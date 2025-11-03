"""
This file is based on the `train_gpt2.py` file in Andrej Karpathy's
`build-nanogpt` github repo.
"""

# =============================================================================
# Imports
# =============================================================================

import asyncio
import dataclasses
import datetime
import itertools
import functools
import json
import logging
import os
import random
import string
import sys
import textwrap
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from os.path import dirname
from typing import Dict, List, Optional

import tiktoken
import torch
import torch.jit
import torch.nn as nn
import torch.profiler
from torch.nn import functional as F
from torch.profiler import ProfilerActivity

from parallelopedia.http.server import (
    HttpApp,
    HttpServer,
    Request,
    route,
)
from parallelopedia.util import ElapsedTimer, get_huggingface_model, join_path

# =============================================================================
# Configuration
# =============================================================================

# If the environment variable PARALLELOPEDIA_DATA_DIR is set, use that instead.
if 'PARALLELOPEDIA_DATA_DIR' in os.environ:
    DATA_DIR = os.environ['PARALLELOPEDIA_DATA_DIR']
else:
    DATA_DIR = join_path(dirname(__file__), '../../data')

# This is the model checkpoint file produced by `train_gpt2.py`.
MODEL_CHECKPOINT = join_path(DATA_DIR, 'model_19072.pt')

# If PARALLELOPEDIA_TORCH_PROFILE is set, profile model initialization.
if 'PARALLELOPEDIA_TORCH_PROFILE' in os.environ:
    TORCH_PROFILE_ACTIVITIES = [
        ProfilerActivity.CPU,
        ProfilerActivity.CUDA,
    ]
else:
    TORCH_PROFILE_ACTIVITIES = None

MODELS_LOCK = threading.Lock()
PRETRAINED_MODELS_LOCK = threading.Lock()

# If PARALLELOPEDIA_CPU_ONLY is set, don't try and use CUDA.
if 'PARALLELOPEDIA_CPU_ONLY' in os.environ or not torch.cuda.is_available():
    CPU_ONLY = True
    NUM_GPUS = 0
    DEVICES = ['cpu']
    MODELS = [None]

    def load_models():
        global MODELS
        MODELS[0] = GPT.from_local_pretrained(
            model_path=MODEL_CHECKPOINT,
            map_location='cpu',
            torch_profile_activities=TORCH_PROFILE_ACTIVITIES,
        )

    def get_next_model_random():
        return MODELS[0]

    def get_next_model_round_robin():
        return MODELS[0]

else:
    # Use all GPUs if available, otherwise use CPU.
    CPU_ONLY = False
    NUM_GPUS = torch.cuda.device_count()
    # Add a CPU version at the end.
    TOTAL_MODELS = NUM_GPUS + 1
    MODELS = [None] * TOTAL_MODELS
    MODELS_ROUND_ROBIN = itertools.cycle(range(TOTAL_MODELS))

    def get_next_model_random():
        # Randomly select a GPU to use.
        return MODELS[random.randint(0, TOTAL_MODELS - 1)]

    def get_next_model_round_robin():
        with MODELS_LOCK:
            index = next(MODELS_ROUND_ROBIN)
        return MODELS[index]

    get_next_model = get_next_model_round_robin

    def load_models():
        global MODELS
        max_workers = min(TOTAL_MODELS, os.cpu_count())
        timer = ElapsedTimer()
        with timer:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(
                        GPT.from_local_pretrained,
                        model_path=MODEL_CHECKPOINT,
                        map_location=f'cuda:{i}',
                        torch_profile_activities=TORCH_PROFILE_ACTIVITIES,
                    ): i for i in range(NUM_GPUS)
                }
                # Add the CPU model.
                futures[executor.submit(
                    GPT.from_local_pretrained,
                    model_path=MODEL_CHECKPOINT,
                    map_location='cpu',
                    torch_profile_activities=TORCH_PROFILE_ACTIVITIES,
                )] = NUM_GPUS
                for future in as_completed(futures):
                    i = futures[future]
                    model = future.result()
                    MODELS[i] = model
        msg = (
            f'Loaded model on {NUM_GPUS} GPU(s) and 1 CPU in '
            f'{timer.elapsed:.3f} seconds.'
        )
        logging.info(msg)

    PRETRAINED_MODELS = [None] * TOTAL_MODELS
    PRETRAINED_MODELS_ROUND_ROBIN = itertools.cycle(range(TOTAL_MODELS))

    def get_next_pretrained_model_random():
        # Randomly select a GPU to use.
        return PRETRAINED_MODELS[random.randint(0, TOTAL_MODELS - 1)]

    def get_next_pretrained_model_round_robin():
        with PRETRAINED_MODELS_LOCK:
            index = next(PRETRAINED_MODELS_ROUND_ROBIN)
        return PRETRAINED_MODELS[index]

    get_next_pretrained_model = get_next_pretrained_model_round_robin

    def load_pretrained_models():
        global PRETRAINED_MODELS
        max_workers = min(TOTAL_MODELS, os.cpu_count())
        timer = ElapsedTimer()
        try:
            with timer:
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = {
                        executor.submit(
                            GPT.from_pretrained,
                            model_name='openai-community/gpt2-xl',
                            map_location=f'cuda:{i}',
                            torch_profile_activities=TORCH_PROFILE_ACTIVITIES,
                        ): i for i in range(NUM_GPUS)
                    }
                    # Add the CPU model.
                    futures[executor.submit(
                        GPT.from_pretrained,
                        model_name='openai-community/gpt2-xl',
                        map_location='cpu',
                        torch_profile_activities=TORCH_PROFILE_ACTIVITIES,
                    )] = NUM_GPUS
                    for future in as_completed(futures):
                        i = futures[future]
                        model = future.result()
                        PRETRAINED_MODELS[i] = model
            msg = (
                f'Loaded gpt2-xl model on {NUM_GPUS} GPU(s) and 1 CPU in '
                f'{timer.elapsed:.3f} seconds.'
            )
            logging.info(msg)
        except Exception as e:
            logging.error(
                f'Failed to load gpt2-xl model on {NUM_GPUS} GPU(s) and 1 CPU: '
                f'{e}'
            )

# #############################################################################
# Helpers
# #############################################################################

def torch_cuda_device_props_to_http_response_header_dict(
    gpu_props
) -> Dict[str, str]:
    gp = gpu_props
    headers = {}
    specs = {
        'X-GPU-Name': lambda: gp.name,
        'X-GPU-Compute-Capability': lambda: f'{gp.major}.{gp.minor}',
        'X-GPU-Num-Multi-Processors': lambda: str(gp.multi_processor_count),
        'X-GPU-PCI': lambda: (
            f'{gp.pci_bus_id}.{gp.pci_device_id}.{gp.pci_domain_id}'
        ),
        'X-GPU-Memory-Total': lambda: str(gp.total_memory),
        'X-GPU-Memory-Bus-Width': lambda: str(gp.memory_bus_width),
        'X-GPU-Memory-Clock-Rate': lambda: (
            f'{gp.memory_clock_rate / 1e3:.0f} MHz'
        ),
        'X-GPU-L2-Cache-Size': lambda: str(gp.l2_cache_size),
        'X-GPU-Clock-Rate': lambda: f'{gp.clock_rate / 1e3:.0f} MHz',
        'X-GPU-Max-Threads-Per-Multi-Processor': lambda: str(
            gp.max_threads_per_multi_processor
        ),
        'X-GPU-Registers-Per-Multiprocessor': lambda: str(
            gp.regs_per_multiprocessor
        ),
        'X-GPU-Shared-Memory-Per-Block': lambda: (
            f'{gp.shared_memory_per_block / 1e3:.0f} KB'
        ),
        'X-GPU-Shared-Memory-Per-Multiprocessor': lambda: (
            f'{gp.shared_memory_per_multiprocessor / 1e3:.0f} KB'
        ),
        'X-GPU-Warp-Size': lambda: f'{gp.warp_size} bytes',
    }

    for key, fn in specs.items():
        try:
            headers[key] = fn()
        except AttributeError:
            continue

    return headers

@functools.cache
def get_gpt2_tokenizer_and_stop() -> Tuple[object, int, str]:
    """
    Get a GPT-2 tokenizer and the stop token ID.

    Returns:

        tuple: A tuple containing the tokenizer object, the stop token ID,
            and the tokenizer type ("tiktoken" or "hf").
    """
    try:
        # Prefer tiktoken if available and has modern API.
        if hasattr(tiktoken, "get_encoding"):
            enc = tiktoken.get_encoding("gpt2")
            stop_id = enc.n_vocab - 1
            return enc, stop_id, "tiktoken"
        if hasattr(tiktoken, "encoding_for_model"):
            enc = tiktoken.encoding_for_model("gpt2")
            stop_id = enc.n_vocab - 1
            return enc, stop_id, "tiktoken"
    except Exception:
        pass
    # Fallback: Hugging Face tokenizer
    try:
        from transformers import GPT2TokenizerFast
        hf_tok = GPT2TokenizerFast.from_pretrained("gpt2")
        class _HFTokenizerWrapper:
            def __init__(self, tok):
                self._tok = tok
                self.n_vocab = tok.vocab_size
            def encode(self, text):
                return self._tok.encode(text, add_special_tokens=False)
            def decode(self, tokens):
                return self._tok.decode(
                    tokens,
                    clean_up_tokenization_spaces=False,
                    skip_special_tokens=False,
                )
        wrapper = _HFTokenizerWrapper(hf_tok)
        stop_id = getattr(hf_tok, "eos_token_id", wrapper.n_vocab - 1)
        return wrapper, stop_id, "hf"
    except Exception as e:
        raise RuntimeError(
            "No usable tokenizer found. Please install 'tiktoken' or 'transformers'."
        ) from e


# =============================================================================
# Globals
# =============================================================================
PRINTABLE = set(c for c in string.printable)
OOPS_NON_PRINTABLE_ENCOUNTERED = (
    'Oops! Non-printable token encountered.  Generation terminated.'
)
DEFAULT_MANUAL_SEED = 42
TRY_JIT_COMPILE = False
TRY_TORCH_COMPILE = False
TORCH_COMPILE_KWDS = {
    'fullgraph': True,
    'mode': 'max-autotune',
}

CUDA_GPU_PROPS = {}
CUDA_GPU_PROPS_HTTP_HEADERS = {}
if not CPU_ONLY:
    for i in range(NUM_GPUS):
        props = torch.cuda.get_device_properties(i)
        CUDA_GPU_PROPS[f'cuda:{i}'] = props
        CUDA_GPU_PROPS_HTTP_HEADERS[f'cuda:{i}'] = (
            torch_cuda_device_props_to_http_response_header_dict(props)
        )

# =============================================================================
# Setup
# =============================================================================

# Use tf32 for matmul precision where possible.
try:
    torch.backends.cudnn.conv.fp32_precision = 'tf32'
    torch.backends.cuda.matmul.allow_tf32 = True
except Exception:
    torch.set_float32_matmul_precision('high')

# =============================================================================
# Classes
# =============================================================================

# N.B. We have simple "no-init" overrides for nn.Embedding and nn.Linear which
#      skip the default initialization routines, significantly reducing the
#      time to load the model by avoiding uniform and random distribution
#      initialization.  As we immediately load all the weights from the model
#      checkpoint straight after creating the model, we don't need the default
#      initialization routines.


class NoInitEmbedding(nn.Embedding):
    def reset_parameters(self):
        # Skip default uniform initialization.
        pass


class NoInitLinear(nn.Linear):
    def reset_parameters(self):
        # Skip default Kaiming initialization.
        pass


class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # Key, query, value projections for all heads, but in a batch.
        self.c_attn = NoInitLinear(config.n_embd, 3 * config.n_embd)

        # Output projection.
        self.c_proj = NoInitLinear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        # Regularization.
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        # Batch size, sequence length, embedding dimensionality.
        B, T, C = (x.size())

        # Calculate query, key, values for all heads in batch and move head
        # forward to be the batch dim.
        #
        # N.B. nh is "number of heads", hs is "head size", and C (number of
        #      channels) is nh * hs.  E.g. in GPT-2 (124M), n_head=12, hs=64,
        #      so nh*hs=C=768 channels in the Transformer.
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)

        head_dim = C // self.n_head

        # (B, nh, T, hs)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)

        # Flash attention.
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Re-assemble all head outputs side by side.
        y = (y.transpose(1, 2).contiguous().view(B, T, C))

        # Output projection.
        y = self.c_proj(y)
        return y


class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc = NoInitLinear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = NoInitLinear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    """
    Configuration class for GPT model.

    Attributes:

        block_size (int): Maximum sequence length.

        vocab_size (int): Number of tokens.  GPT2 from huggingface has a
            vocab size of 50257, which includes 50,000 BPE merges, 256 byte
            tokens, and 1 <|endoftext|> token.  However, Andrej Karpathy's
            `build-nanogpt/train_gpt2.py` uses a vocab size of 50304.  I
            vaguely recall the explanation for this discrepancy as a local
            optimization to yield better alignment sizes, but I'm not 100%
            certain.

            The local GPT2 training that we did on edu_fineweb10b used 50304,
            so we will use that here.

        n_layer (int): Number of layers.

        n_head (int): Number of attention heads.

        n_embd (int): Embedding dimension.
    """
    block_size: int = 1024
    vocab_size: int = 50304
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768


@dataclass
class GPTCheckpoint:
    """
    Checkpoint class for GPT model.

    Mandatory Attributes:

        model (dict): The model state_dict.

        step (int): The step number.

        val_loss (float): The validation loss.

        config (GPTConfig): The configuration.
    """
    model: dict
    step: int
    val_loss: float
    config: GPTConfig

    @classmethod
    def load(cls, checkpoint_path: str, device: str) -> "GPTCheckpoint":
        """
        Load a checkpoint from a file.

        Args:

            checkpoint_path (str): Supplies the path to the checkpoint file.

            device (str): Supplies the device to use for the model.

        Returns:

            GPTCheckpoint: A new GPTCheckpoint instance.

        """
        data = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        checkpoint = cls(
            model=data["model"],
            step=data["step"],
            val_loss=data["val_loss"],
            config=GPTConfig(**data["config"]),
        )
        return checkpoint

    def save(self, checkpoint_path: str) -> None:
        """
        Save the checkpoint to a file.

        Args:

            checkpoint_path (str): Supplies the path to the checkpoint file.
        """
        # N.B. We save config as a raw dictionary to avoid pickling issues
        #      with object namespaces when reloading.
        data = {
            "model": self.model,
            "step": self.step,
            "val_loss": self.val_loss,
            "config": dataclasses.asdict(self.config),
        }
        torch.save(data, checkpoint_path)

    def save_parallel(self, checkpoint_path_prefix: str,
                      num_threads: int) -> None:
        pass


class GPT(nn.Module):

    def __init__(self, checkpoint: GPTCheckpoint,
                 device: Optional[str] = None,
                 manual_seed: Optional[int] = None,
                 torch_profile_activities: Optional[List[type]] = None):
        """
        Initializes a GPT model.

        Arguments:

            checkpoint (GPTCheckpoint): Supplies a checkpoint from which the
                model will be loaded.

            device (str): Optionally supplies the device to use for the model,
                e.g. "cpu" or "cuda".  If None, "cuda" will be used if
                available, otherwise "cpu".

            manual_seed (int): Optionally supplies the manual seed to use for
                generation.

            torch_profile_activities (list): Optionally supplies a list of
                torch.profiler.ProfilerActivity to profile.  This will apply
                for transformer initialization and model weight loading.  The
                torch_profile_init_transformer and torch_profile_load_state
                attributes will be set to the resulting profiler objects.
        """
        super().__init__()

        if device is None:
            if CPU_ONLY:
                device = 'cpu'
            else:
                device = 'cuda'

        assert isinstance(checkpoint.config, GPTConfig), (
            'checkpoint.config must be an instance of GPTConfig.'
        )
        self.config = checkpoint.config
        self.device = device
        self.manual_seed = manual_seed
        self.torch_profile_activities = torch_profile_activities or []
        self.torch_profile_init_transformer = None
        self.torch_profile_load_state = None
        # Populated by the from_local_pretrained() classmethod.
        self.torch_profile_load = None

        self.printable = set(c for c in string.printable)

        timer = ElapsedTimer()

        with timer:
            if not self.torch_profile_activities:
                self._init_transformer()
            else:
                with torch.profiler.profile(
                    activities=self.torch_profile_activities,
                    with_stack=True,
                ) as prof:
                    self._init_transformer()
                self.torch_profile_init_transformer = prof

        msg = f'Initialized GPT model in {timer.elapsed:.3f} seconds.'
        logging.info(msg)

        # Check to see if the first model key starts with '_orig_mod.'; if so,
        # strip that prefix from all keys (if it's present on the first, we
        # can assume it will be present on all).  This happens when loading
        # models straight from build-nanogpt checkpoint saves, for whatever
        # reason.
        model = checkpoint.model
        assert isinstance(model, dict)
        key = next(iter(model))
        if key.startswith('_orig_mod.'):
            new_model = {
                (k.replace('_orig_mod.', '')): v for (k, v) in model.items()
            }
            checkpoint.model = new_model

        with timer:
            if not self.torch_profile_activities:
                self.load_state_dict(checkpoint.model)
            else:
                with torch.profiler.profile(
                    activities=self.torch_profile_activities,
                    with_stack=True,
                ) as prof:
                     self.load_state_dict(checkpoint.model)
                self.torch_profile_load_state = prof

        msg = f'Loaded model weights in {timer.elapsed:.3f} seconds.'
        logging.info(msg)

        enc, stop_token, enc_type = get_gpt2_tokenizer_and_stop()
        # Validate the stop token for tiktoken; Hugging Face decodes EOS
        # differently.
        if enc_type == "tiktoken":
            stop_string = '<|endoftext|>'
            actual = enc.decode([stop_token])
            assert actual == stop_string, f"expected {stop_string}, got {actual}"
        self.enc = enc
        self.stop_token = stop_token

        # Set to eval.
        self.eval()

    def print_profile(self):
        if not self.torch_profile_activities:
            raise RuntimeError('No profiler activities were set.')

        sort_keys = ('cpu_time_total', 'cuda_time_total')

        prof = self.torch_profile_init_transformer
        if prof is not None:
            print('=== [_init_transformer] ===')
            for key in sort_keys:
                print(f'--- {key} ---')
                prof.key_averages().table(sort_by=key)

        prof = self.torch_profile_load_state
        if prof is not None:
            print('=== [load_state_dict] ===')
            for key in sort_keys:
                print(f'--- {key} ---')
                prof.key_averages().table(sort_by=key)

    def _init_transformer(self):
        """
        Initialize the transformer.
        """
        config = self.config
        self.transformer = nn.ModuleDict(
            dict(
                wte=NoInitEmbedding(config.vocab_size, config.n_embd),
                wpe=NoInitEmbedding(config.block_size, config.n_embd),
                h=nn.ModuleList(
                    [Block(config) for _ in range(config.n_layer)]
                ),
                ln_f=nn.LayerNorm(config.n_embd),
            )
        )
        self.lm_head = NoInitLinear(
            config.n_embd,
            config.vocab_size,
            bias=False,
        )
        self.transformer.wte.weight = self.lm_head.weight

    @classmethod
    def from_local_pretrained(
        cls,
        model_path: str,
        map_location: Optional[str] = None,
        manual_seed: Optional[int] = None,
        torch_profile_activities: Optional[List[type]] = None,
    ):
        """
        Load a GPT model from a local checkpoint file (e.g. 'model_19072.pt').

        Arguments:

            model_path (str): Supplies the path to the .pt checkpoint file
                that was produced by `torch.save()`.

            map_location (str): Optionally supplies the device to map the
                loaded tensor parameters to.  If None, "cuda" will be used
                if available, otherwise "cpu".

            manual_seed (int): Optionally supplies the manual seed to use for
                the model.  If None, a random seed will be used.

            torch_profile_activities (list): Optionally supplies a list of
                torch.profiler.ProfilerActivity to profile.

        """
        if manual_seed is None:
            # Use a random seed.
            manual_seed = random.randint(0, 2**32 - 1)

        if map_location is None:
            if torch.cuda.is_available():
                map_location = "cuda"
            else:
                map_location = "cpu"

        torch_profile_load = None

        timer = ElapsedTimer()

        # Load the checkpoint.
        with timer:
            if not torch_profile_activities:
                checkpoint = GPTCheckpoint.load(
                    checkpoint_path=model_path,
                    device=map_location,
                )
            else:
                with torch.profiler.profile(
                    activities=torch_profile_activities,
                    with_stack=True,
                ) as prof:
                    checkpoint = GPTCheckpoint.load(
                        checkpoint_path=model_path,
                        device=map_location,
                    )
                torch_profile_load = prof

        logging.info(
            f'Loaded {model_path} checkpoint in {timer.elapsed:.3f} seconds.'
        )

        # Initialize a new GPT instance using the config.
        with timer:
            model = cls(
                checkpoint=checkpoint,
                device=map_location,
                manual_seed=manual_seed,
                torch_profile_activities=torch_profile_activities,
            )
        logging.info(f'Created GPT model in {timer.elapsed:.3f} seconds.')

        if torch_profile_load is not None:
            model.torch_profile_load = torch_profile_load

        device = map_location
        with timer:
            model.to(device)
        msg = f'Moved model to {device} in {timer.elapsed:.3f} seconds.'
        logging.info(msg)

        logging.info(
            f"Loaded model from step {checkpoint.step}, "
            f"val_loss {checkpoint.val_loss}"
        )
        return model

    @classmethod
    def from_pretrained(cls,
                        model_name: str,
                        map_location: Optional[str] = None,
                        manual_seed: Optional[int] = None,
                        torch_profile_activities: Optional[List[type]] = None,
                        ) -> "GPT":
        """
        Load a GPT model from a pretrained model.

        Arguments:

            model_name (str): Supplies the model name to use.  See the
                docstring for `.util.get_huggingface_safetensors()` for
                more information about the format.

            map_location (str): Optionally supplies the device to map the
                loaded tensor parameters to.  If None, "cuda" will be used
                if available, otherwise "cpu".

            manual_seed (int): Optionally supplies the manual seed to use for
                the model.  If None, a random seed will be used.

            torch_profile_activities (list): Optionally supplies a list of
                torch.profiler.ProfilerActivity to profile.

        """
        if manual_seed is None:
            # Use a random seed.
            manual_seed = random.randint(0, 2**32 - 1)

        if map_location is None:
            if torch.cuda.is_available():
                map_location = "cuda"
            else:
                map_location = "cpu"

        timer = ElapsedTimer()
        with timer:
            hf_model = get_huggingface_model(model_name)
        msg = (
            f'Loaded HuggingFace model {model_name} in '
            f'{timer.elapsed:.3f} seconds.'
        )
        logging.info(msg)

        config = GPTConfig(**{
            'block_size': hf_model.config['n_ctx'],
            'vocab_size': hf_model.config['vocab_size'],
            'n_layer': hf_model.config['n_layer'],
            'n_head': hf_model.config['n_head'],
            'n_embd': hf_model.config['n_embd'],
        })
        checkpoint = GPTCheckpoint(**{
            'model': None,
            'step': 0,
            'val_loss': 0.0,
            'config': config,
        })

        with timer:
            model = cls(
                checkpoint=checkpoint,
                device=map_location,
                manual_seed=manual_seed,
                torch_profile_activities=torch_profile_activities,
            )
        logging.info(f'Created GPT model in {timer.elapsed:.3f} seconds.')

        # This logic is based heavily off build-nanogpt's `train_gpt2.py`;
        # specifically: GPT.from_pretrained().

        exclude = ('.attn.bias', '.attn.masked_bias', 'lm_head.weight')
        transpose = (
            'attn.c_attn.weight',
            'attn.c_proj.weight',
            'mlp.c_fc.weight',
            'mlp.c_proj.weight',
        )

        # Identify the HuggingFace keys we're interested in.
        st = hf_model.safetensors

        # Identify our model keys we're interested in.
        sd = model.state_dict()
        sd_keys = [k for k in sd.keys() if not k.endswith(exclude)]
        hf_keys = [k.replace('transformer.', '') for k in sd_keys]

        # Copying tensors in parallel yields decent speedups, at least on my
        # V100s which have five concurrent copy engines.
        def copy_tensor(hf_key, sd_key):
            hf_tensor = st.get_tensor(hf_key)
            if hf_key.endswith(transpose):
                assert hf_tensor.shape[::-1] == sd[sd_key].shape
                with torch.no_grad():
                    sd[sd_key].copy_(hf_tensor.t())
            else:
                assert hf_tensor.shape == sd[sd_key].shape
                with torch.no_grad():
                    sd[sd_key].copy_(hf_tensor)

        keys = zip(hf_keys, sd_keys)
        max_workers = min(os.cpu_count(), len(sd_keys))
        with timer:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(copy_tensor, hf_key, sd_key):
                        (hf_key, sd_key) for (hf_key, sd_key) in keys
                }
                for future in as_completed(futures):
                    future.result()
        logging.info(
            f'Copied weights with {max_workers} thread(s) '
            f'in {timer.elapsed:.3f} seconds.'
        )

        device = map_location
        with timer:
            model.to(device)
        msg = f'Moved model to {device} in {timer.elapsed:.3f} seconds.'
        logging.info(msg)

        return model

    def _init_weights(self, module):
        # N.B. This is only for new models that you plan on training, not
        #      existing models you've already trained that you want to run
        #      inference on.  It is a verbatim copy of the routine from the
        #      `train_gpt2.py` script and included for posterity, despite us
        #      never using it.
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()

        # Forward the token and position embeddings.

        # Shape (T)
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)

        # Position embeddings of shape (T, n_embd).
        pos_emb = self.transformer.wpe(pos)

        # Token embeddings of shape (B, T, n_embd).
        tok_emb = self.transformer.wte(idx)

        x = tok_emb + pos_emb

        # Forward the blocks of the transformer.
        for block in self.transformer.h:
            x = block(x)

        # Forward the final layernorm and the classifier.
        x = self.transformer.ln_f(x)

        # (B, T, vocab_size)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1)
            )

        return (logits, loss)

    # @torch.compile
    def generate(
        self, text: str, max_length: int = 1024, top_k: int = 50,
        seed: int = None, save_rate: callable = None
    ) -> str:
        """
        Generate text from the model.

        Args:

            text (str): Supplies the prompt to condition on.

            max_length (int): Maximum total length (prompt + generated).

            top_k (int): Number of tokens to consider at each generation step.

            seed (int): Optionally supplies the manual seed to use for the
                generator.  If None, the model's manual seed will be used.

            save_rate (callable): Optionally supplies a callable that will be
                called with the tokens per second rate.

        Returns:

            str: The generated text (including the initial prompt).
        """
        enc = self.enc
        device = self.device
        stop_token = self.stop_token

        # Encode prompt -> tensor of shape (1, T)
        tokens = enc.encode(text)

        x = torch.tensor(
            tokens,
            dtype=torch.long,
            device=device
        ).unsqueeze(0)

        # Create a random generator for reproducibility.
        sample_rng = torch.Generator(device=device)
        if seed is None:
            seed = self.manual_seed
        sample_rng.manual_seed(seed)

        output = []

        # Generate tokens up to our max length, or until we hit the stop token.
        start = time.perf_counter()
        count = 0
        while x.size(1) < max_length:
            count += 1
            with torch.no_grad():
                # Forward pass, ignoring the returned loss.
                (logits, _) = self(x)

            # Take the logits at the last time-step (shape: (1, vocab_size)).
            logits = logits[:, -1, :]

            # Convert to probabilities.
            probs = F.softmax(logits, dim=-1)

            # Top-k sampling.
            topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)

            # Sample the next token.
            next_idx = torch.multinomial(
                topk_probs,
                num_samples=1,
                generator=sample_rng,
            )
            next_token = torch.gather(topk_indices, -1, next_idx)  # (1, 1)

            # If the next token is the stop token, we're done.
            next_token_item = next_token.item()
            if next_token_item == stop_token:
                break

            # Append token to current sequence.  Although we only yield a
            # singular decoded token below, we still need to keep track of
            # the entire sequence for subsequent generation steps.
            x = torch.cat((x, next_token), dim=1)

            # Decode the newly-generated token.
            new_text_fragment = enc.decode([next_token.item()])

            # If the next token isn't printable, terminate generation.  (With
            # our locally-trained GPT2 124M model, this happens quite often.)
            if not all(c in self.printable for c in new_text_fragment):
                break

            output.append(new_text_fragment)

        end = time.perf_counter()
        elapsed = end - start
        tokens_per_sec = float(count) / elapsed
        if save_rate:
            save_rate(tokens_per_sec)

        msg = (
            f'Generated {count} tokens in {elapsed:.2f} seconds '
            f'({tokens_per_sec:.2f} tokens/sec)'
        )
        logging.info(msg)

        return text + ''.join(output)

    # @torch.compile
    def generate_slim(
        self, text_tokens: torch.Tensor, max_length: int = 1024,
        top_k: int = 50, seed: int = None,
    ) -> str:
        """
        Generate text from the model.  This version differs from `generate()`
        in that it does not use any Python code that causes a torch graph
        break.

        Args:

            text_tokens (torch.Tensor): The encoded prompt as a tensor of
                shape (1, T).

            max_length (int): Maximum total length (prompt + generated).

            top_k (int): Number of tokens to consider at each generation step.

            seed (int): Optionally supplies the manual seed to use for the
                generator.  If None, the model's manual seed will be used.

        Returns:

            str: The generated text (including the initial prompt).
        """
        # Initialize alias.
        device = self.device
        stop_token = self.stop_token

        # Create the tensor for capturing predicted tokens.
        x = torch.tensor(
            text_tokens,
            dtype=torch.long,
            device=device
        ).unsqueeze(0)

        # Create a random generator for reproducibility.
        # sample_rng = torch.Generator(device=device)
        # if seed is None:
        #     seed = self.manual_seed
        # sample_rng.manual_seed(seed)

        # Generate tokens up to our max length, or until we hit the stop token.
        for _ in range(max_length):
            with torch.no_grad():
                # Forward pass, ignoring the returned loss.
                (logits, _) = self(x)

            # Take the logits at the last time-step (shape: (1, vocab_size)).
            logits = logits[:, -1, :]

            # Convert to probabilities.
            probs = F.softmax(logits, dim=-1)

            # Top-k sampling.
            topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)

            # Sample the next token.
            next_idx = torch.multinomial(
                topk_probs,
                num_samples=1,
                # generator=sample_rng,
            )
            next_token = torch.gather(topk_indices, -1, next_idx)  # (1, 1)

            # If the next token is the stop token, we're done.
            # next_token_item = next_token.item()
            # if next_token_item == stop_token:
            #    break

            # Append token to current sequence.
            x = torch.cat((x, next_token), dim=1)

        return x

    async def generate_async_for(
        self, text: str, max_length: int = 1024, top_k: int = 50,
        seed: int = None,
    ):
        """
        Asynchronously generate text from the model, yielding tokens
        one at a time as soon as they are available.

        Arguments:

            text (str): Supplies the prompt.

            max_length (int): Supplies the maximum total length,
                including prompt.

            top_k (int): Supplies the number of tokens to consider
                at each generation step.

            seed (int): Optionally supplies the manual seed to use
                for the generator.  If None, the model's manual
                seed will be used.

        Returns:

            str: The generated text (including the initial prompt).

        Yields:

            byte: The newly generated decoded token.  If -1, a non-printable
            token was generated, and generation was terminated.
        """

        enc = self.enc
        stop_token = self.stop_token

        # Encode the prompt -> tensor of shape (1, T)
        tokens = enc.encode(text)
        x = torch.tensor(
            tokens, dtype=torch.long, device=self.device
        ).unsqueeze(0)

        sample_rng = torch.Generator(device=self.device)
        if seed is None:
            seed = self.manual_seed
        sample_rng.manual_seed(seed)

        start_time = time.perf_counter()
        count = 0
        while x.size(1) < max_length:
            count += 1
            with torch.no_grad():
                # Forward pass, ignoring the returned loss.
                (logits, _) = self(x)

            # Take the logits at the last time-step (shape: (1, vocab_size)).
            logits = logits[:, -1, :]

            # Convert to probabilities.
            probs = F.softmax(logits, dim=-1)

            # Top-k sampling.
            topk_probs, topk_indices = torch.topk(probs, k=top_k, dim=-1)

            # Sample the next token.
            next_idx = torch.multinomial(
                topk_probs,
                num_samples=1,
                generator=sample_rng,
            )
            next_token = torch.gather(topk_indices, -1, next_idx)

            # If the next token is the stop token, we're done.
            next_token_item = next_token.item()
            if next_token_item == stop_token:
                break

            # Append token to current sequence.  Although we only yield a
            # singular decoded token below, we still need to keep track of
            # the entire sequence for subsequent generation steps.
            x = torch.cat((x, next_token), dim=1)

            # Decode the newly-generated token.  Note that a single token may
            # decode to multiple characters.
            new_text_fragment = enc.decode([next_token.item()])

            # If any of the next characters in the decoded text representation
            # aren't printable, terminate generation.
            if not all(c in self.printable for c in new_text_fragment):
                yield -1
                break

            yield new_text_fragment

            # Yield control back to the event loop before continuing
            # generation.
            await asyncio.sleep(0)

        elapsed = time.perf_counter() - start_time
        logging.debug(
            f"[generate_async_for] Generated {count} tokens in "
            f"{elapsed:.2f} seconds (~{count / elapsed:.2f} tok/s)"
        )


class Gpt2App(HttpApp):

    def __init__(self, server: HttpServer) -> None:
        super().__init__(server)
        self.printable = PRINTABLE
        self.task = None
        self.keep_alive = None

    @classmethod
    def init_once(cls):
        load_models()
        load_pretrained_models()

        global MODELS, PRETRAINED_MODELS

        # This doesn't work because torch.jit doesn't handle our
        # async generator.
        global TRY_JIT_COMPILE
        if TRY_JIT_COMPILE:
            for (i, model) in enumerate(MODELS):
                model.config = dataclasses.asdict(model.config)
                timer = ElapsedTimer()
                with timer:
                    model = torch.jit.script(model)
                    MODELS[i] = model
                logging.info(
                    f'JIT compiled model {i} in {timer.elapsed:.3f} seconds.'
                )

        kwds = TORCH_COMPILE_KWDS
        global TRY_TORCH_COMPILE
        if TRY_TORCH_COMPILE:
            for (i, model) in enumerate(MODELS):
                model.config = dataclasses.asdict(model.config)
                timer = ElapsedTimer()
                with timer:
                    model = torch.compile(model, **kwds)
                    MODELS[i] = model
                logging.info(
                    f'torch.compiled model {i} in '
                    f'{timer.elapsed:.3f} seconds.'
                )

            for (i, model) in enumerate(PRETRAINED_MODELS):
                model.config = dataclasses.asdict(model.config)
                timer = ElapsedTimer()
                with timer:
                    model = torch.compile(model, **kwds)
                    PRETRAINED_MODELS[i] = model
                logging.info(
                    f'torch.compiled pretrained model {i} in '
                    f'{timer.elapsed:.3f} seconds.'
                )

    def _task_complete(self, task):
        assert self.task is not None
        assert self.task is task
        self.task = None
        if self.keep_alive is False:
            try:
                self.server.transport.close()
            except Exception:
                pass

    def is_connected(self):
        # server.transport will be severed when the client disconnects.
        server = self.server
        transport = None
        try:
            transport = server.transport
        except AttributeError:
            pass
        return transport is not None

    def write(self, response_bytes):
        server = self.server
        transport = None
        try:
            transport = server.transport
        except AttributeError:
            pass
        if transport is not None:
            transport.write(response_bytes)
            return True
        else:
            return False

    async def generate_response(
        self, request: Request, text: str, **kwds: Dict
    ) -> None:

        response = request.response

        response.code = 200
        response.message = 'OK'
        response.chunked_response = True
        response.content_type = 'text/plain'

        if kwds is None:
            kwds = {}
        max_length = min(int(kwds.get('max_length', 100) or 100), 1024)
        top_k = min(int(kwds.get('top_k', 50) or 50), 50)
        seed = kwds.get('seed', None)
        if seed:
            try:
                seed = int(seed)
            except ValueError:
                seed = None
        if not seed:
            seed = random.randint(0, 2**32 - 1)

        device = kwds.get('device', None)

        model_name = kwds.get('model', None)
        if model_name == 'gpt2-xl':
            models = PRETRAINED_MODELS
            get_next = get_next_pretrained_model
        else:
            model_name = 'gpt2'
            models = MODELS
            get_next = get_next_model

        model = None
        gpu_index = None
        if device is not None:
            if device == 'cpu':
                model = models[-1]
            elif device.startswith('cuda:'):
                try:
                    index = int(device[5:])
                except ValueError:
                    index = -1
                if index < 0 or index >= NUM_GPUS:
                    index = -1
                if index != -1:
                    model = models[index]
                    gpu_index = index
            elif device == 'cuda':
                index = random.randint(0, NUM_GPUS - 1)
                model = models[index]
                gpu_index = index

        if not model:
            # Get a model.  If there are multiple models available, e.g. if we
            # have multiple GPUs, this will balance the load a bit.
            model = get_next()

        # If a specific GPU wasn't explicitly chosen above, infer it from the
        # selected model's device so we can attach GPU headers.
        if gpu_index is None and hasattr(model, 'device'):
            dev = model.device
            if isinstance(dev, str) and dev.startswith('cuda:'):
                try:
                    gpu_index = int(dev.split(':', 1)[1])
                except ValueError:
                    pass

        gpu_http_headers = None
        if gpu_index is not None:
            gpu_http_headers = CUDA_GPU_PROPS_HTTP_HEADERS[f'cuda:{gpu_index}']

        expose_headers = (
            'Access-Control-Expose-Headers: '
            'X-Max-Length, '
            'X-Top-K, '
            'X-Seed, '
            'X-Model-Name, '
            'X-Model-Device'
        )

        if gpu_http_headers is not None:
            expose_headers += f', {", ".join(gpu_http_headers.keys())}'

        other_headers = [
            f'X-Max-Length: {max_length}',
            f'X-Top-K: {top_k}',
            f'X-Seed: {seed}',
            f'X-Model-Name: {model_name}',
            f'X-Model-Device: {model.device}',
        ]
        if gpu_http_headers is not None:
            other_headers.extend([
                f'{k}: {v}' for k, v in gpu_http_headers.items()
            ])

        # Ensure custom response headers are visible to browsers.
        response.other_headers.append(expose_headers)
        response.other_headers.extend(other_headers)

        # We want to enable TCP_NODELAY for the duration of the response.
        # This ensures packets are sent immediately without any internal
        # buffering.
        try:
            response.enable_tcp_nodelay()
            enabled_nodelay = True
        except Exception as e:
            logging.error(f'Error enabling TCP_NODELAY: {e}')
            enabled_nodelay = False

        # Write the chunked header immediately.
        response_bytes = bytes(response)
        if not self.write(response_bytes):
            # Encountered a disconnect, return.
            return

        # From herein, all data must be transferred to the client via chunked
        # encoding with `response.send_chunk()`.

        # Send the initial prompt text.
        response.send_chunk(text)

        generate_tokens = model.generate_async_for(
            text,
            max_length=max_length,
            top_k=top_k,
            seed=seed,
        )
        async for decoded_token in generate_tokens:
            if decoded_token == -1:
                # A non-printable token was generated, terminating generation.
                response.send_chunk(OOPS_NON_PRINTABLE_ENCOUNTERED)
                break

            # If the client has forcibly disconnected, terminate generation.
            if not self.is_connected():
                break

            # Otherwise, send the decoded token to the client via chunked
            # encoding.
            response.send_chunk(decoded_token)

        # Send the termination chunk.  This may fail at the socket.send()
        # level if the client has already disconnected, which is harmless.
        response.end_chunks()

        # Disable TCP_NODELAY now that the response is complete.  Again, this
        # may fail at the socket level if the client has already disconnected,
        # which is harmless.
        if enabled_nodelay:
            try:
                response.disable_tcp_nodelay()
            except Exception as e:
                logging.error(f'Error disabling TCP_NODELAY: {e}')

    @route
    def generate(self, request: Request, *args: List, **kwds: Dict) -> None:
        prompt = args[0]
        # Obtain the event loop and schedule the response generation via our
        # async generation coroutine.  We have to do it like this as at this
        # point we're still within the call frame of the data_received()
        # protocol callback, which isn't an async function.
        loop = asyncio.get_running_loop()
        assert self.task is None
        coro = self.generate_response(request, prompt, **kwds)
        self.task = loop.create_task(coro)
        self.keep_alive = request.keep_alive
        self.task.add_done_callback(self._task_complete)


def parse_arguments():
    """
    Parse the command-line arguments for the parallelopedia.gpt2 module.

    Returns:
        argparse.Namespace: The parsed command-line arguments.

    """
    import argparse
    parser = argparse.ArgumentParser(description='Run the GPT2 module.')
    parser.add_argument(
        '--log-level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level.',
    )
    parser.add_argument(
        '--model',
        type=str,
        default='gpt2',
        choices=['gpt2', 'gpt2-xl'],
        help='Select the model to use.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Select the device to use.',
    )
    parser.add_argument(
        '--max-length',
        type=int,
        default=100,
        help='Set the maximum length of the generated text.',
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=50,
        help='Set the top-k value for sampling.',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Set the random seed for generation.',
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default="Einstein's Theory of Relativity states that",
        help='Set the prompt for generation.',
    )
    parser.add_argument(
        '--torch-compile',
        action='store_true',
        help='Compile the models using torch.compile().',
    )
    parser.add_argument(
        '--torch-jit',
        action='store_true',
        help='Compile the models using torch.jit.script().',
    )
    parser.add_argument(
        '--torch-compile-fullgraph',
        action='store_true',
        help='Compile the models using torch.compile() with fullgraph=True.',
    )
    parser.add_argument(
        '--torch-compile-reduce-overhead',
        action='store_true',
        help=(
            'Compile the models using torch.compile() with '
            'mode="reduce-overhead"',
        )
    )
    parser.add_argument(
        '--torch-compile-max-autotune',
        action='store_true',
        help=(
            'Compile the models using torch.compile() with '
            'mode="max_autotune".',
        )
    )
    parser.add_argument(
        '--generate-slim',
        action='store_true',
        help='Use the generate_slim() method for generation.',
    )
    parser.add_argument(
        '--rounds',
        type=int,
        default=3,
        help='Set the number of rounds for generation.',
    )
    parser.add_argument(
        '--wrap',
        type=int,
        default=60,
        help='Set the wrap width for text output.',
    )
    parser.add_argument(
        '--note',
        type=str,
        default='',
        help='Set a note to include in the JSON output.',
    )
    args = parser.parse_args()
    return args


def main():
    """
    Main entry point for the parallelopedia.gpt2 module.
    """
    args = parse_arguments()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    start_time = time.time()
    start_timestamp = datetime.datetime.now().isoformat()

    timer = ElapsedTimer()
    with timer:
        if args.model == 'gpt2-xl':
            model = GPT.from_pretrained(
                model_name='openai-community/gpt2-xl',
                map_location=args.device,
            )
        else:
            model = GPT.from_local_pretrained(
                model_path=MODEL_CHECKPOINT,
                map_location=args.device,
                manual_seed=args.seed,
            )
    logging.info(
        f'Loaded {args.model} on {args.device} '
        f'in {timer.elapsed:.3f} seconds.'
    )

    if args.torch_compile:
        if args.torch_jit:
            msg = 'Cannot specify both --torch-compile and --torch-jit.'
            raise ValueError(msg)
        model.config = dataclasses.asdict(model.config)
        kwds = {}
        if args.torch_compile_fullgraph:
            kwds['fullgraph'] = True
        if args.torch_compile_reduce_overhead:
            if args.torch_compile_max_autotune:
                msg = (
                    'Cannot specify both --torch-compile-reduce-overhead and '
                    '--torch-compile-max-autotune.'
                )
                raise ValueError(msg)
            kwds['mode'] = 'reduce-overhead'
        elif args.torch_compile_max_autotune:
            kwds['mode'] = 'max-autotune'
        with timer:
            model = torch.compile(model, **kwds)
        logging.info(f'torch.compiled model in {timer.elapsed:.3f} seconds.')
    elif args.torch_jit:
        model.config = dataclasses.asdict(model.config)
        with timer:
            model = torch.jit.script(model)
        logging.info(f'JIT compiled model in {timer.elapsed:.3f} seconds.')

    seed = args.seed
    if seed is None or seed == '':
        seed = random.randint(0, 2**32 - 1)

    if args.generate_slim:
        text_tokens = model.enc.encode(args.prompt)
        prompt_token_length = len(text_tokens)

    rates = []
    for i in range(args.rounds):
        logging.info(f'Round {i + 1} of {args.rounds}.')
        if args.generate_slim:
            with timer:
                x = model.generate_slim(
                    text_tokens,
                    max_length=args.max_length,
                    top_k=args.top_k,
                    seed=seed,
                )
            elapsed = timer.elapsed
            count = x.size(1) - prompt_token_length
            tokens_per_sec = count / elapsed
            rates.append(tokens_per_sec)
            logging.info(
                f'Generated {count} tokens in {elapsed:.2f} seconds '
                f'({tokens_per_sec:.2f} tokens/sec)'
            )
            output = model.enc.decode(x[0].tolist())
        else:
            save_rate = lambda x: rates.append(x)
            output = model.generate(
                args.prompt,
                max_length=args.max_length,
                top_k=args.top_k,
                seed=seed,
                save_rate=save_rate,
            )

        if args.wrap:
            output = textwrap.fill(output, width=args.wrap)
        logging.info(f'Output:\n{output}')

    # The filename is of the form:
    #   `gpt2-rates-<yyyy-mm-dd-hh-ss-mm.sss>-[optional].json`
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y-%m-%d-%H-%M-%S-%f')
    filename = f"gpt2-rates-{timestamp}"
    if args.torch_compile:
        filename += '-torch-compile'
        if args.torch_compile_reduce_overhead:
            filename += '-reduce-overhead'
        elif args.torch_compile_max_autotune:
            filename += '-max-autotune'
        if args.torch_compile_fullgraph:
            filename += '-fullgraph'
    if args.generate_slim:
        filename += '-generate-slim'

    conda_env_name = os.environ.get('CONDA_DEFAULT_ENV', 'Unknown')
    filename += f'-{conda_env_name}'

    filename += '.json'

    if not isinstance(model.config, dict):
        model_config = dataclasses.asdict(model.config)
    else:
        model_config = model.config

    end_time = time.time()
    end_timestamp = datetime.datetime.now().isoformat()

    if args.device.startswith('cuda'):
        ix = args.device.find(':')
        if ix == -1:
            device_index = 0
        else:
            device_index = int(args.device[ix+1:])

        device_name = torch.cuda.get_device_name(device_index)
    else:
        device_name = 'CPU'

    try:
        is_gil_enabled = sys._is_gil_enabled()
    except AttributeError:
        is_gil_enabled = False

    # Prepare a dictionary with the details to save.
    run_details = {
        "rates": rates,
        "model_config": model_config,
        "args": vars(args),
        "start_timestamp": start_timestamp,
        "end_timestamp": end_timestamp,
        "elapsed": f'{end_time - start_time:.3f}',
        "device_name": device_name,
        "conda_env_name": conda_env_name,
        "is_gil_enabled": is_gil_enabled,
        "note": args.note,
    }

    # Write the JSON file.
    with open(filename, "w") as json_file:
        json.dump(run_details, json_file, indent=4)

    logging.info(f"Run details saved to {filename}.")

if __name__ == '__main__':
    main()

# vim:set ts=8 sw=4 sts=4 tw=78 et:                                          #
