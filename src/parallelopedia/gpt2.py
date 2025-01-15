"""
This file is based on the `train_gpt2.py` file in Andrej Karpathy's
`build-nanogpt` github repo.
"""

# =============================================================================
# Imports
# =============================================================================

import os
import itertools
import asyncio
import dataclasses
import logging
import random
import string
import time
import threading
from dataclasses import dataclass
from os.path import dirname
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import tiktoken
import torch
import torch.nn as nn
import torch.profiler
from torch.nn import functional as F
from torch.profiler import ProfilerActivity

from parallelopedia.http.server import (
    HttpApp,
    HttpServer,
    Request,
    make_routes,
    router,
)
from parallelopedia.util import ElapsedTimer, join_path

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

# If PARALLELOPEDIA_CPU_ONLY is set, don't try and use CUDA.
if 'PARALLELOPEDIA_CPU_ONLY' in os.environ or not torch.cuda.is_available():
    CPU_ONLY = True
    NUM_GPUS = 0
    DEVICES = ['cpu']
    MODELS = [None]

    def load_models():
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

# =============================================================================
# Globals
# =============================================================================
PRINTABLE = set(c for c in string.printable)
OOPS_NON_PRINTABLE_ENCOUNTERED = (
    'Oops! Non-printable token encountered.  Generation terminated.'
)
DEFAULT_MANUAL_SEED = 42

# =============================================================================
# Setup
# =============================================================================

# Use bfloat16 for matmul precision where possible.
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
        data = torch.load(checkpoint_path, map_location=device)
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

        # Obtain the tokenizer for GPT2, and resolve the stop token.
        enc = tiktoken.get_encoding("gpt2")
        stop_string = '<|endoftext|>'
        stop_token = enc.n_vocab - 1
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
        assert (
            T <= self.config.block_size
        ), (
            f"Cannot forward sequence of length {T}, "
            f"block size is only {self.config.block_size}"
        )

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

    def generate(
        self, text: str, max_length: int = 1024, top_k: int = 50
    ) -> str:
        """
        Generate text from the model, conditioned on `text`.

        Args:

            text (str): The prompt to condition on.

            max_length (int): Maximum total length (prompt + generated).

            top_k (int): Number of tokens to consider at each generation step.

        Returns:

            str: The generated text (including the initial prompt).
        """
        enc = self.enc
        device = self.device
        stop_token = self.stop_token

        # Encode prompt -> tensor of shape (1, T)
        tokens = enc.encode(text)

        x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

        # Create a random generator for reproducibility.
        sample_rng = torch.Generator(device=device)
        sample_rng.manual_seed(self.manual_seed)

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
            if next_token.item() == stop_token:
                break

            # Otherwise, concatenate this token to the sequence and continue
            # generation.
            x = torch.cat((x, next_token), dim=1)

        end = time.perf_counter()
        elapsed = end - start
        tokens_per_sec = float(count) / elapsed

        msg = (
            f'Generated {count} tokens in {elapsed:.2f} seconds '
            f'({tokens_per_sec:.2f} tokens/sec)'
        )
        logging.debug(msg)

        # Decode the generated tokens and return the text, including the
        # prompt.
        output_tokens = x[0].tolist()
        return enc.decode(output_tokens)

    async def generate_async_for(
        self, text: str, max_length: int = 1024, top_k: int = 50,
        seed: int = None,
    ):
        """
        Asynchronously generate text from the model, yielding tokens
        one at a time as soon as they are available.

        Arguments:

            text (str): Supplies the prompt to condition on.

            max_length (int): Maximum total length (prompt + generated).

            top_k (int): Number of tokens to consider at each generation step.

            seed (int): Optionally supplies the manual seed to use for the
                generator.  If None, the model's manual seed will be used.

        Yields:

            byte: The newly generated text token (decoded).  If -1, a
            non-printable token was generated, and generation was terminated.
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

        logging.debug(
            f'[generate_async_for] Starting generation loop for {text} '
            f'with seed {seed}.'
        )

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
    routes = make_routes()
    route = router(routes)

    def __init__(self, server: HttpServer) -> None:
        super().__init__(server)
        self.printable = PRINTABLE

    @classmethod
    def init_once(cls):
        load_models()

    async def generate_response(
        self, request: Request, text: str, **kwds: Dict
    ) -> None:

        server = self.server
        transport = server.transport
        if not transport:
            return

        response = request.response

        response.code = 200
        response.message = 'OK'
        response.chunked_response = True
        response.content_type = 'text/plain'

        if kwds is None:
            kwds = {}
        max_length = min(int(kwds.get('max_length', 100)), 1024)
        top_k = min(int(kwds.get('top_k', 50)), 50)
        seed = kwds.get('seed', None)
        if seed is not None:
            seed = int(seed)
        else:
            seed = random.randint(0, 2**32 - 1)

        # Get a model.  If there are multiple models available, e.g. if we
        # have multiple GPUs, this will balance the load a bit.
        model = get_next_model()

        response.other_headers.extend([
            f'X-Max-Length: {max_length}',
            f'X-Top-K: {top_k}',
            f'X-Seed: {seed}',
            f'X-Model-Device: {model.device}',
        ])

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
        transport.write(response_bytes)

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

            # The HTTP server's `connection_lost()` will clear the server's
            # transport member if the connection is lost.  If this happens,
            # we can stop generating tokens.
            transport = server.transport
            if not transport:
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
        text = args[0]
        # Obtain the event loop and schedule the response generation via our
        # async generation coroutine.  We have to do it like this as at this
        # point we're still within the call frame of the data_received()
        # protocol callback, which isn't an async function.
        loop = asyncio.get_running_loop()
        loop.create_task(self.generate_response(request, text, **kwds))


if __name__ == '__main__':
    logging.basicConfig(
        level=getattr(logging, 'DEBUG'),
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    load_models()
    model = get_next_model()
    result = model.generate("The quick brown fox")
    print(f'Returned: {result}')
