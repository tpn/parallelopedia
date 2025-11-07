"""
Generic LLM inference server and CLI, analogous to `gpt2.py`'s Gpt2App, with
large-model safety (sharded loading), KV cache, context clamping, and
thread-safe lazy caching.

Environment knobs:
- PARALLELOPEDIA_LLM_CPU_ONLY=1          -> force CPU
- PARALLELOPEDIA_LLM_TRUST_REMOTE_CODE=1 -> allow trust_remote_code=True
- PARALLELOPEDIA_LLM_LOAD_IN_8BIT=1      -> attempt 8-bit quantization (bitsandbytes)
- PARALLELOPEDIA_LLM_LOAD_IN_4BIT=1      -> attempt 4-bit quantization (bitsandbytes)
"""

# =============================================================================
# Imports
# =============================================================================

import asyncio
import logging
import os
import random
import string
import threading
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import torch
from torch.nn import functional as F
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from parallelopedia.http.server import (
    HttpApp,
    HttpServer,
    Request,
    route,
)

from .util import torch_cuda_device_props_to_http_response_header_dict

# =============================================================================
# Configuration
# =============================================================================

# Alias -> Hugging Face repo id. You can keep adding friendly names here.
MODEL_VARIANTS = {
    "gpt-oss-20b": "openai/gpt-oss-20b",
    "gpt-oss-120b": "openai/gpt-oss-120b",
    "qwen3-4b": "Qwen/Qwen3-4B",
    "lfm2-8b-a1b": "LiquidAI/LFM2-8B-A1B",
    "llama2-7b": "meta-llama/Llama-2-7b-hf",
}

LLM_DEFAULT_MODEL = "qwen3-4b"

# Device environment
if "PARALLELOPEDIA_LLM_CPU_ONLY" in os.environ or not torch.cuda.is_available():
    CPU_ONLY = True
    NUM_GPUS = 0
else:
    CPU_ONLY = False
    NUM_GPUS = torch.cuda.device_count()

# Safety / features
TRUST_REMOTE_CODE = "PARALLELOPEDIA_LLM_TRUST_REMOTE_CODE" in os.environ
WANT_8BIT = "PARALLELOPEDIA_LLM_LOAD_IN_8BIT" in os.environ
WANT_4BIT = "PARALLELOPEDIA_LLM_LOAD_IN_4BIT" in os.environ
if WANT_8BIT and WANT_4BIT:
    logging.warning("Both 8-bit and 4-bit requested; defaulting to 4-bit.")
    WANT_8BIT = False

# =============================================================================
# Helpers
# =============================================================================


def _normalize_alias(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    name = name.strip().lower()
    if name in MODEL_VARIANTS:
        return name
    if name in ("20b", "oss-20b"):
        return "gpt-oss-20b"
    if name in ("120b", "oss-120b"):
        return "gpt-oss-120b"
    return None


def _device_key(device: Optional[str]) -> str:
    if CPU_ONLY:
        return "cpu"
    if not device:
        # Implicit "multi-GPU if available" (let HF Accelerate shard)
        return "cuda"
    return device


def _is_control_char(ch: str) -> bool:
    code = ord(ch)
    return (0 <= code < 32 and ch not in "\t\n\r") or (127 <= code <= 159)


def _is_stream_printable(s: str) -> bool:
    if "\ufffd" in s:
        return False
    return not any(_is_control_char(c) for c in s)


def resolve_model_id(name_or_id: Optional[str]) -> Tuple[str, str]:
    """
    Returns (model_id, display_name). If name_or_id matches an alias, resolve
    via MODEL_VARIANTS; if it looks like 'org/repo', pass through.
    """
    if not name_or_id:
        alias = LLM_DEFAULT_MODEL
        return MODEL_VARIANTS[alias], alias
    name = name_or_id.strip()
    alias = _normalize_alias(name)
    if alias is not None:
        return MODEL_VARIANTS[alias], alias
    if "/" in name:
        return name, name  # direct repo id
    # Fallback to default alias
    alias = LLM_DEFAULT_MODEL
    return MODEL_VARIANTS[alias], alias


# =============================================================================
# Globals
# =============================================================================

PRINTABLE = set(c for c in string.printable)
OOPS_NON_PRINTABLE_ENCOUNTERED = (
    "Oops! Non-printable token encountered.  Generation terminated."
)

CUDA_GPU_PROPS: Dict[str, object] = {}
CUDA_GPU_PROPS_HTTP_HEADERS: Dict[str, Dict[str, str]] = {}
if not CPU_ONLY:
    for i in range(NUM_GPUS):
        props = torch.cuda.get_device_properties(i)
        CUDA_GPU_PROPS[f"cuda:{i}"] = props
        CUDA_GPU_PROPS_HTTP_HEADERS[f"cuda:{i}"] = (
            torch_cuda_device_props_to_http_response_header_dict(props)
        )

# Thread-safe lazy model cache: (model_id, device_key) -> CausalModel
MODEL_CACHE: Dict[Tuple[str, str], "CausalModel"] = {}
_MODEL_CACHE_LOCKS: Dict[Tuple[str, str], threading.Lock] = defaultdict(threading.Lock)

# =============================================================================
# Model wrapper
# =============================================================================


@dataclass
class CausalModel:
    model_id: str
    device: str = "cuda"  # 'cuda', 'cuda:N', or 'cpu'
    torch_dtype: Optional[torch.dtype] = None
    device_map: Optional[str] = None  # 'auto' (multi-GPU) or None (single target)

    # populated in __post_init__
    tokenizer: AutoTokenizer = None
    model: AutoModelForCausalLM = None
    stop_ids: Set[int] = None
    max_ctx: int = 2048

    def __post_init__(self) -> None:
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            use_fast=True,
            trust_remote_code=TRUST_REMOTE_CODE,
        )

        # Quantization knobs
        load_in_8bit = False
        load_in_4bit = False
        if not CPU_ONLY and (WANT_8BIT or WANT_4BIT):
            try:
                import bitsandbytes as _  # noqa: F401

                load_in_4bit = WANT_4BIT
                load_in_8bit = WANT_8BIT and not load_in_4bit
            except Exception:
                logging.warning(
                    "bitsandbytes not available; ignoring 8/4-bit env flags."
                )
                load_in_8bit = False
                load_in_4bit = False

        # Dtype
        if self.device == "cpu":
            dtype = torch.float32
        else:
            if self.device.startswith("cuda:"):
                try:
                    idx = int(self.device.split(":", 1)[1])
                except Exception:
                    idx = 0
            else:
                idx = 0
            try:
                torch.cuda.set_device(idx)
            except Exception:
                pass
            dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        model_kwargs = {
            "torch_dtype": self.torch_dtype if self.torch_dtype is not None else dtype,
            "trust_remote_code": TRUST_REMOTE_CODE,
            "low_cpu_mem_usage": True,
            "device_map": self.device_map,  # 'auto' for sharded, else None
        }
        # Attach quantization flags if requested
        if not CPU_ONLY:
            if load_in_4bit:
                model_kwargs["load_in_4bit"] = True
            elif load_in_8bit:
                model_kwargs["load_in_8bit"] = True

        try:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_id,
                **model_kwargs,
            )
        except Exception as e:
            if self.device_map == "auto":
                logging.error(
                    "Failed to load model with device_map='auto'. "
                    "Ensure `accelerate` is installed and configured."
                )
            raise

        self.model.eval()

        # Only move the model if not sharded
        if self.device_map is None:
            try:
                self.model.to(self.device)
            except Exception as e:
                logging.error(f"Failed to move model to device {self.device}: {e}")
                raise

        # Stop tokens: may be scalar or list
        eos = getattr(self.model.config, "eos_token_id", None)
        if eos is None:
            eos = getattr(self.tokenizer, "eos_token_id", None)
        self.stop_ids = (
            set(eos if isinstance(eos, (list, tuple, set)) else [eos])
            if eos is not None
            else set()
        )

        # Context window
        max_ctx = getattr(self.model.config, "max_position_embeddings", None)
        if max_ctx is None:
            max_ctx = getattr(self.tokenizer, "model_max_length", 2048)
        if isinstance(max_ctx, int) and max_ctx > 1_000_000:
            max_ctx = 2048
        self.max_ctx = int(max_ctx)

    # --- prompt preparation (optional chat template) -------------------------

    def _prepare_prompt(
        self, text: str, chat: bool, system: Optional[str] = None
    ) -> str:
        if chat and hasattr(self.tokenizer, "apply_chat_template"):
            try:
                messages = []
                if system:
                    messages.append({"role": "system", "content": system})
                messages.append({"role": "user", "content": text})
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                pass
        return text

    def _total_limit(self, prompt_len: int, max_length: int) -> int:
        """Clamp total length (prompt + generated) to the model's context window."""
        return max(0, min(int(max_length), self.max_ctx))

    @torch.inference_mode()
    def generate(
        self,
        text: str,
        max_length: int = 1024,
        top_k: int = 50,
        seed: Optional[int] = None,
        save_rate: Optional[callable] = None,
        chat: bool = False,
        system: Optional[str] = None,
    ) -> str:
        text = self._prepare_prompt(text, chat=chat, system=system)
        tok = self.tokenizer
        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        x_cpu = enc["input_ids"]
        prompt_len = x_cpu.size(1)

        total_limit = self._total_limit(prompt_len, max_length)
        if prompt_len > total_limit:
            x_cpu = x_cpu[:, -total_limit:]
            prompt_len = x_cpu.size(1)

        # Ensure inputs are on the same device as the embedding layer.
        if self.device_map is None:
            input_device = self.device
        else:
            input_device = None
            try:
                device_map = getattr(self.model, "hf_device_map", None)
                if isinstance(device_map, dict):
                    for key in (
                        "model.embed_tokens",
                        "embed_tokens",
                        "transformer.wte",
                        "model.wte",
                        "wte",
                    ):
                        dev = device_map.get(key)
                        if dev:
                            input_device = dev
                            break
                if input_device is None:
                    input_device = next(self.model.parameters()).device
            except Exception:
                try:
                    input_device = next(self.model.parameters()).device
                except Exception:
                    input_device = "cpu"

        x = x_cpu.to(input_device)

        gen = torch.Generator(
            device=("cpu" if str(input_device).startswith("cpu") else str(input_device))
        )
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        gen.manual_seed(seed)

        vocab_size = int(self.model.config.vocab_size)
        k = max(1, min(int(top_k) if top_k is not None else 50, vocab_size))

        count = 0
        start = time.perf_counter()

        outputs = self.model(input_ids=x, use_cache=True)
        past = outputs.past_key_values
        last_logits = outputs.logits[:, -1, :]

        while x.size(1) < total_limit:
            probs = F.softmax(last_logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1)
            next_idx = torch.multinomial(topk_probs, num_samples=1, generator=gen)
            next_token = torch.gather(topk_indices, -1, next_idx)  # (1,1)

            token_id = int(next_token.item())
            if token_id in self.stop_ids:
                break

            next_token = next_token.to(input_device)
            outputs = self.model(
                input_ids=next_token, past_key_values=past, use_cache=True
            )
            past = outputs.past_key_values
            last_logits = outputs.logits[:, -1, :]

            x = torch.cat((x, next_token), dim=1)
            count += 1

        elapsed = time.perf_counter() - start
        if save_rate:
            try:
                save_rate(count / max(elapsed, 1e-8))
            except Exception:
                pass

        return tok.decode(x[0].tolist(), clean_up_tokenization_spaces=False)

    @torch.inference_mode()
    async def generate_async_for(
        self,
        text: str,
        max_length: int = 1024,
        top_k: int = 50,
        seed: Optional[int] = None,
        chat: bool = False,
        system: Optional[str] = None,
    ):
        text = self._prepare_prompt(text, chat=chat, system=system)
        tok = self.tokenizer
        enc = tok(text, return_tensors="pt", add_special_tokens=False)
        x_cpu = enc["input_ids"]
        prompt_len = x_cpu.size(1)

        total_limit = self._total_limit(prompt_len, max_length)
        if prompt_len > total_limit:
            x_cpu = x_cpu[:, -total_limit:]
            prompt_len = x_cpu.size(1)

        # Ensure inputs are on the same device as the embedding layer.
        if self.device_map is None:
            input_device = self.device
        else:
            input_device = None
            try:
                device_map = getattr(self.model, "hf_device_map", None)
                if isinstance(device_map, dict):
                    for key in (
                        "model.embed_tokens",
                        "embed_tokens",
                        "transformer.wte",
                        "model.wte",
                        "wte",
                    ):
                        dev = device_map.get(key)
                        if dev:
                            input_device = dev
                            break
                if input_device is None:
                    input_device = next(self.model.parameters()).device
            except Exception:
                try:
                    input_device = next(self.model.parameters()).device
                except Exception:
                    input_device = "cpu"

        x = x_cpu.to(input_device)

        gen = torch.Generator(
            device=("cpu" if str(input_device).startswith("cpu") else str(input_device))
        )
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        gen.manual_seed(seed)

        vocab_size = int(self.model.config.vocab_size)
        k = max(1, min(int(top_k) if top_k is not None else 50, vocab_size))

        outputs = self.model(input_ids=x, use_cache=True)
        past = outputs.past_key_values
        last_logits = outputs.logits[:, -1, :]

        count = 0
        start = time.perf_counter()

        # For robust streaming across tokenizers (SentencePiece, byte-level),
        # decode cumulatively and emit only the delta since last step. This
        # preserves proper spaces and normalization that may not be present
        # when decoding single tokens in isolation.
        generated_token_ids: List[int] = []
        decoded_so_far: str = ""

        while x.size(1) < total_limit:
            probs = F.softmax(last_logits, dim=-1)
            topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1)
            next_idx = torch.multinomial(topk_probs, num_samples=1, generator=gen)
            next_token = torch.gather(topk_indices, -1, next_idx)

            token_id = int(next_token.item())
            if token_id in self.stop_ids:
                break

            next_token = next_token.to(input_device)
            outputs = self.model(
                input_ids=next_token, past_key_values=past, use_cache=True
            )
            past = outputs.past_key_values
            last_logits = outputs.logits[:, -1, :]

            x = torch.cat((x, next_token), dim=1)
            count += 1

            # Accumulate and decode to compute the incremental fragment.
            generated_token_ids.append(token_id)
            full_decoded = tok.decode(
                generated_token_ids,
                clean_up_tokenization_spaces=False,
            )
            fragment = full_decoded[len(decoded_so_far) :]
            decoded_so_far = full_decoded

            if fragment:
                if not _is_stream_printable(fragment):
                    yield -1
                    break
                yield fragment
            await asyncio.sleep(0)

        elapsed = time.perf_counter() - start
        logging.debug(
            f"[gpt_oss.generate_async_for] Generated {count} tokens in "
            f"{elapsed:.2f}s (~{count / max(elapsed, 1e-8):.2f} tok/s)"
        )


# =============================================================================
# Model loading (thread-safe, large-model aware)
# =============================================================================


def get_or_load_model(
    model_name_or_id: Optional[str], device: Optional[str]
) -> "CausalModel":
    model_id, display_name = resolve_model_id(model_name_or_id)
    dev_key = _device_key(device)
    cache_key = (model_id, dev_key)

    lock = _MODEL_CACHE_LOCKS[cache_key]
    with lock:
        if cache_key in MODEL_CACHE:
            return MODEL_CACHE[cache_key]

        if dev_key == "cpu":
            resolved_device = "cpu"
            dtype = torch.float32
            device_map = None
        else:
            if dev_key == "cuda":
                resolved_device = "cuda"
                device_map = "auto" if NUM_GPUS > 0 else None
            elif dev_key.startswith("cuda"):
                resolved_device = dev_key  # e.g. 'cuda:0'
                device_map = None
            else:
                raise ValueError(f"Unknown device '{dev_key}'")

            dtype = None  # let CausalModel decide

        logging.info(
            f"Loading model '{display_name}' ({model_id}) on {resolved_device} "
            f"(device_map={device_map})"
        )

        model = CausalModel(
            model_id=model_id,
            device=resolved_device,
            torch_dtype=dtype,
            device_map=device_map,
        )
        MODEL_CACHE[cache_key] = model
        return model


# =============================================================================
# HTTP App
# =============================================================================


class CausalModelApp(HttpApp):
    prefix = "llm"

    def __init__(self, server: HttpServer) -> None:
        super().__init__(server)
        self.task = None
        self.keep_alive = None

    @classmethod
    def init_once(cls):
        # Intentionally lazy-load models on first request due to model size.
        pass

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
        server = self.server
        transport = getattr(server, "transport", None)
        return transport is not None

    def write(self, response_bytes):
        transport = getattr(self.server, "transport", None)
        if transport is not None:
            transport.write(response_bytes)
            return True
        return False

    async def generate_response(
        self, request: Request, text: str, **kwds: Dict
    ) -> None:
        response = request.response
        response.code = 200
        response.message = "OK"
        response.chunked_response = True
        response.content_type = "text/plain"

        kwds = kwds or {}

        # Request params
        max_length = int(kwds.get("max_length", 100) or 100)
        max_length = max(1, min(max_length, 1_000_000))
        top_k = int(kwds.get("top_k", 50) or 50)
        top_k = max(1, min(top_k, 100_000))
        chat_flag = str(kwds.get("chat", "0")).lower() in ("1", "true", "yes")
        system_prompt = kwds.get("system", None)

        seed = kwds.get("seed", None)
        if seed is not None:
            try:
                seed = int(seed)
            except ValueError:
                seed = None
        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        device = kwds.get("device", None)  # 'cpu', 'cuda', 'cuda:0', etc.
        model_name_or_id = kwds.get("model", None)

        # Acquire model lazily with robust error handling
        try:
            model = get_or_load_model(model_name_or_id, device)
        except Exception as e:
            logging.exception("Failed to load model")
            response.code = 500
            response.message = "Internal Server Error"
            response.chunked_response = True
            response.content_type = "text/plain"
            self.write(bytes(response))
            response.send_chunk(f"Model load failed: {e}")
            response.end_chunks()
            return

        # Precompute prompt token length (after optional chat templating)
        try:
            prepared = model._prepare_prompt(text, chat=chat_flag, system=system_prompt)
            enc = model.tokenizer(
                prepared, return_tensors="pt", add_special_tokens=False
            )
            prompt_len = enc["input_ids"].size(1)
        except Exception:
            prompt_len = -1

        # Determine GPU headers
        gpu_index = None
        if (
            not CPU_ONLY
            and isinstance(model.device, str)
            and model.device.startswith("cuda")
        ):
            if ":" in model.device:
                try:
                    gpu_index = int(model.device.split(":", 1)[1])
                except Exception:
                    gpu_index = None
            else:
                try:
                    gpu_index = torch.cuda.current_device()
                except Exception:
                    gpu_index = None

        gpu_http_headers = (
            CUDA_GPU_PROPS_HTTP_HEADERS.get(f"cuda:{gpu_index}")
            if gpu_index is not None
            else None
        )

        # Response headers
        model_id, display_name = resolve_model_id(model_name_or_id)
        expose_headers = (
            "Access-Control-Expose-Headers: "
            "X-Max-Length, "
            "X-Top-K, "
            "X-Seed, "
            "X-Model-Alias, "
            "X-Model-ID, "
            "X-Model-Device, "
            "X-Context-Window, "
            "X-Prompt-Tokens, "
            "X-Context-Clipped, "
            "X-Chat-Template"
        )
        if gpu_http_headers is not None:
            expose_headers += f", {', '.join(gpu_http_headers.keys())}"

        total_limit = min(max_length, model.max_ctx)
        context_clipped = prompt_len != -1 and prompt_len > total_limit

        other_headers = [
            f"X-Max-Length: {max_length}",
            f"X-Top-K: {top_k}",
            f"X-Seed: {seed}",
            f"X-Model-Alias: {display_name}",
            f"X-Model-ID: {model.model_id}",
            f"X-Model-Device: {model.device}",
            f"X-Context-Window: {model.max_ctx}",
            f"X-Prompt-Tokens: {prompt_len}",
            f"X-Context-Clipped: {'true' if context_clipped else 'false'}",
            f"X-Chat-Template: {'true' if chat_flag and hasattr(model.tokenizer, 'apply_chat_template') else 'false'}",
        ]
        if gpu_http_headers is not None:
            other_headers.extend([f"{k}: {v}" for k, v in gpu_http_headers.items()])

        response.other_headers.append(expose_headers)
        response.other_headers.extend(other_headers)

        # Enable TCP_NODELAY if available
        try:
            response.enable_tcp_nodelay()
            enabled_nodelay = True
        except Exception as e:
            logging.error(f"Error enabling TCP_NODELAY: {e}")
            enabled_nodelay = False

        # Write the chunked header immediately
        response_bytes = bytes(response)
        if not self.write(response_bytes):
            return

        # Send initial prompt (original text, not the templated one)
        try:
            response.send_chunk(text)
        except Exception:
            return

        # Stream generation
        try:
            generate_tokens = model.generate_async_for(
                text,
                max_length=max_length,
                top_k=top_k,
                seed=seed,
                chat=chat_flag,
                system=system_prompt,
            )
            async for decoded_token in generate_tokens:
                if decoded_token == -1:
                    response.send_chunk(OOPS_NON_PRINTABLE_ENCOUNTERED)
                    break
                if not self.is_connected():
                    break
                response.send_chunk(decoded_token)
        except Exception as e:
            logging.exception("Error during streaming generation")
            try:
                response.send_chunk(f"\n[error] generation failed: {e}")
            except Exception:
                pass

        # End chunked transfer
        try:
            response.end_chunks()
        except Exception:
            pass

        if enabled_nodelay:
            try:
                response.disable_tcp_nodelay()
            except Exception as e:
                logging.error(f"Error disabling TCP_NODELAY: {e}")

    @route
    def generate(self, request: Request, *args: List, **kwds: Dict) -> None:
        prompt = args[0]
        loop = asyncio.get_running_loop()
        assert self.task is None
        coro = self.generate_response(request, prompt, **kwds)
        self.task = loop.create_task(coro)
        self.keep_alive = request.keep_alive
        self.task.add_done_callback(self._task_complete)


# =============================================================================
# CLI
# =============================================================================


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(description="Run the GPT-OSS/HF CausalLM module.")
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=LLM_DEFAULT_MODEL,
        help="Model alias or HF repo id (e.g., gpt-oss-20b or qwen/qwen3-4b-2507).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cpu | cuda | cuda:N. Default: cuda (with sharding) if available, else cpu.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum total length (prompt + generated).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for generation.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="The quick brown fox jumps over the lazy dog",
        help="Prompt for generation.",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="System prompt (only used when --chat is enabled).",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        default=None,
        help="Read prompt from the given file; overrides --prompt.",
    )
    parser.add_argument(
        "--system-file",
        type=str,
        default=None,
        help="Read system prompt from the given file; overrides --system.",
    )
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Use tokenizer.apply_chat_template (if available) for instruction/chat models.",
    )
    parser.add_argument(
        "--wrap",
        type=int,
        default=80,
        help="Wrap width for text output (0 disables).",
    )
    return parser.parse_args()


def main():
    import textwrap

    args = parse_arguments()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # If provided, read prompt text from file and override --prompt
    if getattr(args, "prompt_file", None):
        try:
            with open(args.prompt_file, "r", encoding="utf-8") as f:
                args.prompt = f.read()
        except Exception as e:
            logging.error(f"Failed to read prompt file '{args.prompt_file}': {e}")
            raise SystemExit(1)

    # If provided, read system text from file and override --system
    if getattr(args, "system_file", None):
        try:
            with open(args.system_file, "r", encoding="utf-8") as f:
                args.system = f.read()
        except Exception as e:
            logging.error(f"Failed to read system file '{args.system_file}': {e}")
            raise SystemExit(1)

    model = get_or_load_model(args.model, args.device)

    rates: List[float] = []
    output = model.generate(
        args.prompt,
        max_length=args.max_length,
        top_k=args.top_k,
        seed=args.seed,
        chat=args.chat,
        system=args.system,
        save_rate=lambda r: rates.append(r),
    )

    if args.wrap and args.wrap > 0:
        output = textwrap.fill(output, width=args.wrap)
    logging.info(f"Output:\n{output}")


if __name__ == "__main__":
    main()

# vim:set ts=8 sw=4 sts=4 tw=78 et:
