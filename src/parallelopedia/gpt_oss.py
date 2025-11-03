"""
GPT-OSS inference server and CLI, analogous to `gpt2.py`'s Gpt2App.

This module provides a lightweight wrapper around Hugging Face transformers
to run OpenAI's open-source GPT-OSS models (20B and 120B variants) with
token-by-token streaming over the existing HttpServer framework.
"""

# =============================================================================
# Imports
# =============================================================================

import asyncio
import logging
import os
import random
import string
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch.nn import functional as F

from parallelopedia.http.server import (
    HttpApp,
    HttpServer,
    Request,
    route,
)

# Transformers stack
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)


# =============================================================================
# Configuration
# =============================================================================

# Model id mapping; adjust to the actual Hub ids you intend to use.
MODEL_VARIANTS = {
    'gpt-oss-20b': 'openai/gpt-oss-20b',
    'gpt-oss-120b': 'openai/gpt-oss-120b',
}


# If PARALLELOPEDIA_CPU_ONLY is set, don't try to use CUDA.
if 'PARALLELOPEDIA_CPU_ONLY' in os.environ or not torch.cuda.is_available():
    CPU_ONLY = True
    NUM_GPUS = 0
else:
    CPU_ONLY = False
    NUM_GPUS = torch.cuda.device_count()


# =============================================================================
# Helpers
# =============================================================================

def torch_cuda_device_props_to_http_response_header_dict(
    gpu_props,
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


# =============================================================================
# Globals
# =============================================================================

PRINTABLE = set(c for c in string.printable)
OOPS_NON_PRINTABLE_ENCOUNTERED = (
    'Oops! Non-printable token encountered.  Generation terminated.'
)

CUDA_GPU_PROPS: Dict[str, object] = {}
CUDA_GPU_PROPS_HTTP_HEADERS: Dict[str, Dict[str, str]] = {}
if not CPU_ONLY:
    for i in range(NUM_GPUS):
        props = torch.cuda.get_device_properties(i)
        CUDA_GPU_PROPS[f'cuda:{i}'] = props
        CUDA_GPU_PROPS_HTTP_HEADERS[f'cuda:{i}'] = (
            torch_cuda_device_props_to_http_response_header_dict(props)
        )


# Cache models lazily by (variant, device_key)
MODEL_CACHE: Dict[Tuple[str, str], "GptOss"] = {}


def _normalize_variant_name(name: Optional[str]) -> str:
    if not name:
        return 'gpt-oss-20b'
    name = name.strip().lower()
    if name in MODEL_VARIANTS:
        return name
    # Accept short names like "20b"/"120b"
    if name == '20b':
        return 'gpt-oss-20b'
    if name == '120b':
        return 'gpt-oss-120b'
    # Fallback to 20b
    return 'gpt-oss-20b'


def _device_key(device: Optional[str]) -> str:
    if CPU_ONLY:
        return 'cpu'
    if not device:
        return 'cuda'
    return device


def get_or_load_model(variant: str, device: Optional[str]) -> "GptOss":
    variant = _normalize_variant_name(variant)
    dev_key = _device_key(device)
    cache_key = (variant, dev_key)
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    model_id = MODEL_VARIANTS[variant]

    # Resolve device and dtype
    if CPU_ONLY:
        resolved_device = 'cpu'
        dtype = torch.float32
    else:
        if dev_key == 'cuda' or dev_key.startswith('cuda'):
            resolved_device = dev_key
        else:
            resolved_device = 'cuda'
        # Prefer bfloat16 when supported, else float16
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

    logging.info(
        f"Loading GPT-OSS variant '{variant}' ({model_id}) on {resolved_device}"
    )
    model = GptOss(
        model_id=model_id,
        device=resolved_device,
        torch_dtype=dtype,
    )
    MODEL_CACHE[cache_key] = model
    return model


# =============================================================================
# Classes
# =============================================================================


@dataclass
class GptOss:
    model_id: str
    device: str = 'cuda'
    torch_dtype: Optional[torch.dtype] = None

    def __post_init__(self) -> None:
        tokenizer_kwargs = dict(use_fast=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, **tokenizer_kwargs)

        # Ensure tokenizer has an EOS token id
        if getattr(self.tokenizer, 'eos_token_id', None) is None:
            # Attempt to set eos token if missing
            try:
                self.tokenizer.eos_token = self.tokenizer.sep_token or self.tokenizer.pad_token
            except Exception:
                pass

        model_kwargs = {
            'torch_dtype': self.torch_dtype if self.torch_dtype is not None else 'auto',
            'device_map': None,  # explicit .to(device) below
            'trust_remote_code': True,
            'low_cpu_mem_usage': True,
        }
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs,
        )
        self.model.eval()
        self.model.to(self.device)

        self.printable = PRINTABLE
        self.stop_token = getattr(self.tokenizer, 'eos_token_id', None)

    def generate(
        self,
        text: str,
        max_length: int = 1024,
        top_k: int = 50,
        seed: Optional[int] = None,
        save_rate: Optional[callable] = None,
    ) -> str:
        tokenizer = self.tokenizer
        device = self.device

        # Encode prompt -> tensor of shape (1, T)
        enc = tokenizer(text, return_tensors='pt', add_special_tokens=False)
        x = enc['input_ids'].to(device)

        # Randomness for sampling
        sample_rng = torch.Generator(device=device)
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        sample_rng.manual_seed(seed)

        count = 0
        start = time.perf_counter()
        with torch.no_grad():
            while x.size(1) < max_length:
                outputs = self.model(input_ids=x)
                logits = outputs.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                topk_probs, topk_indices = torch.topk(probs, k=min(top_k, probs.size(-1)))
                next_idx = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)
                next_token = torch.gather(topk_indices, -1, next_idx)  # (1, 1)

                next_token_item = next_token.item()
                if self.stop_token is not None and next_token_item == self.stop_token:
                    break

                x = torch.cat((x, next_token), dim=1)
                count += 1

        elapsed = time.perf_counter() - start
        tokens_per_sec = float(max(count, 1)) / max(elapsed, 1e-8)
        if save_rate:
            try:
                save_rate(tokens_per_sec)
            except Exception:
                pass

        return tokenizer.decode(x[0].tolist(), clean_up_tokenization_spaces=False)

    async def generate_async_for(
        self,
        text: str,
        max_length: int = 1024,
        top_k: int = 50,
        seed: Optional[int] = None,
    ):
        tokenizer = self.tokenizer
        device = self.device

        enc = tokenizer(text, return_tensors='pt', add_special_tokens=False)
        x = enc['input_ids'].to(device)

        sample_rng = torch.Generator(device=device)
        if seed is None:
            seed = random.randint(0, 2**32 - 1)
        sample_rng.manual_seed(seed)

        count = 0
        start = time.perf_counter()

        while x.size(1) < max_length:
            with torch.no_grad():
                outputs = self.model(input_ids=x)
                logits = outputs.logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                k = min(top_k, probs.size(-1))
                topk_probs, topk_indices = torch.topk(probs, k=k, dim=-1)
                next_idx = torch.multinomial(topk_probs, num_samples=1, generator=sample_rng)
                next_token = torch.gather(topk_indices, -1, next_idx)  # (1,1)

            token_id = next_token.item()
            if self.stop_token is not None and token_id == self.stop_token:
                break

            # Append token to sequence first, then decode the single new piece
            x = torch.cat((x, next_token), dim=1)
            count += 1

            fragment = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
            if not all(c in self.printable for c in fragment):
                yield -1
                break

            yield fragment
            await asyncio.sleep(0)

        elapsed = time.perf_counter() - start
        logging.debug(
            f"[gpt_oss.generate_async_for] Generated {count} tokens in "
            f"{elapsed:.2f}s (~{count / max(elapsed, 1e-8):.2f} tok/s)"
        )


class GptOssApp(HttpApp):

    def __init__(self, server: HttpServer) -> None:
        super().__init__(server)
        self.printable = PRINTABLE
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

        max_length = min(int(kwds.get('max_length', 100) or 100), 32768)
        top_k = min(int(kwds.get('top_k', 50) or 50), 100)

        seed = kwds.get('seed', None)
        if seed:
            try:
                seed = int(seed)
            except ValueError:
                seed = None
        if not seed:
            seed = random.randint(0, 2**32 - 1)

        device = kwds.get('device', None)  # e.g. 'cuda', 'cuda:0', etc.
        variant = kwds.get('model', 'gpt-oss-20b')  # 'gpt-oss-20b'|'gpt-oss-120b'

        # Acquire model lazily
        model = get_or_load_model(variant, device)

        # Determine GPU headers
        gpu_index = None
        if not CPU_ONLY and isinstance(model.device, str) and model.device.startswith('cuda:'):
            try:
                gpu_index = int(model.device.split(':', 1)[1])
            except Exception:
                gpu_index = None

        gpu_http_headers = None
        if gpu_index is not None:
            gpu_http_headers = CUDA_GPU_PROPS_HTTP_HEADERS.get(f'cuda:{gpu_index}')

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
            f'X-Model-Name: {_normalize_variant_name(variant)}',
            f'X-Model-Device: {model.device}',
        ]
        if gpu_http_headers is not None:
            other_headers.extend([f'{k}: {v}' for k, v in gpu_http_headers.items()])

        response.other_headers.append(expose_headers)
        response.other_headers.extend(other_headers)

        # Enable TCP_NODELAY if available
        try:
            response.enable_tcp_nodelay()
            enabled_nodelay = True
        except Exception as e:
            logging.error(f'Error enabling TCP_NODELAY: {e}')
            enabled_nodelay = False

        # Write the chunked header immediately
        response_bytes = bytes(response)
        if not self.write(response_bytes):
            return

        # Send initial prompt
        response.send_chunk(text)

        generate_tokens = model.generate_async_for(
            text,
            max_length=max_length,
            top_k=top_k,
            seed=seed,
        )
        async for decoded_token in generate_tokens:
            if decoded_token == -1:
                response.send_chunk(OOPS_NON_PRINTABLE_ENCOUNTERED)
                break

            if not self.is_connected():
                break

            response.send_chunk(decoded_token)

        response.end_chunks()

        if enabled_nodelay:
            try:
                response.disable_tcp_nodelay()
            except Exception as e:
                logging.error(f'Error disabling TCP_NODELAY: {e}')

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
    parser = argparse.ArgumentParser(description='Run the GPT-OSS module.')
    parser.add_argument(
        '--log-level', type=str, default='INFO',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        help='Set the logging level.',
    )
    parser.add_argument(
        '--model', type=str, default='gpt-oss-20b',
        choices=['gpt-oss-20b', 'gpt-oss-120b'],
        help='Select the GPT-OSS model variant.',
    )
    parser.add_argument(
        '--device', type=str, default='cuda',
        help='Select the device (e.g., cuda, cuda:0).',
    )
    parser.add_argument(
        '--max-length', type=int, default=512,
        help='Maximum total length (prompt + generated).',
    )
    parser.add_argument(
        '--top-k', type=int, default=50,
        help='Top-k sampling.',
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for generation.',
    )
    parser.add_argument(
        '--prompt', type=str,
        default="The quick brown fox jumps over the lazy dog",
        help='Prompt for generation.',
    )
    parser.add_argument(
        '--wrap', type=int, default=80,
        help='Wrap width for text output (0 disables).',
    )
    return parser.parse_args()


def main():
    import textwrap

    args = parse_arguments()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(levelname)s - %(message)s',
    )

    model = get_or_load_model(args.model, args.device)

    save_rates: List[float] = []
    output = model.generate(
        args.prompt,
        max_length=args.max_length,
        top_k=args.top_k,
        seed=args.seed,
        save_rate=lambda r: save_rates.append(r),
    )

    if args.wrap and args.wrap > 0:
        output = textwrap.fill(output, width=args.wrap)
    logging.info(f'Output:\n{output}')


if __name__ == '__main__':
    main()

# vim:set ts=8 sw=4 sts=4 tw=78 et:


