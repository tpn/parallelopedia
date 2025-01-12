"""
This file is based on Andrej Karpathy's `build-nanogpt` github repo;
specifically, the `train_gpt2.py` file.
"""
#===============================================================================
# Imports
#===============================================================================
import time
import asyncio
import logging
import tiktoken
from dataclasses import dataclass

from typing import (
    Optional,
)

import torch
import torch.nn as nn
from torch.nn import functional as F

from .util import (
    ElapsedTimer,
)

#===============================================================================
# Globals
#===============================================================================
DEFAULT_MANUAL_SEED = 42

#===============================================================================
# Setup
#===============================================================================

# Use bfloat16 for matmul precision where possible.
torch.set_float32_matmul_precision('high')

#===============================================================================
# Classes
#===============================================================================

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is "number of heads", hs is "head size", and C (number of channels) = nh * hs
        # e.g. in GPT-2 (124M), n_head=12, hs=64, so nh*hs=C=768 channels in the Transformer
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # flash attention
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd)
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

        vocab_size (int): Number of tokens. Includes 50,000 BPE merges,
            256 byte tokens, and 1 <|endoftext|> token.

        n_layer (int): Number of layers.

        n_head (int): Number of attention heads.

        n_embd (int): Embedding dimension.
    """
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):

    def __init__(self, config : GPTConfig, device : str, manual_seed : int):
        """
        Initializes a GPT model.

        Arguments:

            config (GPTConfig): Supplies the configuration for the model.

            device (str): Supplies the device to use for the model, e.g. "cpu"
                or "cuda".

            manual_seed (int): Supplies the manual seed to use for the model.

        """
        super().__init__()
        self.config = config
        self.device = device
        self.manual_seed = manual_seed

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

        # Obtain the tokenizer for GPT2, and resolve the stop token.
        enc = tiktoken.get_encoding("gpt2")
        stop_string = '<|endoftext|>'
        stop_token = enc.n_vocab - 1
        actual = enc.decode([stop_token])
        assert actual == stop_string, f"expected {stop_string}, got {actual}"
        self.enc = enc
        self.stop_token = stop_token

    def _init_weights(self, module):
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
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_local_pretrained(cls, model_path: str,
                              map_location: Optional[str] = None,
                              manual_seed: Optional[int] = None):
        """
        Load a GPT model from a local checkpoint file (e.g. 'model_19072.pt').

        Arguments:

            model_path (str): Supplies the path to the .pt checkpoint file
                that was produced by `torch.save()`.

            map_location (str): Optionally supplies the device to map the
                loaded tensor parameters to.  If None, "cuda" will be used
                if available, otherwise "cpu".

            manual_seed (int): Optionally supplies the manual seed to use for
                the model.  If None, `DEFAULT_MANUAL_SEED` will be used.

        """
        # Extract the stored config.  This is the same GPTConfig instance that
        # was saved in 'train_gpt2.py'.
        config = checkpoint['config']

        if manual_seed is None:
            manual_seed = DEFAULT_MANUAL_SEED

        if map_location is None:
            if torch.cuda.is_available():
                map_location = "cuda"
            else:
                map_location = "cpu"

        timer = ElapsedTimer()

        # Load the checkpoint.
        with timer:
            with torch.serialization.safe_globals([GPTConfig]):
                checkpoint = torch.load(model_path, map_location=map_location)
        logging.info(
            f'Loaded {model_path} checkpoint in {timer.elapsed:.3f} seconds.'
        )

        # Initialize a new GPT instance using the config.
        model = cls(config, device=map_location, manual_seed=manual_seed)

        # Load in the state_dict containing all learned weights.
        with timer:
            model.load_state_dict(checkpoint['model'])
        logging.info(
            f'Loaded model weights in {timer.elapsed:.3f} seconds.'
        )

        # Set the model to eval mode.
        model.eval()

        logging.info(
            f"Loaded model from step {checkpoint['step']}, "
            f"val_loss {checkpoint['val_loss']}"
        )
        return model

    def generate(self, text: str, max_length: int = 1024, top_k: int = 50) -> str:
        """
        Generate text from the model, conditioned on `text`.

        Args:
            text (str): The prompt to condition on.
            max_length (int): Maximum total length (prompt + generated).
            top_k (int): Number of tokens to consider at each generation step (top-k).

        Returns:
            str: The generated text (including the initial prompt).
        """
        enc = self.enc
        stop_token = self.stop_token

        # Encode prompt -> tensor of shape (1, T)
        tokens = enc.encode(text)

        x = torch.tensor(
            tokens,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

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
            next_token = torch.gather(topk_indices, -1, next_idx)                          # (1, 1)

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

        # Decode the generated tokens and return the text, including the prompt.
        output_tokens = x[0].tolist()
        return enc.decode(output_tokens)

    async def generate_async_for(
        self,
        text: str,
        max_length: int = 1024,
        top_k: int = 50
    ):
        """
        Asynchronously generate text from the model, yielding tokens
        one at a time as soon as they are available.

        Arguments:
            text (str): Supplies the prompt to condition on.
            max_length (int): Maximum total length (prompt + generated).
            top_k (int): Number of tokens to consider at each generation step (top-k).

        Yields:
            str: The newly generated text token (decoded).
        """

        enc = self.enc
        stop_token = self.stop_token

        # Encode the prompt -> tensor of shape (1, T)
        tokens = enc.encode(text)
        x = torch.tensor(
            tokens,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(0)

        sample_rng = torch.Generator(device=self.device)
        sample_rng.manual_seed(self.manual_seed)

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
            next_token = torch.gather(topk_indices, -1, next_idx)                          # (1, 1)

            # If the next token is the stop token, we're done.
            if next_token.item() == stop_token:
                break

            # Append token to current sequence.  Although we only yield a
            # singular decoded token below, we still need to keep track of
            # the entire sequence for subsequent generation steps.
            x = torch.cat((x, next_token), dim=1)

            # Decode the newly-generated token, and yield it.
            new_text_fragment = enc.decode([next_token.item()])
            yield new_text_fragment

            # Yield control back to the event loop before continuing generation.
            await asyncio.sleep(0)

        elapsed = time.perf_counter() - start_time
        logging.debug(
            f"[generate_async_for] Generated {count} tokens in "
            f"{elapsed:.2f} seconds (~{count / elapsed:.2f} tok/s)"
        )

class Gpt2App(HttpApp):
    routes = make_routes()
    route = router(routes)

    def __init__(self, server : HttpServer, model : GPT) -> None:
        super().__init__(server)
        self.model = model

    async def generate_response(self, request : Request, text : str,
                                **kwds : Optional[dict] = None) -> None:
        assert isinstance(data, bytes)

        transport = self.server.transport
        response = request.response

        response.code = 200
        response.message = 'OK'
        response.chunked_response = True
        response.content_type = 'text/plain'

        # We want to enable TCP_NODELAY for the duration of the response.
        # This ensures packets are sent immediately without any internal
        # buffering.
        response.enable_tcp_nodelay()

        # Write the chunked header immediately.
        response_bytes = bytes(response)
        transport.write(response_bytes)

        if kwds is None:
            kwds = {}
        max_length = min(kwds.get('max_length', 1024), 1024)
        top_k = min(kwds.get('top_k', 50), 50)

        generate_tokens = self.model.generate_async_for(
            text,
            max_length=max_length,
            top_k=top_k,
        )
        async for decoded_token in generate_tokens:
            response.send_chunk(decoded_token)

        # Send the termination chunk.
        response.end_chunks()

        # Disable TCP_NODELAY now that the response is complete.
        response.disable_tcp_nodelay()

    @route
    def generate(self, request : Request, text : str, **kwds : dict) -> None:
        logging.debug(f'generate: {text}')
        loop = asyncio.get_running_loop()
        loop.create_task(self.generate_response(request))


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        model = GPT.from_local_pretrained(sys.argv[-1], map_location=device)
        model.to(device)
        result = model.generate("The quick brown fox")
        print(f'Returned: {result}')
    else:
        print('Using pretrained GPT...')
        model = GPT.from_pretrained('gpt2')
        model.to(device)
        result = model.generate("The quick brown fox")
        print(f'Returned: {result}')

