from __future__ import annotations

import numpy as np
from .model import SimpleTransformer
from .tokenizer import Tokenizer


def generate_text(model: SimpleTransformer, tokenizer: Tokenizer, prompt: str, max_new_tokens: int = 16, temperature: float = 1.0, top_k: int = 5) -> str:
    if tokenizer.vocab_size == 0:
        raise ValueError("Tokenizer must be initialized before inference.")
    tokens = tokenizer.encode(prompt, max_length=model.config.seq_len)
    tokens = tokens[-model.config.seq_len :]
    for _ in range(max_new_tokens):
        input_tokens = np.array([tokens], dtype=np.int64)
        logits = model.forward(input_tokens)[0, -1]
        scaled = logits / max(temperature, 1e-6)
        top_indices = np.argsort(scaled)[-top_k:]
        probs = np.exp(scaled[top_indices] - np.max(scaled[top_indices]))
        probs /= probs.sum()
        next_token = np.random.choice(top_indices, p=probs)
        tokens.append(int(next_token))
        if len(tokens) > model.config.seq_len:
            tokens = tokens[-model.config.seq_len :]
    return tokenizer.decode(tokens)
