from __future__ import annotations

import numpy as np
from dataclasses import dataclass

@dataclass
class TransformerConfig:
    vocab_size: int
    seq_len: int
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    d_ff: int = 256

class SimpleTransformer:
    def __init__(self, config: TransformerConfig) -> None:
        self.config = config
        self._rng = np.random.default_rng(42)
        self.token_embedding = self._rng.standard_normal((config.vocab_size, config.d_model), dtype=np.float32)
        self.position_embedding = self._rng.standard_normal((config.seq_len, config.d_model), dtype=np.float32)
        self.layers = [
            {
                "wq": self._init_weight((config.d_model, config.d_model)),
                "wk": self._init_weight((config.d_model, config.d_model)),
                "wv": self._init_weight((config.d_model, config.d_model)),
                "wo": self._init_weight((config.d_model, config.d_model)),
                "w1": self._init_weight((config.d_model, config.d_ff)),
                "w2": self._init_weight((config.d_ff, config.d_model)),
                "ln1_gain": np.ones((config.d_model,), dtype=np.float32),
                "ln1_bias": np.zeros((config.d_model,), dtype=np.float32),
                "ln2_gain": np.ones((config.d_model,), dtype=np.float32),
                "ln2_bias": np.zeros((config.d_model,), dtype=np.float32),
            }
            for _ in range(config.n_layers)
        ]
        self.final_norm_gain = np.ones((config.d_model,), dtype=np.float32)
        self.final_norm_bias = np.zeros((config.d_model,), dtype=np.float32)
        self.unembed = self._init_weight((config.d_model, config.vocab_size))

    def _init_weight(self, shape: tuple[int, ...]) -> np.ndarray:
        return self._rng.standard_normal(shape, dtype=np.float32) * 0.02

    def _layer_norm(self, x: np.ndarray, gain: np.ndarray, bias: np.ndarray) -> np.ndarray:
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        normalized = (x - mean) / np.sqrt(var + 1e-5)
        return normalized * gain + bias

    def _scaled_dot_product_attention(self, q: np.ndarray, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        dim = q.shape[-1]
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dim)
        mask = np.triu(np.full(scores.shape, -1e9, dtype=np.float32), 1)
        scores = scores + mask
        weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        weights = weights / np.sum(weights, axis=-1, keepdims=True)
        return weights @ v

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        batch_size, seq_len, d_model = x.shape
        head_dim = d_model // self.config.n_heads
        return x.reshape(batch_size, seq_len, self.config.n_heads, head_dim).transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        batch_size, heads, seq_len, head_dim = x.shape
        return x.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, heads * head_dim)

    def _self_attention(self, x: np.ndarray, layer: dict[str, np.ndarray]) -> np.ndarray:
        q = x @ layer["wq"]
        k = x @ layer["wk"]
        v = x @ layer["wv"]
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)
        attn = self._scaled_dot_product_attention(q, k, v)
        attn = self._merge_heads(attn)
        return attn @ layer["wo"]

    def _feed_forward(self, x: np.ndarray, layer: dict[str, np.ndarray]) -> np.ndarray:
        return np.maximum(0, x @ layer["w1"]) @ layer["w2"]

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        batch_size, seq_len = tokens.shape
        assert seq_len <= self.config.seq_len, "Sequence length exceeds model capacity."
        x = self.token_embedding[tokens] + self.position_embedding[:seq_len]
        for layer in self.layers:
            residual = x
            x = self._layer_norm(x, layer["ln1_gain"], layer["ln1_bias"])
            x = residual + self._self_attention(x, layer)
            residual = x
            x = self._layer_norm(x, layer["ln2_gain"], layer["ln2_bias"])
            x = residual + self._feed_forward(x, layer)
        x = self._layer_norm(x, self.final_norm_gain, self.final_norm_bias)
        logits = x @ self.unembed
        return logits

    def predict(self, tokens: np.ndarray) -> np.ndarray:
        logits = self.forward(tokens)
        return np.argmax(logits, axis=-1)
