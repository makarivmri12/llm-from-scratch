from __future__ import annotations

from typing import List

class Tokenizer:
    """Simple byte-level tokenizer with fixed vocabulary size."""

    def __init__(self) -> None:
        self.vocab = {bytes([i]): i for i in range(256)}
        self.inv_vocab = {i: bytes([i]) for i in range(256)}

    def fit_texts(self, texts: List[str]) -> None:
        # byte-level tokenizer does not require fitting beyond the fixed 256-token vocabulary
        self.vocab = {bytes([i]): i for i in range(256)}
        self.inv_vocab = {i: bytes([i]) for i in range(256)}

    def encode(self, text: str, max_length: int | None = None) -> List[int]:
        tokens = [b for b in text.encode("utf-8", errors="replace")]
        if max_length is not None:
            tokens = tokens[:max_length]
            tokens += [0] * max(0, max_length - len(tokens))
        return tokens

    def decode(self, token_ids: List[int]) -> str:
        return bytes(token_ids).decode("utf-8", errors="replace")

    @property
    def vocab_size(self) -> int:
        return 256
