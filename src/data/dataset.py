from __future__ import annotations

import os
from typing import Iterable
import numpy as np
from ..tokenizer import Tokenizer


def load_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def create_dataset(path: str, tokenizer: Tokenizer, seq_len: int, batch_size: int) -> Iterable[np.ndarray]:
    text = load_text(path)
    tokenizer.fit_texts([text])
    token_ids = tokenizer.encode(text)
    total = len(token_ids) - seq_len
    if total <= 0:
        raise ValueError("Text is too short for the requested sequence length.")
    sequences = [token_ids[i : i + seq_len] for i in range(total)]
    for start in range(0, len(sequences), batch_size):
        batch = np.array(sequences[start : start + batch_size], dtype=np.int64)
        if batch.shape[0] == batch_size:
            yield batch
