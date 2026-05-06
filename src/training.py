from __future__ import annotations

import numpy as np
from .model import SimpleTransformer, TransformerConfig
from typing import Iterable

class TrainingConfig:
    def __init__(self, batch_size: int = 8, epochs: int = 1, learning_rate: float = 1e-3):
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate


def train(model: SimpleTransformer, dataset: Iterable[np.ndarray], config: TrainingConfig) -> SimpleTransformer:
    """Minimal training loop placeholder that iterates through data."""
    for epoch in range(config.epochs):
        for step, batch in enumerate(dataset):
            logits = model.forward(batch)
            targets = batch
            # Use negative log-likelihood surrogate for diagnostics.
            probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
            probs /= np.sum(probs, axis=-1, keepdims=True)
            loss = -np.log(probs[np.arange(batch.shape[0])[:, None], np.arange(batch.shape[1])[None, :], targets] + 1e-9).mean()
            if step % 10 == 0:
                print(f"epoch={epoch+1} step={step} loss={loss:.4f}")
    return model
