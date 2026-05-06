import numpy as np
from src.model import TransformerConfig, SimpleTransformer


def test_model_forward_shape():
    config = TransformerConfig(vocab_size=32, seq_len=16, d_model=32, n_heads=4, n_layers=1, d_ff=64)
    model = SimpleTransformer(config)
    tokens = np.zeros((2, config.seq_len), dtype=np.int64)
    logits = model.forward(tokens)
    assert logits.shape == (2, config.seq_len, config.vocab_size)
