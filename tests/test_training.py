import numpy as np
from src.model import TransformerConfig, SimpleTransformer
from src.training import TrainingConfig, train


def test_training_loop_runs():
    config = TransformerConfig(vocab_size=16, seq_len=8, d_model=16, n_heads=2, n_layers=1, d_ff=32)
    model = SimpleTransformer(config)
    examples = [np.zeros((2, config.seq_len), dtype=np.int64) for _ in range(2)]
    training_config = TrainingConfig(batch_size=2, epochs=1, learning_rate=0.001)
    trained = train(model, examples, training_config)
    assert trained is model
