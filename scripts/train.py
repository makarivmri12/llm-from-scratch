from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from src.utils.config import load_config
from src.tokenizer import Tokenizer
from src.data.dataset import create_dataset
from src.model import TransformerConfig, SimpleTransformer
from src.training import TrainingConfig, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the llm-from-scratch model.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]
    train_cfg = config["training"]

    tokenizer = Tokenizer()
    transformer_config = TransformerConfig(
        vocab_size=model_cfg["vocab_size"],
        seq_len=model_cfg["seq_len"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"],
    )
    model = SimpleTransformer(transformer_config)
    dataset = create_dataset(train_cfg["dataset_path"], tokenizer, transformer_config.seq_len, train_cfg["batch_size"])
    training_config = TrainingConfig(
        batch_size=train_cfg["batch_size"],
        epochs=train_cfg["epochs"],
        learning_rate=train_cfg["learning_rate"],
    )
    train(model, dataset, training_config)


if __name__ == "__main__":
    main()
