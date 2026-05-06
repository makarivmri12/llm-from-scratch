from __future__ import annotations

import argparse
from pathlib import Path

from src.utils.config import load_config
from src.tokenizer import Tokenizer
from src.model import TransformerConfig, SimpleTransformer
from src.inference import generate_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from the llm-from-scratch model.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--prompt", type=str, default="Hello")
    args = parser.parse_args()

    config = load_config(args.config)
    model_cfg = config["model"]

    tokenizer = Tokenizer()
    tokenizer.fit_texts([args.prompt])
    transformer_config = TransformerConfig(
        vocab_size=model_cfg["vocab_size"],
        seq_len=model_cfg["seq_len"],
        d_model=model_cfg["d_model"],
        n_heads=model_cfg["n_heads"],
        n_layers=model_cfg["n_layers"],
        d_ff=model_cfg["d_ff"],
    )
    model = SimpleTransformer(transformer_config)
    output = generate_text(model, tokenizer, args.prompt, max_new_tokens=config["inference"]["max_new_tokens"], temperature=config["inference"]["temperature"], top_k=config["inference"]["top_k"])
    print(output)


if __name__ == "__main__":
    main()
