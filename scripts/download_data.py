from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a sample dataset file for llm-from-scratch.")
    parser.add_argument("--output", type=str, default="data/sample.txt")
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sample_text = (
        "This is a small sample dataset for llm-from-scratch. "
        "It contains simple text that the tokenizer can use to build a vocabulary."
    )
    output_path.write_text(sample_text, encoding="utf-8")
    print(f"Created dataset: {output_path}")


if __name__ == "__main__":
    main()
