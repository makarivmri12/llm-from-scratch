# llm-from-scratch

Minimal repository scaffold for building a simple transformer-based language model from scratch.

## Structure

- `.github/workflows/`: CI workflows for tests and linting.
- `src/`: main library code for tokenization, transformer model, training, inference, and data preparation.
- `scripts/`: CLI entry points for training, generation, and sample data creation.
- `notebooks/`: exploratory notebook assets.
- `tests/`: unit tests for tokenizer, model, and training.
- `configs/`: default YAML configuration files.
- `data/`: sample dataset used by the training example.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run tests

```bash
python -m pytest
```

## Quick start

```bash
python scripts/download_data.py
python scripts/train.py --config configs/base.yaml
python scripts/generate.py --prompt "Hello"
```
