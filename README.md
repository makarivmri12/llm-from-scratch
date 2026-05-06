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

## Development workflow

- Bekerja di branch `development` untuk fitur dan perbaikan.
- Buat branch fitur baru dari `development`, lalu buka pull request ke `development`.
- Jalankan tes lokal:

```bash
python -m pytest
```

- Gunakan `Makefile` untuk perintah umum:

```bash
make install
make test
make lint
make download-data
make train
make generate
```

- Setelah `development` stabil, merge ke `main`.

