.PHONY: install test lint download-data train generate clean

install:
	python -m pip install -r requirements.txt

test:
	python -m pytest

lint:
	python -m py_compile $(find src scripts tests -name '*.py')

download-data:
	python scripts/download_data.py

train:
	python scripts/train.py --config configs/base.yaml

generate:
	python scripts/generate.py --prompt "Hello"

clean:
	rm -rf build dist *.egg-info .pytest_cache
