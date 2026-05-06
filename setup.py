from setuptools import setup, find_packages

setup(
    name="llm_from_scratch",
    version="0.1.0",
    description="Minimal LLM from scratch with tokenizer, transformer model, training, and inference.",
    packages=find_packages(where="."),
    package_dir={"": "."},
    install_requires=[
        "numpy>=1.24",
        "pyyaml>=6.0",
    ],
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
