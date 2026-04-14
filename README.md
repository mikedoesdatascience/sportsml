# SportsML

ML for sports

## Installation

Requires [uv](https://docs.astral.sh/uv/).

### CPU (default)

```sh
uv lock && uv sync
```

Installs PyTorch with CPU-only support. This is the default behavior — no extra flags needed.

### CUDA 12.8

```sh
uv lock && uv sync --extra cu128 --no-group cpu
```

Installs PyTorch with CUDA 12.8 support. The `--no-group cpu` flag is required to disable the default CPU torch group, which conflicts with the `cu128` extra.

## Running

`uv run` requires the same flags to select the correct PyTorch variant.

### CPU

```sh
uv run ...
```

### CUDA 12.8

```sh
uv run --extra cu128 --no-group cpu ...
```
