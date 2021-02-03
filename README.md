# Jack's Jax Utilities

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A collection of utilities I've written for [jax](https://github.com/google/jax).

Currently provides:

- [LOBPCG](jju/linalg/lobpcg/basic.py) implementation for optionally sparse eigendecomposition.
- [vjp](jju/linalg/custom_gradients.py) for partial eigendecomposition, with optionally sparse inputs.

## Pre-commit

This package uses [pre-commit](https://pre-commit.com/) to ensure commits meet minimum criteria. To Install, use

```bash
pip install pre-commit
pre-commit install
```

This will ensure git hooks are run before each commit. While it is not advised to do so, you can skip these hooks with

```bash
git commit --no-verify -m "commit message"
```
