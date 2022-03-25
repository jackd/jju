from typing import Callable, Union

import jax
import jax.numpy as jnp

# function implementing pre-multiplication by a matrix
ArrayFun = Callable[[jnp.ndarray], jnp.ndarray]
ArrayOrFun = Union[jnp.ndarray, ArrayFun]


def matmul_fun(X: jnp.ndarray) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Return a function that pre-multiplies the argument by `X`."""
    return jax.tree_util.Partial(jnp.matmul, X)


def as_array_fun(array_or_fun: ArrayOrFun) -> ArrayFun:
    """Conver ArrayOrFun to an `ArrayFun`."""
    if isinstance(array_or_fun, jnp.ndarray):
        return matmul_fun(array_or_fun)
    assert callable(array_or_fun)
    return array_or_fun
