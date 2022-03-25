from functools import partial
from typing import Optional

import jax
import jax.numpy as jnp

from jju.types import ArrayOrFun, as_array_fun


# decorator allows this to be passed as an argument to jitted functions
@jax.tree_util.Partial
def identity(x):
    return x


@jax.jit
def compute_residual(
    E: jnp.ndarray, X: jnp.ndarray, A: ArrayOrFun, B: Optional[ArrayOrFun]
):
    BX = X if B is None else as_array_fun(B)(X)
    return as_array_fun(A)(X) - BX * E


@jax.jit
def approx_matrix_norm2(A: Optional[ArrayOrFun], ohm: jnp.ndarray):
    """
    Approximation of matrix 2-norm of `A`.

        |A ohm|_fro / |ohm|_fro <= |A|_2

    This function returns the lower bound (left hand side).

    Args:
        A: matrix or callable that simulates matrix multiplication.
        ohm: block-vector used in formula. Should be Gaussian.

    Returns:
        Scalar, lower bound on 2-norm of A.
    """
    A = as_array_fun(A)
    return jnp.linalg.norm(A(ohm), "fro") / jnp.linalg.norm(ohm, "fro")


@jax.jit
def compute_residual_error(
    R: jnp.ndarray, E: jnp.ndarray, X: jnp.ndarray, A_norm: float, B_norm: float
):
    R_norm = jnp.linalg.norm(R, 2, (0,))
    X_norm = jnp.linalg.norm(X, 2, (0,))
    return R_norm / (X_norm * (A_norm + E * B_norm))


def eigh(a, largest: bool = False):
    """
    Get eigenvalues / eigenvectors of hermitian matrix a.

    Args:
        a: square hermitian float matrix
        largest: if True, return order is based on descending eigenvalues, otherwise
            ascending.

    Returns:
        w: [m] eigenvalues
        v: [m, m] eigenvectors
    """
    return _eigh(a, largest)


@partial(jax.jit, static_argnums=1)
def _eigh(a, largest: bool):
    w, v = jnp.linalg.eigh(a)
    if largest:
        w = w[-1::-1]
        v = v[:, -1::-1]
    return w, v


def rayleigh_ritz(
    S: jnp.ndarray, A: ArrayOrFun, B: Optional[ArrayOrFun] = None, largest: bool = False
):
    """

    Based on algorithm2 of [duersch2018](
        https://epubs.siam.org/doi/abs/10.1137/17M1129830)

    Args:
        S: [m, ns] float array, matrix basis for search space. Columns must be linearly
            independent and well-conditioned with respect to `B`.
        A: Callable simulating [m, m] float matrix multiplication.
        B: Callable simulating [m, m] float matrix multiplication.

    Returns:
        (eig_vals, C) satisfying the following:
            C.T @ S.T @ B(S) @ C = jnp.eye(ns)
            C.T @ S.T @ A(S) @ C = jnp.diag(eig_vals)

        eig_vals: [ns] eigenvalues. Sorted in descending order if largest, otherwise
            ascending.
        C: [ns, ns] float matrix satisfying:
    """
    A = as_array_fun(A)
    if B is None:
        BS = S
    else:
        BS = as_array_fun(B)(S)
    SBS = S.T @ BS
    d_right = jnp.diag(SBS) ** -0.5  # d_right * X == X @ D
    d_left = jnp.expand_dims(d_right, 1)  # d_left * X == D @ X
    R_low = jnp.linalg.cholesky(d_left * SBS * d_right)  # upper triangular
    R_up = R_low.T

    # R_inv = jnp.linalg.inv(R_up)
    # RDSASDR = R_inv.T @ (d_left * (S.T @ A(S)) * d_right) @ R_inv

    DSASD = d_left * (S.T @ A(S)) * d_right
    RDSASD = jax.scipy.linalg.solve_triangular(R_low, DSASD, lower=True)
    RDSASDR = jax.scipy.linalg.solve_triangular(R_low, RDSASD.T, lower=True).T

    eig_vals, Z = eigh(RDSASDR, largest)
    if B is not None:
        Z /= jnp.linalg.norm(Z, ord=2, axis=0)
    # C = d_left * R_inv @ Z
    C = d_left * (jax.scipy.linalg.solve_triangular(R_up, Z, lower=False))
    return eig_vals, C


@jax.jit
def projection(R, B, X):
    BX = B(X)
    return R - X @ (BX.T @ R)


@jax.jit
def orthonormalize(V, BV):
    norm = V.max(axis=0) + jnp.finfo(V.dtype).eps
    V = V / norm
    BV = BV / norm
    VBV = V.T @ BV
    if False:
        VBV = jax.scipy.linalg.cholesky(VBV, overwrite_a=True)
        VBV = jax.scipy.linalg.inv(VBV, overwrite_a=True)
        V = V @ VBV
        if B is not None:
            BV = BV @ VBV
        else:
            BV = None
    else:
        gram_VBV = jax.scipy.linalg.cho_factor(VBV)
        V = (jax.scipy.linalg.cho_solve(gram_VBV, V.T)).T
        BV = None
    return V, BV


@jax.jit
def apply_constraints(V, YBY, BY, Y):
    """Changes V in place."""
    YBV = BY.T @ V
    YBY_chol = jax.scipy.linalg.cho_factor(YBY)
    tmp = jax.scipy.linalg.cho_solve(YBY_chol, YBV)
    return V - Y @ tmp
