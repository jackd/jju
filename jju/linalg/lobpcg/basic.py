from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp

from jju.linalg.lobpcg import utils
from jju.types import ArrayFun, ArrayOrFun, as_array_fun


# @partial(jax.jit, static_argnums=(5,))
def lobpcg(
    A: ArrayOrFun,
    X0: jnp.ndarray,
    B: Optional[ArrayOrFun] = None,
    iK: Optional[ArrayOrFun] = None,
    Y: Optional[jnp.ndarray] = None,
    largest: bool = False,
    k: Optional[int] = None,
    tol: Optional[float] = None,
    max_iters: int = 1000,
):
    """Find some of the eigenpairs for the generalized eigenvalue problem (A, B).

    Args:
        A: `[m, m]` hermitian matrix, or function representing pre-multiplication by an
            `[m, m]` hermitian matrix.
        X0: `[m, n]`, `k <= n < m`. Initial guess of eigenvectors.
        B: same type as A. If not given, identity is used.
        iK: Optional inverse preconditioner. If not given, identity is used.
        Y: n-by-sizeY matrix of constraints (non-sparse), sizeY < n
            The iterations will be performed in the B-orthogonal complement
            of the column-space of Y. Y must be full rank.
        largest: if True, return the largest `k` eigenvalues, otherwise the smallest.
        k: number of eigenpairs to return. Uses `n` if not provided.
        tol: tolerance for convergence.
        max_iters: maximum number of iterations.

    Returns:
        w: [k] smallest/largest eigenvalues of generalized eigenvalue problem `(A, B)`.
        v: [n, k] eigenvectors associated with `w`. `v[:, i]` matches `w[i]`.
    """
    # Perform argument checks and fix default / computed arguments
    if B is not None:
        raise NotImplementedError("Implementations with non-None B have issues")
    if iK is not None:
        raise NotImplementedError("Inplementations with non-None iK have issues")
    ohm = jax.random.normal(jax.random.PRNGKey(0), shape=X0.shape, dtype=X0.dtype)
    A = as_array_fun(A)
    A_norm = utils.approx_matrix_norm2(A, ohm)
    if B is None:
        B = utils.identity
        B_norm = jnp.ones((), dtype=X0.dtype)
    else:
        B = as_array_fun(B)
        B_norm = utils.approx_matrix_norm2(B, ohm)
    if iK is None:
        iK = utils.identity
    else:
        iK = as_array_fun(iK)

    if tol is None:
        dtype = X0.dtype
        if dtype == jnp.float32:
            feps = 1.2e-7
        elif dtype == jnp.float64:
            feps = 2.23e-16
        else:
            raise KeyError(dtype)
        tol = feps**0.5

    k = k or X0.shape[1]
    return _lobpcg(A, X0, B, iK, Y, largest, k, tol, max_iters, A_norm, B_norm)


class _BasicState(NamedTuple):
    iteration: int
    eig_vals: jnp.ndarray  # [n]
    X: jnp.ndarray  # [m, n]
    R: jnp.ndarray  # [m, n]
    P: jnp.ndarray  # [m, n]
    rerr: jnp.ndarray  #  error
    mask: jnp.ndarray  # mask of innactive directions


def check(x):
    assert jnp.all(jnp.isfinite(x))


@partial(jax.jit, static_argnums=(5,))
def _lobpcg(
    A: ArrayFun,
    X0: jnp.ndarray,
    B: ArrayFun,
    iK: ArrayFun,
    Y: jnp.ndarray,
    largest: bool,
    k: int,
    tol: float,
    max_iters: int,
    A_norm: float,
    B_norm: float,
):
    @jax.jit
    def cond_fun(s: _BasicState):
        num_converged = jnp.count_nonzero(s.rerr < tol)
        return jnp.logical_and(s.iteration < max_iters, num_converged < k)

    @jax.jit
    def body_fun(s: _BasicState):
        iteration, eig_vals, X, R, P, rerr, mask = s

        mask_full = jnp.outer(jnp.full(m, True, dtype=jnp.bool_), mask)
        R = jnp.where(mask_full, R_init, R)
        P = jnp.where(mask_full, P_init, P)

        # Apply preconditioner T to the active residuals.
        R = iK(R)
        if Y is not None:
            R = utils.apply_constraints(R, YBY, BY, Y)

        # R = projection(R, B, X)
        # R = projection(R, B, jnp.concatenate((X, P), axis=1))

        # B-orthonormalize residues R and P
        if B is not None:
            BR = B(R)
            BP = B(P)
        else:
            BR = R
            BP = P
        R, BR = utils.orthonormalize(R, BR)
        P, BP = utils.orthonormalize(P, BP)

        S = jnp.concatenate((X, R, P), axis=1)
        eig_vals, C = utils.rayleigh_ritz(S, A, B, largest)
        eig_vals = eig_vals[:nx]
        C = C[:, :nx]

        P = S[:, nx:] @ C[nx:]
        X = S[:, :nx] @ C[:nx] + P
        R = utils.compute_residual(eig_vals, X, A, B)
        rerr = utils.compute_residual_error(R, eig_vals, X, A_norm, B_norm)
        mask = jnp.logical_or(mask, rerr < tol)

        return _BasicState(iteration + 1, eig_vals, X, R, P, rerr, mask)

    if Y is not None:
        BY = B(Y)
        YBY = Y.T @ BY

    m, nx = X0.shape
    rng = jax.random.PRNGKey(42)
    S_init = jax.random.normal(rng, (m, 3 * nx)) / jnp.sqrt(m)
    R_init = S_init[:, nx : 2 * nx]
    P_init = S_init[:, 2 * nx :]

    # Initialization
    state = _BasicState(
        1,
        eig_vals=jnp.full(nx, jnp.nan, dtype=X0.dtype),
        X=X0,
        R=R_init,
        P=P_init,
        rerr=jnp.full(nx, jnp.inf, dtype=X0.dtype),
        mask=jnp.full(nx, False, dtype=jnp.bool_),
    )

    state = jax.lax.while_loop(cond_fun, body_fun, state)
    return state.eig_vals, state.X
