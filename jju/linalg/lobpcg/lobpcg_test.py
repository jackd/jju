from typing import Optional

import jax
import jax.numpy as jnp
import numpy as onp
import numpy as np
import scipy
import scipy.sparse.linalg as scipy_lobpcg
from jax.config import config

from jju.linalg.lobpcg.basic import lobpcg
from jju.linalg.lobpcg.utils import identity, rayleigh_ritz
from jju.types import ArrayOrFun, as_array_fun


def rayleigh_ritz_scipy(
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
        C: [ns, ns] float matrix.
    """
    A = as_array_fun(A)
    BS = as_array_fun(B)(S)
    SBS = S.T @ BS

    SAS = S.T @ A(S)

    eig_vals, C = scipy.linalg.eigh(SAS, SBS)
    if largest:
        eig_vals = eig_vals[::-1]
        C = C[:, ::-1]

    return eig_vals, C


def generate_wishart(N=1000, T=1100):
    r = N / T
    X = np.random.randn(T, N)
    W = X.T @ X / T
    return W, X


def generate_data(T, N):
    rng = jax.random.PRNGKey(42)
    X = jax.random.uniform(rng, (T, N))
    W = X.T @ X / T
    return W, X


def test_rayleigh_ritz():
    T = 100
    N = 5000
    W, X = generate_data(T, N)

    rng = jax.random.PRNGKey(42)
    X0 = jax.random.uniform(rng, (N, 3))

    ev_ref, C_ref = rayleigh_ritz_scipy(X0, W, identity, largest=True)

    ev, C = rayleigh_ritz(X0, as_array_fun(W), identity, True)
    assert jnp.allclose(ev, ev_ref, atol=1.0e-3)
    assert jnp.allclose(C, C_ref, atol=1.0e-3)


def test_lobpcg_vs_scipy():

    config.update("jax_enable_x64", True)

    T = 100
    N = 5000
    W, X = generate_data(T, N)
    rng = jax.random.PRNGKey(42)
    X0 = jax.random.uniform(rng, (N, 3))
    assert W.shape == (N, N)
    assert X0.shape == (N, 3)

    # Y = jnp.eye(N, 3)
    ev, X1 = lobpcg(W, X0, largest=True)
    ev.block_until_ready()
    assert ev.shape == (3,)
    assert X1.shape == (N, 3)

    ev_ref, X1_ref = scipy_lobpcg.lobpcg(
        np.array(W).astype(np.float64), np.array(X0).astype(np.float64)
    )
    assert ev_ref.shape == (3,)
    assert X1_ref.shape == (N, 3)

    assert jnp.allclose(ev, ev_ref, atol=1.0e-3)
    assert jnp.abs(ev - jnp.array(ev_ref)).max() < 2.0e-4
    assert jnp.abs(jnp.abs(X1) - jnp.abs(jnp.array(X1_ref))).max().max() < 3.0e-3


def test_lobpcg_vs_torch():
    import torch

    config.update("jax_enable_x64", True)

    T = 100
    N = 5000
    W, X = generate_data(T, N)
    rng = jax.random.PRNGKey(42)
    X0 = jax.random.uniform(rng, (N, 3))
    assert W.shape == (N, N)
    assert X0.shape == (N, 3)

    # Y = jnp.eye(N, 3)
    Y = None
    ev, X1 = lobpcg(W, X0, Y=Y, largest=True)
    ev.block_until_ready()
    assert ev.shape == (3,)
    assert X1.shape == (N, 3)

    Wt = torch.tensor(onp.array(W))
    X0t = torch.tensor(onp.array(X0))
    evt, X1t = torch.lobpcg(Wt, X=X0t, largest=True)
    assert jnp.abs(ev - jnp.array(evt)).max() < 1.0e-9
    assert jnp.abs(jnp.abs(X1) - jnp.abs(jnp.array(X1t))).max().max() < 3.0e-6
