from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from jju.sparse import coo


@jax.jit
def matmul(
    data: jnp.ndarray, indices: jnp.ndarray, indptr: jnp.ndarray, v: jnp.ndarray
):
    """Based on implementation in https://github.com/google/jax/pull/4422."""
    nrows = indptr.shape[0] - 1
    v = jnp.asarray(v)
    dv = jnp.reshape(data, (-1,) + (1,) * (v.ndim - 1)) * v[indices]
    ind = jnp.cumsum(jnp.zeros_like(indices).at[indptr[1:]].add(1))
    return jnp.zeros((nrows, *v.shape[1:]), dv.dtype).at[ind].add(dv)


@jax.jit
def symmetrize(data: jnp.ndarray, indices: jnp.ndarray):
    """
    Get data of `(A + A.T) / 2` assuming `A` has symmetric sparsity.

    Args:
        data, indptr, indices: csr encoding of `A`

    Returns:
        `sym_data`, same shape and dtype as `data`.
    """
    assert data.size == indices.size
    order = jnp.argsort(indices)
    return (data + data[order]) / 2


def rows(indptr: jnp.ndarray, dtype=jnp.int32, total_size: Optional[int] = None):
    return _repeated_rows(
        jnp.arange(indptr.size - 1, dtype=dtype), indptr, total_size=total_size
    )


def _repeated_rows(
    x: jnp.ndarray, indptr: jnp.ndarray, axis=0, total_size: Optional[int] = None
):
    if total_size is None:
        total_size = indptr[-1]
    return jnp.repeat(
        x, indptr[1:] - indptr[:-1], axis=axis, total_repeat_length=total_size
    )


def masked_outer(
    indices: jnp.ndarray, indptr: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray
):
    """Compute (x @ y.T)[row, col] where (row, col) are the nonzero indices."""
    return _repeated_rows(x, indptr, total_size=indices.size) * y[indices]


def masked_inner(
    indices: jnp.ndarray, indptr: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray
):
    """Compute (x.T @ y)[row, col] where (row, col) are the implied nonzero indices."""
    assert x.ndim == 2
    assert y.ndim == 2
    return (
        _repeated_rows(x, indptr, axis=1, total_size=indices.size) * y[:, indices]
    ).sum(axis=0)


def masked_matmul(
    indices: jnp.ndarray, indptr: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray
):
    return masked_inner(indices, indptr, x.T, y)


def matmul_fun(data: jnp.ndarray, indices: jnp.ndarray, indptr: jnp.ndarray):
    return jax.tree_util.Partial(matmul, data, indices, indptr)


def masked_inner_fun(indices: jnp.ndarray, indptr: jnp.ndarray):
    return jax.tree_util.Partial(masked_inner, indices, indptr)


def masked_outer_fun(indices: jnp.ndarray, indptr: jnp.ndarray):
    return jax.tree_util.Partial(masked_outer, indices, indptr)


def masked_matmul_fun(indices: jnp.ndarray, indptr: jnp.ndarray):
    return jax.tree_util.Partial(masked_matmul, indices, indptr)


def to_dense(
    data: jnp.ndarray, indices: jnp.ndarray, indptr: jnp.ndarray, shape: Tuple[int, int]
):
    return coo.to_dense(data, rows(indptr, indices.dtype, indices.size), indices, shape)


def from_dense(dense: jnp.ndarray):
    data, row, col = coo.from_dense(dense)
    row_lengths = jnp.bincount(row, length=dense.shape[0])
    indptr = jnp.concatenate(
        (jnp.zeros((1,), dtype=row_lengths.dtype), jnp.cumsum(row_lengths))
    )
    return data, indptr, col


# @jax.jit
# def sparse_conv(data, indptr, indices, B, kernel):
#     """
#     Args:
#         data: [nnz] sparse matrix values
#         indptr: [nrows + 1] int sparse matrix index pointers / row_splits.
#         indices: [nnz] int column indices of sparse data.
#         B: [Ni, F] input features
#         kernel: [F, ...] kernel.

#     Returns:
#         [No, ...] A @ B @ kernel, where A is the sparse matrix defined in csr format
#           by (data, indptr, indices).
#     """
#     return matmul(data, indptr, indices, B @ kernel)


# @jax.jit
# def sparse_conv_multli(data, indptr, indices, B, kernel):
#     """
#     Args:
#         data: [K, nnz]
#         indptr: [nrows + 1]
#         indices: [nnz]
#         B: [Ni, F] input features
#         kernel: [K, F, ...]

#     Returns:
#         [No, ...]
#     """

#     nrows = indptr.shape[0] - 1
#     B = jnp.asarray(B)
#     B[indices]
#     dv = data * B[indices]
#     ind = jnp.cumsum(jnp.zeros_like(indices).at[indptr[1:]].add(1))
#     return jnp.zeros(nrows, dv.dtype).at[ind].add(dv)


# @jax.jit
# def multi_conv(data, indptr, indices, B, kernel):
#     """
#     Args:
#         data: [nnz, K] sparse matrix values.
#         indptr: [nrows + 1] int sparse matrix index pointers / row_splits
#         indices: [nnz] int column indices of sparse data, all < ncols.
#         B: [ncols, Fi] dense data, right hand operator.
#         kernel: [K, Fi, Fo] kernel values.

#     Returns:
#         [nrows, Fo] dense output.
#     """
#     assert data.ndim == 2
#     assert indptr.ndim == 1
#     assert indices.ndim == 1
#     assert B.ndim == 2
#     assert kernel.ndim == 3
#     K, Fi, Fo = kernel.shape
#     assert data.shape[1] == K
#     assert B.shape[1] == Fi

#     starts = indptr[:-1]
#     stops = indptr[1:]
#     shapes = stops - starts

#     def fn(start, shape):
#         # d = data[start:stop]  # e, K
#         # k = jnp.matmul(d, kernel)  # e, Fi, Fo
#         # bi = B.take(indices[start:stop])  # e, Fi
#         # return jnp.matmul(bi.reshape((1, -1)), k.reshape((-1, Fo))).squeeze(axis=0)
#         # d = data[start:stop]  # e, K
#         # bi = B.take(indices[start:stop])  # e, Fi
#         d = jax.lax.dynamic_slice(data, start, shape)
#         bi = jax.lax.dynamic_slice(data, start, shape)
#         values = jnp.einsum("ei,kio->eko", bi, kernel)  # e, K, Fo
#         return jnp.matmul(d.reshape((1, -1)), values.reshape((-1, Fo))).squeeze(0)

#     return jax.vmap(fn)(starts, shapes)
