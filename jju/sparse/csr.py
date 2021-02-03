import jax
import jax.numpy as jnp


@jax.jit
def matvec(data, indptr, indices, v):
    """Based on implementation in https://github.com/google/jax/pull/4422."""
    nrows = indptr.shape[0] - 1
    v = jnp.asarray(v)
    dv = data * v[indices]
    ind = jnp.cumsum(jnp.zeros_like(indices).at[indptr[1:]].add(1))
    return jnp.zeros(nrows, dv.dtype).at[ind].add(dv)


@jax.jit
def matmul(data, indptr, indices, B):
    """
    Args:
        data: [nnz] sparse matrix values.
        indptr: [nrows + 1] int sparse matrix index pointers / row_splits.
        indices: [nnz] int column indices of sparse data.
        B: [ncols, ...] dense data, right hand operator.

    Returns:
        [nrows, ...] dense result.
    """
    assert data.ndim == 1
    assert indptr.ndim == 1
    assert indices.ndim == 1
    assert B.ndim == 2
    fn = jax.vmap(matvec, in_axes=(None, None, None, 1), out_axes=1)
    return fn(data, indptr, indices, B)


@jax.jit
def sparse_conv(data, indptr, indices, B, kernel):
    """
    Args:
        data: [nnz] sparse matrix values
        indptr: [nrows + 1] int sparse matrix index pointers / row_splits.
        indices: [nnz] int column indices of sparse data.
        B: [Ni, F] input features
        kernel: [F, ...] kernel.

    Returns:
        [No, ...] A @ B @ kernel, where A is the sparse matrix defined in csr format by
          (data, indptr, indices).
    """
    return matmul(data, indptr, indices, B @ kernel)


@jax.jit
def sparse_conv_multli(data, indptr, indices, B, kernel):
    """
    Args:
        data: [K, nnz]
        indptr: [nrows + 1]
        indices: [nnz]
        B: [Ni, F] input features
        kernel: [K, F, ...]

    Returns:
        [No, ...]
    """

    nrows = indptr.shape[0] - 1
    B = jnp.asarray(B)
    B[indices]
    dv = data * B[indices]
    ind = jnp.cumsum(jnp.zeros_like(indices).at[indptr[1:]].add(1))
    return jnp.zeros(nrows, dv.dtype).at[ind].add(dv)


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
#         return jnp.matmul(d.reshape((1, -1)), values.reshape((-1, Fo))).squeeze(axis=0)

#     return jax.vmap(fn)(starts, shapes)
