from typing import Optional, Tuple

import jax
import jax.numpy as jnp


def assert_coo(data, row, col, sized=None):
    assert row.size == col.size == data.size
    assert row.ndim == 1
    assert col.ndim == 1
    assert data.ndim == 1
    if sized is not None:
        assert sized.ndim == 1


def indices_1d(row: jnp.ndarray, col: jnp.ndarray, ncols: Optional[int] = None):
    if ncols is None:
        ncols = col.max() + 1
    return row * ncols + col


# def indices_2d(index_1d, ncols: int):
#     return (indices_1d // ncols, indices_1d % ncols)


@jax.jit
def reorder_perm(row: jnp.ndarray, col: jnp.ndarray, ncols: Optional[int] = None):
    return jnp.argsort(indices_1d(row, col, ncols))


@jax.jit
def reorder(
    data: jnp.ndarray, row: jnp.ndarray, col: jnp.ndarray, ncols: Optional[int] = None
):
    assert_coo(data, row, col)
    perm = reorder_perm(row, col, ncols)
    return data.take(perm), row.take(perm), col.take(perm)


@jax.jit
def is_ordered(row, col, ncols: Optional[int] = None, strict: bool = False) -> bool:
    i0 = indices_1d(row, col, ncols)
    lower = i0[:-1]
    upper = i0[1:]
    valid = lower < upper if strict else lower <= upper
    return jnp.all(valid)


@jax.jit
def symmetrize(
    data: jnp.ndarray, row: jnp.ndarray, col: jnp.ndarray, ncols: Optional[int] = None,
):
    """
    Get data of `(A + A.T) / 2` assuming `A` has symmetric sparsity.

    Args:
        data: values of `A`
        row: row indices of `A`
        col: col indices of `A`
        ncols: number of columns of `A`

    Returns:
        `sym_data`, same shape and dtype as `data`.
    """
    perm = reorder_perm(row=col, col=row, ncols=ncols)
    return (data + data.take(perm)) / 2


@jax.jit
def dot(data, row, col, sized, v):
    """
    Perform matrix-vector product `A @ v` with coo formatted matrix `A`.

    Note the use of `sized` rather than `nrows` is to allow `matvec` to be compiled
    without static arguments. This is a work-around and is likely to change soon.

    See https://github.com/google/jax/issues/5609.

    Args:
        data: nonzero values of `A`.
        row: row indices of nonzero values of `A`.
        col: column indices of nonzero values of `A`.
        sized: array with size `nrows`, the number of rows of `A`.
        v:

    Returns:
        `n
    """
    assert_coo(data, row, col, sized)
    dv = jnp.reshape(data, (-1, *(1,) * (v.ndim - 1))) * v[col]
    return jnp.zeros((sized.size, *v.shape[1:]), dtype=dv.dtype).at[row].add(dv)


def dot_fun(data: jnp.ndarray, row: jnp.ndarray, col: jnp.ndarray, sized: jnp.ndarray):
    assert_coo(data, row, col, sized)
    return jax.tree_util.Partial(dot, data, row, col, sized)


def masked_matmul(row, col, x, y):
    """Compute `(x @ y)[row, col]` at `[row, col]` for rank-2 `x` and `y`."""
    return masked_inner(row, col, x.T, y)
    # return (x.T.take(row, axis=1) * y.take(col, axis=1)).sum(axis=0)


def to_dense(
    data: jnp.ndarray, row: jnp.ndarray, col: jnp.ndarray, shape: Tuple[int, int]
):
    out = jnp.zeros(shape)
    return jax.ops.index_update(out, jax.ops.index[row, col], data)


def from_dense(dense: jnp.ndarray):
    row, col = jnp.where(dense)
    return dense[row, col], row, col


def masked_inner(row, col, x, y):
    """Comput `(x.T @ y)[row, col]`."""
    assert x.ndim == 2
    assert y.ndim == 2
    return (x[:, row] * y[:, col]).sum(axis=0)


def masked_outer(row, col, x, y):
    """Compute `(x @ y.T)[row, col]`."""
    return x[row] * y[col]


def masked_inner_fun(row, col):
    return jax.tree_util.Partial(masked_inner, row, col)


def masked_outer_fun(row, col):
    return jax.tree_util.Partial(masked_outer, row, col)


def masked_matmul_fun(row, col):
    return jax.tree_util.Partial(masked_matmul, row, col)


def is_symmetric(
    row: jnp.ndarray,
    col: jnp.ndarray,
    data: Optional[jnp.ndarray] = None,
    shape: Optional[Tuple[int, int]] = None,
):
    if shape is None:
        nrows = row.max() + 1
        ncols = col.max() + 1
    else:
        nrows, ncols = shape
    conds = [nrows == ncols]
    i0 = indices_1d(row, col, ncols)
    i1 = indices_1d(col, row, nrows)
    if data is None:
        conds.append(jnp.all(i0 == jnp.sort(i1)))
    else:
        perm = jnp.argsort(i1)
        conds.append(jnp.all(i0 == i1.take(perm)))
        conds.append(jnp.all(data.take(perm) == data))
    return jnp.all(jnp.stack(conds))


# def add(
#     d0, r0, c0, d1, r1, c1, ncols: Optional[int] = None
# ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
#     if ncols is None:
#         ncols = jnp.maximum(c0.max(), c1.max()) + 1
#     i = jnp.concatenate((indices_1d(r0, c0, ncols), indices_1d(r1, c1, ncols)))
#     d = jnp.concatenate((d0, d1))

#     iu, di = jnp.unique(i, return_index=True)
#     data = jax.ops.(jnp, d[di])
