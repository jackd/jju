import jax
import jax.numpy as jnp

from jju.types import as_array_fun


def project(x, v):
    assert len(x.shape) == 1
    assert len(v.shape) == 1
    return x - v.conj()[:, jnp.newaxis] @ (v[jnp.newaxis, :] @ x)


def projector(v):
    return jax.tree_util.Partial(project, v=v)


def eigh_rev(grad_w, grad_v, w, v, matmul_fun=jnp.matmul):
    n = w.size
    E = w[jnp.newaxis, :] - w[:, jnp.newaxis]
    vt = v.T

    # grad_a = v (diag(grad_w) + (v^T v.grad / E)) v^T
    #        = v @ inner @ v.T
    inner = jnp.where(jnp.eye(n, dtype=bool), jnp.diag(grad_w), (vt @ grad_v) / E)
    return matmul_fun(v, inner @ vt)


def eigh_partial_rev(grad_w, grad_v, w, v, a, x0, outer_impl=jnp.outer):
    """
    Args:
        grad_w: [k] gradient w.r.t eigenvalues
        grad_v: [m, k] gradient w.r.t eigenvectors
        w: [k] eigenvalues
        v: [m, k] eigenvectors
        a: matmul function
        x0: [m, k] initial solution to (A - w[i]I)x[i] = Proj(grad_v[:, i])

    Returns:
        grad_a: [m, m]
        x0: [m, k]
    """
    # based on
    # https://github.com/fancompute/legume/blob/99dd012feee28156292787330dac5e4f0c41d4c8/legume/primitives.py#L170-L210
    a = as_array_fun(a)
    grad_As = []

    grad_As.append(
        jax.vmap(lambda grad_wi, vi: grad_wi * outer_impl(vi.conj(), vi), (0, 1))(
            grad_w, v
        ).sum(0)
    )
    if grad_v is not None:
        # Add eigenvector part only if non-zero backward signal is present.
        # This can avoid NaN results for degenerate cases if the function
        # depends on the eigenvalues only.

        def f_inner(grad_vi, wi, vi, x0i):
            def if_any(operand):
                grad_vi, wi, vi, x0i = operand

                # Amat = (a - wi * jnp.eye(m, dtype=a.dtype)).T
                Amat = lambda x: (a(x.conj())).conj() - wi * x

                # Projection operator on space orthogonal to v
                P = projector(vi)

                # Find a solution lambda_0 using conjugate gradient
                (l0, _) = jax.scipy.sparse.linalg.cg(
                    Amat, P(grad_vi), x0=P(x0i), atol=0
                )
                # (l0, _) = jax.scipy.sparse.linalg.gmres(Amat, P(grad_vi), x0=P(x0i))
                # l0 = jax.numpy.linalg.lstsq(Amat, P(grad_vi))[0]
                # Project to correct for round-off errors
                # print(Amat(l0) - P(grad_vi))
                l0 = P(l0)
                return -outer_impl(l0, vi), l0

            def if_none(operand):
                x0i = operand[-1]
                return jnp.zeros_like(grad_As[0]), x0i

            operand = (grad_vi, wi, vi, x0i)
            # return if_any(operand) if jnp.any(grad_vi) else if_none(operand)
            return jax.lax.cond(jnp.any(grad_vi), if_any, if_none, operand)

        # x0s = []
        # for k in range(grad_v.shape[1]):
        #     out = f_inner(grad_v[:, k], w[k], v[:, k], x0[:, k])
        #     grad_As.append(out[0])
        #     x0s.append(out[1])
        # x0 = jnp.stack(x0s, axis=0)
        grad_a, x0 = jax.vmap(f_inner, in_axes=(1, 0, 1, 1), out_axes=(0, 1))(
            grad_v, w, v, x0
        )
        grad_As.append(grad_a.sum(0))
    return sum(grad_As), x0
