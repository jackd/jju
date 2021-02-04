from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import test_util as jtu
from jax.config import config

import jju.linalg.custom_gradients as cg
from jju.sparse import coo, csr
from jju.test_utils import (
    coo_components,
    csr_components,
    random_spd_coo,
    random_spd_csr,
)
from jju.utils import standardize_eigenvector_signs, symmetrize

config.parse_flags_with_absl()
config.update("jax_enable_x64", True)


def eigh_partial(a, k: int, largest: bool):
    w, v = jax.numpy.linalg.eigh(a)
    if largest:
        w = w[-1::-1]
        v = v[:, -1::-1]
    w = w[:k]
    v = v[:, :k]
    v = standardize_eigenvector_signs(v)
    return w, v


class LinalgCustomGradientsTest(jtu.JaxTestCase):
    def test_eigh_vjp(self):
        n = 20
        dtype = np.float64
        a = random_spd_coo(n=n, dtype=dtype)
        a = a.todense()

        def eigh(a):
            w, v = jax.numpy.linalg.eigh(a)
            v = standardize_eigenvector_signs(v)
            return w, v

        def eigh_fwd(a):
            w, v = eigh(a)
            return (w, v), (w, v)

        def eigh_rev(res, g):
            grad_w, grad_v = g
            w, v = res
            grad_a = cg.eigh_rev(grad_w, grad_v, w, v)
            grad_a = symmetrize(grad_a)
            return (grad_a,)

        eigh_fun = jax.custom_vjp(eigh)
        eigh_fun.defvjp(eigh_fwd, eigh_rev)
        jtu.check_grads(eigh_fun, (a,), order=1, modes="rev", rtol=1e-3)
        w, v = eigh(a)
        self.assertAllClose(a @ v, v * w, rtol=1e-6)

    def test_eigh_coo_vjp(self):
        n = 20
        dtype = np.float64
        a = random_spd_coo(n=n, dtype=dtype)

        def eigh_coo(data, row, col, size):
            data = coo.symmetrize(data, row, col, size)
            a = coo.to_dense(data, row, col, (size, size))
            w, v = jnp.linalg.eigh(a)
            v = standardize_eigenvector_signs(v)
            return w, v

        def eigh_coo_fwd(data, row, col, size):
            w, v = eigh_coo(data, row, col, size)
            return (w, v), (w, v, row, col)

        def eigh_coo_rev(res, g):
            grad_w, grad_v = g
            w, v, row, col = res
            size = v.shape[0]
            grad_data = cg.eigh_rev(
                grad_w, grad_v, w, v, coo.masked_matmul_fun(row, col)
            )
            grad_data = coo.symmetrize(grad_data, row, col, size)
            return (grad_data, None, None, None)

        eigh = jax.custom_vjp(eigh_coo)
        eigh.defvjp(eigh_coo_fwd, eigh_coo_rev)

        data, row, col, shape = coo_components(a)
        self.assertTrue(coo.is_symmetric(row, col, data, shape))
        jtu.check_grads(
            partial(eigh, row=row, col=col, size=n),
            (data,),
            order=1,
            modes="rev",
            rtol=1e-3,
        )

    def test_eigh_csr_vjp(self):
        n = 20
        dtype = np.float64
        a = random_spd_csr(n=n, dtype=dtype)

        def eigh_csr(data, indices, indptr):
            size = indptr.size - 1
            data = csr.symmetrize(data, indices)
            a = csr.to_dense(data, indices, indptr, (size, size))
            w, v = jnp.linalg.eigh(a)
            v = standardize_eigenvector_signs(v)
            return w, v

        def eigh_csr_fwd(data, indices, indptr):
            w, v = eigh_csr(data, indices, indptr)
            return (w, v), (w, v, indices, indptr)

        def eigh_csr_rev(res, g):
            grad_w, grad_v = g
            w, v, indices, indptr = res
            grad_data = cg.eigh_rev(
                grad_w, grad_v, w, v, csr.masked_matmul_fun(indices, indptr)
            )
            grad_data = csr.symmetrize(grad_data, indices)
            return (grad_data, None, None)

        eigh = jax.custom_vjp(eigh_csr)
        eigh.defvjp(eigh_csr_fwd, eigh_csr_rev)

        data, indices, indptr, _ = csr_components(a)
        jtu.check_grads(
            partial(eigh, indices=indices, indptr=indptr),
            (data,),
            order=1,
            modes="rev",
            rtol=1e-3,
        )

    def test_eigh_partial_vjp(self):
        dtype = np.float64
        n = 20
        k = 4
        largest = False
        a = random_spd_coo(n, dtype=dtype).todense()

        def eigh_partial_fwd(a, k: int, largest: bool):
            w, v = eigh_partial(a, k, largest)
            return (w, v), (w, v, a)

        def eigh_partial_rev(res, g):
            w, v, a = res
            grad_w, grad_v = g
            rng_key = jax.random.PRNGKey(0)
            x0 = jax.random.normal(rng_key, v.shape, dtype=v.dtype)
            grad_a, x0 = cg.eigh_partial_rev(grad_w, grad_v, w, v, a, x0)
            grad_a = symmetrize(grad_a)
            return (grad_a, None, None)

        eigh_partial_fun = jax.custom_vjp(eigh_partial)
        eigh_partial_fun.defvjp(eigh_partial_fwd, eigh_partial_rev)

        jtu.check_grads(
            partial(eigh_partial_fun, k=k, largest=largest),
            (a,),
            1,
            modes=["rev"],
            rtol=1e-3,
        )

    def test_eigh_partial_coo_vjp(self):
        dtype = np.float64
        n = 20
        k = 4
        largest = False
        a = random_spd_coo(n, dtype=dtype)

        def eigh_partial_coo(data, row, col, size, k: int, largest: bool):
            data = coo.symmetrize(data, row, col, size)
            a = coo.to_dense(data, row, col, (size, size))
            w, v = eigh_partial(a, k, largest)
            v = standardize_eigenvector_signs(v)
            return w, v

        def eigh_partial_fwd(data, row, col, size, k: int, largest: bool):
            w, v = eigh_partial_coo(data, row, col, size, k, largest)
            return (w, v), (w, v, data, row, col)

        def eigh_partial_rev(res, g):
            w, v, data, row, col = res
            size = v.shape[0]
            grad_w, grad_v = g
            rng_key = jax.random.PRNGKey(0)
            x0 = jax.random.normal(rng_key, shape=v.shape, dtype=w.dtype)
            grad_data, x0 = cg.eigh_partial_rev(
                grad_w,
                grad_v,
                w,
                v,
                coo.matmul_fun(data, row, col, jnp.zeros((size,))),
                x0,
                outer_impl=coo.masked_outer_fun(row, col),
            )
            grad_data = coo.symmetrize(grad_data, row, col, size)
            return (grad_data, None, None, None, None, None)

        eigh_partial_fn = jax.custom_vjp(eigh_partial_coo)
        eigh_partial_fn.defvjp(eigh_partial_fwd, eigh_partial_rev)

        data, row, col, _ = coo_components(a)
        self.assertTrue(coo.is_symmetric(row, col, data))
        self.assertTrue(coo.is_ordered(row, col))
        jtu.check_grads(
            partial(eigh_partial_fn, k=k, largest=largest, row=row, col=col, size=n),
            (data,),
            1,
            modes=["rev"],
            rtol=1e-3,
        )

    def test_eigh_partial_csr_vjp(self):
        dtype = np.float64
        n = 20
        k = 4
        largest = False
        a = random_spd_csr(n, dtype=dtype)

        def eigh_partial_coo(data, indices, indptr, k: int, largest: bool):
            size = indptr.size - 1
            data = csr.symmetrize(data, indices)
            a = csr.to_dense(data, indices, indptr, (size, size))
            w, v = eigh_partial(a, k, largest)
            v = standardize_eigenvector_signs(v)
            return w, v

        def eigh_partial_fwd(data, indices, indptr, k: int, largest: bool):
            w, v = eigh_partial_coo(data, indices, indptr, k, largest)
            return (w, v), (w, v, data, indices, indptr)

        def eigh_partial_rev(res, g):
            w, v, data, indices, indptr = res
            grad_w, grad_v = g
            rng_key = jax.random.PRNGKey(0)
            x0 = jax.random.normal(rng_key, shape=v.shape, dtype=w.dtype)
            grad_data, x0 = cg.eigh_partial_rev(
                grad_w,
                grad_v,
                w,
                v,
                csr.matmul_fun(data, indices, indptr),
                x0,
                outer_impl=csr.masked_outer_fun(indices, indptr),
            )
            grad_data = csr.symmetrize(grad_data, indices)
            return grad_data, None, None, None, None

        eigh_partial_fn = jax.custom_vjp(eigh_partial_coo)
        eigh_partial_fn.defvjp(eigh_partial_fwd, eigh_partial_rev)

        data, indices, indptr, _ = csr_components(a)
        jtu.check_grads(
            partial(
                eigh_partial_fn, k=k, largest=largest, indices=indices, indptr=indptr
            ),
            (data,),
            1,
            modes=["rev"],
            rtol=1e-3,
        )


if __name__ == "__main__":
    # LinalgCustomGradientsTest().test_eigh_vjp()
    # LinalgCustomGradientsTest().test_eigh_coo_vjp()
    # LinalgCustomGradientsTest().test_eigh_partial_vjp()
    # LinalgCustomGradientsTest().test_eigh_partial_coo_vjp()
    # LinalgCustomGradientsTest().test_eigh_partial_csr_vjp()
    # print("Good!")
    absltest.main(testLoader=jtu.JaxTestLoader())
