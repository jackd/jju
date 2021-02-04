import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import test_util as jtu
from jax.config import config

from jju.sparse import csr
from jju.test_utils import random_csr

config.parse_flags_with_absl()


class CsrTest(jtu.JaxTestCase):
    def test_matvec(self):
        No = 79
        Ni = 61
        rng = np.random.default_rng(0)
        v = rng.normal(size=(Ni,))
        csr_mat = random_csr(rng, (No, Ni), sparsity=0.1)
        expected = csr_mat @ v

        actual = csr.dot(
            jnp.array(csr_mat.data),
            jnp.array(csr_mat.indices),
            jnp.array(csr_mat.indptr),
            jnp.array(v),
        )
        self.assertAllClose(actual, expected, rtol=1e-4)

    def test_matmul(self):
        No = 79
        Ni = 61
        M = 10
        rng = np.random.default_rng(0)
        B = rng.normal(size=(Ni, M))
        csr_mat = random_csr(rng, (No, Ni), sparsity=0.1)

        expected = csr_mat @ B

        actual = csr.dot(
            jnp.array(csr_mat.data),
            jnp.array(csr_mat.indices),
            jnp.array(csr_mat.indptr),
            jnp.array(B),
        )
        self.assertAllClose(actual, expected, rtol=1e-4)

    def test_symmetrize(self):
        n = 50
        shape = (n, n)
        rng = np.random.default_rng(0)
        csr_mat = random_csr(rng, shape, sparsity=0.1)
        csr_mat = ((csr_mat + csr_mat.T) / 2).tocsr()
        csr_mat.sum_duplicates()
        actual = csr.symmetrize(csr_mat.data, csr_mat.indices)

        self.assertAllClose(actual, csr_mat.data)

    def test_masked_matmul(self):
        nx = 53
        ny = 19
        nh = 11

        rng = np.random.default_rng(0)
        dtype = np.float32
        csr_mat = random_csr(rng, (nx, ny), sparsity=0.2, dtype=dtype)
        x = rng.normal(size=(nx, nh)).astype(dtype)
        y = rng.normal(size=(nh, ny)).astype(dtype)

        indices = jnp.asarray(csr_mat.indices)
        indptr = jnp.asarray(csr_mat.indptr)

        actual = csr.masked_matmul(indices, indptr, x, y)
        expected = (x @ y)[csr.rows(indptr), indices]
        self.assertAllClose(actual, expected, rtol=1e-4)

    def test_masked_inner(self):
        nx = 53
        ny = 19
        nh = 11

        rng = np.random.default_rng(0)
        dtype = np.float32
        csr_mat = random_csr(rng, (nx, ny), sparsity=0.2, dtype=dtype)
        x = rng.normal(size=(nh, nx)).astype(dtype)
        y = rng.normal(size=(nh, ny)).astype(dtype)

        indptr = jnp.asarray(csr_mat.indptr)
        indices = jnp.asarray(csr_mat.indices)

        actual = csr.masked_inner(indices, indptr, x, y)
        expected = (x.T @ y)[csr.rows(indptr), indices]
        self.assertAllClose(actual, expected, rtol=1e-4)

    def test_masked_outer(self):
        n = 50
        m = 25
        dtype = np.float32
        rng = np.random.default_rng(0)
        csr_mat = random_csr(rng, (n, m), sparsity=0.2, dtype=dtype)
        indptr = jnp.array(csr_mat.indptr, dtype=jnp.int32)
        indices = jnp.array(csr_mat.indices, dtype=jnp.int32)

        x = rng.normal(size=(n,)).astype(dtype)
        y = rng.normal(size=m).astype(dtype)

        actual = csr.masked_outer(indices, indptr, x, y)
        expected = jnp.outer(x, y)[csr.rows(indptr), indices]
        self.assertAllClose(actual, expected)

    def test_to_dense(self):
        n = 50
        m = 25
        dtype = np.float32
        rng = np.random.default_rng(0)
        csr_mat = random_csr(rng, (n, m), sparsity=0.2, dtype=dtype)
        self.assertAllClose(
            csr_mat.todense(),
            csr.to_dense(csr_mat.data, csr_mat.indices, csr_mat.indptr, csr_mat.shape),
        )

    def test_from_dense(self):
        n = 50
        m = 25
        dtype = np.float32
        rng = np.random.default_rng(0)
        expected = random_csr(rng, (n, m), sparsity=0.2, dtype=dtype)

        data, indptr, indices = csr.from_dense(np.array(expected.todense()))
        self.assertAllClose(data, expected.data)
        self.assertAllClose(indptr, expected.indptr)
        self.assertAllClose(indices, expected.indices)


if __name__ == "__main__":
    # CsrTest().test_masked_outer()
    absltest.main(testLoader=jtu.JaxTestLoader())
