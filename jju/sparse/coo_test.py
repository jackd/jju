import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import test_util as jtu
from jax.config import config

from jju.sparse import coo
from jju.test_utils import coo_components, random_coo, reorder_coo

config.parse_flags_with_absl()


class CooTest(jtu.JaxTestCase):
    def test_to_dense(self):
        shape = (11, 13)
        rng = np.random.default_rng(0)
        coo_mat = random_coo(rng, shape, sparsity=0.1)
        actual = coo.to_dense(coo_mat.data, coo_mat.row, coo_mat.col, shape)
        expected = coo_mat.todense()
        self.assertAllClose(actual, expected)

    def test_from_dense(self):
        shape = (11, 13)
        rng = np.random.default_rng(0)
        expected = random_coo(rng, shape, sparsity=0.1)
        data, row, col = coo.from_dense(np.array(expected.todense()))
        self.assertAllClose(data, expected.data)
        self.assertAllClose(row, expected.row.astype(np.int32))
        self.assertAllClose(col, expected.col.astype(np.int32))

    def test_matvec(self):
        No = 79
        Ni = 61
        rng = np.random.default_rng(0)
        v = rng.normal(size=(Ni,))
        coo_mat = random_coo(rng, (No, Ni), sparsity=0.1)
        expected = coo_mat @ v

        actual = coo.matmul(
            jnp.array(coo_mat.data),
            jnp.array(coo_mat.row),
            jnp.array(coo_mat.col),
            jnp.zeros(No),
            jnp.array(v),
        )
        self.assertAllClose(actual, expected, rtol=1e-4)

    def test_matmul(self):
        No = 79
        Ni = 61
        M = 10
        rng = np.random.default_rng(0)
        B = rng.normal(size=(Ni, M))
        coo_mat = random_coo(rng, (No, Ni), sparsity=0.1)

        expected = coo_mat @ B

        actual = coo.matmul(
            jnp.array(coo_mat.data),
            jnp.array(coo_mat.row),
            jnp.array(coo_mat.col),
            jnp.zeros(No),
            jnp.array(B),
        )
        self.assertAllClose(actual, expected, rtol=1e-4)

    def test_reorder_perm(self):
        ncols = 2
        row = jnp.array([0, 1, 0])
        col = jnp.array([0, 0, 1])

        expected = jnp.array([0, 2, 1])
        actual = coo.reorder_perm(row, col, ncols)
        self.assertAllClose(actual, expected)

    def test_reorder(self):
        No = 11
        Ni = 13
        rng = np.random.default_rng(0)
        coo_mat = random_coo(rng, (No, Ni), sparsity=0.2)
        data, row, col = coo.reorder(coo_mat.data, coo_mat.col, coo_mat.row)
        actual = coo.to_dense(data, row, col, (Ni, No))
        expected = coo.to_dense(coo_mat.data, coo_mat.row, coo_mat.col, (No, Ni)).T

        self.assertAllClose(actual, expected)

    def test_symmetrize(self):
        n = 50
        shape = (n, n)
        rng = np.random.default_rng(0)
        coo_mat = random_coo(rng, shape, sparsity=0.1)
        coo_mat = ((coo_mat + coo_mat.T) / 2).tocoo()
        coo_mat.sum_duplicates()
        coo_mat = reorder_coo(coo_mat)
        actual = coo.symmetrize(coo_mat.data, coo_mat.row, coo_mat.col, n)

        self.assertAllClose(actual, coo_mat.data)

    def test_masked_matmul(self):
        nx = 53
        ny = 19
        nh = 11

        rng = np.random.default_rng(0)
        dtype = np.float32
        coo_mat = random_coo(rng, (nx, ny), sparsity=0.2, dtype=dtype)
        x = rng.normal(size=(nx, nh)).astype(dtype)
        y = rng.normal(size=(nh, ny)).astype(dtype)

        row = jnp.asarray(coo_mat.row)
        col = jnp.asarray(coo_mat.col)

        actual = coo.masked_matmul(row, col, x, y)
        expected = (x @ y)[row, col]
        self.assertAllClose(actual, expected, rtol=1e-4)

    def test_masked_inner(self):
        nx = 53
        ny = 19
        nh = 11

        rng = np.random.default_rng(0)
        dtype = np.float32
        coo_mat = random_coo(rng, (nx, ny), sparsity=0.2, dtype=dtype)
        x = rng.normal(size=(nh, nx)).astype(dtype)
        y = rng.normal(size=(nh, ny)).astype(dtype)

        row = jnp.asarray(coo_mat.row)
        col = jnp.asarray(coo_mat.col)

        actual = coo.masked_inner(row, col, x, y)
        expected = (x.T @ y)[row, col]
        self.assertAllClose(actual, expected, rtol=1e-4)

    def test_masked_outer(self):
        n = 50
        m = 25
        dtype = np.float32
        rng = np.random.default_rng(0)
        coo_mat = random_coo(rng, (n, m), sparsity=0.2, dtype=dtype)
        row = jnp.array(coo_mat.row, dtype=jnp.int32)
        col = jnp.array(coo_mat.col, dtype=jnp.int32)

        x = rng.normal(size=(n,)).astype(dtype)
        y = rng.normal(size=m).astype(dtype)

        actual = coo.masked_outer(row, col, x, y)
        expected = jnp.outer(x, y)[row, col]
        self.assertAllClose(actual, expected)

    def test_is_symmetric(self):
        n = 10
        rng = np.random.default_rng(0)
        a = random_coo(rng, (n, n))
        data, row, col, shape = coo_components(a)
        self.assertFalse(coo.is_symmetric(row, col, shape=shape))
        self.assertFalse(coo.is_symmetric(row, col, data, shape=shape))

        a = (a + a.T).tocoo()
        data, row, col, shape = coo_components(a)
        self.assertTrue(coo.is_symmetric(row, col, shape=shape))
        self.assertTrue(coo.is_symmetric(row, col, data, shape=shape))

        self.assertFalse(coo.is_symmetric(row, col, shape=(shape[0], 2 * shape[1])))
        for i in range(a.nnz):
            if row[i] != col[i]:
                data[i] += 1
                break
        else:
            raise Exception("Test bugged - no non-diagonal entries")
        self.assertTrue(coo.is_symmetric(row, col, shape=shape))
        self.assertFalse(coo.is_symmetric(row, col, data, shape=shape))


if __name__ == "__main__":
    # CooTest().test_from_dense()
    absltest.main(testLoader=jtu.JaxTestLoader())
