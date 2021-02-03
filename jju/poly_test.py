import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import test_util as jtu
from jax.config import config

from jju import poly

config.parse_flags_with_absl()


class PolyTest(jtu.JaxTestCase):
    def test_matrix_poly_vector_prod_from_roots(self):
        m = 5
        nx = 3
        dtype = jnp.float32
        rng = np.random.default_rng(0)
        A = rng.normal(size=(m, m)).astype(dtype)
        x = rng.normal(size=(m, nx)).astype(dtype)

        roots = jnp.array([1.5, 2.0, 0, -3.0], dtype)
        eye = jnp.eye(m, dtype=dtype)

        actual = poly.matrix_poly_vector_prod_from_roots(roots, A, x)
        expected = (A - 1.5 * eye) @ (A - 2 * eye) @ (A - 0 * eye) @ (A + 3.0 * eye) @ x
        self.assertAllClose(actual, expected, rtol=1e-5)

    def test_poly_from_roots(self):
        nx = 3
        dtype = jnp.float32
        rng = np.random.default_rng(0)
        x = rng.normal(size=(nx,)).astype(dtype)
        roots = jnp.array([1.5, 2.0, 0, -3.0], dtype)

        actual = poly.poly_from_roots(roots, x)
        expected = (x - 1.5) * (x - 2) * (x - 0) * (x + 3)
        self.assertAllClose(actual, expected)

    def test_roots_to_coeffs(self):
        roots = jnp.asarray([2, 3, 5])
        expected = jnp.asarray([-30, 31, -10, 1])
        actual = poly.roots_to_coeffs(roots)
        self.assertAllClose(actual, expected)

    def test_matrix_poly_vector_prod_from_coeffs(self):
        m = 5
        nx = 3
        dtype = jnp.float32
        rng = np.random.default_rng(0)

        A = jnp.diag(jnp.arange(m, dtype=dtype))
        x = jnp.arange(m, dtype=dtype)
        A = rng.normal(size=(m, m)).astype(dtype)
        x = rng.normal(size=(m, nx)).astype(dtype)

        coeffs = jnp.array([1.5, 2.0, 0, -3.0], dtype)
        expected = 1.5 * x + 2 * A @ x - 3 * A @ A @ A @ x

        actual = poly.matrix_poly_vector_prod_from_coeffs(coeffs, A, x)

        self.assertAllClose(actual, expected, rtol=1e-5)

    def test_poly_from_coeffs(self):
        nx = 3
        dtype = jnp.float32
        rng = np.random.default_rng(0)
        x = rng.normal(size=(nx,)).astype(dtype)
        coeffs = jnp.array([1.5, 2.0, 0, -3.0], dtype)

        actual = poly.poly_from_coeffs(coeffs, x)
        expected = 1.5 + 2 * x - 3 * x ** 3
        self.assertAllClose(actual, expected)

    def test_poly_from_roots_coeffs_consistent(self):
        dtype = np.float32
        nx = 5
        poly_dim = 7
        rng = np.random.default_rng(0)
        roots = rng.normal(size=(poly_dim,)).astype(dtype)
        x = rng.normal(size=(nx,)).astype(dtype)

        coeffs = poly.roots_to_coeffs(roots)
        eval_roots = poly.poly_from_roots(roots, x)
        eval_coeffs = poly.poly_from_coeffs(coeffs, x)

        self.assertAllClose(eval_roots, eval_coeffs)

    def test_matrix_poly_vector_from_roots_coeffs_consistent(self):
        dtype = np.float32
        m = 5
        nx = 3
        poly_dim = 7
        rng = np.random.default_rng(0)
        roots = rng.normal(size=(poly_dim,)).astype(dtype)
        x = rng.normal(size=(m, nx,)).astype(dtype)
        A = rng.normal(size=(m, m)).astype(dtype)

        coeffs = poly.roots_to_coeffs(roots)
        eval_roots = poly.matrix_poly_vector_prod_from_roots(roots, A, x)
        eval_coeffs = poly.matrix_poly_vector_prod_from_coeffs(coeffs, A, x)

        self.assertAllClose(eval_roots, eval_coeffs, rtol=1e-5)


if __name__ == "__main__":
    # PolyTest().test_matrix_poly_vector_prod_from_coeffs()
    absltest.main(testLoader=jtu.JaxTestLoader())
