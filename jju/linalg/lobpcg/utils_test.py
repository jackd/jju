import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from jax import test_util as jtu
from jax.config import config

from jju.linalg.lobpcg import utils

config.parse_flags_with_absl()
config.update("jax_enable_x64", True)


def symmetrize(A):
    return (A + A.T) / 2


class LobpcgUtilsTest(jtu.JaxTestCase):
    def assertAllLessEqual(self, a, b):
        self.assertEqual(a.shape, b.shape)
        for ai, bi in zip(a, b):
            self.assertLessEqual(ai, bi)

    def assertAllGreaterEqual(self, a, b):
        self.assertEqual(a.shape, b.shape)
        for ai, bi in zip(a, b):
            self.assertGreaterEqual(ai, bi)

    # def test_approx_matrix_norm2(self):
    #     m = 600
    #     nx = 10
    #     rng = np.random.default_rng(1)
    #     A = symmetrize(rng.normal(size=(m, m)))
    #     x = rng.normal(size=(m, nx))

    #     actual = utils.approx_matrix_norm2(A, x)
    #     expected = jnp.linalg.norm(A, 2)
    #     self.assertAllClose(actual, expected)

    def test_eigh(self):
        m = 10
        rng = np.random.default_rng(0)
        A = symmetrize(rng.normal(size=(m, m)))
        w, v = utils.eigh(A, largest=True)
        self.assertAllClose(A @ v, v * w, rtol=1e-10)
        self.assertAllGreaterEqual(w[:-1], w[1:])
        w, v = utils.eigh(A, largest=False)
        self.assertAllClose(A @ v, v * w, rtol=1e-10)
        self.assertAllLessEqual(w[:-1], w[1:])

    def test_rayleigh_ritz(self):
        m = 20
        nx = 5

        rng = np.random.default_rng(0)
        A = symmetrize(rng.normal(size=(m, m)))
        B = symmetrize(rng.normal(size=(m, m)))
        B += m * np.eye(m)
        S = rng.normal(size=(m, nx))

        E, C = utils.rayleigh_ritz(S, A, B, largest=True)
        self.assertAllClose(C.T @ S.T @ B @ S @ C, jnp.eye(nx), atol=1e-12)
        self.assertAllClose(C.T @ S.T @ A @ S @ C, jnp.diag(E), atol=1e-12)
        self.assertAllGreaterEqual(E[:-1], E[1:])

        E, C = utils.rayleigh_ritz(S, A, B, largest=False)
        self.assertAllClose(C.T @ S.T @ B @ S @ C, jnp.eye(nx), atol=1e-12)
        self.assertAllClose(C.T @ S.T @ A @ S @ C, jnp.diag(E), atol=1e-12)
        self.assertAllLessEqual(E[:-1], E[1:])


if __name__ == "__main__":
    # LobpcgUtilsTest().test_rayleigh_ritz()
    # print("Good!")
    absltest.main(testLoader=jtu.JaxTestLoader())
