import numpy as np
from absl.testing import absltest
from jax import test_util as jtu
from jax.config import config

from jju import types

config.parse_flags_with_absl()


class TypesTest(jtu.JaxTestCase):
    def test_matmul_fun(self):
        m = 5
        nx = 3
        rng = np.random.default_rng(0)
        A = rng.normal(size=(m, m))
        x = rng.normal(size=(m, nx))

        A_fun = types.matmul_fun(A)
        self.assertAllClose(A_fun(x), A @ x)


if __name__ == "__main__":
    absltest.main(testLoader=jtu.JaxTestLoader())
