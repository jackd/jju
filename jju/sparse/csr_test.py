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

        actual = csr.matvec(
            jnp.array(csr_mat.data),
            jnp.array(csr_mat.indptr),
            jnp.array(csr_mat.indices),
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

        actual = csr.matmul(
            jnp.array(csr_mat.data),
            jnp.array(csr_mat.indptr),
            jnp.array(csr_mat.indices),
            jnp.array(B),
        )
        self.assertAllClose(actual, expected, rtol=1e-4)


if __name__ == "__main__":
    # CsrTest().test_matmul()
    absltest.main(testLoader=jtu.JaxTestLoader())
