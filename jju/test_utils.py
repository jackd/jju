from typing import NamedTuple, Tuple

import numpy as np
import scipy.sparse as sp


class COOComponents(NamedTuple):
    data: np.ndarray
    row: np.ndarray
    col: np.ndarray
    shape: Tuple[int, int]

    def to_coo(self):
        return sp.coo_matrix((self.data, (self.row, self.col)), shape=self.shape)


def random_coo(rng, shape, sparsity=0.1, dtype=np.float64) -> sp.coo_matrix:
    row, col = np.where(rng.uniform(size=shape) < sparsity)
    nnz = row.size
    data = rng.normal(size=nnz).astype(dtype)
    return sp.coo_matrix((data, (row, col)), shape=shape)


def random_spd_csr(n, sparsity=0.1, dtype=np.float64, seed=0) -> sp.csr_matrix:
    return random_spd_coo(n, sparsity, dtype, seed).tocsr()


def random_spd_coo(n, sparsity=0.1, dtype=np.float64, seed=0) -> sp.coo_matrix:
    rng = np.random.default_rng(seed)
    a = random_coo(rng, (n, n), sparsity=sparsity, dtype=dtype)
    # strengthen diagonal to make eigenvectors distinct
    a += sp.diags(rng.uniform(1, 2, size=n).astype(dtype))
    a = (a @ a.T).tocoo()
    a.sum_duplicates()
    return reorder_coo(a)


def reorder_coo(a: sp.coo_matrix) -> sp.coo_matrix:
    i0 = np.ravel_multi_index((a.row, a.col), a.shape)
    perm = np.argsort(i0)
    return sp.coo_matrix((a.data[perm], (a.row[perm], a.col[perm])), shape=a.shape)


def coo_components(coo: sp.coo_matrix) -> COOComponents:
    return COOComponents(coo.data, coo.row, coo.col, coo.shape)


class CSRComponents(NamedTuple):
    data: np.ndarray
    indices: np.ndarray
    indptr: np.ndarray
    shape: Tuple[int, int]

    def to_csr(self):
        return sp.csr_matrix((self.data, self.indices, self.indptr), shape=self.shape)


def random_csr(rng, shape, sparsity=0.1, dtype=np.float64) -> sp.csr_matrix:
    coo = random_coo(rng, shape, sparsity=sparsity, dtype=dtype)
    return coo.tocsr()


def csr_components(csr: sp.csr_matrix) -> CSRComponents:
    """(data, indices, indptr)."""
    return CSRComponents(csr.data, csr.indices, csr.indptr, csr.shape)
