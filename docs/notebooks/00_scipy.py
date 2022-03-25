# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py,md
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Locally Optimal Block Preconditioned Conjugate Gradient (LOBPCG) method
#
# From [wikipedia page](https://en.wikipedia.org/wiki/LOBPCG):
# > LOBPCG is a matrix-free method for finding the largest (or smallest) eigenvalues and the corresponding eigenvectors of a symmetric generalized eigenvalue problem.
#
#     A x = λ B x
#
#
# I found the method following this recent discussion:
# [What is the most efficient method for calculating eigenvalues and eigenvectors of large matrices?](https://www.researchgate.net/post/What-is-the-most-efficient-method-for-calculating-eigenvalues-and-eigenvectors-of-large-matrices)
#
# In particular, Andrew Knyazev, the author of the LOBPCG, makes multiple interesting remarks.
#
#
# He talks about the  LOBPCG  method:
# > For example, LOBPCG works only for symmetric problems, including
# > generalized, to find <5-10% of extreme eigenpairs, but whether 
# > the matrix is sparse or dense is not so important.
# > For a full review of options, please read this [paper from 1997](https://www.researchgate.net/publication/46619332_Templates_for_the_Solution_of_Algebraic_Eigenvalue_Problems_A_Practical_Guide).
#
#
# >The already mentioned LOBPCG has many implementations. In MATLAB, including support for single precision, distributed or codistributed arrays, and tiling arrays, please see
# https://www.mathworks.com/matlabcentral/fileexchange/48-locally-optimal-block-preconditioned-conjugate-gradient .
# A plain C version is in https://github.com/lobpcg/blopex .
# In other cases, LOBPCG is already included in the hosting package, e.g., Anasazi (Trilinos), SLEPc, hypre, SciPy, Octave, Julia, MAGMA, Pytorch, Rust, RAPIDS cuGraph, and NVIDIA AMGX. For a comprehensive review, see https://en.wikipedia.org/wiki/LOBPCG
#
#
# We can find an implementation in scipy:
# - https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lobpcg.html
#
# And in pytorch:
# - https://pytorch.org/docs/stable/generated/torch.lobpcg.html


# Let's work with some examples.

# %load_ext autoreload
# %autoreload 2

import numpy as np
import scipy.linalg
from scipy.sparse import issparse, spdiags
from scipy.sparse.linalg import LinearOperator, lobpcg


def generate_wishart(N=1000, T=1100):
    X = np.random.randn(T, N)
    W = X.T @ X / T
    return W, X


rng = np.random.default_rng()

N = 5000
T = 100

W, X = generate_wishart(N, T)
X.shape

X0 = rng.random((N, 3))

# %time ev, X1 = lobpcg(W, X0)
ev

# Let's look if we gain in speed if we use the warm start mechanism of LOBPCG:

for i in range(5):
    # %time ev, X0 = lobpcg(W, X0)
ev

# Now Let's try to exploit the structure of the matrix and the fact that in our current example the data matrix $X$ is very thin.

A = LinearOperator(matvec=lambda v: W @ v, shape=(N, N), dtype=float)

X0 = rng.random((N, 3))
# %time ev, X1 = lobpcg(A, X0)
ev

A = LinearOperator(
    matvec=lambda v: (X.T @ (X @ v)) / T,
    # rmatvec=lambda v: (X.T @ (X @ v))/T,
    # matmat=lambda v: (X.T @ (X @ v))/T,
    shape=(N, N),
    dtype=float,
)

X0 = rng.random((N, 3))
# %time ev, X1 = lobpcg(A, X0)
ev

for i in range(6):
    # %time ev, X0 = lobpcg(A, X0)
ev

# We end with a very small execution time!

# Can we use preconditionner to speedup again things

# Initial guess for eigenvectors, should have linearly independent
# columns. Column dimension = number of requested eigenvalues.
#

# Preconditioner in the inverse of A in this example:
#

invA = spdiags([1.0 / (X**2).mean(0)], 0, N, N)

# The preconditiner must be defined by a function.

# The argument x of the preconditioner function is a matrix inside `lobpcg`, thus the use of matrix-matrix product ``@``.
#
# The preconditioner function is passed to lobpcg as a `LinearOperator`:
#

M = LinearOperator(matvec=lambda v: invA @ v, shape=(N, N), dtype=float)

# Let us now solve the eigenvalue problem for the matrix A:
#

rng = np.random.default_rng()
X0 = rng.random((N, 3))

# %time ev, X1 = lobpcg(A, X0, M=M, largest=True)
ev

for i in range(6):
    # %time ev, X0 = lobpcg(A, X0, M=M, largest=True)
X0

# Constraints:

# Note that the vectors passed in Y are the eigenvectors of the 3 smallest
#     eigenvalues. The results returned are orthogonal to those.

# The matrix constraint should be used when solving the eigen problem while embending the problem in a larger dimensional space.

Y = np.eye(N, 5)

rng = np.random.default_rng()
X0 = rng.random((N, 3))

# %time ev, X1 = lobpcg(A, X0, Y=Y, M=M, largest=True)
ev

for i in range(6):
    # %time ev, X0 = lobpcg(A, X0, Y=Y, M=M, largest=True)
X0

for i in range(6):
    # %time ev, X0 = lobpcg(W, X0, Y=Y, largest=True)
X0

# When constraints are specified, then warm start does not matter anymore.
