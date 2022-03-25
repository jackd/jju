---
jupyter:
  jupytext:
    encoding: '# -*- coding: utf-8 -*-'
    formats: ipynb,py:light,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.3
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region -->

# JAX implementations

JAX expose numpy and scipy implementations of `eigh` function but at the moment, advanced options such as `eigvals` are not available.

Here are some interesting links:
- [jackd/jju: Jack's Jax Utilities](https://github.com/jackd/jju)

In particular, it contains a JAX implementation of LOBPCG discussed above:
- https://github.com/jackd/jju/blob/master/jju/linalg/lobpcg/basic.py#L11

The issue https://github.com/google/jax/issues/3112 from JAX's Github is also discussing an implementation.

### Benchmarking with numpy

I found this study:
https://towardsdatascience.com/turbocharging-svd-with-jax-749ae12f93af?gi=398628dbfc88
which compare jax and numy algorithms.

There is in the end no big difference between jax and numpy from the speed perspective.


Other benchmarking results can be found in this study:
- [Separate your filters! Separability, SVD and low-rank approximation of 2D image processing filters | Bart Wronski](https://arxiv.org/pdf/2009.07542.pdf)

This recent paper (2019):
- [Differentiable Programming Tensor Networks
Hai-Jun Liao, Jin-Guo Liu, Lei Wang, Tao Xiang](https://arxiv.org/abs/1903.09650).
discuss which implementations of eigh may be recommended when used in differentiable pipelines.
They have an implementation of differentiable SVD in Pytorch.
<!-- #endregion -->

```python
%load_ext autoreload
%autoreload 2
```

```python
import jax
import jax.numpy as jnp

from jju.linalg.lobpcg.basic import lobpcg

```

```python
T = 100
N = 5000
```

```python
rng = jax.random.PRNGKey(42)
X = jax.random.uniform(rng, (T, N))
W = X.T @ X / T
X0 = jax.random.uniform(rng, (N, 3))
```


## test LOBPCG

```python
X0 = jax.random.uniform(rng, (N, 3))
```


```python
%time ev, _ = scipy.sparse.linalg.lobpcg(onp.array(W).astype(onp.float64), onp.array(X0).astype(onp.float64))
ev
```

```python
Y = None
```


```python
# ??lobpcg
```

```python
rng = jax.random.PRNGKey(42)
X = jax.random.uniform(rng, (T, N))
W = X.T @ X / T
X0 = jax.random.uniform(rng, (N, 3))

%time ev, X1 = lobpcg(W, X0, None, None, None, True, None, None, 1000); ev.block_until_ready(); ev.block_until_ready()
ev
```

Second run is faster:

```python
%time ev, X1 = lobpcg(W, X0, None, None, None, True, None, None, 1000); ev.block_until_ready(); ev.block_until_ready()
ev
```

### Warm-start

```python
%time ev, X1 = lobpcg(W, X1, None, None, None, True, None, None, 1000); ev.block_until_ready(); ev.block_until_ready()
```

## Transpose approach

```python
rng = jax.random.PRNGKey(42)
X = jax.random.uniform(rng, (T, N))
Wsmall = X @ X.T / T
X0 = jax.random.normal(rng, (T, 3))

%time ev, X1 = lobpcg(Wsmall, X0, None, None, None, True, None, None, 1000); ev.block_until_ready(); ev.block_until_ready()
ev
```

```python
%time ev, X1 =  lobpcg(Wsmall, X0, None, None, None, True, None, None, 1000); ev.block_until_ready()
```

```python
nx = len(ev)
```

```python
%time Uhat, s, V = jax.scipy.linalg.svd(X); Uhat.block_until_ready();
```

```python
Q = X1
B = Q.T @ X
%time Uhat, s, V = jax.scipy.linalg.svd(B); Uhat.block_until_ready()
U = Q @ Uhat
U.shape
U = V[:nx].T
U.T @ U
```

```python
Cov = X.T @ X / T
jnp.diag(U.T @ Cov @ U)
```

```python
ev
```

### test with constraints

```python
N, T = 5000, 100
rng = jax.random.PRNGKey(42)
def generate_wishart(N=1000, T=1100):
    X = jax.random.normal(rng, (T, N))
    W = X.T @ X / T
    return W, X
W, X = generate_wishart(N, T)
X0 = jax.random.normal(rng, (N, 3))
Y = jnp.eye(N, 5)
```

```python
Y
```

```python
invA = jnp.diag(1.0 / (X**2).mean(0))
```

```python
%time ev, X1 =  lobpcg(jnp.array(W), jnp.array(X0), Y=jnp.array(Y),largest=True, max_iters=39); ev.block_until_ready();#iK=invA, 
X1
```

```python
ev
```

Starting with max_iters=40 the algorithm give nans.
We have to fix it.

```python
# from jax.config import config; config.update("jax_enable_x64", False)
%time ev, X1 =  lobpcg(jnp.array(W, "float32"), jnp.array(X0, "float32"), Y=jnp.array(Y, "float32"),largest=True, max_iters=40); ev.block_until_ready();#iK=invA, 
X1
```

```python
from jax.config import config; config.update("jax_enable_x64", True)
```

```python
%%time 
ev, X1 =  lobpcg(jnp.array(W, "float64"), jnp.array(X0, "float64"), Y=jnp.array(Y, "float64"), largest=True, max_iters=40); ev.block_until_ready();
#iK=invA, 
X1.block_until_ready()
```

```python
ev
```

## scipy run 


Let's compare to what gives scipy.


```python
if False:
    import numpy as np
    print(N, T)
    N, T = 5000, 100

    def generate_wishart(N=1000, T=1100):
        X = np.random.randn(T, N)
        W = X.T @ X / T
        return W, X
    W, X = generate_wishart(N, T)
    rng = np.random.default_rng()
    X0 = rng.random((N, 3))
    Y = np.eye(N, 5)
```

```python
%time ev, X1 = scipy.sparse.linalg.lobpcg(onp.array(W), X0, Y=Y, largest=True, maxiter=39)
X1
```

```python
ev
```

```python
%time ev, X1 = scipy.sparse.linalg.lobpcg(onp.array(W, float), onp.array(X0, float), Y=onp.array(Y, float), largest=True, maxiter=39)
X1
```

```python
ev
```

It converges better with computations in float64.
