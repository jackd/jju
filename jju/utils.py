import jax
import jax.numpy as jnp


def standardize_signs(v: jnp.ndarray) -> jnp.ndarray:
    """Get `w = s*v` such that `max(abs(w)) == max(w) >= 0` and `abs(s) == 1`."""
    val = v[jnp.argmax(jnp.abs(v))]
    if v.dtype in (jnp.complex64, jnp.complex128):
        return v * jnp.abs(val) / val  # make real
    return v * jnp.sign(val)


def standardize_eigenvector_signs(v: jnp.ndarray) -> jnp.ndarray:
    """Get eigenvectors with standardized signs. See `standardize_signs`."""
    return jax.vmap(standardize_signs, 1, 1)(v)


def symmetrize(A: jnp.ndarray) -> jnp.ndarray:
    """Make symmetric and hermitian."""
    return (A + A.conj().T) / 2


def bilinear_form(x: jnp.ndarray, A: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """Return `x.T @ A @ y`."""
    return x.T @ A @ y
