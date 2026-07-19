"""Small numerical helpers shared across qdesigns modules."""

from __future__ import annotations

import numpy as np

# Default tolerance for numerical certification of design properties.
ATOL = 1e-9


def is_prime(n: int) -> bool:
    """Return True if ``n`` is a prime number."""
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def omega(d: int) -> complex:
    """Primitive ``d``-th root of unity ``exp(2*pi*i / d)``."""
    return np.exp(2j * np.pi / d)


def is_unit_vector(v: np.ndarray, atol: float = ATOL) -> bool:
    """Return True if ``v`` has unit Euclidean norm."""
    return bool(np.isclose(np.vdot(v, v).real, 1.0, atol=atol))


def is_orthonormal_basis(columns: np.ndarray, atol: float = ATOL) -> bool:
    """Return True if the columns of the square matrix form an orthonormal basis.

    ``columns`` is a ``(d, d)`` array whose columns are the candidate vectors;
    the check is ``columns^dagger @ columns == I``.
    """
    d = columns.shape[0]
    gram = columns.conj().T @ columns
    return bool(np.allclose(gram, np.eye(d), atol=atol))
