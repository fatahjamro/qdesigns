"""Mutually unbiased bases (MUBs).

Two orthonormal bases :math:`\\{e_i\\}` and :math:`\\{f_j\\}` of :math:`\\mathbb{C}^d`
are *mutually unbiased* if :math:`|\\langle e_i | f_j \\rangle|^2 = 1/d` for all
``i, j``.  In dimension ``d`` at most ``d + 1`` pairwise-mutually-unbiased bases
can exist, and a complete set of ``d + 1`` is known to exist whenever ``d`` is a
prime power.

This module currently constructs a complete set of ``d + 1`` MUBs for ``d = 2``
and for odd prime ``d`` using the standard quadratic-Gauss-sum construction
(Ivanović / Wootters--Fields).  Prime-power (non-prime) dimensions via Galois
fields are planned.
"""

from __future__ import annotations

import numpy as np

from ._utils import ATOL, is_prime, omega
from .verify import Certificate


class MUBSet:
    """A set of mutually unbiased bases in a fixed dimension.

    Parameters
    ----------
    dimension:
        The Hilbert-space dimension ``d``.
    bases:
        A list of ``(d, d)`` complex arrays.  The columns of each array are the
        vectors of one orthonormal basis.
    construction:
        Human-readable label for how the set was built.
    """

    def __init__(self, dimension: int, bases: list[np.ndarray], construction: str = "") -> None:
        self.dimension = int(dimension)
        self.bases = [np.asarray(b, dtype=complex) for b in bases]
        self.construction = construction

    @property
    def count(self) -> int:
        """Number of bases in the set."""
        return len(self.bases)

    def __repr__(self) -> str:
        return (
            f"MUBSet(dimension={self.dimension}, count={self.count}, "
            f"construction={self.construction!r})"
        )

    def _verify(self, atol: float = ATOL) -> Certificate:
        d = self.dimension
        checks: dict[str, bool] = {}

        # 1. Every basis is orthonormal.
        ortho_ok = True
        for b in self.bases:
            gram = b.conj().T @ b
            if not np.allclose(gram, np.eye(d), atol=atol):
                ortho_ok = False
        checks["each basis orthonormal"] = ortho_ok

        # 2. Every pair of distinct bases is mutually unbiased.
        unbiased_ok = True
        target = 1.0 / d
        for a in range(self.count):
            for b in range(a + 1, self.count):
                overlaps = np.abs(self.bases[a].conj().T @ self.bases[b]) ** 2
                if not np.allclose(overlaps, target, atol=atol):
                    unbiased_ok = False
        checks["pairwise mutually unbiased"] = unbiased_ok

        ok = ortho_ok and unbiased_ok
        return Certificate(
            object_type="mub_set",
            ok=ok,
            checks=checks,
            atol=atol,
            details={"dimension": d, "num_bases": self.count, "construction": self.construction},
        )


def _mubs_dim2() -> list[np.ndarray]:
    """The three MUBs in dimension 2: eigenbases of Z, X, Y."""
    s = 1 / np.sqrt(2)
    z = np.array([[1, 0], [0, 1]], dtype=complex)
    x = np.array([[s, s], [s, -s]], dtype=complex)
    y = np.array([[s, s], [1j * s, -1j * s]], dtype=complex)
    return [z, x, y]


def _mubs_odd_prime(p: int) -> list[np.ndarray]:
    """Complete set of ``p + 1`` MUBs for odd prime ``p``.

    Basis ``k`` (for ``k = 0, ..., p-1``) has column ``a`` with entries
    ``w ** ((a*j + k*j*j) mod p) / sqrt(p)`` for ``j = 0, ..., p-1``, plus the
    computational basis.
    """
    w = omega(p)
    j = np.arange(p)
    bases: list[np.ndarray] = []
    for k in range(p):
        cols = []
        for a in range(p):
            exponent = (a * j + k * j * j) % p
            cols.append(w**exponent / np.sqrt(p))
        bases.append(np.array(cols, dtype=complex).T)  # columns = basis vectors
    bases.append(np.eye(p, dtype=complex))  # computational basis
    return bases


def construct(dimension: int) -> MUBSet:
    """Construct a complete set of ``dimension + 1`` MUBs.

    Supported: ``dimension == 2`` and odd prime ``dimension``.  Non-prime
    prime-power dimensions raise :class:`NotImplementedError` for now.
    """
    d = int(dimension)
    if d < 2:
        raise ValueError("dimension must be >= 2")
    if d == 2:
        return MUBSet(d, _mubs_dim2(), construction="pauli-eigenbases")
    if is_prime(d):
        return MUBSet(d, _mubs_odd_prime(d), construction="wootters-fields (odd prime)")
    raise NotImplementedError(
        f"dimension {d} is not prime; prime-power constructions via Galois "
        "fields are not implemented yet."
    )
