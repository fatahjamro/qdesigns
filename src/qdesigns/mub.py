"""Mutually unbiased bases (MUBs).

Two orthonormal bases :math:`\\{e_i\\}` and :math:`\\{f_j\\}` of :math:`\\mathbb{C}^d`
are *mutually unbiased* if :math:`|\\langle e_i | f_j \\rangle|^2 = 1/d` for all
``i, j``.  In dimension ``d`` at most ``d + 1`` pairwise-mutually-unbiased bases
can exist, and a complete set of ``d + 1`` is known to exist whenever ``d`` is a
prime power.

This module constructs a complete set of ``d + 1`` MUBs for:

- ``d = 2`` (the Pauli eigenbases),
- odd prime ``d`` via the standard quadratic-Gauss-sum construction
  (Ivanović / Wootters--Fields),
- odd prime powers ``d = p**n`` (9, 25, 27, 49, ...) via the Galois-field
  ``GF(p**n)`` trace construction -- see :mod:`qdesigns._field`, and
- even prime powers ``d = 4, 8`` via the Galois ring ``GR(4, r)``
  (Klappenecker & Roetteler, 2004) -- see :mod:`qdesigns._galois`.

Larger even prime powers (16, 32, ...) are planned; non-prime-power dimensions
have no known complete MUB set.  All constructions here are standard, published
prior art.
"""

from __future__ import annotations

import numpy as np

from ._field import GFpn
from ._galois import gr4_mub_bases
from ._utils import ATOL, omega
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


def _mubs_odd_prime_power(p: int, n: int) -> list[np.ndarray]:
    """Complete set of ``p**n + 1`` MUBs for odd prime power ``d = p**n``.

    Uses the Galois-field trace construction: for field elements ``c`` and ``b``,
    the basis vector indexed by ``x`` has entry ``w ** Tr(c*x^2 + b*x) / sqrt(q)``,
    where ``w = exp(2*pi*i/p)`` and ``Tr: GF(q) -> GF(p)`` is the field trace.
    Field elements are indexed by integers ``0 .. q-1``.
    """
    field = GFpn(p, n)
    q = field.q
    add_t, mul_t, tr_t = field.add_table, field.mul_table, field.trace_table
    roots = omega(p) ** np.arange(p)  # p-th roots of unity, indexed by exponent

    x = np.arange(q)
    sq = mul_t[x, x]  # x^2 for every field element x
    bases: list[np.ndarray] = []
    for c in range(q):
        c_sq = mul_t[c, sq]  # c * x^2, vectorized over x
        cols = []
        for b in range(q):
            exponent = tr_t[add_t[c_sq, mul_t[b, x]]]  # Tr(c*x^2 + b*x) over x
            cols.append(roots[exponent] / np.sqrt(q))
        bases.append(np.array(cols, dtype=complex).T)
    bases.append(np.eye(q, dtype=complex))
    return bases


def _prime_power(d: int) -> tuple[int, int] | None:
    """Return ``(p, n)`` if ``d == p**n`` for a prime ``p``, else ``None``."""
    if d < 2:
        return None
    i = 2
    while i * i <= d:
        if d % i == 0:
            n = 0
            x = d
            while x % i == 0:
                x //= i
                n += 1
            return (i, n) if x == 1 else None
        i += 1
    return (d, 1)  # d is prime


def construct(dimension: int) -> MUBSet:
    """Construct a complete set of ``dimension + 1`` MUBs.

    Supported dimensions: ``2``, odd primes, odd prime powers (via ``GF(p**n)``),
    and the even prime powers ``4`` and ``8`` (via the Galois ring ``GR(4, r)``).
    Larger even prime powers (16, 32, ...) and non-prime-power dimensions raise
    :class:`NotImplementedError`.
    """
    d = int(dimension)
    if d < 2:
        raise ValueError("dimension must be >= 2")
    if d == 2:
        return MUBSet(d, _mubs_dim2(), construction="pauli-eigenbases")

    factored = _prime_power(d)
    if factored is None:
        raise NotImplementedError(
            f"dimension {d} is not a prime power; a complete set of MUBs is not "
            "known to exist and no construction is provided."
        )
    p, n = factored

    if n == 1:  # odd prime
        return MUBSet(d, _mubs_odd_prime(d), construction="wootters-fields (odd prime)")
    if p != 2:  # odd prime power
        label = f"galois-field GF({p}^{n}) trace (Wootters-Fields)"
        return MUBSet(d, _mubs_odd_prime_power(p, n), construction=label)
    if d in (4, 8):  # even prime power (currently GR(4, r))
        label = "galois-ring GR(4,r) (Klappenecker-Roetteler 2004)"
        return MUBSet(d, gr4_mub_bases(d), construction=label)
    raise NotImplementedError(
        f"dimension {d} = 2**{n}: even prime powers beyond 8 are not implemented yet."
    )
