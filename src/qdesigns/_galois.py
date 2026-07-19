"""Galois-ring GR(4, r) arithmetic for even prime-power MUB constructions.

For even prime-power dimensions ``d = 2**r`` the Wootters--Fields Galois-*field*
trace formula does not apply.  The standard construction instead uses the Galois
*ring* ``GR(4, r) = Z4[x] / (h(x))``, where ``h`` is a monic basic-primitive
Hensel lift of a primitive degree-``r`` polynomial over GF(2).  The ``d + 1``
mutually unbiased bases are

    B_a  (a in the Teichmuller set T):  v_{a,b}[x] = i ** Tr((a + 2b) x) / sqrt(d)

together with the computational basis.  See Klappenecker & Roetteler,
"Constructions of Mutually Unbiased Bases" (2004).

This module implements ``r = 2`` and ``r = 3`` (dimensions 4 and 8).  The
mathematics here is standard, published prior art.
"""

from __future__ import annotations

import numpy as np

# Low-order coefficients of a primitive polynomial over GF(2):
# r=2 -> x^2 + x + 1 ; r=3 -> x^3 + x + 1.
_GF2_PRIMITIVE = {2: (1, 1), 3: (1, 1, 0)}


class GaloisRing4:
    """Arithmetic in ``GR(4, r)``.  Elements are length-``r`` tuples of ints mod 4."""

    def __init__(self, r: int) -> None:
        if r not in _GF2_PRIMITIVE:
            raise NotImplementedError(f"GR(4, r) is only implemented for r in {{2, 3}}; got r={r}")
        self.r = r
        self.d = 2**r
        self.zero = tuple([0] * r)
        self.one = tuple([1] + [0] * (r - 1))
        self.h_low = self._find_modulus()
        self.T = self._teichmuller()
        self._decomp = self._build_decomposition()

    def add(self, a, b):
        return tuple((x + y) % 4 for x, y in zip(a, b, strict=True))

    def scal(self, c, a):
        return tuple((c * x) % 4 for x in a)

    def _mul_raw(self, a, b, h_low):
        r = self.r
        conv = [0] * (2 * r - 1)
        for i, ai in enumerate(a):
            if ai:
                for j, bj in enumerate(b):
                    conv[i + j] = (conv[i + j] + ai * bj) % 4
        # reduce: x^(r+t) = -sum_i h_low[i] x^(i+t)
        for p in range(2 * r - 2, r - 1, -1):
            c = conv[p]
            if c:
                conv[p] = 0
                for i, hi in enumerate(h_low):
                    conv[p - r + i] = (conv[p - r + i] - c * hi) % 4
        return tuple(conv[:r])

    def mul(self, a, b):
        return self._mul_raw(a, b, self.h_low)

    def _find_modulus(self):
        """Hensel lift: monic ``h`` over Z4 congruent to the primitive polynomial
        mod 2, with ``x`` of multiplicative order exactly ``2**r - 1``."""
        base = _GF2_PRIMITIVE[self.r]
        order = self.d - 1
        xg = tuple([0, 1] + [0] * (self.r - 2))
        for mask in range(2**self.r):
            h_low = tuple((base[i] + 2 * ((mask >> i) & 1)) % 4 for i in range(self.r))
            p = self.one
            ok = True
            for _ in range(1, order):
                p = self._mul_raw(p, xg, h_low)
                if p == self.one:  # order divides j < 2**r - 1 -> reject
                    ok = False
                    break
            if ok and self._mul_raw(p, xg, h_low) == self.one:
                return h_low
        raise RuntimeError(f"No basic-primitive modulus found for r={self.r}")

    def _teichmuller(self):
        xg = tuple([0, 1] + [0] * (self.r - 2))
        T = [self.zero, self.one]
        p = self.one
        for _ in range(self.d - 2):
            p = self.mul(p, xg)
            T.append(p)
        return T

    def _build_decomposition(self):
        """Bijection ``GR(4, r) <-> T x T`` via ``z = a + 2b``."""
        dec = {}
        for a in self.T:
            for b in self.T:
                z = self.add(a, self.scal(2, b))
                dec[z] = (a, b)
        assert len(dec) == 4**self.r
        return dec

    def frobenius(self, z):
        a, b = self._decomp[z]
        return self.add(self.mul(a, a), self.scal(2, self.mul(b, b)))

    def trace(self, z) -> int:
        acc, t = z, z
        for _ in range(self.r - 1):
            t = self.frobenius(t)
            acc = self.add(acc, t)
        assert all(c == 0 for c in acc[1:]), "trace did not land in Z4"
        return acc[0]


def gr4_mub_bases(d: int) -> list[np.ndarray]:
    """Complete set of ``d + 1`` MUBs for ``d = 2**r`` (r = 2, 3) via ``GR(4, r)``.

    Returns a list of ``(d, d)`` complex arrays whose columns are the basis
    vectors, ordered as ``d`` Galois-ring bases followed by the computational
    basis (matching the Wootters--Fields ordering used elsewhere).
    """
    r = {4: 2, 8: 3}.get(d)
    if r is None:
        raise NotImplementedError(f"gr4_mub_bases supports d in {{4, 8}}; got d={d}")
    ring = GaloisRing4(r)
    bases: list[np.ndarray] = []
    for a in ring.T:
        cols = []
        for b in ring.T:
            a2b = ring.add(a, ring.scal(2, b))
            vec = np.array([1j ** ring.trace(ring.mul(a2b, x)) for x in ring.T], dtype=complex)
            cols.append(vec / np.sqrt(d))
        bases.append(np.array(cols, dtype=complex).T)
    bases.append(np.eye(d, dtype=complex))
    return bases
