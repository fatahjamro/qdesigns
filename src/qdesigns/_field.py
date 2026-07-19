"""Minimal finite-field GF(p**n) arithmetic for odd prime-power MUBs.

Elements of ``GF(p**n)`` are addressed by integer index ``0 .. p**n - 1`` (the
base-``p`` digits are the polynomial coefficients over ``GF(p)``).  On
construction the field precomputes addition and multiplication tables and the
field-trace table ``Tr: GF(p**n) -> GF(p)`` (``Tr(z) = sum_k z**(p**k)``), so
downstream constructions are fast and vectorizable.

The module is intentionally small and self-contained; the mathematics is
standard, published prior art.
"""

from __future__ import annotations

import numpy as np


def _trim(a: list[int]) -> list[int]:
    while a and a[-1] == 0:
        a.pop()
    return a


def _pmul(a: list[int], b: list[int], p: int) -> list[int]:
    if not a or not b:
        return []
    r = [0] * (len(a) + len(b) - 1)
    for i, ai in enumerate(a):
        if ai:
            for j, bj in enumerate(b):
                r[i + j] = (r[i + j] + ai * bj) % p
    return _trim(r)


def _pmod(a: list[int], f: list[int], p: int) -> list[int]:
    """Reduce polynomial ``a`` modulo ``f`` over GF(p) (``f`` need not be monic)."""
    a = _trim(a[:])
    f = _trim(f[:])
    df = len(f) - 1
    if df < 0:
        raise ZeroDivisionError("polynomial division by zero")
    inv_lead = pow(f[-1] % p, p - 2, p)  # inverse of leading coeff (p prime, Fermat)
    while len(a) - 1 >= df and a:
        coef = (a[-1] * inv_lead) % p
        shift = len(a) - 1 - df
        for i in range(len(f)):
            a[shift + i] = (a[shift + i] - coef * f[i]) % p
        a = _trim(a)
    return a


def _psub(a: list[int], b: list[int], p: int) -> list[int]:
    m = max(len(a), len(b))
    r = [0] * m
    for i in range(len(a)):
        r[i] = (r[i] + a[i]) % p
    for i in range(len(b)):
        r[i] = (r[i] - b[i]) % p
    return _trim(r)


def _pmulmod(a: list[int], b: list[int], f: list[int], p: int) -> list[int]:
    return _pmod(_pmul(a, b, p), f, p)


def _ppowmod(base: list[int], e: int, f: list[int], p: int) -> list[int]:
    result = [1]
    base = _pmod(base, f, p)
    while e > 0:
        if e & 1:
            result = _pmulmod(result, base, f, p)
        base = _pmulmod(base, base, f, p)
        e >>= 1
    return result


def _pgcd(a: list[int], b: list[int], p: int) -> list[int]:
    a, b = _trim(a[:]), _trim(b[:])
    while b:
        a, b = b, _pmod(a, b, p)
    return a


def _prime_factors(n: int) -> set[int]:
    factors = set()
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors.add(d)
            n //= d
        d += 1
    if n > 1:
        factors.add(n)
    return factors


def _is_irreducible(f: list[int], p: int, n: int) -> bool:
    """Rabin irreducibility test for a monic degree-``n`` polynomial over GF(p)."""
    x = [0, 1]
    if _pmod(_psub(_ppowmod(x, p**n, f, p), x, p), f, p):
        return False
    for r in _prime_factors(n):
        g = _pgcd(_psub(_ppowmod(x, p ** (n // r), f, p), x, p), f, p)
        if len(g) != 1:  # gcd is not a nonzero constant -> reducible
            return False
    return True


def _find_irreducible(p: int, n: int) -> list[int]:
    """Return a monic irreducible polynomial of degree ``n`` over GF(p)."""
    for low in range(p**n):
        coeffs = []
        v = low
        for _ in range(n):
            coeffs.append(v % p)
            v //= p
        f = coeffs + [1]  # monic degree n
        if _is_irreducible(f, p, n):
            return f
    raise RuntimeError(f"No irreducible polynomial of degree {n} over GF({p}) found")


class GFpn:
    """The finite field ``GF(p**n)`` with precomputed operation tables.

    Attributes
    ----------
    add_table, mul_table:
        ``(q, q)`` integer arrays with ``x + y`` and ``x * y`` by element index.
    trace_table:
        length-``q`` integer array with ``Tr(x)`` in ``0 .. p-1``.
    """

    def __init__(self, p: int, n: int) -> None:
        self.p = int(p)
        self.n = int(n)
        self.q = self.p**self.n
        self.modulus = _find_irreducible(self.p, self.n)
        self._coeff_cache = [self._to_coeffs(i) for i in range(self.q)]
        self.add_table = self._build_add_table()
        self.mul_table = self._build_mul_table()
        self.trace_table = self._build_trace_table()

    def _to_coeffs(self, idx: int) -> list[int]:
        out = []
        for _ in range(self.n):
            out.append(idx % self.p)
            idx //= self.p
        return out

    def _to_index(self, coeffs: list[int]) -> int:
        idx = 0
        for i in range(self.n):
            c = coeffs[i] if i < len(coeffs) else 0
            idx += (c % self.p) * (self.p**i)
        return idx

    def _build_add_table(self) -> np.ndarray:
        q, p = self.q, self.p
        table = np.zeros((q, q), dtype=np.int64)
        for a in range(q):
            ca = self._coeff_cache[a]
            for b in range(q):
                cb = self._coeff_cache[b]
                table[a, b] = self._to_index([(ca[i] + cb[i]) % p for i in range(self.n)])
        return table

    def _build_mul_table(self) -> np.ndarray:
        q = self.q
        table = np.zeros((q, q), dtype=np.int64)
        for a in range(q):
            ca = self._coeff_cache[a]
            for b in range(q):
                prod = _pmulmod(ca, self._coeff_cache[b], self.modulus, self.p)
                table[a, b] = self._to_index(prod)
        return table

    def _power(self, a: int, e: int) -> int:
        result, base = 1, a
        while e > 0:
            if e & 1:
                result = int(self.mul_table[result, base])
            base = int(self.mul_table[base, base])
            e >>= 1
        return result

    def _build_trace_table(self) -> np.ndarray:
        q, p, n = self.q, self.p, self.n
        table = np.zeros(q, dtype=np.int64)
        for a in range(q):
            acc = 0
            for k in range(n):
                acc = int(self.add_table[acc, self._power(a, p**k)])
            coeffs = self._coeff_cache[acc]
            assert all(c == 0 for c in coeffs[1:]), "trace did not land in GF(p)"
            table[a] = coeffs[0]
        return table
