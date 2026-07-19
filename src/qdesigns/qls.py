"""Quantum Latin squares (QLS).

A *quantum Latin square* of order ``n`` is an ``n x n`` grid whose cells are unit
vectors in :math:`\\mathbb{C}^n`, such that the vectors in every row form an
orthonormal basis and the vectors in every column form an orthonormal basis
(Musto & Vicary, 2015).  When every cell is a computational basis vector, this
reduces to a classical Latin square.

Representation: a QLS of order ``n`` is stored as a complex array of shape
``(n, n, n)``, where ``cells[i, j]`` is the length-``n`` state vector in row
``i``, column ``j``.
"""

from __future__ import annotations

import numpy as np

from ._utils import ATOL
from .verify import Certificate


class QuantumLatinSquare:
    """An order-``n`` quantum Latin square.

    Parameters
    ----------
    cells:
        Complex array of shape ``(n, n, n)``; ``cells[i, j]`` is the state vector
        at row ``i``, column ``j``.
    name:
        Optional human-readable label.
    """

    def __init__(self, cells: np.ndarray, name: str = "") -> None:
        cells = np.asarray(cells, dtype=complex)
        if cells.ndim != 3 or cells.shape[0] != cells.shape[1] or cells.shape[0] != cells.shape[2]:
            raise ValueError("cells must have shape (n, n, n); got " f"{cells.shape}")
        self.cells = cells
        self.name = name

    @property
    def order(self) -> int:
        """The order ``n`` of the square."""
        return self.cells.shape[0]

    def __repr__(self) -> str:
        return f"QuantumLatinSquare(order={self.order}, name={self.name!r})"

    def _verify(self, atol: float = ATOL) -> Certificate:
        n = self.order
        checks: dict[str, bool] = {}
        eye = np.eye(n)

        rows_ok = True
        for i in range(n):
            basis = self.cells[i].T  # columns are the n row-vectors
            if not np.allclose(basis.conj().T @ basis, eye, atol=atol):
                rows_ok = False
        checks["every row is an orthonormal basis"] = rows_ok

        cols_ok = True
        for jcol in range(n):
            basis = self.cells[:, jcol].T  # columns are the n column-vectors
            if not np.allclose(basis.conj().T @ basis, eye, atol=atol):
                cols_ok = False
        checks["every column is an orthonormal basis"] = cols_ok

        ok = rows_ok and cols_ok
        return Certificate(
            object_type="quantum_latin_square",
            ok=ok,
            checks=checks,
            atol=atol,
            details={"order": n, "name": self.name or "(unnamed)"},
        )


def from_array(cells: np.ndarray, name: str = "") -> QuantumLatinSquare:
    """Build a :class:`QuantumLatinSquare` from an ``(n, n, n)`` array of vectors."""
    return QuantumLatinSquare(cells, name=name)


def from_latin_square(square: np.ndarray, name: str = "") -> QuantumLatinSquare:
    """Build a (classical) QLS from a Latin square of symbols ``0..n-1``.

    Cell ``(i, j)`` becomes the computational basis vector ``|square[i, j]>``.
    """
    square = np.asarray(square, dtype=int)
    n = square.shape[0]
    if square.shape != (n, n):
        raise ValueError("square must be n x n")
    cells = np.zeros((n, n, n), dtype=complex)
    for i in range(n):
        for j in range(n):
            cells[i, j, square[i, j]] = 1.0
    return QuantumLatinSquare(cells, name=name or "classical")


def bell_order4() -> QuantumLatinSquare:
    """The order-4 Bell-state quantum Latin square.

    This is the square studied in the QLS hardware-optimization experiments:
    a 4x4 grid whose cells are the 2-qubit computational basis states and the
    four Bell states, laid out so that every row and column is an orthonormal
    basis of :math:`\\mathbb{C}^4`.  Basis index convention: ``|ab>`` maps to
    index ``2*a + b``.
    """
    s = 1 / np.sqrt(2)
    e = np.eye(4, dtype=complex)
    ket = {"00": e[0], "01": e[1], "10": e[2], "11": e[3]}
    bell = {
        "Phi+": s * (e[0] + e[3]),
        "Phi-": s * (e[0] - e[3]),
        "Psi+": s * (e[1] + e[2]),
        "Psi-": s * (e[1] - e[2]),
    }
    lookup = {**ket, **bell}
    grid = [
        ["00", "01", "10", "11"],
        ["Psi+", "Phi+", "Phi-", "Psi-"],
        ["11", "10", "01", "00"],
        ["Psi-", "Phi-", "Phi+", "Psi+"],
    ]
    cells = np.zeros((4, 4, 4), dtype=complex)
    for i, row in enumerate(grid):
        for j, label in enumerate(row):
            cells[i, j] = lookup[label]
    return QuantumLatinSquare(cells, name="bell-order4")
