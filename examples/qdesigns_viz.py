"""Reusable visualization helpers for qdesigns designs.

Importable from notebooks and scripts:

    import qdesigns_viz as viz
    viz.plot_mub(qd.mub.construct(4))
    viz.plot_qls(qd.qls.bell_order4())

Each function builds and returns a Matplotlib ``Figure``; it does not save or
show it, so the caller (a notebook cell, or a script) decides what to do.
"""

from __future__ import annotations

import numpy as np


def sym_entry(z: complex, d: int | None = None):
    """Return an exact sympy expression for a numeric MUB/QLS amplitude.

    Uses ``nsimplify`` with the square roots that appear in these constructions,
    so entries like ``0.5j`` become ``I/2`` and ``0.577`` becomes ``sqrt(3)/3``.
    Falls back to a rounded sympy Float if no closed form is found.
    """
    import sympy as sp

    if abs(z) < 1e-9:
        return sp.Integer(0)
    consts = [sp.sqrt(2), sp.sqrt(3), sp.sqrt(5), sp.sqrt(7)]
    if d:
        consts.append(sp.sqrt(int(d)))
    try:
        expr = sp.nsimplify(sp.sympify(complex(z)), consts, rational=False)
        # sanity: the symbolic value must match the numeric one
        if abs(complex(expr) - z) < 1e-8:
            return sp.nsimplify(expr)
    except Exception:
        pass
    return sp.N(sp.sympify(complex(z)), 4)


def to_sympy_matrix(basis, d: int | None = None):
    """Convert a numeric basis (columns = vectors) to an exact ``sympy.Matrix``."""
    import sympy as sp

    rows, cols = basis.shape
    return sp.Matrix(rows, cols, lambda i, j: sym_entry(basis[i, j], d))


def mub_symbolic(mubs) -> list:
    """Return the MUB set as a list of exact ``sympy.Matrix`` objects.

    In a Jupyter cell these render as LaTeX, e.g. ``mub_symbolic(m)[0]``.
    """
    return [to_sympy_matrix(b, mubs.dimension) for b in mubs.bases]


def plot_mub(mubs, cmap_phase: str = "twilight", cmap_mag: str = "magma", annotate: bool = False):
    """Visualize a complete MUB set.

    Top row: the phase of every amplitude in each basis (the Fourier-type bases
    show their quadratic-phase structure; the computational basis is the
    identity).  Bottom: the magnitude of all pairwise inner products for the
    stacked bases -- identity diagonal blocks (orthonormal) and uniform
    ``1/sqrt(d)`` off-diagonal blocks (mutually unbiased).

    If ``annotate=True``, the exact sympy value is written into every cell of the
    top row (best for small dimensions, e.g. d <= 5).
    """
    import matplotlib.pyplot as plt

    bases = mubs.bases
    ncol = mubs.count
    d = mubs.dimension

    fig = plt.figure(figsize=(2.0 * ncol, 6.4), constrained_layout=True)
    gs = fig.add_gridspec(2, ncol, height_ratios=[1, 2.2])

    for k, basis in enumerate(bases):
        ax = fig.add_subplot(gs[0, k])
        phase = np.ma.masked_where(np.abs(basis) < 1e-9, np.angle(basis))
        ax.imshow(phase, cmap=cmap_phase, vmin=-np.pi, vmax=np.pi)
        ax.set_title("comp." if k == ncol - 1 else f"B{k}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if annotate:
            _annotate_cells(ax, basis, d)

    stacked = np.hstack(bases)
    gram = np.abs(stacked.conj().T @ stacked)
    axg = fig.add_subplot(gs[1, :])
    im = axg.imshow(gram, cmap=cmap_mag, vmin=0, vmax=1, aspect="auto")
    for b in range(1, ncol):
        axg.axhline(b * d - 0.5, color="white", lw=0.6, alpha=0.5)
        axg.axvline(b * d - 0.5, color="white", lw=0.6, alpha=0.5)
    axg.set_title(
        f"|<e_i|f_j>| for all {ncol} bases (d={d}):  "
        f"diagonal blocks = I (orthonormal),  off-blocks = 1/√{d} ≈ "
        f"{1 / np.sqrt(d):.3f} (unbiased)",
        fontsize=10,
    )
    axg.set_xlabel("stacked basis-vector index")
    axg.set_ylabel("stacked basis-vector index")
    fig.colorbar(im, ax=axg, fraction=0.025, pad=0.01, label="|inner product|")
    return fig


def _canonical(vec: np.ndarray, tol: float = 1e-9) -> tuple:
    """Return a hashable canonical form of a state vector, fixed up to global phase.

    The first component with nonzero magnitude is rotated to be real and positive,
    so two vectors that represent the same physical state (differing only by a
    global phase) map to the same key.  This is what makes distinct-but-
    equal-magnitude cells (e.g. the Bell states Phi+ and Phi-) come out as
    *different* states, unlike ``argmax(|amp|)``.
    """
    v = np.asarray(vec, dtype=complex)
    nz = np.where(np.abs(v) > tol)[0]
    if len(nz):
        v = v * np.conj(v[nz[0]]) / np.abs(v[nz[0]])
    return tuple(np.round(v, 6).tolist())


def _state_ids(cells: np.ndarray) -> np.ndarray:
    """Assign each cell an integer id by state identity (up to global phase)."""
    n = cells.shape[0]
    ids = np.zeros((n, n), dtype=int)
    registry: dict[tuple, int] = {}
    for i in range(n):
        for j in range(n):
            key = _canonical(cells[i, j])
            ids[i, j] = registry.setdefault(key, len(registry))
    return ids


def plot_qls(qls, cmap: str = "tab20", annotate: bool = False):
    """Visualize a quantum Latin square.

    Left: each cell colored by its **state identity** (grouping vectors that are
    equal up to a global phase).  Because every row and column is an orthonormal
    basis, the n states in each row/column are distinct -- so a valid QLS shows n
    different colors per row and per column, i.e. a genuine Latin-square pattern.
    (This is faithful to entangled cells, unlike an ``argmax`` magnitude view,
    which would collapse states such as Phi+ and Phi- that share magnitudes and
    differ only in sign.)  Right: the maximum off-diagonal overlap within each
    row and column -- the QLS axiom requires these to be ~0.

    With ``annotate=True`` each cell also shows its exact state as a ket.
    """
    import matplotlib.pyplot as plt

    cells = qls.cells
    n = qls.order
    eye = np.eye(n)

    ids = _state_ids(cells)
    row_err = np.array([np.abs(cells[i].conj() @ cells[i].T - eye).max() for i in range(n)])
    col_err = np.array([np.abs(cells[:, j].conj() @ cells[:, j].T - eye).max() for j in range(n)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)

    ax1.imshow(ids, cmap=cmap, vmin=0, vmax=max(ids.max(), 1))
    for i in range(n):
        for j in range(n):
            label = f"${_ket_latex(cells[i, j], n)}$" if annotate else str(ids[i, j])
            ax1.text(j, i, label, ha="center", va="center", fontsize=6 if annotate else 9)
    ax1.set_title(f"cell state identity (order {n}) — distinct per row & column")
    ax1.set_xlabel("column")
    ax1.set_ylabel("row")
    ax1.set_xticks(range(n))
    ax1.set_yticks(range(n))

    x = np.arange(n)
    ax2.bar(x - 0.2, np.maximum(row_err, 1e-18), width=0.4, label="rows")
    ax2.bar(x + 0.2, np.maximum(col_err, 1e-18), width=0.4, label="columns")
    ax2.axhline(1e-9, color="red", ls="--", lw=1, label="tolerance 1e-9")
    ax2.set_yscale("log")
    ax2.set_ylim(1e-18, 1.0)
    ax2.set_xticks(x)
    ax2.set_xlabel("row / column index")
    ax2.set_ylabel("max off-diagonal overlap")
    ax2.set_title("orthonormality error (below the line = valid QLS)")
    ax2.legend(fontsize=8)
    return fig


def _ket_expr(vec: np.ndarray, d: int):
    """Exact sympy ket expression for a state vector, e.g. sqrt(2)*(|0> + |3>)/2."""
    import sympy as sp

    expr = sp.Integer(0)
    for k in range(len(vec)):
        c = sym_entry(vec[k], d)
        if c != 0:
            expr += c * sp.Symbol(f"|{k}>")
    return expr


def _ket_latex(vec: np.ndarray, d: int) -> str:
    """LaTeX for a state vector as a ket, suitable for matplotlib mathtext."""
    import sympy as sp

    expr = sp.Integer(0)
    for k in range(len(vec)):
        c = sym_entry(vec[k], d)
        if c != 0:
            expr += c * sp.Symbol(rf"|{k}\rangle")
    try:
        return sp.latex(expr)
    except Exception:
        return str(expr)


def qls_symbolic(qls):
    """Return the QLS cells as an exact ``sympy.Matrix`` of kets.

    In a Jupyter cell ``qls_symbolic(q)`` renders each cell as, e.g.,
    ``sqrt(2)*|0>/2 + sqrt(2)*|3>/2`` -- so Phi+ and Phi- are visibly different.
    """
    import sympy as sp

    cells = qls.cells
    n = qls.order
    return sp.Matrix(n, n, lambda i, j: _ket_expr(cells[i, j], n))


def _annotate_cells(ax, basis, d: int | None) -> None:
    """Overlay the exact sympy value (as LaTeX) on every nonzero cell."""
    import sympy as sp

    n = basis.shape[0]
    box = {"boxstyle": "round,pad=0.12", "fc": "black", "ec": "none", "alpha": 0.35}
    for i in range(n):
        for j in range(n):
            z = basis[i, j]
            if abs(z) < 1e-9:
                continue
            expr = sym_entry(z, d)
            try:
                label = f"${sp.latex(expr)}$"
            except Exception:
                label = str(expr)
            ax.text(j, i, label, ha="center", va="center", fontsize=6, color="white", bbox=box)
