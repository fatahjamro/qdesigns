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


def plot_mub(mubs, cmap_phase: str = "twilight", cmap_mag: str = "magma"):
    """Visualize a complete MUB set.

    Top row: the phase of every amplitude in each basis (the Fourier-type bases
    show their quadratic-phase structure; the computational basis is the
    identity).  Bottom: the magnitude of all pairwise inner products for the
    stacked bases -- identity diagonal blocks (orthonormal) and uniform
    ``1/sqrt(d)`` off-diagonal blocks (mutually unbiased).
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


def plot_qls(qls, cmap: str = "tab20"):
    """Visualize a quantum Latin square.

    Left: the dominant computational component of each cell (an exact Latin
    square when the cells are computational basis vectors).  Right: the maximum
    off-diagonal overlap within each row and each column -- the QLS axiom
    requires every row and column to be an orthonormal basis, so all bars should
    sit far below the tolerance line.
    """
    import matplotlib.pyplot as plt

    cells = qls.cells
    n = qls.order
    eye = np.eye(n)

    symbols = np.argmax(np.abs(cells), axis=2)
    row_err = np.array([np.abs(cells[i].conj() @ cells[i].T - eye).max() for i in range(n)])
    col_err = np.array([np.abs(cells[:, j].conj() @ cells[:, j].T - eye).max() for j in range(n)])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)

    ax1.imshow(symbols, cmap=cmap, vmin=0, vmax=max(n - 1, 1))
    for i in range(n):
        for j in range(n):
            ax1.text(j, i, str(symbols[i, j]), ha="center", va="center", fontsize=9)
    ax1.set_title(f"dominant component per cell (order {n})")
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
