#!/usr/bin/env python3
"""Visualize a complete set of mutually unbiased bases.

Usage
-----
    python examples/visualize_mub.py [d] [--save out.png]

Prints each basis and the verification certificate, and renders two views:

1. The phase of every amplitude in each basis (the d Fourier-type bases show
   their quadratic-phase structure; the computational basis is the identity).
2. The magnitude of all pairwise inner products ``|<e_i|f_j>|`` for the stacked
   bases.  Diagonal blocks are identity (each basis is orthonormal); every
   off-diagonal block is uniform at ``1/sqrt(d)`` (the bases are unbiased).
"""

from __future__ import annotations

import sys

import numpy as np

import qdesigns as qd


def main() -> None:
    d = int(sys.argv[1]) if len(sys.argv) > 1 and not sys.argv[1].startswith("-") else 7
    save = None
    if "--save" in sys.argv:
        save = sys.argv[sys.argv.index("--save") + 1]

    mubs = qd.mub.construct(d)
    print(f"MUB set for d={d}: {mubs.count} bases  [{mubs.construction}]\n")
    for k, basis in enumerate(mubs.bases):
        tag = "computational" if k == mubs.count - 1 else f"basis {k}"
        print(f"--- {tag} (columns are basis vectors) ---")
        print(np.array2string(np.round(basis, 3), max_line_width=120))
        print()

    cert = qd.verify(mubs)
    print(cert.summary())
    print(f"\nexpected off-block overlap |<e_i|f_j>| = 1/sqrt(d) = {1/np.sqrt(d):.4f}")

    try:
        _render(mubs, d, save)
    except Exception as exc:  # matplotlib optional
        print(f"\n(plot skipped: {exc}. Install matplotlib to enable the figure.)")


def _render(mubs, d: int, save: str | None) -> None:
    import matplotlib.pyplot as plt
    import qdesigns_viz as viz

    fig = viz.plot_mub(mubs)
    if save:
        fig.savefig(save, dpi=150, bbox_inches="tight")
        print(f"\nfigure saved -> {save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
