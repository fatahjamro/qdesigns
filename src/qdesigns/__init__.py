"""qdesigns: a verified, machine-readable toolkit for quantum combinatorial designs.

Public API
----------
- :mod:`qdesigns.mub`  -- mutually unbiased bases
- :mod:`qdesigns.qls`  -- quantum Latin squares
- :func:`qdesigns.verify`  -- certify a design object
- :func:`qdesigns.export`  -- serialize a design object to the qdesign schema
"""

from __future__ import annotations

from . import mub, qls
from ._io import export, to_dict
from .verify import Certificate, verify

__version__ = "0.1.0"

__all__ = [
    "mub",
    "qls",
    "verify",
    "Certificate",
    "export",
    "to_dict",
    "__version__",
]
