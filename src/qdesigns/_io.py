"""Serialization of design objects to the machine-readable qdesign schema."""

from __future__ import annotations

import json
from typing import Any

import numpy as np

from .mub import MUBSet
from .qls import QuantumLatinSquare
from .verify import verify

SCHEMA_VERSION = "qdesign/v0.1"


def _complex_array_to_json(arr: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(arr, dtype=complex)
    return {"shape": list(arr.shape), "real": arr.real.tolist(), "imag": arr.imag.tolist()}


def to_dict(obj: Any) -> dict[str, Any]:
    """Serialize a design object to a plain dict following the qdesign schema."""
    cert = verify(obj)
    if isinstance(obj, MUBSet):
        payload = {
            "schema": SCHEMA_VERSION,
            "type": "mub_set",
            "dimension": obj.dimension,
            "num_bases": obj.count,
            "construction": obj.construction,
            "bases": [_complex_array_to_json(b) for b in obj.bases],
            "verification": {"ok": cert.ok, "checks": cert.checks, "atol": cert.atol},
        }
        return payload
    if isinstance(obj, QuantumLatinSquare):
        return {
            "schema": SCHEMA_VERSION,
            "type": "quantum_latin_square",
            "order": obj.order,
            "name": obj.name,
            "cells": _complex_array_to_json(obj.cells),
            "verification": {"ok": cert.ok, "checks": cert.checks, "atol": cert.atol},
        }
    raise TypeError(f"Cannot serialize object of type {type(obj).__name__!r}.")


def export(obj: Any, path: str) -> str:
    """Write a design object to ``path`` as JSON in the qdesign schema.

    Returns the path written.
    """
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(to_dict(obj), fh, indent=2)
    return path
