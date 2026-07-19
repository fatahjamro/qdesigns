"""Tests for schema export."""

import json
import os

import qdesigns as qd


def test_export_mub(tmp_path):
    mubs = qd.mub.construct(3)
    path = os.path.join(tmp_path, "mub_d3.qdesign.json")
    qd.export(mubs, path)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["type"] == "mub_set"
    assert data["dimension"] == 3
    assert data["num_bases"] == 4
    assert data["verification"]["ok"] is True


def test_export_qls(tmp_path):
    q = qd.qls.bell_order4()
    path = os.path.join(tmp_path, "bell.qdesign.json")
    qd.export(q, path)
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["type"] == "quantum_latin_square"
    assert data["order"] == 4
    assert data["verification"]["ok"] is True
