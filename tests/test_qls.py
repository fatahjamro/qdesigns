"""Tests for quantum Latin squares."""

import numpy as np
import pytest

import qdesigns as qd


def test_bell_order4_is_valid_qls():
    q = qd.qls.bell_order4()
    assert q.order == 4
    cert = qd.verify(q)
    assert cert.ok, cert.summary()
    assert cert.checks["every row is an orthonormal basis"]
    assert cert.checks["every column is an orthonormal basis"]


def test_classical_latin_square_is_valid_qls():
    # A cyclic Latin square of order 4.
    n = 4
    square = np.array([[(i + j) % n for j in range(n)] for i in range(n)])
    q = qd.qls.from_latin_square(square)
    assert qd.verify(q).ok


def test_non_latin_square_fails():
    # Row with a repeated symbol is not a Latin square -> not a valid QLS.
    square = np.array([[0, 0], [1, 0]])
    q = qd.qls.from_latin_square(square)
    assert not qd.verify(q).ok


def test_bad_shape_rejected():
    with pytest.raises(ValueError):
        qd.qls.from_array(np.zeros((3, 3), dtype=complex))


def test_from_array_roundtrip():
    q = qd.qls.bell_order4()
    q2 = qd.qls.from_array(q.cells, name="copy")
    assert np.allclose(q.cells, q2.cells)
    assert qd.verify(q2).ok
