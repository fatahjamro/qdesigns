"""Tests for mutually unbiased bases."""

import numpy as np
import pytest

import qdesigns as qd


@pytest.mark.parametrize("d", [2, 3, 5, 7, 11])
def test_complete_set_has_d_plus_one_bases(d):
    mubs = qd.mub.construct(d)
    assert mubs.count == d + 1


@pytest.mark.parametrize("d", [2, 3, 5, 7, 11, 13])
def test_mubs_verify(d):
    mubs = qd.mub.construct(d)
    cert = qd.verify(mubs)
    assert cert.ok, cert.summary()
    assert cert.checks["each basis orthonormal"]
    assert cert.checks["pairwise mutually unbiased"]


@pytest.mark.parametrize("d", [3, 5, 7])
def test_unbiasedness_value(d):
    mubs = qd.mub.construct(d)
    b0, b1 = mubs.bases[0], mubs.bases[1]
    overlaps = np.abs(b0.conj().T @ b1) ** 2
    assert np.allclose(overlaps, 1.0 / d, atol=1e-9)


def test_non_prime_not_implemented():
    with pytest.raises(NotImplementedError):
        qd.mub.construct(6)


def test_dimension_too_small():
    with pytest.raises(ValueError):
        qd.mub.construct(1)


def test_tampered_mub_fails_verification():
    mubs = qd.mub.construct(5)
    mubs.bases[1][0, 0] += 0.3  # break unbiasedness
    cert = qd.verify(mubs)
    assert not cert.ok
