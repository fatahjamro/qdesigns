"""Tests for mutually unbiased bases."""

import numpy as np
import pytest

import qdesigns as qd


@pytest.mark.parametrize("d", [2, 3, 4, 5, 7, 8, 9, 11, 25, 27])
def test_complete_set_has_d_plus_one_bases(d):
    mubs = qd.mub.construct(d)
    assert mubs.count == d + 1


@pytest.mark.parametrize("d", [2, 3, 4, 5, 7, 8, 9, 11, 13, 25, 27, 49])
def test_mubs_verify(d):
    mubs = qd.mub.construct(d)
    cert = qd.verify(mubs)
    assert cert.ok, cert.summary()
    assert cert.checks["each basis orthonormal"]
    assert cert.checks["pairwise mutually unbiased"]


@pytest.mark.parametrize("d", [4, 8])
def test_even_prime_power_uses_galois_ring(d):
    mubs = qd.mub.construct(d)
    assert "galois-ring" in mubs.construction


@pytest.mark.parametrize("d", [9, 25, 27, 49])
def test_odd_prime_power_uses_galois_field(d):
    mubs = qd.mub.construct(d)
    assert "galois-field" in mubs.construction


@pytest.mark.parametrize("d", [3, 5, 7])
def test_unbiasedness_value(d):
    mubs = qd.mub.construct(d)
    b0, b1 = mubs.bases[0], mubs.bases[1]
    overlaps = np.abs(b0.conj().T @ b1) ** 2
    assert np.allclose(overlaps, 1.0 / d, atol=1e-9)


def test_non_prime_power_not_implemented():
    # 6, 10, 12, 15 are not prime powers -> no known complete MUB set.
    for d in (6, 10, 12, 15):
        with pytest.raises(NotImplementedError):
            qd.mub.construct(d)


def test_large_even_prime_power_not_implemented():
    # 16 = 2**4 is a prime power, but the GR(4, r) path currently covers only 4, 8.
    with pytest.raises(NotImplementedError):
        qd.mub.construct(16)


def test_dimension_too_small():
    with pytest.raises(ValueError):
        qd.mub.construct(1)


def test_tampered_mub_fails_verification():
    mubs = qd.mub.construct(5)
    mubs.bases[1][0, 0] += 0.3  # break unbiasedness
    cert = qd.verify(mubs)
    assert not cert.ok
