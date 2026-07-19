<h1 align="center">qdesigns</h1>
<p align="center"><strong>A verified, machine-readable atlas &amp; toolkit for quantum combinatorial designs.</strong></p>
<p align="center"><em>Mutually unbiased bases · complex Hadamard matrices · SIC-POVMs · quantum Latin squares · unitary error bases · AME / k-uniform states · t-designs — constructed, cross-linked, and certified.</em></p>

<p align="center">
  <img alt="status" src="https://img.shields.io/badge/status-early%20development-orange">
  <img alt="license" src="https://img.shields.io/badge/license-Apache%202.0-blue">
  <img alt="python" src="https://img.shields.io/badge/python-3.10%2B-blue">
</p>

---

## What is qdesigns?

`qdesigns` is an open-source library and data collection for the combinatorial and algebraic objects that underpin quantum information: **mutually unbiased bases (MUBs), complex Hadamard matrices, SIC-POVMs, (quantum) Latin squares and orthogonal arrays, unitary error bases, absolutely maximally entangled (AME) / k-uniform states, and unitary/projective t-designs.**

For each object family, `qdesigns` aims to do three things well:

- **Construct** it, across the known families and dimensions, from a clean Python API.
- **Verify** it, producing a re-runnable certificate (certified numerics and/or exact number-field checks) that the object really has the properties it claims.
- **Export** it, in a common machine-readable schema so results are reproducible and consumable by other tools.

The goal is a single, trustworthy, version-controlled source of truth for these objects — the kind of reference the community can cite and build on.

---

## Why it exists

These objects sit at the foundation of quantum error correction, tomography, teleportation, dense coding, benchmarking, and quantum key distribution, and they are deeply interrelated — MUBs are complex projective 2-designs, MUBs are built from Hadamard matrices and from quantum Latin squares, unitary error bases are built from Latin squares and Hadamard matrices, and MUBs and SIC-POVMs are the optimal measurements for quantum state tomography.

Despite this, the existing software and data are fragmented and hard to reuse: catalogues are static web pages with no machine-readable export, exact data is often locked in a single proprietary computer-algebra format, some families (notably MUBs) have no curated collection at all, and there is no standard way to *certify* that a claimed object is valid. `qdesigns` is an effort to fix that with constructions, verification, and a shared data format in one place.

---

## Scope

| Object family | What qdesigns provides |
|---|---|
| Mutually unbiased bases (MUBs) | construction across dimensions, unbiasedness/orthonormality verification, export |
| Quantum Latin squares (QLS) | construction and orthogonality checks, including composite-POVM Latin squares |
| Complex Hadamard matrices | construction and verification of known families |
| SIC-POVMs | construction/ingestion and certification of fiducial vectors |
| Unitary error bases (UEBs) | construction from combinatorial data, with derived protocols/error models |
| AME / k-uniform states | construction from orthogonal arrays and MDS codes |
| Unitary &amp; projective t-designs | construction and frame-potential verification |

Support for these families is being added incrementally; see the issues and commit history for current status.

---

## Quick start

> ⚠️ Early development — the API is not yet stable (coming soon).

```python
import qdesigns as qd

# Construct and verify a complete set of MUBs in a prime-power dimension
mubs = qd.mub.construct(dimension=5)     # (d + 1) mutually unbiased bases
report = qd.verify(mubs)                  # certified unbiasedness + orthonormality
print(report.summary())                   # PASS, with certificate

# Work with quantum Latin squares
qls = qd.qls.from_array(my_quantum_latin_square)
assert qd.verify(qls).ok

# Export in the machine-readable schema
qd.export(mubs, "mub_d5.qdesign.json")
```

---

## Following along

The repository is under active development. The clearest picture of what is implemented and what is in progress is the commit history, the open issues, and the test suite. Contributions — new construction families, verification methods, certified data entries, and honest benchmarks — are welcome; please open an issue to discuss scope before large pull requests, since correctness and reproducibility are the project's first priorities.

---

## License

Licensed under the **Apache License 2.0**. See [`LICENSE`](./LICENSE) for details.

---

<p align="center"><sub>Building the verified backbone of quantum information — one design at a time.</sub></p>
