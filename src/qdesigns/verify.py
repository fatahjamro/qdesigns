"""Verification and certificates for quantum design objects.

Every design object in :mod:`qdesigns` can be handed to :func:`verify`, which
returns a :class:`Certificate` recording exactly which properties were checked,
whether each passed, and the numerical tolerance used.  The certificate is the
"glass box": it is meant to be reproducible and human-readable, not a silent
boolean.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ._utils import ATOL


@dataclass
class Certificate:
    """Result of verifying a design object.

    Attributes
    ----------
    object_type:
        A short string describing what was verified (e.g. ``"mub_set"``).
    ok:
        ``True`` iff every recorded check passed.
    checks:
        Mapping from check name to a boolean pass/fail.
    atol:
        Absolute tolerance used for the numerical checks.
    details:
        Free-form extra information (dimensions, counts, ...).
    """

    object_type: str
    ok: bool
    checks: dict[str, bool] = field(default_factory=dict)
    atol: float = ATOL
    details: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Return a concise, human-readable multi-line report."""
        status = "PASS" if self.ok else "FAIL"
        lines = [f"[{status}] {self.object_type} (atol={self.atol:g})"]
        for name, passed in self.checks.items():
            mark = "ok" if passed else "XX"
            lines.append(f"  [{mark}] {name}")
        for key, value in self.details.items():
            lines.append(f"    - {key}: {value}")
        return "\n".join(lines)

    def __bool__(self) -> bool:  # allows: if verify(obj): ...
        return self.ok


def verify(obj: Any, atol: float = ATOL) -> Certificate:
    """Verify a design object and return a :class:`Certificate`.

    Dispatches on the object's own ``_verify`` method so new design families can
    plug in without editing this function.
    """
    verifier = getattr(obj, "_verify", None)
    if verifier is None:
        raise TypeError(
            f"Object of type {type(obj).__name__!r} is not a verifiable "
            "qdesigns object (no _verify method)."
        )
    return verifier(atol=atol)
