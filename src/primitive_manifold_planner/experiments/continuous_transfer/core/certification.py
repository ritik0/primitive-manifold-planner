"""Thin certification result wrappers for Example 65 strict validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from ..graph_types import StrictValidationFailure


@dataclass(frozen=True)
class CertificationResult:
    """Result wrapper around the existing strict validation failure structure."""

    valid: bool
    failures: list[StrictValidationFailure] = field(default_factory=list)

    @classmethod
    def from_failures(cls, failures: Iterable[StrictValidationFailure]) -> "CertificationResult":
        failure_list = list(failures)
        return cls(valid=len(failure_list) == 0, failures=failure_list)

