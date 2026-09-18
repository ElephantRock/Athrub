"""Backend protocol for interchangeable Athrub implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .contracts import DecisionRequest, DecisionResult


@runtime_checkable
class DecisionBackend(Protocol):
    """Minimal interface required by the Phase 1 benchmark harness."""

    @property
    def name(self) -> str:
        """Stable backend identifier written into benchmark artifacts."""

    def score(self, requests: Sequence[DecisionRequest]) -> Sequence[DecisionResult]:
        """Score requests while preserving request and candidate order."""
