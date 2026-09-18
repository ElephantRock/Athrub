"""Stable data contracts used by Athrub experiments and benchmarks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class DecisionRequest:
    """A bounded probabilistic decision request.

    Candidate order is semantically meaningful for reproducibility. Callers should
    preserve it exactly across reference and optimized implementations.
    """

    request_id: str
    state: str
    question: str
    candidates: Sequence[str]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.question:
            raise ValueError("question must be non-empty")
        if len(self.candidates) < 2:
            raise ValueError("at least two candidates are required")
        if any(not candidate for candidate in self.candidates):
            raise ValueError("candidates must be non-empty strings")


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """Raw and normalized outputs for one decision request."""

    request_id: str
    logits: Sequence[float]
    probabilities: Sequence[float]
    predicted_index: int
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.logits) != len(self.probabilities):
            raise ValueError("logits and probabilities must have equal length")
        if not self.probabilities:
            raise ValueError("result must contain at least one probability")
        if not 0 <= self.predicted_index < len(self.probabilities):
            raise ValueError("predicted_index is out of range")


@dataclass(frozen=True, slots=True)
class BenchmarkRecord:
    """One measured benchmark observation."""

    backend: str
    request_id: str
    candidate_count: int
    prefix_units: int
    candidate_units: int
    latency_ms: float
    peak_allocated_bytes: int | None = None
    peak_reserved_bytes: int | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)
