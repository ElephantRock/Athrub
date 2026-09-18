"""Deterministic test backend for exercising benchmark infrastructure only."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from .contracts import DecisionRequest, DecisionResult


class DeterministicTestBackend:
    """CPU-only backend used by CI; it is not a model or research baseline."""

    def __init__(self, name: str = "deterministic-test") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @staticmethod
    def _logit(request: DecisionRequest, candidate: str) -> float:
        payload = f"{request.state}\n{request.question}\n{candidate}".encode()
        digest = hashlib.sha256(payload).digest()
        integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
        return (integer / float(2**64 - 1) - 0.5) * 8.0

    def score(self, requests: Sequence[DecisionRequest]) -> Sequence[DecisionResult]:
        outputs: list[DecisionResult] = []
        for request in requests:
            logits = np.asarray(
                [self._logit(request, candidate) for candidate in request.candidates],
                dtype=np.float64,
            )
            shifted = logits - logits.max()
            exp = np.exp(shifted)
            probabilities = exp / exp.sum()
            outputs.append(
                DecisionResult(
                    request_id=request.request_id,
                    logits=tuple(float(value) for value in logits),
                    probabilities=tuple(float(value) for value in probabilities),
                    predicted_index=int(np.argmax(probabilities)),
                    metadata={"testing_only": True},
                )
            )
        return outputs
