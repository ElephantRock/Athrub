"""Numerical equivalence metrics for reference and optimized decision backends."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .contracts import DecisionResult


@dataclass(frozen=True, slots=True)
class ComparisonMetrics:
    request_id: str
    argmax_equal: bool
    max_abs_logit_delta: float
    mean_abs_logit_delta: float
    max_abs_probability_delta: float
    mean_abs_probability_delta: float
    total_variation: float
    kl_reference_to_candidate: float


def _kl_divergence(reference: np.ndarray, candidate: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(reference.astype(np.float64), eps, 1.0)
    q = np.clip(candidate.astype(np.float64), eps, 1.0)
    return float(np.sum(p * np.log(p / q)))


def compare_result(reference: DecisionResult, candidate: DecisionResult) -> ComparisonMetrics:
    if reference.request_id != candidate.request_id:
        raise ValueError("cannot compare results with different request ids")

    ref_logits = np.asarray(reference.logits, dtype=np.float64)
    cand_logits = np.asarray(candidate.logits, dtype=np.float64)
    ref_prob = np.asarray(reference.probabilities, dtype=np.float64)
    cand_prob = np.asarray(candidate.probabilities, dtype=np.float64)

    if ref_logits.shape != cand_logits.shape or ref_prob.shape != cand_prob.shape:
        raise ValueError("reference and candidate shapes differ")
    if not math.isclose(float(ref_prob.sum()), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("reference probabilities do not sum to one")
    if not math.isclose(float(cand_prob.sum()), 1.0, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError("candidate probabilities do not sum to one")

    logit_delta = np.abs(ref_logits - cand_logits)
    prob_delta = np.abs(ref_prob - cand_prob)

    return ComparisonMetrics(
        request_id=reference.request_id,
        argmax_equal=reference.predicted_index == candidate.predicted_index,
        max_abs_logit_delta=float(logit_delta.max(initial=0.0)),
        mean_abs_logit_delta=float(logit_delta.mean()),
        max_abs_probability_delta=float(prob_delta.max(initial=0.0)),
        mean_abs_probability_delta=float(prob_delta.mean()),
        total_variation=float(0.5 * prob_delta.sum()),
        kl_reference_to_candidate=_kl_divergence(ref_prob, cand_prob),
    )


def summarize(metrics: Iterable[ComparisonMetrics]) -> dict[str, float | int]:
    rows = list(metrics)
    if not rows:
        raise ValueError("at least one comparison metric is required")

    return {
        "requests": len(rows),
        "argmax_agreement": sum(row.argmax_equal for row in rows) / len(rows),
        "max_abs_logit_delta": max(row.max_abs_logit_delta for row in rows),
        "max_abs_probability_delta": max(row.max_abs_probability_delta for row in rows),
        "mean_total_variation": float(np.mean([row.total_variation for row in rows])),
        "mean_kl_reference_to_candidate": float(
            np.mean([row.kl_reference_to_candidate for row in rows])
        ),
    }
