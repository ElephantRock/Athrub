import pytest

from athrub.comparison import compare_result, summarize
from athrub.contracts import DecisionResult
from athrub.testing import DeterministicTestBackend
from athrub.workloads import synthetic_request


def test_deterministic_backend_is_probability_preserving() -> None:
    backend = DeterministicTestBackend()
    request = synthetic_request(
        request_id="case",
        prefix_units=16,
        candidate_units=4,
        candidate_count=4,
    )

    first = backend.score([request])[0]
    second = backend.score([request])[0]
    metrics = compare_result(first, second)

    assert metrics.argmax_equal
    assert metrics.max_abs_logit_delta == 0.0
    assert metrics.max_abs_probability_delta == 0.0
    assert metrics.total_variation == 0.0
    assert metrics.kl_reference_to_candidate == 0.0


def test_summary_reports_complete_agreement() -> None:
    backend = DeterministicTestBackend()
    requests = [
        synthetic_request(
            request_id=f"case-{count}",
            prefix_units=8,
            candidate_units=2,
            candidate_count=count,
        )
        for count in (2, 4, 8)
    ]

    reference = backend.score(requests)
    candidate = backend.score(requests)
    summary = summarize(compare_result(a, b) for a, b in zip(reference, candidate, strict=True))

    assert summary["requests"] == 3
    assert summary["argmax_agreement"] == 1.0
    assert summary["max_abs_probability_delta"] == 0.0


def _result(probabilities: tuple[float, ...], predicted_index: int) -> DecisionResult:
    logit_base = 3.0
    logits = tuple(logit_base + value for value in probabilities)
    return DecisionResult(
        request_id="near-tie",
        logits=logits,
        probabilities=probabilities,
        predicted_index=predicted_index,
    )


def test_near_tie_diagnostics_flag_small_reference_margin() -> None:
    reference = _result((0.50001, 0.49999), predicted_index=0)
    candidate = _result((0.49999, 0.50001), predicted_index=1)

    flagged = compare_result(reference, candidate, near_tie_threshold=2e-5)
    assert flagged.reference_top1_index == 0
    assert flagged.reference_top2_index == 1
    assert flagged.reference_probability_margin == pytest.approx(2e-5, abs=1e-12)
    assert flagged.near_tie is True
    assert flagged.near_tie_threshold == 2e-5
    # Near-tie status is diagnostic only: the argmax flip still fails the gate.
    assert flagged.argmax_equal is False


def test_near_tie_diagnostics_leave_decisive_margins_unflagged() -> None:
    reference = _result((0.9, 0.05, 0.05), predicted_index=0)
    candidate = _result((0.9, 0.05, 0.05), predicted_index=0)

    metrics = compare_result(reference, candidate, near_tie_threshold=2e-5)
    assert metrics.reference_top1_probability == 0.9
    assert metrics.reference_probability_margin == 0.85
    assert metrics.near_tie is False
    assert metrics.argmax_equal is True


def test_near_tie_diagnostics_default_to_disabled() -> None:
    reference = _result((0.50001, 0.49999), predicted_index=0)
    candidate = _result((0.50001, 0.49999), predicted_index=0)

    metrics = compare_result(reference, candidate)
    assert metrics.near_tie is False
    assert metrics.near_tie_threshold is None
    assert metrics.reference_top1_index == 0
    assert metrics.reference_top2_index == 1
