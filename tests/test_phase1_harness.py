from athrub.comparison import compare_result, summarize
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
