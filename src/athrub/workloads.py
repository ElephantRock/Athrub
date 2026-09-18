"""Controlled workloads for candidate-count and shared-context scaling studies."""

from __future__ import annotations

from collections.abc import Iterable

from .contracts import DecisionRequest


def _repeat_token(label: str, count: int) -> str:
    if count < 0:
        raise ValueError("count must be non-negative")
    return " ".join(f"{label}{index % 97}" for index in range(count))


def synthetic_request(
    *,
    request_id: str,
    prefix_units: int,
    candidate_units: int,
    candidate_count: int,
) -> DecisionRequest:
    """Create a deterministic text workload with controlled logical lengths.

    ``*_units`` are synthetic whitespace-delimited units, not tokenizer-specific
    token counts. Backends should record exact tokenizer counts separately.
    """

    if candidate_count < 2:
        raise ValueError("candidate_count must be at least two")

    state = _repeat_token("state_", prefix_units)
    question = "Select the candidate that best satisfies the decision objective."
    candidates = tuple(
        _repeat_token(f"candidate_{candidate_index}_", candidate_units)
        for candidate_index in range(candidate_count)
    )
    return DecisionRequest(
        request_id=request_id,
        state=state,
        question=question,
        candidates=candidates,
        metadata={
            "prefix_units": prefix_units,
            "candidate_units": candidate_units,
            "synthetic": True,
        },
    )


def phase1_grid(
    prefix_units: Iterable[int] = (128, 512, 1024, 2048, 4096, 8192),
    candidate_counts: Iterable[int] = (2, 4, 8, 16, 32, 64, 128, 255),
    candidate_units: int = 16,
) -> list[DecisionRequest]:
    """Generate the canonical Phase 1 scaling grid."""

    requests: list[DecisionRequest] = []
    for prefix in prefix_units:
        for count in candidate_counts:
            requests.append(
                synthetic_request(
                    request_id=f"p{prefix}-k{count}-c{candidate_units}",
                    prefix_units=prefix,
                    candidate_units=candidate_units,
                    candidate_count=count,
                )
            )
    return requests


def semantic_smoke_requests() -> list[DecisionRequest]:
    """Small cross-domain workload for semantic equivalence checks.

    These examples are not an accuracy benchmark. They exist to ensure that Phase 1
    execution changes are validated on natural decision text in addition to
    shape-controlled synthetic inputs.
    """

    return [
        DecisionRequest(
            request_id="semantic-thermal-control",
            state=(
                "A room is at 29 C, the target is 24 C, the window is closed, "
                "and the cooling system is available."
            ),
            question="Which action most directly reduces the room temperature?",
            candidates=(
                "Activate the cooling system.",
                "Turn on the room lights.",
                "Wait without changing anything.",
            ),
            metadata={"semantic": True, "domain": "control"},
        ),
        DecisionRequest(
            request_id="semantic-service-recovery",
            state=(
                "A stateless service has failed one health check after a transient "
                "network timeout. The next health check has not run yet."
            ),
            question="Which immediate action is least disruptive while gathering evidence?",
            candidates=(
                "Retry the health check.",
                "Delete the service deployment.",
                "Escalate directly to a full environment rebuild.",
                "Disable monitoring.",
            ),
            metadata={"semantic": True, "domain": "operations"},
        ),
        DecisionRequest(
            request_id="semantic-routing",
            state=(
                "A package must reach a destination before 17:00. Route A is shorter "
                "but currently congested; Route B is longer but has stable travel time."
            ),
            question="Which route is more predictable for meeting the deadline?",
            candidates=(
                "Route A, using the congested short path.",
                "Route B, using the stable longer path.",
            ),
            metadata={"semantic": True, "domain": "routing"},
        ),
    ]
