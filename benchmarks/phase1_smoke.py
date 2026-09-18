"""Smoke benchmark for validating the Phase 1 harness.

This uses a deterministic test backend and is not a model benchmark.
"""

from __future__ import annotations

import json

from athrub.benchmark import benchmark_batch, environment_metadata, latency_summary
from athrub.testing import DeterministicTestBackend
from athrub.workloads import synthetic_request


def main() -> None:
    backend = DeterministicTestBackend()
    request = synthetic_request(
        request_id="smoke-p128-k8-c16",
        prefix_units=128,
        candidate_units=16,
        candidate_count=8,
    )
    records = benchmark_batch(backend, [request], warmups=2, repeats=5)
    print(json.dumps({"environment": environment_metadata()}, indent=2, sort_keys=True))
    print(json.dumps({"latency": latency_summary(records)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
