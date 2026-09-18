"""Phase 1 benchmark runner with deterministic measurement metadata."""

from __future__ import annotations

import json
import platform
import statistics
import time
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from pathlib import Path

import torch

from .backends import DecisionBackend
from .contracts import BenchmarkRecord, DecisionRequest, DecisionResult


def _sync_if_needed() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _reset_memory_stats() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _memory_stats() -> tuple[int | None, int | None]:
    if not torch.cuda.is_available():
        return None, None
    return int(torch.cuda.max_memory_allocated()), int(torch.cuda.max_memory_reserved())


def _results_by_request(
    requests: Sequence[DecisionRequest], results: Sequence[DecisionResult]
) -> dict[str, DecisionResult]:
    if len(results) != len(requests):
        raise ValueError("backend returned a different number of results than requests")

    output: dict[str, DecisionResult] = {}
    for request, result in zip(requests, results, strict=True):
        if result.request_id != request.request_id:
            raise ValueError("backend did not preserve request order")
        if result.request_id in output:
            raise ValueError("duplicate request id in benchmark batch")
        output[result.request_id] = result
    return output


def _benchmark_shape(
    request: DecisionRequest, result: DecisionResult
) -> tuple[int, int, dict[str, object]]:
    metadata: dict[str, object] = {
        "candidate_token_counts": result.metadata.get("candidate_token_counts"),
        "path_token_counts": result.metadata.get("path_token_counts"),
        "flat_logical_token_positions": result.metadata.get("flat_logical_token_positions"),
        "substrate_revision": result.metadata.get("substrate_revision"),
        "tokenizer_revision": result.metadata.get("tokenizer_revision"),
        "precision": result.metadata.get("precision"),
    }
    metadata = {key: value for key, value in metadata.items() if value is not None}

    prefix_tokens = result.metadata.get("prefix_tokens")
    candidate_counts = result.metadata.get("candidate_token_counts")
    prefix_units = (
        int(prefix_tokens)
        if prefix_tokens is not None
        else int(request.metadata.get("prefix_units", 0))
    )
    if isinstance(candidate_counts, (tuple, list)):
        candidate_units = sum(int(value) for value in candidate_counts)
    else:
        candidate_units = int(request.metadata.get("candidate_units", 0))
    return prefix_units, candidate_units, metadata


def benchmark_batch(
    backend: DecisionBackend,
    requests: Sequence[DecisionRequest],
    *,
    warmups: int = 5,
    repeats: int = 20,
) -> list[BenchmarkRecord]:
    """Benchmark one fixed request batch.

    The backend is responsible for tokenization and model execution. The timing is
    therefore end-to-end inside ``backend.score``. When a backend reports exact
    tokenizer counts or revision metadata, those values are copied into benchmark
    artifacts instead of relying on synthetic workload units.
    """

    if not requests:
        raise ValueError("benchmark batch must not be empty")
    if warmups < 0 or repeats < 1:
        raise ValueError("warmups must be >= 0 and repeats must be >= 1")

    for _ in range(warmups):
        _results_by_request(requests, backend.score(requests))
    _sync_if_needed()

    records: list[BenchmarkRecord] = []
    for repeat in range(repeats):
        _reset_memory_stats()
        _sync_if_needed()
        start = time.perf_counter_ns()
        results = backend.score(requests)
        _sync_if_needed()
        latency_ms = (time.perf_counter_ns() - start) / 1_000_000.0
        peak_allocated, peak_reserved = _memory_stats()
        indexed = _results_by_request(requests, results)

        for request in requests:
            result = indexed[request.request_id]
            prefix_units, candidate_units, backend_metadata = _benchmark_shape(request, result)
            records.append(
                BenchmarkRecord(
                    backend=backend.name,
                    request_id=request.request_id,
                    candidate_count=len(request.candidates),
                    prefix_units=prefix_units,
                    candidate_units=candidate_units,
                    latency_ms=latency_ms,
                    peak_allocated_bytes=peak_allocated,
                    peak_reserved_bytes=peak_reserved,
                    metadata={
                        "repeat": repeat,
                        "batch_size": len(requests),
                        **backend_metadata,
                    },
                )
            )
    return records


def latency_summary(records: Iterable[BenchmarkRecord]) -> dict[str, float]:
    values = sorted(record.latency_ms for record in records)
    if not values:
        raise ValueError("records must not be empty")

    def percentile(p: float) -> float:
        if len(values) == 1:
            return values[0]
        position = (len(values) - 1) * p
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    return {
        "mean_ms": statistics.fmean(values),
        "p50_ms": percentile(0.50),
        "p90_ms": percentile(0.90),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
    }


def environment_metadata() -> dict[str, object]:
    metadata: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        metadata.update(
            {
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
            }
        )
    return metadata


def write_jsonl(path: str | Path, records: Iterable[BenchmarkRecord]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), sort_keys=True) + "\n")
