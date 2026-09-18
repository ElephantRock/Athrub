"""Run the canonical flat reference benchmark under the tracked attention runtime.

Emits the canonical semantic and shape-controlled benchmark artifacts for a frozen
reference bundle: complete decision outputs (logits, probabilities, tokenizer
accounting), per-repeat latency, and peak CUDA memory, plus benchmark and
environment records that pin the exact attention policy and library versions.

Record layout matches the frozen A0 canonical bundle so regressions compare
decision fields directly.
"""

# Ruff's import sorter classifies this executable's src-layout imports differently from
# the installed package. Keep the dependency groups explicit and suppress only I001.
# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

from athrub.attention_runtime import AttentionPolicy, attention_runtime
from athrub.benchmark import environment_metadata
from athrub.provenance import sha256_path
from athrub.reference import FlatReferenceBackend, ScalarDecisionHead
from athrub.workloads import semantic_smoke_requests, synthetic_request


DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def _driver_version() -> str | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return output.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def _transformers_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("transformers")
    except PackageNotFoundError:  # pragma: no cover - optional dependency metadata
        return None


def _load_backend(config: dict[str, Any], dtype: torch.dtype, substrate_sha: str, tokenizer_sha: str) -> FlatReferenceBackend:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(config["tokenizer_path"]))
    model = AutoModel.from_pretrained(str(config["substrate_path"]), dtype=dtype)
    head = ScalarDecisionHead(int(model.config.hidden_size), bias=bool(config.get("head_bias", True)))
    head.load_state_dict(
        torch.load(str(config["decision_head_path"]), map_location="cpu", weights_only=True),
        strict=True,
    )
    return FlatReferenceBackend(
        model=model,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        head=head,
        device=str(config.get("device", "cuda")),
        dtype=dtype,
        add_bos=bool(config.get("add_bos", True)),
        substrate_revision=f"sha256:{substrate_sha[:16]}",
        tokenizer_revision=f"sha256:{tokenizer_sha[:16]}",
        name="athrub-reference-flat",
    )


def _measured_score(backend: FlatReferenceBackend, requests: list, warmups: int, repeats: int) -> list[dict[str, Any]]:
    for _ in range(warmups):
        backend.score(requests)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter_ns()
        results = backend.score(requests)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        rows.append(
            {
                "repeat": repeat,
                "latency_ms": (time.perf_counter_ns() - start) / 1_000_000.0,
                "peak_allocated": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
                "peak_reserved": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
                "results": results,
            }
        )
    return rows


def _decision_record(
    backend: FlatReferenceBackend,
    request: Any,
    result: Any,
    measurement: dict[str, Any],
    precision: str,
    substrate_sha: str,
    head_sha: str,
) -> dict[str, Any]:
    metadata = result.metadata
    prefix_tokens = metadata.get("prefix_tokens")
    candidate_counts = metadata.get("candidate_token_counts")
    prefix_units = (
        int(prefix_tokens)
        if prefix_tokens is not None
        else int(request.metadata.get("prefix_units", 0))
    )
    candidate_units = (
        sum(int(value) for value in candidate_counts)
        if isinstance(candidate_counts, (tuple, list))
        else int(request.metadata.get("candidate_units", 0))
    )
    return {
        "backend": backend.name,
        "request_id": request.request_id,
        "candidate_count": len(request.candidates),
        "candidates": list(request.candidates),
        "prefix_units": prefix_units,
        "candidate_units": candidate_units,
        "logits": list(result.logits),
        "probabilities": list(result.probabilities),
        "predicted_index": int(result.predicted_index),
        "prefix_tokens": metadata.get("prefix_tokens"),
        "candidate_token_counts": list(metadata.get("candidate_token_counts") or []),
        "path_token_counts": list(metadata.get("path_token_counts") or []),
        "flat_logical_token_positions": metadata.get("flat_logical_token_positions"),
        "precision": precision,
        "substrate_revision": metadata.get("substrate_revision"),
        "tokenizer_revision": metadata.get("tokenizer_revision"),
        "substrate_artifact_sha256": substrate_sha,
        "decision_head_sha256": head_sha,
        "repeat": measurement["repeat"],
        "batch_size": measurement["batch_size"],
        "latency_ms": measurement["latency_ms"],
        "peak_allocated_bytes": measurement["peak_allocated"],
        "peak_reserved_bytes": measurement["peak_reserved"],
    }


def _cross_repeat_drift(rows: list[dict[str, Any]], requests: list) -> float:
    worst = 0.0
    for request_index, request in enumerate(requests):
        baseline = rows[0]["results"][request_index]
        for row in rows[1:]:
            other = row["results"][request_index]
            for a, b in zip(baseline.logits, other.logits, strict=True):
                worst = max(worst, abs(float(a) - float(b)))
    return worst


def _write_line(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _parse_cells(spec: str) -> list[tuple[int, int]]:
    cells = []
    for chunk in spec.split(","):
        prefix, count = chunk.strip().split(":")
        cells.append((int(prefix), int(count)))
    return cells


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--cells", help="optional shape-cell filter, e.g. '128:2,512:16'")
    parser.add_argument("--precisions", help="optional precision filter, e.g. 'fp32' or 'fp32,bf16'")
    parser.add_argument("--suites", help="optional suite filter, e.g. 'semantic', 'shape', 'semantic,shape'")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    policy = AttentionPolicy(**dict(config.get("attention_policy", {})))
    precisions = (
        [value.strip() for value in args.precisions.split(",")]
        if args.precisions
        else list(config.get("precisions", ["fp32", "bf16"]))
    )
    unknown = set(precisions) - set(DTYPES)
    if unknown:
        raise ValueError(f"unsupported precisions: {sorted(unknown)}")
    suites = (
        {value.strip() for value in args.suites.split(",")}
        if args.suites
        else {"semantic", "shape"}
    )

    substrate_path = Path(str(config["substrate_path"]))
    tokenizer_path = Path(str(config["tokenizer_path"]))
    head_path = Path(str(config["decision_head_path"]))
    substrate_sha = sha256_path(substrate_path)
    tokenizer_sha = sha256_path(tokenizer_path)
    head_sha = sha256_path(head_path)

    semantic_cfg = dict(config.get("semantic", {}))
    shape_cfg = dict(config.get("shape", {}))
    prefix_units = [int(value) for value in shape_cfg.get("prefix_units", [128, 512, 1024])]
    candidate_counts = [int(value) for value in shape_cfg.get("candidate_counts", [2, 8, 16])]
    candidate_units = int(shape_cfg.get("candidate_units", 16))
    cells = _parse_cells(args.cells) if args.cells else [
        (prefix, count) for prefix in prefix_units for count in candidate_counts
    ]

    output_dir = Path(str(config.get("output_dir", "artifacts/canonical-benchmark")))
    session_report: dict[str, Any] | None = None

    for precision in precisions:
        backend = _load_backend(config, DTYPES[precision], substrate_sha, tokenizer_sha)
        with attention_runtime(policy) as session:
            session.observe_model_config(backend.model.config)
            if "semantic" in suites:
                requests = semantic_smoke_requests()
                warmups = int(semantic_cfg.get("warmups", 5))
                repeats = int(semantic_cfg.get("repeats", 20))
                rows = _measured_score(backend, requests, warmups, repeats)
                drift = _cross_repeat_drift(rows, requests)
                print(f"semantic/{precision}: {len(rows)} repeats, drift {drift}", flush=True)
                path = output_dir / "semantic" / f"{precision}.jsonl"
                path.unlink(missing_ok=True)
                for row in rows:
                    for request_index, request in enumerate(requests):
                        _write_line(
                            path,
                            _decision_record(
                                backend,
                                request,
                                row["results"][request_index],
                                {**row, "batch_size": len(requests)},
                                precision,
                                substrate_sha,
                                head_sha,
                            ),
                        )
            if "shape" in suites:
                warmups = int(shape_cfg.get("warmups", 2))
                repeats = int(shape_cfg.get("repeats", 10))
                path = output_dir / "shape" / f"{precision}.jsonl"
                path.unlink(missing_ok=True)
                for prefix, count in cells:
                    request = synthetic_request(
                        request_id=f"shape-p{prefix}-k{count}-c{candidate_units}",
                        prefix_units=prefix,
                        candidate_units=candidate_units,
                        candidate_count=count,
                    )
                    rows = _measured_score(backend, [request], warmups, repeats)
                    drift = _cross_repeat_drift(rows, [request])
                    mean_ms = sum(row["latency_ms"] for row in rows) / len(rows)
                    print(
                        f"shape/{precision} p{prefix} k{count}: mean {mean_ms:.2f} ms, drift {drift}",
                        flush=True,
                    )
                    for row in rows:
                        _write_line(
                            path,
                            _decision_record(
                                backend,
                                request,
                                row["results"][0],
                                {**row, "batch_size": 1},
                                precision,
                                substrate_sha,
                                head_sha,
                            ),
                        )
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            session_report = session.metadata()
        del backend
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    benchmark_record = {
        "reference_name": config.get("reference_name"),
        "backend": "athrub-reference-flat",
        "device": str(config.get("device", "cuda")),
        "precisions": precisions,
        "suites": sorted(suites),
        "semantic": {"requests": 3, **semantic_cfg},
        "shape": {
            "cells": [f"p{prefix}-k{count}" for prefix, count in cells],
            "candidate_units": candidate_units,
            "per_cell_execution": "individual",
            **{key: value for key, value in shape_cfg.items() if key not in {"prefix_units", "candidate_counts", "candidate_units"}},
        },
        "benchmarked_artifact": {
            "substrate_path": str(substrate_path),
            "substrate_bundle_sha256": substrate_sha,
            "tokenizer_path": str(tokenizer_path),
            "tokenizer_bundle_sha256": tokenizer_sha,
            "decision_head_path": str(head_path),
            "decision_head_sha256": head_sha,
        },
        **(session_report or {}),
    }
    (output_dir / "benchmark_config.json").write_text(
        json.dumps(benchmark_record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    environment = dict(environment_metadata())
    environment["nvidia_driver"] = _driver_version()
    environment["transformers"] = _transformers_version()
    environment["measurement"] = (
        "per-repeat perf_counter_ns with CUDA synchronization and peak-memory-stat "
        "resets, mirroring athrub.benchmark.benchmark_batch"
    )
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
