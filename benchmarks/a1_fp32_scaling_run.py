"""Issue #15 FP32 scaling campaign orchestrator (protocol v0.1).

GPU-gated: without a subcommand this prints the frozen plan and exits.
``preflight`` runs the resource-only feasibility probes (decision outputs are
discarded unread); ``measure`` runs the timed campaign (correctness, latency,
memory, anchor resolution, A2 fields). Both require explicit authorization.
All constants come from configs/a1_fp32_scaling.v0.1.json; this runner changes
none of them.
"""

# Executable src-layout imports are intentionally grouped by dependency boundary.

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from athrub.attention_runtime import AttentionPolicy, attention_runtime
from athrub.benchmark import environment_metadata
from athrub.comparison import compare_result
from athrub.contracts import DecisionRequest
from athrub.provenance import sha256_path
from athrub.reference import FlatReferenceBackend, ScalarDecisionHead
from athrub.scaling import (
    a2_performance_verdict,
    chunk_safety,
    clip_chunk_candidates,
    deterministic_cell_order,
    flat_chunked_score,
    latency_percentiles,
    repeat_alternation,
    resolve_anchor_set,
    shared_chunked_score,
)
from athrub.shared_context import SharedContextBackend
from athrub.workloads import synthetic_request

RUN1 = Path("artifacts/a0-reference-v0.1-run1")
CONTRACT_PATH = Path("configs/a1_fp32_scaling.v0.1.json")
ISSUE11_MANIFEST = Path("experiments/a1_operational_equivalence/holdout_manifest.v0.2.jsonl")
OUTPUT_DIR = Path("artifacts/a1-fp32-scaling-v0.1")


def _driver_version() -> str | None:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def _git_sha() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def load_contract() -> dict[str, Any]:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    manifest_sha = hashlib.sha256(ISSUE11_MANIFEST.read_bytes()).hexdigest()
    if manifest_sha != contract["semantic_confirmation"]["manifest_sha256"]:
        raise SystemExit("Issue #11 semantic manifest hash mismatch; refusing to execute")
    if sha256_path(RUN1 / "reference_manifest.json") != contract["binding"]["reference_manifest_sha256"]:
        raise SystemExit("frozen A0 manifest hash mismatch; refusing to execute")
    frozen = json.loads((RUN1 / "reference_manifest.json").read_text(encoding="utf-8"))
    for name, actual, recorded in (
        ("substrate", sha256_path(RUN1 / "substrate"), frozen["substrate"]["artifact_sha256"]),
        ("tokenizer", sha256_path(RUN1 / "tokenizer"), frozen["tokenizer"]["artifact_sha256"]),
        ("decision_head", sha256_path(RUN1 / "decision_head.pt"), frozen["decision_head"]["artifact_sha256"]),
    ):
        if actual != recorded:
            raise SystemExit(f"frozen A0 binding failed for {name}; refusing to execute")
    return contract


def load_fp32_pair() -> tuple[FlatReferenceBackend, SharedContextBackend]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(RUN1 / "tokenizer"))
    model = AutoModel.from_pretrained(str(RUN1 / "substrate"), dtype=torch.float32)
    head = ScalarDecisionHead(int(model.config.hidden_size), bias=True)
    head.load_state_dict(
        torch.load(str(RUN1 / "decision_head.pt"), map_location="cpu", weights_only=True), strict=True
    )
    flat = FlatReferenceBackend(
        model=model, tokenizer=tokenizer,  # type: ignore[arg-type]
        head=head, device="cuda", dtype=torch.float32,
        substrate_revision="scaling", tokenizer_revision="scaling",
    )
    shared = SharedContextBackend(
        model=flat.model, tokenizer=flat.tokenizer,  # type: ignore[arg-type]
        head=flat.head, device=flat.device, dtype=flat.dtype, codec=flat.codec,
        add_bos=flat.add_bos, substrate_revision="scaling", tokenizer_revision="scaling",
    )
    return flat, shared


def semantic_requests() -> list[DecisionRequest]:
    requests = []
    for line in ISSUE11_MANIFEST.read_text(encoding="utf-8").splitlines():
        payload = json.loads(line)
        if payload["metadata"].get("suite") != "semantic":
            continue
        requests.append(
            DecisionRequest(
                request_id=payload["request_id"], state=payload["state"], question=payload["question"],
                candidates=payload["candidates"], metadata=payload["metadata"],
            )
        )
    if len(requests) != 24:
        raise SystemExit(f"expected 24 semantic rows, found {len(requests)}")
    return requests


def grid_request(prefix: int, k: int, candidate_units: int) -> DecisionRequest:
    return synthetic_request(
        request_id=f"scale-p{prefix}-k{k}-c{candidate_units}",
        prefix_units=prefix, candidate_units=candidate_units, candidate_count=k,
    )


def probe_cell(
    flat: FlatReferenceBackend,
    shared: SharedContextBackend,
    request: DecisionRequest,
    contract: dict[str, Any],
    physical_bytes: int,
) -> dict[str, Any]:
    thresholds = contract["chunking"]["safety_thresholds"]
    candidates = clip_chunk_candidates(
        contract["chunking"]["feasibility_chunk_candidates"], len(request.candidates)
    )
    if len(request.candidates) == 255 and contract["chunking"]["full_K_probe_for_K_255"]:
        candidates.append(255)
    records = []
    max_safe = {"flat": 0, "shared": 0}
    for chunk in candidates:
        row: dict[str, Any] = {"chunk": chunk}
        for path_name in ("flat", "shared"):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            try:
                if path_name == "flat":
                    flat_chunked_score(flat, request, chunk)
                else:
                    shared_chunked_score(shared, request, chunk)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                decision = chunk_safety(
                    int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
                    int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
                    physical_bytes,
                    thresholds["peak_allocated_le_fraction_of_physical_vram"],
                    thresholds["peak_reserved_le_fraction_of_physical_vram"],
                )
                decision["status"] = "safe" if decision["safe"] else "unsafe"
            except torch.OutOfMemoryError:
                decision = {"safe": False, "status": "oom", "spill_observed": None}
            row[path_name] = decision
            if decision["safe"]:
                max_safe[path_name] = max(max_safe[path_name], chunk)
        records.append(row)
        if row["flat"]["status"] == "oom" and row["shared"]["status"] == "oom":
            break
    return {
        "request_id": request.request_id,
        "max_safe_flat": max_safe["flat"],
        "max_safe_shared": max_safe["shared"],
        "hardware_infeasible": records[0]["flat"]["status"] != "safe" or records[0]["shared"]["status"] != "safe",
        "probe_records": records,
    }


def _timed(score, backend, request, chunk) -> tuple[Any, float, dict[str, int]]:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter_ns()
    result = score(backend, request, chunk)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    memory = {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
    }
    return result, elapsed_ms, memory


def measure_cell(
    flat: FlatReferenceBackend,
    shared: SharedContextBackend,
    request: DecisionRequest,
    paths: dict[str, tuple[Any, Any, int]],
    warmups: int,
    repeats: int,
    correctness_tolerance: float,
) -> dict[str, Any]:
    """Measure named paths (name -> (score_fn, backend, chunk)) under the frozen rule.

    Alternation is by measured repeat: the flat-family group and shared-family
    group alternate going first (flat -> shared, then shared -> flat, repeating);
    within a group, paths run in name order. Every path is timed once per repeat.
    """

    reference = flat.score([request])[0]
    flat_names = sorted(name for name in paths if name.startswith("flat"))
    shared_names = sorted(name for name in paths if name.startswith("shared"))
    if len(flat_names) + len(shared_names) != len(paths):
        raise ValueError("path names must start with 'flat' or 'shared'")

    results = {}
    for name, (score, backend, chunk) in paths.items():
        for _ in range(warmups):
            score(backend, request, chunk)
        results[name] = score(backend, request, chunk)

    latencies: dict[str, list[float]] = {name: [] for name in paths}
    memories: dict[str, list[dict[str, int | None]]] = {name: [] for name in paths}
    for first, second in repeat_alternation(repeats):
        for group in (first, second):
            for name in flat_names if group == "flat" else shared_names:
                score, backend, chunk = paths[name]
                _, elapsed_ms, memory = _timed(score, backend, request, chunk)
                latencies[name].append(elapsed_ms)
                memories[name].append(memory)

    row: dict[str, Any] = {"request_id": request.request_id, "candidate_count": len(request.candidates)}
    for name, (score, backend, chunk) in paths.items():
        metrics = compare_result(reference, results[name])
        row[f"{name}_correctness"] = {
            "max_abs_probability_delta": metrics.max_abs_probability_delta,
            "argmax_equal": metrics.argmax_equal,
            "pass": metrics.max_abs_probability_delta < correctness_tolerance and metrics.argmax_equal,
            **{
                key: value
                for key, value in asdict(metrics).items()
                if key in ("total_variation", "kl_reference_to_candidate", "max_abs_logit_delta", "mean_abs_logit_delta")
            },
        }
        row[name] = {
            "chunk": chunk,
            "model_calls": results[name].metadata["model_calls"],
            "latency_ms": latency_percentiles(latencies[name]),
            "peak_memory": memories[name][-1],
        }
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", nargs="?", choices=["preflight", "measure"],
        help="GPU action; omitted prints the frozen plan only",
    )
    args = parser.parse_args()

    contract = load_contract()
    plan = {
        "protocol_version": contract["protocol_version"],
        "execution_git_sha": _git_sha(),
        "cells": [
            (prefix, k)
            for prefix in contract["grids"]["prefix_units"]
            for k in contract["grids"]["candidate_counts"]
        ],
        "anchor_cells": [tuple(cell) for cell in contract["anchor_suite"]["cells"]],
        "attribution": contract["attribution"],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.action is None:
        print(json.dumps({
            "plan": {key: value for key, value in plan.items() if key != "cells"},
            "cell_count": len(plan["cells"]),
            "note": "GPU actions (preflight, measure) require an explicit subcommand and separate authorization",
        }, indent=2))
        return
    if not torch.cuda.is_available():
        raise SystemExit("GPU action requested but CUDA is unavailable")

    policy = AttentionPolicy(**contract["binding"]["attention_policy"])
    physical_bytes = int(torch.cuda.get_device_properties(0).total_memory)
    provenance = {
        "execution_git_sha": _git_sha(),
        "contract_status": contract["status"],
        "physical_vram_bytes": physical_bytes,
        "environment": {**environment_metadata(), "nvidia_driver": _driver_version()},
        "attention_policy": policy.as_record(),
    }
    flat, shared = load_fp32_pair()
    with attention_runtime(policy):
        cells = deterministic_cell_order(plan["cells"], contract["execution_order"]["cell_order_seed"])
        requests = {cell: grid_request(cell[0], cell[1], contract["grids"]["candidate_units"]) for cell in cells}
        if args.action == "preflight":
            feasibility = {}
            for cell in cells:
                record = probe_cell(flat, shared, requests[cell], contract, physical_bytes)
                feasibility[f"p{cell[0]}-k{cell[1]}"] = record
                print(json.dumps({k: record[k] for k in ("request_id", "max_safe_flat", "max_safe_shared", "hardware_infeasible")}), flush=True)
            (OUTPUT_DIR / "feasibility.json").write_text(json.dumps(feasibility, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(json.dumps({"output": str(OUTPUT_DIR / "feasibility.json"), "cells": len(feasibility)}, indent=2))
            return
        # measure
        feasibility_path = OUTPUT_DIR / "feasibility.json"
        if not feasibility_path.is_file():
            raise SystemExit("measure requires a completed preflight (feasibility.json) first")
        feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
        infeasible = {
            (int(key[1:].split("-k")[0]), int(key.split("-k")[1]))
            for key, record in feasibility.items()
            if record["hardware_infeasible"]
        }
        anchors = resolve_anchor_set(plan["anchor_cells"], contract["grids"]["candidate_counts"], infeasible)
        (OUTPUT_DIR / "anchor_resolution.json").write_text(json.dumps(anchors, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tolerance = contract["correctness_gate"]["max_abs_probability_delta_lt"]
        cadence = contract["cadence"]
        primary_rows = []
        anchor_rows = []
        attribution_rows = []
        for cell in cells:
            request = requests[cell]
            key = f"p{cell[0]}-k{cell[1]}"
            record = feasibility[key]
            if record["hardware_infeasible"]:
                continue
            flat_chunk, shared_chunk = record["max_safe_flat"], record["max_safe_shared"]
            is_anchor = tuple(cell) in [tuple(a) for a in anchors["final_anchor_set"]]
            is_attribution = (
                cell[0] in contract["attribution"]["prefix_units"]
                and cell[1] in contract["attribution"]["candidate_counts"]
            )
            if is_anchor:
                paths = {
                    "flat_chunked": (flat_chunked_score, flat, flat_chunk),
                    "shared_chunked": (shared_chunked_score, shared, shared_chunk),
                }
                anchor_rows.append(
                    measure_cell(flat, shared, request, paths, cadence["anchor"]["warmups"], cadence["anchor"]["repeats"], tolerance)
                )
            elif is_attribution:
                common = min(flat_chunk, shared_chunk, cell[1])
                paths = {
                    "flat_sequential": (flat_chunked_score, flat, 1),
                    "flat_batched": (flat_chunked_score, flat, common),
                    "shared_sequential": (shared_chunked_score, shared, 1),
                    "shared_batched": (shared_chunked_score, shared, common),
                }
                attribution_rows.append(
                    measure_cell(flat, shared, request, paths, cadence["primary"]["warmups"], cadence["primary"]["repeats"], tolerance)
                )
            else:
                paths = {
                    "flat_chunked": (flat_chunked_score, flat, flat_chunk),
                    "shared_chunked": (shared_chunked_score, shared, shared_chunk),
                }
                primary_rows.append(
                    measure_cell(flat, shared, request, paths, cadence["primary"]["warmups"], cadence["primary"]["repeats"], tolerance)
                )
        semantic_rows = []
        for request in semantic_requests():
            record = probe_cell(flat, shared, request, contract, physical_bytes)
            if record["hardware_infeasible"]:
                continue
            paths = {
                "flat_chunked": (flat_chunked_score, flat, record["max_safe_flat"]),
                "shared_chunked": (shared_chunked_score, shared, record["max_safe_shared"]),
            }
            semantic_rows.append(
                measure_cell(flat, shared, request, paths, cadence["semantic"]["warmups"], cadence["semantic"]["repeats"], tolerance)
            )
        for name, rows in (("primary", primary_rows), ("anchor", anchor_rows), ("attribution", attribution_rows), ("semantic", semantic_rows)):
            (OUTPUT_DIR / f"{name}.jsonl").write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8"
            )
        anchor_speedups = []
        for row in anchor_rows:
            flat_mean = row["flat_chunked"]["latency_ms"]["mean"]
            shared_mean = row["shared_chunked"]["latency_ms"]["mean"]
            k = row["candidate_count"]
            anchor_speedups.append((k / shared_mean) / (k / flat_mean))
        summary = {
            "protocol_version": contract["protocol_version"],
            "anchors": anchors,
            "a2": a2_performance_verdict(anchor_speedups, anchors["unique_anchor_count"]) if anchor_speedups else None,
            "infeasible_cells": sorted(f"p{p}-k{k}" for p, k in infeasible),
            "rows": {name: len(rows) for name, rows in (("primary", primary_rows), ("anchor", anchor_rows), ("attribution", attribution_rows), ("semantic", semantic_rows))},
            "correctness_all_pass": all(
                entry["pass"]
                for rows in (primary_rows, anchor_rows, attribution_rows, semantic_rows)
                for row in rows
                for key, entry in row.items()
                if key.endswith("_correctness")
            ),
        }
        (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (OUTPUT_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
