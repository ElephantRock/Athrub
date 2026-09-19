"""Issue #15 FP32 scaling campaign orchestrator (protocol v0.1).

GPU-gated: without a subcommand this prints the frozen plan and exits.
``preflight`` runs the resource-only feasibility probes (decision outputs are
discarded unread) over all grid cells AND the 24 semantic rows, embedding a
binding block in feasibility.json. ``measure`` re-verifies that binding before
trusting the evidence, then runs the timed campaign with per-path FP32
correctness gates. The K=255 full-K probe is probe-only and can never become
the operational chunk. All constants come from the frozen contract.
"""

# Executable src-layout imports are intentionally grouped by dependency boundary.

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    WddmSharedUsageMonitor,
    a2_performance_verdict,
    chunk_safety,
    clip_chunk_candidates,
    deterministic_cell_order,
    feasibility_binding,
    flat_chunked_score,
    is_memory_stop_decision,
    is_oom_exception,
    latency_percentiles,
    repeat_alternation,
    require_complete_feasibility,
    resolve_anchor_set,
    resume_partial_feasibility,
    shared_chunked_score,
    stop_reason,
    suite_membership,
    verdict_thresholds,
    verify_feasibility_binding,
)
from athrub.shared_context import SharedContextBackend
from athrub.workloads import synthetic_request

RUN1 = Path("artifacts/a0-reference-v0.1-run1")
CONTRACT_PATH = Path("configs/a1_fp32_scaling.v0.2.json")
ISSUE11_MANIFEST = Path("experiments/a1_operational_equivalence/holdout_manifest.v0.2.jsonl")
OUTPUT_DIR = Path("artifacts/a1-fp32-scaling-v0.2")
PARTIAL_FEASIBILITY_PATH = OUTPUT_DIR / "feasibility_partial.json"


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
    if contract.get("protocol_version") != "0.2" or "memory_stop_rule" not in contract.get("chunking", {}):
        raise SystemExit("runner implements amended protocol v0.2; contract must declare it and its memory-stop rule")
    if hashlib.sha256(ISSUE11_MANIFEST.read_bytes()).hexdigest() != contract["semantic_confirmation"]["manifest_sha256"]:
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


class WddmTelemetry(WddmSharedUsageMonitor):
    """Continuous per-PID shared-usage windows around each probe.

    The base monitor streams the Windows GPU Process Memory Shared Usage
    counter once per preflight and isolates this process's PID; probe_cell
    opens a window before each probe and closes it after the post-forward
    sync, capturing the in-window peak so transient spills are not missed and
    unrelated desktop GPU activity is never attributed to the harness.
    """



def probe_cell(
    flat: FlatReferenceBackend,
    shared: SharedContextBackend,
    request: DecisionRequest,
    contract: dict[str, Any],
    physical_bytes: int,
    telemetry: WddmTelemetry | None = None,
) -> dict[str, Any]:
    thresholds = contract["chunking"]["safety_thresholds"]
    candidates = clip_chunk_candidates(
        contract["chunking"]["feasibility_chunk_candidates"], len(request.candidates)
    )
    probe_only: set[int] = set()
    if len(request.candidates) == 255 and contract["chunking"]["full_K_probe_for_K_255"]:
        # The full-K run is probe-only: recorded, never selectable as the chunk.
        candidates.append(255)
        probe_only.add(255)
    records = []
    max_safe = {"flat": 0, "shared": 0}
    # Conservative per-path memory stop (amended protocol v0.2): after a path's
    # first memory-unsafe chunk, larger chunks for that path are declared
    # ineligible and unexecuted rather than paged through. The operational
    # chunk is the largest actually observed safe chunk below the stop.
    stopped: dict[str, dict[str, Any] | None] = {"flat": None, "shared": None}
    for chunk in candidates:
        row: dict[str, Any] = {"chunk": chunk, "probe_only": chunk in probe_only}
        for path_name in ("flat", "shared"):
            if stopped[path_name] is not None:
                row[path_name] = {
                    "status": "not_executed_after_memory_stop",
                    "safe": False,
                    "stopped_after_chunk": stopped[path_name]["chunk"],
                    "reason": stopped[path_name]["reason"],
                }
                continue
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            window_token = telemetry.open_window() if telemetry is not None else None
            try:
                if path_name == "flat":
                    flat_chunked_score(flat, request, chunk)
                else:
                    shared_chunked_score(shared, request, chunk)
                # Post-forward synchronization stays inside the protected
                # region so asynchronous OOMs attribute to this probe.
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                usage = telemetry.close_window(window_token) if telemetry is not None else None
                decision = chunk_safety(
                    int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None,
                    int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None,
                    physical_bytes,
                    thresholds["peak_allocated_le_fraction_of_physical_vram"],
                    thresholds["peak_reserved_le_fraction_of_physical_vram"],
                    external_shared_usage_bytes=usage["peak_bytes"] if usage else None,
                    external_shared_baseline_bytes=usage["baseline_bytes"] if usage else None,
                )
                if usage is not None:
                    decision["wddm_window_samples"] = usage["samples_in_window"]
                decision["status"] = "safe" if decision["safe"] else "unsafe"
            except (torch.OutOfMemoryError, RuntimeError) as exception:
                # torch.AcceleratorError subclasses RuntimeError on this build.
                # Only specifically classifiable OOMs become memory results;
                # any other error propagates untouched.
                if not is_oom_exception(exception):
                    raise
                decision = {"safe": False, "status": "oom", "spill_observed": None, "stop_cause": "oom"}
            row[path_name] = decision
            if decision["safe"] and chunk not in probe_only:
                max_safe[path_name] = max(max_safe[path_name], chunk)
            if is_memory_stop_decision(decision):
                stopped[path_name] = {"chunk": chunk, "reason": stop_reason(decision)}
        records.append(row)
    return {
        "request_id": request.request_id,
        "max_safe_flat": max_safe["flat"],
        "max_safe_shared": max_safe["shared"],
        "memory_stop": {name: value for name, value in stopped.items() if value is not None},
        "hardware_infeasible": records[0]["flat"]["status"] != "safe" or records[0]["shared"]["status"] != "safe",
        "probe_records": records,
    }


def _verdict_thresholds(contract: dict[str, Any]) -> dict[str, Any]:
    """CLI wrapper: derive A2 thresholds, converting parse failures to exits."""

    try:
        return verdict_thresholds(contract)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _binding_block(policy: AttentionPolicy, physical_bytes: int) -> dict[str, Any]:
    return feasibility_binding(
        execution_git_sha=_git_sha(),
        contract_sha256=sha256_path(CONTRACT_PATH),
        issue11_manifest_sha256=hashlib.sha256(ISSUE11_MANIFEST.read_bytes()).hexdigest(),
        reference_manifest_sha256=sha256_path(RUN1 / "reference_manifest.json"),
        attention_policy=policy.as_record(),
        physical_vram_bytes=physical_bytes,
        driver_version=_driver_version(),
        torch_version=str(torch.__version__),
    )


def _timed(score, backend, request, chunk) -> tuple[float, int, int]:
    """Measurement timing: synchronize and reset peak stats only.

    Allocator cache clearing is deliberately absent — it would strip the warmed
    allocator state that operational latency depends on. Resource probes keep
    their own cache-clearing discipline.
    """

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter_ns()
    score(backend, request, chunk)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000.0
    return (
        elapsed_ms,
        int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
        int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
    )


def measure_cell(
    flat: FlatReferenceBackend,
    shared: SharedContextBackend,
    request: DecisionRequest,
    paths: dict[str, tuple[Any, Any, int]],
    warmups: int,
    repeats: int,
    correctness_tolerance: float,
) -> dict[str, Any]:
    """Measure named paths under the frozen alternation with correctness gates."""

    row: dict[str, Any] = {"request_id": request.request_id, "candidate_count": len(request.candidates)}
    try:
        reference = flat.score([request])[0]
        correctness_available = True
    except torch.OutOfMemoryError:
        # The canonical oracle batch can OOM where chunks fit. Correctness is
        # then unavailable and this cell's performance is inadmissible; the
        # oracle is never silently replaced.
        reference = None
        correctness_available = False
    row["correctness_available"] = correctness_available
    # The unchunked oracle can leave a large allocator/WDDM state behind; clear
    # once here, before warmups, so timed repeats observe a settled allocator
    # without per-repeat cache clearing (which would strip warmup state).
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    if not correctness_available:
        row["performance_admissible"] = False
        return row

    flat_names = sorted(name for name in paths if name.startswith("flat"))
    shared_names = sorted(name for name in paths if name.startswith("shared"))
    # Warmup cadence is exactly the frozen count. The correctness snapshot
    # reuses the final warmup invocation's outputs rather than adding an extra
    # untimed execution after warmup.
    results: dict[str, Any] = {}
    for name, (score, backend, chunk) in paths.items():
        result = None
        for _ in range(warmups):
            result = score(backend, request, chunk)
        results[name] = result

    latencies: dict[str, list[float]] = {name: [] for name in paths}
    peaks: dict[str, dict[str, int]] = {
        name: {"peak_allocated_bytes": 0, "peak_reserved_bytes": 0} for name in paths
    }
    for first, second in repeat_alternation(repeats):
        for group in (first, second):
            for name in flat_names if group == "flat" else shared_names:
                score, backend, chunk = paths[name]
                elapsed_ms, peak_allocated, peak_reserved = _timed(score, backend, request, chunk)
                latencies[name].append(elapsed_ms)
                peaks[name]["peak_allocated_bytes"] = max(peaks[name]["peak_allocated_bytes"], peak_allocated)
                peaks[name]["peak_reserved_bytes"] = max(peaks[name]["peak_reserved_bytes"], peak_reserved)

    for name, (score, backend, chunk) in paths.items():
        metrics = compare_result(reference, results[name])
        stats = latency_percentiles(latencies[name])
        metadata = results[name].metadata
        prefix_tokens = int(metadata["prefix_tokens"])
        candidate_counts = [int(value) for value in metadata["candidate_token_counts"]]
        flat_positions = sum(prefix_tokens + count for count in candidate_counts)
        shared_positions = prefix_tokens + sum(candidate_counts)
        row[f"{name}_correctness"] = {
            "pass": metrics.max_abs_probability_delta < correctness_tolerance and metrics.argmax_equal,
            **{
                key: value
                for key, value in asdict(metrics).items()
                if key in (
                    "max_abs_probability_delta", "mean_abs_probability_delta",
                    "max_abs_logit_delta", "mean_abs_logit_delta",
                    "total_variation", "kl_reference_to_candidate", "argmax_equal",
                )
            },
        }
        row[name] = {
            "chunk": chunk,
            "model_calls": metadata["model_calls"],
            "latency_ms": stats,
            "requests_per_s": 1000.0 / stats["mean"],
            "candidates_per_s": len(request.candidates) * 1000.0 / stats["mean"],
            "peak_memory_across_repeats": peaks[name],
            "tokenizer_counts": {
                "prefix_tokens": prefix_tokens,
                "candidate_token_counts": candidate_counts,
                "total_candidate_tokens": sum(candidate_counts),
            },
            "flat_logical_token_positions": flat_positions,
            "shared_logical_token_positions": shared_positions if name.startswith("shared") else None,
            "logical_reduction_ratio": (flat_positions / shared_positions) if name.startswith("shared") else None,
        }
    if {"flat_chunked", "shared_chunked"} <= set(paths):
        row["pair"] = {
            "speedup_candidates_per_s": row["shared_chunked"]["candidates_per_s"]
            / row["flat_chunked"]["candidates_per_s"]
        }
    row["performance_admissible"] = all(
        entry["pass"] for key, entry in row.items() if key.endswith("_correctness")
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "action", nargs="?", choices=["preflight", "measure"],
        help="GPU action; omitted prints the frozen plan only",
    )
    args = parser.parse_args()

    contract = load_contract()
    cells = [
        (prefix, k)
        for prefix in contract["grids"]["prefix_units"]
        for k in contract["grids"]["candidate_counts"]
    ]
    plan = {
        "protocol_version": contract["protocol_version"],
        "execution_git_sha": _git_sha(),
        "cell_count": len(cells),
        "anchor_cells": [tuple(cell) for cell in contract["anchor_suite"]["cells"]],
        "attribution": contract["attribution"],
        "note": "GPU actions (preflight, measure) require an explicit subcommand and separate authorization",
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.action is None:
        print(json.dumps(plan, indent=2))
        return
    if not torch.cuda.is_available():
        raise SystemExit("GPU action requested but CUDA is unavailable")

    policy = AttentionPolicy(**contract["binding"]["attention_policy"])
    physical_bytes = int(torch.cuda.get_device_properties(0).total_memory)
    binding = _binding_block(policy, physical_bytes)
    flat, shared = load_fp32_pair()
    feasibility_path = OUTPUT_DIR / "feasibility.json"

    if args.action == "preflight":
        ordered_cells = deterministic_cell_order(cells, contract["execution_order"]["cell_order_seed"])
        semantic_ids = [request.request_id for request in semantic_requests()]
        expected_cell_keys = [f"p{cell[0]}-k{cell[1]}" for cell in ordered_cells]
        feasibility: dict[str, Any] = {"binding": binding, "cells": {}, "semantic": {}}
        telemetry = WddmTelemetry()
        telemetry_available = telemetry.start()
        if not telemetry_available:
            print(json.dumps({"wddm_telemetry": "unavailable; spill detection degrades to internal reserved-beyond-physical only"}), flush=True)
        resumed_units = 0
        if PARTIAL_FEASIBILITY_PATH.is_file():
            # Resume an interrupted preflight: the checkpoint's binding must
            # match this execution exactly, otherwise it is stale and discarded.
            try:
                state = resume_partial_feasibility(
                    json.loads(PARTIAL_FEASIBILITY_PATH.read_text(encoding="utf-8")),
                    binding,
                    expected_cell_keys,
                    semantic_ids,
                )
            except ValueError as exc:
                print(json.dumps({"checkpoint_rejected": str(exc), "action": "fresh preflight"}), flush=True)
            else:
                feasibility["cells"] = state["cells"]
                feasibility["semantic"] = state["semantic"]
                resumed_units = state["resumed_units"]
                print(json.dumps({"checkpoint_resumed_units": resumed_units}), flush=True)

        def checkpoint_partial() -> None:
            # Durable checkpoint after every completed unit: atomic replace,
            # explicitly incomplete, so a crash costs at most the current cell
            # while partial evidence can never enter measurement.
            _atomic_write_json(PARTIAL_FEASIBILITY_PATH, {**feasibility, "complete": False})

        with attention_runtime(policy) as session:
            session.observe_model_config(flat.model.config)
            for cell in ordered_cells:
                key = f"p{cell[0]}-k{cell[1]}"
                if key in feasibility["cells"]:
                    continue  # already completed before an interruption; skip in order
                request = grid_request(cell[0], cell[1], contract["grids"]["candidate_units"])
                record = probe_cell(flat, shared, request, contract, physical_bytes, telemetry)
                feasibility["cells"][key] = record
                checkpoint_partial()
                print(
                    json.dumps({k: record[k] for k in ("request_id", "max_safe_flat", "max_safe_shared", "hardware_infeasible")}),
                    flush=True,
                )
            for request in semantic_requests():
                if request.request_id in feasibility["semantic"]:
                    continue  # already completed before an interruption
                record = probe_cell(flat, shared, request, contract, physical_bytes, telemetry)
                feasibility["semantic"][request.request_id] = record
                checkpoint_partial()
        feasibility["complete"] = True
        _atomic_write_json(feasibility_path, feasibility)
        PARTIAL_FEASIBILITY_PATH.unlink(missing_ok=True)
        telemetry.stop()
        provenance = {
            "action": "preflight",
            "binding": binding,
            "attention_session_evidence": session.metadata(),
            "environment": {**environment_metadata(), "nvidia_driver": _driver_version()},
            "physical_vram_bytes": physical_bytes,
            "resumed_units": resumed_units,
            "wddm_telemetry": (
                "continuous per-PID GPU Process Memory Shared Usage via typeperf; spill = in-window peak above pre-window baseline for this process"
                if telemetry_available
                else "unavailable; internal reserved-beyond-physical only"
            ),
        }
        # Phase-specific provenance files preserve both RuntimeSession records;
        # measure never overwrites the preflight evidence.
        (OUTPUT_DIR / "provenance_preflight.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({"output": str(feasibility_path), "cells": len(feasibility["cells"]), "semantic": len(feasibility["semantic"]), "resumed_units": resumed_units}, indent=2))
        return

    # measure: re-verify the preflight binding before trusting any of it.
    if not feasibility_path.is_file():
        raise SystemExit("measure requires a completed preflight (feasibility.json) first")
    feasibility = json.loads(feasibility_path.read_text(encoding="utf-8"))
    try:
        require_complete_feasibility(feasibility)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    mismatches = verify_feasibility_binding(feasibility, binding)
    if mismatches:
        raise SystemExit(f"stale or mismatched preflight evidence: {mismatches}; rerun preflight")

    with attention_runtime(policy) as session:
        session.observe_model_config(flat.model.config)
        infeasible = {
            (int(key[1:].split("-k")[0]), int(key.split("-k")[1]))
            for key, record in feasibility["cells"].items()
            if record["hardware_infeasible"]
        }
        anchors = resolve_anchor_set(plan["anchor_cells"], contract["grids"]["candidate_counts"], infeasible)
        (OUTPUT_DIR / "anchor_resolution.json").write_text(json.dumps(anchors, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        tolerance = contract["correctness_gate"]["max_abs_probability_delta_lt"]
        cadence = contract["cadence"]
        suites: dict[str, list[dict[str, Any]]] = {"primary": [], "attribution": [], "anchor": [], "semantic": []}
        semantic_records = feasibility["semantic"]
        # Fix 1: anchor membership is judged against the FINAL anchor set, so a
        # resource-substituted cell (e.g. (512,64) standing in for (512,128))
        # is treated as the anchor it replaced.
        final_anchors = [tuple(a) for a in anchors["final_anchor_set"]]

        for cell in deterministic_cell_order(cells, contract["execution_order"]["cell_order_seed"]):
            key = f"p{cell[0]}-k{cell[1]}"
            record = feasibility["cells"][key]
            if record["hardware_infeasible"]:
                continue
            request = grid_request(cell[0], cell[1], contract["grids"]["candidate_units"])
            membership = suite_membership(
                cell, plan["anchor_cells"], contract["attribution"]["prefix_units"], contract["attribution"]["candidate_counts"]
            )
            flat_chunk, shared_chunk = record["max_safe_flat"], record["max_safe_shared"]
            is_anchor = tuple(cell) in final_anchors
            # Fix 2: cadences are independent. Every feasible cell gets its
            # primary 3/10 run; final anchors additionally get a separate
            # 10/50 confirmation run — never a replacement of the primary run.
            pair_row = measure_cell(
                flat, shared, request,
                {
                    "flat_chunked": (flat_chunked_score, flat, flat_chunk),
                    "shared_chunked": (shared_chunked_score, shared, shared_chunk),
                },
                cadence["primary"]["warmups"], cadence["primary"]["repeats"], tolerance,
            )
            # Suites metadata reflects the resolved final anchor set, not the
            # original list, so a substituted anchor cell reports "anchor".
            if is_anchor and "anchor" not in membership:
                membership = [*membership, "anchor"]
            pair_row["suites"] = membership
            suites["primary"].append(pair_row)
            if is_anchor:
                anchor_row = measure_cell(
                    flat, shared, request,
                    {
                        "flat_chunked": (flat_chunked_score, flat, flat_chunk),
                        "shared_chunked": (shared_chunked_score, shared, shared_chunk),
                    },
                    cadence["anchor"]["warmups"], cadence["anchor"]["repeats"], tolerance,
                )
                anchor_row["replaces_original"] = tuple(cell) not in [tuple(a) for a in plan["anchor_cells"]]
                suites["anchor"].append(anchor_row)
            if "attribution" in membership:
                common = min(flat_chunk, shared_chunk, cell[1])
                attribution_row = measure_cell(
                    flat, shared, request,
                    {
                        "flat_sequential": (flat_chunked_score, flat, 1),
                        "flat_batched": (flat_chunked_score, flat, common),
                        "shared_sequential": (shared_chunked_score, shared, 1),
                        "shared_batched": (shared_chunked_score, shared, common),
                    },
                    cadence["primary"]["warmups"], cadence["primary"]["repeats"], tolerance,
                )
                attribution_row["common_chunk"] = common
                suites["attribution"].append(attribution_row)

        for request in semantic_requests():
            record = semantic_records[request.request_id]
            if record["hardware_infeasible"]:
                continue
            suites["semantic"].append(
                measure_cell(
                    flat, shared, request,
                    {
                        "flat_chunked": (flat_chunked_score, flat, record["max_safe_flat"]),
                        "shared_chunked": (shared_chunked_score, shared, record["max_safe_shared"]),
                    },
                    cadence["semantic"]["warmups"], cadence["semantic"]["repeats"], tolerance,
                )
            )

        for name, rows in suites.items():
            (OUTPUT_DIR / f"{name}.jsonl").write_text(
                "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8"
            )
        anchor_speedups = [
            row["pair"]["speedup_candidates_per_s"]
            for row in suites["anchor"]
            if row.get("pair") and row.get("performance_admissible")
        ]
        correctness_all_pass = all(
            row.get("performance_admissible") for rows in suites.values() for row in rows
        )
        thresholds = contract["chunking"]["safety_thresholds"]
        allocated_limit = thresholds["peak_allocated_le_fraction_of_physical_vram"] * physical_bytes
        reserved_limit = thresholds["peak_reserved_le_fraction_of_physical_vram"] * physical_bytes
        vram_acceptable = all(
            entry["peak_memory_across_repeats"]["peak_allocated_bytes"] <= allocated_limit
            and entry["peak_memory_across_repeats"]["peak_reserved_bytes"] <= reserved_limit
            for rows in suites.values()
            for row in rows
            for key, entry in row.items()
            if isinstance(entry, dict) and "peak_memory_across_repeats" in entry
        )
        bands = _verdict_thresholds(contract)
        summary = {
            "protocol_version": contract["protocol_version"],
            "anchors": anchors,
            "a2": a2_performance_verdict(
                anchor_speedups, anchors["unique_anchor_count"],
                correctness_all_pass=correctness_all_pass,
                vram_acceptable=vram_acceptable,
                measured_work_evidence=False,  # no independent measured-work proxy exists in this campaign
                strong_threshold=bands["strong"],
                conditional_threshold=bands["conditional"],
                minimum_anchors=bands["minimum_anchors"],
            ) if anchor_speedups else None,
            "infeasible_cells": sorted(f"p{p}-k{k}" for p, k in infeasible),
            "correctness_all_pass": correctness_all_pass,
            "vram_acceptable": vram_acceptable,
            "rows": {name: len(rows) for name, rows in suites.items()},
            "binding": binding,
        }
        (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        provenance = {
            "action": "measure",
            "binding": binding,
            "attention_session_evidence": session.metadata(),
            "environment": {**environment_metadata(), "nvidia_driver": _driver_version()},
            "physical_vram_bytes": physical_bytes,
        }
        (OUTPUT_DIR / "provenance_measure.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
