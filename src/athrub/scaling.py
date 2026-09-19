"""FP32 shared-context scaling harness primitives (Issue #15, protocol v0.1).

Implements the frozen measurement semantics: chunked flat execution (complete
prefix+candidate paths recomputed per chunk), chunked shared execution (prefix
computed once per request and reused across continuation chunks, one softmax
over the full candidate set), the feasibility safety decision, deterministic
cell ordering, repeat alternation, anchor substitution, and A2 verdict
arithmetic. Every threshold, grid, anchor, and decision rule is supplied by the
frozen contract; this module must not change any of them.
"""

from __future__ import annotations

import math
import os
import random
import re
import subprocess
import threading
import time
from collections.abc import Sequence
from typing import Any

import torch

from .contracts import DecisionRequest, DecisionResult
from .reference import FlatReferenceBackend
from .shared_context import (
    SharedContextBackend,
    _from_legacy_cache,
    _to_legacy_cache,
    expand_legacy_cache,
)


def clip_chunk_candidates(candidates: Sequence[int], k: int) -> list[int]:
    """Feasibility chunk candidates clipped to <= K, ascending and deduplicated."""

    if k < 1:
        raise ValueError("k must be positive")
    return sorted({int(chunk) for chunk in candidates if chunk <= k})


def chunk_slices(k: int, chunk: int) -> list[tuple[int, int]]:
    if chunk < 1:
        raise ValueError("chunk must be positive")
    return [(start, min(start + chunk, k)) for start in range(0, k, chunk)]


def _score_flat_chunk(
    backend: FlatReferenceBackend, prefix_ids: list[int], chunk_token_ids: list[list[int]]
) -> list[float]:
    """Score one flat chunk directly: complete prefix+candidate paths, no cache.

    Chunk scoring must not round-trip through DecisionRequest, whose contract
    requires at least two candidates; the frozen sequential mode is chunk = 1.
    """

    paths = [prefix_ids + ids for ids in chunk_token_ids]
    max_length = max(len(path) for path in paths)
    input_ids = torch.full(
        (len(paths), max_length), backend._pad_token_id(), dtype=torch.long, device=backend.device
    )
    mask = torch.zeros_like(input_ids)
    lengths = []
    for row, path in enumerate(paths):
        input_ids[row, : len(path)] = torch.tensor(path, dtype=torch.long, device=backend.device)
        mask[row, : len(path)] = 1
        lengths.append(len(path))
    with torch.inference_mode():
        output = backend.model(input_ids=input_ids, attention_mask=mask, return_dict=True)
        hidden = output.last_hidden_state
        row_indices = torch.arange(hidden.shape[0], device=hidden.device)
        last_indices = torch.tensor(lengths, dtype=torch.long, device=hidden.device) - 1
        scores = backend.head(hidden[row_indices, last_indices]).float()
    return [float(value) for value in scores.cpu()]


def flat_chunked_score(
    backend: FlatReferenceBackend, request: DecisionRequest, chunk: int
) -> DecisionResult:
    """Chunked flat execution: each chunk recomputes complete prefix+candidate paths."""

    prefix_ids = backend._encode_prefix(request)
    candidate_ids = [backend._encode_candidate(c) for c in request.candidates]
    counts = [len(ids) for ids in candidate_ids]
    logits: list[float] = []
    model_calls = 0
    for start, end in chunk_slices(len(candidate_ids), chunk):
        logits.extend(_score_flat_chunk(backend, prefix_ids, candidate_ids[start:end]))
        model_calls += 1
    scores = torch.tensor(logits, dtype=torch.float32)
    probabilities = torch.softmax(scores, dim=0)
    return DecisionResult(
        request_id=request.request_id,
        logits=tuple(float(value) for value in scores),
        probabilities=tuple(float(value) for value in probabilities),
        predicted_index=int(torch.argmax(probabilities).item()),
        metadata={
            "path": "flat-chunked",
            "chunk": chunk,
            "model_calls": model_calls,
            "prefix_tokens": len(prefix_ids),
            "candidate_token_counts": tuple(counts),
            "flat_logical_token_positions": sum(len(prefix_ids) + count for count in counts),
        },
    )


def shared_chunked_score(
    backend: SharedContextBackend, request: DecisionRequest, chunk: int
) -> DecisionResult:
    """Chunked shared execution: one prefix call, per-chunk continuations, one softmax."""

    prefix_ids = backend._encode_prefix(request)
    candidate_ids = [backend._encode_candidate(c) for c in request.candidates]
    counts = [len(ids) for ids in candidate_ids]
    k = len(candidate_ids)

    prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=backend.device)
    logits: list[float] = []
    model_calls = 0
    with torch.inference_mode():
        prefix_output = backend.model(
            input_ids=prefix_tensor,
            attention_mask=torch.ones_like(prefix_tensor),
            use_cache=True,
            return_dict=True,
        )
        raw_cache = prefix_output.past_key_values
        legacy = _to_legacy_cache(raw_cache)
        model_calls += 1

        for start, end in chunk_slices(k, chunk):
            rows = end - start
            branched = expand_legacy_cache(legacy, rows)
            reconstructed = _from_legacy_cache(branched, raw_cache)
            max_suffix = max(counts[start:end])
            suffix_tensor = torch.full(
                (rows, max_suffix), backend._pad_token_id(), dtype=torch.long, device=backend.device
            )
            suffix_mask = torch.zeros_like(suffix_tensor)
            for offset, ids in enumerate(candidate_ids[start:end]):
                suffix_tensor[offset, : len(ids)] = torch.tensor(ids, dtype=torch.long, device=backend.device)
                suffix_mask[offset, : len(ids)] = 1
            full_mask = torch.cat(
                (
                    torch.ones((rows, len(prefix_ids)), dtype=torch.long, device=backend.device),
                    suffix_mask,
                ),
                dim=1,
            )
            position_ids = (
                torch.arange(
                    len(prefix_ids), len(prefix_ids) + max_suffix, dtype=torch.long, device=backend.device
                )
                .unsqueeze(0)
                .expand(rows, -1)
            )
            continuation = backend.model(
                input_ids=suffix_tensor,
                attention_mask=full_mask,
                position_ids=position_ids,
                past_key_values=reconstructed,
                use_cache=False,
                return_dict=True,
            )
            hidden = continuation.last_hidden_state
            row_indices = torch.arange(rows, device=hidden.device)
            last_indices = torch.tensor(counts[start:end], dtype=torch.long, device=hidden.device) - 1
            scores = backend.head(hidden[row_indices, last_indices]).float()
            logits.extend(float(value) for value in scores.cpu())
            model_calls += 1

    all_scores = torch.tensor(logits, dtype=torch.float32)
    probabilities = torch.softmax(all_scores, dim=0)
    return DecisionResult(
        request_id=request.request_id,
        logits=tuple(float(value) for value in all_scores),
        probabilities=tuple(float(value) for value in probabilities),
        predicted_index=int(torch.argmax(probabilities).item()),
        metadata={
            "path": "shared-chunked",
            "chunk": chunk,
            "model_calls": model_calls,
            "prefix_compute_calls": 1,
            "continuation_compute_calls": len(chunk_slices(k, chunk)),
            "prefix_tokens": len(prefix_ids),
            "candidate_token_counts": tuple(counts),
            "shared_logical_token_positions": len(prefix_ids) + sum(counts),
        },
    )


def chunk_safety(
    peak_allocated_bytes: int | None,
    peak_reserved_bytes: int | None,
    physical_total_bytes: int,
    allocated_fraction: float,
    reserved_fraction: float,
    external_shared_usage_bytes: int | None = None,
    external_shared_baseline_bytes: int | None = None,
) -> dict[str, Any]:
    """Apply the frozen safety thresholds with explicit stop causes.

    The WDDM/shared-memory spill condition is the Windows GPU Process Memory
    ``Shared Usage`` measurement (as observed externally), optionally delta'd
    against a pre-probe baseline so desktop paging noise does not trip the
    rule; reserved beyond physical remains a secondary internal signal. The
    returned ``stop_cause`` names the concrete first violated condition:
    allocated_threshold, reserved_threshold, wddm_shared_spill,
    reserved_beyond_physical, or none when safe.
    """

    if peak_allocated_bytes is None or peak_reserved_bytes is None:
        return {"safe": False, "spill_observed": None, "stop_cause": "no_memory_statistics"}
    within_allocated = peak_allocated_bytes <= allocated_fraction * physical_total_bytes
    within_reserved = peak_reserved_bytes <= reserved_fraction * physical_total_bytes
    reserved_beyond_physical = peak_reserved_bytes > physical_total_bytes
    external_spill = False
    if external_shared_usage_bytes is not None:
        baseline = external_shared_baseline_bytes or 0
        external_spill = (external_shared_usage_bytes - baseline) > 0
    if not within_allocated:
        stop_cause = "allocated_threshold"
    elif not within_reserved:
        stop_cause = "reserved_threshold"
    elif external_spill:
        stop_cause = "wddm_shared_spill"
    elif reserved_beyond_physical:
        stop_cause = "reserved_beyond_physical"
    else:
        stop_cause = "none"
    return {
        "safe": bool(within_allocated and within_reserved and not external_spill and not reserved_beyond_physical),
        "spill_observed": bool(external_spill or reserved_beyond_physical),
        "stop_cause": stop_cause,
        "peak_allocated_bytes": int(peak_allocated_bytes),
        "peak_reserved_bytes": int(peak_reserved_bytes),
        "external_shared_usage_bytes": external_shared_usage_bytes,
        "external_shared_baseline_bytes": external_shared_baseline_bytes,
    }


def deterministic_cell_order(cells: Sequence[tuple[int, int]], seed: int) -> list[tuple[int, int]]:
    """Deterministic shuffled cell order from the frozen seed."""

    ordered = list(cells)
    random.Random(seed).shuffle(ordered)
    return ordered


def repeat_alternation(repeats: int) -> list[tuple[str, str]]:
    """Alternate by measured repeat: flat -> shared, then shared -> flat, repeating."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    return [("flat", "shared") if index % 2 == 0 else ("shared", "flat") for index in range(repeats)]


def latency_percentiles(latency_ms: Sequence[float]) -> dict[str, float]:
    values = sorted(float(value) for value in latency_ms)
    if not values:
        raise ValueError("latency samples must not be empty")

    def percentile(fraction: float) -> float:
        position = (len(values) - 1) * fraction
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - lower
        return values[lower] * (1.0 - weight) + values[upper] * weight

    return {
        "mean": sum(values) / len(values),
        "p50": percentile(0.50),
        "p90": percentile(0.90),
        "p95": percentile(0.95),
        "p99": percentile(0.99),
    }


def geometric_mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("values must not be empty")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def resolve_anchor_set(
    anchor_cells: Sequence[tuple[int, int]],
    k_grid: Sequence[int],
    infeasible_cells: set[tuple[int, int]],
) -> dict[str, Any]:
    """Apply the frozen anchor substitution rule.

    If an anchor is resource-infeasible, substitute the largest lower K from
    the frozen K grid at the same prefix, chosen only from the resource
    preflight; never duplicate an already-selected anchor at that prefix (step
    down again); if no unique lower-K substitute exists, the anchor is
    unavailable.
    """

    selected: list[tuple[int, int]] = []
    substitutions: dict[str, dict[str, Any]] = {}
    unavailable: list[tuple[int, int]] = []
    for prefix, k in anchor_cells:
        if (prefix, k) in selected:
            # Already covered by an earlier substitution at this prefix.
            continue
        if (prefix, k) not in infeasible_cells:
            selected.append((prefix, k))
            continue
        chosen: int | None = None
        for candidate_k in sorted((value for value in k_grid if value < k), reverse=True):
            if (prefix, candidate_k) in infeasible_cells:
                continue
            if (prefix, candidate_k) in selected:
                continue
            chosen = candidate_k
            break
        if chosen is None:
            unavailable.append((prefix, k))
            substitutions[f"p{prefix}-k{k}"] = {"status": "unavailable"}
        else:
            selected.append((prefix, chosen))
            substitutions[f"p{prefix}-k{k}"] = {"status": "substituted", "replacement_k": chosen}
    unique_selected = sorted(set(selected))
    return {
        "final_anchor_set": unique_selected,
        "unique_anchor_count": len(unique_selected),
        "substitutions": substitutions,
        "unavailable": sorted(unavailable),
        "verdicts_enabled": len(unique_selected) >= 4,
    }


def suite_membership(
    cell: tuple[int, int],
    anchor_cells: Sequence[tuple[int, int]],
    attribution_prefixes: Sequence[int],
    attribution_candidate_counts: Sequence[int],
) -> list[str]:
    """Independent suite membership: every feasible cell is primary; overlaps allowed."""

    suites = ["primary"]
    if cell[0] in attribution_prefixes and cell[1] in attribution_candidate_counts:
        suites.append("attribution")
    if cell in list(anchor_cells):
        suites.append("anchor")
    return suites


def feasibility_binding(
    execution_git_sha: str | None,
    contract_sha256: str,
    issue11_manifest_sha256: str,
    reference_manifest_sha256: str,
    attention_policy: dict[str, Any],
    physical_vram_bytes: int,
    driver_version: str | None,
    torch_version: str,
) -> dict[str, Any]:
    """Binding block embedded in feasibility evidence and re-verified at measure time."""

    return {
        "execution_git_sha": execution_git_sha,
        "contract_sha256": contract_sha256,
        "issue11_manifest_sha256": issue11_manifest_sha256,
        "reference_manifest_sha256": reference_manifest_sha256,
        "attention_policy": attention_policy,
        "physical_vram_bytes": physical_vram_bytes,
        "driver_version": driver_version,
        "torch_version": torch_version,
    }


def verify_feasibility_binding(record: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Return the list of binding mismatches; empty means the evidence is admissible."""

    mismatches = []
    for key, expected_value in expected.items():
        actual_value = record.get("binding", {}).get(key)
        if actual_value != expected_value:
            mismatches.append(f"{key}: {actual_value!r} != {expected_value!r}")
    return mismatches


def a2_performance_verdict(
    anchor_speedups: Sequence[float],
    unique_anchor_count: int,
    *,
    correctness_all_pass: bool,
    vram_acceptable: bool,
    measured_work_evidence: bool,
    strong_threshold: float,
    conditional_threshold: float,
    minimum_anchors: int,
) -> dict[str, Any]:
    """Geometric-mean anchor speedup bands under the full frozen gate.

    The band is arithmetic; the A2 performance verdict additionally requires
    correctness on every measured cell, acceptable VRAM, at least the minimum
    unique anchors, and — for the conditional band only — independent measured
    work evidence. Thresholds are supplied by the caller from the frozen
    contract; none are defaulted here.
    """

    if not anchor_speedups:
        raise ValueError("anchor speedups must not be empty")
    mean_speedup = geometric_mean(anchor_speedups)
    if mean_speedup >= strong_threshold:
        band = "strong"
    elif mean_speedup >= conditional_threshold:
        band = "conditional"
    else:
        band = "runtime_bottleneck"
    gates = {
        "anchor_minimum_met": unique_anchor_count >= minimum_anchors,
        "correctness_all_pass": correctness_all_pass,
        "vram_acceptable": vram_acceptable,
        "measured_work_evidence": measured_work_evidence,
    }
    enabled = gates["anchor_minimum_met"] and gates["correctness_all_pass"] and gates["vram_acceptable"]
    if not enabled:
        verdict = "disabled_" + "_".join(
            name for name, passed in gates.items() if not passed and name != "measured_work_evidence"
        )
    elif band == "conditional" and not measured_work_evidence:
        verdict = "conditional_pending_measured_work_evidence"
    else:
        verdict = band
    return {
        "geometric_mean_candidates_per_s_speedup": mean_speedup,
        "band": band,
        "gates": gates,
        "a2_performance_verdict": verdict,
    }


def verdict_thresholds(contract: dict[str, Any]) -> dict[str, Any]:
    """Derive the A2 band thresholds and anchor minimum from the frozen contract.

    The contract stores the interpretation as text (">= 2.0x", "1.5x <= speedup
    < 2.0x", "fewer than four unique feasible anchors"). The numbers are parsed
    out of those strings so no code path carries independent constants; any
    contract drift surfaces as a parse failure rather than a silent mismatch.
    """

    bands = contract["a2_interpretation"]["verdict_bands"]
    strong_match = re.fullmatch(r">= ([0-9.]+)x", bands["strong"])
    # The conditional band text carries a trailing qualification clause; match
    # the numeric range at the start of the string.
    conditional_match = re.match(r"([0-9.]+)x <= speedup < ([0-9.]+)x", bands["conditional"])
    minimum_match = re.search(
        r"fewer than ([a-z]+) unique feasible anchors", contract["a2_interpretation"]["anchor_minimum"]
    )
    if strong_match is None or conditional_match is None or minimum_match is None:
        raise ValueError(
            "cannot derive A2 thresholds from the frozen interpretation text: "
            f"{bands['strong']!r} / {bands['conditional']!r} / {contract['a2_interpretation']['anchor_minimum']!r}"
        )
    number_words = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}
    minimum_anchors = number_words.get(minimum_match.group(1))
    if minimum_anchors is None:
        raise ValueError(f"unparseable anchor minimum word: {minimum_match.group(1)!r}")
    derived = {
        "strong": float(strong_match.group(1)),
        "conditional": float(conditional_match.group(1)),
        "minimum_anchors": minimum_anchors,
    }
    if derived["conditional"] >= derived["strong"]:
        raise ValueError(f"conditional band must sit below the strong band: {derived}")
    return derived


_OOM_TEXT_MARKERS = (
    "out of memory",
    "out-of-memory",
    "cudaerrormemoryallocation",
    "insufficient memory",
)


def is_oom_exception(exception: BaseException) -> bool:
    """Classify an exception as out-of-memory, strictly.

    torch.OutOfMemoryError is always OOM. torch.AcceleratorError and
    RuntimeError qualify only when their message is specifically memory
    related; any other AcceleratorError/RuntimeError must propagate rather
    than being silently recorded as a memory result.
    """

    if isinstance(exception, torch.OutOfMemoryError):
        return True
    accelerator_error = getattr(torch, "AcceleratorError", None)
    is_candidate = isinstance(exception, RuntimeError) or (
        accelerator_error is not None and isinstance(exception, accelerator_error)
    )
    if not is_candidate:
        return False
    text = str(exception).lower()
    return any(marker in text for marker in _OOM_TEXT_MARKERS)


def is_memory_stop_decision(decision: dict[str, Any]) -> bool:
    """True when a probe decision ends the (conservative) per-path ladder.

    Unsafe-by-thresholds, spill, and OOM all stop the ladder; explicitly
    recorded not-executed rows and safe rows do not.
    """

    status = decision.get("status")
    if status in {"oom"}:
        return True
    return bool(decision.get("safe")) is False and status in {"unsafe"}


def stop_reason(decision: dict[str, Any]) -> str:
    """Concrete ladder-stop cause: OOM kind or the threshold decision's stop_cause.

    ``no_memory_statistics`` is a concrete diagnosable cause and is reported
    as-is; only the absent/none sentinel falls back to the raw status.
    """

    if decision.get("status") == "oom":
        return "oom"
    cause = decision.get("stop_cause")
    if isinstance(cause, str) and cause not in {"", "none"}:
        return cause
    return str(decision.get("status", "unknown"))


def require_complete_feasibility(record: dict[str, Any]) -> None:
    """Reject any feasibility artifact not marked complete by a full preflight."""

    if record.get("complete") is not True:
        raise ValueError(
            "feasibility artifact is incomplete (complete != true); "
            "partial preflight evidence is categorically inadmissible for measurement"
        )


def resume_partial_feasibility(
    partial: dict[str, Any],
    binding: dict[str, Any],
    expected_cell_keys: list[str],
    expected_semantic_ids: list[str],
) -> dict[str, Any]:
    """Validate a complete:false checkpoint against the current binding for resume.

    Returns a state dict {cells, semantic, resumed_units} carrying forward all
    previously completed units; raises ValueError when the checkpoint's binding
    does not exactly match (stale evidence must not be resumed) or when the
    checkpoint claims completeness (resume is only for interrupted preflights).
    """

    if partial.get("complete") is True:
        raise ValueError("checkpoint is marked complete; resume applies only to interrupted preflights")
    mismatches = verify_feasibility_binding(partial, binding)
    if mismatches:
        raise ValueError(f"checkpoint binding mismatch: {mismatches}")
    cells = partial.get("cells", {})
    semantic = partial.get("semantic", {})
    for key in cells:
        if key not in expected_cell_keys:
            raise ValueError(f"checkpoint carries unknown cell: {key}")
    for request_id in semantic:
        if request_id not in expected_semantic_ids:
            raise ValueError(f"checkpoint carries unknown semantic request: {request_id}")

    # Order-structure validation: completed cells must form an exact prefix of
    # the frozen deterministic cell order (a gap means the checkpoint was not
    # produced by this runner), and semantic rows may only exist once every
    # grid cell is complete, themselves as a prefix of the semantic order.
    completed = set(cells)
    cell_prefix = 0
    for key in expected_cell_keys:
        if key in completed:
            cell_prefix += 1
        else:
            break
    if completed and cell_prefix != len(completed):
        raise ValueError(
            "checkpoint cells are not a prefix of the frozen deterministic order; refusing resume"
        )
    if semantic and cell_prefix != len(expected_cell_keys):
        raise ValueError("checkpoint carries semantic rows before all grid cells completed; refusing resume")
    completed_semantic = set(semantic)
    semantic_prefix = 0
    for request_id in expected_semantic_ids:
        if request_id in completed_semantic:
            semantic_prefix += 1
        else:
            break
    if completed_semantic and semantic_prefix != len(completed_semantic):
        raise ValueError("checkpoint semantic rows are not a prefix of the frozen order; refusing resume")
    return {"cells": cells, "semantic": semantic, "resumed_units": len(cells) + len(semantic)}


def resolve_complete_feasibility_state(record: dict[str, Any]) -> str:
    """Human-readable completeness state for reporting."""

    if record.get("complete") is True:
        return "complete"
    return f"partial ({len(record.get('cells', {}))} cells, {len(record.get('semantic', {}))} semantic)"


class WddmSharedUsageMonitor:
    r"""Continuous per-process WDDM shared-usage telemetry.

    Streams the Windows ``GPU Process Memory(*)\Shared Usage`` counter once
    per preflight via ``typeperf`` and maintains a timestamped series of the
    total shared bytes attributed to *this process's PID only*, so transient
    spills during a probe are captured and unrelated desktop/browser GPU
    activity is never attributed to the harness. Degrades to unavailable when
    typeperf or the counter is missing; callers then fall back to the internal
    reserved-beyond-physical spill signal alone.
    """

    def __init__(self, pid: int | None = None) -> None:
        self._pid_tag = f"pid_{pid if pid is not None else os.getpid()}_"
        self._process: subprocess.Popen[str] | None = None
        self._reader: threading.Thread | None = None
        self._lock = threading.Lock()
        self._series: list[tuple[float, int]] = []
        self.available = False

    @staticmethod
    def parse_typeperf_header(line: str) -> list[str]:
        """Split a typeperf header line into counter instance names."""

        return [column.strip('"').strip() for column in line.rstrip("\r\n").split('","')]

    @staticmethod
    def parse_typeperf_row(line: str) -> tuple[str, list[str]]:
        """Split a typeperf data row into its timestamp and raw value columns."""

        columns = line.rstrip("\r\n").split('","')
        if not columns:
            return "", []
        return columns[0].strip('"'), [column.strip('"') for column in columns[1:]]

    def _pid_shared_bytes(self, instances: Sequence[str], values: Sequence[str]) -> int | None:
        total = 0
        matched = False
        for instance, value in zip(instances, values, strict=False):
            if self._pid_tag not in instance:
                continue
            matched = True
            if value.strip():
                try:
                    total += int(float(value))
                except ValueError:
                    continue
        return total if matched else None

    def _read_stream(self) -> None:
        assert self._process is not None and self._process.stdout is not None
        instances: list[str] = []
        for line in self._process.stdout:
            line = line.strip()
            if not line or line.startswith(("(Exit Code", "The command completed")):
                continue
            if "Error:" in line:
                return
            if not instances:
                header = self.parse_typeperf_header(line)
                if header and header[0] in {"Timestamp", "timestamp"} or "GPU Process Memory" in line:
                    instances = [column for column in header[1:]] if len(header) > 1 else []
                    if not any(self._pid_tag in column for column in instances):
                        instances = []
                continue
            _, values = self.parse_typeperf_row(line)
            shared = self._pid_shared_bytes(instances, values)
            if shared is not None:
                with self._lock:
                    self._series.append((time.monotonic(), shared))
        return

    def start(self) -> bool:
        """Launch the streaming sampler; returns whether telemetry is available."""

        try:
            self._process = subprocess.Popen(
                ["typeperf", r"\GPU Process Memory(*)\Shared Usage", "-si", "1"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except OSError:
            self.available = False
            return False
        self._reader = threading.Thread(target=self._read_stream, daemon=True)
        self._reader.start()
        self.available = True
        return True

    def stop(self) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def open_window(self) -> float | None:
        """Mark a probe window start; returns the monotonic timestamp token."""

        if not self.available:
            return None
        return time.monotonic()

    def close_window(self, start_token: float | None) -> dict[str, Any] | None:
        """Summarize the window: baseline (pre-window) and in-window peak."""

        if not self.available or start_token is None:
            return None
        with self._lock:
            series = list(self._series)
        baseline: int | None = None
        peak: int | None = None
        samples_in_window = 0
        for timestamp, shared in series:
            if timestamp < start_token:
                baseline = shared
            else:
                samples_in_window += 1
                peak = shared if peak is None else max(peak, shared)
        if peak is None:
            return None
        return {
            "peak_bytes": peak,
            "baseline_bytes": baseline if baseline is not None else 0,
            "samples_in_window": samples_in_window,
        }
