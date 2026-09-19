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
import random
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
) -> dict[str, Any]:
    """Apply the frozen safety thresholds; reserved beyond physical is spill."""

    if peak_allocated_bytes is None or peak_reserved_bytes is None:
        return {"safe": False, "spill_observed": None, "reason": "no-memory-statistics"}
    spill = peak_reserved_bytes > physical_total_bytes
    within_allocated = peak_allocated_bytes <= allocated_fraction * physical_total_bytes
    within_reserved = peak_reserved_bytes <= reserved_fraction * physical_total_bytes
    return {
        "safe": bool(within_allocated and within_reserved and not spill),
        "spill_observed": bool(spill),
        "peak_allocated_bytes": int(peak_allocated_bytes),
        "peak_reserved_bytes": int(peak_reserved_bytes),
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


def a2_performance_verdict(anchor_speedups: Sequence[float], unique_anchor_count: int) -> dict[str, Any]:
    """Geometric-mean anchor speedup bands; <4 anchors disables both verdicts."""

    if not anchor_speedups:
        raise ValueError("anchor speedups must not be empty")
    mean_speedup = geometric_mean(anchor_speedups)
    if mean_speedup >= 2.0:
        band = "strong"
    elif mean_speedup >= 1.5:
        band = "conditional"
    else:
        band = "runtime_bottleneck"
    enabled = unique_anchor_count >= 4
    return {
        "geometric_mean_candidates_per_s_speedup": mean_speedup,
        "band": band,
        "verdicts_enabled": enabled,
        "a2_performance_verdict": band if enabled else "disabled_fewer_than_four_unique_anchors",
    }
