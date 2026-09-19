from __future__ import annotations

import pytest
import torch
from tests.test_shared_context import (
    CacheAwareAccumulatingModel,
    FakeTokenizer,
    make_head,
    make_request,
)

from athrub.reference import FlatReferenceBackend
from athrub.scaling import (
    a2_performance_verdict,
    chunk_safety,
    chunk_slices,
    clip_chunk_candidates,
    deterministic_cell_order,
    feasibility_binding,
    flat_chunked_score,
    geometric_mean,
    latency_percentiles,
    repeat_alternation,
    resolve_anchor_set,
    shared_chunked_score,
    suite_membership,
    verify_feasibility_binding,
)
from athrub.shared_context import SharedContextBackend


def make_flat() -> FlatReferenceBackend:
    return FlatReferenceBackend(
        model=CacheAwareAccumulatingModel(),
        tokenizer=FakeTokenizer(),
        head=make_head(),
        device="cpu",
        dtype=torch.float32,
        substrate_revision="test",
        tokenizer_revision="test",
    )


def make_shared() -> SharedContextBackend:
    return SharedContextBackend(
        model=CacheAwareAccumulatingModel(),
        tokenizer=FakeTokenizer(),
        head=make_head(),
        device="cpu",
        dtype=torch.float32,
        substrate_revision="test",
        tokenizer_revision="test",
    )


def test_chunk_candidates_clip_to_k() -> None:
    assert clip_chunk_candidates([1, 2, 4, 8, 16, 32, 64, 128], 8) == [1, 2, 4, 8]
    assert clip_chunk_candidates([1, 2, 4, 8, 16, 32, 64, 128], 255) == [1, 2, 4, 8, 16, 32, 64, 128]
    assert clip_chunk_candidates([1, 2, 4], 1) == [1]


def test_chunk_slices_cover_all_candidates() -> None:
    assert chunk_slices(8, 3) == [(0, 3), (3, 6), (6, 8)]
    assert chunk_slices(4, 4) == [(0, 4)]
    assert chunk_slices(4, 1) == [(0, 1), (1, 2), (2, 3), (3, 4)]
    with pytest.raises(ValueError):
        chunk_slices(4, 0)


@pytest.mark.parametrize("chunk", [1, 2, 3, 4])
def test_flat_chunked_matches_flat_reference(chunk: int) -> None:
    flat = make_flat()
    request = make_request(candidate_count=4)
    reference = flat.score([request])[0]
    chunked = flat_chunked_score(flat, request, chunk)
    assert chunked.metadata["model_calls"] == len(chunk_slices(4, chunk))
    assert max(abs(a - b) for a, b in zip(reference.logits, chunked.logits, strict=True)) == 0.0
    assert chunked.predicted_index == reference.predicted_index


@pytest.mark.parametrize("chunk", [1, 2, 4])
def test_shared_chunked_matches_shared_and_flat(chunk: int) -> None:
    flat = make_flat()
    shared = make_shared()
    request = make_request(candidate_count=4)
    reference = flat.score([request])[0]
    shared_reference = shared.score([request])[0]
    chunked = shared_chunked_score(shared, request, chunk)
    assert chunked.metadata["prefix_compute_calls"] == 1
    assert chunked.metadata["continuation_compute_calls"] == len(chunk_slices(4, chunk))
    assert chunked.metadata["model_calls"] == 1 + len(chunk_slices(4, chunk))
    assert max(abs(a - b) for a, b in zip(reference.logits, chunked.logits, strict=True)) < 1e-6
    assert max(abs(a - b) for a, b in zip(shared_reference.logits, chunked.logits, strict=True)) == 0.0


def test_chunk_safety_thresholds_and_spill() -> None:
    physical = 12 * 1024**3
    safe = chunk_safety(int(0.89 * physical), int(0.94 * physical), physical, 0.9, 0.95)
    assert safe["safe"] is True and safe["spill_observed"] is False
    over_allocated = chunk_safety(int(0.91 * physical), int(0.94 * physical), physical, 0.9, 0.95)
    assert over_allocated["safe"] is False
    over_reserved = chunk_safety(int(0.5 * physical), int(0.96 * physical), physical, 0.9, 0.95)
    assert over_reserved["safe"] is False
    spill = chunk_safety(int(0.5 * physical), int(1.1 * physical), physical, 0.9, 0.95)
    assert spill["safe"] is False and spill["spill_observed"] is True
    missing = chunk_safety(None, None, physical, 0.9, 0.95)
    assert missing["safe"] is False


def test_cell_order_deterministic() -> None:
    cells = [(p, k) for p in (128, 512, 1024) for k in (2, 8, 32)]
    first = deterministic_cell_order(cells, 271828)
    second = deterministic_cell_order(cells, 271828)
    assert first == second
    assert sorted(first) == sorted(cells)
    assert first != cells  # the seed genuinely shuffles this grid


def test_repeat_alternation_pattern() -> None:
    assert repeat_alternation(4) == [
        ("flat", "shared"),
        ("shared", "flat"),
        ("flat", "shared"),
        ("shared", "flat"),
    ]


def test_latency_percentiles_and_geometric_mean() -> None:
    stats = latency_percentiles([1.0, 2.0, 3.0, 4.0])
    assert stats["mean"] == 2.5
    assert stats["p50"] == 2.5
    assert stats["p99"] == pytest.approx(3.97)
    assert geometric_mean([2.0, 8.0]) == pytest.approx(4.0)


def test_anchor_substitution_rule() -> None:
    anchors = [(512, 32), (512, 128), (1024, 16), (1024, 64), (2048, 16), (2048, 64)]
    k_grid = [2, 4, 8, 16, 32, 64, 128, 255]
    resolved = resolve_anchor_set(anchors, k_grid, {(512, 128)})
    assert (512, 32) in resolved["final_anchor_set"]
    # Largest lower K at the same prefix that is not already selected: 64.
    assert (512, 64) in resolved["final_anchor_set"]
    assert resolved["substitutions"]["p512-k128"] == {"status": "substituted", "replacement_k": 64}
    assert resolved["verdicts_enabled"] is True

    duplicate_pressure = resolve_anchor_set(anchors, k_grid, {(512, 128), (512, 64), (512, 32)})
    # (512,32) substitutes down to 16; (512,128) then steps past 64 and 32
    # (infeasible) and 16 (already selected) to 8, without duplication.
    assert (512, 16) in duplicate_pressure["final_anchor_set"]
    assert (512, 8) in duplicate_pressure["final_anchor_set"]
    assert duplicate_pressure["substitutions"]["p512-k32"] == {"status": "substituted", "replacement_k": 16}
    assert duplicate_pressure["substitutions"]["p512-k128"] == {"status": "substituted", "replacement_k": 8}

    exhausted = resolve_anchor_set([(512, 2)], [2, 4], {(512, 2)})
    assert exhausted["final_anchor_set"] == []
    assert exhausted["substitutions"]["p512-k2"]["status"] == "unavailable"
    assert exhausted["verdicts_enabled"] is False


def test_a2_verdict_bands_and_anchor_minimum() -> None:
    strong = a2_performance_verdict(
        [2.5, 2.0, 2.2, 2.1], 4,
        correctness_all_pass=True, vram_acceptable=True, measured_work_evidence=False,
        strong_threshold=2.0, conditional_threshold=1.5, minimum_anchors=4,
    )
    assert strong["band"] == "strong"
    assert strong["a2_performance_verdict"] == "strong"
    conditional = a2_performance_verdict(
        [1.6, 1.7, 1.5, 1.9], 4,
        correctness_all_pass=True, vram_acceptable=True, measured_work_evidence=True,
        strong_threshold=2.0, conditional_threshold=1.5, minimum_anchors=4,
    )
    assert conditional["band"] == "conditional"
    assert conditional["a2_performance_verdict"] == "conditional"
    bottleneck = a2_performance_verdict(
        [1.1, 1.2, 1.0, 1.3], 4,
        correctness_all_pass=True, vram_acceptable=True, measured_work_evidence=False,
        strong_threshold=2.0, conditional_threshold=1.5, minimum_anchors=4,
    )
    assert bottleneck["band"] == "runtime_bottleneck"
    assert bottleneck["a2_performance_verdict"] == "runtime_bottleneck"


def test_a2_verdict_gates() -> None:
    common = {
        "correctness_all_pass": True, "vram_acceptable": True, "measured_work_evidence": False,
        "strong_threshold": 2.0, "conditional_threshold": 1.5, "minimum_anchors": 4,
    }
    disabled = a2_performance_verdict([2.5, 2.5, 2.5], 3, **common)
    assert disabled["a2_performance_verdict"].startswith("disabled_")
    assert disabled["gates"]["anchor_minimum_met"] is False

    correctness_failed = a2_performance_verdict([2.5] * 4, 4, **{**common, "correctness_all_pass": False})
    assert correctness_failed["band"] == "strong"
    assert correctness_failed["a2_performance_verdict"].startswith("disabled_")

    vram_failed = a2_performance_verdict([2.5] * 4, 4, **{**common, "vram_acceptable": False})
    assert vram_failed["a2_performance_verdict"].startswith("disabled_")

    pending = a2_performance_verdict([1.6] * 4, 4, **{**common, "measured_work_evidence": False})
    assert pending["band"] == "conditional"
    assert pending["a2_performance_verdict"] == "conditional_pending_measured_work_evidence"


def test_suite_membership_overlap_is_independent() -> None:
    anchors = [(512, 32), (512, 128), (1024, 16), (1024, 64), (2048, 16), (2048, 64)]
    attribution_prefixes = [128, 512, 1024]
    attribution_ks = [2, 8, 32]
    # (512, 32) is both an attribution cell and an anchor, and is always primary.
    assert suite_membership((512, 32), anchors, attribution_prefixes, attribution_ks) == [
        "primary", "attribution", "anchor",
    ]
    # Every feasible grid cell participates in the primary comparison.
    assert suite_membership((8192, 255), anchors, attribution_prefixes, attribution_ks) == ["primary"]
    assert suite_membership((128, 8), anchors, attribution_prefixes, attribution_ks) == ["primary", "attribution"]


def test_probe_only_full_k_never_selected() -> None:
    # Mirror of the runner's probe bookkeeping: a safe full-K 255 run is
    # recorded but excluded from max_safe because it is probe-only.
    probe_only = {255}
    safe_chunks = [1, 2, 4, 8, 16, 32, 64, 128, 255]
    max_safe = max(chunk for chunk in safe_chunks if chunk not in probe_only)
    assert max_safe == 128
    assert 255 in safe_chunks  # recorded in probe_records nonetheless


def test_stale_preflight_rejected_by_binding() -> None:
    expected = feasibility_binding(
        execution_git_sha="aaaa",
        contract_sha256="bbbb",
        issue11_manifest_sha256="cccc",
        reference_manifest_sha256="dddd",
        attention_policy={"backend": "efficient_sdpa"},
        physical_vram_bytes=12,
        driver_version="616.64",
        torch_version="2.11.0",
    )
    fresh = {"binding": dict(expected)}
    assert verify_feasibility_binding(fresh, expected) == []
    stale = {"binding": {**expected, "execution_git_sha": "zzzz"}}
    mismatches = verify_feasibility_binding(stale, expected)
    assert any("execution_git_sha" in mismatch for mismatch in mismatches)
    missing = {"binding": {}}
    assert len(verify_feasibility_binding(missing, expected)) == len(expected)


def test_measure_row_contains_required_metric_fields() -> None:
    # The measured-row schema must carry every contract-required field; this
    # fixes the orchestrator against silently dropping metrics again.
    required_latency = {"mean", "p50", "p90", "p95", "p99"}
    required_path_fields = {
        "chunk", "model_calls", "latency_ms", "requests_per_s", "candidates_per_s",
        "peak_memory_across_repeats", "tokenizer_counts",
        "flat_logical_token_positions", "shared_logical_token_positions",
        "logical_reduction_ratio",
    }
    required_correctness = {
        "pass", "max_abs_probability_delta", "mean_abs_probability_delta",
        "max_abs_logit_delta", "mean_abs_logit_delta",
        "total_variation", "kl_reference_to_candidate", "argmax_equal",
    }
    required_memory = {"peak_allocated_bytes", "peak_reserved_bytes"}
    # Schema check against a synthetic row shaped like measure_cell output.
    synthetic = {
        "chunk": 8, "model_calls": 2,
        "latency_ms": {key: 1.0 for key in required_latency},
        "requests_per_s": 1.0, "candidates_per_s": 8.0,
        "peak_memory_across_repeats": {key: 1 for key in required_memory},
        "tokenizer_counts": {"prefix_tokens": 10, "candidate_token_counts": [2, 2], "total_candidate_tokens": 4},
        "flat_logical_token_positions": 24, "shared_logical_token_positions": 14,
        "logical_reduction_ratio": 24 / 14,
    }
    assert required_path_fields <= set(synthetic)
    assert required_latency <= set(synthetic["latency_ms"])
    assert required_memory <= set(synthetic["peak_memory_across_repeats"])
    assert required_correctness >= {"pass"}
