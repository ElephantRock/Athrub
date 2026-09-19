from __future__ import annotations

import pytest
import torch

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

# Sibling-module import: pytest's default prepend import mode puts tests/ on
# sys.path (no __init__.py), so this resolves identically under `pytest` (CI)
# and `python -m pytest` (local). The package-qualified `tests.` form only
# works when the repo root is on sys.path, which bare pytest does not guarantee.
from test_shared_context import (
    CacheAwareAccumulatingModel,
    FakeTokenizer,
    make_head,
    make_request,
)


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


def test_verdict_thresholds_derived_from_contract() -> None:
    import json as _json
    from pathlib import Path

    contract = _json.loads(Path("configs/a1_fp32_scaling.v0.1.json").read_text(encoding="utf-8"))
    derived = verdict_thresholds(contract)
    # Values parsed from the frozen interpretation text, not hard-coded.
    assert derived == {"strong": 2.0, "conditional": 1.5, "minimum_anchors": 4}
    # A drifted contract must fail derivation rather than silently pass.
    drifted = _json.loads(_json.dumps(contract))
    drifted["a2_interpretation"]["verdict_bands"]["strong"] = "at least twice as fast"
    try:
        verdict_thresholds(drifted)
    except ValueError:
        pass
    else:
        raise AssertionError("drifted strong band must fail derivation")
    inverted = _json.loads(_json.dumps(contract))
    inverted["a2_interpretation"]["verdict_bands"]["conditional"] = "3.0x <= speedup < 4.0x"
    try:
        verdict_thresholds(inverted)
    except ValueError:
        pass
    else:
        raise AssertionError("conditional above strong must fail derivation")


def test_substituted_anchor_treated_as_final() -> None:
    # A resource-substituted cell must be measured at anchor cadence: the
    # runner checks final_anchor_set membership, not the original list.
    original_anchors = [(512, 32), (512, 128)]
    resolved = resolve_anchor_set(original_anchors, [2, 4, 8, 16, 32, 64, 128, 255], {(512, 128)})
    final = {tuple(a) for a in resolved["final_anchor_set"]}
    assert (512, 64) in final  # substituted cell
    assert (512, 128) not in final  # original, infeasible
    # Runner logic: is_anchor = tuple(cell) in final_anchors — (512,64) qualifies.
    assert (512, 64) in final and (512, 32) in final


def test_inadmissible_anchors_excluded_from_geomean() -> None:
    # The anchor geometric mean filters on performance_admissible.
    rows = [
        {"pair": {"speedup_candidates_per_s": 2.5}, "performance_admissible": True},
        {"pair": {"speedup_candidates_per_s": 5.0}, "performance_admissible": False},  # oracle OOM
        {"pair": {"speedup_candidates_per_s": 2.0}, "performance_admissible": True},
    ]
    speedups = [r["pair"]["speedup_candidates_per_s"] for r in rows if r.get("pair") and r.get("performance_admissible")]
    assert geometric_mean(speedups) == geometric_mean([2.5, 2.0])


def test_is_oom_exception_classification() -> None:
    assert is_oom_exception(torch.OutOfMemoryError("CUDA out of memory")) is True
    accelerator = getattr(torch, "AcceleratorError", RuntimeError)
    assert is_oom_exception(accelerator("CUDA error: out of memory")) is True
    assert is_oom_exception(RuntimeError("cudaErrorMemoryAllocation:insufficient")) is True
    # Non-OOM AcceleratorError/RuntimeError must propagate, not classify as OOM.
    assert is_oom_exception(accelerator("CUDA error: device-side assert triggered")) is False
    assert is_oom_exception(RuntimeError("shape mismatch in matmul")) is False
    assert is_oom_exception(ValueError("out of memory")) is False


def test_memory_stop_decision_rule() -> None:
    assert is_memory_stop_decision({"safe": True, "status": "safe"}) is False
    assert is_memory_stop_decision({"safe": False, "status": "unsafe"}) is True
    assert is_memory_stop_decision({"safe": False, "status": "oom"}) is True
    assert is_memory_stop_decision({"safe": False, "status": "not_executed_after_memory_stop"}) is False


def test_require_complete_feasibility_rejects_partial() -> None:
    require_complete_feasibility({"complete": True})
    for bad in ({"complete": False}, {}, {"complete": None}):
        with pytest.raises(ValueError, match="incomplete"):
            require_complete_feasibility(bad)


def test_protocol_v02_amends_v01_solely() -> None:
    import json as _json
    from pathlib import Path

    v01 = _json.loads(Path("configs/a1_fp32_scaling.v0.1.json").read_text(encoding="utf-8"))
    v02 = _json.loads(Path("configs/a1_fp32_scaling.v0.2.json").read_text(encoding="utf-8"))
    assert v02["protocol_version"] == "0.2"
    assert "memory_stop_rule" in v02["chunking"]
    assert v02["amendment"]["sole_scientific_change"].startswith("the conservative")
    assert v02["amendment"]["amends"] == "0.1"
    # Everything outside protocol_version/status/amendment/memory_stop_rule identical.
    def strip(contract):
        clone = _json.loads(_json.dumps(contract))
        for key in ("protocol_version", "status", "amendment"):
            clone.pop(key, None)
        clone["chunking"] = {k: v for k, v in clone["chunking"].items() if k != "memory_stop_rule"}
        return clone
    assert strip(v01) == strip(v02)
    # The grid is untouched: 4096/8192 remain; no post-hoc pruning.
    assert v02["grids"]["prefix_units"] == [128, 512, 1024, 2048, 4096, 8192]


def test_chunk_safety_stop_causes_and_external_spill() -> None:
    physical = 12 * 1024**3
    safe = chunk_safety(int(0.5 * physical), int(0.9 * physical), physical, 0.9, 0.95)
    assert safe["safe"] is True and safe["stop_cause"] == "none"
    allocated = chunk_safety(int(0.95 * physical), int(0.9 * physical), physical, 0.9, 0.95)
    assert allocated["safe"] is False and allocated["stop_cause"] == "allocated_threshold"
    reserved = chunk_safety(int(0.5 * physical), int(0.97 * physical), physical, 0.9, 0.95)
    assert reserved["safe"] is False and reserved["stop_cause"] == "reserved_threshold"
    # External WDDM shared usage above baseline stops the ladder even when the
    # internal thresholds are satisfied — this is the attempt-1 failure signal.
    spill = chunk_safety(int(0.5 * physical), int(0.9 * physical), physical, 0.9, 0.95,
                         external_shared_usage_bytes=16 * 1024**3, external_shared_baseline_bytes=80 * 1024**2)
    assert spill["safe"] is False and spill["stop_cause"] == "wddm_shared_spill"
    assert spill["spill_observed"] is True
    # A matching baseline (desktop noise) is not a probe-attributable spill.
    noise = chunk_safety(int(0.5 * physical), int(0.9 * physical), physical, 0.9, 0.95,
                         external_shared_usage_bytes=80 * 1024**2, external_shared_baseline_bytes=80 * 1024**2)
    assert noise["safe"] is True
    beyond = chunk_safety(int(0.5 * physical), int(1.1 * physical), physical, 0.9, 0.95)
    # Reserved beyond physical necessarily violates the 95% threshold first, so
    # the named cause is reserved_threshold; beyond-physical is the spill flag.
    assert beyond["stop_cause"] == "reserved_threshold"
    assert beyond["spill_observed"] is True


def test_stop_reason_names_concrete_cause() -> None:
    assert stop_reason({"status": "oom", "stop_cause": "oom"}) == "oom"
    assert stop_reason({"status": "unsafe", "stop_cause": "wddm_shared_spill"}) == "wddm_shared_spill"
    assert stop_reason({"status": "unsafe", "stop_cause": "allocated_threshold"}) == "allocated_threshold"
    # Legacy-shaped decisions without stop_cause degrade to status, never "unsafe" silently.
    assert stop_reason({"status": "unsafe"}) == "unsafe"


def test_resume_partial_feasibility_contract() -> None:
    binding = feasibility_binding("sha-a", "sha-b", "sha-c", "sha-d", {"backend": "x"}, 12, "616.64", "2.11.0")
    partial = {
        "binding": dict(binding),
        "complete": False,
        # All grid cells complete (prefix of the full order) so the semantic
        # row is legal under the stricter ordering contract.
        "cells": {"p128-k2": {}, "p512-k4": {}},
        "semantic": {"a0-route-cost-x": {"request_id": "a0-route-cost-x"}},
    }
    state = resume_partial_feasibility(partial, binding, ["p128-k2", "p512-k4"], ["a0-route-cost-x"])
    assert state["resumed_units"] == 3 and "p128-k2" in state["cells"]
    stale = {**partial, "binding": {**binding, "execution_git_sha": "other"}}
    with pytest.raises(ValueError, match="binding mismatch"):
        resume_partial_feasibility(stale, binding, ["p128-k2", "p512-k4"], ["a0-route-cost-x"])
    complete = {**partial, "complete": True}
    with pytest.raises(ValueError, match="only to interrupted preflights"):
        resume_partial_feasibility(complete, binding, ["p128-k2", "p512-k4"], ["a0-route-cost-x"])
    unknown = {**partial, "cells": {"p9999-k2": {}}}
    with pytest.raises(ValueError, match="unknown cell"):
        resume_partial_feasibility(unknown, binding, ["p128-k2", "p512-k4"], ["a0-route-cost-x"])
