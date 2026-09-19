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
    flat_chunked_score,
    geometric_mean,
    latency_percentiles,
    repeat_alternation,
    resolve_anchor_set,
    shared_chunked_score,
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
    strong = a2_performance_verdict([2.5, 2.0, 2.2, 2.1], 4)
    assert strong["band"] == "strong" and strong["verdicts_enabled"] is True
    conditional = a2_performance_verdict([1.6, 1.7, 1.5, 1.9], 4)
    assert conditional["band"] == "conditional"
    bottleneck = a2_performance_verdict([1.1, 1.2, 1.0, 1.3], 4)
    assert bottleneck["band"] == "runtime_bottleneck"
    disabled = a2_performance_verdict([2.5, 2.5, 2.5], 3)
    assert disabled["verdicts_enabled"] is False
    assert disabled["a2_performance_verdict"] == "disabled_fewer_than_four_unique_anchors"
