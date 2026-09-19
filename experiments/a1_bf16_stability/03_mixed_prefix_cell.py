"""Experiment 4-5: mixed-precision prefix diagnostics (Issue #9).

03 runs the single p128 x k2 mixed cell (M vs A/B/F). 04 runs the full
nine-row controlled mixed-prefix suite. M = FP32 prefix (frozen BF16-valued
weights promoted; no higher-precision weights exist on disk) -> BF16-cast KV
cache -> view branch -> cache-family reconstruction -> BF16 continuation/head.
Evidence: artifacts/a1-bf16-stability/{mixed_prefix_p128k2,mixed_prefix_suite}.json
"""

# ruff: noqa: I001

from __future__ import annotations

import json
from dataclasses import asdict

import torch

from _support import (
    OUT,
    RUN1,
    attention_runtime,
    load_flat,
    load_head_bf16,
    load_shared,
    mixed_execution,
    outputs,
    verify_frozen_a0_manifest,
)
from athrub.attention_runtime import CANONICAL_A0_POLICY
from athrub.comparison import compare_result
from athrub.workloads import synthetic_request

FROZEN = RUN1 / "canonical-benchmark" / "shape" / "bf16.jsonl"


def main() -> None:
    verify_frozen_a0_manifest()
    flat32 = load_flat(torch.float32)
    flat16 = load_flat(torch.bfloat16)
    shared16 = load_shared(flat16)
    head16 = load_head_bf16()
    request = synthetic_request(
        request_id="shape-p128-k2-c16", prefix_units=128, candidate_units=16, candidate_count=2
    )

    frozen = {}
    for line in FROZEN.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        frozen.setdefault(row["request_id"], row)

    with attention_runtime(CANONICAL_A0_POLICY):
        a = flat16.score([request])[0]
        b = shared16.score([request])[0]
        f = flat32.score([request])[0]
        m, transition = mixed_execution(flat32.model, flat16.model, head16, shared16, request)

    frozen_row = frozen[request.request_id]
    a_matches_frozen = (
        max(abs(x - y) for x, y in zip(a.logits, frozen_row["logits"], strict=True)) == 0.0
        and a.predicted_index == frozen_row["predicted_index"]
    )
    if not a_matches_frozen:
        raise SystemExit("ABORT: flat BF16 does not match the frozen canonical record")

    threshold = 2e-3
    comparisons = {
        "M_vs_A": compare_result(a, m, near_tie_threshold=threshold),
        "M_vs_F": compare_result(f, m, near_tie_threshold=threshold),
        "M_vs_B": compare_result(b, m, near_tie_threshold=threshold),
    }
    primary = comparisons["M_vs_A"]
    report = {
        "experiment": "diagnostic mixed-precision prefix, bf16 p128 k2",
        "attention_policy": CANONICAL_A0_POLICY.as_record(),
        "weights_note": "frozen BF16-valued weights promoted to FP32 for the prefix; no higher-precision weights exist on disk",
        "cache_dtype_transition_layers": transition["cast_layers"],
        "a_matches_frozen_canonical": a_matches_frozen,
        "outputs": {"A_flat_bf16": outputs(a), "B_shared_bf16": outputs(b), "M_mixed": outputs(m), "F_flat_fp32": outputs(f)},
        "comparisons": {name: asdict(metrics) for name, metrics in comparisons.items()},
        "diagnostic_targets": {
            "max_abs_probability_delta_lt_1e-3": primary.max_abs_probability_delta < 1e-3,
            "argmax_match": primary.argmax_equal,
        },
        "mixed_precision_operational_candidate": "PASS"
        if (primary.max_abs_probability_delta < 1e-3 and primary.argmax_equal)
        else "FAIL",
    }
    (OUT / "mixed_prefix_p128k2.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("a_matches_frozen_canonical", "diagnostic_targets", "mixed_precision_operational_candidate")}, indent=2))


if __name__ == "__main__":
    main()
