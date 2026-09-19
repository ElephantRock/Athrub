"""Experiment 6: full nine-row controlled mixed-prefix suite (Issue #9).

The same 9 request cases as the A1 correctness matrix (3 semantic individually
+ 6 shape cells), one request per score invocation, comparing A = flat BF16,
B = shared BF16, M = FP32-prefix/BF16-continuation, F = flat FP32. Gate:
max |dp(M, A)| < 1e-3 and argmax(M) == argmax(A) for ALL nine rows, plus the
centered-logit decomposition (softmax-invariant common mode vs residuals).
Evidence: artifacts/a1-bf16-stability/mixed_prefix_suite.json
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
    centered_decomposition,
    load_flat,
    load_head_bf16,
    load_shared,
    mixed_execution,
    outputs,
    verify_frozen_a0_manifest,
)
from athrub.attention_runtime import CANONICAL_A0_POLICY
from athrub.comparison import compare_result
from athrub.provenance import sha256_path
from athrub.workloads import semantic_smoke_requests, synthetic_request

CELLS = [(128, 2), (128, 8), (512, 4), (512, 16), (1024, 2), (1024, 8)]


def frozen_rows(path):
    if not path.is_file():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows.setdefault(row["request_id"], row)
    return rows


def main() -> None:
    verify_frozen_a0_manifest()
    flat32 = load_flat(torch.float32)
    flat16 = load_flat(torch.bfloat16)
    shared16 = load_shared(flat16)
    head16 = load_head_bf16()

    requests = [("semantic", r) for r in semantic_smoke_requests()]
    requests += [
        ("shape", synthetic_request(request_id=f"shape-p{p}-k{k}-c16", prefix_units=p, candidate_units=16, candidate_count=k))
        for p, k in CELLS
    ]

    frozen_semantic = frozen_rows(RUN1 / "canonical-benchmark" / "semantic" / "bf16.jsonl")
    frozen_shape = frozen_rows(RUN1 / "canonical-benchmark" / "shape" / "bf16.jsonl")
    rows = []

    with attention_runtime(CANONICAL_A0_POLICY):
        for suite, request in requests:
            a = flat16.score([request])[0]
            b = shared16.score([request])[0]
            f = flat32.score([request])[0]
            m, transition = mixed_execution(flat32.model, flat16.model, head16, shared16, request)

            frozen = (frozen_semantic if suite == "semantic" else frozen_shape).get(request.request_id)
            if frozen is None:
                frozen_status = "no_frozen_counterpart"
            else:
                exact = max(abs(x - y) for x, y in zip(a.logits, frozen["logits"], strict=True)) == 0.0 and (
                    a.predicted_index == frozen["predicted_index"]
                )
                frozen_status = "exact" if exact else "non_exact_provenance_diagnostic"

            m_vs_a = compare_result(a, m, near_tie_threshold=2e-3)
            rows.append(
                {
                    "suite": suite,
                    "request_id": request.request_id,
                    "candidate_count": len(request.candidates),
                    **asdict(m_vs_a),
                    "b_vs_a_max_abs_probability_delta": compare_result(a, b).max_abs_probability_delta,
                    "m_vs_f_max_abs_probability_delta": compare_result(f, m).max_abs_probability_delta,
                    "m_vs_f_argmax_equal": compare_result(f, m).argmax_equal,
                    **centered_decomposition(a.logits, m.logits),
                    "numerical_pass": m_vs_a.max_abs_probability_delta < 1e-3,
                    "argmax_pass": m_vs_a.argmax_equal,
                    "flat_vs_frozen_canonical": frozen_status,
                    "cache_dtype_transition": {"layers": transition["cast_layers"], "transition": "fp32->bf16"},
                    "token_counts": {
                        "prefix_tokens": m.metadata["prefix_tokens"],
                        "candidate_token_counts": list(m.metadata["candidate_token_counts"]),
                        "flat_positions": m.metadata["flat_logical_token_positions"],
                        "shared_positions": m.metadata["shared_logical_token_positions"],
                        "prefix_compute_calls": m.metadata["prefix_compute_calls"],
                        "continuation_compute_calls": m.metadata["continuation_compute_calls"],
                    },
                    "outputs": {
                        "A_flat_bf16": outputs(a),
                        "B_shared_bf16": outputs(b),
                        "M_mixed": outputs(m),
                        "F_flat_fp32": outputs(f),
                    },
                }
            )

    all_numerical = all(row["numerical_pass"] for row in rows)
    all_argmax = all(row["argmax_pass"] for row in rows)
    report = {
        "experiment": "full controlled mixed-prefix suite (issue #9)",
        "attention_policy": CANONICAL_A0_POLICY.as_record(),
        "frozen_a0_verification": "passed",
        "reference_manifest_sha256": sha256_path(RUN1 / "reference_manifest.json"),
        "rows": rows,
        "gate": {
            "all_nine_max_abs_probability_delta_lt_1e-3": all_numerical,
            "all_nine_argmax_equal": all_argmax,
            "mixed_precision_operational_candidate": "PASS" if (all_numerical and all_argmax) else "FAIL",
            "failing_rows": [
                {
                    "request_id": row["request_id"],
                    "suite": row["suite"],
                    "candidate_count": row["candidate_count"],
                    "max_abs_probability_delta": row["max_abs_probability_delta"],
                    "argmax_equal": row["argmax_pass"],
                    "logit_delta_common_mode": row["logit_delta_common_mode"],
                    "max_abs_centered_logit_delta": row["max_abs_centered_logit_delta"],
                }
                for row in rows
                if not (row["numerical_pass"] and row["argmax_pass"])
            ],
        },
    }
    (OUT / "mixed_prefix_suite.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["gate"], indent=2))


if __name__ == "__main__":
    main()
