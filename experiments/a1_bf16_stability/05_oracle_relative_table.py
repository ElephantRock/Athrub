"""Oracle-relative development table for Issue #11 (Issue #9 evidence analysis).

Reads the existing mixed-prefix suite artifact (no GPU execution) and, for all
nine rows, compares A (flat BF16), B (shared BF16), and M (mixed prefix) against
F (flat FP32, the semantic oracle):

  max/mean |dz|, common-mode delta, max/mean centered |dz|,
  max/mean |dp|, TV, KL, argmax_equal,
  FP32 top-1/top-2 probabilities, margin, and near-tie (defined from F).

Derived per row: E_A = d(A, F), E_B = d(B, F), and dE = E_B - E_A for
max |dp|, TV, and KL. Positive dE means sharing adds low-precision error beyond
what flat BF16 already introduces relative to the oracle.

These nine rows are the DEVELOPMENT SET for the Issue #11 criterion: they must
not be turned into thresholds. A fresh holdout suite is required after the
criterion is predeclared.
"""

# ruff: noqa: I001

from __future__ import annotations

import json

from athrub.comparison import compare_result
from athrub.contracts import DecisionResult

from _support import OUT, outputs

SUITE = OUT / "mixed_prefix_suite.json"
NEAR_TIE_THRESHOLD = 2e-3


def to_result(payload: dict) -> DecisionResult:
    return DecisionResult(
        request_id=payload.get("request_id", "row"),
        logits=payload["logits"],
        probabilities=payload["probabilities"],
        predicted_index=payload["predicted_index"],
    )


def pair_metrics(reference: DecisionResult, candidate: DecisionResult) -> dict[str, object]:
    metrics = compare_result(reference, candidate, near_tie_threshold=NEAR_TIE_THRESHOLD)
    return {
        "max_abs_logit_delta": metrics.max_abs_logit_delta,
        "mean_abs_logit_delta": metrics.mean_abs_logit_delta,
        **centered(reference.logits, candidate.logits),
        "max_abs_probability_delta": metrics.max_abs_probability_delta,
        "mean_abs_probability_delta": metrics.mean_abs_probability_delta,
        "total_variation": metrics.total_variation,
        "kl_reference_to_candidate": metrics.kl_reference_to_candidate,
        "argmax_equal": metrics.argmax_equal,
        "fp32_top1_probability": metrics.reference_top1_probability,
        "fp32_top2_probability": metrics.reference_top2_probability,
        "fp32_margin": metrics.reference_probability_margin,
        "fp32_near_tie": metrics.near_tie,
    }


def centered(reference_logits, candidate_logits) -> dict[str, float]:
    deltas = [float(c) - float(r) for r, c in zip(reference_logits, candidate_logits, strict=True)]
    common = sum(deltas) / len(deltas)
    residuals = [d - common for d in deltas]
    return {
        "logit_delta_common_mode": common,
        "max_abs_centered_logit_delta": max(abs(r) for r in residuals),
        "mean_abs_centered_logit_delta": sum(abs(r) for r in residuals) / len(residuals),
    }


def main() -> None:
    suite = json.loads(SUITE.read_text(encoding="utf-8"))
    rows = []
    for row in suite["rows"]:
        payload = row["outputs"]
        f = to_result({**payload["F_flat_fp32"], "request_id": row["request_id"]})
        a = to_result({**payload["A_flat_bf16"], "request_id": row["request_id"]})
        b = to_result({**payload["B_shared_bf16"], "request_id": row["request_id"]})
        m = to_result({**payload["M_mixed"], "request_id": row["request_id"]})

        metrics = {
            "A_vs_F": pair_metrics(f, a),
            "B_vs_F": pair_metrics(f, b),
            "M_vs_F": pair_metrics(f, m),
        }
        derived = {
            "dE_max_abs_probability_delta": metrics["B_vs_F"]["max_abs_probability_delta"]
            - metrics["A_vs_F"]["max_abs_probability_delta"],
            "dE_total_variation": metrics["B_vs_F"]["total_variation"] - metrics["A_vs_F"]["total_variation"],
            "dE_kl": metrics["B_vs_F"]["kl_reference_to_candidate"]
            - metrics["A_vs_F"]["kl_reference_to_candidate"],
        }
        rows.append(
            {
                "suite": row["suite"],
                "request_id": row["request_id"],
                "candidate_count": row["candidate_count"],
                **metrics,
                **derived,
                "outputs": {key: outputs(to_result({**value, "request_id": row["request_id"]})) for key, value in payload.items()},
            }
        )

    summary = {
        "A_vs_F": aggregate(rows, "A_vs_F"),
        "B_vs_F": aggregate(rows, "B_vs_F"),
        "M_vs_F": aggregate(rows, "M_vs_F"),
        "dE_positive_rows": {
            "max_abs_probability_delta": sum(r["dE_max_abs_probability_delta"] > 0 for r in rows),
            "total_variation": sum(r["dE_total_variation"] > 0 for r in rows),
            "kl": sum(r["dE_kl"] > 0 for r in rows),
        },
        "argmax_vs_f": {
            "A": sum(r["A_vs_F"]["argmax_equal"] for r in rows),
            "B": sum(r["B_vs_F"]["argmax_equal"] for r in rows),
            "M": sum(r["M_vs_F"]["argmax_equal"] for r in rows),
        },
    }
    report = {
        "analysis": "oracle-relative development table (issue #11 evidence; issue #9 artifacts)",
        "semantic_oracle": "F = flat FP32",
        "near_tie_threshold_from_F": NEAR_TIE_THRESHOLD,
        "development_set_warning": "these nine rows are the development set; thresholds chosen from them would be post-hoc",
        "rows": rows,
        "summary": summary,
    }
    (OUT / "oracle_relative_dev_table.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"{'request':<28} {'K':>2} | {'dp(A,F)':>9} {'dp(B,F)':>9} {'dp(M,F)':>9} | {'dE_dp':>9} | {'argA':>4} {'argB':>4} {'argM':>4}")
    for row in rows:
        print(
            f"{row['request_id']:<28} {row['candidate_count']:>2} | "
            f"{row['A_vs_F']['max_abs_probability_delta']:>9.3e} {row['B_vs_F']['max_abs_probability_delta']:>9.3e} "
            f"{row['M_vs_F']['max_abs_probability_delta']:>9.3e} | {row['dE_max_abs_probability_delta']:>9.3e} | "
            f"{row['A_vs_F']['argmax_equal']!s:>4} {row['B_vs_F']['argmax_equal']!s:>4} {row['M_vs_F']['argmax_equal']!s:>4}"
        )
    print(json.dumps(summary, indent=2))


def aggregate(rows: list[dict], pair: str) -> dict[str, float | int]:
    return {
        "worst_max_abs_probability_delta": max(r[pair]["max_abs_probability_delta"] for r in rows),
        "worst_total_variation": max(r[pair]["total_variation"] for r in rows),
        "worst_kl": max(r[pair]["kl_reference_to_candidate"] for r in rows),
        "argmax_equal_count": sum(r[pair]["argmax_equal"] for r in rows),
    }


if __name__ == "__main__":
    main()
