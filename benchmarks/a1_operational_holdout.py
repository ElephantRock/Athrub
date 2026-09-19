"""Issue #11 scored holdout: BF16 operational equivalence against the FP32 oracle.

Executes the frozen 42-row holdout manifest under the frozen contract v0.2.

Phase 0 (resource gate): load the FP32 pair only, re-measure the largest cell
(shape-p384-k12-c24) on F and G, discard outputs unread, then release.
Phase 1 (architecture control): score F and G on every row in frozen order, one
request per invocation; any violation of max |dp| < 1e-5 or exact argmax stops
BF16 adjudication.
Phase 2 (adjudication): load the BF16 pair; score A (diagnostic only) and B;
evaluate B against the stored FP32 oracle exactly per contract v0.2 (per-row
max |dp| <= 1e-2, TV <= 2e-2, 2e-2 oracle ambiguity radius decision rule, no
failure budget).
"""

# Executable src-layout imports are intentionally grouped by dependency boundary.

from __future__ import annotations

import argparse
import gc
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
from athrub.shared_context import SharedContextBackend

RUN1 = Path("artifacts/a0-reference-v0.1-run1")
CONTRACT_PATH = Path("configs/a1_operational_equivalence.contract.v0.2.json")
MANIFEST_PATH = Path("experiments/a1_operational_equivalence/holdout_manifest.v0.2.jsonl")
OUTPUT_DIR = Path("artifacts/a1-operational-holdout-v0.2")
REMEASURE_REQUEST_ID = "shape-p384-k12-c24"


def _driver_version() -> str | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        )
        return output.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def _load_pair(dtype: torch.dtype) -> tuple[FlatReferenceBackend, SharedContextBackend]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(RUN1 / "tokenizer"))
    model = AutoModel.from_pretrained(str(RUN1 / "substrate"), dtype=dtype)
    head = ScalarDecisionHead(int(model.config.hidden_size), bias=True)
    head.load_state_dict(
        torch.load(str(RUN1 / "decision_head.pt"), map_location="cpu", weights_only=True), strict=True
    )
    flat = FlatReferenceBackend(
        model=model,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        head=head,
        device="cuda",
        dtype=dtype,
        substrate_revision="holdout",
        tokenizer_revision="holdout",
    )
    shared = SharedContextBackend(
        model=flat.model,
        tokenizer=flat.tokenizer,  # type: ignore[arg-type]
        head=flat.head,
        device=flat.device,
        dtype=flat.dtype,
        codec=flat.codec,
        add_bos=flat.add_bos,
        substrate_revision="holdout",
        tokenizer_revision="holdout",
    )
    return flat, shared


def _score_one(backend, request: DecisionRequest):
    return backend.score([request])[0]


def _timed_score_one(backend, request: DecisionRequest):
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter_ns()
    result = _score_one(backend, request)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, (time.perf_counter_ns() - start) / 1_000_000.0


def _outputs(result) -> dict[str, Any]:
    return {
        "logits": list(result.logits),
        "probabilities": list(result.probabilities),
        "predicted_index": result.predicted_index,
    }


def _centered(reference_logits, candidate_logits) -> dict[str, float]:
    deltas = [float(c) - float(r) for r, c in zip(reference_logits, candidate_logits, strict=True)]
    common = sum(deltas) / len(deltas)
    residuals = [d - common for d in deltas]
    return {
        "logit_delta_common_mode": common,
        "max_abs_centered_logit_delta": max(abs(r) for r in residuals),
        "mean_abs_centered_logit_delta": sum(abs(r) for r in residuals) / len(residuals),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", default=str(CONTRACT_PATH))
    args = parser.parse_args()

    contract = json.loads(Path(args.contract).read_text(encoding="utf-8"))
    manifest_bytes = MANIFEST_PATH.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    if manifest_sha != contract["holdout"]["manifest_sha256"]:
        raise SystemExit("raw-byte manifest sha does not match the frozen contract; refusing to execute")

    if sha256_path(RUN1 / "reference_manifest.json") != contract["binding"]["reference_manifest_sha256"]:
        raise SystemExit("reference manifest sha does not match the contract binding; refusing to execute")
    frozen_manifest = json.loads((RUN1 / "reference_manifest.json").read_text(encoding="utf-8"))
    for name, actual, recorded in (
        ("substrate", sha256_path(RUN1 / "substrate"), frozen_manifest["substrate"]["artifact_sha256"]),
        ("tokenizer", sha256_path(RUN1 / "tokenizer"), frozen_manifest["tokenizer"]["artifact_sha256"]),
        ("decision_head", sha256_path(RUN1 / "decision_head.pt"), frozen_manifest["decision_head"]["artifact_sha256"]),
    ):
        if actual != recorded:
            raise SystemExit(f"frozen A0 binding failed for {name}; refusing to execute")

    requests = [
        DecisionRequest(
            request_id=payload["request_id"],
            state=payload["state"],
            question=payload["question"],
            candidates=payload["candidates"],
            metadata=payload["metadata"],
        )
        for payload in (json.loads(line) for line in manifest_bytes.decode("utf-8").splitlines())
    ]
    if len(requests) != 42:
        raise SystemExit(f"expected 42 holdout rows, found {len(requests)}")

    policy = AttentionPolicy(**contract["binding"]["attention_policy"])
    radius = float(contract["oracle_ambiguity_radius"])
    prob_gate = float(contract["bf16_gates_per_row"]["max_abs_probability_delta_le"])
    tv_gate = float(contract["bf16_gates_per_row"]["total_variation_le"])
    fp32_tol = float(contract["fp32_precondition"]["max_abs_probability_delta_lt"])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        execution_git_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10, check=True
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        execution_git_sha = None
    provenance = {
        "contract_version": contract["contract_version"],
        "execution_git_sha": execution_git_sha,
        "manifest_sha256_raw": manifest_sha,
        "reference_manifest_sha256": sha256_path(RUN1 / "reference_manifest.json"),
        "attention_policy": policy.as_record(),
        "environment": {**environment_metadata(), "nvidia_driver": _driver_version()},
        "request_grouping": "one DecisionRequest per score invocation, frozen manifest order",
    }
    (OUTPUT_DIR / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    with attention_runtime(policy):
        # ---------------- phase 0: pair-loaded largest-cell resource gate ----------------
        flat32, shared32 = _load_pair(torch.float32)
        remeasure_request = next(r for r in requests if r.request_id == REMEASURE_REQUEST_ID)
        remeasure = {}
        for name, backend in (("F_flat_fp32", flat32), ("G_shared_fp32", shared32)):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            start = time.perf_counter_ns()
            try:
                result = _score_one(backend, remeasure_request)
                elapsed = (time.perf_counter_ns() - start) / 1_000_000.0
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                del result  # resource gate: outputs discarded unread
                remeasure[name] = {
                    "status": "ok",
                    "latency_ms": elapsed,
                    "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
                    "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
                }
            except torch.OutOfMemoryError:
                remeasure[name] = {"status": "oom"}
        (OUTPUT_DIR / "phase0_remeasure.json").write_text(json.dumps(remeasure, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if any(row["status"] != "ok" for row in remeasure.values()):
            raise SystemExit(f"phase 0 failed: {remeasure}")
        print(f"phase0 remeasure (F+G only, outputs unread): {json.dumps(remeasure)}", flush=True)

        # ---------------- phase 1: FP32 architecture control ----------------
        f_outputs: dict[str, dict[str, Any]] = {}
        precondition_rows = []
        for request in requests:
            f_result = _score_one(flat32, request)
            g_result = _score_one(shared32, request)
            f_outputs[request.request_id] = _outputs(f_result)
            metrics = compare_result(f_result, g_result)
            precondition_rows.append(
                {
                    "request_id": request.request_id,
                    **asdict(metrics),
                    "precondition_pass": metrics.max_abs_probability_delta < fp32_tol and metrics.argmax_equal,
                }
            )
        precondition_failures = [row for row in precondition_rows if not row["precondition_pass"]]
        (OUTPUT_DIR / "fp32_precondition.jsonl").write_text(
            "\n".join(json.dumps(row, sort_keys=True) for row in precondition_rows) + "\n", encoding="utf-8"
        )
        print(f"phase1 FP32 precondition: {42 - len(precondition_failures)}/42 pass", flush=True)
        if precondition_failures:
            report = {
                "fp32_precondition_pass": False,
                "bf16_adjudication_executed": False,
                "failures": precondition_failures,
            }
            (OUTPUT_DIR / "summary.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            raise SystemExit("FP32 architecture control failed; BF16 adjudication stopped per contract")
        # The phase-0 remeasure loop leaves `backend` (and `name`) bound to the
        # shared FP32 backend; those references keep the model alive just like
        # the original helper bug did, so they are deleted alongside the pairs.
        del flat32, shared32, backend, name
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        post_fp32_release_memory = {
            "cuda_memory_allocated_bytes": int(torch.cuda.memory_allocated()),
            "cuda_memory_reserved_bytes": int(torch.cuda.memory_reserved()),
        }
        print(f"phase transition (FP32 pair released): {post_fp32_release_memory}", flush=True)

        # ---------------- phase 2: BF16 adjudication ----------------
        flat16, shared16 = _load_pair(torch.bfloat16)
        rows = []
        for request in requests:
            f_payload = f_outputs[request.request_id]
            a_result = _score_one(flat16, request)
            b_result = _score_one(shared16, request)

            from athrub.contracts import DecisionResult

            f_result = DecisionResult(
                request_id=request.request_id,
                logits=f_payload["logits"],
                probabilities=f_payload["probabilities"],
                predicted_index=f_payload["predicted_index"],
            )
            f_prob = f_result.probabilities
            f_top1 = max(range(len(f_prob)), key=f_prob.__getitem__)
            ambiguity_set = {i for i, p in enumerate(f_prob) if p >= f_prob[f_top1] - radius}
            inside = len(ambiguity_set) > 1

            b_metrics = compare_result(f_result, b_result, near_tie_threshold=radius)
            probability_pass = b_metrics.max_abs_probability_delta <= prob_gate
            tv_pass = b_metrics.total_variation <= tv_gate
            if inside:
                decision_pass = b_result.predicted_index in ambiguity_set
            else:
                decision_pass = b_result.predicted_index == f_result.predicted_index
            regret = float(f_prob[f_top1] - f_prob[b_result.predicted_index]) if b_result.predicted_index != f_result.predicted_index else 0.0

            a_metrics = compare_result(f_result, a_result, near_tie_threshold=radius)
            rows.append(
                {
                    "request_id": request.request_id,
                    "suite": request.metadata.get("suite", "shape"),
                    "candidate_count": len(request.candidates),
                    "fp32_margin": b_metrics.reference_probability_margin,
                    "inside_ambiguity_region": inside,
                    "ambiguity_set_size": len(ambiguity_set),
                    **{f"b_{key}": value for key, value in asdict(b_metrics).items()},
                    **{f"b_{key}": value for key, value in _centered(f_result.logits, b_result.logits).items()},
                    "b_probability_gate_pass": probability_pass,
                    "b_tv_gate_pass": tv_pass,
                    "b_decision_rule_pass": decision_pass,
                    "b_row_pass": probability_pass and tv_pass and decision_pass,
                    "b_argmax_equal": b_metrics.argmax_equal,
                    "oracle_decision_regret": regret,
                    **{f"a_{key}": value for key, value in asdict(a_metrics).items()},
                    "outputs": {"F_flat_fp32": f_payload, "A_flat_bf16": _outputs(a_result), "B_shared_bf16": _outputs(b_result)},
                }
            )
        del flat16, shared16
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (OUTPUT_DIR / "rows.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n", encoding="utf-8"
    )
    inside = sum(row["inside_ambiguity_region"] for row in rows)
    flips = [row for row in rows if not row["b_argmax_equal"]]
    summary = {
        "contract_version": contract["contract_version"],
        "rows": len(rows),
        "fp32_precondition_pass": True,
        "post_fp32_release_memory": post_fp32_release_memory,
        "bf16_probability_gate_pass": all(row["b_probability_gate_pass"] for row in rows),
        "bf16_tv_gate_pass": all(row["b_tv_gate_pass"] for row in rows),
        "decision_rule_pass": all(row["b_decision_rule_pass"] for row in rows),
        "worst_b_max_abs_probability_delta": max(row["b_max_abs_probability_delta"] for row in rows),
        "worst_b_total_variation": max(row["b_total_variation"] for row in rows),
        "exact_b_f_argmax_agreement": sum(row["b_argmax_equal"] for row in rows) / len(rows),
        "rows_inside_ambiguity_region": inside,
        "rows_outside_ambiguity_region": len(rows) - inside,
        "argmax_flips": [row["request_id"] for row in flips],
        # A flip is permitted only when the decision rule itself passes: inside
        # the ambiguity radius AND B selected a candidate within F's ambiguity
        # set. Region membership alone is not permission.
        "permitted_flips": [
            {"request_id": row["request_id"], "regret": row["oracle_decision_regret"]}
            for row in flips
            if row["b_decision_rule_pass"]
        ],
        "unpermitted_flips": [
            {"request_id": row["request_id"], "regret": row["oracle_decision_regret"]}
            for row in flips
            if not row["b_decision_rule_pass"]
        ],
        "failing_rows": [row["request_id"] for row in rows if not row["b_row_pass"]],
        "shared_bf16_operational_status": "SUPPORTED"
        if all(row["b_row_pass"] for row in rows)
        else "UNSUPPORTED",
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
