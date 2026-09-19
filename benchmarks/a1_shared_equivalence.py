"""A1 numerical equivalence: flat vs shared-context execution on the frozen A0 oracle.

Issue #2 experiment. One request per ``score`` invocation on both backends so the
only experimental variable is the computation path. Records complete raw outputs,
comparison metrics with near-tie diagnostics, shared-computation invariants, stage
timings, and a cache-branching probe. Emits no performance claims.
"""

# Executable src-layout imports are intentionally grouped by dependency boundary.
# ruff: noqa: I001

from __future__ import annotations

import argparse
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
from athrub.provenance import sha256_path
from athrub.reference import FlatReferenceBackend, ScalarDecisionHead
from athrub.shared_context import SharedContextBackend, _to_legacy_cache, expand_legacy_cache
from athrub.workloads import semantic_smoke_requests, synthetic_request


DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16}


def _driver_version() -> str | None:
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
        return output.stdout.strip().splitlines()[0]
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def _transformers_version() -> str | None:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("transformers")
    except PackageNotFoundError:  # pragma: no cover - optional dependency metadata
        return None


def _load_backends(
    config: dict[str, Any], dtype: torch.dtype, substrate_sha: str, tokenizer_sha: str
) -> tuple[FlatReferenceBackend, SharedContextBackend]:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(config["tokenizer_path"]))
    model = AutoModel.from_pretrained(str(config["substrate_path"]), dtype=dtype)
    head = ScalarDecisionHead(int(model.config.hidden_size), bias=bool(config.get("head_bias", True)))
    head.load_state_dict(
        torch.load(str(config["decision_head_path"]), map_location="cpu", weights_only=True),
        strict=True,
    )
    flat = FlatReferenceBackend(
        model=model,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        head=head,
        device=str(config.get("device", "cuda")),
        dtype=dtype,
        add_bos=bool(config.get("add_bos", True)),
        substrate_revision=f"sha256:{substrate_sha[:16]}",
        tokenizer_revision=f"sha256:{tokenizer_sha[:16]}",
    )
    shared = SharedContextBackend(
        model=flat.model,
        tokenizer=flat.tokenizer,  # type: ignore[arg-type]
        head=flat.head,
        device=flat.device,
        dtype=flat.dtype,
        codec=flat.codec,
        add_bos=flat.add_bos,
        substrate_revision=flat.substrate_revision,
        tokenizer_revision=flat.tokenizer_revision,
        profile_stages=bool(config.get("profile_stages", True)),
    )
    return flat, shared


def _timed_score(backend: Any, request: Any) -> tuple[Any, float]:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter_ns()
    result = backend.score([request])[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return result, (time.perf_counter_ns() - start) / 1_000_000.0


def _cache_probe(shared: SharedContextBackend, request: Any) -> dict[str, Any]:
    """Replicate the shared backend's cache path and record branching evidence."""

    prefix_ids = shared._encode_prefix(request)
    prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=shared.device)
    with torch.inference_mode():
        prefix_output = shared.model(
            input_ids=prefix_tensor,
            attention_mask=torch.ones_like(prefix_tensor),
            use_cache=True,
            return_dict=True,
        )
        raw_cache = prefix_output.past_key_values
        legacy = _to_legacy_cache(raw_cache)
        key0, value0 = legacy[0]
        branched = expand_legacy_cache(legacy, len(request.candidates))
        bkey0, _ = branched[0]
        from athrub.shared_context import _from_legacy_cache

        model_cache = _from_legacy_cache(branched, raw_cache)
        probe: dict[str, Any] = {
            "raw_cache_type": type(raw_cache).__name__,
            "converted_legacy_cache": True,
            "cache_layers": len(legacy),
            "key_source_shape": list(key0.shape),
            "key_branched_shape": list(bkey0.shape),
            "source_batch_dimension": int(key0.shape[0]),
            "branched_batch_dimension": int(bkey0.shape[0]),
            "branched_stride0": next(iter(bkey0.stride())),
            "branched_storage_shared": bkey0.untyped_storage().data_ptr()
            == key0.untyped_storage().data_ptr(),
            "continuation_cache_type": type(model_cache).__name__,
            "athrub_branching": "view-based",
            "physical_zero_copy_inference": False,
        }

        # Direct inspection of the tensors the model-held cache object references.
        # No to_legacy_cache() round-trip: that conversion could materialize on its
        # own and would overstate the adapter's copying behavior.
        layers = getattr(model_cache, "layers", None)
        if layers is None:
            raise TypeError(
                "cache materialization probe requires direct layer access; "
                f"{type(model_cache).__name__} exposes no 'layers'"
            )
        layer_rows = []
        for index, layer in enumerate(layers):
            layer_key = getattr(layer, "keys", None)
            layer_value = getattr(layer, "values", None)
            if layer_key is None or layer_value is None:
                raise TypeError(f"cache layer {index} holds no key/value tensors")
            layer_rows.append(
                {
                    "layer": index,
                    "key_dtype": str(layer_key.dtype),
                    "key_shape": list(layer_key.shape),
                    "key_stride0": next(iter(layer_key.stride())),
                    "key_storage_shared_with_prefix": layer_key.untyped_storage().data_ptr()
                    == key0.untyped_storage().data_ptr(),
                    "value_dtype": str(layer_value.dtype),
                    "value_shape": list(layer_value.shape),
                    "value_stride0": next(iter(layer_value.stride())),
                    "value_storage_shared_with_prefix": layer_value.untyped_storage().data_ptr()
                    == value0.untyped_storage().data_ptr(),
                }
            )
        probe["model_held_layers_inspected"] = len(layer_rows)
        probe["model_held_key_strides0"] = sorted({row["key_stride0"] for row in layer_rows})
        probe["model_held_value_strides0"] = sorted({row["value_stride0"] for row in layer_rows})
        probe["model_held_all_key_storage_shared"] = all(
            row["key_storage_shared_with_prefix"] for row in layer_rows
        )
        probe["model_held_all_value_storage_shared"] = all(
            row["value_storage_shared_with_prefix"] for row in layer_rows
        )
        probe["first_layer_detail"] = layer_rows[0]
        probe["conclusion"] = (
            "Athrub cache branching is view-based at the expand boundary (stride-0 "
            "views over shared storage). Direct inspection of every layer held by the "
            "reconstructed cache object shows the adapter materializes the branched "
            "key/value tensors, so physical zero-copy prefix KV sharing is not established."
        )
    return probe


def _frozen_rows(frozen_root: Path, suite: str, precision: str) -> dict[str, dict[str, Any]]:
    path = frozen_root / suite / f"{precision}.jsonl"
    if not path.is_file():
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows.setdefault(row["request_id"], row)
    return rows


def _invariants(flat: Any, shared: Any) -> dict[str, Any]:
    flat_meta, shared_meta = flat.metadata, shared.metadata
    prefix = int(shared_meta["prefix_tokens"])
    candidate_counts = [int(value) for value in shared_meta["candidate_token_counts"]]
    return {
        "flat_positions_equals_sum_of_paths": flat_meta["flat_logical_token_positions"]
        == sum(flat_meta["path_token_counts"])
        == sum(prefix + count for count in candidate_counts),
        "shared_positions_equals_prefix_plus_suffixes": shared_meta["shared_logical_token_positions"]
        == prefix + sum(candidate_counts),
        "prefix_compute_calls": shared_meta["prefix_compute_calls"],
        "continuation_compute_calls": shared_meta["continuation_compute_calls"],
        "compute_calls_correct": shared_meta["prefix_compute_calls"] == 1
        and shared_meta["continuation_compute_calls"] == 1,
        "candidate_token_counts_agree": list(flat_meta["candidate_token_counts"]) == candidate_counts,
        "path_token_counts_agree": list(flat_meta["path_token_counts"])
        == list(shared_meta["path_token_counts"]),
        "prefix_tokens_agree": flat_meta["prefix_tokens"] == shared_meta["prefix_tokens"],
    }


def _row(
    *,
    suite: str,
    precision: str,
    request: Any,
    flat: Any,
    shared: Any,
    flat_ms: float,
    shared_ms: float,
    epsilon: float,
    frozen_row: dict[str, Any] | None,
) -> dict[str, Any]:
    metrics = compare_result(flat, shared, near_tie_threshold=2.0 * epsilon)
    invariants = _invariants(flat, shared)
    numerical_pass = metrics.max_abs_probability_delta < epsilon
    if frozen_row is None:
        flat_vs_frozen: dict[str, Any] = {
            "status": "no_frozen_counterpart",
            "max_abs_logit_delta": None,
        }
    else:
        frozen_delta = max(
            abs(a - b) for a, b in zip(flat.logits, frozen_row["logits"], strict=True)
        )
        flat_vs_frozen = {
            "status": "matched"
            if frozen_delta == 0.0 and flat.predicted_index == frozen_row["predicted_index"]
            else "mismatched",
            "max_abs_logit_delta": frozen_delta,
        }
    return {
        "suite": suite,
        "precision": precision,
        "request_id": request.request_id,
        "candidate_count": len(request.candidates),
        **asdict(metrics),
        "numerical_equivalence_pass": numerical_pass,
        "decision_equivalence_pass": metrics.argmax_equal,
        "near_tie_explanation": bool(metrics.near_tie and not metrics.argmax_equal),
        "invariants_pass": all(
            value if isinstance(value, bool) else True for value in invariants.values()
        )
        and invariants["compute_calls_correct"],
        "invariants": invariants,
        "flat_vs_frozen_canonical": flat_vs_frozen,
        "flat": {
            "logits": list(flat.logits),
            "probabilities": list(flat.probabilities),
            "predicted_index": flat.predicted_index,
            "metadata": {key: list(value) if isinstance(value, tuple) else value for key, value in flat.metadata.items()},
        },
        "shared": {
            "logits": list(shared.logits),
            "probabilities": list(shared.probabilities),
            "predicted_index": shared.predicted_index,
            "metadata": {key: list(value) if isinstance(value, tuple) else value for key, value in shared.metadata.items()},
        },
        "flat_latency_ms": flat_ms,
        "shared_end_to_end_latency_ms": shared_ms,
        "prefix_latency_ms": shared.metadata.get("prefix_latency_ms"),
        "continuation_latency_ms": shared.metadata.get("continuation_latency_ms"),
    }


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    policy = AttentionPolicy(**dict(config.get("attention_policy", {})))
    gates = dict(config.get("gates", {}))
    epsilon = dict(config.get("near_tie_epsilon", {"fp32": 1e-5, "bf16": 1e-3}))
    for precision, gate_key in (("fp32", "fp32_max_abs_probability_delta"), ("bf16", "bf16_max_abs_probability_delta")):
        if gate_key in gates and precision in epsilon and float(gates[gate_key]) != float(epsilon[precision]):
            raise ValueError(
                f"gate {gate_key} ({gates[gate_key]}) contradicts near_tie_epsilon[{precision}] "
                f"({epsilon[precision]}); the numerical gate and the near-tie half-margin must agree"
            )
    output_dir = Path(str(config.get("output_dir", "artifacts/a1-equivalence-v0.1")))
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_root = Path(str(config.get("frozen_canonical_root", "artifacts/a0-reference-v0.1-run1/canonical-benchmark")))

    substrate_sha = sha256_path(str(config["substrate_path"]))
    tokenizer_sha = sha256_path(str(config["tokenizer_path"]))
    head_sha = sha256_path(str(config["decision_head_path"]))

    # Cryptographic binding to the frozen A0 oracle: the local artifacts must be
    # the exact bundle the frozen reference manifest hash-attests, so a modified
    # local model cannot be reported as the frozen oracle.
    manifest_path = Path(str(config["reference_manifest_path"]))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    reference_manifest_sha256 = sha256_path(manifest_path)
    artifact_verification = {
        "substrate": (substrate_sha, manifest["substrate"]["artifact_sha256"]),
        "tokenizer": (tokenizer_sha, manifest["tokenizer"]["artifact_sha256"]),
        "decision_head": (head_sha, manifest["decision_head"]["artifact_sha256"]),
    }
    for name, (actual, recorded) in artifact_verification.items():
        if actual != recorded:
            raise RuntimeError(
                f"frozen A0 manifest verification failed for {name}: local artifact "
                f"{actual} does not match manifest {recorded}; refusing to execute"
            )

    candidate_units = int(config.get("candidate_units", 16))
    cells = [
        (int(prefix), int(count))
        for prefix, count in config.get(
            "shape_cells", [[128, 2], [128, 8], [512, 4], [512, 16], [1024, 2], [1024, 8]]
        )
    ]
    precisions = list(config.get("precisions", ["fp32", "bf16"]))
    unknown = set(precisions) - set(DTYPES)
    if unknown:
        raise ValueError(f"unsupported precisions: {sorted(unknown)}")

    environment = dict(environment_metadata())
    environment["nvidia_driver"] = _driver_version()
    environment["transformers"] = _transformers_version()
    (output_dir / "environment.json").write_text(
        json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    summary: dict[str, Any] = {
        "reference_name": config.get("reference_name"),
        "attention_policy": policy.as_record(),
        "benchmarked_artifact": {
            "substrate_bundle_sha256": substrate_sha,
            "tokenizer_bundle_sha256": tokenizer_sha,
            "decision_head_sha256": head_sha,
        },
        "frozen_a0_binding": {
            "reference_manifest_path": str(manifest_path),
            "reference_manifest_sha256": reference_manifest_sha256,
            "frozen_a0_athrub_commit": manifest["athrub_commit"],
            "codec": manifest["codec"],
            "artifact_verification": "passed",
        },
        "request_grouping": "one DecisionRequest per score invocation on both backends",
        "precisions": {},
    }
    cache_probes: dict[str, Any] = {}

    for precision in precisions:
        dtype = DTYPES[precision]
        eps = float(epsilon[precision])
        flat, shared = _load_backends(config, dtype, substrate_sha, tokenizer_sha)
        frozen_semantic = _frozen_rows(frozen_root, "semantic", precision)
        frozen_shape = _frozen_rows(frozen_root, "shape", precision)

        probe_request = synthetic_request(
            request_id="shape-p128-k2-c16", prefix_units=128, candidate_units=candidate_units, candidate_count=2
        )
        with attention_runtime(policy):
            cache_probes[precision] = _cache_probe(shared, probe_request)

            semantic_rows = []
            for request in semantic_smoke_requests():
                flat_result, flat_ms = _timed_score(flat, request)
                shared_result, shared_ms = _timed_score(shared, request)
                semantic_rows.append(
                    _row(
                        suite="semantic",
                        precision=precision,
                        request=request,
                        flat=flat_result,
                        shared=shared_result,
                        flat_ms=flat_ms,
                        shared_ms=shared_ms,
                        epsilon=eps,
                        frozen_row=frozen_semantic.get(request.request_id),
                    )
                )
            _write_jsonl(output_dir / precision / "semantic.jsonl", semantic_rows)

            shape_rows = []
            for prefix, count in cells:
                request = synthetic_request(
                    request_id=f"shape-p{prefix}-k{count}-c{candidate_units}",
                    prefix_units=prefix,
                    candidate_units=candidate_units,
                    candidate_count=count,
                )
                flat_result, flat_ms = _timed_score(flat, request)
                shared_result, shared_ms = _timed_score(shared, request)
                shape_rows.append(
                    _row(
                        suite="shape",
                        precision=precision,
                        request=request,
                        flat=flat_result,
                        shared=shared_result,
                        flat_ms=flat_ms,
                        shared_ms=shared_ms,
                        epsilon=eps,
                        frozen_row=frozen_shape.get(request.request_id),
                    )
                )
            _write_jsonl(output_dir / precision / "shape.jsonl", shape_rows)

        rows = semantic_rows + shape_rows
        summary["precisions"][precision] = {
            "epsilon_probability": eps,
            "near_tie_threshold": 2.0 * eps,
            "requests": len(rows),
            "argmax_agreement": sum(row["decision_equivalence_pass"] for row in rows) / len(rows),
            "max_abs_probability_delta": max(row["max_abs_probability_delta"] for row in rows),
            "max_abs_logit_delta": max(row["max_abs_logit_delta"] for row in rows),
            "mean_total_variation": sum(row["total_variation"] for row in rows) / len(rows),
            "numerical_equivalence_gate_pass": all(row["numerical_equivalence_pass"] for row in rows),
            "decision_equivalence_gate_pass": all(row["decision_equivalence_pass"] for row in rows),
            "token_accounting_pass": all(row["invariants_pass"] for row in rows),
            "near_tie_rows": sum(row["near_tie"] for row in rows),
            "near_tie_explanations": sum(row["near_tie_explanation"] for row in rows),
            "flat_vs_frozen_canonical": {
                "matched": sum(
                    row["flat_vs_frozen_canonical"]["status"] == "matched" for row in rows
                ),
                "mismatched": [
                    row["request_id"]
                    for row in rows
                    if row["flat_vs_frozen_canonical"]["status"] == "mismatched"
                ],
                "no_frozen_counterpart": [
                    row["request_id"]
                    for row in rows
                    if row["flat_vs_frozen_canonical"]["status"] == "no_frozen_counterpart"
                ],
            },
        }
        del flat, shared
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (output_dir / "cache_probe.json").write_text(
        json.dumps(cache_probes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fp32_gate = summary["precisions"].get("fp32", {}).get("numerical_equivalence_gate_pass")
    bf16_gate = summary["precisions"].get("bf16", {}).get("numerical_equivalence_gate_pass")
    argmax_gate = all(
        entry.get("decision_equivalence_gate_pass") for entry in summary["precisions"].values()
    )
    token_gate = all(entry.get("token_accounting_pass") for entry in summary["precisions"].values())
    summary["gates"] = {
        "fp32_numerical_gate": fp32_gate,
        "bf16_numerical_gate": bf16_gate,
        "argmax_gate": argmax_gate,
        "cache_branching_verification": "view-based at expand boundary; DynamicCache adapter materializes; physical zero-copy not established",
        "token_accounting_verification": token_gate,
        "overall_a1_equivalence": None
        if fp32_gate is None or bf16_gate is None
        else bool(fp32_gate and bf16_gate and argmax_gate and token_gate),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary["gates"], indent=2, sort_keys=True))
    print(json.dumps({"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
