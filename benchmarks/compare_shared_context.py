"""Compare Athrub flat and shared-context execution on the same frozen weights."""

# Executable src-layout imports are intentionally grouped by dependency boundary.
# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from athrub.benchmark import benchmark_batch, environment_metadata, latency_summary, write_jsonl
from athrub.comparison import compare_result, summarize
from athrub.reference import FlatReferenceBackend
from athrub.shared_context import SharedContextBackend
from athrub.workloads import semantic_smoke_requests


DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/reference.example.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    dtype_name = str(config.get("dtype", "float32"))
    if dtype_name not in DTYPES:
        raise ValueError(f"unsupported dtype: {dtype_name}")

    reference = FlatReferenceBackend.from_pretrained(
        model_id=str(config["model_id"]),
        revision=str(config["revision"]),
        tokenizer_id=config.get("tokenizer_id"),
        tokenizer_revision=config.get("tokenizer_revision"),
        head_path=str(config["head_path"]),
        device=str(config.get("device", "cuda")),
        dtype=DTYPES[dtype_name],
        head_bias=bool(config.get("head_bias", True)),
        add_bos=bool(config.get("add_bos", True)),
    )
    shared = SharedContextBackend(
        model=reference.model,
        tokenizer=reference.tokenizer,
        head=reference.head,
        device=reference.device,
        dtype=reference.dtype,
        codec=reference.codec,
        add_bos=reference.add_bos,
        model_revision=reference.model_revision,
        tokenizer_revision=reference.tokenizer_revision,
        profile_stages=bool(config.get("profile_stages", False)),
    )

    requests = semantic_smoke_requests()
    reference_results = reference.score(requests)
    shared_results = shared.score(requests)
    comparisons = [
        compare_result(reference_result, shared_result)
        for reference_result, shared_result in zip(
            reference_results, shared_results, strict=True
        )
    ]

    warmups = int(config.get("warmups", 5))
    repeats = int(config.get("repeats", 20))
    reference_records = benchmark_batch(reference, requests, warmups=warmups, repeats=repeats)
    shared_records = benchmark_batch(shared, requests, warmups=warmups, repeats=repeats)

    output_dir = Path(str(config.get("comparison_output_dir", "artifacts/benchmarks/shared")))
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "reference.jsonl", reference_records)
    write_jsonl(output_dir / "shared.jsonl", shared_records)
    with (output_dir / "comparisons.jsonl").open("w", encoding="utf-8") as handle:
        for comparison in comparisons:
            handle.write(json.dumps(asdict(comparison), sort_keys=True) + "\n")

    report = {
        "environment": environment_metadata(),
        "precision": dtype_name,
        "comparison": summarize(comparisons),
        "reference_latency": latency_summary(reference_records),
        "shared_latency": latency_summary(shared_records),
        "output_dir": str(output_dir),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
