"""Run the canonical Athrub flat reference backend on semantic Phase 1 workloads."""

# Ruff's import sorter classifies this executable's src-layout imports differently from
# the installed package. Keep the dependency groups explicit and suppress only I001.
# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from athrub.benchmark import benchmark_batch, environment_metadata, latency_summary, write_jsonl
from athrub.reference import FlatReferenceBackend
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

    backend = FlatReferenceBackend.from_pretrained(
        substrate_id=str(config["substrate_id"]),
        substrate_revision=str(config["substrate_revision"]),
        tokenizer_id=config.get("tokenizer_id"),
        tokenizer_revision=config.get("tokenizer_revision"),
        head_path=str(config["head_path"]),
        device=str(config.get("device", "cuda")),
        dtype=DTYPES[dtype_name],
        head_bias=bool(config.get("head_bias", True)),
        add_bos=bool(config.get("add_bos", True)),
    )

    requests = semantic_smoke_requests()
    records = benchmark_batch(
        backend,
        requests,
        warmups=int(config.get("warmups", 5)),
        repeats=int(config.get("repeats", 20)),
    )
    output = Path(str(config.get("output", "artifacts/benchmarks/reference-semantic.jsonl")))
    write_jsonl(output, records)

    report = {
        "environment": environment_metadata(),
        "reference_name": config.get("reference_name"),
        "backend": backend.name,
        "requests": len(requests),
        "records": len(records),
        "latency": latency_summary(records),
        "output": str(output),
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
