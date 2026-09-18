"""Generate Athrub-native programmatic supervision for the first A0 run."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
from pathlib import Path

from athrub.a0_data import GENERATOR_VERSION, GENERATORS, dataset_summary, generate_decisions, write_decision_jsonl
from athrub.provenance import sha256_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/a0-data-v0.1")
    parser.add_argument("--train-count", type=int, default=50000)
    parser.add_argument("--validation-count", type=int, default=5000)
    parser.add_argument("--train-seed", type=int, default=1729)
    parser.add_argument("--validation-seed", type=int, default=2718)
    parser.add_argument("--candidate-min", type=int, default=2)
    parser.add_argument("--candidate-max", type=int, default=8)
    parser.add_argument(
        "--families",
        nargs="*",
        default=sorted(GENERATORS),
        choices=sorted(GENERATORS),
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train = generate_decisions(
        count=args.train_count,
        seed=args.train_seed,
        candidate_min=args.candidate_min,
        candidate_max=args.candidate_max,
        families=args.families,
    )
    validation = generate_decisions(
        count=args.validation_count,
        seed=args.validation_seed,
        candidate_min=args.candidate_min,
        candidate_max=args.candidate_max,
        families=args.families,
    )

    train_path = output_dir / "train.jsonl"
    validation_path = output_dir / "validation.jsonl"
    write_decision_jsonl(train_path, train)
    write_decision_jsonl(validation_path, validation)

    manifest = {
        "dataset_name": "Athrub A0 Programmatic Bootstrap v0.1",
        "generator_version": GENERATOR_VERSION,
        "families": list(args.families),
        "train_seed": args.train_seed,
        "validation_seed": args.validation_seed,
        "candidate_min": args.candidate_min,
        "candidate_max": args.candidate_max,
        "train": {
            "path": train_path.name,
            "sha256": sha256_path(train_path),
            "summary": dataset_summary(train),
        },
        "validation": {
            "path": validation_path.name,
            "sha256": sha256_path(validation_path),
            "summary": dataset_summary(validation),
        },
        "scope_note": (
            "Programmatically verifiable A0 bootstrap supervision. It is not evidence of "
            "cross-domain generalization or production capability."
        ),
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "manifest": str(manifest_path)}, indent=2))


if __name__ == "__main__":
    main()
