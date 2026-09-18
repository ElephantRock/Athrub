"""Emit the identity-neutral public manifest for a frozen Athrub A0 reference."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
from pathlib import Path

from athrub.provenance import ReferenceFreezeInputs, build_reference_manifest, write_reference_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/freeze_reference.example.json")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    inputs = ReferenceFreezeInputs(
        reference_name=str(config["reference_name"]),
        athrub_commit=str(config["athrub_commit"]),
        substrate_path=Path(config["substrate_path"]),
        substrate_revision=str(config["substrate_revision"]),
        substrate_private_provenance_id=str(config["substrate_private_provenance_id"]),
        tokenizer_path=Path(config["tokenizer_path"]),
        tokenizer_revision=str(config["tokenizer_revision"]),
        tokenizer_private_provenance_id=str(config["tokenizer_private_provenance_id"]),
        decision_head_path=Path(config["decision_head_path"]),
        codec_source_commit=str(config["codec_source_commit"]),
        training_config_path=Path(config["training_config_path"]),
        data_manifest_path=Path(config["data_manifest_path"]),
        benchmark_config_path=Path(config["benchmark_config_path"]),
        benchmark_artifact_path=Path(config["benchmark_artifact_path"]),
        environment_manifest_path=Path(config["environment_manifest_path"]),
        precision=tuple(str(value) for value in config["precision"]),
        training_stages=tuple(
            str(value)
            for value in config.get(
                "training_stages", ["head-warmup", "brief-full-adaptation"]
            )
        ),
        parameter_count=(
            int(config["parameter_count"]) if config.get("parameter_count") is not None else None
        ),
        head_bias=bool(config.get("head_bias", True)),
    )
    manifest = build_reference_manifest(inputs)
    output = Path(str(config.get("output", "artifacts/reference/reference_manifest.json")))
    write_reference_manifest(output, manifest)
    print(json.dumps({"reference_name": inputs.reference_name, "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
