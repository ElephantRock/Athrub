"""Identity-neutral provenance helpers for freezing Athrub reference artifacts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path


def _hash_chunks(chunks: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: str | Path) -> str:
    """Hash a file or directory deterministically.

    Directory hashes include normalized relative paths and file contents in sorted order,
    which makes a local reference-substrate/tokenizer bundle addressable without exposing
    its source identity in the public manifest.
    """

    root = Path(path)
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        with root.open("rb") as handle:
            return _hash_chunks(iter(lambda: handle.read(1024 * 1024), b""))

    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"directory contains no files: {root}")
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class ReferenceFreezeInputs:
    reference_name: str
    athrub_commit: str
    substrate_path: Path
    substrate_revision: str
    substrate_private_provenance_id: str
    tokenizer_path: Path
    tokenizer_revision: str
    tokenizer_private_provenance_id: str
    decision_head_path: Path
    codec_source_commit: str
    training_config_path: Path
    data_manifest_path: Path
    benchmark_config_path: Path
    benchmark_artifact_path: Path
    environment_manifest_path: Path
    precision: tuple[str, ...]
    parameter_count: int | None = None
    head_bias: bool = True


def build_reference_manifest(inputs: ReferenceFreezeInputs) -> dict[str, object]:
    """Build the public A0 manifest without source/vendor/product identity."""

    if inputs.reference_name != "Athrub A0 Reference v0.1":
        raise ValueError("A0 v0.1 freeze must use the canonical reference name")
    if not inputs.precision:
        raise ValueError("at least one precision mode is required")
    unsupported = set(inputs.precision) - {"fp32", "bf16"}
    if unsupported:
        raise ValueError(f"unsupported precision modes: {sorted(unsupported)}")

    substrate: dict[str, object] = {
        "artifact_sha256": sha256_path(inputs.substrate_path),
        "immutable_revision": inputs.substrate_revision,
        "private_provenance_id": inputs.substrate_private_provenance_id,
        "architecture_class": "causal-transformer-reference-substrate",
    }
    if inputs.parameter_count is not None:
        if inputs.parameter_count < 1:
            raise ValueError("parameter_count must be positive")
        substrate["parameter_count"] = inputs.parameter_count

    return {
        "schema_version": "0.1",
        "reference_name": inputs.reference_name,
        "athrub_commit": inputs.athrub_commit,
        "substrate": substrate,
        "tokenizer": {
            "artifact_sha256": sha256_path(inputs.tokenizer_path),
            "immutable_revision": inputs.tokenizer_revision,
            "private_provenance_id": inputs.tokenizer_private_provenance_id,
        },
        "decision_head": {
            "artifact_sha256": sha256_path(inputs.decision_head_path),
            "head_type": "scalar-linear",
            "bias": inputs.head_bias,
        },
        "codec": {
            "name": "TextDecisionCodec",
            "version": "0.1",
            "source_commit": inputs.codec_source_commit,
        },
        "training": {
            "config_sha256": sha256_path(inputs.training_config_path),
            "data_manifest_sha256": sha256_path(inputs.data_manifest_path),
            "objective": "categorical-cross-entropy",
            "stages": ["head-warmup", "brief-full-adaptation"],
        },
        "benchmark": {
            "config_sha256": sha256_path(inputs.benchmark_config_path),
            "artifact_sha256": sha256_path(inputs.benchmark_artifact_path),
            "environment_sha256": sha256_path(inputs.environment_manifest_path),
            "precision": list(inputs.precision),
        },
    }


def write_reference_manifest(path: str | Path, manifest: dict[str, object]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
