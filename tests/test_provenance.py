from __future__ import annotations

from pathlib import Path

import pytest

from athrub.provenance import ReferenceFreezeInputs, build_reference_manifest, sha256_path


def _write(path: Path, value: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    return path


def test_directory_hash_is_deterministic_and_path_sensitive(tmp_path: Path) -> None:
    first = tmp_path / "bundle"
    _write(first / "a.txt", "alpha")
    _write(first / "nested" / "b.txt", "beta")

    digest = sha256_path(first)
    assert digest == sha256_path(first)
    assert len(digest) == 64

    _write(first / "nested" / "b.txt", "changed")
    assert sha256_path(first) != digest


def test_build_reference_manifest_exposes_hashes_not_source_names(tmp_path: Path) -> None:
    substrate = _write(tmp_path / "substrate.bin", "substrate")
    tokenizer = _write(tmp_path / "tokenizer.json", "tokenizer")
    head = _write(tmp_path / "head.pt", "head")
    training = _write(tmp_path / "training.json", "{}")
    data = _write(tmp_path / "data.json", "{}")
    benchmark_config = _write(tmp_path / "benchmark-config.json", "{}")
    benchmark = _write(tmp_path / "benchmark.jsonl", "{}\n")
    environment = _write(tmp_path / "environment.json", "{}")

    inputs = ReferenceFreezeInputs(
        reference_name="Athrub A0 Reference v0.1",
        athrub_commit="a" * 40,
        substrate_path=substrate,
        substrate_revision="immutable-substrate-revision",
        substrate_private_provenance_id="PRIVATE-SUBSTRATE-001",
        tokenizer_path=tokenizer,
        tokenizer_revision="immutable-tokenizer-revision",
        tokenizer_private_provenance_id="PRIVATE-TOKENIZER-001",
        decision_head_path=head,
        codec_source_commit="b" * 40,
        training_config_path=training,
        data_manifest_path=data,
        benchmark_config_path=benchmark_config,
        benchmark_artifact_path=benchmark,
        environment_manifest_path=environment,
        precision=("fp32", "bf16"),
        parameter_count=600_000_000,
    )

    manifest = build_reference_manifest(inputs)

    assert manifest["reference_name"] == "Athrub A0 Reference v0.1"
    assert manifest["substrate"]["artifact_sha256"] == sha256_path(substrate)
    assert manifest["tokenizer"]["artifact_sha256"] == sha256_path(tokenizer)
    assert manifest["decision_head"]["artifact_sha256"] == sha256_path(head)
    assert "source_name" not in manifest["substrate"]
    assert "vendor" not in manifest["substrate"]


def test_build_reference_manifest_rejects_unknown_precision(tmp_path: Path) -> None:
    artifact = _write(tmp_path / "artifact", "x")
    inputs = ReferenceFreezeInputs(
        reference_name="Athrub A0 Reference v0.1",
        athrub_commit="a" * 40,
        substrate_path=artifact,
        substrate_revision="substrate-revision",
        substrate_private_provenance_id="PRIVATE-SUBSTRATE-001",
        tokenizer_path=artifact,
        tokenizer_revision="tokenizer-revision",
        tokenizer_private_provenance_id="PRIVATE-TOKENIZER-001",
        decision_head_path=artifact,
        codec_source_commit="b" * 40,
        training_config_path=artifact,
        data_manifest_path=artifact,
        benchmark_config_path=artifact,
        benchmark_artifact_path=artifact,
        environment_manifest_path=artifact,
        precision=("fp8",),
    )

    with pytest.raises(ValueError, match="unsupported precision"):
        build_reference_manifest(inputs)
