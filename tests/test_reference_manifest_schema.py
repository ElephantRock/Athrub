# ruff: noqa: I001

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = ROOT / "schemas" / "reference_manifest.schema.json"


def test_reference_manifest_schema_is_identity_neutral_and_complete() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    assert schema["title"] == "Athrub A0 Reference Manifest"
    required = set(schema["required"])
    assert {
        "schema_version",
        "reference_name",
        "athrub_commit",
        "substrate",
        "tokenizer",
        "decision_head",
        "codec",
        "training",
        "benchmark",
    } <= required

    substrate_required = set(schema["properties"]["substrate"]["required"])
    assert {"artifact_sha256", "immutable_revision", "private_provenance_id"} <= substrate_required

    serialized = json.dumps(schema).lower()
    assert "vendor" not in serialized
    assert "product_name" not in serialized
    assert "upstream_name" not in serialized
