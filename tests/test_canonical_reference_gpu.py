"""GPU regression of the tracked attention runtime against the frozen A0 bundle.

Disabled by default: set ``ATHRUB_GPU_REGRESSION=1`` on a CUDA machine that also
holds the frozen ``artifacts/a0-reference-v0.1-run1`` bundle. CI (CPU, no bundle)
always skips. Acceptance: logits, probabilities, predicted indices, and tokenizer
accounting reproduce the frozen canonical outputs bit-exactly.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import torch

from athrub.attention_runtime import CANONICAL_A0_POLICY, attention_runtime
from athrub.reference import FlatReferenceBackend, ScalarDecisionHead
from athrub.workloads import semantic_smoke_requests, synthetic_request

RUN1 = Path("artifacts/a0-reference-v0.1-run1")
FROZEN = RUN1 / "canonical-benchmark"
_REGRESSION_REQUESTED = os.environ.get("ATHRUB_GPU_REGRESSION") == "1"
_BUNDLE_PRESENT = (RUN1 / "reference_manifest.json").is_file() and FROZEN.is_dir()

pytestmark = pytest.mark.skipif(
    not (_REGRESSION_REQUESTED and _BUNDLE_PRESENT and torch.cuda.is_available()),
    reason="set ATHRUB_GPU_REGRESSION=1 on a CUDA machine with the frozen A0 bundle",
)


def _frozen_rows(relative: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in (FROZEN / relative).read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows.setdefault(row["request_id"], row)
    return rows


def _load_backend(dtype: torch.dtype, substrate_sha: str, tokenizer_sha: str) -> FlatReferenceBackend:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(RUN1 / "tokenizer"))
    model = AutoModel.from_pretrained(str(RUN1 / "substrate"), dtype=dtype)
    head = ScalarDecisionHead(int(model.config.hidden_size), bias=True)
    head.load_state_dict(
        torch.load(RUN1 / "decision_head.pt", map_location="cpu", weights_only=True),
        strict=True,
    )
    return FlatReferenceBackend(
        model=model,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        head=head,
        device="cuda",
        dtype=dtype,
        substrate_revision=f"sha256:{substrate_sha[:16]}",
        tokenizer_revision=f"sha256:{tokenizer_sha[:16]}",
    )


@pytest.fixture(scope="module")
def bundle_hashes() -> dict[str, str]:
    from athrub.provenance import sha256_path

    return {
        "substrate": sha256_path(RUN1 / "substrate"),
        "tokenizer": sha256_path(RUN1 / "tokenizer"),
    }


@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_semantic_and_shape_reproduce_frozen_bundle(
    precision: str, bundle_hashes: dict[str, str]
) -> None:
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[precision]
    backend = _load_backend(dtype, bundle_hashes["substrate"], bundle_hashes["tokenizer"])
    semantic_frozen = _frozen_rows(f"semantic/{precision}.jsonl")
    shape_frozen = _frozen_rows(f"shape/{precision}.jsonl")

    cells = [
        synthetic_request(request_id="shape-p128-k2-c16", prefix_units=128, candidate_units=16, candidate_count=2),
    ]
    if precision == "bf16":
        cells.append(
            synthetic_request(request_id="shape-p1024-k16-c16", prefix_units=1024, candidate_units=16, candidate_count=16)
        )

    with attention_runtime(CANONICAL_A0_POLICY) as session:
        session.observe_model_config(backend.model.config)
        assert session.last_expansion_ratio in (None, 2)

        for result in backend.score(semantic_smoke_requests()):
            row = semantic_frozen[result.request_id]
            assert result.predicted_index == row["predicted_index"]
            assert max(abs(a - b) for a, b in zip(result.logits, row["logits"], strict=True)) == 0.0
            assert max(abs(a - b) for a, b in zip(result.probabilities, row["probabilities"], strict=True)) == 0.0
            assert result.metadata["prefix_tokens"] == row["prefix_tokens"]
            assert list(result.metadata["path_token_counts"]) == row["path_token_counts"]

        # Canonical discipline: every shape cell is scored individually. Batching
        # different-length cells together changes padding and can shift BF16
        # reductions by one ulp, which flips near-tie argmax results.
        for cell in cells:
            for result in backend.score([cell]):
                row = shape_frozen[result.request_id]
                assert result.predicted_index == row["predicted_index"]
                assert max(abs(a - b) for a, b in zip(result.logits, row["logits"], strict=True)) == 0.0
                assert max(abs(a - b) for a, b in zip(result.probabilities, row["probabilities"], strict=True)) == 0.0

        assert session.query_heads == 16
        assert session.kv_heads == 8
        assert session.last_expansion_ratio == 2
        assert session.expansion_calls > 0
