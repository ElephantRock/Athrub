"""Shared helpers for the A1 BF16 stability diagnostics (Issue #9).

These diagnostics run against the frozen A0 reference bundle and always execute
under the canonical attention policy. They are research evidence tools, not
production paths; SharedContextBackend itself is unchanged.
"""


from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from athrub.attention_runtime import CANONICAL_A0_POLICY, attention_runtime
from athrub.provenance import sha256_path
from athrub.reference import FlatReferenceBackend, ScalarDecisionHead
from athrub.shared_context import (
    SharedContextBackend,
    _from_legacy_cache,
    _to_legacy_cache,
    expand_legacy_cache,
)

RUN1 = Path("artifacts/a0-reference-v0.1-run1")
OUT = Path("artifacts/a1-bf16-stability")


def verify_frozen_a0_manifest() -> str:
    """Refuse to execute unless the local bundle matches the frozen manifest."""

    manifest = json.loads((RUN1 / "reference_manifest.json").read_text(encoding="utf-8"))
    checks = {
        "substrate": (sha256_path(RUN1 / "substrate"), manifest["substrate"]["artifact_sha256"]),
        "tokenizer": (sha256_path(RUN1 / "tokenizer"), manifest["tokenizer"]["artifact_sha256"]),
        "decision_head": (sha256_path(RUN1 / "decision_head.pt"), manifest["decision_head"]["artifact_sha256"]),
    }
    for name, (actual, recorded) in checks.items():
        if actual != recorded:
            raise SystemExit(f"frozen A0 manifest verification failed for {name}; refusing to execute")
    return sha256_path(RUN1 / "reference_manifest.json")


def load_flat(dtype: torch.dtype) -> FlatReferenceBackend:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(RUN1 / "tokenizer"))
    model = AutoModel.from_pretrained(str(RUN1 / "substrate"), dtype=dtype)
    head = ScalarDecisionHead(int(model.config.hidden_size), bias=True)
    head.load_state_dict(
        torch.load(RUN1 / "decision_head.pt", map_location="cpu", weights_only=True), strict=True
    )
    return FlatReferenceBackend(
        model=model,
        tokenizer=tokenizer,  # type: ignore[arg-type]
        head=head,
        device="cuda",
        dtype=dtype,
        substrate_revision="diagnostic",
        tokenizer_revision="diagnostic",
    )


def load_shared(flat: FlatReferenceBackend) -> SharedContextBackend:
    return SharedContextBackend(
        model=flat.model,
        tokenizer=flat.tokenizer,  # type: ignore[arg-type]
        head=flat.head,
        device=flat.device,
        dtype=flat.dtype,
        codec=flat.codec,
        add_bos=flat.add_bos,
        substrate_revision="diagnostic",
        tokenizer_revision="diagnostic",
    )


def load_head_bf16() -> ScalarDecisionHead:
    head = ScalarDecisionHead(1024, bias=True)
    head.load_state_dict(
        torch.load(RUN1 / "decision_head.pt", map_location="cpu", weights_only=True), strict=True
    )
    return head.to(device="cuda", dtype=torch.bfloat16)


def stats(a: torch.Tensor, b: torch.Tensor) -> dict[str, object]:
    delta = (a.float() - b.float()).abs()
    return {
        "max_abs_delta": float(delta.max()),
        "mean_abs_delta": float(delta.mean()),
        "rms_delta": float(delta.pow(2).mean().sqrt()),
        "final_prefix_token_delta": float(delta[-1].max()),
        "dtype": str(a.dtype),
        "shape": list(a.shape),
    }


def centered_decomposition(reference_logits, candidate_logits) -> dict[str, float]:
    """Split logit deltas into a softmax-invariant common mode and residuals."""

    deltas = [float(m) - float(a) for a, m in zip(reference_logits, candidate_logits, strict=True)]
    common = sum(deltas) / len(deltas)
    residuals = [d - common for d in deltas]
    return {
        "logit_delta_common_mode": common,
        "max_abs_centered_logit_delta": max(abs(r) for r in residuals),
        "mean_abs_centered_logit_delta": sum(abs(r) for r in residuals) / len(residuals),
    }


def mixed_execution(
    prefix_model: torch.nn.Module,
    continuation_model: torch.nn.Module,
    head: torch.nn.Module,
    template: SharedContextBackend,
    request,
) -> tuple[Any, dict[str, object]]:
    """FP32 prefix -> BF16-cast KV cache -> view branch -> BF16 continuation/head."""

    prefix_ids = template._encode_prefix(request)
    candidate_ids = [template._encode_candidate(c) for c in request.candidates]
    candidate_counts = [len(tokens) for tokens in candidate_ids]
    candidate_count = len(candidate_ids)

    prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=template.device)
    with torch.inference_mode():
        prefix_output = prefix_model(
            input_ids=prefix_tensor,
            attention_mask=torch.ones_like(prefix_tensor),
            use_cache=True,
            return_dict=True,
        )
        raw_cache = prefix_output.past_key_values
        legacy_bf16 = []
        for key, value in _to_legacy_cache(raw_cache):
            key16, value16 = key.to(torch.bfloat16), value.to(torch.bfloat16)
            if key16.shape != key.shape or value16.shape != value.shape:
                raise ValueError("cache shape changed during cast")
            if not torch.isfinite(key16.float()).all() or not torch.isfinite(value16.float()).all():
                raise ValueError("non-finite cache tensor after cast")
            legacy_bf16.append((key16, value16))
        branched = expand_legacy_cache(tuple(legacy_bf16), candidate_count)
        reconstructed = _from_legacy_cache(branched, raw_cache)

        max_suffix = max(candidate_counts)
        suffix_tensor = torch.full(
            (candidate_count, max_suffix), template._pad_token_id(), dtype=torch.long, device=template.device
        )
        suffix_mask = torch.zeros_like(suffix_tensor)
        for row, tokens in enumerate(candidate_ids):
            suffix_tensor[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long, device=template.device)
            suffix_mask[row, : len(tokens)] = 1
        full_mask = torch.cat(
            (torch.ones((candidate_count, len(prefix_ids)), dtype=torch.long, device=template.device), suffix_mask),
            dim=1,
        )
        position_ids = (
            torch.arange(len(prefix_ids), len(prefix_ids) + max_suffix, dtype=torch.long, device=template.device)
            .unsqueeze(0)
            .expand(candidate_count, -1)
        )
        continuation_output = continuation_model(
            input_ids=suffix_tensor,
            attention_mask=full_mask,
            position_ids=position_ids,
            past_key_values=reconstructed,
            use_cache=False,
            return_dict=True,
        )
        hidden = continuation_output.last_hidden_state
        row_indices = torch.arange(candidate_count, device=hidden.device)
        last_indices = torch.tensor(candidate_counts, dtype=torch.long, device=hidden.device) - 1
        scores = head(hidden[row_indices, last_indices]).squeeze(-1).float()
        probabilities = torch.softmax(scores, dim=0)

    from athrub.contracts import DecisionResult

    result = DecisionResult(
        request_id=request.request_id,
        logits=tuple(float(v) for v in scores.cpu()),
        probabilities=tuple(float(v) for v in probabilities.cpu()),
        predicted_index=int(torch.argmax(probabilities).item()),
        metadata={
            "prefix_tokens": len(prefix_ids),
            "candidate_token_counts": tuple(candidate_counts),
            "flat_logical_token_positions": sum(len(prefix_ids) + c for c in candidate_counts),
            "shared_logical_token_positions": len(prefix_ids) + sum(candidate_counts),
            "prefix_compute_calls": 1,
            "continuation_compute_calls": 1,
            "cache_dtype_transition": "fp32->bf16",
            "cache_layers": len(legacy_bf16),
        },
    )
    return result, {"cast_layers": len(legacy_bf16)}


def frozen_rows(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        rows.setdefault(row["request_id"], row)
    return rows


def outputs(result) -> dict[str, object]:
    return {
        "logits": list(result.logits),
        "probabilities": list(result.probabilities),
        "predicted_index": result.predicted_index,
    }


__all__ = [
    "CANONICAL_A0_POLICY",
    "OUT",
    "RUN1",
    "attention_runtime",
    "centered_decomposition",
    "frozen_rows",
    "load_flat",
    "load_head_bf16",
    "load_shared",
    "mixed_execution",
    "outputs",
    "stats",
    "verify_frozen_a0_manifest",
]
