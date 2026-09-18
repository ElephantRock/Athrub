from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from athrub.attention_runtime import (
    CANONICAL_A0_POLICY,
    AttentionPolicy,
    RuntimeSession,
    attention_runtime,
    expand_kv_heads,
)


def test_policy_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unsupported attention backend"):
        AttentionPolicy(backend="math_sdpa")


def test_policy_rejects_silent_fallback() -> None:
    with pytest.raises(ValueError, match="fail loudly"):
        AttentionPolicy(fallback_allowed=True)


def test_policy_rejects_unknown_gqa_strategy() -> None:
    with pytest.raises(ValueError, match="unsupported GQA strategy"):
        AttentionPolicy(gqa_strategy="expand_query_heads")


def test_canonical_policy_shape() -> None:
    assert CANONICAL_A0_POLICY.backend == "efficient_sdpa"
    assert CANONICAL_A0_POLICY.gqa_strategy == "expand_kv_heads"
    assert CANONICAL_A0_POLICY.q_to_kv_ratio == 2
    assert CANONICAL_A0_POLICY.fallback_allowed is False


def test_expand_kv_heads_matches_explicit_reference_math() -> None:
    generator = torch.Generator().manual_seed(1729)
    query = torch.randn(2, 4, 8, 16, generator=generator)
    key = torch.randn(2, 2, 8, 16, generator=generator)
    value = torch.randn(2, 2, 8, 16, generator=generator)

    expanded_key, expanded_value = expand_kv_heads(key, value, 2)
    assert expanded_key.shape == (2, 4, 8, 16)
    # Each expanded head must equal the source KV head it materializes.
    assert torch.equal(expanded_key[:, 0], key[:, 0])
    assert torch.equal(expanded_key[:, 1], key[:, 0])
    assert torch.equal(expanded_key[:, 2], key[:, 1])
    assert torch.equal(expanded_key[:, 3], key[:, 1])

    scores = query @ expanded_key.transpose(-1, -2) / (query.shape[-1] ** 0.5)
    reference = torch.softmax(scores, dim=-1) @ expanded_value
    computed = F.scaled_dot_product_attention(query, expanded_key, expanded_value)
    assert torch.allclose(computed, reference, atol=1e-5, rtol=1e-5)


def test_runtime_session_validates_declared_ratio() -> None:
    session = RuntimeSession(policy=CANONICAL_A0_POLICY)
    session.observe_model_config(SimpleNamespace(num_attention_heads=16, num_key_value_heads=8))
    assert session.query_heads == 16
    assert session.kv_heads == 8

    with pytest.raises(ValueError, match="contradicts declared"):
        session.observe_model_config(SimpleNamespace(num_attention_heads=16, num_key_value_heads=7))

    session.record_expansion(2)
    metadata = session.metadata()
    assert metadata["requested_attention_policy"] == CANONICAL_A0_POLICY.as_record()
    assert metadata["query_head_count"] == 16
    assert metadata["kv_head_count"] == 8
    assert metadata["gqa_expansion_ratio"] == 2
    assert metadata["gqa_expansion_calls"] == 1


@pytest.mark.skipif(torch.cuda.is_available(), reason="CUDA machines exercise the runtime directly")
def test_attention_runtime_requires_cuda() -> None:
    with (
        pytest.raises(RuntimeError, match="requires CUDA"),
        attention_runtime(CANONICAL_A0_POLICY),
    ):
        pass
