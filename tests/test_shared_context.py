from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from athrub.comparison import compare_result
from athrub.contracts import DecisionRequest
from athrub.reference import FlatReferenceBackend, ScalarDecisionHead
from athrub.shared_context import SharedContextBackend, expand_legacy_cache


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        output: list[int] = []
        for token in text.split():
            digest = hashlib.sha256(token.encode()).digest()
            output.append(3 + int.from_bytes(digest[:2], "big") % 997)
        return output


class CacheAwareAccumulatingModel(nn.Module):
    """Tiny causal model whose cached and full-path executions are equivalent."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool,
        use_cache: bool = False,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        position_ids: torch.Tensor | None = None,
    ) -> SimpleNamespace:
        assert return_dict is True
        self.calls += 1
        current_mask = attention_mask[:, -input_ids.shape[1] :].float()
        current_values = input_ids.float() * current_mask

        if past_key_values is None:
            offset = torch.zeros(input_ids.shape[0], device=input_ids.device)
            previous_history = None
        else:
            previous_history = past_key_values[0][0][:, 0, :, 0]
            offset = previous_history[:, -1]
            if position_ids is not None:
                expected = previous_history.shape[1]
                assert torch.all(position_ids[:, 0] == expected)

        cumulative = torch.cumsum(current_values, dim=1) + offset.unsqueeze(1)
        positions = attention_mask[:, -input_ids.shape[1] :].float().cumsum(dim=1)
        if previous_history is not None:
            positions = positions + previous_history.shape[1]

        hidden = torch.stack(
            (
                cumulative,
                cumulative * 0.1,
                cumulative * 0.01,
                positions,
            ),
            dim=-1,
        )

        cache = None
        if use_cache:
            history = cumulative
            if previous_history is not None:
                history = torch.cat((previous_history, cumulative), dim=1)
            key = history[:, None, :, None]
            value = key * 0.5
            cache = ((key, value),)

        return SimpleNamespace(last_hidden_state=hidden, past_key_values=cache)


def make_head() -> ScalarDecisionHead:
    head = ScalarDecisionHead(4, bias=False)
    with torch.no_grad():
        head.projection.weight.copy_(torch.tensor([[0.01, 0.02, -0.03, 0.04]]))
    return head


def make_request(candidate_count: int = 4) -> DecisionRequest:
    candidates = tuple(f"candidate option {index}" for index in range(candidate_count))
    return DecisionRequest(
        request_id=f"equivalence-k{candidate_count}",
        state="shared state with several facts and constraints",
        question="Which candidate best satisfies the objective?",
        candidates=candidates,
    )


def test_cache_branching_uses_views_without_explicit_tensor_copy() -> None:
    key = torch.arange(24, dtype=torch.float32).reshape(1, 2, 3, 4)
    value = key + 1
    expanded = expand_legacy_cache(((key, value),), batch_size=8)
    expanded_key = expanded[0][0]
    expanded_value = expanded[0][1]

    assert expanded_key.shape == (8, 2, 3, 4)
    assert expanded_value.shape == (8, 2, 3, 4)
    assert expanded_key.stride(0) == 0
    assert expanded_value.stride(0) == 0
    assert expanded_key.untyped_storage().data_ptr() == key.untyped_storage().data_ptr()
    assert expanded_value.untyped_storage().data_ptr() == value.untyped_storage().data_ptr()


@pytest.mark.parametrize("candidate_count", [2, 4, 8, 16])
def test_shared_context_matches_flat_reference(candidate_count: int) -> None:
    tokenizer = FakeTokenizer()
    reference_model = CacheAwareAccumulatingModel()
    shared_model = CacheAwareAccumulatingModel()
    reference = FlatReferenceBackend(
        model=reference_model,
        tokenizer=tokenizer,
        head=make_head(),
        device="cpu",
        dtype=torch.float32,
        substrate_revision="test-substrate",
        tokenizer_revision="test-tokenizer",
    )
    shared = SharedContextBackend(
        model=shared_model,
        tokenizer=tokenizer,
        head=make_head(),
        device="cpu",
        dtype=torch.float32,
        substrate_revision="test-substrate",
        tokenizer_revision="test-tokenizer",
    )
    request = make_request(candidate_count)

    reference_result = reference.score((request,))[0]
    shared_result = shared.score((request,))[0]
    metrics = compare_result(reference_result, shared_result)

    assert reference_model.calls == 1
    assert shared_model.calls == 2
    assert metrics.argmax_equal
    assert metrics.max_abs_logit_delta < 1e-6
    assert metrics.max_abs_probability_delta < 1e-6
    assert metrics.total_variation < 1e-6
    assert metrics.kl_reference_to_candidate < 1e-9
    assert reference_result.metadata["substrate_revision"] == "test-substrate"
    assert shared_result.metadata["substrate_revision"] == "test-substrate"


def test_shared_context_reports_reduced_logical_work() -> None:
    tokenizer = FakeTokenizer()
    shared = SharedContextBackend(
        model=CacheAwareAccumulatingModel(),
        tokenizer=tokenizer,
        head=make_head(),
        device="cpu",
        dtype=torch.float32,
    )
    request = make_request(candidate_count=16)

    result = shared.score((request,))[0]
    prefix_tokens = int(result.metadata["prefix_tokens"])
    candidate_counts = tuple(int(value) for value in result.metadata["candidate_token_counts"])
    expected_flat = sum(prefix_tokens + count for count in candidate_counts)
    expected_shared = prefix_tokens + sum(candidate_counts)

    assert result.metadata["flat_logical_token_positions"] == expected_flat
    assert result.metadata["shared_logical_token_positions"] == expected_shared
    assert result.metadata["logical_token_reduction_ratio"] == pytest.approx(
        expected_flat / expected_shared
    )
    assert result.metadata["cache_branching"] == "tensor-expand-view"
    assert result.metadata["prefix_compute_calls"] == 1
    assert result.metadata["continuation_compute_calls"] == 1
