from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from athrub.benchmark import benchmark_batch
from athrub.contracts import DecisionRequest
from athrub.reference import FlatReferenceBackend, ScalarDecisionHead, TextDecisionCodec


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


class ContextAccumulatingModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool,
    ) -> SimpleNamespace:
        assert return_dict is True
        self.calls += 1
        masked = input_ids.float() * attention_mask.float()
        cumulative = torch.cumsum(masked, dim=1)
        hidden = torch.stack(
            (
                cumulative,
                cumulative * 0.1,
                cumulative * 0.01,
                attention_mask.float().cumsum(dim=1),
            ),
            dim=-1,
        )
        return SimpleNamespace(last_hidden_state=hidden)


def make_backend() -> tuple[FlatReferenceBackend, ContextAccumulatingModel]:
    model = ContextAccumulatingModel()
    head = ScalarDecisionHead(4, bias=False)
    with torch.no_grad():
        head.projection.weight.copy_(torch.tensor([[0.01, 0.02, -0.03, 0.04]]))
    return (
        FlatReferenceBackend(
            model=model,
            tokenizer=FakeTokenizer(),
            head=head,
            device="cpu",
            dtype=torch.float32,
            substrate_revision="substrate-revision-test",
            tokenizer_revision="tokenizer-revision-test",
        ),
        model,
    )


def test_codec_keeps_candidate_out_of_shared_prefix() -> None:
    codec = TextDecisionCodec()
    request = DecisionRequest(
        request_id="codec",
        state="shared state",
        question="shared question?",
        candidates=("candidate alpha", "candidate beta"),
    )

    prefix = codec.prefix_text(request)
    assert "candidate alpha" not in prefix
    assert "candidate beta" not in prefix
    assert codec.candidate_text("candidate alpha").startswith("Candidate:\n")
    assert codec.candidate_text("candidate alpha").endswith("Decision:\n")


def test_flat_backend_preserves_request_and_candidate_order_in_one_model_call() -> None:
    backend, model = make_backend()
    request_a = DecisionRequest(
        request_id="a",
        state="system state alpha",
        question="Which option?",
        candidates=("first option", "second option", "third option"),
    )
    request_b = DecisionRequest(
        request_id="b",
        state="system state beta",
        question="Which option?",
        candidates=("third option", "first option"),
    )

    results = backend.score((request_a, request_b))

    assert model.calls == 1
    assert [result.request_id for result in results] == ["a", "b"]
    assert [len(result.logits) for result in results] == [3, 2]
    for result in results:
        assert sum(result.probabilities) == pytest.approx(1.0)
        assert result.predicted_index == max(
            range(len(result.probabilities)), key=result.probabilities.__getitem__
        )


def test_candidate_reordering_only_reorders_corresponding_scores() -> None:
    backend, _ = make_backend()
    forward = DecisionRequest(
        request_id="forward",
        state="same state",
        question="same question",
        candidates=("alpha", "beta", "gamma"),
    )
    reverse = DecisionRequest(
        request_id="reverse",
        state="same state",
        question="same question",
        candidates=("gamma", "beta", "alpha"),
    )

    forward_result, reverse_result = backend.score((forward, reverse))

    assert reverse_result.logits == pytest.approx(tuple(reversed(forward_result.logits)))
    assert reverse_result.probabilities == pytest.approx(tuple(reversed(forward_result.probabilities)))


def test_reference_backend_emits_exact_token_accounting() -> None:
    backend, _ = make_backend()
    request = DecisionRequest(
        request_id="tokens",
        state="one two three four",
        question="Which candidate is preferred?",
        candidates=("alpha candidate", "beta", "gamma candidate with detail"),
        metadata={"prefix_units": 999, "candidate_units": 999},
    )

    result = backend.score((request,))[0]
    candidate_counts = result.metadata["candidate_token_counts"]
    path_counts = result.metadata["path_token_counts"]

    assert isinstance(candidate_counts, tuple)
    assert isinstance(path_counts, tuple)
    assert len(candidate_counts) == len(request.candidates)
    assert len(path_counts) == len(request.candidates)
    assert result.metadata["prefix_tokens"] > 0
    assert result.metadata["flat_logical_token_positions"] == sum(path_counts)
    assert result.metadata["substrate_revision"] == "substrate-revision-test"
    assert result.metadata["tokenizer_revision"] == "tokenizer-revision-test"


def test_benchmark_prefers_exact_backend_token_counts() -> None:
    backend, _ = make_backend()
    request = DecisionRequest(
        request_id="benchmark-tokens",
        state="state text",
        question="Question?",
        candidates=("alpha", "beta"),
        metadata={"prefix_units": 999, "candidate_units": 999},
    )

    result = backend.score((request,))[0]
    records = benchmark_batch(backend, (request,), warmups=0, repeats=1)
    record = records[0]

    assert record.prefix_units == result.metadata["prefix_tokens"]
    assert record.candidate_units == sum(result.metadata["candidate_token_counts"])
    assert record.metadata["candidate_token_counts"] == result.metadata["candidate_token_counts"]
    assert record.metadata["flat_logical_token_positions"] == result.metadata[
        "flat_logical_token_positions"
    ]
    assert record.metadata["substrate_revision"] == "substrate-revision-test"
