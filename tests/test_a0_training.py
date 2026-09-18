from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from athrub.a0_training import (
    DecisionTrainingExample,
    decision_logits,
    decision_loss,
    evaluate_by_family,
    example_from_mapping,
    load_decision_jsonl,
    metrics_from_logits,
    prepare_decision_batch,
    run_decision_epoch,
    set_substrate_trainable,
)
from athrub.reference import ScalarDecisionHead


class FakeTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        values: list[int] = []
        for token in text.split():
            digest = hashlib.sha256(token.encode()).digest()
            values.append(3 + int.from_bytes(digest[:2], "big") % 97)
        return values


class TinyTrainableModel(nn.Module):
    def __init__(self, hidden_size: int = 4) -> None:
        super().__init__()
        self.embedding = nn.Embedding(128, hidden_size)

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool,
    ) -> SimpleNamespace:
        assert return_dict is True
        embedded = self.embedding(input_ids) * attention_mask.unsqueeze(-1)
        hidden = torch.cumsum(embedded, dim=1)
        return SimpleNamespace(last_hidden_state=hidden)


def examples() -> list[DecisionTrainingExample]:
    return [
        DecisionTrainingExample(
            request_id="a",
            state="temperature is high",
            question="Which action reduces temperature?",
            candidates=("cool", "heat"),
            target_distribution=(1.0, 0.0),
            task_family="control",
        ),
        DecisionTrainingExample(
            request_id="b",
            state="service timeout is transient",
            question="What should happen next?",
            candidates=("retry", "delete", "ignore"),
            target_distribution=(1.0, 0.0, 0.0),
            task_family="operations",
        ),
    ]


def test_mapping_accepts_hard_label_and_soft_distribution() -> None:
    hard = example_from_mapping(
        {
            "request_id": "hard",
            "state": "s",
            "question": "q",
            "candidates": ["a", "b", "c"],
            "label": 1,
            "task_family": "hard-family",
        }
    )
    soft = example_from_mapping(
        {
            "request_id": "soft",
            "question": "q",
            "candidates": ["a", "b"],
            "target_distribution": [0.75, 0.25],
        }
    )

    assert hard.target_distribution == (0.0, 1.0, 0.0)
    assert hard.task_family == "hard-family"
    assert soft.target_distribution == (0.75, 0.25)
    assert soft.task_family is None


def test_mapping_rejects_ambiguous_supervision() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        example_from_mapping(
            {
                "request_id": "bad",
                "question": "q",
                "candidates": ["a", "b"],
                "label": 0,
                "target_distribution": [1.0, 0.0],
            }
        )


def test_jsonl_loader_reports_invalid_line(tmp_path: Path) -> None:
    dataset = tmp_path / "data.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "request_id": "valid",
                "question": "q",
                "candidates": ["a", "b"],
                "label": 0,
            }
        )
        + "\n"
        + "not-json\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=":2:"):
        load_decision_jsonl(dataset)


def test_prepared_batch_preserves_variable_candidate_groups() -> None:
    prepared = prepare_decision_batch(examples(), FakeTokenizer(), device="cpu")

    assert prepared.input_ids.shape[0] == 5
    assert prepared.attention_mask.shape == prepared.input_ids.shape
    assert prepared.group_ranges == ((0, 2), (2, 5))
    assert tuple(target.shape[0] for target in prepared.targets) == (2, 3)


def test_loss_and_metrics_use_complete_distribution() -> None:
    logits = (torch.tensor([3.0, 0.0]), torch.tensor([0.0, 1.0, 2.0]))
    targets = (torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0]))

    loss = decision_loss(logits, targets)
    metrics = metrics_from_logits(logits, targets)

    assert loss.item() > 0.0
    assert metrics.examples == 2
    assert metrics.accuracy == 1.0
    assert metrics.nll == pytest.approx(loss.item())
    assert metrics.brier >= 0.0


def test_head_warmup_changes_head_but_not_frozen_substrate() -> None:
    torch.manual_seed(7)
    model = TinyTrainableModel()
    head = ScalarDecisionHead(4)
    set_substrate_trainable(model, False)
    before_model = [parameter.detach().clone() for parameter in model.parameters()]
    before_head = [parameter.detach().clone() for parameter in head.parameters()]
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.05)

    metrics = run_decision_epoch(
        model=model,
        head=head,
        tokenizer=FakeTokenizer(),
        examples=examples(),
        batch_size=2,
        device="cpu",
        optimizer=optimizer,
    )

    assert metrics.examples == 2
    assert model.training is False
    assert all(
        torch.equal(before, after.detach())
        for before, after in zip(before_model, model.parameters(), strict=True)
    )
    assert any(
        not torch.equal(before, after.detach())
        for before, after in zip(before_head, head.parameters(), strict=True)
    )


def test_decision_logits_preserve_request_grouping() -> None:
    torch.manual_seed(11)
    model = TinyTrainableModel()
    head = ScalarDecisionHead(4)
    prepared = prepare_decision_batch(examples(), FakeTokenizer(), device="cpu")

    logits = decision_logits(model, head, prepared)

    assert len(logits) == 2
    assert logits[0].shape == (2,)
    assert logits[1].shape == (3,)


def test_family_evaluation_reports_each_family() -> None:
    torch.manual_seed(13)
    model = TinyTrainableModel()
    head = ScalarDecisionHead(4)

    metrics = evaluate_by_family(
        model=model,
        head=head,
        tokenizer=FakeTokenizer(),
        examples=examples(),
        batch_size=2,
        device="cpu",
    )

    assert set(metrics) == {"control", "operations"}
    assert metrics["control"].examples == 1
    assert metrics["operations"].examples == 1
