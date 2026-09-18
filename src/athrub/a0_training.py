"""Training primitives for creating a meaningful Athrub A0 reference.

A0 training is intentionally conventional. It exists only to produce a frozen decision
reference for A1 execution experiments; it is not the final Athrub architecture.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .reference import ScalarDecisionHead, TextDecisionCodec, TokenizerLike


@dataclass(frozen=True, slots=True)
class DecisionTrainingExample:
    """One supervised bounded-decision example."""

    request_id: str
    state: str
    question: str
    candidates: tuple[str, ...]
    target_distribution: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.question:
            raise ValueError("question must be non-empty")
        if len(self.candidates) < 2:
            raise ValueError("at least two candidates are required")
        if len(self.candidates) != len(self.target_distribution):
            raise ValueError("target_distribution must match candidate count")
        if any(not candidate for candidate in self.candidates):
            raise ValueError("candidates must be non-empty strings")
        if any(not math.isfinite(value) or value < 0.0 for value in self.target_distribution):
            raise ValueError("target probabilities must be finite and non-negative")
        total = sum(self.target_distribution)
        if not math.isclose(total, 1.0, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError("target_distribution must sum to one")


@dataclass(frozen=True, slots=True)
class DecisionMetrics:
    examples: int
    accuracy: float
    nll: float
    brier: float


@dataclass(frozen=True, slots=True)
class PreparedDecisionBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    path_lengths: tuple[int, ...]
    group_ranges: tuple[tuple[int, int], ...]
    targets: tuple[torch.Tensor, ...]


def _normalize_target(record: dict[str, Any], candidate_count: int) -> tuple[float, ...]:
    has_label = "label" in record
    has_distribution = "target_distribution" in record
    if has_label == has_distribution:
        raise ValueError("each training record must contain exactly one of label or target_distribution")

    if has_label:
        label = int(record["label"])
        if not 0 <= label < candidate_count:
            raise ValueError("label is outside candidate range")
        return tuple(1.0 if index == label else 0.0 for index in range(candidate_count))

    values = tuple(float(value) for value in record["target_distribution"])
    if len(values) != candidate_count:
        raise ValueError("target_distribution must match candidate count")
    return values


def example_from_mapping(record: dict[str, Any]) -> DecisionTrainingExample:
    candidates = tuple(str(value) for value in record["candidates"])
    target = _normalize_target(record, len(candidates))
    return DecisionTrainingExample(
        request_id=str(record["request_id"]),
        state=str(record.get("state", "")),
        question=str(record["question"]),
        candidates=candidates,
        target_distribution=target,
    )


def load_decision_jsonl(path: str | Path) -> list[DecisionTrainingExample]:
    """Load Athrub bounded-decision supervision from JSONL."""

    source = Path(path)
    examples: list[DecisionTrainingExample] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                record = json.loads(stripped)
                if not isinstance(record, dict):
                    raise TypeError("record must be a JSON object")
                examples.append(example_from_mapping(record))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"invalid decision record at {source}:{line_number}: {exc}") from exc
    if not examples:
        raise ValueError(f"decision dataset is empty: {source}")
    return examples


def _encode_prefix(
    example: DecisionTrainingExample,
    tokenizer: TokenizerLike,
    codec: TextDecisionCodec,
    *,
    add_bos: bool,
) -> list[int]:
    class _RequestView:
        state = example.state
        question = example.question

    token_ids = tokenizer.encode(codec.prefix_text(_RequestView()), add_special_tokens=False)  # type: ignore[arg-type]
    bos_token_id = getattr(tokenizer, "bos_token_id", None)
    if add_bos and bos_token_id is not None:
        token_ids = [int(bos_token_id), *token_ids]
    if not token_ids:
        raise ValueError("tokenized training prefix is empty")
    return [int(value) for value in token_ids]


def _encode_candidate(candidate: str, tokenizer: TokenizerLike, codec: TextDecisionCodec) -> list[int]:
    token_ids = tokenizer.encode(codec.candidate_text(candidate), add_special_tokens=False)
    if not token_ids:
        raise ValueError("tokenized training candidate is empty")
    return [int(value) for value in token_ids]


def _pad_token_id(tokenizer: TokenizerLike) -> int:
    for attribute in ("pad_token_id", "eos_token_id", "bos_token_id"):
        token_id = getattr(tokenizer, attribute, None)
        if token_id is not None:
            return int(token_id)
    return 0


def prepare_decision_batch(
    examples: Sequence[DecisionTrainingExample],
    tokenizer: TokenizerLike,
    *,
    device: str | torch.device,
    codec: TextDecisionCodec | None = None,
    add_bos: bool = True,
) -> PreparedDecisionBatch:
    """Flatten variable-cardinality decisions into complete candidate paths."""

    if not examples:
        raise ValueError("training batch must not be empty")
    codec = codec or TextDecisionCodec()
    paths: list[list[int]] = []
    ranges: list[tuple[int, int]] = []
    targets: list[torch.Tensor] = []

    for example in examples:
        prefix = _encode_prefix(example, tokenizer, codec, add_bos=add_bos)
        start = len(paths)
        for candidate in example.candidates:
            paths.append([*prefix, *_encode_candidate(candidate, tokenizer, codec)])
        ranges.append((start, len(paths)))
        targets.append(torch.tensor(example.target_distribution, dtype=torch.float32, device=device))

    max_length = max(len(path) for path in paths)
    input_ids = torch.full(
        (len(paths), max_length),
        fill_value=_pad_token_id(tokenizer),
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    lengths: list[int] = []
    for row, path in enumerate(paths):
        length = len(path)
        lengths.append(length)
        input_ids[row, :length] = torch.tensor(path, dtype=torch.long, device=device)
        attention_mask[row, :length] = 1

    return PreparedDecisionBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        path_lengths=tuple(lengths),
        group_ranges=tuple(ranges),
        targets=tuple(targets),
    )


def _last_hidden_state(output: Any) -> torch.Tensor:
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is None and isinstance(output, (tuple, list)) and output:
        hidden = output[0]
    if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
        raise TypeError("substrate output must provide a rank-3 last_hidden_state tensor")
    return hidden


def decision_logits(
    model: nn.Module,
    head: nn.Module,
    batch: PreparedDecisionBatch,
) -> tuple[torch.Tensor, ...]:
    """Return one logit vector per original decision request."""

    output = model(
        input_ids=batch.input_ids,
        attention_mask=batch.attention_mask,
        return_dict=True,
    )
    hidden = _last_hidden_state(output)
    row_indices = torch.arange(hidden.shape[0], device=hidden.device)
    last_indices = torch.tensor(batch.path_lengths, dtype=torch.long, device=hidden.device) - 1
    scores = head(hidden[row_indices, last_indices])
    if not isinstance(scores, torch.Tensor):
        raise TypeError("decision head must return a tensor")
    if scores.ndim == 2 and scores.shape[-1] == 1:
        scores = scores.squeeze(-1)
    if scores.ndim != 1 or scores.shape[0] != len(batch.path_lengths):
        raise ValueError("decision head must return one scalar per candidate path")
    return tuple(scores[start:end] for start, end in batch.group_ranges)


def decision_loss(logits: Sequence[torch.Tensor], targets: Sequence[torch.Tensor]) -> torch.Tensor:
    if len(logits) != len(targets) or not logits:
        raise ValueError("logits and targets must contain the same non-zero number of decisions")
    losses: list[torch.Tensor] = []
    for decision_logits_, target in zip(logits, targets, strict=True):
        if decision_logits_.shape != target.shape:
            raise ValueError("logit and target shapes differ")
        losses.append(-(target * torch.log_softmax(decision_logits_.float(), dim=0)).sum())
    return torch.stack(losses).mean()


def metrics_from_logits(
    logits: Sequence[torch.Tensor], targets: Sequence[torch.Tensor]
) -> DecisionMetrics:
    if len(logits) != len(targets) or not logits:
        raise ValueError("logits and targets must contain the same non-zero number of decisions")

    correct = 0
    nll = 0.0
    brier = 0.0
    for decision_logits_, target in zip(logits, targets, strict=True):
        probabilities = torch.softmax(decision_logits_.float(), dim=0)
        correct += int(torch.argmax(probabilities).item() == torch.argmax(target).item())
        nll += float(-(target * torch.log_softmax(decision_logits_.float(), dim=0)).sum().item())
        brier += float(torch.sum((probabilities - target) ** 2).item())
    count = len(logits)
    return DecisionMetrics(
        examples=count,
        accuracy=correct / count,
        nll=nll / count,
        brier=brier / count,
    )


def merge_metrics(rows: Iterable[DecisionMetrics]) -> DecisionMetrics:
    metrics = list(rows)
    total = sum(row.examples for row in metrics)
    if total < 1:
        raise ValueError("cannot merge empty metrics")
    return DecisionMetrics(
        examples=total,
        accuracy=sum(row.accuracy * row.examples for row in metrics) / total,
        nll=sum(row.nll * row.examples for row in metrics) / total,
        brier=sum(row.brier * row.examples for row in metrics) / total,
    )


def set_substrate_trainable(model: nn.Module, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(trainable)


def trainable_parameter_count(*modules: nn.Module) -> int:
    return sum(
        parameter.numel()
        for module in modules
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def run_decision_epoch(
    *,
    model: nn.Module,
    head: ScalarDecisionHead,
    tokenizer: TokenizerLike,
    examples: Sequence[DecisionTrainingExample],
    batch_size: int,
    device: str | torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    codec: TextDecisionCodec | None = None,
    add_bos: bool = True,
) -> DecisionMetrics:
    """Train or evaluate one deterministic pass over a decision dataset.

    Passing an optimizer enables training. Omitting it evaluates under inference mode.
    Dataset shuffling is intentionally owned by the caller so the exact order can be
    recorded in the A0 provenance bundle.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not examples:
        raise ValueError("examples must not be empty")

    training = optimizer is not None
    substrate_training = training and any(parameter.requires_grad for parameter in model.parameters())
    model.train(substrate_training)
    head.train(training)
    rows: list[DecisionMetrics] = []

    for offset in range(0, len(examples), batch_size):
        batch_examples = examples[offset : offset + batch_size]
        prepared = prepare_decision_batch(
            batch_examples,
            tokenizer,
            device=device,
            codec=codec,
            add_bos=add_bos,
        )
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
            logits = decision_logits(model, head, prepared)
            loss = decision_loss(logits, prepared.targets)
            loss.backward()
            optimizer.step()
            detached = tuple(value.detach() for value in logits)
        else:
            with torch.inference_mode():
                detached = decision_logits(model, head, prepared)
        rows.append(metrics_from_logits(detached, prepared.targets))

    return merge_metrics(rows)
