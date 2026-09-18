"""Canonical flat candidate-path backend for Athrub Phase 1.

The reference backend deliberately recomputes the shared context for every candidate.
Phase 1 optimized backends must preserve its candidate ordering, logits, and probability
semantics while changing only the execution strategy.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
from torch import nn

from .contracts import DecisionRequest, DecisionResult


class TokenizerLike(Protocol):
    """Tokenizer surface required by the reference backend."""

    bos_token_id: int | None
    eos_token_id: int | None
    pad_token_id: int | None

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]: ...


@dataclass(frozen=True, slots=True)
class TextDecisionCodec:
    """Stable text serialization for Phase 1 bounded decisions.

    The prefix ends immediately before the candidate-specific region. Shared-context
    implementations must use the same serialization so numerical comparisons isolate
    computation rather than prompt-format changes.
    """

    state_label: str = "State"
    question_label: str = "Question"
    candidate_label: str = "Candidate"
    decision_label: str = "Decision"

    def prefix_text(self, request: DecisionRequest) -> str:
        return (
            f"{self.state_label}:\n{request.state}\n"
            f"{self.question_label}:\n{request.question}\n"
        )

    def candidate_text(self, candidate: str) -> str:
        return (
            f"{self.candidate_label}:\n{candidate}\n"
            f"{self.decision_label}:\n"
        )


class ScalarDecisionHead(nn.Module):
    """Minimal scalar scoring head applied to the final candidate-path state."""

    def __init__(self, hidden_size: int, *, bias: bool = True) -> None:
        super().__init__()
        self.projection = nn.Linear(hidden_size, 1, bias=bias)

    def forward(self, hidden_state: torch.Tensor) -> torch.Tensor:
        return self.projection(hidden_state).squeeze(-1)


class FlatReferenceBackend:
    """Canonical Phase 1 backend using complete candidate paths in one flat batch.

    All candidate paths repeat the full state/question prefix. This is intentional:
    the implementation is the correctness baseline against which shared-context
    execution will be compared.
    """

    def __init__(
        self,
        *,
        model: nn.Module,
        tokenizer: TokenizerLike,
        head: nn.Module,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        codec: TextDecisionCodec | None = None,
        add_bos: bool = True,
        model_revision: str | None = None,
        tokenizer_revision: str | None = None,
        name: str = "athrub-reference-flat",
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.head = head
        self.codec = codec or TextDecisionCodec()
        self.add_bos = add_bos
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self._name = name

        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self.device = torch.device(device)
        self.dtype = dtype

        self.model.to(self.device)
        self.head.to(self.device)
        if dtype is not None:
            self.model.to(dtype=dtype)
            self.head.to(dtype=dtype)
        self.model.eval()
        self.head.eval()

    @property
    def name(self) -> str:
        return self._name

    def _encode_prefix(self, request: DecisionRequest) -> list[int]:
        token_ids = self.tokenizer.encode(
            self.codec.prefix_text(request), add_special_tokens=False
        )
        bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
        if self.add_bos and bos_token_id is not None:
            token_ids = [int(bos_token_id), *token_ids]
        if not token_ids:
            raise ValueError("tokenized decision prefix is empty")
        return [int(token_id) for token_id in token_ids]

    def _encode_candidate(self, candidate: str) -> list[int]:
        token_ids = self.tokenizer.encode(
            self.codec.candidate_text(candidate), add_special_tokens=False
        )
        if not token_ids:
            raise ValueError("tokenized candidate suffix is empty")
        return [int(token_id) for token_id in token_ids]

    def _pad_token_id(self) -> int:
        for attribute in ("pad_token_id", "eos_token_id", "bos_token_id"):
            token_id = getattr(self.tokenizer, attribute, None)
            if token_id is not None:
                return int(token_id)
        return 0

    @staticmethod
    def _last_hidden_state(output: Any) -> torch.Tensor:
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None and isinstance(output, (tuple, list)) and output:
            hidden = output[0]
        if not isinstance(hidden, torch.Tensor) or hidden.ndim != 3:
            raise TypeError("model output must provide a rank-3 last_hidden_state tensor")
        return hidden

    def score(self, requests: Sequence[DecisionRequest]) -> Sequence[DecisionResult]:
        if not requests:
            return []

        paths: list[list[int]] = []
        groups: list[tuple[DecisionRequest, int, int, int, tuple[int, ...]]] = []

        for request in requests:
            prefix_ids = self._encode_prefix(request)
            candidate_token_counts: list[int] = []
            start = len(paths)
            for candidate in request.candidates:
                candidate_ids = self._encode_candidate(candidate)
                candidate_token_counts.append(len(candidate_ids))
                paths.append([*prefix_ids, *candidate_ids])
            groups.append(
                (
                    request,
                    start,
                    len(paths),
                    len(prefix_ids),
                    tuple(candidate_token_counts),
                )
            )

        max_length = max(len(path) for path in paths)
        pad_token_id = self._pad_token_id()
        input_ids = torch.full(
            (len(paths), max_length),
            fill_value=pad_token_id,
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.zeros_like(input_ids)
        lengths: list[int] = []
        for row, path in enumerate(paths):
            length = len(path)
            lengths.append(length)
            input_ids[row, :length] = torch.tensor(path, dtype=torch.long, device=self.device)
            attention_mask[row, :length] = 1

        with torch.inference_mode():
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
            )
            hidden = self._last_hidden_state(output)
            row_indices = torch.arange(hidden.shape[0], device=hidden.device)
            last_indices = torch.tensor(lengths, device=hidden.device, dtype=torch.long) - 1
            final_hidden = hidden[row_indices, last_indices]
            scores = self.head(final_hidden)

        if not isinstance(scores, torch.Tensor):
            raise TypeError("decision head must return a tensor")
        if scores.ndim == 2 and scores.shape[-1] == 1:
            scores = scores.squeeze(-1)
        if scores.ndim != 1 or scores.shape[0] != len(paths):
            raise ValueError("decision head must return one scalar per candidate path")

        scores = scores.float()
        results: list[DecisionResult] = []
        precision = str(self.dtype or next(self.model.parameters(), torch.empty(0)).dtype)
        for request, start, end, prefix_count, candidate_counts in groups:
            logits = scores[start:end]
            probabilities = torch.softmax(logits, dim=0)
            path_counts = tuple(prefix_count + count for count in candidate_counts)
            results.append(
                DecisionResult(
                    request_id=request.request_id,
                    logits=tuple(float(value) for value in logits.cpu()),
                    probabilities=tuple(float(value) for value in probabilities.cpu()),
                    predicted_index=int(torch.argmax(probabilities).item()),
                    metadata={
                        "prefix_tokens": prefix_count,
                        "candidate_token_counts": candidate_counts,
                        "path_token_counts": path_counts,
                        "flat_logical_token_positions": sum(path_counts),
                        "model_revision": self.model_revision,
                        "tokenizer_revision": self.tokenizer_revision,
                        "precision": precision,
                    },
                )
            )
        return results

    @classmethod
    def from_pretrained(
        cls,
        *,
        model_id: str,
        head_path: str | Path,
        revision: str,
        tokenizer_id: str | None = None,
        tokenizer_revision: str | None = None,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
        head_bias: bool = True,
        add_bos: bool = True,
    ) -> "FlatReferenceBackend":
        """Load a reproducible reference backbone and trained scalar decision head.

        ``revision`` is required rather than silently tracking a mutable default branch.
        ``head_path`` must contain a PyTorch state dict for :class:`ScalarDecisionHead`.
        The optional ``reference`` dependency group installs the model loader.
        """

        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise RuntimeError(
                "from_pretrained requires the 'reference' optional dependency"
            ) from exc

        tokenizer_source = tokenizer_id or model_id
        tokenizer_revision = tokenizer_revision or revision
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            revision=tokenizer_revision,
        )
        model = AutoModel.from_pretrained(
            model_id,
            revision=revision,
            torch_dtype=dtype,
        )

        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(model.config, "n_embd", None)
        if hidden_size is None:
            raise ValueError("unable to infer backbone hidden size")

        head = ScalarDecisionHead(int(hidden_size), bias=head_bias)
        state_dict = torch.load(Path(head_path), map_location="cpu", weights_only=True)
        if not isinstance(state_dict, dict):
            raise TypeError("decision-head checkpoint must contain a state dict")
        head.load_state_dict(state_dict, strict=True)

        return cls(
            model=model,
            tokenizer=tokenizer,
            head=head,
            device=device,
            dtype=dtype,
            add_bos=add_bos,
            model_revision=revision,
            tokenizer_revision=tokenizer_revision,
        )
