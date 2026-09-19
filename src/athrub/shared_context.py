"""Shared-context inference for Athrub Phase 1.

This backend computes the state/question prefix once per decision request, branches the
resulting causal-attention cache across candidates using tensor views, and processes all
candidate suffixes in one continuation batch.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import torch

from .contracts import DecisionRequest, DecisionResult
from .reference import FlatReferenceBackend


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def _to_legacy_cache(cache: Any) -> tuple[Any, ...]:
    """Return an immutable tuple cache suitable for view-based batch expansion.

    Current model libraries may return a cache object instead of the historical nested
    tuple. When the object exposes ``to_legacy_cache`` we convert it explicitly. Athrub
    keeps this compatibility boundary isolated so later cache-native implementations can
    replace it without changing the decision contract.
    """

    if isinstance(cache, tuple):
        return cache
    converter = getattr(cache, "to_legacy_cache", None)
    if callable(converter):
        converted = converter()
        if isinstance(converted, tuple):
            return converted
    raise TypeError(
        "shared-context inference requires a tuple cache or an object exposing "
        "to_legacy_cache()"
    )


def _from_legacy_cache(cache: tuple[Any, ...]) -> Any:
    """Convert a branched legacy tuple cache back into the model's expected cache object.

    Complements ``_to_legacy_cache``: branching happens on the immutable tuple
    representation, while current model libraries require a cache object exposing
    ``get_seq_length`` at the continuation boundary. When the cache library is not
    installed (dependency-free tests), the tuple is passed through unchanged.
    """

    try:
        from transformers import DynamicCache
    except ImportError:
        return cache
    return DynamicCache.from_legacy_cache(cache)


def _expand_cache_value(value: Any, batch_size: int) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return value
        if value.shape[0] != 1:
            raise ValueError(
                "view-based cache branching expects a single-request cache with batch dimension 1"
            )
        return value.expand(batch_size, *value.shape[1:])
    if isinstance(value, tuple):
        return tuple(_expand_cache_value(item, batch_size) for item in value)
    if isinstance(value, list):
        return [_expand_cache_value(item, batch_size) for item in value]
    if value is None:
        return None
    raise TypeError(f"unsupported cache value type: {type(value)!r}")


def expand_legacy_cache(cache: tuple[Any, ...], batch_size: int) -> tuple[Any, ...]:
    """Branch a single-request legacy cache without explicit tensor copies.

    ``Tensor.expand`` creates zero-stride views. Downstream kernels may choose to
    materialize contiguous storage internally, but Athrub itself does not duplicate the
    prefix cache at this boundary.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    return tuple(_expand_cache_value(layer, batch_size) for layer in cache)


class SharedContextBackend(FlatReferenceBackend):
    """Phase 1 prototype that reuses one computed prefix across all candidates.

    The first implementation intentionally handles requests independently: one prefix
    call and one packed continuation call per request. This isolates the shared-context
    hypothesis. Cross-request prefix packing is a separate optimization and must not be
    conflated with the Phase 1 equivalence experiment.
    """

    def __init__(self, *args: Any, profile_stages: bool = False, **kwargs: Any) -> None:
        kwargs.setdefault("name", "athrub-shared-context")
        super().__init__(*args, **kwargs)
        self.profile_stages = profile_stages

    def _timed_forward(self, **kwargs: Any) -> tuple[Any, float | None]:
        if not self.profile_stages:
            return self.model(**kwargs), None
        _synchronize(self.device)
        started = time.perf_counter_ns()
        output = self.model(**kwargs)
        _synchronize(self.device)
        elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
        return output, elapsed_ms

    def _score_request(self, request: DecisionRequest) -> DecisionResult:
        prefix_ids = self._encode_prefix(request)
        candidate_ids = [self._encode_candidate(candidate) for candidate in request.candidates]
        candidate_counts = tuple(len(tokens) for tokens in candidate_ids)
        candidate_count = len(candidate_ids)

        prefix_tensor = torch.tensor([prefix_ids], dtype=torch.long, device=self.device)
        prefix_mask = torch.ones_like(prefix_tensor)

        with torch.inference_mode():
            prefix_output, prefix_ms = self._timed_forward(
                input_ids=prefix_tensor,
                attention_mask=prefix_mask,
                use_cache=True,
                return_dict=True,
            )
            raw_cache = getattr(prefix_output, "past_key_values", None)
            if raw_cache is None:
                raise TypeError("model did not return past_key_values for the shared prefix")
            prefix_cache = _to_legacy_cache(raw_cache)
            branched_cache = expand_legacy_cache(prefix_cache, candidate_count)

            max_candidate_length = max(candidate_counts)
            suffix_tensor = torch.full(
                (candidate_count, max_candidate_length),
                fill_value=self._pad_token_id(),
                dtype=torch.long,
                device=self.device,
            )
            suffix_mask = torch.zeros_like(suffix_tensor)
            for row, tokens in enumerate(candidate_ids):
                length = len(tokens)
                suffix_tensor[row, :length] = torch.tensor(
                    tokens, dtype=torch.long, device=self.device
                )
                suffix_mask[row, :length] = 1

            shared_mask = torch.ones(
                (candidate_count, len(prefix_ids)),
                dtype=prefix_mask.dtype,
                device=self.device,
            )
            full_attention_mask = torch.cat((shared_mask, suffix_mask), dim=1)
            position_ids = torch.arange(
                len(prefix_ids),
                len(prefix_ids) + max_candidate_length,
                dtype=torch.long,
                device=self.device,
            ).unsqueeze(0).expand(candidate_count, -1)

            continuation_output, continuation_ms = self._timed_forward(
                input_ids=suffix_tensor,
                attention_mask=full_attention_mask,
                position_ids=position_ids,
                past_key_values=_from_legacy_cache(branched_cache),
                use_cache=False,
                return_dict=True,
            )
            hidden = self._last_hidden_state(continuation_output)
            row_indices = torch.arange(candidate_count, device=hidden.device)
            last_indices = torch.tensor(
                candidate_counts, dtype=torch.long, device=hidden.device
            ) - 1
            final_hidden = hidden[row_indices, last_indices]
            scores = self.head(final_hidden)

        if not isinstance(scores, torch.Tensor):
            raise TypeError("decision head must return a tensor")
        if scores.ndim == 2 and scores.shape[-1] == 1:
            scores = scores.squeeze(-1)
        if scores.ndim != 1 or scores.shape[0] != candidate_count:
            raise ValueError("decision head must return one scalar per candidate path")

        logits = scores.float()
        probabilities = torch.softmax(logits, dim=0)
        prefix_count = len(prefix_ids)
        path_counts = tuple(prefix_count + count for count in candidate_counts)
        flat_positions = sum(path_counts)
        shared_positions = prefix_count + sum(candidate_counts)
        metadata: dict[str, object] = {
            "prefix_tokens": prefix_count,
            "candidate_token_counts": candidate_counts,
            "path_token_counts": path_counts,
            "flat_logical_token_positions": flat_positions,
            "shared_logical_token_positions": shared_positions,
            "logical_token_reduction_ratio": flat_positions / shared_positions,
            "prefix_compute_calls": 1,
            "continuation_compute_calls": 1,
            "cache_branching": "tensor-expand-view",
            "substrate_revision": self.substrate_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "precision": str(
                self.dtype or next(self.model.parameters(), torch.empty(0)).dtype
            ),
        }
        if prefix_ms is not None:
            metadata["prefix_latency_ms"] = prefix_ms
        if continuation_ms is not None:
            metadata["continuation_latency_ms"] = continuation_ms

        return DecisionResult(
            request_id=request.request_id,
            logits=tuple(float(value) for value in logits.cpu()),
            probabilities=tuple(float(value) for value in probabilities.cpu()),
            predicted_index=int(torch.argmax(probabilities).item()),
            metadata=metadata,
        )

    def score(self, requests: Sequence[DecisionRequest]) -> Sequence[DecisionResult]:
        return [self._score_request(request) for request in requests]
