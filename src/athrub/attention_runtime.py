"""Explicit, fail-loud attention runtime policy for Athrub reference execution.

The frozen A0 benchmark semantics were produced under one exact attention runtime:
memory-efficient SDPA with GQA key/value-head expansion. Two failure modes make
that policy load-bearing rather than cosmetic:

1. This torch Windows wheel ships no flash attention and its memory-efficient
   kernel rejects dense GQA inputs, so unpinned execution silently falls back to
   the math backend, which materializes full L^2 attention and exhausts GPU
   memory on larger FP32 shape cells.
2. Even when memory suffices, a silent kernel change silently changes numerics,
   turning a controlled comparison into a different experiment.

``attention_runtime`` therefore pins the backend and installs the declared GQA
strategy, and fails loudly when the requested policy cannot execute. Silent
fallback is not representable: ``fallback_allowed=True`` is rejected at policy
construction time.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

_SUPPORTED_BACKENDS: dict[str, SDPBackend] = {
    "efficient_sdpa": SDPBackend.EFFICIENT_ATTENTION,
}
_SUPPORTED_GQA_STRATEGIES = {"expand_kv_heads", "none"}


@dataclass(frozen=True, slots=True)
class AttentionPolicy:
    """Declared attention execution contract for canonical Athrub benchmarks."""

    backend: str = "efficient_sdpa"
    gqa_strategy: str = "expand_kv_heads"
    q_to_kv_ratio: int | None = None
    fallback_allowed: bool = False

    def __post_init__(self) -> None:
        if self.backend not in _SUPPORTED_BACKENDS:
            raise ValueError(f"unsupported attention backend: {self.backend}")
        if self.gqa_strategy not in _SUPPORTED_GQA_STRATEGIES:
            raise ValueError(f"unsupported GQA strategy: {self.gqa_strategy}")
        if self.q_to_kv_ratio is not None and self.q_to_kv_ratio < 1:
            raise ValueError("q_to_kv_ratio must be positive")
        if self.fallback_allowed:
            raise ValueError(
                "silent attention fallback is not supported: canonical Athrub execution "
                "must fail loudly when the requested policy cannot execute"
            )

    def as_record(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "gqa_strategy": self.gqa_strategy,
            "q_to_kv_ratio": self.q_to_kv_ratio,
            "fallback_allowed": self.fallback_allowed,
        }


CANONICAL_A0_POLICY = AttentionPolicy(
    backend="efficient_sdpa",
    gqa_strategy="expand_kv_heads",
    q_to_kv_ratio=2,
    fallback_allowed=False,
)


@dataclass
class RuntimeSession:
    """Observed state of one attention-runtime installation."""

    policy: AttentionPolicy
    query_heads: int | None = None
    kv_heads: int | None = None
    expansion_calls: int = 0
    last_expansion_ratio: int | None = None
    _expansion_ratio_history: set[int] = field(default_factory=set)

    def observe_model_config(self, config: Any) -> None:
        """Record and validate head counts against the declared policy."""

        query_heads = getattr(config, "num_attention_heads", None)
        kv_heads = getattr(config, "num_key_value_heads", None)
        if query_heads is None or kv_heads is None:
            raise ValueError("model config must expose num_attention_heads and num_key_value_heads")
        query_heads = int(query_heads)
        kv_heads = int(kv_heads)
        if self.policy.q_to_kv_ratio is not None:
            actual = query_heads // kv_heads
            if query_heads % kv_heads != 0 or actual != self.policy.q_to_kv_ratio:
                raise ValueError(
                    f"model GQA ratio {query_heads}:{kv_heads} contradicts declared "
                    f"q_to_kv_ratio {self.policy.q_to_kv_ratio}"
                )
        self.query_heads = query_heads
        self.kv_heads = kv_heads

    def record_expansion(self, ratio: int) -> None:
        self.expansion_calls += 1
        self.last_expansion_ratio = ratio
        self._expansion_ratio_history.add(ratio)

    def metadata(self) -> dict[str, Any]:
        return {
            "requested_attention_policy": self.policy.as_record(),
            "effective_attention_policy": self.policy.as_record(),
            "query_head_count": self.query_heads,
            "kv_head_count": self.kv_heads,
            "gqa_expansion_ratio": self.last_expansion_ratio,
            "gqa_expansion_calls": self.expansion_calls,
            "gqa_expansion_ratios_observed": sorted(self._expansion_ratio_history),
        }


def expand_kv_heads(key: torch.Tensor, value: torch.Tensor, ratio: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize grouped query attention by repeating key/value heads.

    ``ratio`` consecutive query heads share one key/value head, matching the
    standard ``repeat_interleave`` GQA materialization used by attention kernels.
    """

    if ratio < 1:
        raise ValueError("expansion ratio must be positive")
    return key.repeat_interleave(ratio, dim=1), value.repeat_interleave(ratio, dim=1)


@contextmanager
def attention_runtime(policy: AttentionPolicy) -> Iterator[RuntimeSession]:
    """Install one attention policy for the duration of the context.

    Installs a wrapper around ``torch.nn.functional.scaled_dot_product_attention``
    that applies the declared GQA strategy, and pins the declared SDPA backend.
    With ``fallback_allowed=False`` (the only supported value), an input the
    pinned backend cannot execute raises ``RuntimeError`` instead of silently
    falling back to the math kernel.
    """

    if not torch.cuda.is_available():
        raise RuntimeError("attention_runtime requires CUDA; canonical execution is CUDA-defined")
    session = RuntimeSession(policy=policy)
    original_sdpa = F.scaled_dot_product_attention

    def expanding_sdpa(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if (
            policy.gqa_strategy == "expand_kv_heads"
            and query.ndim == 4
            and key.ndim == 4
            and query.shape[1] != key.shape[1]
            and query.shape[1] % key.shape[1] == 0
        ):
            ratio = query.shape[1] // key.shape[1]
            key, value = expand_kv_heads(key, value, ratio)
            session.record_expansion(ratio)
        return original_sdpa(query, key, value, *args, **kwargs)

    F.scaled_dot_product_attention = expanding_sdpa
    try:
        with sdpa_kernel([_SUPPORTED_BACKENDS[policy.backend]]):
            yield session
    finally:
        F.scaled_dot_product_attention = original_sdpa
