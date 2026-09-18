# Athrub canonical attention runtime

## Purpose

Athrub A1 compares flat complete-path execution with shared-context execution on the same frozen A0 weights. The comparison is only meaningful when both paths use the same attention runtime. Kernel selection, grouped-query-attention (GQA) handling, request grouping, and precision are therefore controlled experimental variables rather than incidental implementation details.

## Canonical policy

The canonical A0/A1 runtime policy on the reference workstation is:

```text
backend: efficient_sdpa
gqa_strategy: expand_kv_heads
q_to_kv_ratio: 2
fallback_allowed: false
```

`src/athrub/attention_runtime.py` materializes GQA key/value heads with `repeat_interleave` before the SDPA call and pins the memory-efficient SDPA backend. Silent fallback is deliberately unsupported: if the requested backend cannot execute, the run must fail rather than switch kernels unnoticed.

Flat and shared-context executions used for A1 equivalence or performance claims must run under the same declared policy and record the effective runtime metadata.

## Frozen A0 latency caveat

The frozen A0 canonical **FP32 `p1024×k16` latency is WDDM shared-memory spill contaminated** on the RTX 3080 Ti reference workstation. Its decision outputs are valid and deterministic, but its latency must **not** be used as an A1 performance baseline or as evidence of flat-path throughput.

A1 performance measurements must establish a fresh controlled baseline under the tracked runtime and must distinguish dedicated-VRAM execution from WDDM spill.

## Request grouping discipline

Shape cells are executed individually. Mixing different sequence-length cells in one batch changes padding and may change BF16 reduction order in the memory-efficient kernel. A one-ULP shift can change argmax on near-tied synthetic cases even when probability deltas remain small.

Therefore the first A1 equivalence suite must keep request grouping and batch composition identical between flat and shared-context execution. Any later batching optimization is a separate experimental variable.

## Scope

This runtime policy does not change frozen A0 weights, tokenizer, codec, candidate ordering, decision head, or probability normalization semantics. It exists only to make the execution environment explicit and reproducible before A1 numerical-equivalence work begins.
