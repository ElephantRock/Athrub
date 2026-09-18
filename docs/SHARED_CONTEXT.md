# Athrub Phase 1 Shared-Context Inference

## Research question

Can Athrub compute common decision context once, reuse that computation across every candidate, and preserve the reference probability distribution?

## Execution model

The flat reference backend evaluates complete candidate paths:

```text
prefix + candidate A -> score A
prefix + candidate B -> score B
prefix + candidate C -> score C
```

The Phase 1 shared-context prototype evaluates:

```text
prefix -> causal cache
             |-- candidate A -> score A
             |-- candidate B -> score B
             `-- candidate C -> score C
```

The prefix and candidate text serialization are inherited from the reference backend. Only execution changes.

## Cache branching

The initial implementation converts the single-request prefix cache to the legacy nested-tensor representation when necessary and branches batch dimension 1 to candidate count `K` with `Tensor.expand`.

This creates zero-stride tensor views rather than explicit Athrub-side copies of the prefix cache. Downstream model kernels may still materialize contiguous buffers internally; GPU profiling is required before making physical-memory claims.

## Candidate packing

All candidate suffixes for one request are padded into one continuation batch. The model therefore performs:

1. one prefix forward pass;
2. one candidate-continuation forward pass.

This first prototype handles separate decision requests independently. Cross-request prefix packing is intentionally deferred so that the shared-context hypothesis is not confounded with unrelated batching optimizations.

## Logical work accounting

For prefix length `L_p` and candidate suffix lengths `L_c[i]`, flat execution processes the logical token positions:

```text
sum(L_p + L_c[i])
```

The shared execution target is:

```text
L_p + sum(L_c[i])
```

The backend records both quantities and their ratio. These counts are architecture-level work indicators, not direct wall-clock speedup estimates.

## Numerical equivalence

Every shared-context result must be compared with the flat reference result using complete logits and probability vectors.

Phase 1 targets:

- FP32 maximum absolute probability delta: `< 1e-5`
- BF16 maximum absolute probability delta: `< 1e-3`
- argmax agreement: `100%` on the controlled equivalence suite

Also record total variation and KL divergence. Any threshold failure blocks performance claims until understood.

## Performance gate

For representative multi-candidate workloads, the Phase 1 go gate is numerical equivalence plus either:

- `>= 2x` effective throughput, or
- `>= 50%` reduction in measured computational work,

without operationally unacceptable VRAM growth.

## Current limitations

The initial prototype:

- relies on causal-cache support from the backbone;
- uses a legacy tuple-cache compatibility boundary for view-based branching;
- does not yet batch multiple independent prefixes in one prefix call;
- does not claim that downstream kernels preserve zero-copy cache views internally;
- has not yet been validated on the frozen production/reference hardware matrix.

These limitations are deliberate. Phase 1 prioritizes falsifiable equivalence and scaling measurements over premature kernel optimization.
