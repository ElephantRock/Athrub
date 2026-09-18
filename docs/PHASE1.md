# Phase 1 — Shared Computation

## Prerequisite

Phase 1 performance claims begin only after `Athrub A0 Reference v0.1` is frozen under `docs/A0_REFERENCE.md`.

The A0 reference substrate is an experimental dependency rather than Athrub architecture identity. A1 changes execution only; it does not change the frozen weights, tokenizer, decision head, serialization, or candidate semantics.

## Research question

Can Athrub encode common decision context once, reuse that computation across candidate evaluations, and preserve the resulting probability distribution?

Phase 1 deliberately changes inference computation before changing model weights, datasets, training objectives, candidate representation, or scoring topology.

## Hypothesis

For a shared prefix of length `L_p`, `K` candidates, and average candidate suffix length `L_c`, a flat implementation repeatedly processes approximately:

```text
K * (L_p + L_c)
```

logical token positions.

A shared-context implementation should move toward:

```text
L_p + K * L_c
```

The theoretical reduction is not assumed to translate directly into wall-clock speedup. Kernel launch cost, memory movement, attention implementation, batching efficiency, and cache layout must be measured separately.

## Implementations

Phase 1 compares four execution paths:

1. **Reference flat** — canonical candidate-path execution.
2. **Optimized flat** — equivalent semantics with packing/batching improvements only.
3. **Shared context** — shared prefix state with branched candidate continuation.
4. **Shared packed** — shared prefix state plus packed/batched candidate continuation.

Implementation 2 is a required control: ordinary packing improvements must not be misattributed to shared computation.

Candidate-compatibility scoring, independent candidate encoders, runtime label representations, and late cross-attention/refinement are deliberately excluded from these four paths. They are A2 research variables and would confound A1 attribution.

## Workload matrix

Primary candidate counts:

```text
2, 4, 8, 16, 32, 64, 128, 255
```

Primary context lengths:

```text
128, 512, 1024, 2048, 4096, 8192
```

At minimum, include short-prefix/high-cardinality, balanced, and long-prefix/low-cardinality regimes.

Synthetic workloads exist to control shape precisely. Research conclusions must also be reproduced on representative semantic workloads.

## Correctness metrics

For every decision compare the complete outputs, not only the winning candidate.

Record:

- argmax agreement
- maximum absolute logit difference
- mean absolute logit difference
- maximum absolute probability difference
- mean absolute probability difference
- total variation distance
- KL divergence from reference to candidate implementation

FP32 is the primary numerical reference. BF16 is evaluated separately.

### Initial equivalence targets

FP32 target:

```text
max absolute probability delta < 1e-5
```

BF16 target:

```text
max absolute probability delta < 1e-3
100% argmax agreement on controlled equivalence suite
```

These are engineering targets, not immutable scientific thresholds; any relaxation must be documented with evidence.

## Performance metrics

Record separately:

- warm p50/p90/p95/p99 latency
- end-to-end backend latency
- throughput in requests/s
- throughput in candidates/s
- exact tokenizer token counts when available
- estimated FLOPs
- measured GPU kernel time where available
- peak allocated VRAM
- peak reserved VRAM

Do not infer architectural efficiency from latency alone.

## Benchmark discipline

Every primary benchmark artifact must record:

- Athrub git commit
- frozen reference-substrate revision/hash
- tokenizer revision/hash
- decision-head hash
- codec/source revision
- private-provenance opaque identifier
- hardware model
- driver/CUDA version
- PyTorch version
- precision
- batch shape
- warmup count
- repeat count
- random seed where applicable

Primary comparisons must use identical hardware and software environments.

## Go / no-go gate

Proceed to trainable shared computation when all of the following hold:

1. Numerical equivalence is demonstrated within documented tolerances.
2. Candidate ordering and decision semantics remain unchanged.
3. Shared computation produces at least one material efficiency benefit on representative multi-candidate workloads, initially targeted as either:
   - >= 2x effective throughput, or
   - >= 50% reduction in measured computational work.
4. VRAM growth does not erase the operational benefit.

If theoretical work falls dramatically while wall-clock improvement remains small, the next investigation is kernel/memory-layout optimization rather than an immediate claim of architectural speedup.

Only after this gate passes may A2 test whether candidate continuation itself can be replaced by a trainable compatibility mechanism under the same bounded probability contract.

## Non-goals

Phase 1 does not attempt to:

- train a new general model
- replace candidate continuation with compatibility scoring
- introduce independent candidate encoders or candidate-conditioned late-interaction heads
- introduce runtime candidate-label semantics as a new trainable architecture
- reduce parameter count
- introduce new reinforcement-learning objectives
- claim domain generality
- establish production safety or calibration

Those are later phases and would confound the first architectural question.
