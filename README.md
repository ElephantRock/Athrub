# Athrub

Athrub is an independent research project exploring compact probabilistic decision models that separate shared contextual understanding from candidate-specific computation.

The first research objective is deliberately narrow:

> Can Athrub encode common decision context once, reuse that computation across candidate evaluations, and preserve the resulting decision probabilities?

## Project identity

Athrub architecture and terminology are Athrub-native. A temporary pretrained causal Transformer may be used as an **A0 reference substrate** to establish a meaningful frozen correctness oracle, but that substrate is an experimental dependency rather than the Athrub architecture or model identity.

Identity-bearing acquisition details for any external research substrate stay in private research provenance. Public Athrub manifests use immutable content hashes, opaque provenance identifiers, and Athrub-native terminology.

## A0: Frozen reference

Before Phase 1 performance work, Athrub freezes one meaningful decision model as `Athrub A0 Reference v0.1`.

A0 keeps the architecture intentionally simple:

- causal Transformer reference substrate near the 0.6B scale
- fixed tokenizer
- `TextDecisionCodec` v0.1
- scalar linear decision head
- categorical cross-entropy
- head-only warm-up followed, if justified, by brief full decision adaptation
- initial 2–16 candidate training range
- initial planning envelope of 50k–200k bounded decisions

The complete freeze protocol is specified in `docs/A0_REFERENCE.md`. The public reference manifest must validate against `schemas/reference_manifest.schema.json`.

## Phase 1: Shared Computation

Phase 1 establishes the benchmark evidence required to test shared-context inference rigorously.

Primary deliverables:

- reproducible Athrub reference baseline
- shared-context inference prototype
- latency, FLOP, throughput, and VRAM benchmarks
- candidate-count and context-length scaling study
- exact logit and probability comparison
- numerical stability checks in FP32 and BF16

Success requires preserving decision quality while materially reducing repeated computation for multi-candidate workloads.

A1 deliberately preserves candidate continuation semantics. Runtime candidate encoders, candidate-compatibility heads, late-interaction redesigns, semantic-state enrichment, and offline objective compilation are excluded until the A1 equivalence/efficiency gate is passed.

## A2: Trainable Shared Architecture

After A1 succeeds, Athrub will test whether candidate-specific Transformer continuation itself can be reduced or removed while preserving the same bounded probability contract.

The A2 specification compares:

- trainable shared continuation as the control;
- independent context/candidate encoding with lightweight compatibility scoring;
- independent encoding with a small late-interaction/refinement stage.

Every A2 arm still returns one aligned logit per supplied candidate and applies request-local softmax over exactly that candidate set. The detailed research contract is in `docs/A2_TRAINABLE_ARCHITECTURE.md`.

## Future research sidecar: semantic-state enrichment

A separate, non-gating research hypothesis asks whether a validated Athrub shared decision representation could later be enriched by an independently computed semantic state.

This is not an A1 optimization and not a fourth A2 architecture arm. Unlike shared-prefix reuse, enrichment intentionally introduces additional information and may change the decision distribution. It can only be assessed after the shared architecture is established, and any claim must account for the complete system rather than only the receiver model.

The research boundary, controls, failure model, and full-system efficiency requirements are specified in `docs/SEMANTIC_STATE_ENRICHMENT.md`.

## Future research sidecar: offline objective compilation

Another non-gating hypothesis asks whether an expensive offline teacher, search, ensemble, or reward process can generate bounded-decision supervision that is later learned by a compact Athrub serving model.

This is not shared-prefix reuse and not an A2 architecture arm. A compiled student is a newly trained model, so useful behavior, calibration, teacher/student disagreement, holdout contamination, offline synthesis/training cost, and serving economics must be measured independently. Low serving latency or parameter count is not evidence of exact teacher equivalence or low total learning cost.

The research boundary and required controls are specified in `docs/OFFLINE_OBJECTIVE_COMPILATION.md`.

## Core decision contract

Athrub models bounded probabilistic decisions of the form:

```text
(state, question, candidates) -> probability distribution
```

The project is designed around typed decision outputs rather than free-form text generation.

## Phase 1 reference backend

`FlatReferenceBackend` is the canonical correctness baseline. It intentionally builds one complete path per candidate and processes all paths in a flat batch, including repeated state/question context. Later shared-computation backends must preserve its ordering and probability semantics.

The backend records exact tokenizer accounting for each request:

- shared-prefix token count
- token count for every candidate suffix
- complete path lengths
- total logical token positions processed by flat execution
- reference-substrate and tokenizer revisions
- numerical precision

Install the optional reference-loading dependencies with:

```bash
pip install -e '.[reference]'
```

Copy `configs/reference.example.json`, replace the placeholder reference-substrate/checkpoint values with immutable revisions and a trained Athrub scalar-head checkpoint, then run:

```bash
python benchmarks/reference_backend.py --config path/to/reference.json
```

Synthetic workloads control shape precisely; `semantic_smoke_requests()` supplies natural decision text for equivalence checks. Neither substitutes for later accuracy or calibration benchmarks.

## Repository layout

```text
src/athrub/                 Core package
experiments/shared_context/ Phase 1 research code and notes
benchmarks/                 Benchmark entry points
configs/                    Public identity-neutral experiment templates
docs/                       Architecture and research specifications
schemas/                    Machine-readable artifact contracts
tests/                      Numerical and contract tests
```

Key architecture documents include `docs/ROADMAP.md`, `docs/PHASE1.md`, `docs/A2_TRAINABLE_ARCHITECTURE.md`, `docs/SEMANTIC_STATE_ENRICHMENT.md`, and `docs/OFFLINE_OBJECTIVE_COMPILATION.md`. Architecture-pattern research remains non-binding until linked through an explicit roadmap/specification decision.

## Research principles

1. Change one variable at a time.
2. Preserve full probability distributions, not only argmax decisions.
3. Measure wall-clock performance and computational work separately.
4. Treat calibration, numerical stability, and out-of-distribution behavior as first-class properties.
5. Record code, substrate, tokenizer, head, hardware, precision, configuration, and environment identity for every canonical benchmark.
6. Keep external research identity in private provenance; keep Athrub public architecture vocabulary independent.
7. Treat runtime candidate representation and compatibility scoring as post-A1 hypotheses until controlled A2 evidence exists.
8. Treat semantic-state enrichment as post-A2 research unless an explicit Athrub decision promotes a bounded trial; count adviser compute, memory, latency, and negative transfer as part of the system claim.
9. Treat offline objective compilation as post-A2 research unless a measured quality/scale/transfer/serving-cost pressure exists; separate student utility from teacher equivalence and serving cost from offline synthesis/training cost.

## Status

The flat and shared-context execution prototypes exist and pass CPU contract/equivalence tests. The A0 milestone remains open until a trained reference substrate, tokenizer, scalar-head checkpoint, dataset manifest, and canonical benchmark artifact are frozen together.
