# Athrub

Athrub is an independent research project exploring compact probabilistic decision models that separate shared contextual understanding from candidate-specific computation.

The first research objective is deliberately narrow:

> Can Athrub encode common decision context once, reuse that computation across candidate evaluations, and preserve the resulting decision probabilities?

## Phase 1: Shared Computation

Phase 1 establishes the reference implementation and the benchmark harness required to test shared-context inference rigorously.

Primary deliverables:

- reproducible Athrub reference baseline
- shared-context inference prototype
- latency, FLOP, throughput, and VRAM benchmarks
- candidate-count and context-length scaling study
- exact logit and probability comparison
- numerical stability checks in FP32 and BF16

Success requires preserving decision quality while materially reducing repeated computation for multi-candidate workloads.

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
- model and tokenizer revisions
- numerical precision

Install the optional model-loading dependencies with:

```bash
pip install -e '.[reference]'
```

Copy `configs/reference.example.json`, replace the placeholder model/checkpoint values with immutable revisions and a trained Athrub scalar-head checkpoint, then run:

```bash
python benchmarks/reference_backend.py --config path/to/reference.json
```

Synthetic workloads control shape precisely; `semantic_smoke_requests()` supplies natural decision text for equivalence checks. Neither is intended to substitute for later accuracy or calibration benchmarks.

## Repository layout

```text
src/athrub/                 Core package
experiments/shared_context/ Phase 1 research code and notes
benchmarks/                 Benchmark entry points
configs/                    Reproducible experiment configurations
docs/                       Architecture and research specifications
tests/                      Numerical and contract tests
```

## Research principles

1. Change one variable at a time.
2. Preserve full probability distributions, not only argmax decisions.
3. Measure wall-clock performance and computational work separately.
4. Treat calibration, numerical stability, and out-of-distribution behavior as first-class properties.
5. Record model, tokenizer, code, hardware, precision, and environment revisions for every benchmark.

## Status

Phase 1 reference-backend and shared-context prototypes are implemented. The next milestone is complete only after an Athrub checkpoint, tokenizer revision, scalar-head checkpoint, and canonical benchmark artifact are frozen together.
