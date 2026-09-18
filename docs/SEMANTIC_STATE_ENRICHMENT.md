# Semantic-State Enrichment — Research Sidecar

## Status

This document records a future Athrub research hypothesis. It is **non-normative**, does not create a roadmap gate, and does not authorize implementation.

It must not alter A0/A1 acceptance criteria or be used to reinterpret A1 results. A1 asks whether repeated shared context can be computed once and reused while preserving the same decision distribution. Semantic-state enrichment intentionally introduces new information and may change the distribution.

The earliest reassessment point is after the A1 go gate and after a trainable shared architecture has been established in A2. Any implementation trial requires an explicit Athrub decision.

## Research motivation

Comparative research indicates that a Transformer's internal causal state can retain useful contextual information beyond token identity, and that a learned mapping can make one model's contextual state useful to another frozen model. The relevant Athrub question is not whether to reproduce any external system, but whether a reusable Athrub shared decision representation can later accept bounded semantic enrichment without losing the decision contract or obscuring system economics.

This creates a future question distinct from the current shared-computation hypothesis:

```text
A1:
Can the same shared decision context be computed once,
reused across candidate evaluation,
and preserve the resulting probability distribution?

Future enrichment question:
Can an independently computed semantic state augment Athrub's
shared decision representation and improve decision quality
without making total compute, memory, calibration, or failure
behavior unacceptable?
```

## Candidate mechanism

The candidate mechanism is **residual semantic-state enrichment**.

Conceptually:

```text
state + question
      │
      ├──────────────→ Athrub shared representation H
      │
      └──────────────→ semantic adviser representation S
                              │
                         learned mapping
                              │
                         bounded contribution
                              │
                              ▼
                    H' = H + enrich(H, S)
                              │
                  candidate-specific evaluation
                              │
                    request-local probabilities
```

The Athrub representation remains authoritative. External or auxiliary state is treated as an optional contribution rather than a replacement for the receiver state.

A mature mechanism may include:

- learned projection between representation spaces;
- input- or layer-conditioned contribution weights;
- explicit gates that can suppress harmful enrichment;
- adviser-confidence or compatibility signals;
- deterministic fallback to the receiver-only path.

These are candidate mechanisms, not accepted architecture.

## Architectural boundary

Semantic-state enrichment is not shared-prefix reuse.

Shared-prefix reuse changes execution while preserving reference semantics:

```text
reference shared state
      ↓ reuse
same decision distribution
```

Semantic-state enrichment changes the information available to the decision model:

```text
receiver state + adviser state
      ↓ learned enrichment
potentially different decision distribution
```

Therefore:

- it is excluded from A1;
- it cannot be used as evidence of A1 equivalence;
- its quality gains must be evaluated as a new trained architecture;
- its compute and memory costs must include the adviser and enrichment path;
- a smaller Athrub receiver does not make the overall system "small" if a much larger adviser runs for every request.

## Required controls

Any future experiment must compare at least:

1. **Receiver only** — Athrub without enrichment.
2. **Text-mediated adviser** — adviser produces an intermediate textual representation that Athrub consumes through a declared interface.
3. **Semantic-state adviser** — adviser provides a mapped internal representation without autoregressive intermediate text.
4. **Self-enrichment control where useful** — the receiver's own additional representation path, to separate cross-model transfer from generic extra capacity.

All paths must preserve the same bounded request and output semantics.

## Required quality evidence

Measure at minimum:

- task accuracy or task-appropriate quality metric;
- negative log-likelihood;
- Brier score where applicable;
- expected calibration error or documented alternative;
- complete candidate probability vectors;
- receiver-correct / enriched-wrong regressions;
- receiver-wrong / enriched-correct recoveries;
- per-domain and per-candidate-count behavior;
- abstention or escalation behavior when later production policy is studied.

The key safety/correctness question is not only whether foreign state can be mapped, but whether Athrub can determine when that information should influence the decision.

## Required efficiency evidence

Do not describe removal of intermediate text generation as total-system efficiency without full accounting.

Record separately:

- receiver logical work;
- adviser logical work;
- enrichment/projection work;
- estimated FLOPs for all components;
- prefill time;
- any autoregressive communication time in text-mediated controls;
- enrichment time;
- candidate evaluation time;
- warm p50/p90/p95/p99 latency;
- requests/s and candidates/s;
- peak allocated/reserved VRAM;
- model residency cost;
- cache/state materialization and transfer cost;
- total cost per bounded decision where measurable.

A semantic-state path may be faster than text-mediated model collaboration while still being slower and more expensive than receiver-only inference. Those are separate claims.

## Scaling questions

A future study should vary independently:

```text
receiver size:    ~100M / ~150M / ~300M / ~600M where available
adviser size:     compact / medium / larger
candidate count:  2 / 4 / 8 / 16 / 32 / 64 / 128 / 255
context length:   representative A1/A2 anchors
```

The study should identify whether enrichment moves the **quality-versus-total-system-cost frontier**, rather than only whether quality increases when another model is added.

## Reuse and conditional execution

The mechanism becomes materially more interesting if adviser cost can be amortized or avoided. Candidate future conditions include:

- adviser execution only on uncertain requests;
- reuse of adviser state across multiple candidate sets;
- reuse across repeated decisions sharing the same source context;
- a compact adviser rather than a much larger always-on model;
- asynchronous or tiered execution in production;
- escalation to the adviser only when the receiver's calibrated uncertainty exceeds a policy threshold.

These are later production/architecture hypotheses. None is assumed by this note.

## Representation-alignment caution

A successful learned mapping between two model states demonstrates utility for the tested pair and workload. It does not establish that arbitrary model representation spaces are universally interchangeable.

Any Athrub trial must therefore bind evidence to:

- exact model/revision identities;
- layer/state interface used;
- tokenizer or position-alignment method where relevant;
- precision;
- training data and objective for the mapping;
- supported context lengths and shapes;
- measured failure cases.

If tokenization differs, string-level token matching is only one possible approximation and must not be treated as an exact semantic alignment guarantee.

## Failure model

Expected failure classes include:

- adviser state is wrong and degrades an otherwise correct receiver decision;
- mapped representations are numerically valid but semantically unhelpful;
- enrichment helps one domain and harms another;
- layer or position alignment is incompatible;
- the projection adds substantial trainable parameters;
- VRAM/model residency removes any practical latency benefit;
- adviser execution dominates total cost;
- gating learns to remain effectively always-on or always-off;
- a favorable batch-1 latency result fails under concurrent serving;
- a smaller receiver is misrepresented as a smaller total system.

Every trial must preserve an explicit receiver-only fallback and report negative transfer.

## Adoption trigger

Reassess this pattern only when all of the following are true:

1. A1 has passed its numerical-equivalence and efficiency gate.
2. A2 has established a trainable shared representation under the canonical bounded probability contract.
3. There is a measured quality, domain-transfer, or model-size pressure that receiver-only Athrub cannot meet efficiently enough.
4. A bounded experiment can isolate enrichment from model-size, serialization, and serving changes.
5. Total-system compute, latency, throughput, and VRAM can be measured rather than inferred.

Until then, semantic-state enrichment remains research-only.

## Relationship to later Athrub phases

Potential relevance is primarily after A2:

- **A3:** test whether enrichment changes the quality/compute frontier for smaller receivers, while counting adviser cost explicitly;
- **A4/A5:** test whether complementary domain knowledge transfers under complete domain holdouts;
- **A6:** place adviser routing behind calibration/uncertainty rather than embedding policy into core probability semantics;
- **A7:** qualify conditional adviser use only inside a bounded observable workflow.

This document does not change any existing A0–A7 exit condition.

## Summary invariant

```text
A1 proves reuse.
A2 proves trainable shared decision representation.
Only then may Athrub test semantic-state enrichment.

Enrichment must improve the bounded decision system,
not merely the receiver model in isolation,
and every quality gain must be accounted against
full-system compute, memory, calibration, and failure behavior.
```
