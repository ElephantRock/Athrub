# Offline Objective Compilation — Research Sidecar

## Status

This document records a future Athrub research hypothesis. It is **non-normative**, does not create a roadmap gate, and does not authorize implementation.

It must not alter A0/A1 acceptance criteria, be used to reinterpret A1 equivalence evidence, or be treated as an additional A2 architecture arm. The earliest meaningful reassessment is after A2 has established a trainable shared architecture and Athrub has measured a quality, model-size, transfer, multi-objective, or serving-cost pressure that direct training/serving does not meet efficiently enough.

Any implementation trial requires an explicit Athrub decision.

## Research question

A strong teacher, search procedure, ensemble, reward optimizer, or other expensive process may discover useful bounded-decision behavior that is too costly to run for every production request. Athrub may later test whether that behavior can be converted into frozen training supervision for a compact serving model.

```text
Can expensive offline decision optimization be converted into
provenance-bound Athrub supervision such that a compact serving
model preserves useful decision quality and calibration at
materially lower serving cost?
```

This is distinct from A1:

```text
A1:
same model semantics + less repeated inference computation

Offline objective compilation:
newly trained compact model + behavior learned from an expensive
offline process, with empirical rather than exact equivalence
```

## Candidate mechanism

```text
bounded training requests
        ↓
expensive teacher / search / reward process
        ↓
frozen decision supervision
        ↓
compact Athrub student training
        ↓
compact serving model
        ↓
request-local bounded probability distribution
```

Compiled supervision may include hard decisions, complete logits or probability vectors, calibrated soft targets, reward-ranked candidate structures, or other explicitly typed decision targets. The target contract must be frozen before the primary experiment.

The expensive process is a training-time instrument. It is not part of Athrub's normal serving path unless separately promoted.

## Architectural boundary

Offline objective compilation is not shared-prefix reuse.

Therefore:

- it is excluded from A1;
- it is not an A2 architecture arm;
- it cannot repair or justify a failed A1 equivalence result;
- teacher quality does not establish student quality;
- low student inference cost does not establish low total learning cost;
- a specialized compact student is not evidence of general-purpose capability;
- synthetic or reward-shaped supervision cannot cross declared evaluation holdout boundaries.

A1 may require numerical equivalence within defined tolerances. A compiled student is a newly trained model and is evaluated through quality, calibration, robustness, and economics rather than by assuming exact teacher equivalence.

## Adoption trigger

Reassess this pattern only when all of the following are true:

1. A1 has passed its numerical-equivalence and efficiency gate.
2. A2 has established a trainable shared architecture under the canonical bounded probability contract.
3. A measured forcing function exists, such as:
   - A3 model-size reduction causes unacceptable quality or calibration loss;
   - A4/A5 complete-domain holdouts show insufficient transfer;
   - a desired multi-objective behavior is too expensive to optimize at inference;
   - direct serving of a stronger model/process violates latency, throughput, memory, or cost constraints.
4. The offline teacher/search/reward process can be versioned and reproduced.
5. Evaluation holdouts can be isolated from target generation and reward/judge construction.
6. Offline generation/training cost and serving cost can be measured separately.

Until then, the mechanism remains research-only.

## Required controls

Any future trial should compare at least:

1. **Compact receiver, direct supervision** — same student architecture without compiled targets.
2. **Compact receiver, data-volume-matched control** — controls for gains caused merely by more examples.
3. **Compact receiver, compiled supervision** — the candidate mechanism.
4. **Stronger direct process** — the teacher/optimizer or nearest feasible quality upper control.
5. **Escalation baseline where relevant** — compact receiver with calibrated routing to a stronger model instead of permanently distilling that behavior.

If reward optimization is used, include ablations for individual reward components and inspect high-reward failure cases.

## Required quality evidence

Measure at minimum:

- primary task quality;
- negative log-likelihood;
- Brier score where applicable;
- expected calibration error or a documented alternative;
- complete candidate probability vectors;
- teacher/student disagreement;
- teacher-correct / student-wrong and student-correct / teacher-wrong slices;
- per-candidate-count behavior;
- per-domain behavior;
- complete-domain holdouts before any generality claim.

A high reward or judge score is evidence that the defined objective was optimized. It is not by itself evidence that the intended decision behavior was learned.

## Data provenance and contamination boundary

Every generated training artifact must bind, where applicable:

- source request/data revision;
- teacher/search/optimizer identity;
- objective/reward configuration;
- generation seed or sampling configuration;
- candidate-set construction;
- generated target;
- filtering/selection rule;
- student-training dataset version.

Evaluation domains and canonical test requests must not contribute labels, reward feedback, teacher selection, prompt tuning, rejection filters, or other supervision to the compiled training set.

Random example separation is insufficient for claims involving unseen domains.

## Offline cost accounting

Report separately:

- number of source training requests;
- teacher/optimizer samples per request;
- accepted/retained targets;
- generation accelerator hours and FLOPs where measurable;
- filtering/judging cost;
- student training steps, examples/tokens, FLOPs, and accelerator hours;
- generated-artifact storage;
- regeneration cost for the canonical supervision set.

A compact serving model can still require an expensive learning pipeline.

## Serving cost accounting

Record separately:

- student parameter count;
- logical context and candidate work;
- estimated FLOPs;
- warm p50/p90/p95/p99 latency;
- requests/s and candidates/s;
- peak allocated/reserved VRAM;
- model residency;
- cache/materialization cost;
- candidate-count/context-length scaling;
- cost per bounded decision where measurable.

The intended economic claim is an amortization claim:

```text
higher offline learning cost
        ↓
lower repeated serving cost
```

Whether that trade is favorable depends on deployment volume, refresh frequency, drift, and quality retention.

## Relationship to later phases

Potential relevance is primarily post-A2:

- **A3:** test whether compiled supervision recovers quality/calibration lost during parameter reduction;
- **A4:** test transfer under complete domain holdouts without contaminating those holdouts;
- **A5:** study whether one compact model can absorb reusable behavior across heterogeneous task families;
- **A6:** keep calibration/adaptation explicit instead of hiding local policy inside the core predictor;
- **A7:** qualify the compact student against direct stronger-model and escalation baselines in a bounded observable workflow.

This document does not change any existing A0–A7 exit condition.

## Summary invariant

```text
A1 proves equivalent reuse.
A2 establishes the trainable Athrub architecture.
Only then may Athrub consider compiling expensive offline behavior
into a compact serving model when a measured forcing function exists.

Student utility is not teacher equivalence.
Compact serving cost is not total learning cost.
Synthetic volume is not domain generality.
```
