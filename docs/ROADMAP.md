# Athrub Roadmap

Athrub is developed through explicit research gates. Later phases are contingent on evidence from earlier ones.

## A0 — Frozen reference

Goal: create a meaningful, reproducible decision-model correctness oracle without treating the temporary reference substrate as Athrub architecture identity.

Work:

- short head-only warm-up
- brief full decision adaptation only if justified by warm-up quality
- freeze tokenizer, substrate weights, scalar decision head, codec, training/data configuration, and benchmark artifacts
- publish an identity-neutral manifest backed by private research provenance

Exit condition: a trained `Athrub A0 Reference v0.1` can be regenerated from its recorded provenance bundle and exceeds documented trivial/random controls on the selected bounded-decision suite.

## A1 — Shared computation

Goal: encode common context once and reuse it across candidate-specific computation without materially changing probabilities.

A1 changes execution only. Candidate serialization, model weights, training objective, decision semantics, and candidate-continuation behavior remain fixed so the repeated-context hypothesis can be tested without architectural confounds.

Exit condition: numerical equivalence plus a material efficiency gain on representative workloads.

## A2 — Trainable shared architecture

Goal: train shared computation end to end and test whether full candidate-specific Transformer continuation is still necessary once a reusable shared decision representation exists.

A2 begins only after the A1 go gate passes. It compares three controlled architecture families under the same bounded probability contract:

1. **Trainable shared continuation** — the trainable form of the A1 decomposition and the primary control.
2. **Independent context/candidate encoding + lightweight compatibility scoring** — context and candidate representations are produced separately, then one aligned logit is produced per candidate.
3. **Independent encoding + late interaction** — the same separation with a small bounded interaction/refinement stage before scoring.

Every arm must return one logit per supplied candidate and normalize over exactly the candidates in that request. Runtime candidate semantics may be explored, but independent sigmoid scores do not replace the current mutually exclusive Athrub probability distribution.

Detailed experiment contract: `docs/A2_TRAINABLE_ARCHITECTURE.md`.

Exit condition: <1 percentage point absolute quality loss against the reference target, preserved probability semantics and acceptable calibration, plus a substantial measured compute/latency/throughput/memory advantage in a representative multi-candidate regime.

## A3 — Architecture scaling

Goal: identify the smallest architecture that preserves useful decision quality.

Planned scale sequence:

```text
~600M -> ~300M -> ~150M -> ~100M
```

Model-size reduction is introduced only after the A2 architecture is validated. Size and candidate-computation structure must not be changed simultaneously in the primary A2 comparison.

### Non-gating research sidecar — semantic-state enrichment

After A2 establishes a trainable shared decision representation, Athrub may separately reassess whether an independently computed semantic state can enrich that representation and move the **quality-versus-total-system-cost frontier**.

This is not an A3 requirement, does not change the A3 scaling sequence, and creates no implementation authority by itself. Any later trial must count the full adviser path when measuring parameters, FLOPs, latency, throughput, VRAM, model residency, and cost. A smaller receiver plus a larger always-on adviser is not treated as a smaller total system.

The relevant comparisons are receiver-only, text-mediated adviser, semantic-state adviser, and appropriate controls. Negative transfer—cases where enrichment turns a correct receiver decision into an incorrect one—must be measured explicitly.

Research boundary: `docs/SEMANTIC_STATE_ENRICHMENT.md`.

### Non-gating research sidecar — offline objective compilation

After A2, Athrub may also reassess whether an expensive offline teacher, search, ensemble, or reward process can generate provenance-bound bounded-decision supervision that recovers quality or calibration in a smaller serving model.

This is not an A3 requirement and does not change the scale sequence. It becomes relevant only when a measured forcing function exists—for example, parameter reduction causes unacceptable quality loss, domain transfer remains insufficient, a desired multi-objective behavior is too expensive to optimize online, or direct serving of a stronger process violates latency/cost constraints.

Any trial must separate student serving economics from offline synthesis/training economics. A compact student is not evidence of exact teacher equivalence, low total learning cost, or general-purpose capability. Data-volume-matched controls, teacher/student disagreement, calibration, holdout contamination checks, and immutable generation provenance are mandatory.

Research boundary: `docs/OFFLINE_OBJECTIVE_COMPILATION.md`.

## A4 — Multi-domain decision benchmark

Goal: evaluate whether the learned decision capability transfers across unrelated semantic domains.

Key requirement: complete domain holdouts, not only random example splits.

If semantic-state enrichment is ever trialed in A4/A5 research, receiver-only and enriched paths must be reported separately so cross-domain capability is not confused with adviser capability.

If offline objective compilation is trialed, held-out domains must be isolated from target generation, reward feedback, teacher selection, filtering, and other training supervision. Synthetic volume must not be presented as domain-transfer evidence.

## A5 — General Athrub model

Goal: train one compact decision model across heterogeneous task families and measure zero-shot domain transfer.

Target: useful zero-shot behavior with efficient specialization from limited domain examples.

Any optional adviser mechanism remains separable from the general Athrub predictor. Generality claims must identify whether they describe the receiver alone or the complete enriched system.

Any compiled offline supervision remains separable from the generality claim. Athrub must report which domains supplied teacher/optimizer targets and retain complete-domain holdouts before describing the compact student as general-purpose.

## A6 — Domain adaptation and calibration

Goal: separate reusable prediction capability from local policy and domain calibration.

Artifacts may include lightweight adapters, calibration parameters, and abstention/escalation policies.

If later evidence justifies conditional semantic-state enrichment, routing belongs behind explicit calibration/uncertainty policy rather than being silently embedded into core probability semantics.

If compiled supervision is used for specialization, local reward functions, thresholds, or policy preferences remain explicit adaptation artifacts rather than universal Athrub probability semantics.

## A7 — Production runtime

Goal: validate Athrub in a bounded, measurable real-world workflow where decision latency, cost, quality, uncertainty, and escalation can all be observed.

The production gate requires comparison against deterministic rules and appropriate larger-model baselines.

Any adviser/enrichment path must be qualified as a full-system deployment path, including fallback behavior, adviser availability, negative transfer, concurrency, memory residency, and total cost per bounded decision.

Any compact model created through offline objective compilation must be compared with direct stronger-model and calibrated escalation baselines, while reporting offline refresh/retraining cost separately from serving cost.
