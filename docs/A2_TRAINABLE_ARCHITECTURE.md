# A2 — Trainable Shared Architecture

## Status and prerequisite

A2 is a future research phase. It is not authorized to begin until the A1 go gate in `docs/PHASE1.md` passes with frozen numerical and performance evidence.

A2 may change trainable computation, representation topology, and candidate scoring, but it must preserve the canonical Athrub decision contract:

```text
(state, question, candidates) -> probability distribution
```

The output remains one aligned logit per supplied candidate followed by normalization over exactly that request's mutually exclusive candidate set.

## Research question

Can Athrub train a reusable shared decision representation end to end and reduce candidate-specific computation further by replacing full candidate continuation with bounded candidate-compatibility scoring, without materially degrading decision quality or calibration?

A2 therefore asks two progressively stronger questions:

1. Can the A1 shared-computation decomposition be trained end to end?
2. Once shared context exists, is full Transformer continuation for every candidate still necessary?

The second question is subordinate to the first. Candidate-compatibility mechanisms must not be used to retroactively justify A1.

## Fixed semantics

For request `r` with state `s`, question `q`, and ordered candidates `c_1 ... c_K`, every A2 architecture must return:

```text
z = [z_1, z_2, ..., z_K]
p = softmax(z)
```

Requirements:

- `z_i` corresponds to candidate `c_i` in the original request order;
- softmax covers exactly the candidates for one request;
- logits from different requests are never normalized together;
- duplicate textual candidates at different positions retain distinct candidate identities;
- complete logits and probabilities are retained for evaluation;
- independent sigmoid/confidence outputs are not a substitute for the current mutually exclusive Athrub probability distribution.

Any future multilabel task requires a separately declared decision contract rather than a silent change to this one.

## A2 architecture arms

A2 compares at least three trainable arms under matched data, evaluation, parameter-accounting, and deployment conditions.

### A2-A — Trainable shared continuation control

Train the shared-context decomposition while preserving candidate continuation semantics.

Conceptually:

```text
H = shared_context(state, question)
for candidate_i:
    z_i = continue_and_score(H, candidate_i)
p = softmax(z)
```

Purpose:

- establish the trainable version of the A1 mechanism;
- provide the control needed to attribute any later gain to compatibility scoring rather than to training alone.

### A2-B — Independent context/candidate encoding with lightweight scoring

Encode context and candidates independently, then score their compatibility using a bounded scorer such as dot product, bilinear interaction, or a small MLP.

```text
H = context_encoder(state, question)
C_i = candidate_encoder(candidate_i)
z_i = compatibility(H, C_i)
p = softmax(z)
```

A context representation is considered reusable only if `H` is invariant when only the candidate set changes.

This arm tests whether candidate-specific Transformer continuation can be removed while retaining useful decision quality.

### A2-C — Independent encoding with late interaction

Use independently encoded context and candidates, but permit a small late interaction/refinement stage before scoring.

```text
H = context_encoder(state, question)
C_i = candidate_encoder(candidate_i)
C'_i = late_interaction(C_i, H)
z_i = score(C'_i)
p = softmax(z)
```

Candidate late interaction may use a small number of cross-attention or equivalent refinement layers. Complexity must remain explicitly measured; a late interaction module that recreates full candidate re-encoding does not satisfy the intended decomposition.

## Runtime candidate representation

A2 may treat candidate semantics as runtime model input instead of binding every candidate to a fixed learned output index.

This can support changing candidate vocabularies without changing the classifier head, but candidate wording becomes part of the effective input contract. Evaluation must therefore measure:

- candidate paraphrase sensitivity;
- synonymous or near-duplicate candidates;
- seen versus unseen candidate descriptions;
- candidate ordering invariance where semantics permit it;
- large candidate-set behavior;
- calibration as candidate count and wording change.

Candidate text or typed candidate encodings must remain versioned with benchmark artifacts.

## Experimental controls

A2 changes more than A1, so causal controls are mandatory.

For each architecture arm hold constant where applicable:

- dataset and train/eval split;
- decision codec and semantic request structure;
- candidate ordering;
- optimization budget;
- tokenizer or candidate encoder identity;
- parameter accounting;
- precision;
- hardware/runtime environment;
- benchmark shape population;
- evaluation metrics.

Do not simultaneously reduce model size during the primary A2 comparison. Parameter reduction belongs to A3 after the architecture question is resolved.

## Quality and calibration metrics

Measure at minimum:

- task accuracy / task-appropriate quality metric;
- absolute quality delta versus the frozen A0/A1 reference target;
- negative log-likelihood;
- Brier score where applicable;
- expected calibration error or a documented alternative;
- complete probability-vector comparison on shared supported cases;
- per-domain and per-candidate-count breakdowns where the dataset permits them.

A2 is allowed to learn different probabilities from A1 because it is a newly trained architecture. It is not allowed to change what the probabilities mean.

## Computational metrics

Record separately:

- logical context work;
- logical candidate work;
- estimated FLOPs;
- warm p50/p90/p95/p99 latency;
- requests/s;
- candidates/s;
- peak allocated/reserved VRAM;
- context-encoder time;
- candidate-encoder time;
- compatibility/late-interaction time;
- cache construction and materialization cost where applicable.

A theoretical change from candidate continuation to compatibility scoring is not itself a measured speedup.

## Scaling matrix

Retain the A1 candidate-count anchors where feasible:

```text
K = 2, 4, 8, 16, 32, 64, 128, 255
```

Retain representative context lengths from A1 and add explicit candidate-length sweeps.

Primary analysis should identify when cost is dominated by:

- shared context encoding;
- candidate encoding;
- late interaction;
- normalization/packing overhead;
- memory movement.

For independent encoding, separately test reuse across multiple candidate sets for one fixed context.

## Candidate-cache validity

If candidate representations are cached, the cache identity must bind at minimum:

- candidate text or typed representation;
- candidate encoder code/model revision;
- tokenizer/codec revision;
- relevant precision/configuration.

Stale candidate embeddings must fail explicitly or be invalidated deterministically.

## A2 go / no-go gate

A2 can nominate an architecture for A3 only when all of the following hold:

1. The canonical bounded probability contract is preserved.
2. Candidate-to-logit identity and request-local normalization pass adversarial tests.
3. Quality loss against the reference target is less than 1 percentage point absolute on the primary decision benchmark, unless an explicit revised threshold is approved before evaluation.
4. Calibration does not regress materially without a documented trade-off.
5. The selected architecture demonstrates a substantial measured compute, throughput, latency, or memory advantage in at least one target multi-candidate regime without hiding offsetting costs.
6. Candidate-count and context-length scaling are characterized rather than inferred from one favorable shape.
7. All primary evidence is reproducible from immutable provenance.

A compatibility architecture that is faster but loses unacceptable quality does not pass. A high-quality architecture with no meaningful computational advantage remains useful research evidence but does not justify A3 as the Athrub scaling path.

## Future research sidecar — semantic-state enrichment

A separate future hypothesis asks whether Athrub's trainable shared representation can be **augmented by an independently computed semantic state**, rather than only reused or scored more efficiently.

This is intentionally outside the A2 architecture comparison and outside the A2 go/no-go gate. It changes the information available to the receiver and therefore may change the probability distribution by design. It must never be treated as an A1 equivalence mechanism or as evidence that repeated context computation was removed.

The earliest meaningful reassessment is after A2 has established a trainable shared decision representation. A bounded later experiment would compare receiver-only, text-mediated adviser, and semantic-state adviser paths while accounting for total-system quality, calibration, FLOPs, latency, throughput, VRAM, model residency, and negative transfer.

The public Athrub research boundary is specified in `docs/SEMANTIC_STATE_ENRICHMENT.md`.

## Non-goals

A2 does not attempt to:

- reduce parameter count as the primary variable;
- claim cross-domain generality;
- add production policy thresholds or escalation logic;
- introduce domain-specific adapters as the main architecture;
- add cross-model semantic-state enrichment as an A2 architecture arm;
- claim production value from offline results;
- import external project terminology into Athrub architecture.

Those questions remain assigned to later gates or non-binding research sidecars.

## Architecture register linkage

A2 is the first intended research surface for these candidate patterns:

- APR-030 — Runtime Candidate Representation;
- APR-031 — Independent Context and Candidate Encoding;
- APR-032 — Bounded Candidate Compatibility Scoring.

Their presence in the register or this specification means they are eligible for controlled comparison after A1. It does not mean any of them is already adopted, implemented, or verified.

Semantic-state enrichment remains a separate research-only hypothesis. Its documentation does not add an A2 arm, change the A2 exit gate, or create implementation authority.
