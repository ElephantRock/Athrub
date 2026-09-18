# Athrub Phase 1 Reference Backend

The Phase 1 reference backend is a **correctness oracle for execution experiments**, not the final Athrub architecture.

Its frozen weights are produced by the A0 reference protocol in `docs/A0_REFERENCE.md`.

The temporary pretrained causal Transformer used to establish A0 is called the **reference substrate**. It is an experimental dependency and must not become Athrub model-family or architecture identity.

It intentionally processes one complete sequence per candidate:

```text
shared state + question + candidate A -> scalar score A
shared state + question + candidate B -> scalar score B
shared state + question + candidate C -> scalar score C
```

All complete candidate paths are evaluated in one flat batch. The state/question context is therefore recomputed for every candidate by design.

## Stable serialization

`TextDecisionCodec` serializes a request as a shared prefix followed by a candidate-specific suffix.

Shared prefix:

```text
State:
{state}
Question:
{question}
```

Candidate suffix:

```text
Candidate:
{candidate}
Decision:
```

Phase 1 optimized backends must use this exact semantic partition unless a separate experiment explicitly changes serialization.

## Scoring

The reference substrate returns the final hidden state for every complete candidate path. `ScalarDecisionHead` maps that state to one scalar logit. Candidate logits belonging to the same request are normalized with softmax.

The reference backend emits the complete logit vector and the complete probability vector. Argmax-only comparison is insufficient.

## A0 freeze requirements

A canonical reference artifact is valid only when all of the following are recorded together:

- Athrub git commit
- immutable reference-substrate revision and artifact hash
- immutable tokenizer revision and artifact hash
- trained scalar decision-head checkpoint hash
- text codec version / source commit
- training configuration hash
- dataset manifest hash
- numerical precision
- hardware and runtime environment
- benchmark configuration
- complete raw benchmark output
- public identity-neutral reference manifest

The public manifest must conform to `schemas/reference_manifest.schema.json`.

Identity-bearing source/acquisition details for the temporary reference substrate belong in private research provenance and are represented publicly only by opaque provenance identifiers plus immutable content/revision hashes.

Do not record `main`, `latest`, or another mutable ref as a canonical substrate/tokenizer revision.

## Decision-head checkpoint

`FlatReferenceBackend.from_pretrained()` expects `head_path` to contain a PyTorch state dict compatible with `ScalarDecisionHead`.

Example:

```python
import torch

from athrub.reference import ScalarDecisionHead

head = ScalarDecisionHead(hidden_size=1024, bias=True)
torch.save(head.state_dict(), "decision_head.pt")
```

For canonical runs, the head must be trained and frozen through the A0 adaptation protocol. Random initialization is not a research baseline.

## Exact token accounting

For each request the backend records:

- `prefix_tokens`
- `candidate_token_counts`
- `path_token_counts`
- `flat_logical_token_positions`
- `substrate_revision`
- `tokenizer_revision`

For `K` candidates with shared prefix length `L_p` and suffix lengths `L_c[i]`:

```text
flat_logical_token_positions = sum(L_p + L_c[i] for i in candidates)
```

This is the direct computational quantity that Phase 1 shared-context execution is intended to reduce.

## Acceptance gate for Issue #1

The implementation alone does not close the reference milestone. Issue #1 closes only after a meaningful A0 decision checkpoint/head is frozen, the semantic and controlled workload suites run successfully, and the resulting benchmark/reference manifests are committed or otherwise immutably referenced with hashes.
