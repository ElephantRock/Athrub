# Athrub Phase 1 Reference Backend

The Phase 1 reference backend is a **correctness oracle for execution experiments**, not the final Athrub architecture.

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

The reference backbone returns the final hidden state for every complete candidate path. `ScalarDecisionHead` maps that state to one scalar logit. Candidate logits belonging to the same request are normalized with softmax.

The reference backend emits the complete logit vector and the complete probability vector. Argmax-only comparison is insufficient.

## Checkpoint freeze requirements

A canonical reference artifact is valid only when all of the following are recorded together:

- Athrub git commit
- immutable backbone model revision
- immutable tokenizer revision
- scalar decision-head checkpoint hash
- text codec version / source commit
- numerical precision
- hardware and runtime environment
- benchmark configuration
- complete raw benchmark output

Do not record `main`, `latest`, or another mutable ref as a canonical model/tokenizer revision.

## Decision-head checkpoint

`FlatReferenceBackend.from_pretrained()` expects `head_path` to contain a PyTorch state dict compatible with `ScalarDecisionHead`.

Example:

```python
import torch

from athrub.reference import ScalarDecisionHead

head = ScalarDecisionHead(hidden_size=1024, bias=True)
torch.save(head.state_dict(), "decision_head.pt")
```

For canonical runs, the head must be trained/frozen separately; random initialization is not a research baseline.

## Exact token accounting

For each request the backend records:

- `prefix_tokens`
- `candidate_token_counts`
- `path_token_counts`
- `flat_logical_token_positions`

For `K` candidates with shared prefix length `L_p` and suffix lengths `L_c[i]`:

```text
flat_logical_token_positions = sum(L_p + L_c[i] for i in candidates)
```

This is the direct computational quantity that Phase 1 shared-context execution is intended to reduce.

## Acceptance gate for reference baseline

The implementation alone does not close the reference milestone. It closes only after a real Athrub checkpoint/head is frozen, the semantic and controlled workload suites run successfully, and the resulting benchmark artifact is committed or otherwise immutably referenced with hashes.
