"""Experiment 1: layerwise BF16 p128 x k2 prefix decomposition (Issue #9).

Compares prefix-position hidden states across A (flat full-path batch),
B (prefix-only batch 1), and C (prefix-only batch 2), plus the flat cross-row
prefix identity check. Evidence: artifacts/a1-bf16-stability/layerwise_p128k2.json
"""

# ruff: noqa: I001

from __future__ import annotations

import json

import torch

from _support import OUT, attention_runtime, load_flat, stats
from athrub.attention_runtime import CANONICAL_A0_POLICY
from athrub.workloads import synthetic_request


def main() -> None:
    backend = load_flat(torch.bfloat16)
    request = synthetic_request(
        request_id="shape-p128-k2-c16", prefix_units=128, candidate_units=16, candidate_count=2
    )

    with attention_runtime(CANONICAL_A0_POLICY):
        prefix_ids = backend._encode_prefix(request)
        candidate_ids = [backend._encode_candidate(c) for c in request.candidates]
        paths = [prefix_ids + tokens for tokens in candidate_ids]
        prefix_length = len(prefix_ids)
        max_length = max(len(path) for path in paths)

        flat_ids = torch.full((2, max_length), backend._pad_token_id(), dtype=torch.long, device=backend.device)
        flat_mask = torch.zeros_like(flat_ids)
        for row, path in enumerate(paths):
            flat_ids[row, : len(path)] = torch.tensor(path, dtype=torch.long, device=backend.device)
            flat_mask[row, : len(path)] = 1

        prefix_single = torch.tensor([prefix_ids], dtype=torch.long, device=backend.device)
        prefix_double = prefix_single.repeat(2, 1)

        with torch.inference_mode():
            out_a = backend.model(input_ids=flat_ids, attention_mask=flat_mask, return_dict=True, output_hidden_states=True, use_cache=False)
            out_b = backend.model(input_ids=prefix_single, attention_mask=torch.ones_like(prefix_single), return_dict=True, output_hidden_states=True, use_cache=False)
            out_c = backend.model(input_ids=prefix_double, attention_mask=torch.ones_like(prefix_double), return_dict=True, output_hidden_states=True, use_cache=False)

        comparisons = ("A_vs_B", "A_vs_C", "B_vs_C", "A_row0_vs_row1", "C_row0_vs_row1")
        first_nonzero = {c: None for c in comparisons}
        layers = []
        for index in range(len(out_a.hidden_states)):
            a0 = out_a.hidden_states[index][0, :prefix_length]
            a1 = out_a.hidden_states[index][1, :prefix_length]
            b0 = out_b.hidden_states[index][0]
            c0 = out_c.hidden_states[index][0]
            c1 = out_c.hidden_states[index][1]
            row = {
                "layer": index,
                "A_vs_B": stats(a0, b0),
                "A_vs_C": stats(a0, c0),
                "B_vs_C": stats(b0, c0),
                "A_row0_vs_row1": stats(a0, a1),
                "C_row0_vs_row1": stats(c0, c1),
            }
            layers.append(row)
            for comparison in comparisons:
                if first_nonzero[comparison] is None and row[comparison]["max_abs_delta"] > 0.0:
                    first_nonzero[comparison] = index

        report = {
            "experiment": "layerwise bf16 p128 k2 prefix decomposition",
            "attention_policy": CANONICAL_A0_POLICY.as_record(),
            "prefix_tokens": prefix_length,
            "path_lengths": [len(path) for path in paths],
            "hidden_state_tensors": len(out_a.hidden_states),
            "first_nonzero_layer": first_nonzero,
            "worst_prefix_max_abs_delta": {
                c: max(row[c]["max_abs_delta"] for row in layers) for c in comparisons
            },
            "layers": layers,
        }
        (OUT / "layerwise_p128k2.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: report[k] for k in ("first_nonzero_layer", "worst_prefix_max_abs_delta")}, indent=2))


if __name__ == "__main__":
    main()
