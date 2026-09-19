"""Experiments 2-3: first-block sub-operation trace and D/E shape controls (Issue #9).

The trace localizes the first prefix divergence inside transformer block 0
(input norm, q/k/v projections, attention, o_proj, norms, MLP). The shape
controls add D (batch 1, full length) and E (batch 2, full length, duplicated
suffix) and compare prefix positions against A per layer.
Evidence: artifacts/a1-bf16-stability/{first_block_trace_p128k2,shape_controls_p128k2}.json
"""

# ruff: noqa: I001

from __future__ import annotations

import json

import torch

from _support import OUT, attention_runtime, load_flat, stats
from athrub.attention_runtime import CANONICAL_A0_POLICY
from athrub.workloads import synthetic_request


def build_batch(backend, rows: list[list[int]]) -> tuple[torch.Tensor, torch.Tensor]:
    max_length = max(len(row) for row in rows)
    ids = torch.full((len(rows), max_length), backend._pad_token_id(), dtype=torch.long, device=backend.device)
    mask = torch.zeros_like(ids)
    for index, row in enumerate(rows):
        ids[index, : len(row)] = torch.tensor(row, dtype=torch.long, device=backend.device)
        mask[index, : len(row)] = 1
    return ids, mask


def main() -> None:
    backend = load_flat(torch.bfloat16)
    request = synthetic_request(
        request_id="shape-p128-k2-c16", prefix_units=128, candidate_units=16, candidate_count=2
    )

    with attention_runtime(CANONICAL_A0_POLICY):
        prefix_ids = backend._encode_prefix(request)
        candidate_ids = [backend._encode_candidate(c) for c in request.candidates]
        prefix_length = len(prefix_ids)

        flat_ids, flat_mask = build_batch(backend, [prefix_ids + c for c in candidate_ids])
        prefix_single = torch.tensor([prefix_ids], dtype=torch.long, device=backend.device)
        prefix_double = prefix_single.repeat(2, 1)
        filler_ids, filler_mask = build_batch(backend, [prefix_ids + candidate_ids[0]])
        matched_ids, matched_mask = build_batch(backend, 2 * [prefix_ids + candidate_ids[0]])

        block = backend.model.layers[0]
        captures: dict[str, list[torch.Tensor]] = {}
        hooks = []

        def capture(name: str):
            def hook(_module, _inputs, output):
                captures.setdefault(name, []).append(output[0] if isinstance(output, tuple) else output)
            return hook

        for module_name, label in (
            ("input_layernorm", "input_layernorm"),
            ("self_attn.q_proj", "q_proj"),
            ("self_attn.k_proj", "k_proj"),
            ("self_attn.v_proj", "v_proj"),
            ("self_attn", "attention_output"),
            ("self_attn.o_proj", "o_proj"),
            ("post_attention_layernorm", "post_attention_layernorm"),
            ("mlp.gate_proj", "mlp_gate_proj"),
            ("mlp.up_proj", "mlp_up_proj"),
            ("mlp.down_proj", "mlp_down_proj"),
        ):
            module = backend.model.layers[0]
            for attribute in module_name.split("."):
                module = getattr(module, attribute)
            hooks.append(module.register_forward_hook(capture(label)))
        hooks.append(block.register_forward_hook(capture("block_output")))

        def forward(ids: torch.Tensor, mask: torch.Tensor):
            captures.clear()
            with torch.inference_mode():
                output = backend.model(
                    input_ids=ids, attention_mask=mask, return_dict=True,
                    output_hidden_states=True, use_cache=False,
                )
            return output, {name: tensors[0].detach().clone() for name, tensors in captures.items()}

        out_a, cap_a = forward(flat_ids, flat_mask)
        out_b, cap_b = forward(prefix_single, torch.ones_like(prefix_single))
        out_c, cap_c = forward(prefix_double, torch.ones_like(prefix_double))
        for hook in hooks:
            hook.remove()

        stages = [
            "input_layernorm", "q_proj", "k_proj", "v_proj", "attention_output", "o_proj",
            "post_attention_layernorm", "mlp_gate_proj", "mlp_up_proj", "mlp_down_proj", "block_output",
        ]
        trace_rows = {}
        for stage in stages:
            a0, b0, c0 = cap_a[stage][0, :prefix_length], cap_b[stage][0, :prefix_length], cap_c[stage][0, :prefix_length]
            trace_rows[stage] = {"A_vs_B": stats(a0, b0), "A_vs_C": stats(a0, c0), "B_vs_C": stats(b0, c0)}
        trace_report = {
            "experiment": "first-block sub-operation trace, bf16 p128 k2, prefix positions",
            "block_input_embeddings": {
                "A_vs_B": stats(out_a.hidden_states[0][0, :prefix_length], out_b.hidden_states[0][0, :prefix_length]),
                "A_vs_C": stats(out_a.hidden_states[0][0, :prefix_length], out_c.hidden_states[0][0, :prefix_length]),
                "B_vs_C": stats(out_b.hidden_states[0][0, :prefix_length], out_c.hidden_states[0][0, :prefix_length]),
            },
            "stages": trace_rows,
        }
        (OUT / "first_block_trace_p128k2.json").write_text(json.dumps(trace_report, indent=2) + "\n", encoding="utf-8")

        out_d, _ = forward(filler_ids, filler_mask)
        out_e, _ = forward(matched_ids, matched_mask)

        shape_comparisons = ("B_vs_A", "D_vs_A", "E_vs_A", "E1_vs_A1")
        first_nonzero = {c: None for c in shape_comparisons}
        layer_rows = []
        for index in range(len(out_a.hidden_states)):
            a0 = out_a.hidden_states[index][0, :prefix_length]
            row = {
                "layer": index,
                "B_vs_A": stats(out_b.hidden_states[index][0], a0),
                "D_vs_A": stats(out_d.hidden_states[index][0, :prefix_length], a0),
                "E_vs_A": stats(out_e.hidden_states[index][0, :prefix_length], a0),
                "E1_vs_A1": stats(out_e.hidden_states[index][1, :prefix_length], out_a.hidden_states[index][1, :prefix_length]),
            }
            layer_rows.append(row)
            for comparison in shape_comparisons:
                if first_nonzero[comparison] is None and row[comparison]["max_abs_delta"] > 0.0:
                    first_nonzero[comparison] = index

        shape_report = {
            "experiment": "shape-matched bf16 controls, p128 k2, prefix positions",
            "executions": {
                "A": "batch 2, length 598, real candidate suffixes",
                "B": "batch 1, length 507, prefix only",
                "C": "batch 2, length 507, prefix only",
                "D": "batch 1, length 598, prefix + one real suffix",
                "E": "batch 2, length 598, prefix + same suffix duplicated",
            },
            "prefix_tokens": prefix_length,
            "path_length": len(prefix_ids + candidate_ids[0]),
            "first_nonzero_layer": first_nonzero,
            "worst_prefix_max_abs_delta": {
                c: max(row[c]["max_abs_delta"] for row in layer_rows) for c in shape_comparisons
            },
            "layers": layer_rows,
        }
        (OUT / "shape_controls_p128k2.json").write_text(json.dumps(shape_report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({k: shape_report[k] for k in ("first_nonzero_layer", "worst_prefix_max_abs_delta")}, indent=2))


if __name__ == "__main__":
    main()
