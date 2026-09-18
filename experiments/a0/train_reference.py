"""Train and export the Athrub A0 decision reference from private/local inputs."""

# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import platform
import random
from dataclasses import asdict
from pathlib import Path

import torch

from athrub.a0_training import (
    DecisionTrainingExample,
    load_decision_jsonl,
    run_decision_epoch,
    set_substrate_trainable,
    trainable_parameter_count,
)
from athrub.reference import ScalarDecisionHead, TextDecisionCodec


DTYPES = {
    "fp32": torch.float32,
    "bf16": torch.bfloat16,
}


def _configured_dtype(config: dict[str, object]) -> torch.dtype:
    precision = str(config.get("precision", "fp32"))
    if precision not in DTYPES:
        raise ValueError(f"unsupported precision: {precision}")
    return DTYPES[precision]


def _load_reference_substrate(config: dict[str, object]) -> tuple[torch.nn.Module, object]:
    try:
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("A0 adaptation requires the 'reference' optional dependency") from exc

    substrate_id = str(config["substrate_id"])
    substrate_revision = str(config["substrate_revision"])
    tokenizer_id = config.get("tokenizer_id") or substrate_id
    tokenizer_revision = config.get("tokenizer_revision") or substrate_revision

    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_id),
        revision=str(tokenizer_revision),
    )
    model = AutoModel.from_pretrained(
        substrate_id,
        revision=substrate_revision,
        torch_dtype=_configured_dtype(config),
    )
    return model, tokenizer


def _hidden_size(model: torch.nn.Module) -> int:
    config = getattr(model, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        hidden_size = getattr(config, "n_embd", None)
    if hidden_size is None:
        raise ValueError("unable to infer reference-substrate hidden size")
    return int(hidden_size)


def _chance_accuracy(examples: list[DecisionTrainingExample]) -> float:
    return sum(1.0 / len(example.candidates) for example in examples) / len(examples)


def _optimizer(
    parameters: object,
    stage: dict[str, object],
) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        parameters,  # type: ignore[arg-type]
        lr=float(stage["learning_rate"]),
        weight_decay=float(stage.get("weight_decay", 0.0)),
    )


def _run_stage(
    *,
    stage_name: str,
    model: torch.nn.Module,
    head: ScalarDecisionHead,
    tokenizer: object,
    train_examples: list[DecisionTrainingExample],
    validation_examples: list[DecisionTrainingExample],
    stage: dict[str, object],
    device: torch.device,
    seed: int,
) -> list[dict[str, object]]:
    epochs = int(stage.get("epochs", 1))
    batch_size = int(stage.get("batch_size", 8))
    optimizer = _optimizer(
        (parameter for parameter in (*model.parameters(), *head.parameters()) if parameter.requires_grad),
        stage,
    )
    rows: list[dict[str, object]] = []

    for epoch in range(epochs):
        shuffled = list(train_examples)
        random.Random(seed + epoch).shuffle(shuffled)
        train_metrics = run_decision_epoch(
            model=model,
            head=head,
            tokenizer=tokenizer,  # type: ignore[arg-type]
            examples=shuffled,
            batch_size=batch_size,
            device=device,
            optimizer=optimizer,
            codec=TextDecisionCodec(),
        )
        validation_metrics = run_decision_epoch(
            model=model,
            head=head,
            tokenizer=tokenizer,  # type: ignore[arg-type]
            examples=validation_examples,
            batch_size=batch_size,
            device=device,
            optimizer=None,
            codec=TextDecisionCodec(),
        )
        rows.append(
            {
                "stage": stage_name,
                "epoch": epoch + 1,
                "train": asdict(train_metrics),
                "validation": asdict(validation_metrics),
                "trainable_parameters": trainable_parameter_count(model, head),
            }
        )
    return rows


def _environment() -> dict[str, object]:
    output: dict[str, object] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        output.update(
            {
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "gpu_count": torch.cuda.device_count(),
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/a0_adaptation.example.json")
    args = parser.parse_args()

    config_path = Path(args.config)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("reference_name") != "Athrub A0 Reference v0.1":
        raise ValueError("A0 trainer requires the canonical reference_name")

    seed = int(config.get("seed", 1729))
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = torch.device(str(config.get("device", "cuda")))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    data = dict(config["data"])
    train_examples = load_decision_jsonl(str(data["train_jsonl"]))
    validation_examples = load_decision_jsonl(str(data["validation_jsonl"]))
    minimum_candidates = int(config.get("candidate_count_train_min", 2))
    maximum_candidates = int(config.get("candidate_count_train_max", 16))
    for example in [*train_examples, *validation_examples]:
        if not minimum_candidates <= len(example.candidates) <= maximum_candidates:
            raise ValueError(
                f"{example.request_id}: candidate count outside configured A0 range "
                f"[{minimum_candidates}, {maximum_candidates}]"
            )

    dtype = _configured_dtype(config)
    model, tokenizer = _load_reference_substrate(config)
    model.to(device=device, dtype=dtype)
    head_config = dict(config.get("head", {}))
    head = ScalarDecisionHead(
        _hidden_size(model),
        bias=bool(head_config.get("bias", True)),
    ).to(device=device, dtype=dtype)

    output_dir = Path(str(config.get("output_dir", "artifacts/a0-reference-v0.1")))
    output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    executed_stages: list[str] = []

    warmup = dict(config["head_warmup"])
    if not bool(warmup.get("enabled", True)):
        raise ValueError("A0 requires head-warmup")
    set_substrate_trainable(model, False)
    for parameter in head.parameters():
        parameter.requires_grad_(True)
    history.extend(
        _run_stage(
            stage_name="head-warmup",
            model=model,
            head=head,
            tokenizer=tokenizer,
            train_examples=train_examples,
            validation_examples=validation_examples,
            stage=warmup,
            device=device,
            seed=seed,
        )
    )
    executed_stages.append("head-warmup")

    validation_row = history[-1]["validation"]
    if not isinstance(validation_row, dict):
        raise TypeError("validation history must be a mapping")
    warmup_accuracy = float(validation_row["accuracy"])
    chance_accuracy = _chance_accuracy(validation_examples)
    gate = dict(config.get("warmup_gate", {}))
    minimum_margin = float(gate.get("minimum_accuracy_margin", 0.02))
    require_above_uniform = bool(gate.get("require_accuracy_above_uniform", True))
    warmup_gate_passed = (not require_above_uniform) or (
        warmup_accuracy >= chance_accuracy + minimum_margin
    )

    full = dict(config["brief_full_adaptation"])
    full_requested = bool(full.get("enabled", True))
    if full_requested and warmup_gate_passed:
        set_substrate_trainable(model, True)
        for parameter in head.parameters():
            parameter.requires_grad_(True)
        history.extend(
            _run_stage(
                stage_name="brief-full-adaptation",
                model=model,
                head=head,
                tokenizer=tokenizer,
                train_examples=train_examples,
                validation_examples=validation_examples,
                stage=full,
                device=device,
                seed=seed + 10_000,
            )
        )
        executed_stages.append("brief-full-adaptation")

    substrate_output = output_dir / "substrate"
    tokenizer_output = output_dir / "tokenizer"
    save_model = getattr(model, "save_pretrained", None)
    save_tokenizer = getattr(tokenizer, "save_pretrained", None)
    if not callable(save_model) or not callable(save_tokenizer):
        raise TypeError("A0 runtime requires save_pretrained support for substrate and tokenizer")
    save_model(substrate_output)
    save_tokenizer(tokenizer_output)
    torch.save(head.state_dict(), output_dir / "decision_head.pt")

    (output_dir / "training_config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "environment.json").write_text(
        json.dumps(_environment(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "reference_name": config["reference_name"],
        "train_examples": len(train_examples),
        "validation_examples": len(validation_examples),
        "chance_accuracy": chance_accuracy,
        "warmup_accuracy": warmup_accuracy,
        "warmup_gate_passed": warmup_gate_passed,
        "full_adaptation_requested": full_requested,
        "executed_stages": executed_stages,
        "output_dir": str(output_dir),
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
