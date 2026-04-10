"""Continual learning training across time periods."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from src.model.backbone import apply_lora, load_backbone, load_trained_model
from src.model.docid_tokenizer import setup_tokenizer_and_embeddings
from src.training.dataset import ChronoDocIDCollator, ChronoDocIDDataset, ReplayDataset
from src.training.lora_merging import LoRAMerger, compose_loras, prune_loras_fifo, prune_loras_lra
from src.training.replay_buffer import ReplayBuffer

logger = logging.getLogger(__name__)


def _load_period_dataset(
    config: dict, period_name: str
) -> ChronoDocIDDataset:
    """Load dataset for a specific time period."""
    exp_cfg = config["experiment"]
    data_dir = Path(exp_cfg["output_dir"]) / "splits" / period_name
    examples_path = data_dir / "train_examples.jsonl"
    docid_to_tokenids_path = Path(exp_cfg["output_dir"]) / "docid_to_tokenids.json"

    return ChronoDocIDDataset(
        examples_path=examples_path,
        docid_to_tokenids_path=docid_to_tokenids_path,
        filter_ratio=config.get("evaluation", {}).get("filter_query_ratio", 0.3),
    )


def _make_trainer(
    model, tokenizer, dataset, collator, config: dict,
    period_name: str, output_dir: Path, strategy: str,
) -> Seq2SeqTrainer:
    """Create a Seq2SeqTrainer with standard CL training args."""
    train_cfg = config["training"]["d0"]  # reuse D0 hyperparams for CL periods
    exp_cfg = config["experiment"]

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=train_cfg["num_epochs"],
        per_device_train_batch_size=train_cfg["batch_size"],
        learning_rate=train_cfg["learning_rate"],
        warmup_ratio=train_cfg.get("warmup_ratio", 0.04),
        weight_decay=train_cfg.get("weight_decay", 0.01),
        lr_scheduler_type=train_cfg.get("scheduler", "cosine"),
        bf16=train_cfg.get("bf16", True),
        fp16=train_cfg.get("fp16", False),
        logging_steps=exp_cfg.get("logging_steps", 50),
        save_strategy=exp_cfg.get("save_strategy", "epoch"),
        seed=exp_cfg.get("seed", 42),
        report_to="wandb" if exp_cfg.get("wandb_project") else "none",
        run_name=f"cl_{strategy}_{period_name}",
        remove_unused_columns=False,
        predict_with_generate=False,
    )

    return Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )


def train_period(
    config_path: str,
    period_name: str,
    strategy: str,
    prev_checkpoint: str | None = None,
    output_dir: str | None = None,
) -> str:
    """Train on a new time period using the specified CL strategy.

    Strategies:
        replay:     Single LoRA + replay buffer. Load prev checkpoint, continue
                    training with mix of new + replay data.
        merge:      Train fresh LoRA on new period. Merge with accumulated weights.
        compose:    Train fresh LoRA per period. Keep separate, route at inference.
        null_space: Project gradients into null space of previous periods.

    Returns:
        Path to the output checkpoint.
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    cl_cfg = config["training"]["cl"]
    exp_cfg = config["experiment"]

    if output_dir is None:
        output_dir = str(Path(exp_cfg["output_dir"]) / "cl" / strategy / period_name)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    collator = ChronoDocIDCollator(
        tokenizer_path_or_name=model_cfg["backbone"],
        max_query_length=config["training"]["d0"].get("max_query_length", 64),
    )

    period_dataset = _load_period_dataset(config, period_name)

    if strategy == "replay":
        _train_replay(config, period_name, period_dataset, collator, prev_checkpoint, output_path)
    elif strategy == "merge":
        _train_merge(config, period_name, period_dataset, collator, prev_checkpoint, output_path)
    elif strategy == "compose":
        _train_compose(config, period_name, period_dataset, collator, prev_checkpoint, output_path)
    elif strategy == "null_space":
        _train_null_space(config, period_name, period_dataset, collator, prev_checkpoint, output_path)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return str(output_path)


def _train_replay(
    config: dict, period_name: str, dataset: ChronoDocIDDataset,
    collator: ChronoDocIDCollator, prev_checkpoint: str | None, output_path: Path,
) -> None:
    """Strategy A: Single LoRA with replay buffer."""
    cl_cfg = config["training"]["cl"]["replay"]
    exp_cfg = config["experiment"]

    # Load previous model
    if prev_checkpoint is None:
        raise ValueError("replay strategy requires a prev_checkpoint")
    model, tokenizer = load_trained_model(prev_checkpoint, config["model"]["backbone"])

    # Load or create replay buffer
    buffer_path = Path(exp_cfg["output_dir"]) / "cl" / "replay" / "replay_buffer.json"
    if buffer_path.exists():
        replay_buffer = ReplayBuffer.load(buffer_path)
    else:
        replay_buffer = ReplayBuffer(max_size=cl_cfg.get("buffer_size", 10000))

    # Build replay dataset
    train_dataset = ReplayDataset(
        current_dataset=dataset,
        replay_buffer=replay_buffer,
        replay_ratio=cl_cfg.get("replay_ratio", 0.3),
    )

    trainer = _make_trainer(
        model, tokenizer, train_dataset, collator, config,
        period_name, output_path, "replay",
    )

    logger.info("Strategy A (replay): training on %d examples (%d current + %d replay)",
                len(train_dataset), len(dataset), len(train_dataset) - len(dataset))
    trainer.train()
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Update replay buffer with current period
    replay_buffer.add_from_dataset(dataset, period_name)
    replay_buffer.save(buffer_path)


def _train_merge(
    config: dict, period_name: str, dataset: ChronoDocIDDataset,
    collator: ChronoDocIDCollator, prev_checkpoint: str | None, output_path: Path,
) -> None:
    """Strategy B: Train fresh LoRA, then merge with accumulated state."""
    cl_cfg = config["training"]["cl"]["merge"]
    model_cfg = config["model"]
    exp_cfg = config["experiment"]

    # Load base model (not from checkpoint -- fresh LoRA each period)
    dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))
    model, tokenizer = load_backbone(model_cfg["backbone"], dtype)
    model, tokenizer = setup_tokenizer_and_embeddings(model, tokenizer, config=config)

    lora_cfg = model_cfg.get("lora", {})
    model = apply_lora(
        model,
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
    )

    trainer = _make_trainer(
        model, tokenizer, dataset, collator, config,
        period_name, output_path, "merge",
    )

    logger.info("Strategy B (merge): training fresh LoRA on %d examples", len(dataset))
    trainer.train()

    # Save this period's LoRA
    period_lora_path = output_path / "period_lora"
    trainer.save_model(str(period_lora_path))

    # Merge with accumulated state
    merger_path = Path(exp_cfg["output_dir"]) / "cl" / "merge" / "merger_state"
    if merger_path.exists():
        merger = LoRAMerger.load(merger_path)
    else:
        merger = LoRAMerger(alpha_decay=cl_cfg.get("alpha_decay", 0.9))

    merger.merge(period_lora_path)
    merger.apply_to_model(model)
    merger.save(merger_path)

    # Save final merged model
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))


def _train_compose(
    config: dict, period_name: str, dataset: ChronoDocIDDataset,
    collator: ChronoDocIDCollator, prev_checkpoint: str | None, output_path: Path,
) -> None:
    """Strategy C: Train fresh LoRA per period, keep separate for routing."""
    model_cfg = config["model"]
    cl_cfg = config["training"]["cl"]["compose"]

    # Train a fresh LoRA for this period
    dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))
    model, tokenizer = load_backbone(model_cfg["backbone"], dtype)
    model, tokenizer = setup_tokenizer_and_embeddings(model, tokenizer, config=config)

    lora_cfg = model_cfg.get("lora", {})
    model = apply_lora(
        model,
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
    )

    trainer = _make_trainer(
        model, tokenizer, dataset, collator, config,
        period_name, output_path, "compose",
    )

    logger.info("Strategy C (compose): training period LoRA on %d examples", len(dataset))
    trainer.train()
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))

    # Prune if over capacity
    exp_cfg = config["experiment"]
    compose_dir = Path(exp_cfg["output_dir"]) / "cl" / "compose"
    existing_loras = sorted(compose_dir.glob("*/adapter_model.bin"))
    lora_dirs = [p.parent for p in existing_loras]

    max_loras = cl_cfg.get("max_loras", 8)
    pruning = cl_cfg.get("pruning", "fifo")

    if len(lora_dirs) > max_loras:
        if pruning == "fifo":
            prune_loras_fifo(lora_dirs, max_loras)
        else:
            usage_path = compose_dir / "usage_counts.json"
            usage_counts = {}
            if usage_path.exists():
                with open(usage_path) as f:
                    usage_counts = json.load(f)
            prune_loras_lra(lora_dirs, usage_counts, max_loras)


def _train_null_space(
    config: dict, period_name: str, dataset: ChronoDocIDDataset,
    collator: ChronoDocIDCollator, prev_checkpoint: str | None, output_path: Path,
) -> None:
    """Strategy D: Null-space projection (gradient projection into null space of past tasks).

    This is a placeholder -- full implementation requires computing the Fisher
    information matrix or representation matrix from previous periods and
    projecting gradients during training.
    """
    model_cfg = config["model"]
    exp_cfg = config["experiment"]

    if prev_checkpoint is None:
        raise ValueError("null_space strategy requires a prev_checkpoint")
    model, tokenizer = load_trained_model(prev_checkpoint, model_cfg["backbone"])

    # TODO: Compute null-space projection matrix from previous period representations.
    # For now, fall back to standard fine-tuning (equivalent to no projection).
    logger.warning(
        "Strategy D (null_space): gradient projection not yet implemented, "
        "falling back to standard fine-tuning."
    )

    trainer = _make_trainer(
        model, tokenizer, dataset, collator, config,
        period_name, output_path, "null_space",
    )

    logger.info("Strategy D (null_space): training on %d examples", len(dataset))
    trainer.train()
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))


def run_continual_learning(config_path: str, strategy: str, output_dir: str | None = None) -> None:
    """Run full CL pipeline across all time periods.

    1. Load manifest to get ordered list of periods.
    2. D0 should already be trained.
    3. For each subsequent period, call train_period().
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    exp_cfg = config["experiment"]
    manifest_path = Path(exp_cfg["output_dir"]) / "splits_manifest.json"

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Sort periods by start_date
    periods = sorted(manifest.items(), key=lambda x: x[1]["start_date"])
    period_names = [name for name, _ in periods]

    if len(period_names) < 2:
        logger.warning("Only %d period(s) found. CL requires at least 2.", len(period_names))
        return

    d0_name = period_names[0]
    d0_checkpoint = str(Path(exp_cfg["output_dir"]) / "d0" / d0_name)

    if not Path(d0_checkpoint).exists():
        raise FileNotFoundError(
            f"D0 checkpoint not found at {d0_checkpoint}. Train D0 first with train_d0.py."
        )

    prev_checkpoint = d0_checkpoint
    for period_name in period_names[1:]:
        logger.info("=== CL period: %s (strategy=%s) ===", period_name, strategy)
        prev_checkpoint = train_period(
            config_path, period_name, strategy,
            prev_checkpoint=prev_checkpoint, output_dir=output_dir,
        )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(description="Continual learning training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument(
        "--strategy",
        choices=["replay", "merge", "compose", "null_space"],
        required=True,
    )
    parser.add_argument("--period", type=str, default=None, help="Train specific period (or all)")
    parser.add_argument("--prev-checkpoint", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.period:
        train_period(args.config, args.period, args.strategy, args.prev_checkpoint, args.output_dir)
    else:
        run_continual_learning(args.config, args.strategy, args.output_dir)
