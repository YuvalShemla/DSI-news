"""Train the initial DSI model (D0) on the first time period."""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

from src.model.backbone import apply_lora, load_backbone
from src.model.docid_tokenizer import setup_tokenizer_and_embeddings
from src.training.dataset import ChronoDocIDCollator, ChronoDocIDDataset

logger = logging.getLogger(__name__)


def train_d0(config_path: str, split_name: str, output_dir: str | None = None) -> None:
    """Main D0 training function.

    Steps:
        1. Load config
        2. Load T5Gemma 2 backbone
        3. Extend tokenizer with chrono-semantic tokens
        4. Optionally initialize RQ embeddings from centroids
        5. Apply LoRA
        6. Load D0 dataset
        7. Train with Seq2SeqTrainer
        8. Save model + tokenizer
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_cfg = config["model"]
    train_cfg = config["training"]["d0"]
    exp_cfg = config["experiment"]

    # Resolve output directory
    if output_dir is None:
        output_dir = str(Path(exp_cfg["output_dir"]) / "d0" / split_name)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load backbone
    dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))
    model, tokenizer = load_backbone(model_cfg["backbone"], dtype)

    # 2. Extend tokenizer and initialize embeddings
    rq_centroids = None
    centroids_path = Path(exp_cfg["output_dir"]) / "rq_centroids.npy"
    if centroids_path.exists():
        rq_centroids = np.load(centroids_path)
        logger.info("Loaded RQ centroids from %s", centroids_path)

    model, tokenizer = setup_tokenizer_and_embeddings(
        model, tokenizer, rq_centroids=rq_centroids, config=config
    )

    # 3. Apply LoRA
    lora_cfg = model_cfg.get("lora", {})
    model = apply_lora(
        model,
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        target_modules=lora_cfg.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
    )

    # 4. Load dataset
    data_dir = Path(exp_cfg["output_dir"]) / "splits" / split_name
    examples_path = data_dir / "train_examples.jsonl"
    docid_to_tokenids_path = Path(exp_cfg["output_dir"]) / "docid_to_tokenids.json"

    dataset = ChronoDocIDDataset(
        examples_path=examples_path,
        docid_to_tokenids_path=docid_to_tokenids_path,
        filter_ratio=config.get("evaluation", {}).get("filter_query_ratio", 0.3),
    )

    collator = ChronoDocIDCollator(
        tokenizer_path_or_name=model_cfg["backbone"],
        max_query_length=train_cfg.get("max_query_length", 64),
    )

    # 5. Training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_path),
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
        run_name=f"d0_{split_name}",
        remove_unused_columns=False,
        predict_with_generate=False,
    )

    # 6. Train
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    logger.info("Starting D0 training on split '%s' (%d examples)", split_name, len(dataset))
    trainer.train()

    # 7. Save
    trainer.save_model(str(output_path))
    tokenizer.save_pretrained(str(output_path))
    logger.info("D0 training complete. Model saved to %s", output_path)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    parser = argparse.ArgumentParser(description="Train initial DSI model (D0)")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--split-name", type=str, required=True, help="Name of D0 split")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    train_d0(args.config, args.split_name, args.output_dir)
