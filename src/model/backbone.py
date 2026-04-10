"""T5Gemma 2 backbone with PEFT LoRA for DSI."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

logger = logging.getLogger(__name__)


def load_backbone(
    model_name: str = "google/t5gemma-2-270m-270m",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Load T5Gemma 2 base model and tokenizer."""
    logger.info("Loading backbone: %s (dtype=%s)", model_name, torch_dtype)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch_dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def apply_lora(
    model,
    r: int = 16,
    lora_alpha: int = 32,
    target_modules: list[str] | None = None,
    lora_dropout: float = 0.05,
):
    """Apply LoRA adapter to the model. Returns PEFT model."""
    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
    )
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model


def load_trained_model(
    checkpoint_path: str | Path,
    base_model_name: str = "google/t5gemma-2-270m-270m",
    torch_dtype: torch.dtype = torch.bfloat16,
):
    """Load a trained PEFT model from checkpoint."""
    logger.info("Loading trained model from %s", checkpoint_path)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_name, torch_dtype=torch_dtype
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    return model, tokenizer


def load_model_from_config(config: dict):
    """Load model + LoRA from a config dict (from YAML).

    Expects config keys: model.backbone, model.torch_dtype, model.lora.*
    """
    model_cfg = config["model"]
    dtype = getattr(torch, model_cfg.get("torch_dtype", "bfloat16"))
    model, tokenizer = load_backbone(model_cfg["backbone"], dtype)

    lora_cfg = model_cfg.get("lora", {})
    model = apply_lora(
        model,
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("lora_alpha", 32),
        target_modules=lora_cfg.get(
            "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
        ),
        lora_dropout=lora_cfg.get("lora_dropout", 0.05),
    )
    return model, tokenizer


if __name__ == "__main__":
    import yaml

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model, tokenizer = load_model_from_config(cfg)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Total parameters: %d", total)
    logger.info("Trainable parameters: %d (%.2f%%)", trainable, 100.0 * trainable / total)
    logger.info("Vocab size: %d", len(tokenizer))
