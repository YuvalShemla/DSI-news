"""Extend tokenizer with chrono-semantic DocID tokens and initialize embeddings from RQ centroids."""

from __future__ import annotations

import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def generate_special_tokens(
    year_range: tuple[int, int] = (2017, 2025),
    num_codebooks: int = 6,
    codebook_size: int = 256,
) -> list[str]:
    """Generate all 1557 special token strings in canonical order.

    Order: year tokens, month tokens, then RQ codebook tokens.
    """
    tokens = []
    for y in range(year_range[0], year_range[1] + 1):
        tokens.append(f"<year_{y}>")
    for m in range(1, 13):
        tokens.append(f"<month_{m:02d}>")
    for cb in range(num_codebooks):
        for code in range(codebook_size):
            tokens.append(f"<rq_{cb}_{code}>")
    return tokens


def extend_tokenizer(
    tokenizer,
    model,
    year_range: tuple[int, int] = (2017, 2025),
    num_codebooks: int = 6,
    codebook_size: int = 256,
) -> int:
    """Add special tokens to tokenizer and resize model embeddings.

    Returns the number of tokens added.
    """
    new_tokens = generate_special_tokens(year_range, num_codebooks, codebook_size)
    num_added = tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    logger.info(
        "Added %d special tokens (expected %d). New vocab size: %d",
        num_added, len(new_tokens), len(tokenizer),
    )
    return num_added


def initialize_rq_embeddings(
    model,
    tokenizer,
    rq_centroids: np.ndarray,
    num_codebooks: int = 6,
    codebook_size: int = 256,
) -> None:
    """Initialize RQ token embeddings from FAISS centroids.

    Args:
        rq_centroids: Array of shape (num_codebooks, codebook_size, centroid_dim).
                      If centroid_dim != model embed dim, a random projection is used.
    """
    embedding_weight = model.get_input_embeddings().weight
    model_dim = embedding_weight.shape[1]
    centroid_dim = rq_centroids.shape[2]

    if centroid_dim != model_dim:
        logger.info(
            "Projecting RQ centroids from %d to %d dims", centroid_dim, model_dim
        )
        projection = torch.randn(centroid_dim, model_dim) * (1.0 / centroid_dim**0.5)
        rq_projected = torch.from_numpy(rq_centroids).float() @ projection
    else:
        rq_projected = torch.from_numpy(rq_centroids).float()

    with torch.no_grad():
        for cb in range(num_codebooks):
            for code in range(codebook_size):
                token_str = f"<rq_{cb}_{code}>"
                token_id = tokenizer.convert_tokens_to_ids(token_str)
                embedding_weight.data[token_id] = rq_projected[cb, code].to(
                    embedding_weight.dtype
                )

    logger.info("Initialized %d RQ token embeddings", num_codebooks * codebook_size)


def initialize_chrono_embeddings(
    model,
    tokenizer,
    year_range: tuple[int, int] = (2017, 2025),
) -> None:
    """Initialize year/month token embeddings from mean embedding + noise."""
    embedding_weight = model.get_input_embeddings().weight
    mean_embed = embedding_weight.data.mean(dim=0)

    with torch.no_grad():
        for y in range(year_range[0], year_range[1] + 1):
            token_id = tokenizer.convert_tokens_to_ids(f"<year_{y}>")
            noise = torch.randn_like(mean_embed) * 0.01
            embedding_weight.data[token_id] = mean_embed + noise

        for m in range(1, 13):
            token_id = tokenizer.convert_tokens_to_ids(f"<month_{m:02d}>")
            noise = torch.randn_like(mean_embed) * 0.01
            embedding_weight.data[token_id] = mean_embed + noise

    num_chrono = (year_range[1] - year_range[0] + 1) + 12
    logger.info("Initialized %d chrono token embeddings", num_chrono)


def setup_tokenizer_and_embeddings(
    model,
    tokenizer,
    rq_centroids: np.ndarray | None = None,
    config: dict | None = None,
):
    """Full setup: extend tokenizer, initialize all embeddings.

    This is the main entry point for preparing the tokenizer and model
    embeddings for chrono-semantic DocIDs.
    """
    year_range = tuple(config["docid"]["year_range"]) if config else (2017, 2025)
    num_codebooks = config["docid"]["rq_codebooks"] if config else 6
    codebook_size = 2 ** config["docid"]["rq_bits"] if config else 256

    extend_tokenizer(tokenizer, model, year_range, num_codebooks, codebook_size)

    if rq_centroids is not None:
        initialize_rq_embeddings(
            model, tokenizer, rq_centroids, num_codebooks, codebook_size
        )

    initialize_chrono_embeddings(model, tokenizer, year_range)

    return model, tokenizer


def get_token_id_ranges(
    tokenizer,
    year_range: tuple[int, int] = (2017, 2025),
    num_codebooks: int = 6,
    codebook_size: int = 256,
) -> dict:
    """Get token ID ranges for each category. Useful for constrained decoding.

    Returns:
        Dict with keys "year" (list), "month" (list), "rq" (dict[int, list]).
    """
    year_ids = [
        tokenizer.convert_tokens_to_ids(f"<year_{y}>")
        for y in range(year_range[0], year_range[1] + 1)
    ]
    month_ids = [
        tokenizer.convert_tokens_to_ids(f"<month_{m:02d}>")
        for m in range(1, 13)
    ]
    rq_ids = {}
    for cb in range(num_codebooks):
        rq_ids[cb] = [
            tokenizer.convert_tokens_to_ids(f"<rq_{cb}_{code}>")
            for code in range(codebook_size)
        ]
    return {"year": year_ids, "month": month_ids, "rq": rq_ids}


if __name__ == "__main__":
    from pathlib import Path

    import yaml
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )

    config_path = Path(__file__).resolve().parent.parent.parent / "configs" / "default.yaml"
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model_name = cfg["model"]["backbone"]
    dtype = getattr(torch, cfg["model"].get("torch_dtype", "bfloat16"))

    logger.info("Loading base model: %s", model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    base_vocab = len(tokenizer)
    logger.info("Base vocab size: %d", base_vocab)

    model, tokenizer = setup_tokenizer_and_embeddings(model, tokenizer, config=cfg)

    logger.info("Extended vocab size: %d", len(tokenizer))
    logger.info("Embedding shape: %s", model.get_input_embeddings().weight.shape)

    ranges = get_token_id_ranges(tokenizer)
    logger.info("Year token IDs: %s", ranges["year"])
    logger.info("Month token IDs: %s", ranges["month"])
    logger.info("RQ codebook 0 first 5 IDs: %s", ranges["rq"][0][:5])
