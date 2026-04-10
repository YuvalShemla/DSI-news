"""
Encode documents into dense vectors using sentence-transformers.

Outputs numpy memmap files and text_ids.tsv for downstream RQ codebook training.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def embed_documents(
    parquet_path: str | Path,
    text_col: str,
    id_col: str,
    output_path: str | Path,
    model_name: str = "sentence-transformers/all-mpnet-base-v2",
    batch_size: int = 256,
) -> np.ndarray:
    """Encode documents and save as numpy memmap.

    Args:
        parquet_path: Path to Parquet file with documents.
        text_col: Column containing document text.
        id_col: Column containing document IDs.
        output_path: Directory to save embeddings (.mmap) and text_ids.tsv.
        model_name: Sentence-transformer model name.
        batch_size: Encoding batch size.

    Returns:
        Embeddings array of shape (n_docs, embed_dim).
    """
    from sentence_transformers import SentenceTransformer

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Loading documents from %s", parquet_path)
    df = pd.read_parquet(parquet_path)

    # Filter out empty texts
    df = df[df[text_col].notna() & (df[text_col].str.strip().str.len() > 0)].reset_index(drop=True)
    texts = df[text_col].tolist()
    doc_ids = df[id_col].tolist()
    logger.info("Encoding %d documents with %s", len(texts), model_name)

    model = SentenceTransformer(model_name)
    embed_dim = model.get_sentence_embedding_dimension()

    # Encode in batches with progress bar
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding"):
        batch = texts[i : i + batch_size]
        embs = model.encode(batch, show_progress_bar=False, convert_to_numpy=True)
        all_embeddings.append(embs)

    embeddings = np.vstack(all_embeddings).astype(np.float32)
    logger.info("Embeddings shape: %s", embeddings.shape)

    # Save as memmap
    mmap_path = output_path / "embeddings.mmap"
    fp = np.memmap(mmap_path, dtype="float32", mode="w+", shape=embeddings.shape)
    fp[:] = embeddings[:]
    fp.flush()
    del fp

    # Save shape metadata for reloading
    meta = {"shape": list(embeddings.shape), "dtype": "float32"}
    with open(output_path / "embeddings_meta.json", "w") as f:
        json.dump(meta, f)

    # Save doc_id -> row index mapping
    tsv_path = output_path / "text_ids.tsv"
    with open(tsv_path, "w") as f:
        for idx, doc_id in enumerate(doc_ids):
            f.write(f"{idx}\t{doc_id}\n")

    logger.info("Saved embeddings to %s (%d docs, %d dim)", mmap_path, *embeddings.shape)
    return embeddings


def embed_all_splits(manifest_path: str | Path, config_path: str | Path) -> dict:
    """Embed all splits referenced in a manifest file.

    Returns:
        Dict mapping split_name -> embedding directory path.
    """
    manifest_path = Path(manifest_path)
    config_path = Path(config_path)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    with open(manifest_path) as f:
        manifest = json.load(f)

    docid_cfg = cfg["docid"]
    model_name = docid_cfg.get("encoder_model", "sentence-transformers/all-mpnet-base-v2")
    batch_size = 256

    # Determine dataset name from manifest path (e.g. data/splits/bbc_news/splits_manifest.json)
    dataset_name = manifest_path.parent.name

    # Find the right data config for id_col/text_col
    data_cfg = cfg["data"].get(dataset_name, {})
    text_col = data_cfg.get("text_col", "text")
    id_col = data_cfg.get("id_col", "url")

    embedding_paths = {}
    for split_name, split_info in sorted(manifest.items()):
        parquet_path = Path(split_info["parquet_path"])
        output_dir = PROJECT_ROOT / "data" / "embeddings" / dataset_name / split_name

        if (output_dir / "embeddings.mmap").exists():
            logger.info("Embeddings already exist for %s, skipping", split_name)
            embedding_paths[split_name] = str(output_dir)
            continue

        logger.info("=== Embedding split %s (%d docs) ===", split_name, split_info["num_docs"])
        embed_documents(
            parquet_path=parquet_path,
            text_col=text_col,
            id_col=id_col,
            output_path=output_dir,
            model_name=model_name,
            batch_size=batch_size,
        )
        embedding_paths[split_name] = str(output_dir)

    return embedding_paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Embed documents using sentence-transformers")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--split-manifest", type=str, default=None,
                        help="Path to splits_manifest.json (if omitted, embeds all datasets)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.split_manifest:
        embed_all_splits(args.split_manifest, args.config)
    else:
        # Find all manifests
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        for dataset_name in cfg["data"]:
            manifest_path = PROJECT_ROOT / "data" / "splits" / dataset_name / "splits_manifest.json"
            if manifest_path.exists():
                logger.info("Processing %s", dataset_name)
                embed_all_splits(manifest_path, args.config)
            else:
                logger.warning("No manifest found for %s at %s. Run temporal_splits.py first.",
                               dataset_name, manifest_path)


if __name__ == "__main__":
    main()
