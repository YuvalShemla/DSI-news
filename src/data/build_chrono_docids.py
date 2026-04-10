"""
Train FAISS Residual Quantizer and build chrono-semantic DocIDs.

DocID format: [year_code, month_code, rq_0, rq_1, rq_2, rq_3, rq_4, rq_5]
  - year_code: 0-8 indexing into [2017..2025]
  - month_code: 0-11 indexing into [Jan..Dec]
  - rq_0..rq_5: 0-255 from FAISS Residual Quantizer (M=6, nbits=8)

Special tokens (1557 total):
  9 year tokens + 12 month tokens + 6*256 RQ tokens
"""

import argparse
import json
import logging
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Special token generation
# ---------------------------------------------------------------------------

def generate_special_tokens(year_range: tuple[int, int] = (2017, 2025)) -> list[str]:
    """Generate the 1557 special token strings in canonical order.

    Order: year tokens, month tokens, RQ tokens.
    """
    tokens = []

    # Year tokens: <year_2017> .. <year_2025>
    for y in range(year_range[0], year_range[1] + 1):
        tokens.append(f"<year_{y}>")

    # Month tokens: <month_01> .. <month_12>
    for m in range(1, 13):
        tokens.append(f"<month_{m:02d}>")

    # RQ tokens: <rq_0_0> .. <rq_5_255>  (6 codebooks x 256 centroids)
    for cb in range(6):
        for c in range(256):
            tokens.append(f"<rq_{cb}_{c}>")

    return tokens


# ---------------------------------------------------------------------------
# RQ training and encoding
# ---------------------------------------------------------------------------

def train_rq_codebook(embeddings: np.ndarray, M: int = 6, nbits: int = 8) -> faiss.Index:
    """Train a FAISS Residual Quantizer on document embeddings.

    Args:
        embeddings: (n, d) float32 array.
        M: Number of codebooks.
        nbits: Bits per codebook (256 centroids when nbits=8).

    Returns:
        Trained faiss.IndexResidualQuantizer.
    """
    n, d = embeddings.shape
    logger.info("Training RQ: n=%d, d=%d, M=%d, nbits=%d", n, d, M, nbits)

    index = faiss.IndexResidualQuantizer(d, M, nbits)
    index.train(embeddings)
    logger.info("RQ training complete")
    return index


def compute_rq_codes(rq_index: faiss.Index, embeddings: np.ndarray,
                     M: int = 6, nbits: int = 8) -> list[list[int]]:
    """Compute RQ codes for each document.

    Returns:
        List of M-length code lists, one per document.
    """
    rq = rq_index.rq
    uint8_codes = rq.compute_codes(embeddings)

    doc_encodings = []
    for u8_code in uint8_codes:
        bs = faiss.BitstringReader(faiss.swig_ptr(u8_code), uint8_codes.shape[1])
        code = []
        for _ in range(M):
            code.append(bs.read(nbits))
        doc_encodings.append(code)

    return doc_encodings


# ---------------------------------------------------------------------------
# DocID construction
# ---------------------------------------------------------------------------

def _year_code(year: int, year_range: tuple[int, int]) -> int:
    """Map year to 0-based index. Clamp to range."""
    return max(0, min(year - year_range[0], year_range[1] - year_range[0]))


def _month_code(month: int) -> int:
    """Map month (1-12) to 0-based index."""
    return max(0, min(month - 1, 11))


def _build_token_id_lookup(special_tokens: list[str], vocab_offset: int) -> dict[str, int]:
    """Build token string -> absolute vocab index mapping."""
    return {tok: vocab_offset + i for i, tok in enumerate(special_tokens)}


def build_chrono_docids(
    manifest_path: str | Path,
    embeddings_dir: str | Path,
    output_dir: str | Path,
    config_path: str | Path,
) -> dict:
    """Train RQ on D0, assign chrono-semantic DocIDs to all documents.

    Returns:
        docid_to_smtid dict.
    """
    manifest_path = Path(manifest_path)
    embeddings_dir = Path(embeddings_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    with open(manifest_path) as f:
        manifest = json.load(f)

    docid_cfg = cfg["docid"]
    M = docid_cfg.get("rq_codebooks", 6)
    nbits = docid_cfg.get("rq_bits", 8)
    year_range = tuple(docid_cfg.get("year_range", [2017, 2025]))

    dataset_name = manifest_path.parent.name
    data_cfg = cfg["data"].get(dataset_name, {})
    date_col = data_cfg.get("date_col", "date")
    id_col = data_cfg.get("id_col", "url")

    sorted_splits = sorted(manifest.keys())
    d0_split = sorted_splits[0]
    logger.info("D0 split (for RQ training): %s", d0_split)

    # Load D0 embeddings
    d0_emb_dir = embeddings_dir / d0_split
    d0_meta = json.load(open(d0_emb_dir / "embeddings_meta.json"))
    d0_shape = tuple(d0_meta["shape"])
    d0_embeddings = np.memmap(d0_emb_dir / "embeddings.mmap", dtype="float32",
                              mode="r", shape=d0_shape)

    # Train RQ on D0
    rq_index = train_rq_codebook(np.array(d0_embeddings), M=M, nbits=nbits)

    # Save RQ index
    rq_path = output_dir / "rq_index.faiss"
    faiss.write_index(rq_index, str(rq_path))
    logger.info("Saved RQ index to %s", rq_path)

    # Generate special tokens and token ID lookup
    special_tokens = generate_special_tokens(year_range)
    # vocab_offset will be set when adding to tokenizer; for docid_to_tokenids we use
    # a placeholder offset of 0 and store relative indices. The model pipeline will
    # recompute absolute token IDs after vocab extension.
    n_year = year_range[1] - year_range[0] + 1
    n_month = 12

    # Process all splits
    docid_to_smtid = {}
    smtid_to_docids: dict[str, list[str]] = {}

    for split_name in sorted_splits:
        split_info = manifest[split_name]
        parquet_path = Path(split_info["parquet_path"])
        emb_dir = embeddings_dir / split_name

        logger.info("Processing split %s (%d docs)", split_name, split_info["num_docs"])

        # Load embeddings
        meta = json.load(open(emb_dir / "embeddings_meta.json"))
        shape = tuple(meta["shape"])
        embeddings = np.memmap(emb_dir / "embeddings.mmap", dtype="float32",
                               mode="r", shape=shape)

        # Load doc IDs and dates
        doc_ids = []
        with open(emb_dir / "text_ids.tsv") as f:
            for line in f:
                parts = line.strip().split("\t", 1)
                if len(parts) == 2:
                    doc_ids.append(parts[1])

        # Load dates from parquet
        df = pd.read_parquet(parquet_path, columns=[id_col, date_col])
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        date_lookup = dict(zip(df[id_col].astype(str), df[date_col]))

        # Compute RQ codes for this split
        rq_codes = compute_rq_codes(rq_index, np.array(embeddings), M=M, nbits=nbits)

        for i, doc_id in enumerate(doc_ids):
            doc_id_str = str(doc_id)
            dt = date_lookup.get(doc_id_str)

            if pd.isna(dt) or dt is None:
                yc, mc = 0, 0
            else:
                yc = _year_code(dt.year, year_range)
                mc = _month_code(dt.month)

            smtid = [yc, mc] + rq_codes[i]
            docid_to_smtid[doc_id_str] = smtid

            smtid_key = str(smtid)
            if smtid_key not in smtid_to_docids:
                smtid_to_docids[smtid_key] = []
            smtid_to_docids[smtid_key].append(doc_id_str)

    # Build docid_to_tokenids (relative token indices within special token list)
    docid_to_tokenids = {}
    for doc_id_str, smtid in docid_to_smtid.items():
        yc, mc = smtid[0], smtid[1]
        rq = smtid[2:]
        token_ids = [
            yc,                          # year token index (0..8)
            n_year + mc,                 # month token index (9..20)
        ]
        for cb_idx, code in enumerate(rq):
            token_ids.append(n_year + n_month + cb_idx * 256 + code)
        docid_to_tokenids[doc_id_str] = token_ids

    # Save outputs
    with open(output_dir / "docid_to_smtid.json", "w") as f:
        json.dump(docid_to_smtid, f)
    with open(output_dir / "docid_to_tokenids.json", "w") as f:
        json.dump(docid_to_tokenids, f)
    with open(output_dir / "smtid_to_docids.json", "w") as f:
        json.dump(smtid_to_docids, f)
    with open(output_dir / "special_tokens.json", "w") as f:
        json.dump(special_tokens, f)

    # Print uniqueness stats
    total_docs = len(docid_to_smtid)
    unique_smtids = len(smtid_to_docids)
    collisions = sum(1 for v in smtid_to_docids.values() if len(v) > 1)
    max_collision = max(len(v) for v in smtid_to_docids.values()) if smtid_to_docids else 0

    print(f"\n{'='*60}")
    print(f"  DocID Statistics -- {dataset_name}")
    print(f"{'='*60}")
    print(f"  Total documents:    {total_docs:,}")
    print(f"  Unique DocIDs:      {unique_smtids:,}")
    print(f"  Collision buckets:  {collisions:,} ({100*collisions/max(unique_smtids,1):.1f}%)")
    print(f"  Max collision size: {max_collision}")
    print(f"  Special tokens:     {len(special_tokens)}")
    print(f"{'='*60}\n")

    logger.info("Saved docid files to %s", output_dir)
    return docid_to_smtid


# ---------------------------------------------------------------------------
# Incremental merge for continual learning
# ---------------------------------------------------------------------------

def merge_docids_for_new_period(
    existing_docids_path: str | Path,
    new_parquet_path: str | Path,
    new_embeddings_path: str | Path,
    rq_index_path: str | Path,
    output_path: str | Path,
    date_col: str = "date",
    id_col: str = "url",
    year_range: tuple[int, int] = (2017, 2025),
    M: int = 6,
    nbits: int = 8,
) -> dict:
    """Compute DocIDs for new period docs and merge with existing.

    Uses the frozen RQ codebook from D0 training.
    """
    existing_docids_path = Path(existing_docids_path)
    new_embeddings_path = Path(new_embeddings_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load existing
    with open(existing_docids_path) as f:
        docid_to_smtid = json.load(f)

    # Load frozen RQ
    rq_index = faiss.read_index(str(rq_index_path))

    # Load new embeddings
    meta = json.load(open(new_embeddings_path / "embeddings_meta.json"))
    shape = tuple(meta["shape"])
    embeddings = np.memmap(new_embeddings_path / "embeddings.mmap",
                           dtype="float32", mode="r", shape=shape)

    # Load new doc IDs
    doc_ids = []
    with open(new_embeddings_path / "text_ids.tsv") as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                doc_ids.append(parts[1])

    # Load dates
    df = pd.read_parquet(new_parquet_path, columns=[id_col, date_col])
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    date_lookup = dict(zip(df[id_col].astype(str), df[date_col]))

    # Compute RQ codes
    rq_codes = compute_rq_codes(rq_index, np.array(embeddings), M=M, nbits=nbits)

    new_count = 0
    for i, doc_id in enumerate(doc_ids):
        doc_id_str = str(doc_id)
        if doc_id_str in docid_to_smtid:
            continue  # already assigned

        dt = date_lookup.get(doc_id_str)
        if pd.isna(dt) or dt is None:
            yc, mc = 0, 0
        else:
            yc = _year_code(dt.year, year_range)
            mc = _month_code(dt.month)

        docid_to_smtid[doc_id_str] = [yc, mc] + rq_codes[i]
        new_count += 1

    # Save merged
    with open(output_path / "docid_to_smtid.json", "w") as f:
        json.dump(docid_to_smtid, f)

    logger.info("Merged %d new DocIDs (total: %d)", new_count, len(docid_to_smtid))
    return docid_to_smtid


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build chrono-semantic DocIDs")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Process only this dataset (default: all)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    datasets = [args.dataset] if args.dataset else list(cfg["data"].keys())

    for dataset_name in datasets:
        manifest_path = PROJECT_ROOT / "data" / "splits" / dataset_name / "splits_manifest.json"
        embeddings_dir = PROJECT_ROOT / "data" / "embeddings" / dataset_name
        output_dir = PROJECT_ROOT / "data" / "docids" / dataset_name

        if not manifest_path.exists():
            logger.warning("No manifest for %s at %s. Run temporal_splits.py first.",
                           dataset_name, manifest_path)
            continue
        if not embeddings_dir.exists():
            logger.warning("No embeddings for %s at %s. Run embed_documents.py first.",
                           dataset_name, embeddings_dir)
            continue

        build_chrono_docids(manifest_path, embeddings_dir, output_dir, args.config)


if __name__ == "__main__":
    main()
