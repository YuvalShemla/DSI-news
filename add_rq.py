import argparse
import os
import shutil
from copy import deepcopy
from typing import List

import faiss
import numpy as np
import torch
from index_rq import RQIndexer, rq_retrieve
from t5_pretrainer.dataset import CollectionDatasetPreLoad, T5DenseCollectionDataLoader
from t5_pretrainer.index import dense_indexing
from t5_pretrainer.ripor import T5DenseEncoder
from t5_pretrainer.utils.utils import set_seed
from tqdm import tqdm

set_seed(42)


class RQIndexerAdd(RQIndexer):
    def __init__(self, args):
        faiss.omp_set_num_threads(20)
        self.d = args.d  # dimension of the vectors
        self.m = args.codebook_num  # number of subquantizers, 24
        self.n_bits = args.codebook_bits  # bits allocated per subquantizer, 8
        # int d, size_t M, size_t nbits,
        self.rq = faiss.read_index(os.path.join(args.rq_index_dir, "model.index"))
        self.index_ids_path = os.path.join(args.flat_index_dir, "text_ids.tsv")
        self.idx_to_docid = self._read_text_ids(self.index_ids_path)

        self.args = args

    def add(self):
        doc_embeds = np.memmap(
            os.path.join(self.args.new_flat_index_dir, "doc_embeds.mmap"),
            dtype=np.float32,
            mode="r",
        ).reshape(-1, 768)

        faiss.omp_set_num_threads(20)
        self.rq.add(doc_embeds)
        os.makedirs(self.args.new_rq_index_dir, exist_ok=True)

        print(
            f"writing index to {os.path.join(self.args.new_rq_index_dir, 'model.index')}"
        )

        faiss.write_index(
            self.rq, os.path.join(self.args.new_rq_index_dir, "model.index")
        )


def main():
    print("starting...")
    parser = argparse.ArgumentParser()
    # RQ settings
    parser.add_argument("--index_retrieve_batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--d", type=int, default=768)
    parser.add_argument("--codebook_num", type=int, default=8)
    parser.add_argument("--codebook_bits", type=int, default=11)

    # Previous split
    parser.add_argument("--collection_path", type=str, required=True)
    parser.add_argument("--pretrained_path", type=str, required=True)
    parser.add_argument("--flat_index_dir", type=str, required=True)
    parser.add_argument("--rq_index_dir", type=str, required=True)

    # Current split
    parser.add_argument("--new_collection_path", type=str, required=True)
    parser.add_argument("--new_flat_index_dir", type=str, required=True)
    parser.add_argument("--new_rq_index_dir", type=str, required=True)
    parser.add_argument("--q_collection_paths", type=str, required=True)
    parser.add_argument("--eval_qrel_path", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--eval_metric", type=List[str], default=["mrr_10", "recall"])

    args = parser.parse_args()

    # Make a copy of args for dense indexing
    args_dense = deepcopy(args)
    args_dense.collection_path = args.new_collection_path
    args_dense.mmap_dir = args.new_flat_index_dir
    args_dense.index_dir = args.new_flat_index_dir

    # First step: dense indexing
    dense_indexing(args_dense)

    # 2. Update the text_ids.tsv (Note: for dO of MSMARCO, the text_ids.tsv is copied directly from the mmap_rq folder, since MSMARCO does not have a flat index folder, whatever???)

    # Copy the text_ids.tsv from previous split to current split index folder
    shutil.copy(
        os.path.join(args.flat_index_dir, "text_ids.tsv"),
        os.path.join(args.new_flat_index_dir, "previous_text_ids.tsv"),
    )
    # Append the text_ids.tsv from current split to the previous split
    with open(
        os.path.join(args.new_flat_index_dir, "previous_text_ids.tsv"), "a"
    ) as f_previous:
        with open(os.path.join(args.new_flat_index_dir, "text_ids.tsv"), "r") as f_current:
            for i, line in enumerate(f_current):
                f_previous.write(line)
    os.system(f"rm -rf {os.path.join(args.new_flat_index_dir, 'text_ids.tsv')}")
    os.system(f"mv {os.path.join(args.new_flat_index_dir, 'previous_text_ids.tsv')} {os.path.join(args.new_flat_index_dir, 'text_ids.tsv')}")

    # 3. add to D0's RQ index
    rq_indexer = RQIndexerAdd(args)
    rq_indexer.add()

    # 4. (Optional) evaluate the rq index performance
    # args_dense.mmap_dir = args.new_flat_index_dir
    # args_dense.index_dir = args.new_rq_index_dir
    # rq_retrieve(args_dense)


if __name__ == "__main__":
    main()
