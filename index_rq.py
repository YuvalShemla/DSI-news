import os
from pathlib import Path
from pdb import set_trace as st

import faiss
import numpy as np
import torch
import ujson
from t5_pretrainer.arguments import EvalArguments
from t5_pretrainer.dataset import CollectionDatasetPreLoad, T5DenseCollectionDataLoader
from t5_pretrainer.index import dense_indexing, evaluate
from t5_pretrainer.ripor import T5DenseEncoder
from t5_pretrainer.utils.utils import get_dataset_name, is_first_worker, set_seed
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
from transformers import HfArgumentParser
from transformers.modeling_utils import unwrap_model

set_seed(42)

class RQIndexer:
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args

    @staticmethod
    def index(mmap_dir, codebook_num, index_dir, codebook_bits):
        doc_embeds = np.memmap(
            os.path.join(mmap_dir, "doc_embeds.mmap"), dtype=np.float32, mode="r"
        ).reshape(-1, 768)
        faiss.omp_set_num_threads(16)

        d = doc_embeds.shape[1] # dimension of the vectors
        m = codebook_num        # number of subquantizers, 24
        n_bits = codebook_bits  # bits allocated per subquantizer, 8
        # int d, size_t M, size_t nbits,
        rq = faiss.IndexResidualQuantizer(d, m, n_bits, faiss.METRIC_INNER_PRODUCT)
        print("m: {}, n_bits: {}".format(m, n_bits))
        print("start rq training")
        print("shape of doc_embeds: ", doc_embeds.shape)

        rq.verbose = True
        rq.train(doc_embeds)
        rq.add(doc_embeds)

        print(f"writing index to {os.path.join(index_dir, 'model.index')}")
        faiss.write_index(rq, os.path.join(index_dir, "model.index"))

    def search(self, collection_dataloader, topk, index_path, out_dir, index_ids_path):
        query_embs, query_ids = self._get_embeddings_from_scratch(
            collection_dataloader, use_fp16=False, is_query=True
        )
        query_embs = (
            query_embs.astype(np.float32)
            if query_embs.dtype == np.float16
            else query_embs
        )

        index = faiss.read_index(index_path)
        # index = self._convert_index_to_gpu(index, list(range(8)), False)
        idx_to_docid = self._read_text_ids(index_ids_path)

        qid_to_rankdata = {}
        all_scores, all_idxes = index.search(query_embs, topk)
        for qid, scores, idxes in zip(query_ids, all_scores, all_idxes):
            docids = [idx_to_docid[idx] for idx in idxes]
            qid_to_rankdata[str(qid)] = {}
            for docid, score in zip(docids, scores):
                qid_to_rankdata[str(qid)][str(docid)] = float(score)

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)

        with open(os.path.join(out_dir, "run.json"), "w") as fout:
            ujson.dump(qid_to_rankdata, fout)

    def flat_index_search(self, collection_loader, topk, index_path, out_dir):
        query_embs, query_ids = self._get_embeddings_from_scratch(
            collection_loader, use_fp16=False, is_query=True
        )
        query_embs = (
            query_embs.astype(np.float32)
            if query_embs.dtype == np.float16
            else query_embs
        )

        index = faiss.read_index(index_path)
        index = self._convert_index_to_gpu(index, list(range(8)), False)

        nn_scores, nn_doc_ids = index.search(query_embs, topk)

        qid_to_ranks = {}
        for qid, docids, scores in zip(query_ids, nn_doc_ids, nn_scores):
            for docid, s in zip(docids, scores):
                if str(qid) not in qid_to_ranks:
                    qid_to_ranks[str(qid)] = {str(docid): float(s)}
                else:
                    qid_to_ranks[str(qid)][str(docid)] = float(s)

        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        with open(os.path.join(out_dir, "run.json"), "w") as fout:
            ujson.dump(qid_to_ranks, fout)

    def _get_embeddings_from_scratch(
        self, collection_loader, use_fp16=False, is_query=True
    ):
        model = self.model

        embeddings = []
        embeddings_ids = []
        for _, batch in tqdm(
            enumerate(collection_loader),
            disable=not is_first_worker(),
            desc=f"encode # {len(collection_loader)} seqs",
            total=len(collection_loader),
        ):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs = {k: v.cuda() for k, v in batch.items() if k != "id"}
                    # reps = model(**inputs)
                    if is_query:
                        reps = unwrap_model(model).query_encode(**inputs)
                    else:
                        raise NotImplementedError
                    text_ids = batch["id"].tolist()

            embeddings.append(reps.cpu().numpy())
            assert isinstance(text_ids, list)
            embeddings_ids.extend(text_ids)

        embeddings = np.concatenate(embeddings)

        assert len(embeddings_ids) == embeddings.shape[0]
        assert isinstance(embeddings_ids[0], int)
        print(f"# nan in embeddings: {np.sum(np.isnan(embeddings))}")

        return embeddings, embeddings_ids

    def _convert_index_to_gpu(self, index, faiss_gpu_index, useFloat16=False):
        if type(faiss_gpu_index) == list and len(faiss_gpu_index) == 1:
            faiss_gpu_index = faiss_gpu_index[0]
        if isinstance(faiss_gpu_index, int):
            res = faiss.StandardGpuResources()
            res.setTempMemory(1024 * 1024 * 1024)
            co = faiss.GpuClonerOptions()
            co.useFloat16 = useFloat16
            index = faiss.index_cpu_to_gpu(res, faiss_gpu_index, index, co)
        else:
            gpu_resources = []
            for i in range(torch.cuda.device_count()):
                res = faiss.StandardGpuResources()
                res.setTempMemory(256 * 1024 * 1024)
                gpu_resources.append(res)
            print(f"length of gpu_resources : {len(gpu_resources)}.")

            assert isinstance(faiss_gpu_index, list)
            vres = faiss.GpuResourcesVector()
            vdev = faiss.IntVector()
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True
            co.useFloat16 = useFloat16
            for i in faiss_gpu_index:
                vdev.push_back(i)
                vres.push_back(gpu_resources[i])
            index = faiss.index_cpu_to_gpu_multiple(vres, vdev, index, co)

        return index

    def _read_text_ids(self, text_ids_path):
        idx_to_docid = {}
        with open(text_ids_path) as fin:
            for idx, line in enumerate(fin):
                docid = line.strip()
                idx_to_docid[idx] = docid

        print("size of idx_to_docid = {}".format(len(idx_to_docid)))
        return idx_to_docid


def rq_indexing(args):
    if not os.path.exists(args.index_dir):
        os.mkdir(args.index_dir)
        
    print("start rq indexing")
    RQIndexer.index(
        args.mmap_dir, index_dir=args.index_dir, codebook_num=args.codebook_num, codebook_bits=args.codebook_bits)


def rq_retrieve(args):
    print("pretrained_path: ", args.pretrained_path)
    model = T5DenseEncoder.from_pretrained(args.pretrained_path)
    model.cuda()

    rq_indexer = RQIndexer(model, args)
    batch_size = 1024

    if not os.path.exists(args.out_dir):
        os.mkdir(args.out_dir)

    # if len(args.q_collection_paths) == 1:
    #     args.q_collection_paths = ujson.loads(args.q_collection_paths[0])
    #     print(args.q_collection_paths)

    if not isinstance(args.q_collection_paths, list):
        args.q_collection_paths = [args.q_collection_paths]

    for data_dir in args.q_collection_paths:
        q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        q_loader = T5DenseCollectionDataLoader(dataset=q_collection, tokenizer_type=args.pretrained_path,
                                        max_length=args.max_length, batch_size=batch_size,
                                        shuffle=False, num_workers=4)

        print("index_path: ", os.path.join(args.index_dir, "model.index"))
        print("out_dir: ", os.path.join(args.out_dir, get_dataset_name(data_dir)))
        print("index_ids_path: ", os.path.join(args.mmap_dir, "text_ids.tsv"))
        rq_indexer.search(
            q_loader, 
            topk=1000, 
            index_path=os.path.join(args.index_dir, "model.index"),
            out_dir=os.path.join(args.out_dir, get_dataset_name(data_dir)),
            index_ids_path=os.path.join(str(Path(args.mmap_dir).parent / "mmap_rq"), "text_ids.tsv")
        )

    evaluate(args)


def main():
    print("starting...")
    parser = HfArgumentParser((EvalArguments))
    args = parser.parse_args_into_dataclasses()[0]
    
    if args.index_only:
        rq_indexing(args)
    elif args.retrieve_only:
        rq_retrieve(args)
    else:
        dense_indexing(args)
        rq_indexing(args)
        rq_retrieve(args)


if __name__ == "__main__":
    main()
