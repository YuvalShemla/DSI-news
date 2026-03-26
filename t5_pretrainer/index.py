import json
import os
import pickle
import time
from pdb import set_trace as st

import faiss
import numpy as np
import torch
import torch.distributed as dist
import ujson
from t5_pretrainer.dataset import CollectionDatasetPreLoad, T5DenseCollectionDataLoader
from t5_pretrainer.losses.regulariaztion import L0, L1
from t5_pretrainer.ripor import T5DenseEncoder
from t5_pretrainer.utils.metrics import load_and_evaluate
from t5_pretrainer.utils.utils import (
    ddp_setup,
    get_dataset_name,
    is_first_worker,
    makedir,
)
from torch.distributed import destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers.modeling_utils import unwrap_model

local_rank = int(os.environ.get("LOCAL_RANK", 0))


class DenseIndexing:
    def __init__(self, model, args):
        self.index_dir = args.index_dir
        if is_first_worker():
            if self.index_dir is not None:
                makedir(self.index_dir)
        self.model = model
        self.model.eval()
        self.args = args

    def index(self, collection_loader, use_fp16=False, hidden_dim=768):
        model = self.model

        faiss.omp_set_num_threads(20)
        index = faiss.IndexFlatIP(hidden_dim)
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
        index = faiss.IndexIDMap(index)

        for idx, batch in tqdm(
            enumerate(collection_loader),
            disable=not is_first_worker(),
            total=len(collection_loader),
        ):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs = {k: v.cuda() for k, v in batch.items() if k != "id"}
                    reps = model(**inputs)
                    text_ids = batch["id"].numpy()

            index.add_with_ids(reps.cpu().numpy().astype(np.float32), text_ids)

        faiss.write_index(index, index_path)

        return index

    def store_embs(
        self,
        collection_loader,
        local_rank,
        chunk_size=50_000,
        use_fp16=False,
        is_query=False,
        idx_to_id=None,
    ):
        model = self.model
        index_dir = self.index_dir
        write_freq = chunk_size // collection_loader.batch_size
        if is_first_worker():
            print(
                "write_freq: {}, batch_size: {}, chunk_size: {}".format(
                    write_freq, collection_loader.batch_size, chunk_size
                )
            )

        embeddings = []
        embeddings_ids = []

        chunk_idx = 0
        for idx, batch in tqdm(
            enumerate(collection_loader),
            disable=not is_first_worker(),
            desc=f"encode # {len(collection_loader)} seqs",
            total=len(collection_loader),
        ):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs = {
                        k: v.to(model.device) for k, v in batch.items() if k != "id"
                    }
                    if is_query:
                        raise NotImplementedError
                    else:
                        reps = unwrap_model(model).doc_encode(**inputs)
                    # reps = model(**inputs)
                    text_ids = batch["id"].tolist()

            embeddings.append(reps.cpu().numpy())
            assert isinstance(text_ids, list)
            embeddings_ids.extend(text_ids)

            if (idx + 1) % write_freq == 0:
                embeddings = np.concatenate(embeddings)
                embeddings_ids = np.array(embeddings_ids, dtype=np.int64)
                assert len(embeddings) == len(embeddings_ids), (
                    len(embeddings),
                    len(embeddings_ids),
                )

                text_path = os.path.join(
                    index_dir, "embs_{}_{}.npy".format(local_rank, chunk_idx)
                )
                id_path = os.path.join(
                    index_dir, "ids_{}_{}.npy".format(local_rank, chunk_idx)
                )
                np.save(text_path, embeddings)
                np.save(id_path, embeddings_ids)

                del embeddings, embeddings_ids
                embeddings, embeddings_ids = [], []

                chunk_idx += 1

        if len(embeddings) != 0:
            embeddings = np.concatenate(embeddings)
            embeddings_ids = np.array(embeddings_ids, dtype=np.int64)
            assert len(embeddings) == len(embeddings_ids), (
                len(embeddings),
                len(embeddings_ids),
            )
            print("last embedddings shape = {}".format(embeddings.shape))
            text_path = os.path.join(
                index_dir, "embs_{}_{}.npy".format(local_rank, chunk_idx)
            )
            id_path = os.path.join(
                index_dir, "ids_{}_{}.npy".format(local_rank, chunk_idx)
            )
            np.save(text_path, embeddings)
            np.save(id_path, embeddings_ids)

            del embeddings, embeddings_ids
            chunk_idx += 1

        plan = {
            "nranks": dist.get_world_size(),
            "num_chunks": chunk_idx,
            "index_path": os.path.join(index_dir, "model.index"),
        }
        print("plan: ", plan)

        if is_first_worker():
            with open(os.path.join(self.index_dir, "plan.json"), "w") as fout:
                ujson.dump(plan, fout)

    def stat_sparse_project_encoder(
        self, collection_loader, use_fp16=False, apply_log_relu_logit=False
    ):
        model = self.model
        l0_scores, l1_scores = [], []
        docid_to_info = {}
        l0_fn = L0()
        l1_fn = L1()
        for i, batch in tqdm(
            enumerate(collection_loader),
            disable=not is_first_worker(),
            total=len(collection_loader),
        ):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs = {
                        k: v.to(model.device) for k, v in batch.items() if k != "id"
                    }
                    _, logits = unwrap_model(model).doc_encode_and_logit(**inputs)

            if apply_log_relu_logit:
                l0_scores.append(l0_fn(torch.log(1 + torch.relu(logits))).cpu().item())
                l1_scores.append(l1_fn(torch.log(1 + torch.relu(logits))).cpu().item())
            else:
                l0_scores.append(l0_fn(logits).cpu().item())
                l1_scores.append(l1_fn(logits).cpu().item())

            top_scores, top_cids = torch.topk(logits, k=128, dim=1)
            top_scores, top_cids = top_scores.cpu().tolist(), top_cids.cpu().tolist()
            for docid, scores, cids in zip(batch["id"], top_scores, top_cids):
                docid_to_info[docid.cpu().item()] = {"scores": scores, "cids": cids}

        return docid_to_info, np.mean(l0_scores), np.mean(l1_scores)

    @staticmethod
    def aggregate_embs_to_index(index_dir):
        with open(os.path.join(index_dir, "plan.json")) as fin:
            plan = ujson.load(fin)

        print("index_dir is: {}".format(index_dir))
        print("plan: ", plan)

        nranks = plan["nranks"]
        num_chunks = plan["num_chunks"]
        index_path = plan["index_path"]

        # start index
        text_embs, text_ids = [], []
        for i in range(nranks):
            for chunk_idx in range(num_chunks):
                text_embs.append(
                    np.load(
                        os.path.join(index_dir, "embs_{}_{}.npy".format(i, chunk_idx))
                    )
                )
                text_ids.append(
                    np.load(
                        os.path.join(index_dir, "ids_{}_{}.npy".format(i, chunk_idx))
                    )
                )

        text_embs = np.concatenate(text_embs)
        text_embs = (
            text_embs.astype(np.float32) if text_embs.dtype == np.float16 else text_embs
        )
        text_ids = np.concatenate(text_ids)

        assert len(text_embs) == len(text_ids), (len(text_embs), len(text_ids))
        assert text_ids.ndim == 1, text_ids.shape
        print("embs dtype: ", text_embs.dtype, "embs size: ", text_embs.shape)
        print("ids dtype: ", text_ids.dtype)

        index = faiss.IndexFlatIP(text_embs.shape[1])
        index = faiss.IndexIDMap(index)

        # assert isinstance(text_ids, list)
        # text_ids = np.array(text_ids)

        index.add_with_ids(text_embs, text_ids)
        faiss.write_index(index, index_path)

        meta = {"text_ids": text_ids, "num_embeddings": len(text_ids)}
        print("meta data for index: {}".format(meta))
        with open(os.path.join(index_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

        # remove embs, ids
        for i in range(nranks):
            for chunk_idx in range(num_chunks):
                os.remove(
                    os.path.join(index_dir, "embs_{}_{}.npy".format(i, chunk_idx))
                )
                os.remove(os.path.join(index_dir, "ids_{}_{}.npy".format(i, chunk_idx)))

    @staticmethod
    def aggregate_embs_to_mmap(mmap_dir):
        with open(os.path.join(mmap_dir, "plan.json")) as fin:
            plan = ujson.load(fin)

        print("mmap_dir is: {}".format(mmap_dir))
        print("plan: ", plan)

        nranks = plan["nranks"]
        num_chunks = plan["num_chunks"]
        index_path = plan["index_path"]

        # start index
        text_embs, text_ids = [], []
        for i in range(nranks):
            for chunk_idx in range(num_chunks):
                text_embs.append(
                    np.load(
                        os.path.join(mmap_dir, "embs_{}_{}.npy".format(i, chunk_idx))
                    )
                )
                text_ids.append(
                    np.load(
                        os.path.join(mmap_dir, "ids_{}_{}.npy".format(i, chunk_idx))
                    )
                )

        text_embs = np.concatenate(text_embs)
        text_embs = (
            text_embs.astype(np.float32) if text_embs.dtype == np.float16 else text_embs
        )
        text_ids = np.concatenate(text_ids)

        assert len(text_embs) == len(text_ids), (len(text_embs), len(text_ids))
        assert text_ids.ndim == 1, text_ids.shape
        print("embs dtype: ", text_embs.dtype, "embs size: ", text_embs.shape)
        print("ids dtype: ", text_ids.dtype)

        faiss.omp_set_num_threads(20)
        index = faiss.IndexFlatIP(text_embs.shape[1])
        index = faiss.IndexIDMap(index)

        index.add_with_ids(text_embs, text_ids)
        faiss.write_index(index, index_path)

        fp = np.memmap(
            os.path.join(mmap_dir, "doc_embeds.mmap"),
            dtype=np.float32,
            mode="w+",
            shape=text_embs.shape,
        )

        total_num = 0
        chunksize = 5_000
        for i in range(0, len(text_embs), chunksize):
            # generate some data or load a chunk of your data here. Replace `np.random.rand(chunksize, shape[1])` with your data.
            data_chunk = text_embs[i : i + chunksize]
            total_num += len(data_chunk)
            # make sure that the last chunk, which might be smaller than chunksize, is handled correctly
            if data_chunk.shape[0] != chunksize:
                fp[i : i + data_chunk.shape[0]] = data_chunk
            else:
                fp[i : i + chunksize] = data_chunk
        assert total_num == len(text_embs), (total_num, len(text_embs))

        with open(os.path.join(mmap_dir, "text_ids.tsv"), "w") as fout:
            for tid in text_ids:
                fout.write(f"{tid}\n")

        meta = {"text_ids": text_ids, "num_embeddings": len(text_ids)}
        print("meta data for index: {}".format(meta))
        with open(os.path.join(mmap_dir, "meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

        # remove embs, ids
        for i in range(nranks):
            for chunk_idx in range(num_chunks):
                os.remove(os.path.join(mmap_dir, "embs_{}_{}.npy".format(i, chunk_idx)))
                os.remove(os.path.join(mmap_dir, "ids_{}_{}.npy".format(i, chunk_idx)))


class DenseRetriever:
    def __init__(self, model, args, dataset_name, is_beir=False):
        self.index_dir = args.index_dir
        self.out_dir = (
            os.path.join(args.out_dir, dataset_name)
            if (dataset_name is not None and not is_beir)
            else args.out_dir
        )
        if is_first_worker():
            makedir(self.out_dir)
        if self.index_dir is not None:
            self.index_path = os.path.join(self.index_dir, "model.index")
        self.model = model
        self.model.eval()
        self.args = args

    def retrieve(
        self, collection_loader, topk, save_run=True, index=None, use_fp16=False
    ):
        query_embs, query_ids = self._get_embeddings_from_scratch(
            collection_loader, use_fp16=use_fp16, is_query=True
        )
        query_embs = (
            query_embs.astype(np.float32)
            if query_embs.dtype == np.float16
            else query_embs
        )

        if index is None:
            index = faiss.read_index(self.index_path)
            # index = self._convert_index_to_gpu(index, list(range(8)), False)
            index = self._convert_index_to_gpu(index, list(range(1)), False)

        start_time = time.time()
        nn_scores, nn_doc_ids = self._index_retrieve(index, query_embs, topk, batch=128)
        print("Flat index time spend: {:.3f}".format(time.time() - start_time))

        qid_to_ranks = {}
        for qid, docids, scores in zip(query_ids, nn_doc_ids, nn_scores):
            for docid, s in zip(docids, scores):
                if str(qid) not in qid_to_ranks:
                    qid_to_ranks[str(qid)] = {str(docid): float(s)}
                else:
                    qid_to_ranks[str(qid)][str(docid)] = float(s)

        if save_run:
            with open(os.path.join(self.out_dir, "run.json"), "w") as fout:
                ujson.dump(qid_to_ranks, fout)
            return {"retrieval": qid_to_ranks}
        else:
            return {"retrieval": qid_to_ranks}

    @staticmethod
    def get_first_smtid(model, collection_loader, use_fp16=False, is_query=True):
        model = model.base_model
        print("the flag for model.decoding: ", model.config.decoding)

        text_ids = []
        semantic_ids = []
        for _, batch in tqdm(
            enumerate(collection_loader),
            disable=not is_first_worker(),
            desc=f"encode # {len(collection_loader)} seqs",
            total=len(collection_loader),
        ):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs = {k: v.cuda() for k, v in batch.items() if k != "id"}
                    logits = model(**inputs).logits[0]  # [bz, vocab]
            smtids = torch.argmax(logits, dim=1).cpu().tolist()
            text_ids.extend(batch["id"].tolist())
            semantic_ids.extend(smtids)

        return text_ids, semantic_ids

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

    def _index_retrieve(self, index, query_embeddings, topk, batch=None):
        if batch is None:
            nn_scores, nearest_neighbors = index.search(query_embeddings, topk)
        else:
            query_offset_base = 0
            pbar = tqdm(total=len(query_embeddings))
            nearest_neighbors = []
            nn_scores = []
            while query_offset_base < len(query_embeddings):
                batch_query_embeddings = query_embeddings[
                    query_offset_base : query_offset_base + batch
                ]
                batch_nn_scores, batch_nn = index.search(batch_query_embeddings, topk)
                nearest_neighbors.extend(batch_nn.tolist())
                nn_scores.extend(batch_nn_scores.tolist())
                query_offset_base += len(batch_query_embeddings)
                pbar.update(len(batch_query_embeddings))
            pbar.close()

        return nn_scores, nearest_neighbors


def dense_indexing(args):
    ddp_setup()

    # # parallel initialiaztion
    assert local_rank != -1
    model = T5DenseEncoder.from_pretrained(args.pretrained_path)
    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    d_collection = CollectionDatasetPreLoad(data_dir=args.collection_path, id_style="row_id")
    d_loader = T5DenseCollectionDataLoader(dataset=d_collection, tokenizer_type=args.pretrained_path,
                                    max_length=args.max_length,
                                    batch_size=1024,
                                    num_workers=4,
                                sampler=DistributedSampler(d_collection, shuffle=False))

    print("start dense indexing")
    print("Pretrained path: ", args.pretrained_path)
    evaluator = DenseIndexing(model=model, args=args)
    evaluator.store_embs(d_loader, local_rank, use_fp16=False)

    print("done dense indexing")
    print("start aggregating embeddings to mmap")
    evaluator.aggregate_embs_to_mmap(args.mmap_dir)

    destroy_process_group()


def evaluate(args):

    if args.eval_qrel_path is None:
        print("missing eval_qrel_path")
        return

    if len(args.eval_qrel_path) == 1:
        args.eval_qrel_path = ujson.loads(args.eval_qrel_path[0])
    eval_qrel_path = args.eval_qrel_path
    eval_metric = [args.eval_metric] * len(eval_qrel_path)
    if hasattr(args, "out_dir"):
        out_dir = args.out_dir
    if hasattr(args, "output_dir"):
        out_dir = args.output_dir
    
    if not isinstance(eval_qrel_path, list):
        eval_qrel_path = [eval_qrel_path]
    
    res_all_datasets = {}
    for i, (qrel_file_path, eval_metrics) in enumerate(
        zip(eval_qrel_path, eval_metric)
    ):
        if qrel_file_path is not None:
            res = {}
            dataset_name = get_dataset_name(qrel_file_path)
            print("-" * 50)
            print(f"Evaluating on {dataset_name} queries from: ", qrel_file_path)
            print(dataset_name, eval_metrics)
            for metric in eval_metrics:
                res.update(
                    load_and_evaluate(
                        qrel_file_path=qrel_file_path,
                        run_file_path=os.path.join(out_dir, dataset_name, "run.json"),
                        metric=metric,
                    ) # type: ignore
                ) # type: ignore
            if dataset_name in res_all_datasets.keys():
                res_all_datasets[dataset_name].update(res)
            else:
                res_all_datasets[dataset_name] = res

            with open(os.path.join(out_dir, dataset_name, "perf.json"), "a") as fout:
                fout.write("\n")
            json.dump(res, open(os.path.join(out_dir, dataset_name, "perf.json"), "a"), indent=4)

    with open(
        os.path.join(out_dir, dataset_name, "perf_all_datasets.json"), "a"
    ) as fout:
        fout.write("\n")
    json.dump(
        res_all_datasets,
        open(os.path.join(out_dir, "perf_all_datasets.json"), "a"),
        indent=4,
    )
    return res_all_datasets


def dense_retrieve(args):
    model = T5DenseEncoder.from_pretrained(args.pretrained_path)
    model.cuda()

    batch_size = 256

    if not isinstance(args.q_collection_paths, list):
        args.q_collection_paths = [args.q_collection_paths]

    for data_dir in args.q_collection_paths:
        q_collection = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        q_loader = T5DenseCollectionDataLoader(
            dataset=q_collection,
            tokenizer_type=args.pretrained_path,
            max_length=args.max_length,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
        )
        print("q_collection: ", data_dir)
        print("get_dataset_name(data_dir)", get_dataset_name(data_dir))
        evaluator = DenseRetriever(
            model=model, args=args, dataset_name=get_dataset_name(data_dir)
        )
        evaluator.retrieve(q_loader, topk=args.topk)
    evaluate(args)
