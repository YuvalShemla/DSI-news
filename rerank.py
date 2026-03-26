import os

import numpy as np
import torch
import ujson
from t5_pretrainer.arguments import RerankArguments
from t5_pretrainer.dataset import CrossEncRerankDataLoader, RerankDataset
from t5_pretrainer.ripor import CrossEncoder
from t5_pretrainer.utils.utils import ddp_setup, is_first_worker, set_seed
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import HfArgumentParser
from transformers.tokenization_utils_base import BatchEncoding

local_rank = int(os.environ.get("LOCAL_RANK", 0))
set_seed(42)


class Reranker:
    def __init__(self, model, dataloader, config, dtype=None, dataset_name=None, is_beir=False, write_to_disk=False, local_rank=-1):
        self.model = model
        self.dataloader = dataloader
        self.config = config
        self.model.eval()
        
        self.write_to_disk = write_to_disk
        if write_to_disk:
            self.out_dir = os.path.join(config["out_dir"], dataset_name) if (dataset_name is not None and not is_beir) \
            else config["out_dir"]
            if is_first_worker():
                if not os.path.exists(self.out_dir):
                    os.makedirs(self.out_dir)
        
        self.dtype = dtype
        self.local_rank = local_rank
        

    def reranking(self, name=None, is_biencoder=False, use_fp16=True, run_json_output=True):
        all_pair_ids = []
        all_scores = []
        for i, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    qd_kwargs = self.kwargs_to_cuda(batch["qd_kwargs"])
                    if hasattr(self.model, "module"):
                        scores = self.model.module.rerank_forward(qd_kwargs)["scores"].cpu().tolist()
                    else:
                        scores = self.model.rerank_forward(qd_kwargs)["scores"].cpu().tolist()
                    all_scores.extend(scores)
                    assert isinstance(batch["pair_ids"], list)
                    all_pair_ids.extend(batch["pair_ids"])
                        
        qid_to_rankdata = self.pair_ids_to_json_output(all_scores, all_pair_ids)
        
        if self.write_to_disk:
            out_path = os.path.join(self.out_dir, "rerank{}.json".format(f"_{name}" if name is not None else ""))
            print("Write reranking to path: {}".format(out_path))
            print("Contain {} queries, avg_doc_length = {:.3f}".format(len(qid_to_rankdata), 
                                                              np.mean([len(xs) for xs in qid_to_rankdata.values()])))
            with open(out_path, "w") as fout:
                if run_json_output:
                    ujson.dump(qid_to_rankdata, fout)
                else:
                    raise NotImplementedError
        
        return qid_to_rankdata

    def reranking_for_same_prefix_pair(self, name=None, is_biencoder=False, use_fp16=True, run_json_output=True, prefix_name=None):
        all_triple_ids = []
        all_scores = []
        for i, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    qd_kwargs = self.kwargs_to_cuda(batch["qd_kwargs"])
                    if hasattr(self.model, "module"):
                        scores = self.model.module.rerank_forward(qd_kwargs)["scores"].cpu().tolist()
                    else:
                        scores = self.model.rerank_forward(qd_kwargs)["scores"].cpu().tolist()
                    all_scores.extend(scores)
                    assert isinstance(batch["triple_ids"], list)
                    all_triple_ids.extend(batch["triple_ids"])
                        
        qid_to_rankdata = self.triple_ids_to_json_output(all_scores, all_triple_ids)
        
        if self.write_to_disk:
            if prefix_name is not None:
                out_path = os.path.join(self.out_dir, "{}{}.json".format(prefix_name, f"_{name}" if name is not None else ""))
            else:
                out_path = os.path.join(self.out_dir, "qid_to_smtid_to_rerank{}.json".format(f"_{name}" if name is not None else ""))
            print("Write qid_to_smtid_to_rerank to path: {}".format(out_path))
            #print("Contain {} queries, avg_doc_length = {:.3f}".format(len(qid_to_rankdata), 
            #                                                  np.mean([len(xs) for xs in qid_to_rankdata.values()])))
            with open(out_path, "w") as fout:
                if run_json_output:
                    ujson.dump(qid_to_rankdata, fout)
                else:
                    raise NotImplementedError
        
        return qid_to_rankdata

    def query_to_smtid_reranking(self, name=None, use_fp16=False, run_json_output=True):
        all_pair_ids = []
        all_scores = []
        for i, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs = {"tokenized_query": self.kwargs_to_cuda(batch["tokenized_query"]),
                     "labels": batch["labels"].to(self.local_rank)}
                    if hasattr(self.model, "module"):
                        scores = self.model.module.get_query_smtids_score(**inputs) #[bz, seq_len]
                    else:
                        scores = self.model.get_query_smtids_score(**inputs) #[bz, seq_len]

            scores = torch.sum(scores, dim=-1).cpu().tolist() #[bz]
            all_scores.extend(scores)
            all_pair_ids.extend(batch["pair_ids"])

        qid_to_rankdata = self.pair_ids_to_json_output(all_scores, all_pair_ids)
        if self.write_to_disk:
            out_path = os.path.join(self.out_dir, "qid_smtids_rerank{}.json".format(f"_{name}" if name is not None else ""))
            print("Write reranking to path: {}".format(out_path))
            print("Contain {} queries, avg_doc_length = {:.3f}".format(len(qid_to_rankdata), 
                                                              np.mean([len(xs) for xs in qid_to_rankdata.values()])))
            with open(out_path, "w") as fout:
                if run_json_output:
                    ujson.dump(qid_to_rankdata, fout)
                else:
                    raise NotImplementedError
        
        return qid_to_rankdata
    
    def cond_prev_smtid_t5seq_encoder_reranking(self, name=None, use_fp16=False, run_json_output=True):
        all_pair_ids = []
        all_scores = []
        for i, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader)):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    inputs = {
                        "tokenized_query": self.kwargs_to_cuda(batch["tokenized_query"]),
                        "tokenized_doc": self.kwargs_to_cuda(batch["tokenized_doc"])}
                    inputs["prev_smtids"] = batch["prev_smtids"].to(self.local_rank)
                    if hasattr(self.model, "module"):
                        scores = self.model.module.cond_prev_smtid_query_doc_score(**inputs) #[bz, seq_len]
                    else:
                        scores = self.model.cond_prev_smtid_query_doc_score(**inputs) #[bz, seq_len]

            all_scores.extend(scores.cpu().tolist())
            all_pair_ids.extend(batch["pair_ids"])

        qid_to_rankdata = self.pair_ids_to_json_output(all_scores, all_pair_ids)
        if self.write_to_disk:
            out_path = os.path.join(self.out_dir, "cond_prev_smtid_rerank{}.json".format(f"_{name}" if name is not None else ""))
            print("Write reranking to path: {}".format(out_path))
            print("Contain {} queries, avg_doc_length = {:.3f}".format(len(qid_to_rankdata), 
                                                              np.mean([len(xs) for xs in qid_to_rankdata.values()])))
            with open(out_path, "w") as fout:
                if run_json_output:
                    ujson.dump(qid_to_rankdata, fout)
                else:
                    raise NotImplementedError
        
        return qid_to_rankdata

    def assign_scores_for_pseudo_queries(self, name=None, is_biencoder=False, use_fp16=True, run_json_output=True):
        all_pair_ids = []
        all_scores = []
        for i, batch in tqdm(enumerate(self.dataloader), total=len(self.dataloader), disable=self.local_rank>0):
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    qd_kwargs = self.kwargs_to_cuda(batch["qd_kwargs"])
                    if hasattr(self.model, "module"):
                        scores = self.model.module.rerank_forward(qd_kwargs)["scores"].cpu().tolist()
                    else:
                        scores = self.model.rerank_forward(qd_kwargs)["scores"].cpu().tolist()
                    all_scores.extend(scores)
                    assert isinstance(batch["pair_ids"], list)
                    all_pair_ids.extend(batch["pair_ids"])
                        
        pid_to_rankdata = self.pair_ids_to_docid_to_qids_json_output(all_scores, all_pair_ids)
        
        if self.write_to_disk:
            out_path = os.path.join(self.out_dir, "pid_qids_rerank_scores{}.json".format(f"_{name}" if name is not None else ""))
            print("Write reranking to path: {}".format(out_path))
            print("Contain {} docs, avg_doc_length = {:.3f}".format(len(pid_to_rankdata), 
                                                              np.mean([len(xs) for xs in pid_to_rankdata.values()])))
            with open(out_path, "w") as fout:
                if run_json_output:
                    ujson.dump(pid_to_rankdata, fout)
                else:
                    raise NotImplementedError
        
        return pid_to_rankdata
    
    def kwargs_to_cuda(self, input_kwargs):
        assert isinstance(input_kwargs, BatchEncoding) or isinstance(input_kwargs, dict), (type(input_kwargs))
        
        if self.local_rank != -1:
            return {k:v.to(self.local_rank) for k, v in input_kwargs.items()}
        else:
            return {k:v.cuda() for k, v in input_kwargs.items()}
    
    @staticmethod
    def pair_ids_to_json_output(scores, pair_ids):
        qid_to_rankdata = {}
        assert len(scores) == len(pair_ids)
        for i in range(len(scores)):
            s = float(scores[i])
            qid, pid = pair_ids[i]
            qid, pid = str(qid), str(pid)
            
            if qid not in qid_to_rankdata:
                qid_to_rankdata[qid] = {pid: s}
            else:
                qid_to_rankdata[qid][pid] = s
                
        return qid_to_rankdata
    
    @staticmethod
    def triple_ids_to_json_output(scores, triple_ids):
        qid_to_smtid_to_rankdata = {}
        for i in range(len(scores)):
            s = float(scores[i])
            qid, docid, smtid = triple_ids[i]
            qid, docid, smtid = str(qid), str(docid), str(smtid)

            if qid not in qid_to_smtid_to_rankdata:
                qid_to_smtid_to_rankdata[qid] = {smtid: [(docid, s)]}
            else:
                if smtid not in qid_to_smtid_to_rankdata[qid]:
                    qid_to_smtid_to_rankdata[qid][smtid] = [(docid, s)]
                else:
                    qid_to_smtid_to_rankdata[qid][smtid] += [(docid, s)]
        return qid_to_smtid_to_rankdata

    @staticmethod
    def pair_ids_to_docid_to_qids_json_output(scores, pair_ids):
        pid_to_rankdata = {}
        assert len(scores) == len(pair_ids)
        for i in range(len(scores)):
            s = float(scores[i])
            qid, pid = pair_ids[i]
            qid, pid = str(qid), str(pid)

            if pid not in pid_to_rankdata:
                pid_to_rankdata[pid] = {qid: s}
            else:
                pid_to_rankdata[pid][qid] = s 

        return pid_to_rankdata


def rerank_for_create_trainset(args):
    ddp_setup()
    print("model_name_or_path: ", args.model_name_or_path)
    model = CrossEncoder(args.model_name_or_path)
    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    run_json_path = args.run_json_path
    q_collection_path = args.q_collection_path

    rerank_dataset = RerankDataset(
        run_json_path=run_json_path,
        document_dir=args.collection_path,
        query_dir=q_collection_path,
        json_type=args.json_type,
    )
    rerank_loader = CrossEncRerankDataLoader(
        dataset=rerank_dataset,
        tokenizer_type=args.model_name_or_path,
        max_length=args.max_length,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        sampler=DistributedSampler(rerank_dataset) if local_rank != -1 else None,
    )
    reranker = Reranker(
        model=model,
        dataloader=rerank_loader,
        config={"out_dir": args.out_dir},
        write_to_disk=True,
        local_rank=local_rank,
    )
    reranker.reranking(name=f"{local_rank}", use_fp16=True)

    if os.path.exists(os.path.join(args.out_dir, "qid_pids_rerank_scores.train.json")):
        print("old qid_pids_rerank_scores.train.json exisit.")
        os.remove(os.path.join(args.out_dir, "qid_pids_rerank_scores.train.json"))

    sub_rerank_paths = [p for p in os.listdir(args.out_dir) if "rerank" in p]
    assert len(sub_rerank_paths) == torch.cuda.device_count(), (
        len(sub_rerank_paths),
        torch.cuda.device_count(),
    )
    qid_to_rankdata = {}
    for sub_path in sub_rerank_paths:
        with open(os.path.join(args.out_dir, sub_path)) as fin:
            sub_qid_to_rankdata = ujson.load(fin)
        if len(qid_to_rankdata) == 0:
            qid_to_rankdata.update(sub_qid_to_rankdata)
        else:
            for qid, rankdata in sub_qid_to_rankdata.items():
                if qid not in qid_to_rankdata:
                    qid_to_rankdata[qid] = rankdata
                else:
                    qid_to_rankdata[qid].update(rankdata)
    print(
        "length of qids and avg rankdata length in qid_to_rankdata: {}, {}".format(
            len(qid_to_rankdata), np.mean([len(xs) for xs in qid_to_rankdata.values()])
        )
    )

    qid_to_sorteddata = {}
    for qid, rankdata in qid_to_rankdata.items():
        qid_to_sorteddata[qid] = dict(
            sorted(rankdata.items(), key=lambda x: x[1], reverse=True)
        )

    with open(
        os.path.join(args.out_dir, "qid_docids_teacher_scores.train.json"), "w"
    ) as fout:
        for qid, rankdata in qid_to_sorteddata.items():
            example = {"qid": qid, "docids": [], "scores": []}
            for i, (pid, score) in enumerate(rankdata.items()):
                example["docids"].append(pid)
                example["scores"].append(score)
            fout.write(ujson.dumps(example) + "\n")
    print("end write to json")

    for sub_path in sub_rerank_paths:
        os.remove(os.path.join(args.out_dir, sub_path))


def main():
    parser = HfArgumentParser((RerankArguments))
    args = parser.parse_args_into_dataclasses()[0]
    rerank_for_create_trainset(args)


if __name__ == "__main__":
    main()
