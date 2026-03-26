import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import orjson
from transformers import TrainingArguments

local_rank = int(os.environ.get("LOCAL_RANK", 0))


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="t5-base")
    num_decoder_layers: Optional[int] = field(default=None)


@dataclass
class Arguments:
    teacher_score_path: Optional[str] = field(default=None)
    collection_path: str = field(default="./data/data/msmarco/full_collection")
    queries_path: str = field(default="./data/data/msmarco/train_queries/queries")
    qrels_path: str = field(default="./data/data/msmarco/train_queries/qrels.json")
    output_dir: str = field(default="./data/t5_pretrainer/t5_pretrainer/experiments/")
    example_path: Optional[str] = field(default=None)
    pseudo_queries_to_docid_path: Optional[str] = field(default=None)
    pseudo_queries_to_mul_docid_path: Optional[str] = field(default=None)
    docid_to_smtid_path: Optional[str] = field(default=None)
    docid_to_tokenids_path: Optional[str] = field(default=None)
    qid_to_smtid_path: Optional[str] = field(default=None)
    first_centroid_path: Optional[str] = field(default=None)
    second_centroid_path: Optional[str] = field(default=None)
    third_centroid_path: Optional[str] = field(default=None)
    qid_to_rrpids_path: Optional[str] = field(default=None)
    docid_decode_eval_path: Optional[str] = field(default=None)
    centroid_path: Optional[str] = field(default=None)
    centroid_idx: Optional[int] = field(default=None)
    triple_margin_mse_path: Optional[str] = field(default=None)
    query_to_docid_path: Optional[str] = field(default=None)
    teacher_rerank_nway_path: Optional[str] = field(default=None)
    bce_example_path: Optional[str] = field(default=None)
    smt_docid_to_smtid_path: Optional[str] = field(default=None)
    lex_docid_to_smtid_path: Optional[str] = field(default=None)

    run_name: str = field(default="t5_pretrainer_marginmse_tmp")
    pretrained_path: Optional[str] = field(default=None)
    loss_type: str = field(default="margin_mse")
    model_type: str = field(default="t5_docid_gen_encoder")
    do_eval: bool = field(default=False)

    max_length: int = field(default=256)
    learning_rate: float = field(default=1e-3)
    warmup_ratio: float = field(default=0.04)
    per_device_train_batch_size: int = field(default=256)
    logging_steps: int = field(default=50)
    max_steps: int = field(default=-1)
    epochs: int = field(default=3)
    task_names: Optional[str] = field(default=None)

    ln_to_weight: Dict[str, float] = field(
        default_factory=lambda: {"rank": 1.0}
    )
    nway_label_type: Optional[str] = field(default=None)
    nway_rrpids: int = field(default=24)
    nway: int = field(default=12)
    eval_steps: int = field(default=50)
    use_fp16: bool = field(default=False)
    multi_vocab_sizes: bool = field(default=False)
    save_steps: int = field(default=15_000)
    wandb_project_name: str = field(default="mixloradsi")
    pad_token_id: Optional[int] = field(default=None)
    smtid_as_docid: bool = field(default=False)
    apply_lex_loss: bool = field(default=False)

    # for evaluation
    eval_collection_path: Optional[str] = field(default=None)
    eval_queries_path: Optional[str] = field(default=None)
    full_rank_eval_qrel_path: Optional[str] = field(default=None)
    full_rank_eval_topk: int = field(default=200)
    full_rank_index_dir: Optional[str] = field(default=None)
    full_rank_out_dir: Optional[str] = field(default=None)

    def __post_init__(self):
        self.output_dir = os.path.join(self.output_dir, self.run_name, "checkpoint")
        if local_rank <= 0:
            os.makedirs(self.output_dir, exist_ok=True)

        if isinstance(self.task_names, list):
            pass
        elif isinstance(self.task_names, str):
            self.task_names = orjson.loads(self.task_names)
        else:
            raise ValueError(f"task_names: {self.task_names} don't have valid type.")

        # mannualy set ln_to_weight
        self.ln_to_weight = {}
        for ln in self.task_names:
            if ln == "rank":
                self.ln_to_weight[ln] = 1.0
            elif ln == "rank_4":
                self.ln_to_weight[ln] = 1.0
            elif ln == "query_reg":
                self.ln_to_weight[ln] = 0.01
            elif ln == "doc_reg":
                self.ln_to_weight[ln] = 0.008
            elif ln == "lexical_rank":
                self.ln_to_weight[ln] = 1.0
            elif ln == "dense_rank":
                self.ln_to_weight[ln] = 1.0
            else:
                print(type(self.task_names), self.task_names[0], self.task_names)
                raise ValueError(f"loss name: {ln} is not valid")
        print("task_names", self.task_names, self.ln_to_weight)


@dataclass
class TermEncoder_TrainingArguments(TrainingArguments):
    task_names: Optional[list] = field(default=None)
    ln_to_weight: Dict[str, float] = field(default=None)
    num_tasks: Optional[int] = field(default=None, init=False)
    full_rank_eval_qrel_path: Optional[str] = field(default=None)

    def __post_init__(self):
        super().__post_init__()

        self.num_tasks = len(self.task_names)


@dataclass
class EvalArguments:
    collection_path: str = field(
        default="./pag-data/msmarco/full_collection"
    )
    pretrained_path: str = field(default="")
    index_dir: str = field(default="")
    mmap_dir: str = field(default="")
    initial_mmap_dir: str = field(default="")
    target_mmap_dir: str = field(default="")
    out_dir: str = field(default="")
    flat_index_dir: str = field(default=None)
    model_name_or_path: str = field(default="t5-base")
    max_length: int = field(default=256)
    index_retrieve_batch_size: int = field(default=256)
    local_rank: int = field(default=-1)
    task: str = field(default="")
    model_cls_name: str = field(default="mixloradsi")
    q_collection_paths: str = field(default=None)
    eval_qrel_path: str = field(default=None)
    eval_metric: List[str] = field(default_factory=lambda: ["mrr_10", "recall"])
    docid_to_smtid_path: Optional[str] = field(default=None)
    docid_to_tokenids_path: Optional[str] = field(default=None)
    initial_docid_to_smtid_path: Optional[str] = field(default=None)
    qid_to_smtid_path: Optional[str] = field(default=None)
    use_fp16: bool = field(default=False)
    num_return_sequences: int = field(default=10)
    encoder_type: str = field(default="standard_encoder")
    train_query_dir: Optional[str] = field(default=None)
    out_qid_to_smtid_dir: Optional[str] = field(default=None)
    centroid_path: Optional[str] = field(default=None)
    centroid_idx: Optional[int] = field(default=None)
    dev_queries_path: str = field(
        default="./pag-data/msmarco/dev_queries/raw.tsv"
    )
    dev_qrels_path: str = field(
        default="./pag-data/msmarco-full/dev_qrel.json"
    )
    retrieve_json_path: str = field(default=None)
    codebook_num: int = field(default=8)
    codebook_bits: int = field(default=8)

    batch_size: Optional[int] = field(default=1024)
    topk: int = field(default=200)
    apply_log_softmax_for_scores: Optional[bool] = field(default=False)
    max_new_token: Optional[int] = field(default=None)
    max_new_token_for_docid: int = field(default=32)
    num_links: int = field(default=100)
    ef_construct: int = field(default=128)
    splade_threshold: float = field(default=0.0)
    smt_docid_to_smtid_path: Optional[str] = field(default=None)
    lex_docid_to_smtid_path: Optional[str] = field(default=None)
    lex_out_dir: Optional[str] = field(default=None)
    smt_out_dir: Optional[str] = field(default=None)
    lex_constrained: Optional[str] = field(default=None)
    get_qid_smtid_rankdata: bool = field(default=False)
    beir_dataset_path: str = field(default=None)
    beir_dataset: str = field(default=None)
    bow_topk: int = field(default=64)
    pooling: str = field(default="max")
    index_only: bool = field(default=False)
    retrieve_only: bool = field(default=False)
    mixlora_config_json_path: str = field(default=None)

    def __post_init__(self):
        assert self.encoder_type in [
            "standard_encoder",
            "sparse_project_encoder",
            "sparse_subset_k_project_encoder",
            "sparse_project_pretrain_encoder",
            "t5seq_pretrain_encoder",
            "t5seq_gumbel_max_encoder",
        ]


@dataclass
class RerankArguments:
    collection_path: str = field(default="./pag-data/msmarco/full_collection")
    out_dir: str = field(default="")
    model_name_or_path: str = field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    max_length: int = field(default=256)
    batch_size: int = field(default=256)
    q_collection_path: str = field(default="")
    run_json_path: str = field(default="")
    local_rank: int = field(default=-1)
    task: str = field(default=None)
    pseudo_queries_path: Optional[str] = field(default=None)
    docid_pseudo_qids_path: Optional[str] = field(default=None)
    json_type: str = field(default="jsonl")
    qid_smtid_docids_path: Optional[str] = field(default=None)

    # for sparse query_to_smtid evaluate 
    docid_to_smtid_path: str = field(default="")
    docid_to_tokenids_path: str = field(default="")
    pretrained_path: str = field(default="")
    dev_queries_path: str = field(default="./pag-data/msmarco/dev_queries/raw.tsv")
    dev_qrels_path: str = field(default="./pag-data/msmarco-full/dev_qrel.json")
    qid_docids_path: str = field(default="./pag-data/msmarco/bm25_run/top1000.dev.rerank.json")
    query_to_smtid_tokenizer_type: str = field(default="t5-base")
    qid_smtid_rank_path: str = field(default="")
    train_qrels_path: str = field(default="./pag-data/msmarco/train_queries/qrels.json")
    train_queries_path: str = field(default="./pag-data/msmarco/train_queries/queries/raw.tsv")
    qid_to_reldocid_hard_docids_path: Optional[str] = field(default=None)
    eval_qrel_path: Optional[str] = field(default=None)
    eval_metrics: Optional[str] = field(default=None)
    eval_qrel_path: Optional[str] = field(default=None)
    eval_metrics: Optional[str] = field(default=None)
