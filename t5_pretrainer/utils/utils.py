import importlib.metadata
import importlib.util
import logging
import os
import random
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import ujson
from packaging import version
from torch.distributed import init_process_group


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_params_info(model):
    all_param = 0
    trainable_param = 0
    mixlora_param = 0
    mixlora_components = ["experts", "to_gates", "gate_"]
    
    print("\nAll trainable parameters:")
    for name, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            if "shared" in name:
                trainable_param += 768 * 2048 * 8
            else:
                trainable_param += param.numel()
            print(name, param.numel())
        if any([comp in name for comp in mixlora_components]):
            mixlora_param += param.numel()
    from pdb import set_trace as st; st()
    print_params_info(all_param, trainable_param, mixlora_param)


def print_params_info(all_param, trainable_param, mixlora_param):
    print(f" # all param       : {all_param}")
    print(f" # trainable param : {trainable_param}")
    print(f" # mixlora param   : {mixlora_param}")
    print(f" # % trainable parameters: {trainable_param/all_param*100:.2f}%")
    print(f" # % mixlora parameters  : {mixlora_param/all_param*100:.2f}%")


def ddp_setup():
    init_process_group(backend="nccl")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def is_package_available(
    pkg_name: str, pkg_version: Optional[str] = None
) -> Union[Tuple[bool, str], bool]:
    # Check we're not importing a "pkg_name" directory somewhere but the actual library by trying to grab the version
    package_exists = importlib.util.find_spec(pkg_name) is not None
    package_version = "N/A"
    if package_exists:
        try:
            package_version = importlib.metadata.version(pkg_name)
            package_exists = True
        except importlib.metadata.PackageNotFoundError:
            package_exists = False
        logging.debug(f"Detected {pkg_name} version {package_version}")
    if pkg_version is not None:
        return package_exists and version.parse(package_version) >= version.parse(
            pkg_version
        )
    else:
        return package_exists


class Unsubscribable:
    def __init__(self) -> None:
        raise RuntimeError(f"Instant unsubscribable class {__class__}")


# Class Placeholder for Bitsandbytes
class Linear8bitLt(Unsubscribable):
    def __init__(self) -> None:
        super().__init__()


class Linear4bit(Unsubscribable):
    def __init__(self) -> None:
        super().__init__()


def is_first_worker():
    return not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0


def makedir(dir_):
    if is_first_worker():
        if not os.path.exists(dir_):
            os.makedirs(dir_)
            

def get_dataset_name(path):
    # small (hard-coded !) snippet to get a dataset name from a Q_COLLECTION_PATH or a EVAL_QREL_PATH (full paths)
    path = str(path).lower()

    # return "longeval_train_2022-10"

    dataset_name = ""
    if "nq320k" in path:
        dataset_name = "nq320k"
    elif "msmarco" in path:
        dataset_name = "msmarco"
    elif "longeval" in path:
        dataset_name = "other_dataset"

    train_val_test = ""
    if "eval" in path:
        train_val_test = "eval"
    elif "train" in path:
        train_val_test = "train"

    splits = {"d0": "d0", "d1": "d1", "d2": "d2", "d3": "d3", "d4": "d4", "d5": "d5"}

    for k, v in splits.items():
        if k in path:
            split = v

    final_name = f"{dataset_name}_{train_val_test}_{split}"
    return final_name


def flatten_list(nested_list):
    out_list = []
    for cur_list in nested_list:
        for elem in cur_list:
            assert type(elem) != list, elem
            out_list.append(elem)

    return out_list


def convert_ptsmtids_to_strsmtid(input_smtids, seq_length):
    assert input_smtids.dim() == 3, input_smtids.dim()
    assert input_smtids.size(2) == seq_length + 1, (input_smtids.size(1), seq_length)

    input_smtids = input_smtids.cpu().tolist()
    out_list = []
    for beam_smtids in input_smtids:
        beam_list = []
        for smtids in beam_smtids:
            beam_list.append("_".join([str(x) for x in smtids[1:]]))
        out_list.append(beam_list)

    return out_list


def form_strsmtid_from_prefix_and_lastsmtids(
    prefix_smtid, last_smtids, last_smtids_scores
):
    assert isinstance(prefix_smtid, list) and isinstance(last_smtids, list), (
        prefix_smtid,
        last_smtids,
    )

    all_strsmtids, all_scores = [], []
    for prefix, lasts, lasts_scores in zip(
        prefix_smtid, last_smtids, last_smtids_scores
    ):
        strsmtids, scores = [], []
        for last, score in zip(lasts, lasts_scores):
            strsmtid = "_".join([str(p) for p in prefix]) + "_" + str(last)
            strsmtids.append(strsmtid)
            scores.append(score)
        all_strsmtids.append(strsmtids)
        all_scores.append(scores)

    return all_strsmtids, all_scores


def partition_fn(lst, num_partitions):
    if len(lst) < num_partitions:
        raise ValueError(
            f"list size is {len(lst)} which is smaller num_partitions: {num_partitions}"
        )
    partition_size = len(lst) // num_partitions
    partitions = [
        lst[i * partition_size : (i + 1) * partition_size]
        for i in range(num_partitions)
    ]
    # If the number of elements does not divide evenly into the number of partitions,
    # the last partition will take all remaining elements
    partitions[-1].extend(lst[num_partitions * partition_size :])
    return partitions


def sample_from_partitions(lst, num_partitions, num_samples):
    partitions = partition_fn(lst, num_partitions)
    num_samples_per_partition = num_samples // num_partitions
    remainder = num_samples % num_partitions

    samples = []

    for i, partition in enumerate(partitions):
        if i < remainder:
            # If there is a remainder, add one more sample to the first 'remainder' partitions
            samples.extend(random.sample(partition, num_samples_per_partition + 1))
        else:
            samples.extend(random.sample(partition, num_samples_per_partition))

    return samples


def from_qrel_to_qsmtid_rel(docid_to_smtid_path, qrels_path, truncate_smtid):
    with open(docid_to_smtid_path) as fin:
        docid_to_smtid = ujson.load(fin)

    if not truncate_smtid:
        docid_to_strsmtid = {}
        for docid, smtid in docid_to_smtid.items():
            assert smtid[0] == -1, smtid
            str_smtid = "_".join([str(x) for x in smtid[1:]])
            docid_to_strsmtid[docid] = str_smtid
    else:
        print("Warning: we truncate the smtid for smtid_tree evaluation")
        all_smtids = [list(xs) for xs in docid_to_smtid.values()]
        smtid_lengths = [len(xs) for xs in all_smtids]
        max_new_tokens = max(smtid_lengths) - 2  # -1 and eos_token_id is added

        docid_to_strsmtid = {}
        for docid, smtid in docid_to_smtid.items():
            assert smtid[0] == -1, smtid
            str_smtid = "_".join([str(x) for x in smtid[1 : 1 + max_new_tokens]])
            # print(str_smtid)
            docid_to_strsmtid[docid] = str_smtid

    with open(qrels_path) as fin:
        qrel_data = ujson.load(fin)
    qid_to_relsmtid_data = {}
    for qid in qrel_data:
        qid_to_relsmtid_data[qid] = {}
        for docid, s in qrel_data[qid].items():
            rel_smtid = docid_to_strsmtid[docid]
            qid_to_relsmtid_data[qid][rel_smtid] = s

    return qid_to_relsmtid_data


def to_list(tensor):
    return tensor.detach().cpu().tolist()


def to_device(batch, device):
    for key, value in batch.items():
        if isinstance(value, dict):  # If value is a dict, recurse
            batch[key] = to_device(value, device)
        elif isinstance(value, torch.Tensor):  # If value is a tensor, move to CUDA
            batch[key] = value.to(device=device)
    return batch


def get_qid_smtid_scores(qid_to_rankdata, docid_to_tokenids):
    docid_to_smtid = {}
    for i, (docid, tokenids) in enumerate(docid_to_tokenids.items()):
        if i == 0:
            if is_first_worker():
                print(
                    "length of tokenids: ",
                    len(tokenids),
                    "its type: ",
                    type(tokenids[0]),
                )
        smtid = "_".join([str(x) for x in tokenids])
        docid_to_smtid[str(docid)] = smtid

    qid_to_smtid_to_score = {}
    for qid, rankdata in qid_to_rankdata.items():
        qid_to_smtid_to_score[qid] = {}
        for docid, score in rankdata.items():
            smtid = docid_to_smtid[docid]
            if smtid not in qid_to_smtid_to_score[qid]:
                qid_to_smtid_to_score[qid][smtid] = score
            else:
                qid_to_smtid_to_score[qid][smtid] = max(
                    score, qid_to_smtid_to_score[qid][smtid]
                )

    return qid_to_smtid_to_score
