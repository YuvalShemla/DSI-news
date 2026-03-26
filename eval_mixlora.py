import os
from pathlib import Path
from pdb import set_trace as st

import msgspec
import numpy as np
import torch
import ujson
from t5_pretrainer.arguments import EvalArguments
from t5_pretrainer.dataset import (
    CollectionDataLoaderForRiporGeneration,
    CollectionDatasetPreLoad,
)
from t5_pretrainer.index import evaluate
from t5_pretrainer.mixlora import MixLoraDSI
from t5_pretrainer.mixlora_config import MixLoraConfig
from t5_pretrainer.promptdsi import PromptDSI
from t5_pretrainer.ripor import RiporForSeq2seq
from t5_pretrainer.utils.prefixer import Prefixer
from t5_pretrainer.utils.utils import (
    convert_ptsmtids_to_strsmtid,
    ddp_setup,
    get_dataset_name,
    set_seed,
)

# from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import AutoTokenizer, HfArgumentParser
from transformers.generation import GenerationConfig

local_rank = 0 # int(os.environ["LOCAL_RANK"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)

model_dict = {
    "ripor": RiporForSeq2seq,
    "mixloradsi": MixLoraDSI,
    "promptdsi": PromptDSI,
}


def get_model(args):
    with open(args.mixlora_config_json_path, "rb") as f:
        mixlora_config = msgspec.json.Decoder().decode(f.read())
    mixlora_config = MixLoraConfig.from_config(mixlora_config)

    model_cls_name = args.model_cls_name.lower()
    model_cls = model_dict[model_cls_name]

    model = model_cls.from_pretrained(
        model_name_or_path=args.pretrained_path,
        mixlora_config=mixlora_config,
    )

    checkpoint = {}
    if (Path(args.pretrained_path) / "model.safetensors").exists():
        with safe_open(Path(args.pretrained_path) / "model.safetensors", framework="pt", device="cpu") as f:  # type: ignore
            # This is most likely t5-self-neg checkpoint
            for k in f.keys():
                checkpoint[k] = f.get_tensor(k)
    elif (Path(args.pretrained_path) / "pytorch_model.bin").exists():
        checkpoint = torch.load(
            os.path.join(args.pretrained_path, "pytorch_model.bin"), map_location="cpu"
        )

    return model, checkpoint


### Override the beam search method of GenerationMixin in  transformers/generation/utils.py


def constrained_decode_doc(
    model,
    dataloader,
    prefixer,
    smtid_to_docids,
    max_new_token,
    device,
    out_dir,
    local_rank,
    topk=100,
    get_qid_smtid_rankdata=False,
):

    qid_to_rankdata = {}

    generation_config = GenerationConfig.from_model_config(model.config)

    for i, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        with torch.no_grad():
            inputs = {k:v.to(device) for k, v in batch.items() if k != "id"}

            outputs = model.generate(
                inputs=inputs["input_ids"].long(),
                generation_config=generation_config,
                prefix_allowed_tokens_fn=prefixer,
                attention_mask=inputs["attention_mask"].long(),
                max_new_tokens=max_new_token,
                output_scores=True,
                return_dict=True,
                return_dict_in_generate=True,
                num_beams=topk,
                num_return_sequences=topk,
            )

        batch_qids = batch["id"].cpu().tolist()
        str_smtids = convert_ptsmtids_to_strsmtid(
            outputs.sequences.view(-1, topk, max_new_token + 1), max_new_token
        )
        relevant_scores = outputs.sequences_scores.view(-1, topk).cpu().tolist()

        for qid, ranked_smtids, rel_scores in zip(
            batch_qids, str_smtids, relevant_scores
        ):
            qid_to_rankdata[qid] = {}
            if get_qid_smtid_rankdata:  # Does not use
                for smtid, rel_score in zip(ranked_smtids, rel_scores):
                    qid_to_rankdata[qid][smtid] = {}
                    if smtid in smtid_to_docids:
                        for docid in smtid_to_docids[smtid]:
                            qid_to_rankdata[qid][smtid][docid] = (
                                rel_score * max_new_token
                            )
                    else:
                        print(f"smtid: {smtid} not in smtid_to_docid")
            else:
                for smtid, rel_score in zip(ranked_smtids, rel_scores):
                    if smtid not in smtid_to_docids:
                        # pass
                        print(f"smtid: {smtid} not in smtid_to_docid")
                    else:
                        for docid in smtid_to_docids[smtid]:
                            qid_to_rankdata[qid][docid] = rel_score * max_new_token

    if get_qid_smtid_rankdata:
        out_path = os.path.join(out_dir, f"qid_smtid_rankdata_{local_rank}.json")
        with open(out_path, "w") as fout:
            ujson.dump(qid_to_rankdata, fout)
    else:
        out_path = os.path.join(out_dir, f"run_{local_rank}.json")
        with open(out_path, "w") as fout:
            ujson.dump(qid_to_rankdata, fout)


def constrained_beam_search_for_qid_rankdata(args):
    # ddp_setup()
    model, _ = get_model(args)

    print("Pretrained model path: ", args.pretrained_path)

    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_path)

    # load prefixer
    # This is the docid maps to token ids of the PQ codes
    prefix_dir = os.path.dirname(args.docid_to_tokenids_path)
    if os.path.exists(os.path.join(prefix_dir, "prefix.pickle")):
        prefixer = Prefixer(
            docid_to_tokenids_path=None,
            tokenizer=None,
            prefix_path=os.path.join(prefix_dir, "prefix.pickle"),
        )
    else:
        prefixer = Prefixer(
            docid_to_tokenids_path=args.docid_to_tokenids_path, tokenizer=tokenizer
        )
    os.makedirs(args.out_dir, exist_ok=True)

    # define parameters for decoding
    with open(args.docid_to_tokenids_path) as fin:
        docid_to_tokenids = ujson.load(fin)
    assert args.max_new_token_for_docid in [
        2,
        4,
        6,
        8,
        16,
        24,
        32,
    ], args.max_new_token_for_docid

    # This add _ to connect the token ids
    smtid_to_docids = {}
    for docid, tokenids in docid_to_tokenids.items():
        assert tokenids[0] != -1, tokenids
        sid = "_".join([str(x) for x in tokenids[: args.max_new_token_for_docid]])
        if sid not in smtid_to_docids:
            smtid_to_docids[sid] = [docid]
        else:
            smtid_to_docids[sid] += [docid]

    max_new_token = args.max_new_token_for_docid
    assert len(sid.split("_")) == max_new_token, (sid, max_new_token)

    if local_rank <= 0:
        print(
            "distribution of docids length per smtid: ",
            np.quantile(
                [len(xs) for xs in smtid_to_docids.values()],
                [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0],
            ),
        )
        print(
            "avergen length = {:.3f}".format(
                np.mean([len(xs) for xs in smtid_to_docids.values()])
            )
        )
        print("smtid: ", sid)
    if not isinstance(args.q_collection_paths, list):
        args.q_collection_paths = [args.q_collection_paths]

    for data_dir in args.q_collection_paths:
        dev_dataset = CollectionDatasetPreLoad(data_dir=data_dir, id_style="row_id")
        dev_loader = CollectionDataLoaderForRiporGeneration(
            dataset=dev_dataset,
            tokenizer_type=args.pretrained_path,
            max_length=64,
            batch_size=args.batch_size,
            num_workers=4,
            # sampler=DistributedSampler(dev_dataset, shuffle=False),
        )

        model.to(device)
        if args.get_qid_smtid_rankdata:
            out_dir = args.out_dir
        else:
            out_dir = os.path.join(args.out_dir, get_dataset_name(data_dir))
        print("out_dir: ", out_dir)

        os.makedirs(out_dir, exist_ok=True)

        model.base_model.return_logits = True
        constrained_decode_doc(
            model.base_model,
            dev_loader,
            prefixer,
            smtid_to_docids,
            max_new_token,
            device=local_rank,
            out_dir=out_dir,
            local_rank=local_rank,
            topk=args.topk,
            get_qid_smtid_rankdata=args.get_qid_smtid_rankdata,
        )


def constrained_beam_search_for_qid_rankdata_2(args):

    if not isinstance(args.q_collection_paths, list):
        args.q_collection_paths = [args.q_collection_paths]

    for data_dir in args.q_collection_paths:
        out_dir = os.path.join(args.out_dir, get_dataset_name(data_dir))

        # remove old
        if os.path.exists(os.path.join(out_dir, "run.json")):
            print("old run.json exisit.")
            os.remove(os.path.join(out_dir, "run.json"))

        # merge
        qid_to_rankdata = {}
        sub_paths = [p for p in os.listdir(out_dir) if "run" in p]
        # assert len(sub_paths) == torch.cuda.device_count()
        for sub_path in sub_paths:
            with open(os.path.join(out_dir, sub_path)) as fin:
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
            "length of pids and avg rankdata length in qid_to_rankdata: {}, {}".format(
                len(qid_to_rankdata),
                np.mean([len(xs) for xs in qid_to_rankdata.values()]),
            )
        )

        with open(os.path.join(out_dir, "run.json"), "w") as fout:
            ujson.dump(qid_to_rankdata, fout, indent=4)

        for sub_path in sub_paths:
            sub_path = os.path.join(out_dir, sub_path)
            os.remove(sub_path)

    evaluate(args)


def main():
    parser = HfArgumentParser((EvalArguments))
    args = parser.parse_args_into_dataclasses()[0]

    constrained_beam_search_for_qid_rankdata(args)
    constrained_beam_search_for_qid_rankdata_2(args)


if __name__ == "__main__":
    main()
