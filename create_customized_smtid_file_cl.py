import argparse
import os
import shutil
from pathlib import Path
from pdb import set_trace as st

import faiss
import numpy as np
import ujson
from tqdm import tqdm
from transformers import AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codebook_num", default=8, type=int)
    parser.add_argument("--codebook_bits", default=11, type=int)
    parser.add_argument("--previous_out_dir", default=None, type=str)
    parser.add_argument("--new_out_dir", default=None, type=str)
    parser.add_argument("--new_flat_index_dir", default=None, type=str)
    parser.add_argument("--new_rq_index_dir", default=None, type=str)
    parser.add_argument("--dataset", default="nq320k", type=str)
    # parser.add_argument("--id_to_key_path", default=None, type=str)
    return parser.parse_args()

def main():
    args = get_args()
    M = args.codebook_num
    bits = args.codebook_bits
    new_rq_index_dir = args.new_rq_index_dir
    previous_out_dir = args.previous_out_dir
    new_out_dir = args.new_out_dir
    if not os.path.exists(new_out_dir):
        os.mkdir(new_out_dir)

    ### Read new document embeddings ###
    doc_embeds = np.memmap(os.path.join(args.new_flat_index_dir, "doc_embeds.mmap"), dtype=np.float32, mode="r").reshape(-1,768)

    ### Read text_ids.tsv ###
    idx_to_docid = {}
    print("text_ids_path: ", os.path.join(args.new_flat_index_dir, "text_ids.tsv"))
    with open(os.path.join(args.new_flat_index_dir, "text_ids.tsv")) as fin:
        tsv = fin.readlines()
        for i, line in enumerate(tsv):
            docid = line.strip()
            idx_to_docid[i] = docid
    print("size of idx_to_docid = {}".format(len(idx_to_docid)))
    length_previous = len(idx_to_docid) - len(doc_embeds)

    print("new_rq_index_dir: ", new_rq_index_dir)
    index = faiss.read_index(os.path.join(new_rq_index_dir, "model.index"))

    if "pq" in Path(new_rq_index_dir).stem:
        pq = index.pq
    elif "rq" in Path(new_rq_index_dir).stem:
        pq = index.rq

    ### Retrieve new document RQ codes ###
    doc_encodings = []
    unit8_codes = pq.compute_codes(doc_embeds)
    for u8_code in unit8_codes:
        bs = faiss.BitstringReader(faiss.swig_ptr(u8_code), unit8_codes.shape[1])
        code = []
        for i in range(M): 
            code.append(bs.read(bits)) 
        doc_encodings.append(code)
        assert len(doc_encodings[-1]) == M, (len(doc_encodings[-1]), M)

    ### Save new document RQ codes ###
    docid_to_smtid = {}
    for idx, doc_enc in enumerate(doc_encodings):
        docid = idx_to_docid[length_previous+idx]
        docid_to_smtid[docid] = doc_enc
    print("size of docid_to_smtid = {}".format(len(docid_to_smtid)))

    print("writing docid_to_smtid to {}".format(os.path.join(new_out_dir, "docid_to_smtid.json")))
    print("-" * 50)
    with open(os.path.join(new_out_dir, "docid_to_smtid.json"), "w") as fout:
        ujson.dump(docid_to_smtid, fout)

    ### Print some statistic about the new document RQ codes ###
    smtid_to_docids = {}
    for docid, smtids in docid_to_smtid.items():
        smtid = "_".join([str(x) for x in smtids])
        if smtid not in smtid_to_docids:
            smtid_to_docids[smtid] = [docid]
        else:
            smtid_to_docids[smtid] += [docid] # Several documents have the same pq codes

    total_smtid = len(smtid_to_docids)
    lengths = np.array([len(x) for x in smtid_to_docids.values()]) # number of documents with the same pq codes
    unique_smtid_num = np.sum(lengths == 1)
    print("unique_smtid_num = {}, total_smtid = {}".format(unique_smtid_num, total_smtid))
    print("percentage of smtid is unique = {:.3f}".format(unique_smtid_num / total_smtid))
    print("distribution of lengths: ", np.quantile(lengths, [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]))
    print("-" * 50)

    ### Merge current split's docid_to_smtid.json with previosu split's docid_to_smtid.json ###
    shutil.copy(
        os.path.join(previous_out_dir, "docid_to_smtid.json"),
        os.path.join(new_out_dir, "previous_docid_to_smtid.json"),
    )

    with open(os.path.join(new_out_dir, "previous_docid_to_smtid.json")) as f_previous:
        previous_json = ujson.load(f_previous)
        length_previous = len(previous_json)
        with open(
            os.path.join(new_out_dir, "docid_to_smtid.json")
        ) as f_current:
            current_json = ujson.load(f_current)
            for k, v in current_json.items():
                previous_json[k] = v
        ujson.dump(previous_json, open(os.path.join(new_out_dir, "docid_to_smtid.json"), "w"))
    os.system(f"rm -rf {os.path.join(new_out_dir, 'previous_docid_to_smtid.json')}")

    with open(os.path.join(args.new_out_dir, "docid_to_smtid.json")) as fin:
        docid_to_smtid = ujson.load(fin)

    ### Create docid_to_tokenids.json

    # The tokenizer is fixed to D0 extended model
    # tokenizer = AutoTokenizer.from_pretrained(
    #     f"./mixloradsi/{args.dataset}/d0/experiments/t5-self-neg-marginmse-5e-4/extended_rq_token_checkpoint"
    # )
    tokenizer = AutoTokenizer.from_pretrained(
        f"./mixloradsi/longeval/d0/experiments/t5-bm25-marginmse-5e-4/extended_rq_token_checkpoint"
    )

    docid_to_tokenids = {}
    for docid, smtids in tqdm(docid_to_smtid.items(), total=len(docid_to_smtid)):
        tokenids = []
        for i, j in enumerate(smtids):
            token = f"<docid_{i}_{j}>"
            tokenids.append(tokenizer.convert_tokens_to_ids(token))
        if docid == "0":
            print(tokenids)
        docid_to_tokenids[docid] = tokenids

    print(
        "writing docid_to_tokenids to {}".format(
            os.path.join(new_out_dir, "docid_to_tokenids.json")
        )
    )
    print("-" * 50)
    with open(os.path.join(new_out_dir, "docid_to_tokenids.json"), "w") as fout:
        ujson.dump(docid_to_tokenids, fout)


if __name__ == "__main__":
    main()
