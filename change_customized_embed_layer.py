import argparse
import os
from pathlib import Path
from pdb import set_trace as st

import faiss
import torch
import torch.nn as nn
import ujson
from t5_pretrainer.ripor import Ripor
from t5_pretrainer.utils.prefixer import generate_special_token_list
from tqdm import tqdm
from transformers import AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_dir", 
                        default=None,
                        type=str)
    parser.add_argument("--K",
                        default=256,
                        type=int)
    parser.add_argument("--codebook_num", default=24, type=int)
    parser.add_argument("--codebook_bits", default=8, type=int)
    parser.add_argument("--out_dir", default=None, type=str)
    parser.add_argument("--mmap_dir", default=None, type=str)
    parser.add_argument("--extended_model_out_dir", default=None, type=str)

    return parser.parse_args()

def main():
    args = get_args()

    m = args.codebook_num
    nbits = args.codebook_bits
    model_dir = args.model_dir
    mmap_dir = args.mmap_dir
    d_model = 768
    extended_model_out_dir = args.extended_model_out_dir

    pretrained_path = os.path.join(model_dir, "checkpoint")
    index_path = os.path.join(mmap_dir, "model.index")

    index = faiss.read_index(index_path)

    if "pq" in Path(mmap_dir).stem:
        print("Product quantization")
        pq = index.pq
    elif "rq" in Path(mmap_dir).stem:
        print("Residual quantization")
        pq = index.rq # For the sake of reducing boilerplate code
    else:
        raise ValueError("Unknown index type")

    pq.M = m
    pq.d = d_model
    print("num of subquantizers: ", pq.M)
    print("bits per subquantizer: ", nbits)
    print("dimension of the vectors: ", pq.d)

    if "pq" in Path(mmap_dir).stem:
        centroids = faiss.vector_to_array(pq.centroids).reshape(pq.M, 2**nbits, d_model // pq.M)
    elif "rq" in Path(mmap_dir).stem:
        centroids = faiss.vector_to_array(pq.codebooks).reshape(pq.M, 2**nbits, 768)
    else:
        raise ValueError("Unknown index type")
    centroids = torch.FloatTensor(centroids)

    model = Ripor.from_pretrained(pretrained_path)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_path)

    # change embed layers and corresponding config params
    # For reference if the number does not make sense: https://github.com/huggingface/transformers/issues/4875 (Inconsistent number of vocab from pretrained T5Tokenizer and T5ForConditionalGeneration)
    print("original embedding size = {}".format(len(tokenizer)), model.base_model.get_input_embeddings().weight.size(0))

    new_tokens = generate_special_token_list(num_code=pq.M, codebook_size=2**nbits)
    tokenizer.add_tokens(new_tokens)

    assert len(new_tokens) == pq.M * 2**nbits, (len(new_tokens), pq.M * 2**nbits)

    # resize model embeds and assign new embeddings
    model.base_model.resize_token_embeddings(len(tokenizer))
    embedding_weight = model.base_model.get_input_embeddings().weight

    if "rq" in Path(mmap_dir).stem:
        print("Residual quantization token initialization")

        for i in range(pq.M):
            for j in range(2**nbits):
                # Token to find
                token = f"<docid_{i}_{j}>"

                # Find the index of the token in the tokenizer
                token_index = tokenizer.convert_tokens_to_ids(token)
                assert token_index >= 32100, token_index

                # Get the corresponding centroid embedding
                vec = centroids[i, j]
                # print(vec.shape)

                # Assign the embedding vector to the corresponding index in the embedding weight
                embedding_weight.data[token_index] = vec
    elif "pq" in Path(mmap_dir).stem:
        print(
            "Skip because the centroid size (32) is mismatched with the embedding size (768)"
        )
        print("Product quantization will have tokens randomly initialized")

    print("new embedding size = {}".format(len(tokenizer)), model.base_model.get_input_embeddings().weight.size(0))

    # create docid_to_tokenids
    with open(os.path.join(args.out_dir, "docid_to_smtid.json")) as fin:
        docid_to_smtids = ujson.load(fin)

    docid_to_tokenids = {}
    for docid, smtids in tqdm(docid_to_smtids.items(), total=len(docid_to_smtids)):
        tokenids = []
        for i, j in enumerate(smtids):
            token = f"<docid_{i}_{j}>"
            tokenids.append(tokenizer.convert_tokens_to_ids(token))
        if docid == "0":
            print(tokenids)
        docid_to_tokenids[docid] = tokenids
    with open(
        os.path.join(args.out_dir, "docid_to_tokenids.json"), "w"
    ) as fout:
        ujson.dump(docid_to_tokenids, fout)
    # save model
    model.save_pretrained(extended_model_out_dir)
    tokenizer.save_pretrained(extended_model_out_dir)


if __name__ == "__main__":
    main()
