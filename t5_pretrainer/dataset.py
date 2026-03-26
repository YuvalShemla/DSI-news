import os
import random
from copy import deepcopy

import msgspec
import torch
import ujson
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer

local_rank = int(os.environ.get("LOCAL_RANK", 0))


class DataLoaderWrapper(DataLoader):
    def __init__(self, tokenizer_type, max_length, **kwargs):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        """
        try:
            print("use auto tokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
        except:
            print("use t5 tokenizer")
            self.tokenizer = T5Tokenizer.from_pretrained(tokenizer_type, cache_dir='cache')
        """
        super().__init__(collate_fn=self.collate_fn, **kwargs, pin_memory=True)

    def collate_fn(self, batch):
        raise NotImplementedError("must implement this method")


class CrossEncRerankDataLoader(DataLoaderWrapper):
    def collate_fn(self, batch):
        pair_ids, queries, docs = [], [], []
        for elem in batch:
            pair_ids.append(elem["pair_id"])
            queries.append(elem["query"])
            docs.append(elem["doc"])

        qd_kwargs = self.tokenizer(
            queries,
            docs,
            padding=True,
            truncation="longest_first",
            return_attention_mask=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        return {"pair_ids": pair_ids, "qd_kwargs": qd_kwargs}


class RerankDataset(Dataset):
    def __init__(self, run_json_path, document_dir, query_dir, json_type="jsonl"):
        self.document_dataset = CollectionDatasetPreLoad(
            document_dir, id_style="content_id"
        )
        self.query_dataset = CollectionDatasetPreLoad(query_dir, id_style="content_id")
        if json_type == "jsonl":
            self.all_pair_ids = []
            with open(run_json_path) as fin:
                for line in fin:
                    example = ujson.loads(line)
                    qid, docids = example["qid"], example["docids"]
                    for docid in docids:
                        self.all_pair_ids.append((qid, docid))
        else:
            with open(run_json_path) as fin:
                qid_to_rankdata = ujson.load(fin)

            self.all_pair_ids = []
            for qid, rankdata in qid_to_rankdata.items():
                for pid, _ in rankdata.items():
                    self.all_pair_ids.append((str(qid), str(pid)))

    def __len__(self):
        return len(self.all_pair_ids)

    def __getitem__(self, idx):
        pair_id = self.all_pair_ids[idx]
        qid, pid = pair_id

        query = self.query_dataset[qid][1]
        doc = self.document_dataset[pid][1]

        return {
            "pair_id": pair_id,
            "query": query,
            "doc": doc,
        }


class CollectionDatasetPreLoad(Dataset):
    """
    dataset to iterate over a document/query collection, format per line: format per line: doc_id \t doc
    we preload everything in memory at init
    """

    def __init__(self, data_dir, id_style):
        self.data_dir = data_dir
        assert id_style in ("row_id", "content_id"), "provide valid id_style"
        # id_style indicates how we access the doc/q (row id or doc/q id)
        self.id_style = id_style
        self.data_dict = {}
        self.line_dict = {}
        
        print("Preloading dataset")
        with open(os.path.join(self.data_dir, "raw.tsv")) as reader:
            for i, line in enumerate(tqdm(reader)):
                if len(line) > 1:
                    id_, *data = line.split("\t")  # first column is id
                    data = " ".join(" ".join(data).splitlines())
                    if self.id_style == "row_id":
                        self.data_dict[i] = data
                        self.line_dict[i] = id_.strip()
                    else:
                        self.data_dict[id_] = data.strip()
        self.nb_ex = len(self.data_dict)

    def __len__(self):
        return self.nb_ex

    def __getitem__(self, idx):
        if self.id_style == "row_id":
            return self.line_dict[idx], self.data_dict[idx]
        else:
            return str(idx), self.data_dict[str(idx)]


class MarginMSEDataset(Dataset):
    def __init__(self, example_path, document_dir, query_dir):
        self.document_dataset = CollectionDatasetPreLoad(
            document_dir, id_style="content_id"
        )
        self.query_dataset = CollectionDatasetPreLoad(query_dir, id_style="content_id")

        self.examples = []
        with open(example_path) as fin:
            for line in fin:
                example = ujson.loads(line)
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        qid, docids, scores = (
            self.examples[idx]["qid"],
            self.examples[idx]["docids"],
            self.examples[idx]["scores"],
        )

        pos_docid = docids[0]
        pos_score = scores[0]

        neg_idx = random.sample(range(1, len(docids)), k=1)[0]
        neg_docid = docids[neg_idx]
        neg_score = scores[neg_idx]

        query = self.query_dataset[qid][1]
        pos_doc = self.document_dataset[pos_docid][1]
        neg_doc = self.document_dataset[neg_docid][1]

        return query, pos_doc, neg_doc, pos_score, neg_score


class T5DenseCollectionDataLoader(DataLoaderWrapper):
    def collate_fn(self, batch):
        """
        batch is a list of tuples, each tuple has 2 (text) items (id_, doc)
        """
        id_, d = zip(*batch)
        processed_passage = self.tokenizer(
            list(d),
            add_special_tokens=True,
            padding="longest",  # pad to max sequence length in batch
            truncation="longest_first",  # truncates to self.max_length
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # add special token for decoder_input_ids
        start_token_id = self.tokenizer.pad_token_id
        batch_size = processed_passage["input_ids"].shape[0]
        processed_passage["decoder_input_ids"] = torch.full(
            (batch_size, 1), start_token_id, dtype=torch.long
        )

        return {
            **{k: v for k, v in processed_passage.items()},
            "id": torch.tensor([int(i) for i in id_], dtype=torch.long),
        }


class RiporForSeq2seqDataset(Dataset):
    def __init__(self, example_path, docid_to_smtid_path):        
        decoder = msgspec.json.Decoder()
        
        if local_rank <= 0:
            print("Docid to smtid path: {}".format(docid_to_smtid_path))
        with open(docid_to_smtid_path, 'rb') as f:
            docid_to_smtid = decoder.decode(f.read())

        self.examples = []
        
        # Get one examples only
        with open(example_path, 'rb') as f:
            fin = decoder.decode_lines(f.read())

        for example in tqdm(fin):
            docid, query = example["doc_id"], example["query"]
            smtid = docid_to_smtid[str(docid)]
            self.examples.append((query, smtid))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        query, smtid = self.examples[idx]

        assert len(smtid) in {4, 8, 16, 24}
        # assert smtid[0] != -1 and smtid[0] >= 32000, smtid

        return query, smtid


class RiporForSeq2seqCollator:
    def __init__(self, tokenizer_type, max_length):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)

    def __call__(self, batch):
        query, smtid = [list(x) for x in zip(*batch)]

        tokenized_query = self.tokenizer(
            query,
            add_special_tokens=True,
            padding="longest",  # pad to max sequence length in batch
            truncation="longest_first",  # truncates to self.max_length
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )
        labels = torch.LongTensor(smtid)

        batch_size = labels.size(0)
        prefix_tensor = torch.full(
            (batch_size, 1), self.tokenizer.pad_token_id, dtype=torch.long
        )
        tokenized_query["decoder_input_ids"] = torch.hstack(
            (prefix_tensor, labels[:, :-1])
        )

        return {
            "tokenized_query": tokenized_query,
            "labels": labels,
        }


class CollectionDataLoaderForRiporGeneration(DataLoaderWrapper):
    def collate_fn(self, batch):
        """
        batch is a list of tuples, each tuple has 2 (text) items (id_, doc)
        """
        id_, d = zip(*batch)
        processed_passage = self.tokenizer(
            list(d),
            add_special_tokens=True,
            padding="longest",  # pad to max sequence length in batch
            truncation="longest_first",  # truncates to self.max_length
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # add special token for decoder_input_ids
        start_token_id = self.tokenizer.pad_token_id
        batch_size = processed_passage["input_ids"].shape[0]
        processed_passage["decoder_input_ids"] = torch.full(
            (batch_size, 1), start_token_id, dtype=torch.long
        )

        return {
            **{k: v for k, v in processed_passage.items()},
            "id": torch.tensor([int(i) for i in id_], dtype=torch.long),
        }


class T5DenseMarginMSECollator:
    def __init__(self, tokenizer_type, max_length):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)

    def __call__(self, batch):
        query, pos_doc, neg_doc, pos_score, neg_score = zip(*batch)

        tokenized_query = self.tokenizer(
            list(query),
            add_special_tokens=True,
            padding="longest",  # pad to max sequence length in batch
            truncation="longest_first",  # truncates to self.max_length
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        pos_tokenized_doc = self.tokenizer(
            list(pos_doc),
            add_special_tokens=True,
            padding="longest",  # pad to max sequence length in batch
            truncation="longest_first",  # truncates to self.max_length
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        neg_tokenized_doc = self.tokenizer(
            list(neg_doc),
            add_special_tokens=True,
            padding="longest",  # pad to max sequence length in batch
            truncation="longest_first",  # truncates to self.max_length
            max_length=self.max_length,
            return_attention_mask=True,
            return_tensors="pt",
        )

        # add special token for decoder_input_ids
        start_token_id = self.tokenizer.pad_token_id
        batch_size = tokenized_query["input_ids"].shape[0]
        # print("start_token_idx for T5: {}".format(start_token_id))

        decoder_input_ids = torch.full(
            (batch_size, 1), start_token_id, dtype=torch.long
        )
        tokenized_query["decoder_input_ids"] = decoder_input_ids
        pos_tokenized_doc["decoder_input_ids"] = deepcopy(decoder_input_ids)
        neg_tokenized_doc["decoder_input_ids"] = deepcopy(decoder_input_ids)

        # teacher score
        teacher_pos_scores = torch.FloatTensor(pos_score)
        teacher_neg_scores = torch.FloatTensor(neg_score)

        return {
            "tokenized_query": tokenized_query,
            "pos_tokenized_doc": pos_tokenized_doc,
            "neg_tokenized_doc": neg_tokenized_doc,
            "teacher_pos_scores": teacher_pos_scores,
            "teacher_neg_scores": teacher_neg_scores,
        }


class RiporForMarginMSEDataset(Dataset):
    def __init__(self, dataset_path, document_dir, query_dir,
                 docid_to_smtid_path, smtid_as_docid=False):
        self.document_dataset = CollectionDatasetPreLoad(document_dir, id_style="content_id")
        self.query_dataset = CollectionDatasetPreLoad(query_dir, id_style="content_id")
        
        self.examples = []
        with open(dataset_path) as fin:
            for line in fin:
                self.examples.append(ujson.loads(line))

        self.smtid_as_docid = smtid_as_docid

        if self.smtid_as_docid:
            self.docid_to_smtid = None
            assert docid_to_smtid_path == None
        else:
            if docid_to_smtid_path is not None:
                with open(docid_to_smtid_path) as fin: 
                    self.docid_to_smtid = ujson.load(fin)
                tmp_docids = list(self.docid_to_smtid.keys())
                assert self.docid_to_smtid[tmp_docids[0]][0] != -1, self.docid_to_smtid[tmp_docids[0]]
            else:
                self.docid_to_smtid = None 

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]
        query = example["qid"]
        if self.smtid_as_docid:
            positive = example["smtids"][0]
        else:
            positive = example["docids"][0]
        s_pos = example["scores"][0]

        if self.smtid_as_docid:
            neg_idx = random.sample(range(1, len(example["smtids"])), k=1)[0]
            negative = example["smtids"][neg_idx]
        else:
            neg_idx = random.sample(range(1, len(example["docids"])), k=1)[0]
            negative = example["docids"][neg_idx]
        s_neg = example["scores"][neg_idx]

        q = self.query_dataset[str(query)][1]

        if self.smtid_as_docid:
            pos_doc_encoding = [int(x) for x in positive.split("_")]
            neg_doc_encoding = [int(x) for x in negative.split("_")]
        else:
            pos_doc_encoding = self.docid_to_smtid[str(positive)]
            neg_doc_encoding = self.docid_to_smtid[str(negative)]
        
        assert len(pos_doc_encoding) == len(neg_doc_encoding)
        assert len(pos_doc_encoding) in {4, 8, 16, 32}, len(pos_doc_encoding)
        q_pos = q.strip()
        q_neg = q.strip()

        return q_pos, q_neg, pos_doc_encoding, neg_doc_encoding, s_pos, s_neg


class RiporForMarginMSECollator:
    """ Can also be used for T5AQDataset """
    def __init__(self, tokenizer_type, max_length):
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_type)
    
    def __call__(self, batch):
        q_pos, q_neg, pos_doc_encoding, neg_doc_encoding, s_pos, s_neg = [list(x) for x in zip(*batch)]
        assert pos_doc_encoding[0][0] >= len(self.tokenizer), (pos_doc_encoding, len(self.tokenizer))

        q_pos = self.tokenizer(q_pos,
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")
        q_neg = self.tokenizer(q_neg,
                           add_special_tokens=True,
                           padding="longest",  # pad to max sequence length in batch
                           truncation="longest_first",  # truncates to self.max_length
                           max_length=self.max_length,
                           return_attention_mask=True,
                           return_tensors="pt")

        pos_doc_encoding = torch.LongTensor(pos_doc_encoding)
        neg_doc_encoding = torch.LongTensor(neg_doc_encoding)
        batch_size = q_pos["input_ids"].size(0)
        prefix_tensor = torch.full((batch_size, 1), self.tokenizer.pad_token_id, dtype=torch.long)
        q_pos["decoder_input_ids"] = torch.hstack((prefix_tensor, pos_doc_encoding[:, :-1]))
        q_neg["decoder_input_ids"] = torch.hstack((prefix_tensor, neg_doc_encoding[:, :-1]))

        return {
            "pos_tokenized_query": q_pos,
            "neg_tokenized_query": q_neg,
            "pos_doc_encoding": pos_doc_encoding,
            "neg_doc_encoding": neg_doc_encoding,
            "teacher_pos_scores": torch.FloatTensor(s_pos),
            "teacher_neg_scores": torch.FloatTensor(s_neg)
        }