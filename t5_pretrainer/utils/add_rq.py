from pathlib import Path
from pdb import set_trace as st

import matplotlib.pyplot as plt
import msgspec
import numpy as np
import torch
import torch.nn as nn
from safetensors import safe_open

encoder = msgspec.json.Encoder()
decoder = msgspec.json.Decoder()

docid_to_tokenids_path = "./nq320k/d0/experiments/rq_smtid/docid_to_tokenids.json"

with open(docid_to_tokenids_path, 'rb') as fin:
    docid_to_tokenids = decoder.decode(fin.read())

start_token_id = 32100
k = 2048
m = 8

# checkpoint = {}
# baseline_path = "./nq320k/d0/experiments/t5-self-neg-marginmse-5e-4/extended_rq_token_checkpoint/model.safetensors"

# with safe_open(baseline_path, framework="pt", device=0) as f:  # type: ignore
#     # This is most likely t5-self-neg checkpoint
#     for key in f.keys():
#         checkpoint[key] = f.get_tensor(key)

# rq_centroids = checkpoint["shared.weight"]
