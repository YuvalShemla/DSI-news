from __future__ import annotations

import copy
import math
import os
import warnings
from collections import namedtuple
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from pdb import set_trace as st
from typing import List, Optional, Tuple, Union

import einx
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from adapters import AutoAdapterModel
from beartype import beartype
from colt5_attention import topk as maybe_differentiable_topk
from einops import pack, rearrange, reduce, unpack
from safetensors import safe_open
from torch import einsum
from torch.nn import CrossEntropyLoss, Module, ModuleList
from transformers.activations import ACT2FN
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
)
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import (
    _CONFIG_FOR_DOC,
    DEPARALLELIZE_DOCSTRING,
    PARALLELIZE_DOCSTRING,
    T5_INPUTS_DOCSTRING,
    T5Block,
    T5LayerCrossAttention,
    T5LayerNorm,
    T5LayerSelfAttention,
    T5PreTrainedModel,
    T5Stack,
)
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

from t5_pretrainer.ewc import EWC
from t5_pretrainer.mixlora_config import MixLoraConfig
from t5_pretrainer.stmoe_distributed import (
    AllGather,
    gather_sizes,
    has_only_one_value,
    pad_dim_to,
    split_by_rank,
)
from t5_pretrainer.utils.utils import get_params_info

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "T5Config"

MIN_EXPERT_CAPACITY = 8


def _get_submodules(model, key):
    parent = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target = model.get_submodule(key)
    return parent, target, target_name


def inject_adapter_in_model(
    model,
    config: MixLoraConfig,
    # weights: Dict[str, torch.Tensor],
):
    encoder = config.encoder_attention
    decoder = config.decoder_attention

    if not (encoder or decoder):
        print("No MixLora injection to attentions")
        return

    key_list = [key for key, _ in model.named_modules()]
    target_modules = [
        (target, inject) for target, inject in config.attention_target_modules.items()
    ]

    for key in key_list:
        parent, target, target_name = _get_submodules(model, key)

        # Encoder attentions
        if encoder and isinstance(parent, T5LayerSelfAttention):
            for proj_name, inject in target_modules:
                if not inject or not hasattr(target, proj_name):
                    continue
                base_layer = getattr(target, proj_name)
                setattr(
                    target,
                    proj_name,
                    LoraLinear(
                        base_layer,
                        config,
                        weight=None,
                    ),
                )
                # print(f"added lora to encoder {type(parent)}'s {target_name}")

        # Decoder attentions
        if decoder and isinstance(parent, T5LayerCrossAttention):
            for proj_name, inject in target_modules:
                if not inject or not hasattr(target, proj_name):
                    continue
                base_layer = getattr(target, proj_name)
                print(type(parent), target_name)
                setattr(
                    target,
                    proj_name,
                    Lora(
                        base_layer,
                        config,
                        weight=None,
                    ),
                )
                # print(f"added lora to decoder {type(parent)}'s {target_name}")

    if encoder:
        print("Enabled MixLora in encoder")
    if decoder:
        print("Enabled MixLora in decoder")

        # if isinstance(parent, MixloraT5LayerFF):
        #     """
        #     class T5LayerFF(Module):
        #         def __init__(self, config: T5Config):
        #             super().__init__()
        #             if config.is_gated_act:
        #                 self.DenseReluDense = T5DenseGatedActDense(config)
        #             else:
        #                 self.DenseReluDense = T5DenseActDense(config)

        #             self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        #             self.dropout = nn.Dropout(config.dropout_rate)

        #         def forward(self, hidden_states):
        #             forwarded_states = self.layer_norm(hidden_states)
        #             forwarded_states = self.DenseReluDense(forwarded_states)
        #             hidden_states = hidden_states + self.dropout(forwarded_states)
        #             return hidden_states
        #     """

        #     if target_name != "DenseReluDense":
        #         continue

        #     # Turn off dictionary to let mixlora moes layer to be sent to gpu
        #     # if not hasattr(target, "mixlora_moes"):
        #     #     target.mixlora_moes = {}

        #     print(key)
        #     name = (
        #         "mixlora."
        #         + key.split(".")[0]
        #         + ".layer"
        #         + key.split(".")[2]
        #         + f".{config.num_experts_}-experts"
        #         + f".top-{config.top_k_}"
        #     )
        #     moe_layer = MixLoraSparseMoe(target, config, parent.layer_norm, name=name)
        #     moe_layer.gate_ = torch.nn.Linear(768, config.num_experts_).to(
        #         config.dtype_
        #     )

        #     # target.mixlora_moes[config.adapter_name_] = moe_layer
        #     target.mixlora_moes = moe_layer
        #     target.forward = moe_layer.forward

        #     for proj_name, inject in target_modules:
        #         if not inject or not hasattr(target, proj_name):
        #             continue

        #         base_layer = getattr(target, proj_name)
        #         print(base_layer, proj_name)
        #         for expert_idx in range(config.num_experts_):
        #             moe_layer.experts_[f"experts.{expert_idx}.{proj_name}"] = (
        #                 LoraLinear(
        #                     base_layer,
        #                     config,
        #                     (None, None),
        #                 )
        #             )
        #         print(f"added router to {type(parent)}'s {target_name}")


class Lora(Module):
    def __init__(
        self,
        base_layer,
        config: MixLoraConfig,
        weight = None,
        device: Optional[str] = None,
    ):
        super().__init__()
            
        in_features, out_features = base_layer.in_features, base_layer.out_features
        # self.base_layer = base_layer
        self.device = torch.device(device) if device else base_layer.weight.device
        self.dtype = base_layer.weight.dtype

        self.r = config.r
        self.alpha = config.lora_alpha
        self.scaling = self.alpha / self.r

        self.in_features = in_features
        self.out_features = out_features

        self.dropout = nn.Dropout(p=config.lora_dropout)

        self.lora_A = nn.Linear(self.in_features, self.r, bias=False, dtype=self.dtype, device=self.device)
        self.lora_B = nn.Linear(self.r, self.out_features, bias=False, dtype=self.dtype, device=self.device)

        # Always replace the weight to random initialization, since nn.Linear sometimes has NaN in their weight
        self.lora_A.weight.data = torch.randn_like(self.lora_A.weight, dtype=self.dtype, device=self.device)
        self.lora_B.weight.data = torch.randn_like(self.lora_B.weight, dtype=self.dtype, device=self.device)

        self.reset_lora_parameters()
        if self.lora_A.weight.isnan().any() or self.lora_B.weight.isnan().any():
            print("nan in lora_A or lora_B")

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lora_B(self.lora_A(self.dropout(hidden_states.to(self.dtype)))) * self.scaling

######### MixLoRA #########
MixtureOfExpertsReturn = namedtuple('MixtureOfExpertsReturn', [
    'outputs',
    'total_aux_loss',
    'balance_loss',
    'router_z_loss',
    'router_logits'
])

def exists(val):
    return val is not None

def default(val, default):
    if exists(val):
        return val

    return default() if callable(default) else default

def divisible_by(num, den):
    return (num % den) == 0

def chunk_num(num, chunks):
    num_per_chunk, remainder = divmod(num, chunks)

    out = []
    for i in range(chunks):
        n = num_per_chunk
        out.append(n + int(i < remainder))

    return out

def pack_one(t, pattern):
    return pack([t], pattern)

def unpack_one(t, ps, pattern):
    return unpack(t, ps, pattern)[0]

def cast_tuple(el, len = 1):
    return el if isinstance(el, tuple) else ((el,) * len)

def Sequential(*modules):
    return nn.Sequential(*filter(exists, modules))

def cumsum_exclusive(t, dim = -3):
    assert dim < 0
    num_pad_dims = -dim - 1
    pre_padding = (0, 0) * num_pad_dims
    return F.pad(t, (*pre_padding, 1, -1)).cumsum(dim = dim)

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

# pytorch one hot throws an error if there are out of bound indices.
# tensorflow, in contrast, does not throw an error
def safe_one_hot(indexes, max_length):
    max_index = indexes.max() + 1
    one_hot_classes = max(max_index + 1, max_length)
    return F.one_hot(indexes, one_hot_classes)[..., :max_length]


class TopNGating(Module):

    @beartype
    def __init__(
        self,
        dim,
        num_gates,
        eps=1e-9,
        top_n=2,
        threshold_train: float | Tuple[float, ...] = 0.2,
        threshold_eval: float | Tuple[float, ...] = 0.2,
        capacity_factor_train=1.25,
        capacity_factor_eval=2.0,
        straight_through_dispatch_tensor=True,
        differentiable_topk=False,
        differentiable_topk_fused=True,
        mixlora_config: MixLoraConfig = None,
        name=None
    ):
        super().__init__()
        self.name = name
        self.dim = dim
        self.eps = eps
        self.num_gates = num_gates
        self.to_gates = nn.Linear(dim, num_gates, bias=False)
        self.to_gates.weight.data = torch.randn_like(self.to_gates.weight)
        torch.nn.init.normal_(self.to_gates.weight, std=768**-0.5 * 1.0)
        self.renorm_keep_std(self.to_gates.weight, dim=1)
        assert self.to_gates.weight.isnan().sum() == 0, "NaNs in to_gates"
        self.differentiable_topk = differentiable_topk

        self.topk = partial(
            maybe_differentiable_topk,
            non_differentiable=not differentiable_topk,
            fused=differentiable_topk_fused,  # use triton fused coordinate descent if possible by default
        )

        assert top_n >= 2, "must be 2 or more experts"
        self.top_n = top_n
        top_n_minus_1 = top_n - 1

        threshold_train = cast_tuple(threshold_train, top_n_minus_1)
        threshold_eval = cast_tuple(threshold_eval, top_n_minus_1)

        assert len(threshold_train) == len(threshold_eval) == top_n_minus_1

        self.register_buffer("threshold_train", torch.tensor([eps, *threshold_train]))
        self.register_buffer("threshold_eval", torch.tensor([eps, *threshold_eval]))

        self.capacity_factor_train = capacity_factor_train
        self.capacity_factor_eval = capacity_factor_eval

        self.straight_through_dispatch_tensor = straight_through_dispatch_tensor
        self.register_buffer("zero", torch.zeros((1,)), persistent=False)
        self.mixlora_config = mixlora_config

    def renorm_keep_std(self, weight: torch.Tensor, dim: int = 0):
        with torch.no_grad():
            std = weight.std()
            weight.div_(weight.norm(dim=dim, keepdim=True))
            weight.mul_(std / weight.std())

    def forward(self, x, noise_gates=False, noise_mult=1.0):
        """
        einstein notation:

        b - batch
        n - sequence
        e - experts
        k - top-n experts
        c - capacity
        """
        *_, b, group_size, dim, dtype, top_n, num_gates, eps = (
            *x.shape,
            x.dtype,
            self.top_n,
            self.num_gates,
            self.eps,
        )

        # threshold, capacity depending on training or eval

        suffix = "train" if self.training else "eval"

        threshold = getattr(self, f"threshold_{suffix}")
        capacity_factor = getattr(self, f"capacity_factor_{suffix}")

        # Each sequence sends (at most?) expert_capacity positions to each expert.
        # Static expert_capacity dimension is needed for expert batch sizes

        expert_capacity = min(
            group_size, int((group_size * capacity_factor) / num_gates)
        )
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)

        # gate logits and gates

        gate_logits = self.to_gates(x)

        maybe_noised_gate_logits = gate_logits
        maybe_noised_gate_logits_for_logging = gate_logits.clone()

        if noise_gates:
            noise = gumbel_noise(maybe_noised_gate_logits)
            maybe_noised_gate_logits = maybe_noised_gate_logits + noise * noise_mult

        raw_gates = maybe_noised_gate_logits.softmax(dim=-1)

        # find top N experts per position

        topk_return = self.topk(raw_gates, k=top_n)

        gate_indices = topk_return.indices

        if self.differentiable_topk:
            # allow for differentiable topk using coordinate descent
            # used successfully for routing from CoLT5 paper https://github.com/lucidrains/CoLT5-attention

            gates = topk_return.coor_descent_values
        else:
            gates = topk_return.values

        gates = rearrange(gates, "... k -> k ...")
        gate_indices = rearrange(gate_indices, "... k -> k ...")

        # masks

        one_hot_gate_indices = F.one_hot(gate_indices, num_gates)
        mask = one_hot_gate_indices.float()

        mask_1 = mask[0]  # needed for balancing loss

        # normalize top-n gate scores

        denom = reduce(gates, "k ... -> 1 ...", "sum").clamp(min=eps)
        gates = gates / denom
        # best performing policy was to route to the second expert, with probability of min(1., score / threshold), where score = gate2 / (gate1 + gate2)
        # optimal threshold was ~ 0.2
        # generalized to more than 2 experts

        probs = torch.zeros_like(gates).uniform_(0.0, 1.0)

        should_route = probs < einx.divide(
            "k b n, k -> k b n", gates, threshold.clamp(min=eps)
        )

        # tokens should always be routed to first expert
        # threshold for first expert already set to very small number, but just in case

        should_route[0, ...] = True

        mask *= rearrange(should_route.float(), "... -> ... 1")

        mask_cumsum = cumsum_exclusive(mask, dim=-2)  # along sequence dimension

        # compute assignment to experts - (batch, seq, experts)

        # This is the position within the expert's mini-batch for this sequence

        positions = []
        prev_expert_count = 0.0

        for n in range(self.top_n):
            position_in_expert = (mask_cumsum[n] + prev_expert_count) * mask[n]

            # Remove the elements that don't fit. (batch, sequence, experts)
            mask[n] *= (position_in_expert < expert_capacity_f).float()

            # How many examples in this sequence go to this expert - needed for the next iteration as offset
            prev_expert_count = (
                reduce(mask[n], "... n e -> ... 1 e", "sum") + prev_expert_count
            )

            # (batch, sequence)
            position_in_expert = reduce(position_in_expert, "... n e -> ... n", "sum")
            positions.append(position_in_expert)

        positions = torch.stack(positions)

        # (k, batch, sequence) - mostly ones, but zeros where something didn't fit
        mask_flat = reduce(mask, "... n e -> ... n", "sum")

        # (k, batch, sequence) - weighted assignment
        # following https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py#L1903
        gates = gates * mask_flat

        # (batch, sequence, experts, expert_capacity)

        combine_tensor = einx.multiply(
            "k b n, k b n, k b n e, k b n c -> k b n e c",
            gates,
            mask_flat,
            one_hot_gate_indices,
            safe_one_hot(positions.long(), expert_capacity),
        )

        combine_tensor = reduce(combine_tensor, "k b n e c -> b n e c", "sum")

        # dispatch tensor

        dispatch_tensor = combine_tensor.bool().type(dtype)

        if self.straight_through_dispatch_tensor:
            dispatch_tensor = dispatch_tensor + combine_tensor - combine_tensor.detach()

        # balance losses - (batch, experts)
        # We want to equalize the fraction of the batch assigned to each expert

        if self.training:
            density_1 = reduce(mask_1, "... n e -> ... e", "mean")
            density_1_proxy = reduce(
                raw_gates, "... n e -> ... e", "mean"
            )  # Something continuous that is correlated with what we want to equalize.

            balance_loss = (density_1_proxy * density_1).mean() * float(num_gates**2)
        else:
            balance_loss = self.zero

        # calculate the router z-loss proposed in paper

        if self.training:
            router_z_loss = torch.logsumexp(gate_logits, dim=-1)
            router_z_loss = torch.square(router_z_loss)
            router_z_loss = router_z_loss.mean()
        else:
            router_z_loss = self.zero

        return (
            dispatch_tensor,
            combine_tensor,
            balance_loss,
            router_z_loss,
            maybe_noised_gate_logits_for_logging,
        )


class TopNGating_NaiveExpand(TopNGating):

    @beartype
    def __init__(
        self,
        dim,
        num_gates,
        eps=1e-9,
        top_n=2,
        threshold_train: float | Tuple[float, ...] = 0.2,
        threshold_eval: float | Tuple[float, ...] = 0.2,
        capacity_factor_train=1.25,
        capacity_factor_eval=2.0,
        straight_through_dispatch_tensor=True,
        differentiable_topk=False,
        differentiable_topk_fused=True,
        mixlora_config: MixLoraConfig = None,
        name=None,
    ):
        super().__init__(dim, num_gates, eps, top_n, threshold_train, threshold_eval, capacity_factor_train, capacity_factor_eval, straight_through_dispatch_tensor, differentiable_topk, differentiable_topk_fused, mixlora_config, name)
        self.cosine_classifier = self.mixlora_config.cosine_classifier

    def grow_expert(self, embedding=None):
        """
        Only need to grow the to_gates linear layer, which is the router.
        """
        new_to_gates = nn.Linear(self.dim, self.num_gates + 1, bias=False)
        new_to_gates.weight.data = torch.randn_like(new_to_gates.weight)
        # new_to_gates.weight.data.normal_(mean=0.0, std=1.0)
        torch.nn.init.normal_(new_to_gates.weight, mean=0.0, std=768**-0.5 * 1.0)
        self.renorm_keep_std(new_to_gates.weight, dim=1)
        assert new_to_gates.weight.isnan().sum() == 0, "NaNs in to_gates"

        with torch.no_grad():
            new_to_gates.weight[: self.num_gates].copy_(self.to_gates.weight)
            if embedding is not None:
                print(
                    "Initializing new router weight with the given embedding"
                )
                new_to_gates.weight[self.num_gates].copy_(embedding)
            else:
                print("Randomly initializing new router weight")
            self.to_gates = new_to_gates

        self.num_gates += 1

    def forward(self, x, noise_gates=False, noise_mult=1.0):
        """
        einstein notation:

        b - batch
        n - sequence
        e - experts
        k - top-n experts
        c - capacity
        """
        *_, b, group_size, dim, dtype, top_n, num_gates, eps = (
            *x.shape,
            x.dtype,
            self.top_n,
            self.num_gates,
            self.eps,
        )

        # threshold, capacity depending on training or eval

        suffix = "train" if self.training else "eval"

        threshold = getattr(self, f"threshold_{suffix}")
        capacity_factor = getattr(self, f"capacity_factor_{suffix}")

        # Each sequence sends (at most?) expert_capacity positions to each expert.
        # Static expert_capacity dimension is needed for expert batch sizes

        expert_capacity = min(
            group_size, int((group_size * capacity_factor) / num_gates)
        )
        expert_capacity = max(expert_capacity, MIN_EXPERT_CAPACITY)
        expert_capacity_f = float(expert_capacity)

        # gate logits and gates

        # Modify the router behavior
        if self.cosine_classifier:
            self.to_gates.weight.data = F.normalize(
                self.to_gates.weight.data, p=2, dim=-1
            )
            gate_logits = self.to_gates(F.normalize(x, p=2, dim=-1))
        else:
            gate_logits = self.to_gates(x)

        maybe_noised_gate_logits = gate_logits
        maybe_noised_gate_logits_for_logging = gate_logits.clone()

        if noise_gates:
            noise = gumbel_noise(maybe_noised_gate_logits)
            maybe_noised_gate_logits = maybe_noised_gate_logits + noise * noise_mult

        raw_gates = maybe_noised_gate_logits.softmax(dim=-1)

        # find top N experts per position

        topk_return = self.topk(raw_gates, k=top_n)

        gate_indices = topk_return.indices

        if self.differentiable_topk:
            # allow for differentiable topk using coordinate descent
            # used successfully for routing from CoLT5 paper https://github.com/lucidrains/CoLT5-attention

            gates = topk_return.coor_descent_values
        else:
            gates = topk_return.values

        gates = rearrange(gates, "... k -> k ...")
        gate_indices = rearrange(gate_indices, "... k -> k ...")

        # masks

        one_hot_gate_indices = F.one_hot(gate_indices, num_gates)
        mask = one_hot_gate_indices.float()

        mask_1 = mask[0]  # needed for balancing loss

        # normalize top-n gate scores

        denom = reduce(gates, "k ... -> 1 ...", "sum").clamp(min=eps)
        gates = gates / denom

        # best performing policy was to route to the second expert, with probability of min(1., score / threshold), where score = gate2 / (gate1 + gate2)
        # optimal threshold was ~ 0.2
        # generalized to more than 2 experts

        # probs = torch.zeros_like(gates).uniform_(0.0, 1.0)

        # should_route = probs < einx.divide(
        #     "k b n, k -> k b n", gates, threshold.clamp(min=eps)
        # )

        # # tokens should always be routed to first expert
        # # threshold for first expert already set to very small number, but just in case

        # should_route[0, ...] = True

        probs = torch.zeros_like(gates).uniform_(0.0, 1.0)

        should_route = probs < einx.divide(
            "k b n, k -> k b n", gates, threshold.clamp(min=eps)
        )
        # tokens should always be routed to first expert
        # threshold for first expert already set to very small number, but just in case

        should_route[0, ...] = True
        # else:
        #     should_route = torch.ones_like(gates).bool()

        mask *= rearrange(should_route.float(), "... -> ... 1")

        mask_cumsum = cumsum_exclusive(mask, dim=-2)  # along sequence dimension

        # compute assignment to experts - (batch, seq, experts)

        # This is the position within the expert's mini-batch for this sequence

        positions = []
        prev_expert_count = 0.0

        for n in range(self.top_n):
            position_in_expert = (mask_cumsum[n] + prev_expert_count) * mask[n]

            # Remove the elements that don't fit. (batch, sequence, experts)
            mask[n] *= (position_in_expert < expert_capacity_f).float()

            # How many examples in this sequence go to this expert - needed for the next iteration as offset
            prev_expert_count = (
                reduce(mask[n], "... n e -> ... 1 e", "sum") + prev_expert_count
            )

            # (batch, sequence)
            position_in_expert = reduce(position_in_expert, "... n e -> ... n", "sum")
            positions.append(position_in_expert)

        positions = torch.stack(positions)

        # (k, batch, sequence) - mostly ones, but zeros where something didn't fit
        mask_flat = reduce(mask, "... n e -> ... n", "sum")

        # (k, batch, sequence) - weighted assignment
        # following https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py#L1903
        gates = gates * mask_flat

        # (batch, sequence, experts, expert_capacity)

        combine_tensor = einx.multiply(
            "k b n, k b n, k b n e, k b n c -> k b n e c",
            gates,
            mask_flat,
            one_hot_gate_indices,
            safe_one_hot(positions.long(), expert_capacity),
        )

        combine_tensor = reduce(combine_tensor, "k b n e c -> b n e c", "sum")

        # dispatch tensor

        dispatch_tensor = combine_tensor.bool().type(dtype)

        if self.straight_through_dispatch_tensor:
            dispatch_tensor = dispatch_tensor + combine_tensor - combine_tensor.detach()

        # balance losses - (batch, experts)
        # We want to equalize the fraction of the batch assigned to each expert

        balance_loss = torch.tensor(0.0).to(gate_logits.device)
        router_z_loss = torch.tensor(0.0).to(gate_logits.device)
        
        if self.training and not self.mixlora_config.no_aux_loss:
            density_1 = reduce(mask_1, "... n e -> ... e", "mean")
            density_1_proxy = reduce(
                raw_gates, "... n e -> ... e", "mean"
            )  # Something continuous that is correlated with what we want to equalize.

            balance_loss = (density_1_proxy * density_1).mean() * float (num_gates**2)
            
            # calculate the router z-loss proposed in paper
            router_z_loss = torch.logsumexp(gate_logits, dim=-1)
            router_z_loss = torch.square(router_z_loss)
            router_z_loss = router_z_loss.mean()

        return (
            dispatch_tensor,
            combine_tensor,
            balance_loss,
            router_z_loss,
            maybe_noised_gate_logits_for_logging,
        )


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return F.normalize(x, dim=-1) * self.gamma * self.scale


# expert class
# best performing was ff geglu with multiplicative bias (just after gating)
class GEGLU(Module):
    def __init__(self, dim, mult_bias=True):
        super().__init__()
        self.mult_bias = nn.Parameter(torch.ones(dim)) if mult_bias else 1.0

    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        return F.gelu(gate) * x * self.mult_bias


class Expert(Module):
    def __init__(
        self,
        dim,
        hidden_mult = 4,
        mult_bias = True,
        prenorm = False
    ):
        super().__init__()
        dim_hidden = int(dim * hidden_mult * 2 / 3)

        self.net = Sequential(
            RMSNorm(dim) if prenorm else None,
            nn.Linear(dim, dim_hidden * 2),
            GEGLU(dim_hidden, mult_bias = mult_bias),
            nn.Linear(dim_hidden, dim)
        )

        self.apply(self.init_)

    def init_(self, module):
        if isinstance(module, nn.Linear):
            dim = module.weight.shape[0]
            std = dim ** -0.5

            module.weight.data.uniform_(-std, std)
            module.bias.data.uniform_(-std, std)

    def forward(self, x):
        print(x.shape)
        return self.net(x)


class Experts(Module):
    def __init__(
        self,
        experts,
        is_distributed=None,
        allow_var_seq_len=False,  # whether to handle variable sequence length
    ):
        super().__init__()
        self.num_experts = len(experts)
        self.experts = ModuleList(experts)

        # distributed related settings

        self.is_distributed = is_distributed
        if not exists(self.is_distributed):
            self.is_distributed = dist.is_initialized() and dist.get_world_size() > 1

        self.all_gather = AllGather()

        self.allow_var_seq_len = allow_var_seq_len

        # device tracker, since need to manually move experts not in use to CPU in distributed

        self.register_buffer("dummy", torch.ones(1), persistent=False)

    @property
    def device(self):
        return self.dummy.device

    def all_experts_to_cpu_besides(self, selection):
        if isinstance(selection, int):
            experts = [self.experts[selection]]
        if isinstance(selection, slice):
            experts = self.experts[selection]
        else:
            experts = selection

        experts_set = set(experts)

        for expert in self.experts:
            device = self.device if expert in experts_set else "cpu"
            expert.to(device)

    def forward(self, x, is_distributed=None):
        """
        einops notation:
        b - batch
        r - rank (device / machines)
        e - experts
        n - sequence (number of tokens per expert)
        d - feature dimension
        """

        # declare some variables

        is_distributed = default(is_distributed, self.is_distributed)
        shape, num_experts = x.shape, self.num_experts
        seq_len = shape[-2]

        # for now naively all gather across batch dimension if distributed, optimize later

        world_size = 1
        rank = 0

        if is_distributed:
            seq_sizes = gather_sizes(x, dim=-2)
            var_seq_len = not has_only_one_value(seq_sizes)

            assert (
                self.allow_var_seq_len or not var_seq_len
            ), "number of tokens per expert must be the same - if you want the framework to handle it, set `allow_var_seq_len = True` on `Experts`"

            # if variable sequence length, pad

            if var_seq_len:
                max_seq_size = seq_sizes.amax().item()
                x = pad_dim_to(x, max_seq_size, dim=-2)

            # gather and concat across batches, accounting for variable batch sizes

            x, batch_sizes = self.all_gather(x)
            total_batch_size = batch_sizes.sum().item()

            world_size = dist.get_world_size()
            rank = dist.get_rank()

        # the experts in use on the rank

        num_experts_per_rank = num_experts
        expert_slice = slice(0, num_experts)

        if is_distributed:
            if world_size <= num_experts:
                num_experts_across_ranks = chunk_num(num_experts, world_size)
                start_indices = cumsum_exclusive(
                    torch.tensor(num_experts_across_ranks), dim=-1
                )

                num_experts_per_rank = num_experts_across_ranks[rank]
                num_experts_batches_across_ranks = tuple(
                    i * total_batch_size for i in num_experts_across_ranks
                )

                expert_start_index = start_indices[rank].item()
            else:
                num_batch_chunks = world_size // num_experts
                total_ranks_in_use = num_batch_chunks * num_experts

                expert_start_index = rank // num_batch_chunks

                batch_splits = chunk_num(total_batch_size, num_batch_chunks)
                num_experts_batches_across_ranks = batch_splits * num_experts

                # for now, remaining machines just process nothing

                remain_ranks = world_size % num_experts
                num_experts_batches_across_ranks += (0,) * remain_ranks

                num_experts_per_rank = int(rank < total_ranks_in_use)

            assert len(num_experts_batches_across_ranks) == world_size

            expert_slice = slice(
                expert_start_index, expert_start_index + num_experts_per_rank
            )

        # if distributed, each machine only handles subset of experts and batch

        x = rearrange(x, "b e n d -> e b n d")

        if is_distributed:
            x, expert_batch_packed_shape = pack_one(x, "* n d")

            x = x.split(num_experts_batches_across_ranks, dim=0)
            x, experts_per_rank_sizes = split_by_rank(x)

            if num_experts_per_rank > 0:
                x = rearrange(x, "(e b) n d -> e b n d", e=num_experts_per_rank)
            else:
                x = x.reshape(num_experts, *x.shape)

        # get the experts in use

        self.all_experts_to_cpu_besides(expert_slice)

        experts = self.experts[expert_slice]

        # route tokens to appropriate experts

        outs = []

        for expert, expert_input in zip(experts, x):
            out = expert(expert_input)
            outs.append(out)

        if len(outs) > 0:
            outs = torch.stack(outs)
        else:
            outs = torch.empty_like(x, requires_grad=self.training)

        # all gather across merged expert batches dimensions
        # then split the batch dimension back

        if is_distributed:
            outs = rearrange(outs, "e b n d -> (e b) n d")
            outs, _ = self.all_gather(outs, sizes=experts_per_rank_sizes)
            outs = unpack_one(outs, expert_batch_packed_shape, "* n d")

        outs = rearrange(outs, "e b n d -> b e n d")

        if is_distributed:
            outs = outs.split(batch_sizes.tolist())
            outs, _ = split_by_rank(outs)

            # account for padded sequence length
            outs = outs[..., :seq_len, :]

        # Ignore this since LoRA for Wi will produce 3072 dimensions, which is the hidden size of the T5 model's FFN
        # assert outs.shape == shape
        return outs

    def grow_expert(self, mixlora_config, layer_weight):
        self.experts.append(
            Lora(layer_weight, config=mixlora_config, weight=None,)
        )
        self.num_experts += 1


class MoE(Module):

    @beartype
    def __init__(
        self,
        config: T5Config,
        mixlora_config: MixLoraConfig,
        dim,
        num_experts=16,
        expert_hidden_mult=4,
        threshold_train=0.2,
        threshold_eval=0.2,
        capacity_factor_train=1.25,
        capacity_factor_eval=2.0,
        gating_top_n=2,
        balance_loss_coef=1e-2,
        router_z_loss_coef=1e-3,
        experts: Module | None = None,
        straight_through_dispatch_tensor=True,
        differentiable_topk=False,
        differentiable_topk_fused=True,
        is_distributed=None,
        allow_var_seq_len=False,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.mixlora_config = mixlora_config
        # gate_class = (
        #     TopNGating_NaiveExpand if mixlora_config.naive_expand_lora else TopNGating
        # )
        self.gate = TopNGating_NaiveExpand(
            dim,
            top_n=gating_top_n,
            num_gates=num_experts,
            straight_through_dispatch_tensor=straight_through_dispatch_tensor,
            differentiable_topk=differentiable_topk,
            threshold_train=threshold_train,
            threshold_eval=threshold_eval,
            capacity_factor_train=capacity_factor_train,
            capacity_factor_eval=capacity_factor_eval,
            mixlora_config=mixlora_config,
        )

        # Mixlorat5layerff
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Original T5DenseActDense
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

        # experts = default(experts, lambda: [Expert(dim = dim, hidden_mult = expert_hidden_mult) for _ in range(num_experts)])

        wi_experts = default(
            experts,
            lambda: [
                Lora(
                    self.wi,
                    config=mixlora_config,
                    weight=None,
                )
                for _ in range(num_experts)
            ],
        )
        wo_experts = default(
            experts,
            lambda: [
                Lora(
                    self.wo,
                    config=mixlora_config,
                    weight=None,
                )
                for _ in range(num_experts)
            ],
        )
        self.wi_experts = Experts(
            wi_experts,
            is_distributed=is_distributed,
            allow_var_seq_len=allow_var_seq_len,
        )
        self.wo_experts = Experts(
            wo_experts,
            is_distributed=is_distributed,
            allow_var_seq_len=allow_var_seq_len,
        )

        self.balance_loss_coef = balance_loss_coef
        self.router_z_loss_coef = router_z_loss_coef

    def forward(
        self,
        x,
        noise_gates=False,
        noise_mult=1.0,
        idx=None,
    ):

        # T5LayerFF :forwarded_states = self.layer_norm(hidden_states)
        wi_inputs = self.layer_norm(x)
        
        # T5LayerFF: forwarded_states = self.DenseReluDense(forwarded_states)
        wi_output = self.wi(wi_inputs)
        wi_output = self.act(wi_output)
        wi_output = self.dropout(wi_output)

        # router
        dispatch_tensor, combine_tensor, balance_loss, router_z_loss, router_logits = (
            self.gate(wi_inputs, noise_gates=noise_gates, noise_mult=noise_mult)
        )

        ######### Wi
        # dispatch
        wi_expert_inputs = einsum(
            "b n d, b n e c -> b e c d", wi_inputs, dispatch_tensor
        )
        # feed the expert inputs through the experts.
        wi_expert_outputs = self.wi_experts(wi_expert_inputs)
        # combine
        wi_expert_outputs = einsum(
            "b e c d, b n e c -> b n d", wi_expert_outputs, combine_tensor
        )

        # Residual: W (wi_output) + Delta W (wi_expert_outputs)
        wi_output = wi_output + wi_expert_outputs.to(wi_output.dtype)

        ######### Wo
        if (
            isinstance(self.wo.weight, torch.Tensor)
            and x.dtype != self.wo.weight.dtype
            and self.wo.weight.dtype != torch.int8
        ):
            wi_output = wi_output.to(self.wo.weight.dtype)
            
        # T5DenseActDense: hidden_states = self.wo(hidden_states)
        wo_output = self.wo(wi_output)

        # dispatch
        wo_expert_inputs = einsum(
            "b n d, b n e c -> b e c d", wi_output, dispatch_tensor
        )

        # feed the expert inputs through the experts
        wo_expert_outputs = self.wo_experts(wo_expert_inputs)

        # combine
        wo_expert_outputs = einsum(
            "b e c d, b n e c -> b n d", wo_expert_outputs, combine_tensor
        )

        # Residual: W (wo_output) + Delta W (wo_expert_outputs)
        wo_output = wo_output + wo_expert_outputs.to(wo_output.dtype)
        
        # Final residual
        # T5LayerFF: hidden_states = hidden_states + self.dropout(forwarded_states)
        output = x + self.dropout(wo_output)
        # losses
        weighted_balance_loss = balance_loss * self.balance_loss_coef
        weighted_router_z_loss = router_z_loss * self.router_z_loss_coef

        # combine the losses
        total_aux_loss = weighted_balance_loss + weighted_router_z_loss
        return MixtureOfExpertsReturn(
            output,
            total_aux_loss,
            balance_loss,
            router_z_loss,
            router_logits,
        )

    def grow_expert(self, embedding=None):
        self.gate.grow_expert(embedding)
        self.wi_experts.grow_expert(self.mixlora_config, layer_weight=self.wi)
        self.wo_experts.grow_expert(self.mixlora_config, layer_weight=self.wo)
        self.num_experts += 1


@dataclass
class MixloraT5BaseModelOutput(BaseModelOutput):
    total_aux_loss: Optional[torch.FloatTensor] = None
    balance_loss: Optional[torch.FloatTensor] = None
    router_z_loss: Optional[torch.FloatTensor] = None
    all_router_logits: Optional[List[torch.FloatTensor]] = None


@dataclass
class MixloraT5BaseModelOutputWithPastAndCrossAttentions(BaseModelOutputWithPastAndCrossAttentions):
    total_aux_loss: Optional[torch.FloatTensor] = None
    balance_loss: Optional[torch.FloatTensor] = None
    router_z_loss: Optional[torch.FloatTensor] = None
    all_router_logits: Optional[List[torch.FloatTensor]] = None


@dataclass
class MixloraT5ForSemanticOutput(ModelOutput):
    semantic_output: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    router_logits: torch.FloatTensor = None
    balance_loss: torch.FloatTensor = None
    router_z_loss: torch.FloatTensor = None
    total_aux_loss: torch.FloatTensor = None
    all_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class MixloraT5Block(Module):
    def __init__(self, config, mixlora_config, has_relative_attention_bias=False, name=None):
        super().__init__()
        self.name = name
        self.is_decoder = config.is_decoder
        self.layer = ModuleList()
        self.layer.append(
            T5LayerSelfAttention(
                config, has_relative_attention_bias=has_relative_attention_bias
            )
        )
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        layer_id = int(name.split(".")[-1])
        self.layer.append(
            MoE(
                config, 
                mixlora_config,
                dim=config.d_model,
                num_experts = mixlora_config.num_experts[layer_id],
                gating_top_n = mixlora_config.top_k,
                balance_loss_coef=mixlora_config.balance_loss_coef,
                router_z_loss_coef=mixlora_config.router_z_loss_coef,
            )
        )

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_bias=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        encoder_decoder_position_bias=None,
        layer_head_mask=None,
        cross_attn_layer_head_mask=None,
        past_key_value=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
    ):
        if past_key_value is not None:
            if not self.is_decoder:
                logger.warning(
                    "`past_key_values` is passed to the encoder. Please make sure this is intended."
                )
            expected_num_past_key_values = 2 if encoder_hidden_states is None else 4

            if len(past_key_value) != expected_num_past_key_values:
                raise ValueError(
                    f"There should be {expected_num_past_key_values} past states. "
                    f"{'2 (key / value) for cross attention. ' if expected_num_past_key_values == 4 else ''}"
                    f"Got {len(past_key_value)} past key / value states"
                )

            self_attn_past_key_value = past_key_value[:2]
            cross_attn_past_key_value = past_key_value[2:]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=layer_head_mask,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[
            2:
        ]  # Keep self-attention outputs and relative position weights

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        do_cross_attention = self.is_decoder and encoder_hidden_states is not None
        if do_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )
            hidden_states = cross_attention_outputs[0]

            # clamp inf values to enable fp16 training
            if hidden_states.dtype == torch.float16:
                clamp_value = torch.where(
                    torch.isinf(hidden_states).any(),
                    torch.finfo(hidden_states.dtype).max - 1000,
                    torch.finfo(hidden_states.dtype).max,
                )
                hidden_states = torch.clamp(
                    hidden_states, min=-clamp_value, max=clamp_value
                )

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = (
                    present_key_value_state + cross_attention_outputs[1]
                )

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        # Type: MixtureOfExpertsReturn
        hidden_states = self.layer[-1](hidden_states)
        hidden_states, total_aux_loss, balance_loss, router_z_loss, router_logits = hidden_states

        # clamp inf values to enable fp16 training
        if hidden_states.dtype == torch.float16:
            clamp_value = torch.where(
                torch.isinf(hidden_states).any(),
                torch.finfo(hidden_states.dtype).max - 1000,
                torch.finfo(hidden_states.dtype).max,
            )
            hidden_states = torch.clamp(
                hidden_states, min=-clamp_value, max=clamp_value
            )

        outputs = (hidden_states,)

        if use_cache:
            outputs = outputs + (present_key_value_state,) + attention_outputs
        else:
            outputs = outputs + attention_outputs

        return (
            outputs,
            total_aux_loss,
            balance_loss,
            router_z_loss,
            router_logits,
        )

    def grow_expert(self, embedding=None):
        self.layer[-1].grow_expert(embedding)


class MixloraT5Stack(T5PreTrainedModel):
    def __init__(self, config, mixlora_config, embed_tokens=None):
        super().__init__(config)

        self.mixlora_config = mixlora_config
        self.embed_tokens = embed_tokens
        self.is_decoder = config.is_decoder

        # Adding MixLoRA
        self.block = ModuleList([])
        if config.is_decoder:
            mixlora_target_layers_dict = mixlora_config.decoder_target_layers
            name_prefix = "decoder.layer."
        else:
            mixlora_target_layers_dict = mixlora_config.encoder_target_layers
            name_prefix = "encoder.layer."

        for i in range(config.num_layers):
            if mixlora_target_layers_dict[i]:
                self.block.append(
                    MixloraT5Block(
                        config, 
                        has_relative_attention_bias=bool(i == 0),   mixlora_config=mixlora_config,
                        name=f"{name_prefix}{i}"
                    )
                )
            else:
                self.block.append(
                    T5Block(
                        config, 
                        has_relative_attention_bias=bool(i == 0), 
                    )
                )

        self.final_layer_norm = T5LayerNorm(
            config.d_model, eps=config.layer_norm_epsilon
        )
        self.dropout = nn.Dropout(config.dropout_rate)

        # Initialize weights and apply final processing
        self.post_init()
        # Model parallel
        self.model_parallel = False
        self.devicemap = None
        self.gradient_checkpointing = False

    def grow_expert(self, layer_id_list, embedding=None):
        """
        layer_id_list: list of layer ids to grow expert
        """
        for layer_id in layer_id_list:
            if self.mixlora_config.decoder_target_layers[
                layer_id
            ]:  # Check if the layer is MixLoRA layer
                self.block[
                    layer_id
                ].grow_expert(embedding)  # Grow 1 expert for the MixloraT5Block
                self.mixlora_config.num_experts[
                    layer_id
                ] += 1  # Update the number of experts in the config
        return self.mixlora_config

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5Stack.parallelize` is deprecated and will be removed in v5 of Transformers, you should load your model"
            " with `device_map='balanced'` in the call to `from_pretrained`. You can also provide your own"
            " `device_map` but it needs to be a dictionary module_name to device, so for instance {'block.0': 0,"
            " 'block.1': 1, ...}",
            FutureWarning,
        )
        # Check validity of device_map
        self.devicemap = (
            get_device_map(len(self.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.devicemap, len(self.block))
        self.model_parallel = True
        self.first_device = (
            "cpu"
            if "cpu" in self.devicemap.keys()
            else "cuda:" + str(min(self.devicemap.keys()))
        )
        self.last_device = "cuda:" + str(max(self.devicemap.keys()))
        # Load onto devices
        for k, v in self.devicemap.items():
            for layer in v:
                cuda_device = "cuda:" + str(k)
                self.block[layer] = self.block[layer].to(cuda_device)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.model_parallel = False
        self.devicemap = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        for i in range(len(self.block)):
            self.block[i] = self.block[i].to("cpu")
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Model parallel
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = (
            output_attentions
            if output_attentions is not None
            else self.config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else self.config.output_hidden_states
        )
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if input_ids is not None and inputs_embeds is not None:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You cannot specify both {err_msg_prefix}input_ids and {err_msg_prefix}inputs_embeds at the same time"
            )
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(
                f"You have to specify either {err_msg_prefix}input_ids or {err_msg_prefix}inputs_embeds"
            )

        if inputs_embeds is None:
            if self.embed_tokens is None:
                raise ValueError(
                    "You have to initialize the model with valid token embeddings"
                )
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = (
            past_key_values[0][0].shape[2] + seq_length
            if past_key_values is not None
            else seq_length
        )

        if use_cache is True:
            if not self.is_decoder:
                raise ValueError(
                    f"`use_cache` can only be set to `True` if {self} is used as a decoder"
                )

        # initialize past_key_values with `None` if past does not exist
        if past_key_values is None:
            past_key_values = [None] * len(self.block)

        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, mask_seq_length, device=inputs_embeds.device
            )

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(
            attention_mask, input_shape
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = (
                encoder_hidden_states.size()
            )
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(
                    encoder_hidden_shape, device=inputs_embeds.device, dtype=torch.long
                )
            encoder_extended_attention_mask = self.invert_attention_mask(
                encoder_attention_mask
            )
        else:
            encoder_extended_attention_mask = None

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        cross_attn_head_mask = self.get_head_mask(
            cross_attn_head_mask, self.config.num_layers
        )
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)
        all_router_logits = {}
        total_aux_loss = 0.0
        balance_loss = 0.0
        router_z_loss = 0.0

        for i, (layer_module, past_key_value) in enumerate(
            zip(self.block, past_key_values)
        ):
            layer_head_mask = head_mask[i]
            cross_attn_layer_head_mask = cross_attn_head_mask[i]
            # Model parallel
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                # Ensure that attention_mask is always on the same device as hidden_states
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if position_bias is not None:
                    position_bias = position_bias.to(hidden_states.device)
                if encoder_hidden_states is not None:
                    encoder_hidden_states = encoder_hidden_states.to(
                        hidden_states.device
                    )
                if encoder_extended_attention_mask is not None:
                    encoder_extended_attention_mask = (
                        encoder_extended_attention_mask.to(hidden_states.device)
                    )
                if encoder_decoder_position_bias is not None:
                    encoder_decoder_position_bias = encoder_decoder_position_bias.to(
                        hidden_states.device
                    )
                if layer_head_mask is not None:
                    layer_head_mask = layer_head_mask.to(hidden_states.device)
                if cross_attn_layer_head_mask is not None:
                    cross_attn_layer_head_mask = cross_attn_layer_head_mask.to(
                        hidden_states.device
                    )
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            router_logits = None
            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.forward,
                    hidden_states,
                    extended_attention_mask,
                    position_bias,
                    encoder_hidden_states,
                    encoder_extended_attention_mask,
                    encoder_decoder_position_bias,
                    layer_head_mask,
                    cross_attn_layer_head_mask,
                    None,  # past_key_value is always None with gradient checkpointing
                    use_cache,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask=extended_attention_mask,
                    position_bias=position_bias,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    encoder_decoder_position_bias=encoder_decoder_position_bias,
                    layer_head_mask=layer_head_mask,
                    cross_attn_layer_head_mask=cross_attn_layer_head_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                )

            if isinstance(layer_module, MixloraT5Block):
                layer_outputs, _total_aux_loss, _balance_loss, _router_zloss, router_logits = layer_outputs
                total_aux_loss += _total_aux_loss
                balance_loss += _balance_loss
                router_z_loss += _router_zloss
                all_router_logits[i] = router_logits
            else:
                all_router_logits[i] = None

            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)
            if use_cache is False:
                layer_outputs = layer_outputs[:1] + (None,) + layer_outputs[1:]

            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention position bias), (self-attention weights),
            # (cross-attention position bias), (cross-attention weights)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[
                    4 if output_attentions else 3
                ]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (
                    present_key_value_state,
                )

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            # Model Parallel: If it's the last layer for that device, put things on the next device
            if self.model_parallel:
                for k, v in self.devicemap.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    present_key_value_states,
                    all_hidden_states,
                    all_attentions,
                    all_cross_attentions,
                    total_aux_loss,
                    balance_loss,
                    router_z_loss,
                ]
                if v is not None
            )
        return MixloraT5BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
            total_aux_loss=total_aux_loss,
            balance_loss=balance_loss,
            router_z_loss=router_z_loss,
            all_router_logits=all_router_logits,
        )


class MixloraT5ForConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
        "lm_head.weight",
    ]

    def grow_expert(self, layer_id_list, embedding=None):
        # Only need to grow the decoder for now
        self.mixlora_config = self.decoder.grow_expert(layer_id_list, embedding)
        return self.mixlora_config

    def __init__(self, config: T5Config, mixlora_config: MixLoraConfig):
        super().__init__(config)
        self.mixlora_config = mixlora_config
        self.model_dim = config.d_model

        # MixLora configs
        self.mixlora_encoder = mixlora_config.encoder
        self.mixlora_decoder = mixlora_config.decoder
        self.mixlora_encoder_attention = mixlora_config.encoder_attention
        self.mixlora_decoder_attention = mixlora_config.decoder_attention
        self.num_experts = mixlora_config.num_experts

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        if self.mixlora_encoder:
            self.encoder = MixloraT5Stack(encoder_config, mixlora_config, self.shared)
            print("Enabled Mixlora in encoder")
        else:
            self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers

        if self.mixlora_decoder:
            self.decoder = MixloraT5Stack(decoder_config, mixlora_config, self.shared)
            print("Enabled Mixlora in decoder")
        else:
            self.decoder = T5Stack(decoder_config, self.shared)

        # if self.mixlora_encoder_attention or self.mixlora_decoder_attention:
        #     inject_adapter_in_model(self, config=mixlora_config)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.devicemap = None

    @add_start_docstrings(PARALLELIZE_DOCSTRING)
    def parallelize(self, device_map=None):
        warnings.warn(
            "`T5ForConditionalGeneration.parallelize` is deprecated and will be removed in v5 of Transformers, you"
            " should load your model with `device_map='balanced'` in the call to `from_pretrained`. You can also"
            " provide your own `device_map` but it needs to be a dictionary module_name to device, so for instance"
            " {'encoder.block.0': 0, 'encoder.block.1': 1, ...}",
            FutureWarning,
        )
        self.devicemap = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.devicemap, len(self.encoder.block))
        self.encoder.parallelize(self.devicemap)
        self.decoder.parallelize(self.devicemap)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    @add_start_docstrings(DEPARALLELIZE_DOCSTRING)
    def deparallelize(self):
        warnings.warn(
            "Like `parallelize`, `deparallelize` is deprecated and will be removed in v5 of Transformers.",
            FutureWarning,
        )
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.lm_head = self.lm_head.to("cpu")
        self.model_parallel = False
        self.devicemap = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(
        output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC
    )
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[-100, 0, ...,
            config.vocab_size - 1]`. All labels set to `-100` are ignored (masked), the loss is only computed for
            labels in `[0, ..., config.vocab_size]`

        Returns:

        Examples:

        ```python
        >>> from transformers import AutoTokenizer, T5ForConditionalGeneration

        >>> tokenizer = AutoTokenizer.from_pretrained("google-t5/t5-small")
        >>> model = T5ForConditionalGeneration.from_pretrained("google-t5/t5-small")

        >>> # training
        >>> input_ids = tokenizer("The <extra_id_0> walks in <extra_id_1> park", return_tensors="pt").input_ids
        >>> labels = tokenizer("<extra_id_0> cute dog <extra_id_1> the <extra_id_2>", return_tensors="pt").input_ids
        >>> outputs = model(input_ids=input_ids, labels=labels)
        >>> loss = outputs.loss
        >>> logits = outputs.logits

        >>> # inference
        >>> input_ids = tokenizer(
        ...     "summarize: studies have shown that owning a dog is good for you", return_tensors="pt"
        ... ).input_ids  # Batch size 1
        >>> outputs = model.generate(input_ids)
        >>> print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        >>> # studies have shown that owning a dog is good for you.
        ```"""
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning)  # type: ignore
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

        # Decode

        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            labels = labels.to(lm_logits.device)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        decoder_attention_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        # cut decoder_input_ids if past_key_values is used
        if past_key_values is not None:
            past_length = past_key_values[0][0].shape[2]

            # Some generation methods already pass only the last input ID
            if input_ids.shape[1] > past_length:
                remove_prefix_length = past_length
            else:
                # Default to old behavior: keep only final ID
                remove_prefix_length = input_ids.shape[1] - 1

            input_ids = input_ids[:, remove_prefix_length:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past_key_values,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "decoder_attention_mask": decoder_attention_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return self._shift_right(labels)

    def _reorder_cache(self, past_key_values, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past_key_values is None:
            logger.warning(
                "You might want to consider setting `use_cache=True` to speed up decoding"
            )
            return past_key_values

        reordered_decoder_past = ()
        for layer_past_states in past_key_values:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states:
                # need to set correct `past` for each of the four key / value states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(
                        0, beam_idx.to(layer_past_state.device)
                    ),
                )

            if reordered_layer_past_states[0].shape != layer_past_states[0].shape:
                raise ValueError(
                    f"reordered_layer_past_states[0] shape {reordered_layer_past_states[0].shape} and layer_past_states[0] shape {layer_past_states[0].shape} mismatched"
                )
            if len(reordered_layer_past_states) != len(layer_past_states):
                raise ValueError(
                    f"length of reordered_layer_past_states {len(reordered_layer_past_states)} and length of layer_past_states {len(layer_past_states)} mismatched"
                )

            reordered_decoder_past = reordered_decoder_past + (
                reordered_layer_past_states,
            )
        return reordered_decoder_past


class MixloraT5ForSemanticGeneration(MixloraT5ForConditionalGeneration):
    def __init__(self, config: T5Config, mixlora_config: MixLoraConfig):
        super().__init__(config, mixlora_config)
        self.return_logits = False

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        decoder_head_mask: Optional[torch.FloatTensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = True, # Optional[bool] = None,
        return_dict: Optional[bool] = None,
        idx: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if head_mask is not None and decoder_head_mask is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                warnings.warn(__HEAD_MASK_WARNING_MSG, FutureWarning) # type: ignore
                decoder_head_mask = head_mask

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = MixloraT5BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # if isinstance(encoder_outputs, MixloraT5BaseModelOutputWithPastAndCrossAttentions):
        #     encoder_total_aux_loss, encoder_balance_loss, encoder_router_z_loss, all_encoder_router_logits = (
        #         encoder_outputs["total_aux_loss"],
        #         encoder_outputs["balance_loss"],
        #         encoder_outputs["router_z_loss"],
        #         encoder_outputs["all_router_logits"]
        #     )
        # else:
        encoder_total_aux_loss, encoder_balance_loss, encoder_router_z_loss = (
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
        )
        all_encoder_router_logits = {}
        hidden_states = encoder_outputs[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if (
            labels is not None
            and decoder_input_ids is None
            and decoder_inputs_embeds is None
        ):
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(
                    self.decoder.first_device
                )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # router logits shape: (sequence_length, num_adapters)
        # expect all_router logits: list of 12 * 2 = 24 layers router logits
        # Thus: all_encoder_router_logits = list of len 12 (layers) with each element is [bs*encode_sequence_length, num_adapters]
        # Thus: all_decoder_router_logits = list of len 12 (layers) with each element is [bs*decode_sequence_length, num_adapters]

        if isinstance(decoder_outputs, MixloraT5BaseModelOutputWithPastAndCrossAttentions):
            decoder_total_aux_loss, decoder_balance_loss, decoder_router_z_loss, all_decoder_router_logits = (
                decoder_outputs["total_aux_loss"], decoder_outputs["balance_loss"], decoder_outputs["router_z_loss"], decoder_outputs["all_router_logits"]
            )
        else:
            decoder_total_aux_loss, decoder_balance_loss, decoder_router_z_loss = (
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
            )
            all_decoder_router_logits = {}

        all_router_logits = {
            "encoder": all_encoder_router_logits, 
            "decoder": all_decoder_router_logits,
        }

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        # if self.config.tie_word_embeddings:
        # Rescale output before projecting on vocab
        # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
        #    sequence_output = sequence_output * (self.model_dim**-0.5)

        loss = None
        if labels is not None:
            raise NotImplementedError
            # loss_fct = CrossEntropyLoss(ignore_index=-100)
            # move labels to correct device to enable PP
            # labels = labels.to(lm_logits.device)
            # loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (sequence_output) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        if self.return_logits:
            return MixloraT5ForSemanticOutput(
                logits=self.lm_head(sequence_output),
                router_logits=all_router_logits,
                semantic_output=sequence_output,
                balance_loss=encoder_balance_loss + decoder_balance_loss,
                router_z_loss=encoder_router_z_loss + decoder_router_z_loss,
                total_aux_loss=encoder_total_aux_loss + decoder_total_aux_loss,
                all_hidden_states=decoder_outputs.hidden_states,
            )
        else:
            return MixloraT5ForSemanticOutput(
                semantic_output=sequence_output  # [bz smtid_length, d_model]
            )


class MixLoraDSI(Module): # Now implemented a masked head    

    def __init__(self, model_name_or_path, mixlora_config, base_model_cls=MixloraT5ForSemanticGeneration):

        super().__init__()
        # Configs
        config = T5Config.from_pretrained(model_name_or_path)
        self.config = config
        self.mixlora_config = mixlora_config
        self.router_z_loss_coef = mixlora_config.router_z_loss_coef

        # Model
        self.base_model = base_model_cls
        self.base_model = self.base_model.from_pretrained(
            model_name_or_path, config=config, 
            mixlora_config=self.mixlora_config,
            ignore_mismatched_sizes=True
        )
        self.device = self.base_model.device
        self.base_model.return_logits = True
        self.rq_specific_mask_head = self.mixlora_config.rq_specific_mask_head # For RQ specific masked head

        checkpoint = {}
        if (Path(model_name_or_path) / "model.safetensors").exists():
            with safe_open(Path(model_name_or_path) / "model.safetensors", framework="pt", device=0) as f: # type: ignore
                # This is most likely t5-self-neg checkpoint
                for k in f.keys():
                    checkpoint[k] = f.get_tensor(k)
        elif (Path(model_name_or_path) / "pytorch_model.bin").exists():
            checkpoint = torch.load(os.path.join(model_name_or_path, "pytorch_model.bin"))

        if mixlora_config.encoder:
            encoder_stack = getattr(self.base_model, "encoder")
            for i_block, block in enumerate(encoder_stack.block):
                moe_layer = getattr(block, "layer")[1]
                if isinstance(moe_layer, MoE) and f"encoder.block.{i_block}.layer.1.DenseReluDense.wi.weight" in checkpoint.keys():
                    wi_weight = checkpoint[f"encoder.block.{i_block}.layer.1.DenseReluDense.wi.weight"]
                    wo_weight = checkpoint[f"encoder.block.{i_block}.layer.1.DenseReluDense.wo.weight"]
                    moe_layer.wi.weight.data = wi_weight
                    moe_layer.wo.weight.data = wo_weight

                    assert moe_layer.wi.weight.sum() == wi_weight.sum()
                    print(f"Initialized encoder block {i_block} moe layer with wi,wo weights")

        if mixlora_config.decoder:
            decoder_stack = getattr(self.base_model, "decoder")
            for i_block, block in enumerate(decoder_stack.block):
                moe_layer = getattr(block, "layer")[2]
                if (
                    isinstance(moe_layer, MoE)
                    and f"decoder.block.{i_block}.layer.2.DenseReluDense.wi.weight"
                    in checkpoint.keys()
                ):
                    wi_weight = checkpoint[f"decoder.block.{i_block}.layer.2.DenseReluDense.wi.weight"]
                    wo_weight = checkpoint[f"decoder.block.{i_block}.layer.2.DenseReluDense.wo.weight"]
                    moe_layer.wi.weight.data = wi_weight
                    moe_layer.wo.weight.data = wo_weight
                    assert moe_layer.wi.weight.sum() == wi_weight.sum()
                    print(f"Initialized decoder block {i_block} moe layer with wi,wo weights")

        if self.rq_specific_mask_head:
            self.register_buffer(
                "masks", torch.zeros((8, 32100 + 2048 * 7), dtype=torch.int64)
            )
            temp = [
                [i for i in range(32100 + 2048 * j)]
                + [i for i in range(32100 + 2048 * (j + 1), 32100 + 2048 * 8)]
                for j in range(8)
            ]
            for i, mask in enumerate(temp):
                self.masks[i] = torch.tensor(mask, dtype=torch.int64)

        # Losses
        self.rank_loss_fn = torch.nn.CrossEntropyLoss()
        self.previous_checkpoint = None

    def forward(self, **inputs):
        """
        Args:
            tokenized_query: [bz, max_length] + [bz, smtid_length]
            labels: [bz, smtid_length]
        """
        output, logits = self._minimal_forward(inputs)
        
        if self.training:
            rank_loss, total_aux_loss, balance_loss, router_z_loss = self._minimal_calc_loss(inputs, output, logits)
            
            all_router_logits = output.router_logits

            return {
                "loss": rank_loss + total_aux_loss,
                "rank_loss": rank_loss,
                "total_aux_loss": total_aux_loss,
                "balance_loss": balance_loss,
                "router_z_loss": router_z_loss,
                "all_router_logits": all_router_logits,
            }

    def _minimal_calc_loss(self, inputs, output, logits):
        bz, smtid_length = inputs["labels"].size()
        rank_loss = self.rank_loss_fn(
                logits.view(bz * smtid_length, -1), inputs["labels"].view(-1)
            )

        total_aux_loss = output.total_aux_loss
        balance_loss = output.balance_loss
        router_z_loss = output.router_z_loss
        return rank_loss,total_aux_loss,balance_loss,router_z_loss

    def _minimal_forward(self, inputs):
        output = self.base_model(**inputs["tokenized_query"])

        logits = output.logits  # [bz, smtid_length, vocab_size]
        if self.rq_specific_mask_head:
            for i, mask in enumerate(self.masks):
                logits[:, i, :] = logits[:, i, :].index_fill_(
                    dim=-1, index=mask, value=float("-inf")
                )
                
        return output,logits

    @classmethod
    def from_pretrained(cls, model_name_or_path, mixlora_config):
        return cls(model_name_or_path, mixlora_config)

    def save_pretrained(self, save_dir, safe_serialization=False):
        self.base_model.save_pretrained(
            save_dir,
            safe_serialization=safe_serialization,
        )

    def get_model_checkpoint(self, model_name_or_path):
        checkpoint = {}
        if (Path(model_name_or_path) / "model.safetensors").exists():
            with safe_open(Path(model_name_or_path) / "model.safetensors", framework="pt", device=0) as f:  # type: ignore
                # This is most likely t5-self-neg checkpoint
                for k in f.keys():
                    checkpoint[k] = f.get_tensor(k)
        elif (Path(model_name_or_path) / "pytorch_model.bin").exists():
            checkpoint = torch.load(
                os.path.join(model_name_or_path, "pytorch_model.bin")
            )
        return checkpoint

    def _freeze_base_model(self, freeze_vocab=True):
        mixlora_components = ["experts", "to_gates", "gate_"]
        if not freeze_vocab:
            mixlora_components.append("shared") # Adding shared to trainable params
        for name, param in self.base_model.named_parameters():
            if not any([comp in name for comp in mixlora_components]):
                param.requires_grad = False
                
        print("\n##### Params info after freezing base model #####")
        get_params_info(self.base_model)


class MixLoraDSI_sequential_finetuning(MixLoraDSI):
    def __init__(self, model_name_or_path, mixlora_config, base_model_cls=MixloraT5ForSemanticGeneration):
        super().__init__(model_name_or_path, mixlora_config, base_model_cls)
        self.previous_checkpoint = None

    def _before_task(self):
        get_params_info(self.base_model)


class CLEVER(MixLoraDSI_sequential_finetuning):
    def __init__(self, model_name_or_path, mixlora_config, base_model_cls=MixloraT5ForSemanticGeneration):
        super().__init__(model_name_or_path, mixlora_config, base_model_cls)
        self.task_id = 0
        self.ewc = EWC()
        self.fisher_dict = {}
        self.optpar_dict = {}

    def _before_task(self, dataloader, task_id):
        self.task_id = task_id
        self.ewc.before_task(self.base_model, dataloader, task_id=task_id-1) # EWC is calculated for the previous task, thus task_id-1
        get_params_info(self.base_model)

    # def _after_task(self, task_id):
    #     self.task_id = task_id
    #     self.ewc.after_task(self.base_model, self.task_id)

    def forward(self, **inputs):
        """
        Args:
            tokenized_query: [bz, max_length] + [bz, smtid_length]
            labels: [bz, smtid_length]
        """
        output = self.base_model(**inputs["tokenized_query"])

        logits = output.logits  # [bz, smtid_length, vocab_size]
        if self.rq_specific_mask_head:
            for i, mask in enumerate(self.masks):
                logits[:, i, :] = logits[:, i, :].index_fill_(
                    dim=-1, index=mask, value=float("-inf")
                )

        bz, smtid_length = inputs["labels"].size()
        rank_loss = self.rank_loss_fn(
            logits.view(bz * smtid_length, -1), inputs["labels"].view(-1)
        )

        # Source: https://github.com/ContinualAI/colab/blob/master/notebooks/intro_to_continual_learning.ipynb
        ewc_loss = self.ewc(self.base_model, task_id=self.task_id)

        return {
            "loss": rank_loss + ewc_loss,
            "rank_loss": rank_loss,
            "ewc_loss": ewc_loss,
        }


class MixLoraDSI_naive_expand(MixLoraDSI):
    def __init__(
        self,
        model_name_or_path,
        mixlora_config,
        base_model_cls=MixloraT5ForSemanticGeneration,
    ):
        super().__init__(model_name_or_path, mixlora_config, base_model_cls)
        self.kl_loss_coef = mixlora_config.kl_loss_coef
        self.cosine_sim_loss_coef = mixlora_config.cosine_sim_loss_coef
        self.router_contrastive_loss_coef = (
            self.mixlora_config.router_contrastive_loss_coef
        )
        self.cosine_loss = nn.CosineEmbeddingLoss(reduction="mean")

    def forward(self, **inputs):
        if self.previous_checkpoint is not None:
            with torch.no_grad():
                self.base_model.shared.weight[:32100].copy_(self.previous_checkpoint.base_model.shared.weight.detach()[:32100])
        output, logits = self._minimal_forward(inputs)

        if self.training:
            rank_loss, total_aux_loss, balance_loss, router_z_loss = (
                self._minimal_calc_loss(inputs, output, logits)
            )

            kl_loss = self._calc_kl_loss(inputs, logits)

            cosine_sim_loss = self._calc_cosine_sim_loss(output, logits)

            router_contrastive_loss = self._calc_router_contrastive_loss(output, logits)

            all_router_logits = output.router_logits

            return {
                "loss": rank_loss
                + total_aux_loss
                + kl_loss
                + cosine_sim_loss
                + router_contrastive_loss,
                "rank_loss": rank_loss,
                "kl_loss": kl_loss,
                "cosine_sim_loss": cosine_sim_loss,
                "router_contrastive_loss": router_contrastive_loss,
                "total_aux_loss": total_aux_loss,
                "balance_loss": balance_loss,
                "router_z_loss": router_z_loss,
                "all_router_logits": all_router_logits,
            }

    def _before_task(self, train_loader=None, taskid=-1, freeze_vocab=False):
        self._freeze_base_model(freeze_vocab=freeze_vocab)
        if taskid == 1:
            return  # No need to expand for the first task

        layer_idx_to_grow_list = []
        for layer_idx, enable in self.mixlora_config.decoder_target_layers.items():
            if enable:
                layer_idx_to_grow_list.append(layer_idx)
        self.mixlora_config = self.base_model.grow_expert(layer_idx_to_grow_list)
        self._freeze_previous_router_experts(freeze_vocab=freeze_vocab)

    def _freeze_router(self):
        mixlora_components = ["to_gates", "gate_"]
        for name, param in self.base_model.named_parameters():
            if any([comp in name for comp in mixlora_components]):
                param.requires_grad = False

    def _freeze_previous_router_experts(self, freeze_vocab=False):
        with torch.no_grad():
            for name, param in self.base_model.named_parameters():
                if (
                    not freeze_vocab and "shared" in name
                ):  # During continual learning, we enable the vocab embedding to be trainable
                    param.requires_grad = True
                elif "to_gates" in name:  # Router weights
                    param.requires_grad = True
                elif "experts" in name:  # Lora experts
                    layer = int(name.split(".")[2])
                    expert_id = int(name.split(".")[-3])
                    num_experts = self.mixlora_config.num_experts[layer]
                    param.requires_grad = False

                    if expert_id >= num_experts - 1:
                        param.requires_grad = True
                else:
                    param.requires_grad = False
        print("##### Params info after naive expanding #####")
        get_params_info(self.base_model)
        print("##### Params info after naive expanding #####")

    def grow_expert(self, layer_idx, embedding=None):
        """
        layer_idx: a list of integers representing the layer indices where the expert should be added
        """
        self.mixlora_config = self.base_model.grow_expert(layer_idx, embedding)

    def _calc_kl_loss(self, inputs, logits):
        kl_loss = torch.tensor(0.0).to(logits.device)
        if self.previous_checkpoint is None or not self.mixlora_config.kl_loss:
            return kl_loss

        self.previous_checkpoint.eval()
        with torch.no_grad():
            previous_output = self.previous_checkpoint(**inputs["tokenized_query"])

        previous_logits = (
            previous_output.logits.detach()
        )  # [bz, smtid_length, vocab_size]

        if self.rq_specific_mask_head:
            for i, mask in enumerate(self.masks):
                previous_logits[:, i, :] = previous_logits[:, i, :].index_fill_(
                    dim=-1, index=mask, value=float("-inf")
                )

        for i, _ in enumerate(self.masks):
            kl_loss += F.kl_div(
                F.log_softmax(
                    logits[:, i, 32100 + 2048 * i : 32100 + 2048 * (i + 1)], dim=-1
                ),
                F.softmax(
                    previous_logits[:, i, 32100 + 2048 * i : 32100 + 2048 * (i + 1)],
                    dim=-1,
                ),
                reduction="batchmean",
            )
        kl_loss = self.kl_loss_coef * kl_loss
        return kl_loss

    def _calc_cosine_sim_loss(self, output, logits):
        # This cosine loss consider all tokens of the current task need to be similar to the new router weight.
        cosine_sim_loss = torch.tensor(0.0).to(logits.device)
        if (
            not self.mixlora_config.cosine_sim_loss
            or not self.mixlora_config.cosine_classifier
        ):
            return -cosine_sim_loss

        for k, v in output.router_logits["decoder"].items():
            if v is not None:
                # Note that v is already the cosine simiarlty
                v = v.reshape(-1, v.shape[-1])  # Flatten the router logits
                if (
                    self.mixlora_config.num_experts[k] > 2
                ):  # Since this is naive expansion, if there are more than 2 experts, we only need to consider the last expert
                    v = v[:, [-1]]
                cosine_sim_loss += (torch.ones_like(v, device=v.device) - v).mean()
        cosine_sim_loss = self.cosine_sim_loss_coef * cosine_sim_loss
        return cosine_sim_loss

    def _calc_router_contrastive_loss(self, output, logits):
        router_contrastive_loss = torch.tensor(0.0).to(logits.device)

        if not self.mixlora_config.router_contrastive_loss:
            return router_contrastive_loss

        for k, v in output.router_logits["decoder"].items():
            if v is not None:
                router_contrastive_loss += self._contrastive_loss(router_weights=self.base_model.decoder.block[k].layer[-1].gate.to_gates.weight)
        return self.router_contrastive_loss_coef * router_contrastive_loss

    def _contrastive_loss(
        self,
        router_weights: torch.Tensor,
    ) -> torch.Tensor:
        """
        Einstein notation:
        b - batch
        n - sequence
        d - dim
        e - experts/router weights
        """

        # This implementation only consider the contrastive loss between router weights, not the features, which should be the case for energy-based contrastive loss

        router_contrastive_loss = torch.tensor(0.0).to(router_weights.device)

        previous_router_weights = router_weights[:-1]  # (e, d)
        new_router_weights = router_weights[[-1]]  # (1, d)

        if len(previous_router_weights.shape) < 2:
            previous_router_weights = previous_router_weights.unsqueeze(0)

        # The cosine_embedding loss: creates a criterion that measures the loss given input tensors  x1, x2 ​and a Tensor label  y with values 1 or -1. Use (y=1) to maximize the cosine similarity of two inputs, and (y=−1) otherwise. This is typically used for learning nonlinear embeddings or semi-supervised learning.
        new_router_weights = new_router_weights.repeat(previous_router_weights.shape[0], 1)

        targets = -torch.ones(previous_router_weights.shape[0], device=router_weights.device)

        router_contrastive_loss = self.cosine_loss(input1=new_router_weights,input2=previous_router_weights,target=targets)
        return router_contrastive_loss


class CorpusBrainPlusPlus(Module):

    def __init__(
        self,
        model_name_or_path,
        mixlora_config,
        base_model_cls=MixloraT5ForSemanticGeneration,
    ):
        super().__init__()
        # Configs
        config = T5Config.from_pretrained(model_name_or_path)
        self.config = config
        self.mixlora_config = mixlora_config
        self.router_z_loss_coef = mixlora_config.router_z_loss_coef
        self.adapter_config = None

        # Model
        self.base_model = AutoAdapterModel.from_pretrained(model_name_or_path)
        temp = base_model_cls.from_pretrained(model_name_or_path, mixlora_config)
        self.base_model._tied_weights_keys = [
            "encoder.embed_tokens.weight",
            "decoder.embed_tokens.weight",
            "lm_head.weight",
        ]

        self.device = self.base_model.device
        self.base_model.return_logits = True
        self.rq_specific_mask_head = (
            self.mixlora_config.rq_specific_mask_head
        )  # For RQ specific masked head

        if self.rq_specific_mask_head:
            self.register_buffer(
                "masks", torch.zeros((8, 32100 + 2048 * 7), dtype=torch.int64)
            )
            temp = [
                [i for i in range(32100 + 2048 * j)]
                + [i for i in range(32100 + 2048 * (j + 1), 32100 + 2048 * 8)]
                for j in range(8)
            ]
            for i, mask in enumerate(temp):
                self.masks[i] = torch.tensor(mask, dtype=torch.int64)

        # Losses
        self.rank_loss_fn = torch.nn.CrossEntropyLoss()
        self.previous_checkpoint = None

    def _before_task(self):
        for name, param in self.base_model.named_parameters():
            if "shared" in name or "adapter" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        self.get_params_info()
        
    def forward(self, **inputs):
        """
        Args:
            tokenized_query: [bz, max_length] + [bz, smtid_length]
            labels: [bz, smtid_length]
        """
        logits = self.base_model(
            **inputs["tokenized_query"]
        ).logits  # [bz, smtid_length, vocab_size]

        # if self.masked_head:
        #     for i, mask in enumerate(self.masks):
        #         logits[:, i, :] = logits[:, i, :].index_fill_(
        #             dim=-1, index=mask, value=float("-inf")
        #         )
        if not self.training:
            return logits

        bz, smtid_length = inputs["labels"].size()
        rank_loss = self.rank_loss_fn(
            logits.view(bz * smtid_length, -1), inputs["labels"].view(-1)
        )

        return {
            "loss": rank_loss,
        }

    def _minimal_calc_loss(self, inputs, output, logits):
        bz, smtid_length = inputs["labels"].size()
        rank_loss = self.rank_loss_fn(
                logits.view(bz * smtid_length, -1), inputs["labels"].view(-1)
            )

        st()
        total_aux_loss = output.total_aux_loss
        balance_loss = output.balance_loss
        router_z_loss = output.router_z_loss
        return rank_loss,total_aux_loss,balance_loss,router_z_loss

    def _minimal_forward(self, inputs):
        output = self.base_model(**inputs["tokenized_query"])

        logits = output.logits  # [bz, smtid_length, vocab_size]
        if self.rq_specific_mask_head:
            for i, mask in enumerate(self.masks):
                logits[:, i, :] = logits[:, i, :].index_fill_(
                    dim=-1, index=mask, value=float("-inf")
                )
                
        return output,logits

    @classmethod
    def from_pretrained(cls, model_name_or_path, mixlora_config):
        return cls(model_name_or_path, mixlora_config)

    def save_pretrained(self, save_dir, safe_serialization=False):
        self.base_model.save_pretrained(
            save_dir,
            safe_serialization=safe_serialization,
        )

    def get_model_checkpoint(self, model_name_or_path):
        checkpoint = {}
        if (Path(model_name_or_path) / "model.safetensors").exists():
            with safe_open(Path(model_name_or_path) / "model.safetensors", framework="pt", device=0) as f:  # type: ignore
                # This is most likely t5-self-neg checkpoint
                for k in f.keys():
                    checkpoint[k] = f.get_tensor(k)
        elif (Path(model_name_or_path) / "pytorch_model.bin").exists():
            checkpoint = torch.load(
                os.path.join(model_name_or_path, "pytorch_model.bin")
            )
        return checkpoint

    def _freeze_base_model(self, freeze_vocab=True):
        mixlora_components = ["experts", "to_gates", "gate_"]
        if not freeze_vocab:
            mixlora_components.append("shared") # Adding shared to trainable params
        for name, param in self.base_model.named_parameters():
            if not any([comp in name for comp in mixlora_components]):
                param.requires_grad = False

        print("\n##### Params info after freezing base model #####")
        self.get_params_info()
        
    def get_params_info(self):
        all_param = 0
        trainable_param = 0
        mixlora_param = 0
        mixlora_components = ["adapter"]
        print("All trainable parameters:")
        for name, param in self.base_model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                if "shared" in name:
                    trainable_param += 768 * 2048 * 8
                else:
                    trainable_param += param.numel()
                print(name, param.numel())
            if any([comp in name for comp in mixlora_components]):
                mixlora_param += param.numel()
        print(f" # all param       : {all_param}")
        print(f" # trainable param : {trainable_param}")
        print(f" # adapter param   : {mixlora_param}")
        print(f" # % trainable parameters: {trainable_param/all_param*100:.2f}%")
        print(f" # % adapter parameters  : {mixlora_param/all_param*100:.2f}%")