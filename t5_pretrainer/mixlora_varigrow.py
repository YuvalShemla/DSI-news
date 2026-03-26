from __future__ import annotations

import copy
import math
import warnings
from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass
from pdb import set_trace as st
from pprint import pprint
from typing import List, Optional, Tuple, Union

import einx
import torch
import torch.nn as nn
import torch.nn.functional as F
from beartype import beartype
from einops import rearrange, reduce
from t5_pretrainer.mixlora import (
    Experts,
    Lora,
    MixLoraDSI_naive_expand,
    MixloraT5BaseModelOutput,
    TopNGating_NaiveExpand,
    cumsum_exclusive,
    default,
    gumbel_noise,
    safe_one_hot,
)
from t5_pretrainer.mixlora_config import MixLoraConfig
from t5_pretrainer.utils.utils import get_params_info
from torch import einsum
from torch.nn import CrossEntropyLoss, Module, ModuleList, Parameter
from tqdm import tqdm
from transformers import AutoTokenizer
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
    replace_return_docstrings,
)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map

torch.set_printoptions(sci_mode=False)
MIN_EXPERT_CAPACITY = 8

######### MixLoRA #########
MixtureOfExpertsReturn = namedtuple(
    "MixtureOfExpertsReturn",
    [
        "outputs",
        "total_aux_loss",
        "balance_loss",
        "router_z_loss",
        "router_logits",
        "energy_scores",
        "gate_indices",
    ],
)


def get_energy_score(logits, temperature=1.0):
    return -temperature * torch.logsumexp(logits / temperature, dim=-1)


def dist_logsumexp(x: torch.Tensor, dim: int, keepdim: bool = False) -> torch.Tensor:
    # Calculate numerically stable distributed logsumexp
    xmax = x.max(dim=dim, keepdim=True).values
    torch.distributed.all_reduce(xmax, op=torch.distributed.ReduceOp.MAX)

    xe = (x - xmax).exp().sum(dim=dim, keepdim=True)
    torch.distributed.all_reduce(xe, op=torch.distributed.ReduceOp.SUM)

    res = xmax + xe.log()
    if not keepdim:
        res = res.squeeze(dim)

    return res


def log_mean(x: torch.Tensor, dim: int = 0):
    return x.logsumexp(dim) - math.log(x.shape[dim])


def entropy_l(l: torch.Tensor) -> torch.Tensor:
    return -(l * l.exp()).sum(-1)


@dataclass
class MixloraT5ForSemanticOutput(ModelOutput):
    semantic_output: torch.FloatTensor = None
    logits: torch.FloatTensor = None
    router_logits: torch.FloatTensor = None
    balance_loss: torch.FloatTensor = None
    router_z_loss: torch.FloatTensor = None
    total_aux_loss: torch.FloatTensor = None
    energy_scores: torch.FloatTensor = None
    gate_indices: torch.LongTensor = None
    all_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


@dataclass
class MixloraT5BaseModelOutputWithPastAndCrossAttentions(
    BaseModelOutputWithPastAndCrossAttentions
):
    total_aux_loss: Optional[torch.FloatTensor] = None
    balance_loss: Optional[torch.FloatTensor] = None
    router_z_loss: Optional[torch.FloatTensor] = None
    all_router_logits: Optional[List[torch.FloatTensor]] = None
    all_energy_scores: Optional[List[torch.FloatTensor]] = None
    all_gate_indices: Optional[List[torch.LongTensor]] = None


class TopNGating_Varigrow(TopNGating_NaiveExpand):

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
        super().__init__(
            dim,
            num_gates,
            eps,
            top_n,
            threshold_train,
            threshold_eval,
            capacity_factor_train,
            capacity_factor_eval,
            straight_through_dispatch_tensor,
            differentiable_topk,
            differentiable_topk_fused,
            mixlora_config,
            name,
        )
        # For Varigrow
        self.energy_score_temperature = self.mixlora_config.energy_score_temperature
        
        self.to_update = None
        self.before_task = False
        self.embedding = torch.zeros(0, 8, 768)
        self.count = 0

    def forward(self, x, noise_gates=False, noise_mult=1.0, idx=None):
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
        # x: bs, seq, dim
        # gate_logits: bs, seq, num_gates
        if self.cosine_classifier:
            self.to_gates.weight.data = F.normalize(
                self.to_gates.weight.data, p=2, dim=-1
            )
            gate_logits = self.to_gates(F.normalize(x, p=2, dim=-1))
        else:
            gate_logits = self.to_gates(x)

        energy_scores = get_energy_score(
            gate_logits, temperature=self.energy_score_temperature
        ).detach()

        maybe_noised_gate_logits = gate_logits

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

        # Slightly modify the evaluation routing policy
        # if self.training:
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
            maybe_noised_gate_logits,
            energy_scores,
            mask.argmax(dim=-1),
        )


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
        router_z_loss_coef=1e-2,  # 1e-3
        experts: Module | None = None,
        straight_through_dispatch_tensor=True,
        differentiable_topk=False,
        differentiable_topk_fused=True,
        is_distributed=None,
        allow_var_seq_len=False,
        name=None,
    ):
        super().__init__()
        self.dim = dim
        self.num_experts = num_experts
        self.mixlora_config = mixlora_config

        # Mixlorat5layerff
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        # Original T5DenseActDense
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]

        self.gate = TopNGating_Varigrow(
            dim,
            num_gates=num_experts,
            top_n=gating_top_n,
            straight_through_dispatch_tensor=straight_through_dispatch_tensor,
            differentiable_topk=differentiable_topk,
            threshold_train=threshold_train,
            threshold_eval=threshold_eval,
            capacity_factor_train=capacity_factor_train,
            capacity_factor_eval=capacity_factor_eval,
            mixlora_config=mixlora_config,
            name=name,
        )
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

        wi_inputs = self.layer_norm(x)
        wi_output = self.wi(wi_inputs)
        wi_output = self.act(wi_output)
        wi_output = self.dropout(wi_output)

        # router
        (
            dispatch_tensor,
            combine_tensor,
            balance_loss,
            router_z_loss,
            router_logits,
            energy_scores,
            gate_indices,
        ) = self.gate(
            wi_inputs, noise_gates=noise_gates, noise_mult=noise_mult, idx=idx
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
        wi_output = wi_output + wi_expert_outputs

        ######### Wo
        wo_output = self.wo(wi_output)

        # dispatch
        wo_expert_inputs = einsum(
            "b n d, b n e c -> b e c d", wi_output, dispatch_tensor
        )

        # feed the expert inputs through the experts.
        wo_expert_outputs = self.wo_experts(wo_expert_inputs)

        # combine
        wo_expert_outputs = einsum(
            "b e c d, b n e c -> b n d", wo_expert_outputs, combine_tensor
        )

        # Residual: W (wo_output) + Delta W (wo_expert_outputs)
        wo_output = wo_output + wo_expert_outputs
        # Final residual
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
            energy_scores,
            gate_indices,
        )

    def grow_expert(self, embedding=None):
        self.gate.grow_expert(embedding)
        self.wi_experts.grow_expert(self.mixlora_config, layer_weight=self.wi)
        self.wo_experts.grow_expert(self.mixlora_config, layer_weight=self.wo)
        self.num_experts += 1


class MixloraT5Block(Module):
    def __init__(
        self,
        config,
        mixlora_config,
        has_relative_attention_bias=False,
        name=None,
        num_experts=8,
    ):
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
                num_experts=mixlora_config.num_experts[layer_id],
                gating_top_n=mixlora_config.top_k,
                balance_loss_coef=mixlora_config.balance_loss_coef,
                router_z_loss_coef=mixlora_config.router_z_loss_coef,
                name=layer_id,
            )
        )

    def grow_expert(self, embedding=None):
        self.layer[-1].grow_expert(embedding)

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
        idx=None,
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
        hidden_states = self.layer[-1](hidden_states, idx=idx)
        (
            hidden_states,
            total_aux_loss,
            balance_loss,
            router_z_loss,
            router_logits,
            energy_scores,
            gate_indices,
        ) = hidden_states

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
            energy_scores,
            gate_indices,
        )  # hidden-states, present_key_value_states, (self-attention position bias), (self-attention weights), (cross-attention position bias), (cross-attention weights)


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
                        has_relative_attention_bias=bool(i == 0),
                        mixlora_config=mixlora_config,
                        name=f"{name_prefix}{i}",
                        num_experts=mixlora_config.num_experts[i],
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
                self.block[layer_id].grow_expert(
                    embedding
                )  # Grow 1 expert for the MixloraT5Block
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
        idx=None,
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
        all_energy_scores = {}
        all_gate_indices = {}
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
                (
                    layer_outputs,
                    _total_aux_loss,
                    _balance_loss,
                    _router_zloss,
                    router_logits,
                    energy_scores,
                    gate_indices,
                ) = layer_outputs
                total_aux_loss += _total_aux_loss
                balance_loss += _balance_loss
                router_z_loss += _router_zloss
                all_router_logits[i] = router_logits
                all_energy_scores[i] = energy_scores
                all_gate_indices[i] = gate_indices
            else:
                all_router_logits[i] = None
                all_energy_scores[i] = None
                all_gate_indices[i] = None

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
            all_energy_scores=all_energy_scores,
            all_gate_indices=all_gate_indices,
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
        self.decoder_base_novelty_mean = None
        self.decoder_base_novelty_var = None
        self.decoder_novelty_means = None
        self.decoder_novelty_vars = None

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
        output_hidden_states: Optional[bool] = True,# Optional[bool] = None,
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
            encoder_outputs = MixloraT5BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # if isinstance(
        #     encoder_outputs, MixloraT5BaseModelOutputWithPastAndCrossAttentions
        # ):
        #     (
        #         encoder_total_aux_loss,
        #         encoder_balance_loss,
        #         encoder_router_z_loss,
        #         all_encoder_router_logits,
        #         all_encoder_energy_scores,
        #         all_encoder_gate_indices,
        #     ) = (
        #         encoder_outputs["total_aux_loss"],
        #         encoder_outputs["balance_loss"],
        #         encoder_outputs["router_z_loss"],
        #         encoder_outputs["all_router_logits"],
        #         encoder_outputs["all_energy_scores"],
        #         encoder_outputs["all_gate_indices"],
        #     )
        # else:
        (
            encoder_total_aux_loss,
            encoder_balance_loss,
            encoder_router_z_loss,
        ) = (
            torch.tensor(0.0),
            torch.tensor(0.0),
            torch.tensor(0.0),
        )
        all_encoder_router_logits = {}
        all_encoder_energy_scores = {}
        all_encoder_gate_indices = {}
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
            idx=idx,
        )

        # router logits shape: (sequence_length, num_adapters)
        # expect all_router logits: list of 12 * 2 = 24 layers router logits
        # Thus: all_encoder_router_logits = list of len 12 (layers) with each element is [bs*encode_sequence_length, num_adapters]
        # Thus: all_decoder_router_logits = list of len 12 (layers) with each element is [bs*decode_sequence_length, num_adapters]

        if isinstance(
            decoder_outputs, MixloraT5BaseModelOutputWithPastAndCrossAttentions
        ):
            (
                decoder_total_aux_loss,
                decoder_balance_loss,
                decoder_router_z_loss,
                all_decoder_router_logits,
                all_decoder_energy_scores,
                all_decoder_gate_indices,
            ) = (
                decoder_outputs["total_aux_loss"],
                decoder_outputs["balance_loss"],
                decoder_outputs["router_z_loss"],
                decoder_outputs["all_router_logits"],
                decoder_outputs["all_energy_scores"],
                decoder_outputs["all_gate_indices"],
            )
        else:
            (
                decoder_total_aux_loss,
                decoder_balance_loss,
                decoder_router_z_loss,
            ) = (
                torch.tensor(0.0),
                torch.tensor(0.0),
                torch.tensor(0.0),
            )
            all_decoder_router_logits = {}
            all_decoder_energy_scores = {}
            all_decoder_gate_indices = {}

        all_router_logits = {
            "encoder": all_encoder_router_logits,
            "decoder": all_decoder_router_logits,
        }
        all_gate_indices = {
            "encoder": all_encoder_gate_indices,
            "decoder": all_decoder_gate_indices,
        }
        all_energy_scores = {
            "encoder": all_encoder_energy_scores,
            "decoder": all_decoder_energy_scores,
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
                energy_scores=all_energy_scores,
                gate_indices=all_gate_indices,
                all_hidden_states=decoder_outputs.hidden_states,
            )
        else:
            return MixloraT5ForSemanticOutput(
                semantic_output=sequence_output  # [bz smtid_length, d_model]
            )


class BufferDict(nn.Module):
    def __init__(self, input_dict):
        super().__init__()
        for k, v in input_dict.items():
            self.register_buffer(str(k), v)


##### VARIGROW #####
class MixLoraDSI_Varigrow(MixLoraDSI_naive_expand):

    def __init__(
        self,
        model_name_or_path,
        mixlora_config,
        base_model_cls=MixloraT5ForSemanticGeneration,
    ):
        super().__init__(model_name_or_path, mixlora_config, base_model_cls)
        self.novelty_result = None
        self.taskid = -1

        # Varigrow
        self.initialize_novelty_stats()

        checkpoint = self.get_model_checkpoint(model_name_or_path)

        self.sanity_check(mixlora_config, checkpoint)

    def sanity_check(self, mixlora_config, checkpoint):
        if mixlora_config.encoder:
            encoder_stack = getattr(self.base_model, "encoder")
            for i_block, block in enumerate(encoder_stack.block):
                moe_layer = getattr(block, "layer")[1]
                if (
                    isinstance(moe_layer, MoE)
                    and f"encoder.block.{i_block}.layer.1.DenseReluDense.wi.weight"
                    in checkpoint.keys()
                ):
                    wi_weight = checkpoint[
                        f"encoder.block.{i_block}.layer.1.DenseReluDense.wi.weight"
                    ]
                    wo_weight = checkpoint[
                        f"encoder.block.{i_block}.layer.1.DenseReluDense.wo.weight"
                    ]
                    moe_layer.wi.weight.data = wi_weight
                    moe_layer.wo.weight.data = wo_weight

                    assert moe_layer.wi.weight.sum() == wi_weight.sum()
                    print(
                        f"Initialized encoder block {i_block} moe layer with wi,wo weights"
                    )

        if mixlora_config.decoder:
            decoder_stack = getattr(self.base_model, "decoder")
            for i_block, block in enumerate(decoder_stack.block):
                moe_layer = getattr(block, "layer")[2]
                if (
                    isinstance(moe_layer, MoE)
                    and f"decoder.block.{i_block}.layer.2.DenseReluDense.wi.weight"
                    in checkpoint.keys()
                ):
                    wi_weight = checkpoint[
                        f"decoder.block.{i_block}.layer.2.DenseReluDense.wi.weight"
                    ]
                    wo_weight = checkpoint[
                        f"decoder.block.{i_block}.layer.2.DenseReluDense.wo.weight"
                    ]
                    moe_layer.wi.weight.data = wi_weight
                    moe_layer.wo.weight.data = wo_weight
                    assert moe_layer.wi.weight.sum() == wi_weight.sum()
                    print(f"Loaded weights for LoRAs in {i_block} MoE layer", end="\t")

        missing, _ = self.base_model.load_state_dict(checkpoint, strict=False)
        if len(missing) == 0:
            assert torch.allclose(
                checkpoint["decoder_novelty_means.layer2_expert1"],
                self.base_model.decoder_novelty_means.state_dict()["layer2_expert1"],
            )
            print("Loaded novelty scores successfully")

    def initialize_novelty_stats(self):
        novelty_means = {
            "decoder": {
                f"layer{k}_expert{v}": torch.zeros(
                    8, device="cuda", requires_grad=False
                )
                for k, v_ in self.mixlora_config.num_experts.items()
                for v in range(v_)
                if self.mixlora_config.decoder_target_layers[k]
            }
        }

        novelty_vars = {
            "decoder": {
                f"layer{k}_expert{v}": torch.zeros(
                    8, device="cuda", requires_grad=False
                )
                for k, v_ in self.mixlora_config.num_experts.items()
                for v in range(v_)
                if self.mixlora_config.decoder_target_layers[k]
            }
        }

        base_novelty_mean = {
            "decoder": {
                f"layer{k}_expert{v}": torch.zeros(
                    8, device="cuda", requires_grad=False
                )
                for k, v_ in self.mixlora_config.num_experts.items()
                for v in range(2)
                if self.mixlora_config.decoder_target_layers[k]
            }
        }
        base_novelty_var = {
            "decoder": {
                f"layer{k}_expert{v}": torch.zeros(
                    8, device="cuda", requires_grad=False
                )
                for k, v_ in self.mixlora_config.num_experts.items()
                for v in range(2)
                if self.mixlora_config.decoder_target_layers[k]
            }
        }

        self.base_model.decoder_base_novelty_mean = BufferDict(
            base_novelty_mean["decoder"]
        )
        self.base_model.decoder_base_novelty_var = BufferDict(
            base_novelty_var["decoder"]
        )
        self.base_model.decoder_novelty_means = BufferDict(novelty_means["decoder"])
        self.base_model.decoder_novelty_vars = BufferDict(novelty_vars["decoder"])

    def forward(self, **inputs):
        
        if self.previous_checkpoint is not None:
            with torch.no_grad():
                self.base_model.shared.weight[:32100].copy_(self.previous_checkpoint.base_model.shared.weight.detach()[:32100])
        
        output, logits = self._minimal_forward(inputs)

        if self.training:
            rank_loss, total_aux_loss, balance_loss, router_z_loss = self._minimal_calc_loss(inputs, output, logits)

            kl_loss = self._calc_kl_loss(inputs, logits)

            cosine_sim_loss = self._calc_cosine_sim_loss(output, logits)

            router_contrastive_loss = self._calc_router_contrastive_loss(output, logits)

            all_router_logits = output.router_logits

            self.update_exponential_moving_average(
                self.base_model.decoder_novelty_means,
                self.base_model.decoder_novelty_vars,
                output,
            )

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

    def update_base_novelty_stats(self):

        for k, v in self.base_model.decoder_novelty_means.state_dict().items():
            if isinstance(v, torch.Tensor):
                self.base_model.decoder_base_novelty_mean.state_dict()[str(k)].copy_(
                    v.clone().detach().to(self.device)
                )

        for k, v in self.base_model.decoder_novelty_vars.state_dict().items():
            if isinstance(v, torch.Tensor):
                self.base_model.decoder_base_novelty_var.state_dict()[str(k)].copy_(
                    v.clone().detach().to(self.device)
                )

    def _before_task(self, train_loader=None, taskid=-1, freeze_vocab=False):
        self._freeze_base_model(freeze_vocab=freeze_vocab)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.base_model.to(self.device)
        self.base_model.eval()

        # Reset the layer-wise gate embeddings to zero
        for layer in range(12):
            if isinstance(self.base_model.decoder.block[layer].layer[-1], MoE):
                self.base_model.decoder.block[layer].layer[-1].gate.embedding.copy_(
                    torch.zeros(8, 768)
                )
                self.base_model.decoder.block[layer].layer[-1].gate.before_task = True

        assert self.base_model.decoder_novelty_means, self.base_model.decoder_novelty_vars
        novelty_threshold = self.base_model.decoder_novelty_means.state_dict()
        novelty_var = self.base_model.decoder_novelty_vars.state_dict()

        # Create a mask to identify OOD tokens and prepare the threshold for novelty detection
        ood_token_mask = {}
        two_std_novelty_threshold = {}
        for k, v in self.mixlora_config.num_experts.items():
            if not self.mixlora_config.decoder_target_layers[k]:
                continue

            layer = k
            num_experts = v
            two_std_novelty_threshold[layer] = torch.stack(
                [
                    novelty_threshold[f"layer{layer}_expert{expert}"]
                    + self.mixlora_config.ood_sigma_threshold*torch.sqrt(novelty_var[f"layer{layer}_expert{expert}"]) 
                    for expert in range(num_experts) 
                ]
            )
            ood_token_mask[k] = []

        # Adding the two std threshold as attribute for training use
        self.two_std_novelty_threshold = two_std_novelty_threshold

        with torch.no_grad():
            # Loop over the training data to calculate the novelty statistics
            n = 0.0
            num_ood_tokens = {
                0: torch.zeros(8),
                1: torch.zeros(8),
                2: torch.zeros(8),
                3: torch.zeros(8),
                4: torch.zeros(8),
            }
            for ix, inputs in tqdm(enumerate(train_loader), total=len(train_loader)):

                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                outputs = self.base_model(**inputs["tokenized_query"])
                energy_scores = outputs.energy_scores["decoder"]

                for k, v in energy_scores.items():
                    if v is not None:
                        ood_threshold = two_std_novelty_threshold[k]
                        is_ood_token = torch.all(v.unsqueeze(1) > ood_threshold, dim=1)
                        num_ood_tokens[k] += is_ood_token.sum(dim=0).cpu()
                        ood_token_mask[k].append(is_ood_token)
                n += len(inputs["labels"])
            
            print("num_ood_tokens", num_ood_tokens)
            print(n)
            
            for k, v in ood_token_mask.items():
                ood_token_mask[k] = torch.concat(v, dim=0)

            for layer in range(12):
                if isinstance(self.base_model.decoder.block[layer].layer[-1], MoE):
                    self.base_model.decoder.block[layer].layer[-1].gate.before_task = False

            self.mixlora_config.novelty_result = {}
            is_novel = {}
            for k, v in ood_token_mask.items():
                not_ood_samples = torch.all(v == False, dim=1)  # Check if all experts think all tokens of a sample are ID -> True if ID, False if OOD
                ood_samples = torch.where(not_ood_samples == False)[0] # Get the indices of OOD samples
                ood_samples_percentage = round(len(ood_samples) / n * 100, 4)

                token_wise_ood = torch.round(torch.sum(v, dim=0) / len(v) * 100, decimals=4)
                is_novel[k] = (ood_samples_percentage >= self.mixlora_config.layerwise_novelty_threshold)

                self.mixlora_config.novelty_result[k] = (ood_samples_percentage, token_wise_ood)

                print("Layer", k+1)
                print("two_std_novelty_threshold:", two_std_novelty_threshold[k])
                print(f"Number of OOD samples: {ood_samples_percentage:.2f}%")
                print(f"Token-wise statistics: 1: {token_wise_ood[0]:.2f}, "
                f"2: {token_wise_ood[1]:.2f}, "
                f"3: {token_wise_ood[2]:.2f}, "
                f"4: {token_wise_ood[3]:.2f}, "
                f"5: {token_wise_ood[4]:.2f}, "
                f"6: {token_wise_ood[5]:.2f}, "
                f"7: {token_wise_ood[6]:.2f}, "
                f"8: {token_wise_ood[7]:.2f}")
                print("Novelty decision:", is_novel[k])
                print("\n")

            self.novelty_result = is_novel

        grad_mask_dict = self._freeze_expand_with_novelty(
            ood_token_mask, taskid=taskid, freeze_vocab=freeze_vocab
        )

        print("\n##### Params info after freezing with energy-based OOD score #####")
        get_params_info(self.base_model)

        return self.mixlora_config.novelty_result, grad_mask_dict

    def grow_novelty_stats(self, layer_idx):
        current_decoder_novelty_means = copy.deepcopy(
            self.base_model.decoder_novelty_means.state_dict()
        )
        current_decoder_novelty_vars = copy.deepcopy(
            self.base_model.decoder_novelty_vars.state_dict()
        )

        new_expert_id = self.mixlora_config.num_experts[layer_idx] - 1  # Index based 0
        new_num_experts = self.mixlora_config.num_experts[layer_idx]

        # Re-calibrate the running novelty mean of the expanded layer
        for i in range(2):
            previous_base_novelty_mean = self.base_model.decoder_base_novelty_mean.state_dict()[f"layer{layer_idx}_expert{i}"].clone()
            self.base_model.decoder_base_novelty_mean.state_dict()[f"layer{layer_idx}_expert{i}"].copy_(
                previous_base_novelty_mean + math.log(new_num_experts-1) - math.
                log(new_num_experts)
            )

        # Re-calibrate the base novelty mean of the expanded layer
        for i in range(new_num_experts - 1):
            previous_novelty_mean = current_decoder_novelty_means[f"layer{layer_idx}_expert{i}"]

            # Re-calibrate the novelty mean of the previous experts
            current_decoder_novelty_means[f"layer{layer_idx}_expert{i}"] = previous_novelty_mean + math.log(new_num_experts-1) - math.log(new_num_experts)

        current_decoder_novelty_means[f"layer{layer_idx}_expert{new_expert_id}"] = (
            # Initialize the new expert with the base expert
            # Why expert 0? Since both expert 0 and 1 are the same, we can initialize the new expert with expert 0, because of the top-2 nature of the router and we start with 2 experts
            self.base_model.decoder_base_novelty_mean.state_dict()[
                f"layer{layer_idx}_expert0"
            ].clone()
        )
        current_decoder_novelty_vars[f"layer{layer_idx}_expert{new_expert_id}"] = (
            self.base_model.decoder_base_novelty_var.state_dict()[f"layer{layer_idx}_expert0"].clone()
        )
        self.base_model.decoder_novelty_means = BufferDict(
            current_decoder_novelty_means
        )
        self.base_model.decoder_novelty_vars = BufferDict(current_decoder_novelty_vars)

    def update_exponential_moving_average(
        self, running_mean, running_var, curr_samples, momentum=0.1
    ):
        energy_scores = curr_samples["energy_scores"]["decoder"]
        gate_indices = curr_samples["gate_indices"]["decoder"]
        has_novelty_result = self.mixlora_config.novelty_result is not None

        with torch.no_grad():
            for layer in range(12):
                if energy_scores[layer] is None:
                    continue

                if has_novelty_result and self.mixlora_config.novelty_result[layer][0] < self.mixlora_config.layerwise_novelty_threshold:
                    continue

                layer_energy_scores = energy_scores[layer]  # bs, seq_len
                indices = gate_indices[layer]  # top-k, bs, seq_len

                for expert_id in range(self.mixlora_config.num_experts[layer]):
                    self.update_statistics(running_mean, running_var, momentum, layer, layer_energy_scores, indices, expert_id)

    def update_statistics(self, running_mean, running_var, momentum, layer, layer_energy_scores, indices, expert_id):
        scores = torch.zeros_like(layer_energy_scores, device=self.device)
        var = torch.zeros(8, device=self.device)
        n = torch.ones(8, device=self.device) # Counting OOD token in each RQ position; Start with 1 to avoid division by zero, trade-off between accuracy and stability
        
        running_mean_vals = running_mean.state_dict()[f"layer{layer}_expert{expert_id}"].clone()
        running_var_vals = running_var.state_dict()[f"layer{layer}_expert{expert_id}"].clone()

        for top_k in range(indices.shape[0]):
            expert_mask = indices[top_k] == expert_id
            n += expert_mask.sum(dim=0)
            scores += layer_energy_scores * expert_mask

        mean = scores.sum(dim=0) / n # Making the mean more accurate
        for ix, m in enumerate(mean):
            if n[ix] <= 1:
                continue
            _score = scores[:, ix][torch.nonzero(scores[:, ix])]
            var[ix] = (_score - m).pow(2).sum(dim=0) / (n[ix]-1)
            
            running_mean_vals[ix] = momentum * m + (1 - momentum) * running_mean_vals[ix] 
            
            running_var_vals[ix] = momentum * var[ix] + (1 - momentum) * running_var_vals[ix] # Removing the Bessel correction (might be incorrect) since tensor.var() already has such correction

        running_mean.state_dict()[f"layer{layer}_expert{expert_id}"].copy_(running_mean_vals)
        running_var.state_dict()[f"layer{layer}_expert{expert_id}"].copy_(running_var_vals)

    def _freeze_lora_layer(self, layer_idx, lora_idx_list):
        """
        layer_idx: int: the layer index to freeze
        loar_idx_list: list of integers: the expert indices to stay trainable
        """

        num_experts = self.mixlora_config.num_experts[layer_idx]
        grad_mask = torch.zeros(num_experts, device=self.device)

        for name, param in self.base_model.named_parameters():

            # If the router does not expand, simply freeze it
            if (
                "gate" in name
                and int(name.split(".")[2]) == layer_idx
                and self.mixlora_config.novelty_result[layer_idx][0] < self.mixlora_config.layerwise_novelty_threshold
            ):
                param.requires_grad = False

            if not ("decoder" in name and "expert" in name):
                continue
            layer = int(name.split(".")[2])
            expert_idx = int(name.split(".")[-3])

            if layer == layer_idx:
                # The expanded router and expert are trainable
                if expert_idx in lora_idx_list and self.mixlora_config.novelty_result[layer_idx][0] >= self.mixlora_config.layerwise_novelty_threshold:
                    param.requires_grad = True
                    grad_mask[expert_idx] = 1.0  # Activate the expert router  weight
                    print(f"Layer {layer_idx} Expert {expert_idx} is trainable {name}")
                elif (
                    expert_idx in lora_idx_list
                    and not self.mixlora_config.update_only_expanded
                ):
                    # Allow selected non-expanded experts to be updated
                    param.requires_grad = True
                    print(f"Layer {layer_idx} Expert {expert_idx} is trainable {name}")
                else:
                    param.requires_grad = False
        return grad_mask

    def _freeze_expand_with_novelty(
        self, ood_token_mask, taskid=-1, freeze_vocab=False
    ):
        grad_mask_dict = {}
        for k, v in self.mixlora_config.novelty_result.items():
            # The first element of the tuple is the percentage of OOD samples, while the second element is the token-wise OOD percentage

            ood_percentage = v[0]
            if ood_percentage > self.mixlora_config.layerwise_novelty_threshold:
                print(f"\nLayer {k+1} needs to be expanded")
                # Get the avreage embeddings of the OOD tokens
                # if self.mixlora_config.average_ood_embedding:
                #     embedding = self.base_model.decoder.block[k].layer[-1].gate.embedding * ood_token_mask[k].long().cpu().unsqueeze(-1)
                #     embedding = torch.mean(embedding.reshape(-1, 768), dim=0)
                # else:
                embedding = None

                # Extend the router initialized with the average embedding of the OOD tokens
                self.grow_expert([k], embedding=embedding)
                self.grow_novelty_stats(layer_idx=k)
                grad_mask_dict[k] = self._freeze_lora_layer(
                    layer_idx=k,
                    lora_idx_list=[self.mixlora_config.num_experts[k] - 1],
                ).unsqueeze(-1)

                assert torch.nonzero(grad_mask_dict[k], as_tuple=True)[0].cpu().tolist() == [self.mixlora_config.num_experts[k] - 1]
            else:
                print(f"\nLayer {k+1} is not expanded")
                # If the layer is not expanded, simply freeze the router and the experts
                grad_mask_dict[k] = self._freeze_lora_layer(
                    layer_idx=k, lora_idx_list=[], # 
                )
                # The grad mask is simply zeros, meaning that the router which not expand are not updated
                grad_mask_dict[k] = torch.zeros(
                    self.mixlora_config.num_experts[k], device=self.device
                ).unsqueeze(-1)
        return grad_mask_dict


class MixLoraDSI_Varigrow_DirectLngKnpMarginMSE(MixLoraDSI_Varigrow):
    def __init__(self, model_name_or_path, model_args=None):
        super().__init__(model_name_or_path, model_args)
        self.loss_fn = torch.nn.MSELoss()
        
    
    def decode(self, text_encodings):
        """
        Args:
            text_encodings: [bz, smtid_length]
        Returns:
            text_embeds: [bz, smtid_length, d_model]
        """
        text_embeds = torch.nn.functional.embedding(
            text_encodings, self.base_model.lm_head.weight
        )
        return text_embeds

        
    
    def _minimal_forward(self, inputs):
        output = self.base_model(**inputs["tokenized_query"])

        logits = output.logits  # [bz, smtid_length, vocab_size]
        if self.rq_specific_mask_head:
            for i, mask in enumerate(self.masks):
                logits[:, i, :] = logits[:, i, :].index_fill_(
                    dim=-1, index=mask, value=float("-inf")
                )
                
        return output,logits


    def forward(self, **inputs):

        if self.training:

            pos_query_embeds = self.base_model(**inputs["pos_tokenized_query"]).semantic_output #[bz, smtid_length, d_model]
            neg_query_embeds = self.base_model(**inputs["neg_tokenized_query"]).semantic_output #[bz, smtid_length, d_model]
            pos_doc_embeds = self.decode(inputs["pos_doc_encoding"]) #[bz, smtid_length, d_model]
            neg_doc_embeds = self.decode(inputs["neg_doc_encoding"]) #[bz, smtid_length, d_model]

            assert pos_doc_embeds.size(1) == 8, pos_doc_embeds.size()

            # rank_4
            early_pos_score = (pos_query_embeds[:, :4, :].clone() * pos_doc_embeds[:, :4, :].clone()).sum(-1).sum(-1)
            early_neg_score = (neg_query_embeds[:, :4, :].clone() * neg_doc_embeds[:, :4, :].clone()).sum(-1).sum(-1)
            early_student_margin = early_pos_score - early_neg_score
            early_teacher_margin = (inputs["teacher_pos_scores"].clone() - inputs["teacher_neg_scores"].clone()) * 0.5
            rank_4_loss = self.loss_fn(early_student_margin, early_teacher_margin)

            # rank
            student_margin = (pos_query_embeds * pos_doc_embeds).sum(-1).sum(-1) - (neg_query_embeds * neg_doc_embeds).sum(-1).sum(-1)
            teacher_margin = inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"]
            rank_loss = self.loss_fn(student_margin, teacher_margin)

            return {
                "loss": rank_loss + rank_4_loss,
                "rank_loss": rank_loss,
                "rank_4_loss": rank_4_loss,
            }