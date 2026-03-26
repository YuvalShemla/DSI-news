from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Optional, Union


def default_num_experts():
    return defaultdict(lambda: 2, {str(i): 2 for i in range(12)})

def default_encoder_target_layers():
    return defaultdict(lambda: False, {str(i): False for i in range(12)})

def default_decoder_target_layers():
    return defaultdict(lambda: False, {
        "0": True,
        "1": True,
        "2": True,
        "3": True,
        "4": True,
        "5": False,
        "6": False,
        "7": False,
        "8": False,
        "9": False,
        "10": False,
        "11": False,
    })


def default_attention_target_modules():
    return defaultdict(
        lambda: False,
        {"q": True, "k": False, "v": True, "o": False, "wi": True, "wo": True},
    )


@dataclass
class MixLoraConfig:

    # ------------------------ LoRA config ------------------------
    r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    top_k: int = 2
    router_loss: bool = True
    router_aux_loss_coef: float = 0.01
    encoder: bool = False
    decoder: bool = True # We only add the LoRAs to the decoder
    encoder_attention: bool = False
    decoder_attention: bool = False
    attention_target_modules: Dict[str, bool] = field(
        default_factory=lambda: {
            "q": True, "k": False, "v": True, "o": False, "wi": True, "wo": True
        })
    num_experts: Dict[str, int] = field(
        default_factory=lambda: {str(i): 2 for i in range(12)}
    )
    encoder_target_layers: Dict[str, bool] = field(
        default_factory=lambda: {str(i): False for i in range(12)}
    )
    decoder_target_layers: Dict[str, bool] = field(
        default_factory=lambda: {
            "0": True,
            "1": True,
            "2": True,
            "3": True,
            "4": True,
            "5": False,
            "6": False,
            "7": False,
            "8": False,
            "9": False,
            "10": False,
            "11": False,
        }
    )
    model_description: str = "Empty"

    # Original router config
    no_aux_loss: bool = True  # Remove the original auxiliary loss
    balance_loss_coef: float = 1e-2
    router_z_loss_coef: float = 1e-3

    # ------------------------ MixLoRA-DSI specific ------------------------
    update_only_expanded: bool = True

    # RQ-based docids CL strategies
    rq_specific_mask_head: bool = True
    freeze_vocab: bool = False
    slow_learn_rq: bool = True

    kl_loss: bool = True
    kl_loss_coef: float = 0.1

    # Improved router params
    cosine_classifier: bool = True
    cosine_sim_loss: bool = True
    cosine_sim_loss_coef: float = 1.0 # Lambda_1

    router_contrastive_loss: bool = True
    router_contrastive_loss_temperature: float = 1.0
    router_contrastive_loss_coef: float = 1.0  # Lambda_2

    # OOD params
    energy_score_temperature: float = 1.0
    layerwise_novelty_threshold: float = 20.0
    ood_sigma_threshold: float = 0.0
    novelty_result: Dict[str, Union[int, list]] = None

    # CorpusBrain++
    reduction_factor: int = 96
    leave_out: list[int] = field(default_factory=list)

    # Archived, not in use
    # l2_loss: bool = False
    # l2_loss_coef: float = 0.1
    # calibrate_router_logits: bool = False
    # use_sigmoid: bool = False
    # init_avg_experts: bool = True
    # normalized_router_weight: bool = False
    # average_ood_embedding: bool = False
    # naive_expand_lora: bool = False

    @staticmethod
    def from_config(config: Dict[str, any]) -> "MixLoraConfig":
        base_config = MixLoraConfig()

        base_config.r = config.get("r", base_config.r)
        base_config.lora_alpha = config.get("lora_alpha", base_config.lora_alpha)
        base_config.lora_dropout = config.get("lora_dropout", base_config.lora_dropout)
        base_config.top_k = config.get("top_k", base_config.top_k)
        base_config.router_loss = config.get("router_loss", base_config.router_loss)
        base_config.router_aux_loss_coef = config.get("router_aux_loss_coef", base_config.router_aux_loss_coef)
        base_config.encoder = config.get("encoder", base_config.encoder)
        base_config.decoder = config.get("decoder", base_config.decoder)
        base_config.encoder_attention = config.get("encoder_attention", base_config.encoder_attention)
        base_config.decoder_attention = config.get("decoder_attention", base_config.decoder_attention)
        base_config.attention_target_modules = config.get(
            "attention_target_modules", base_config.attention_target_modules
        )
        base_config.num_experts = config.get("num_experts", base_config.num_experts)
        base_config.encoder_target_layers = config.get("encoder_target_layers", base_config.encoder_target_layers)
        base_config.decoder_target_layers = config.get("decoder_target_layers", base_config.decoder_target_layers)
        base_config.model_description = config.get("model_description", base_config.model_description)

        # Original router config
        base_config.no_aux_loss = config.get("no_aux_loss", base_config.no_aux_loss)
        base_config.balance_loss_coef = config.get("balance_loss_coef", base_config.balance_loss_coef)
        base_config.router_z_loss_coef = config.get("router_z_loss_coef", base_config.router_z_loss_coef)

        # ------------------------ MixLoRA-DSI specific ------------------------
        base_config.update_only_expanded = config.get(
            "update_only_expanded", base_config.update_only_expanded
        )

        # RQ-based docids CL strategies
        base_config.rq_specific_mask_head = config.get("rq_specific_mask_head", base_config.rq_specific_mask_head)
        base_config.freeze_vocab = config.get("freeze_vocab", base_config.freeze_vocab)
        base_config.slow_learn_rq = config.get("slow_learn_rq", base_config.slow_learn_rq)
        base_config.model_description = config.get("model_description", base_config.model_description)

        base_config.kl_loss = config.get("kl_loss", base_config.kl_loss)
        base_config.kl_loss_coef = config.get("kl_loss_coef", base_config.kl_loss_coef)

        # Improved router params
        base_config.cosine_classifier = config.get("cosine_classifier", base_config.cosine_classifier)
        base_config.cosine_sim_loss = config.get("cosine_sim_loss", base_config.cosine_sim_loss)
        base_config.cosine_sim_loss_coef = config.get("cosine_sim_loss_coef", base_config.cosine_sim_loss_coef)

        base_config.router_contrastive_loss = config.get("router_contrastive_loss", base_config.router_contrastive_loss)
        base_config.router_contrastive_loss_temperature = config.get(
            "router_contrastive_loss_temperature", base_config.router_contrastive_loss_temperature
        )
        base_config.router_contrastive_loss_coef = config.get("router_contrastive_loss_coef", base_config.router_contrastive_loss_coef)

        if base_config.encoder_target_layers is not None:
            base_config.encoder_target_layers = {
                int(k): v for k, v in base_config.encoder_target_layers.items()
            }

        if base_config.decoder_target_layers is not None:
            base_config.decoder_target_layers = {
                int(k): v for k, v in base_config.decoder_target_layers.items()
            }

        if base_config.num_experts is not None:
            base_config.num_experts = {
                int(k): v for k, v in base_config.num_experts.items()
            }

        # OOD params
        base_config.energy_score_temperature = config.get("energy_score_temperature", base_config.energy_score_temperature)
        base_config.layerwise_novelty_threshold = config.get("layerwise_novelty_threshold", base_config.layerwise_novelty_threshold)
        base_config.ood_sigma_threshold = config.get("ood_sigma_threshold", base_config.ood_sigma_threshold)

        base_config.novelty_result = config.get("novelty_result", base_config.novelty_result)
        if base_config.novelty_result is not None:
            base_config.novelty_result = {
                int(k): v for k, v in base_config.novelty_result.items()
            }

        # CorpusBrain++
        base_config.reduction_factor = config.get("reduction_factor", base_config.reduction_factor)
        base_config.leave_out = [x for x in config.get("leave_out", base_config.leave_out)]

        return base_config
