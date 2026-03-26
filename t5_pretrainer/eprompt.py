import torch
import torch.nn as nn

topic_embeddings_dict = {
    # "nq320k_t5": "./mixlora_dsi/nq320k/bertopic/t5_nq320k_d0_topic_embeddings_310.bin",
    "nq320k_t5": "./mixloradsi/nq320k/bertopic/topic_embeddings.bin",
    "msmarco_t5": "./mixlora_dsi/nq320k/bertopic/t5_msmarco_d0_topic_embeddings_6611.bin",
}

topic_pool_size_dict = {
    "nq320k_t5": 283, #310,
    "msmarco_t5": 6611,
}


class EPromptWithTopicModelling(nn.Module):  # Cosine similarity
    def __init__(
        self,
        embed_dim=768,
        prompt_init="uniform",
        num_layers=1,
        top_k=1,
        num_heads=12,
        prompt_length=10,
        prompt_allocation=10,
        contrastive_loss=False,
        base_data_dir="nq320k",
    ):
        super().__init__()
        dataset_name = "nq320k" if "nq320k" in base_data_dir else "msmarco"
        model_name = "t5" # "sbert" if "sbert" in model_encoder else "bert"
        name = dataset_name + "_" + model_name
        topic_embeddings = torch.load(topic_embeddings_dict[name])["topic_embeddings"]
        self.pool_size = topic_pool_size_dict[name]

        self.top_k = top_k
        self.prompt_key = nn.Parameter(topic_embeddings)
        self.prompt_allocation = prompt_allocation
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.contrastive_loss = contrastive_loss
        self.length = prompt_length

        assert embed_dim % self.num_heads == 0
        prompt_pool_shape = (
            self.num_layers,
            2,
            self.pool_size,
            self.length,
            self.num_heads,
            embed_dim // self.num_heads,
        )
        if prompt_init == "zero":
            self.prompt = nn.Parameter(torch.zeros(prompt_pool_shape))
        elif prompt_init == "uniform":
            self.prompt = nn.Parameter(
                torch.randn(prompt_pool_shape)
            )  # num_layers, 2, pool_size, length, num_heads, embed_dim // num_heads
            nn.init.uniform_(self.prompt, -1, 1)

    def l2_normalize(self, x, dim=None, epsilon=1e-12):
        """Normalizes a given vector or matrix."""
        square_sum = torch.sum(x**2, dim=dim, keepdim=True)
        x_inv_norm = torch.rsqrt(
            torch.maximum(square_sum, torch.tensor(epsilon, device=x.device))
        )
        return x * x_inv_norm

    def forward(
        self,
        cls_features=None,
    ):
        out = dict()
        x_embed_mean = cls_features

        prompt_pool = self.prompt
        prompt_key = self.prompt_key

        prompt_key_norm = self.l2_normalize(prompt_key, dim=-1)  # Pool_size, C
        x_embed_norm = self.l2_normalize(x_embed_mean, dim=-1)  # B, C

        similarity = torch.matmul(
            prompt_key_norm, x_embed_norm.t()
        )  # pool_size, B or Pool_size, #class, B
        similarity = similarity.t()  # B, pool_size

        if len(similarity.shape) == 1:
            similarity = similarity.unsqueeze(1)

        (similarity_top_k, idx) = torch.topk(
            similarity, k=self.top_k, dim=1
        )  # B, top_k
        out["similarity"] = similarity_top_k
        out["idx"] = idx

        batched_prompt_raw = prompt_pool[:, :, idx]  # num_layers, B, top_k, length, C
        (
            num_layers,
            dual,
            batch_size,
            top_k,
            length,
            num_heads,
            heads_embed_dim,
        ) = batched_prompt_raw.shape
        batched_prompt = batched_prompt_raw.reshape(
            num_layers,
            batch_size,
            dual,
            top_k * length,
            num_heads,
            heads_embed_dim,
        )
        x_embed_norm = x_embed_norm.unsqueeze(1)  # B, 1, C
        out["batched_prompt"] = batched_prompt

        return out
