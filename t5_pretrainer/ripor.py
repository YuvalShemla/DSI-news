import os
from dataclasses import dataclass, field
from pdb import set_trace as st
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    BertConfig,
    BertModel,
)
from transformers.file_utils import ModelOutput
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.models.t5.modeling_t5 import T5Config  # type: ignore
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration, T5Model


class CrossEncoder(nn.Module):
    def __init__(self, model_name_or_path):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name_or_path)
        config.num_labels = 1
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, config=config
        )

        self.cls_loss = nn.BCEWithLogitsLoss()
        self.model_args = None  # incompatible with previous models

    def forward(self, **inputs):
        logits = self.model(**inputs["qd_kwargs"]).logits.view(-1)
        if "labels" in inputs:
            loss = self.cls_loss(logits, inputs["labels"])
            return {"cls": loss}
        else:
            return logits

    def rerank_forward(self, qd_kwargs):
        out = {}
        scores = self.model(**qd_kwargs).logits.view(-1)
        out.update({"scores": scores})

        return out

    @classmethod
    def from_pretrained(cls, model_name_or_path):
        return cls(model_name_or_path)

    def save_pretrained(self, save_dir):
        self.model.save_pretrained(save_dir)


class BertDenseEncoder(torch.nn.Module):
    def __init__(self, model_name_or_path, model_args=None):
        super().__init__()

        config = BertConfig.from_pretrained("bert-base-uncased")
        if model_args is not None:
            config.num_decoder_layers = model_args.num_decoder_layers
        self.base_model = BertModel.from_pretrained(model_name_or_path, config=config)
        self.model_args = model_args

        self.rank_loss_fn = torch.nn.MSELoss()

    def forward(self, **inputs):
        raise NotImplementedError

    def encode(self, **inputs):
        hidden_state = self.base_model(**inputs, return_dict=True).last_hidden_state[:,0,:]
        # assert hidden_state.dim() == 3 and hidden_state.size(1) == 1

        return hidden_state.squeeze(1)

    def doc_encode(self, **inputs):
        return self.encode(**inputs)

    def query_encode(self, **inputs):
        return self.encode(**inputs)

    @classmethod
    def from_pretrained(cls, model_name_or_path, model_args=None):
        return cls(model_name_or_path, model_args)

    def save_pretrained(self, save_dir):
        self.base_model.save_pretrained(save_dir)


class T5DenseEncoder(torch.nn.Module):
    def __init__(self, model_name_or_path, model_args=None):
        super().__init__()

        config = T5Config.from_pretrained(model_name_or_path)
        if model_args is not None:
            config.num_decoder_layers = model_args.num_decoder_layers
        self.base_model = T5Model.from_pretrained(model_name_or_path, config=config)
        self.model_args = model_args

        self.rank_loss_fn = torch.nn.MSELoss()

    def forward(self, **inputs):
        raise NotImplementedError

    def encode(self, **inputs):
        hidden_state = self.base_model(**inputs, return_dict=True).last_hidden_state
        assert hidden_state.dim() == 3 and hidden_state.size(1) == 1

        return hidden_state.squeeze(1)

    def doc_encode(self, **inputs):
        return self.encode(**inputs)

    def query_encode(self, **inputs):
        return self.encode(**inputs)

    @classmethod
    def from_pretrained(cls, model_name_or_path, model_args=None):
        return cls(model_name_or_path, model_args)

    def save_pretrained(self, save_dir):
        self.base_model.save_pretrained(save_dir)


class T5DenseEncoderForMarginMSE(T5DenseEncoder):
    def __init__(self, model_name_or_path, model_args=None):
        super().__init__(model_name_or_path, model_args)

        self.rank_loss_fn = torch.nn.MSELoss()

    def forward(self, **inputs):
        query_rep = self.encode(**inputs["tokenized_query"])  # [bz, vocab_size]
        pos_doc_rep = self.encode(**inputs["pos_tokenized_doc"])
        neg_doc_rep = self.encode(**inputs["neg_tokenized_doc"])

        student_margin = (query_rep * pos_doc_rep).sum(dim=-1) - (
            query_rep * neg_doc_rep
        ).sum(dim=-1)
        teacher_margin = inputs["teacher_pos_scores"] - inputs["teacher_neg_scores"]

        rank_loss = self.rank_loss_fn(student_margin, teacher_margin)
        return {"rank": rank_loss}


@dataclass
class T5ForSemanticOutput(ModelOutput):
    semantic_output: torch.FloatTensor = None
    logits: torch.FloatTensor = None


class T5ForSemanticGeneration(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
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
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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
            return T5ForSemanticOutput(
                logits=self.lm_head(sequence_output), semantic_output=sequence_output
            )
        else:
            return T5ForSemanticOutput(
                semantic_output=sequence_output  # [bz smtid_length, d_model]
            )


class T5ForPQOnestepDecoding(T5ForConditionalGeneration):
    def __init__(self, config: T5Config):
        super().__init__(config)
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
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
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

        return sequence_output


class Ripor(torch.nn.Module):
    def __init__(self, model_name_or_path, model_args=None):
        super().__init__()

        config = T5Config.from_pretrained(model_name_or_path)
        if model_args is not None:
            config.num_decoder_layers = model_args.num_decoder_layers

        self.base_model = T5ForSemanticGeneration.from_pretrained(
            model_name_or_path, config=config
        )
        self.model_args = model_args
        self.config = config

    def forward(self, **inputs):
        raise NotImplementedError

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

    @classmethod
    def from_pretrained(cls, model_name_or_path, model_args=None):
        return cls(model_name_or_path, model_args)

    def save_pretrained(self, save_dir, safe_serialization=False):
        self.base_model.save_pretrained(save_dir, safe_serialization=safe_serialization)

    def beam_search(self):
        raise NotImplementedError


class RiporForSeq2seq(Ripor):
    _keys_to_ignore_on_save = None
    def __init__(self, model_name_or_path, model_args=None):
        super().__init__(model_name_or_path, model_args)
        self.rank_loss_fn = torch.nn.CrossEntropyLoss()
        self.base_model.return_logits = True
        self.masked_head = True # Masked RQ Specific head

        if self.masked_head:
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

        # rq_reconstruction_loss = torch.tensor(0.0).to(logits.device)
        # rq_centroids_ground_truth = self.base_model.get_input_embeddings()(inputs["labels"]).sum(dim=1)

        # predicted_rq_centroids = []
        # for i in range(8):
        #     predicted_rq_centroids.append(
        #         torch.matmul(
        #             F.softmax(logits[:, i, 32100 + 2048*i:32100 + 2048 * (i+1)], dim=-1),
        #             self.base_model.get_input_embeddings()(torch.arange(32100 + 2048*i, 32100 + 2048 * (i+1), device=logits.device))
        #         )
        #     )

        # predicted_rq_centroids = torch.stack(predicted_rq_centroids, dim=1).sum(dim=1)

        # rq_reconstruction_loss = F.mse_loss(predicted_rq_centroids, rq_centroids_ground_truth, reduction="mean")

        return {
            "loss": rank_loss,
            "rank_loss": rank_loss,
            # "rq_reconstruction_loss": rq_reconstruction_loss,
        }


class RiporForSeq2seqPQOnestepDecoding(Ripor):

    def __init__(
        self,
        model_name_or_path,
        pq_M=24,
        pq_nbits=8,
    ):
        super().__init__(model_name_or_path, model_args=None)
        self.base_model = T5ForPQOnestepDecoding.from_pretrained(model_name_or_path)
        self.pq_M = pq_M
        self.pq_K = 2**pq_nbits
        self.chunk_size = int(self.config.d_model // self.pq_M)
        self.linear_softmax_head = nn.ModuleDict({})

        # Add pq_M softmax linear heads
        for i in range(pq_M):
            self.linear_softmax_head[f"head{i}"] = PQClassficationHead(
                config=self.config,
                pq_M=pq_M,
                pq_K=self.pq_K,
            )

        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.base_model.return_logits = True

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path,
        pq_M=24,
        pq_nbits=8,
    ):
        return cls(
            model_name_or_path,
            pq_M,
            pq_nbits
        )

    def forward(self, **inputs):
        """
        Args:
            tokenized_query: [bz, max_length] + [bz, smtid_length]
            labels: [bz, smtid_length]
        """

        # Evaluation
        if not self.training:
            sequence_output = self.base_model(**inputs)

            all_head_logits = torch.stack(
                [
                    self.linear_softmax_head[f"head{i}"](
                        sequence_output[
                            :, -1, self.chunk_size * i : self.chunk_size * (i + 1)
                        ]  # The -1 actually collapses (squeezes) the single dimension
                    )
                    for i in range(self.pq_M)
                ],
                dim=0,
            )

            return all_head_logits

        sequence_output = self.base_model(
            **inputs["tokenized_query"]
        )

        # Map sequence_output to the linear_softmax_head
        all_head_logits = torch.stack(
            [
                self.linear_softmax_head[f"head{i}"](
                    sequence_output[
                        :, -1, self.chunk_size * i : self.chunk_size * (i + 1)
                    ]  # The -1 actually collapses (squeezes) the single dimension
                )
                for i in range(self.pq_M)
            ],
            dim=0,
        )

        cls_loss = 0.0
        for i in range(self.pq_M):
            cls_loss += self.cls_loss_fn(all_head_logits[i], inputs["smtid"][:, i])
        cls_loss /= self.pq_M

        return {
            "loss": cls_loss,
            "cls_loss": cls_loss,
        }

    def save_pretrained(self, save_dir, safe_serialization=False):
        self.base_model.save_pretrained(
            save_dir,
            safe_serialization=safe_serialization,
        )
        torch.save(self.linear_softmax_head, os.path.join(save_dir, "linear_softmax_head.pth"))


class RiporForDirectLngKnpMarginMSE(Ripor):
    def __init__(self, model_name_or_path, model_args=None):
        super().__init__(model_name_or_path, model_args)
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, **inputs):
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
