from typing import Tuple, Dict, Union

import torch
from torch import nn
from transformers import PreTrainedModel, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput


class EmqaBaseModel(nn.Module):

    def forward(self, question_tokens, question_mask,
                video_features, video_mask,
                answer_tokens, answer_mask,
                batch_sample_ids,
                moment_localization_labels,
                teacher_forcing=True) -> Union[Tuple[Dict[str, torch.Tensor], torch.Tensor],
                                               torch.Tensor]:
        if teacher_forcing:
            return self.teacher_forcing_forward(question_tokens, question_mask,
                                                video_features, video_mask,
                                                answer_tokens, answer_mask,
                                                batch_sample_ids, moment_localization_labels)
        else:
            return self.autoregressive_forward(question_tokens, question_mask,
                                               video_features, video_mask)

    def teacher_forcing_forward(self, question_tokens, question_mask,
                                video_features, video_mask,
                                answer_tokens, answer_mask,
                                batch_sample_ids,
                                moment_localization_labels
                                ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Forward in teacher forcing mode.

        :param question_tokens: B x Q
        :param question_mask: B x Q
        :param video_features: B x N x H
        :param video_mask: B x N x H
        :param answer_tokens: B x A
        :param answer_mask: B x A
        :param batch_sample_ids: List of sample ids for this batch.
        :param moment_localization_labels: B x N, with entries between 0.0 and 1.0
                (smoothed label saying if each frame is part of ground truth moment). Might be None.
        :return: tuple with loss_dict and anwer logits.
                 loss_dict should at least contain "total_loss" Tensor.
        """
        raise NotImplementedError

    def autoregressive_forward(self, question_tokens, question_mask,
                               video_features, video_mask
                               ) -> torch.Tensor:
        """
        Forward in autoregressive mode.

        :param question_tokens: B x Q
        :param question_mask: B x Q
        :param video_features: B x N x H
        :param video_mask: B x N x H
        :return: tensor with answer tokens. B x A
        """
        raise NotImplementedError


class MemoryAugmentedTransformerEmqaModel(EmqaBaseModel):
    def __init__(self,
                 pretrained_enc_dec: str
                 ) -> None:
        super().__init__()
        self.transformer: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(pretrained_enc_dec)

    def teacher_forcing_forward(self, question_tokens, question_mask,
                                video_features, video_mask,
                                answer_tokens, answer_mask,
                                batch_sample_ids,
                                moment_localization_labels) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        context, context_mask, final_memory, mem_mask, enc_add_loss = self.forward_encoders(
            question_tokens, question_mask, video_features, video_mask, moment_localization_labels)
        output = self.transformer(labels=answer_tokens, decoder_attention_mask=answer_mask,
                                  encoder_outputs=(context,), attention_mask=context_mask)
        return {
                   'lm_loss': output.loss,
                   **enc_add_loss,
                   **self.calc_additional_loss(question_tokens, question_mask,
                                               video_features, video_mask,
                                               answer_tokens, answer_mask, batch_sample_ids,
                                               context, context_mask, final_memory, mem_mask,
                                               output, moment_localization_labels)
               }, output.logits

    def forward_encoders(self, question_tokens, question_mask, video_features, video_mask, moment_localization_labels):
        q_encoder_out = self.transformer.encoder(input_ids=question_tokens, attention_mask=question_mask)
        q_avg_pooled = q_encoder_out.last_hidden_state.mean(dim=1)

        mem_out = self.forward_memory(video_features, video_mask, moment_localization_labels, q_avg_pooled)
        if isinstance(mem_out, tuple):
            final_memory, additional_loss = mem_out
        else:
            assert torch.is_tensor(mem_out)
            final_memory = mem_out
            additional_loss = {}
        mem_mask = torch.ones(final_memory.shape[:-1], device=final_memory.device, dtype=torch.bool)

        context, context_mask = self._prepare_context(final_memory, mem_mask, question_mask=question_mask,
                                                      q_encoder_out=q_encoder_out)
        return context, context_mask, final_memory, mem_mask, additional_loss

    def _prepare_context(self, final_memory, mem_mask, question_tokens=None, question_mask=None, q_encoder_out=None):
        if q_encoder_out:
            encoder_out = q_encoder_out
        else:
            encoder_out = self.transformer.encoder(input_ids=question_tokens, attention_mask=question_mask)
        context = torch.cat([
            final_memory,
            encoder_out.last_hidden_state
        ], dim=1)
        context_mask = torch.cat([
            mem_mask,
            question_mask
        ], dim=1)
        return context, context_mask

    def autoregressive_forward(self, question_tokens, question_mask, video_features, video_mask) -> torch.Tensor:
        context, context_mask, _, _, _ = self.forward_encoders(question_tokens, question_mask, video_features,
                                                               video_mask, None)
        # noinspection PyTypeChecker
        enc_out = BaseModelOutput(last_hidden_state=context)
        return self.transformer.generate(encoder_outputs=enc_out, attention_mask=context_mask)

    def forward_memory(self, video_features, video_mask,
                       moment_localization_labels,
                       question_encoding) -> Union[torch.Tensor,
                                                   Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward video input to memory.

        :param video_features: Tensor of shape Batch x Sequence x Features
        :param video_mask: Tensor of shape Batch x Sequence
        :param moment_localization_labels: Tensor of shape Batch x Sequence, with entries between 0.0 and 1.0
                        (smoothed label saying if each frame is part of ground truth moment). Might be None.
        :param question_encoding: Tensor of shape Batch x Hidden, Mean-pooled representation of the question.
                                  Should only be used for additional loss, not memory construction!
        :return: memory, Tensor of shape Batch x MemoryLength x Hidden
                 or tuple of (memory, additional_loss_dict)
        """
        raise NotImplementedError

    # noinspection PyMethodMayBeStatic
    def calc_additional_loss(self, question_tokens, question_mask,
                             video_features, video_mask,
                             answer_tokens, answer_mask, batch_sample_ids,
                             context, context_mask, final_memory, mem_mask,
                             transformer_output, moment_localization_labels) -> Dict[str, torch.Tensor]:
        return {}
