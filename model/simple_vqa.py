from typing import Dict

import torch
from torch import nn
from transformers.modeling_outputs import Seq2SeqLMOutput

from model.base import MemoryAugmentedTransformerEmqaModel
from model.moment_loc import TransformerMomentLocalizationLossModule


# noinspection PyAbstractClass
class SimpleVqaModel(MemoryAugmentedTransformerEmqaModel):
    # Actually this is not really a MemoryAugmentedTransformerEmqaModel since it uses full attention over the input

    def __init__(self, input_size: int,
                 pretrained_enc_dec: str,
                 moment_localization_loss: TransformerMomentLocalizationLossModule = None) -> None:
        super().__init__(pretrained_enc_dec)
        self.moment_localization_loss = moment_localization_loss
        self.transformer.get_decoder().config.output_attentions = True

        hidden = self.transformer.get_input_embeddings().embedding_dim
        if input_size != hidden:
            self.transform_visual = nn.Linear(input_size, hidden, bias=False)
        else:
            self.transform_visual = nn.Identity()

    def forward_encoders(self, question_tokens, question_mask, video_features, video_mask, moment_localization_labels):
        visual_seq = self.transform_visual(video_features)
        context, context_mask = self._prepare_context(visual_seq, video_mask, question_tokens, question_mask)
        return context, context_mask, visual_seq, video_mask, {}

    def calc_additional_loss(self, question_tokens, question_mask, video_features, video_mask, answer_tokens,
                             answer_mask, batch_sample_ids, context, context_mask, final_memory, mem_mask,
                             transformer_output: Seq2SeqLMOutput,
                             moment_localization_labels) -> Dict[str, torch.Tensor]:
        if self.moment_localization_loss:
            return {
                'moment_localization': self.moment_localization_loss(
                    question_tokens, transformer_output, moment_localization_labels, video_mask)
            }
        else:
            return {}
