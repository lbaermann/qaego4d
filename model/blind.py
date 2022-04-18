from typing import Tuple, Dict

import torch
from transformers import PreTrainedModel, AutoModelForSeq2SeqLM

from model.base import EmqaBaseModel


class BlindVqaModel(EmqaBaseModel):

    def __init__(self,
                 pretrained_model: str
                 ) -> None:
        super().__init__()
        self.transformer: PreTrainedModel = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model)
        assert self.transformer.config.is_encoder_decoder

    def teacher_forcing_forward(self, question_tokens, question_mask, video_features, video_mask, answer_tokens,
                                answer_mask, batch_sample_ids,
                                moment_localization_labels) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        output = self.transformer(input_ids=question_tokens, attention_mask=question_mask,
                                  labels=answer_tokens, decoder_attention_mask=answer_mask)
        return {'lm_loss': output.loss}, output.logits

    def autoregressive_forward(self, question_tokens, question_mask, video_features, video_mask) -> torch.Tensor:
        return self.transformer.generate(inputs=question_tokens, attention_mask=question_mask)
