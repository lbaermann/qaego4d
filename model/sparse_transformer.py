from typing import Dict, Tuple

import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput
from transformers.models.longformer.modeling_longformer import LongformerModel
from transformers.models.bigbird_pegasus.modeling_bigbird_pegasus import BigBirdPegasusForConditionalGeneration

from model.base import EmqaBaseModel
from model.moment_loc import TransformerMomentLocalizationLossModule


# noinspection PyAbstractClass
class LongformerVqaModel(EmqaBaseModel):
    # Actually this is not really a EMQA model since it uses full attention over the input

    def __init__(self, input_size: int,
                 pretrained_enc_dec: str,
                 pretrained_longformer: str,
                 moment_localization_loss: TransformerMomentLocalizationLossModule = None) -> None:
        super().__init__()
        # Actually, only using the decoder of the enc_dec model. Just want to use LM head + cross attention
        self.moment_localization_loss = moment_localization_loss
        self.enc_dec = AutoModelForSeq2SeqLM.from_pretrained(pretrained_enc_dec)
        self.enc_dec.encoder.block = None
        self.enc_dec.encoder.final_layer_norm = None
        self.enc_dec.decoder.config.output_attentions = True

        self.longformer = LongformerModel.from_pretrained(pretrained_longformer, add_pooling_layer=False)

        longformer_h = self.longformer.get_input_embeddings().embedding_dim
        if input_size != longformer_h:
            self.transform_visual = nn.Linear(input_size, longformer_h, bias=False)
        else:
            self.transform_visual = nn.Identity()
        decoder_h = self.enc_dec.get_input_embeddings().embedding_dim
        if longformer_h != decoder_h:
            self.transform_context = nn.Linear(longformer_h, decoder_h, bias=False)
        else:
            self.transform_context = nn.Identity()

    def teacher_forcing_forward(self, question_tokens, question_mask,
                                video_features, video_mask,
                                answer_tokens, answer_mask,
                                batch_sample_ids,
                                moment_localization_labels) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        context, context_mask = self.forward_encoders(question_tokens, question_mask, video_features, video_mask)
        output = self.enc_dec(labels=answer_tokens, decoder_attention_mask=answer_mask,
                              encoder_outputs=(context,), attention_mask=context_mask)
        loss_dict = {'lm_loss': output.loss}
        if self.moment_localization_loss:
            loss_dict['moment_localization'] = self.moment_localization_loss(
                question_tokens, output, moment_localization_labels, video_mask)
        return loss_dict, output.logits

    def autoregressive_forward(self, question_tokens, question_mask, video_features, video_mask) -> torch.Tensor:
        context, context_mask = self.forward_encoders(question_tokens, question_mask, video_features, video_mask)
        # noinspection PyTypeChecker
        enc_out = BaseModelOutput(last_hidden_state=context)
        return self.enc_dec.generate(encoder_outputs=enc_out, attention_mask=context_mask)

    def forward_encoders(self, question_tokens, question_mask, video_features, video_mask):
        longformer_in = torch.cat([
            self.transform_visual(video_features),
            self.longformer.get_input_embeddings()(question_tokens)
        ], dim=1)
        longformer_mask = torch.cat([
            video_mask,
            question_mask
        ], dim=1)
        # initialize to global attention to be deactivated for all tokens
        global_attention_mask = torch.zeros_like(longformer_mask)
        # Set global attention to question tokens
        global_attention_mask[:, -question_tokens.shape[1]:] = 1

        context = self.longformer(
            inputs_embeds=longformer_in, attention_mask=longformer_mask,
            global_attention_mask=global_attention_mask,
        ).last_hidden_state
        context = self.transform_context(context)
        return context, longformer_mask


class BigBirdVqaModel(EmqaBaseModel):
    # Actually this is not really a EMQA model since it uses full attention over the input

    def __init__(self, input_size: int,
                 pretrained_bigbird: str,
                 moment_localization_loss: TransformerMomentLocalizationLossModule = None,
                 gradient_checkpointing=False) -> None:
        super().__init__()
        self.moment_localization_loss = moment_localization_loss
        self.bigbird = BigBirdPegasusForConditionalGeneration.from_pretrained(pretrained_bigbird)

        bigbird_h = self.bigbird.get_input_embeddings().embedding_dim
        if input_size != bigbird_h:
            self.transform_visual = nn.Linear(input_size, bigbird_h, bias=False)
        else:
            self.transform_visual = nn.Identity()

        self.gradient_checkpointing = gradient_checkpointing
        if gradient_checkpointing:
            self.bigbird.gradient_checkpointing_enable()

    def teacher_forcing_forward(self, question_tokens, question_mask,
                                video_features, video_mask,
                                answer_tokens, answer_mask,
                                batch_sample_ids,
                                moment_localization_labels) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        enc_in, enc_mask = self.prepare_encoder_input(question_tokens, question_mask, video_features, video_mask)
        output = self.bigbird(labels=answer_tokens, decoder_attention_mask=answer_mask,
                              inputs_embeds=enc_in, attention_mask=enc_mask,
                              use_cache=not self.gradient_checkpointing)
        loss_dict = {'lm_loss': output.loss}
        if self.moment_localization_loss:
            loss_dict['moment_localization'] = self.moment_localization_loss(
                question_tokens, output, moment_localization_labels, video_mask)
        return loss_dict, output.logits

    def autoregressive_forward(self, question_tokens, question_mask, video_features, video_mask) -> torch.Tensor:
        enc_in, enc_mask = self.prepare_encoder_input(question_tokens, question_mask, video_features, video_mask)
        return self.bigbird.generate(inputs_embeds=enc_in, attention_mask=enc_mask)

    def prepare_encoder_input(self, question_tokens, question_mask, video_features, video_mask):
        encoder_in = torch.cat([
            self.transform_visual(video_features),
            self.bigbird.get_input_embeddings()(question_tokens)
        ], dim=1)
        encoder_mask = torch.cat([
            video_mask,
            question_mask
        ], dim=1)
        return encoder_in, encoder_mask
