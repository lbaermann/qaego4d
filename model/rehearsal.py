import math
from typing import Dict, Optional, Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModel, PreTrainedModel
from transformers.models.roberta import RobertaModel, RobertaConfig

from model.base import MemoryAugmentedTransformerEmqaModel


class RehearsalTrainingModule(nn.Module):

    def __init__(self,
                 input_size: int,
                 mem_hidden_size: int,
                 num_samples: int,
                 sample_length: int,
                 positive_mask_ratio: float = 0.5,
                 negative_replacement_ratio: float = 0.5,
                 invert_teacher_sequence: bool = False,
                 pretrained_decoder: Optional[str] = None,
                 decoder_params: Optional[Dict[str, Any]] = None,
                 sampling_teacher_weights_file: Optional[str] = None
                 ) -> None:
        super().__init__()
        if pretrained_decoder is not None:
            self.decoder = RobertaModel.from_pretrained(pretrained_decoder, config=RobertaConfig.from_pretrained(
                pretrained_decoder, is_decoder=True, add_cross_attention=True
            ))
        else:
            model_cfg = RobertaConfig(**decoder_params, is_decoder=True, add_cross_attention=True)
            self.decoder = RobertaModel(model_cfg)
        hidden_size = self.decoder.config.hidden_size
        self.input_dimension_adjustment_layer = nn.Linear(input_size, hidden_size, bias=False)
        self.em_dimension_adjustment_layer = nn.Linear(mem_hidden_size, hidden_size, bias=False)
        self.num_samples = num_samples
        self.sample_length = sample_length
        self.positive_mask_ratio = positive_mask_ratio
        self.negative_replacement_ratio = negative_replacement_ratio
        self.invert_teacher_sequence = invert_teacher_sequence

        empty = torch.empty(2, 1, hidden_size)
        nn.init.kaiming_uniform_(empty)
        self.class_token_emb = nn.Parameter(empty[0])
        self.mask_token_emb = nn.Parameter(empty[1])
        self.pos_neg_projection = nn.Sequential(
            nn.Linear(hidden_size, 1, bias=False),
            nn.Sigmoid()
        )

        if sampling_teacher_weights_file:
            self.teacher_sampling_weights = torch.load(sampling_teacher_weights_file)
        else:
            self.teacher_sampling_weights = None

    def forward(self, memory, mem_mask, original_input, input_mask, batch_sample_ids):
        bsz, l_x = original_input.shape[0:2]
        h = self.decoder.config.hidden_size
        sample_length = min(self.sample_length, l_x)
        num_tokens_to_mask = int(sample_length * self.positive_mask_ratio)
        num_tokens_to_replace = int((sample_length - num_tokens_to_mask) * self.negative_replacement_ratio)
        dims = bsz, l_x, h, sample_length, num_tokens_to_mask, num_tokens_to_replace
        assert bsz > 1  # Negative sampling from other batches requires at least bsz=2

        samples, negative_samples, masked_items_map, padding_mask, start_indices = self._construct_samples(
            original_input, input_mask, dims, batch_sample_ids)
        hiddens_neg, hiddens_pos = self._forward_transformer(memory, mem_mask, samples, negative_samples,
                                                             padding_mask, dims)

        recollection_loss = self._calc_recollection_loss(original_input, hiddens_pos,
                                                         masked_items_map, start_indices, dims)
        familiarity_loss = self._calc_familiarity_loss(hiddens_neg, hiddens_pos, dims)

        return recollection_loss, familiarity_loss

    def _construct_samples(self, original_input, input_padding_mask, dims, batch_sample_ids):
        original_input = self.input_dimension_adjustment_layer(original_input)
        # original_input: bsz x l_x x h
        # input_padding_mask: bsz x l_x
        bsz, l_x, h, sample_length, num_tokens_to_mask, num_tokens_to_replace = dims
        start_indices = self._choose_start_indices(dims, batch_sample_ids)
        samples = torch.stack([
            torch.stack([
                torch.cat((self.class_token_emb, original_input[i, start:start + sample_length]), dim=0)
                for start in start_indices[i]
            ])
            for i in range(bsz)
        ])  # bsz x num_samples x sample_length x hidden
        negative_samples = samples.clone()
        # masked_items_map is 1 for masked input items (!) i.e. 1 for items that are replaced with mask token.
        masked_items_map = torch.zeros(bsz, self.num_samples, 1 + sample_length,  # + 1 for CLS token!
                                       device=samples.device, dtype=torch.bool)
        padding_mask = torch.ones_like(masked_items_map)  # one for items that should be used, 0 for padding
        all_indices = range(1, 1 + sample_length)
        for i in range(bsz):
            for s in range(self.num_samples):
                start = start_indices[i, s].item()
                padding_mask[i, s, 1:] = input_padding_mask[i, start:start + sample_length]
                padding_indices = (~padding_mask).nonzero()
                mask_indices = np.random.choice(all_indices, num_tokens_to_mask, replace=False)
                # Set masked_items_map only if these items are not padded. padding_mask is zero for non-padded entries
                masked_items_map[i, s, mask_indices] = padding_mask[i, s, mask_indices]
                unmasked_indices = list(set(all_indices) - set(mask_indices) - set(padding_indices))
                replacement_indices = np.random.choice(unmasked_indices, num_tokens_to_replace, replace=False)
                neg_sample_original_indices = replacement_indices + start - 1  # -1 because of CLS token
                negative_samples[i, s, replacement_indices] = original_input[(i + 1) % bsz, neg_sample_original_indices]
        samples[masked_items_map] = self.mask_token_emb
        negative_samples[masked_items_map] = self.mask_token_emb
        return samples, negative_samples, masked_items_map, padding_mask, start_indices

    def _choose_start_indices(self, dims, batch_sample_ids):
        bsz, l_x, _, sample_length, _, _ = dims
        num_fragments = l_x - sample_length + 1
        if self.teacher_sampling_weights is None:
            # uniform random sampling
            start_indices = np.stack([np.random.choice(max(1, num_fragments), self.num_samples, replace=False)
                                      for _ in range(bsz)])
        else:
            # biased random sampling guided by unconstrained teacher model (see section "What to rehearse?" in RM paper)
            teacher_attn_weights = [
                self.teacher_sampling_weights[sample_id]
                for sample_id in batch_sample_ids
            ]
            sample_distribution = torch.nn.utils.rnn.pad_sequence(teacher_attn_weights, batch_first=True)
            assert sample_distribution.shape[1] == num_fragments, f'{sample_distribution.shape}, {num_fragments}'
            if self.invert_teacher_sequence:
                sample_distribution = sample_distribution.flip(dims=(1,))
            cum_distribution = sample_distribution.cumsum(dim=-1)
            rand_sources = torch.rand(bsz, self.num_samples)
            # noinspection PyTypeChecker
            start_indices = torch.sum(cum_distribution[:, None, :] < rand_sources[:, :, None], dim=-1)
        return start_indices

    def _forward_transformer(self, memory, mem_mask, samples, negative_samples, padding_mask, dims):
        bsz, _, h, sample_length, _, _ = dims
        sample_length = 1 + sample_length  # CLS token at the beginning

        model_in = torch.cat((samples.view(-1, sample_length, h), negative_samples.view(-1, sample_length, h)), dim=0)
        model_padding_mask = padding_mask.view(-1, sample_length).repeat(2, 1)
        extended_attn_mask = model_padding_mask[:, None, :]  # Extend here already so that no causal mask is added.
        memory = self.em_dimension_adjustment_layer(memory)
        # bsz must be repeated for each sample and again twice (positive vs. negative sample)
        memory = memory.repeat(2 * self.num_samples, 1, 1)
        mem_mask = mem_mask.repeat(2 * self.num_samples, 1)
        output = self.decoder(inputs_embeds=model_in, attention_mask=extended_attn_mask,
                              encoder_hidden_states=memory, encoder_attention_mask=mem_mask)
        hiddens_pos, hiddens_neg = output.last_hidden_state.split(model_in.shape[0] // 2)
        hiddens_pos = hiddens_pos.reshape(bsz, self.num_samples, sample_length, h)
        hiddens_neg = hiddens_neg.reshape(bsz, self.num_samples, sample_length, h)
        return hiddens_neg, hiddens_pos

    def _calc_recollection_loss(self, original_input, hiddens_pos, masked_items_map, start_indices, dims):
        bsz, _, _, _, num_tokens_to_mask, _ = dims

        recollection_loss = torch.scalar_tensor(0, device=hiddens_pos.device, dtype=hiddens_pos.dtype)
        for i in range(bsz):
            for s in range(self.num_samples):
                mask_indices = masked_items_map[i, s].nonzero().squeeze(dim=-1)
                # -1 because masked_items respects CLS tokens at position 0, which is not part of original_input
                original_masked_item_indices = mask_indices + start_indices[i, s].item() - 1
                reconstructed_items = F.linear(hiddens_pos[i, s, mask_indices],
                                               self.input_dimension_adjustment_layer.weight.T)
                # other neg. sampling strategy?
                #  (here, the contrastive loss is calculated between all the masked items from the original input.
                #   This might be bad, since neighboring video clips might have actually similar features?
                #   Other option would be to sample randomly from another batch)
                #  Also introduce another hyperparameter: number of sampled items. RM uses 30 for ActivityNetQA
                target_items = original_input[i, original_masked_item_indices]
                assert reconstructed_items.shape == target_items.shape, \
                    f'{reconstructed_items.shape} != {target_items.shape}'
                products = torch.inner(reconstructed_items, target_items)  # x,y = sum(reconstructed[x] * original[y])
                recollection_loss += torch.log_softmax(products, dim=1).trace()
        recollection_loss = - recollection_loss / (num_tokens_to_mask * self.num_samples * bsz)
        return recollection_loss

    def _calc_familiarity_loss(self, hiddens_neg, hiddens_pos, dims):
        bsz = dims[0]
        pos_cls_output = hiddens_pos[..., 0, :]
        neg_cls_output = hiddens_neg[..., 0, :]
        pos_scores = self.pos_neg_projection(pos_cls_output)
        neg_scores = self.pos_neg_projection(neg_cls_output)
        # This has wrong sign in the RM Paper (Eq. 4), leading to NaN
        #   neg_score has goal 0 => 1-neg_score has goal 1 => want to maximize its log
        #   (thus needs negative sign due to overall loss minimization)
        #   since log(x) in (-inf, 0) for x in (0, 1)
        #   analogous for pos_score, which has goal 1 directly
        familiarity_loss = pos_scores.log() + (1 - neg_scores).log()
        familiarity_loss = -torch.sum(familiarity_loss) / (self.num_samples * bsz)
        return familiarity_loss


class RehearsalMemoryMachine(nn.Module):

    def __init__(self,
                 pretrained_encoder: str,
                 input_dim: int,
                 mem_hidden_size: int,
                 num_memory_slots: int,
                 segment_length: int,
                 slot_to_item_num_heads: int = 1,
                 use_independent_gru_per_mem_slot=False
                 ) -> None:
        super().__init__()
        self.segment_length = segment_length
        self.num_memory_slots = num_memory_slots
        self.mem_hidden_size = mem_hidden_size
        self.encoder: PreTrainedModel = AutoModel.from_pretrained(pretrained_encoder)
        if hasattr(self.encoder, 'get_encoder'):  # In case pretrained_encoder is actually encoder-decoder model
            self.encoder = self.encoder.get_encoder()
        feature_size = self.encoder.get_input_embeddings().embedding_dim
        if input_dim == feature_size:
            self.input_transform = nn.Identity()
        else:
            self.input_transform = nn.Linear(input_dim, feature_size, bias=False)
        self.slot_to_item_attn = nn.MultiheadAttention(embed_dim=mem_hidden_size,
                                                       kdim=feature_size, vdim=feature_size,
                                                       num_heads=slot_to_item_num_heads,
                                                       batch_first=True)
        num_recurrent_units = num_memory_slots if use_independent_gru_per_mem_slot else 1
        self.recurrent_units = nn.ModuleList([
            nn.GRUCell(input_size=mem_hidden_size,
                       hidden_size=mem_hidden_size,
                       bias=False)
            for _ in range(num_recurrent_units)
        ])

    def forward(self, input_items, input_mask) -> torch.Tensor:
        """
        Process the input items, and return memory state at the end.
        :param input_items: input of shape Batch x Sequence x InputHidden
        :param input_mask: mask of shape Batch x Sequence, 1 = valid token, 0 = masked token
        :return: memory state of shape Batch x NumMemorySlots x MemHidden
        """
        bsz, seq, h = input_items.shape
        assert input_mask.shape == (bsz, seq)

        x = self.input_transform(input_items)

        num_segments = math.ceil(seq / self.segment_length)
        memory = torch.zeros(bsz, self.num_memory_slots, self.mem_hidden_size,
                             device=x.device, dtype=x.dtype)

        for t in range(num_segments):
            current_slice = slice(t * self.segment_length, (t + 1) * self.segment_length)
            x_t = x[:, current_slice]
            mask_t = input_mask[:, current_slice]
            active_batches = mask_t.sum(dim=1) > 0
            if not active_batches.any():
                raise RuntimeError('Bad batch - padding only?')

            f_t = self.encoder(inputs_embeds=x_t[active_batches],
                               attention_mask=mask_t[active_batches]).last_hidden_state

            l_t, attn_weights = self.slot_to_item_attn(query=memory[active_batches],
                                                       key=f_t, value=f_t,
                                                       key_padding_mask=~mask_t[active_batches])
            # l_t : bsz x num_memory_slots x mem_hidden
            if len(self.recurrent_units) == 1:
                flattened_l_t = l_t.reshape(-1, self.mem_hidden_size)
                flattened_mem = memory[active_batches].reshape(-1, self.mem_hidden_size)
                new_mem = self.recurrent_units[0](input=flattened_l_t, hx=flattened_mem)
                active_bsz = active_batches.sum()
                memory[active_batches] = new_mem.view(active_bsz, self.num_memory_slots, self.mem_hidden_size)
            else:
                for i in range(self.num_memory_slots):
                    memory[active_batches, i, :] = self.recurrent_units[i](input=l_t[:, i, :],
                                                                           hx=memory[active_batches, i, :])

        return memory


class RehearsalMemoryEmqaModel(MemoryAugmentedTransformerEmqaModel):

    def __init__(self,
                 rehearsal_machine: RehearsalMemoryMachine,
                 rehearsal_trainer: RehearsalTrainingModule,
                 pretrained_enc_dec: str
                 ) -> None:
        super().__init__(pretrained_enc_dec)
        self.rehearsal_machine = rehearsal_machine
        self.rehearsal_trainer = rehearsal_trainer

    def forward_memory(self, video_features, video_mask,
                       moment_localization_labels, question_encoding  # unused
                       ):
        return self.rehearsal_machine(video_features, video_mask)

    def calc_additional_loss(self, question_tokens, question_mask, video_features, video_mask, answer_tokens,
                             answer_mask, batch_sample_ids, context, context_mask, final_memory, mem_mask,
                             transformer_output, moment_localization_labels):
        if self.rehearsal_trainer:
            loss_rec, loss_fam = self.rehearsal_trainer(final_memory, mem_mask,
                                                        video_features, video_mask,
                                                        batch_sample_ids)
        else:
            loss_rec, loss_fam = torch.zeros(2, dtype=context.dtype, device=context.device)
        return {
            'recollection_loss': loss_rec,
            'familiarity_loss': loss_fam
        }
