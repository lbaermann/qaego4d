import torch
from torch import nn


class SeqMomentLocalizationLossModule(nn.Module):

    def __init__(self,
                 seq_hidden_dim: int,
                 question_hidden_dim: int) -> None:
        super().__init__()
        total = seq_hidden_dim + question_hidden_dim
        self.projection_layer = nn.Sequential(
            nn.Linear(total, total // 2, bias=True),
            nn.LeakyReLU(),
            nn.Linear(total // 2, 1, bias=False),
            nn.Tanh()
        )
        self.loss = nn.MSELoss(reduction='none')

    def forward(self, output_sequence, sequence_mask,
                moment_localization_labels, question_encoding) -> torch.Tensor:
        """
        Calculate moment localization loss based on output sequence.

        :param output_sequence: Tensor of shape Batch x Sequence x Hidden
        :param sequence_mask: Tensor of shape Batch x Sequence
        :param moment_localization_labels: Tensor of shape Batch x Sequence, with entries between 0.0 and 1.0
                        (smoothed label saying if each frame is part of ground truth moment). Might be None.
        :param question_encoding: Tensor of shape Batch x Hidden, Mean-pooled representation of the question.
        :return: scalar Tensor containing moment localization loss
        """
        bsz, seq = output_sequence.shape[:2]
        question = question_encoding[:, None, :].expand(bsz, seq, -1)
        seq_with_question = torch.cat([output_sequence, question],
                                      dim=-1)  # Batch x Sequence x 2*Hidden
        output_scores = self.projection_layer(seq_with_question)  # Batch x Sequence x 1
        loss_per_item = self.loss(output_scores.squeeze(), moment_localization_labels)
        loss_per_item = loss_per_item * sequence_mask
        return loss_per_item.mean()


class TransformerMomentLocalizationLossModule(nn.Module):

    def __init__(self, softmax_temperature, att_loss_type, hinge_margin, lse_alpha):
        # softmax_temperature: softmax softmax_temperature when rescaling video-only attention scores
        # sum_attentions: whether to use LSE loss on sum of attention weights (target vs non-target interval)
        super().__init__()
        self.att_loss_type = att_loss_type
        self.softmax_temperature = softmax_temperature
        self.hinge_margin = hinge_margin
        self.lse_alpha = lse_alpha

    def forward(self, question_tokens, transformer_output,
                moment_localization_labels, video_mask):
        """

        :param question_tokens: Batch x QuestionSequence
        :param transformer_output: transformer.ModelOutputWithCrossAttentions. Input to cross attention is assumed as
                                   concatenated [video, question], so that transformer_output.cross_attentions has shape
                                   Batch x NumHeads x DecoderSequence x (VideoSeq + QuestionSeq)
        :param moment_localization_labels: Batch x VideoSequence. 1 for target frames, 0 elsewhere
        :param video_mask: Batch x VideoSequence. 0 for masked indices, 1 for valid
        :return: loss
        """
        # Tuple of tensors (one for each layer) of shape (batch, num_heads, dec_seq_len, enc_seq_len)
        cross_attentions = transformer_output.cross_attentions
        # (batch, num_layers, num_heads, dec_seq_len, enc_seq_len)
        cross_attentions = torch.stack(cross_attentions, dim=1)
        question_len = question_tokens.shape[1]

        # Need to take care of masked indices before re-applying softmax
        cross_attn_on_video = cross_attentions[:, :, :, :, :-question_len] / self.softmax_temperature
        mask_value = torch.scalar_tensor(-1e6, dtype=cross_attentions.dtype, device=cross_attentions.device)
        cross_attn_on_video = torch.where(video_mask[:, None, None, None, :], cross_attn_on_video, mask_value)
        cross_attn_on_video = cross_attn_on_video.softmax(dim=-1)

        # loss per batch
        loss = self.calc_attention_loss(cross_attn_on_video, moment_localization_labels, video_mask)
        return loss.mean()

    def calc_attention_loss(self, cross_attn_on_video, moment_localization_labels, video_mask):
        raise NotImplementedError

    def ranking_loss(self, pos_scores, neg_scores):
        if self.att_loss_type == "hinge":
            # max(0, m + S_pos - S_neg)
            loss = torch.clamp(self.hinge_margin + neg_scores - pos_scores, min=0)
        elif self.att_loss_type == "lse":
            # log[1 + exp(scale * (S_pos - S_neg))]
            loss = torch.log1p(torch.exp(self.lse_alpha * (neg_scores - pos_scores)))
        else:
            raise NotImplementedError("Only support hinge and lse")
        return loss


class SummedAttentionTransformerMomentLocLoss(TransformerMomentLocalizationLossModule):

    def __init__(self, softmax_temperature=0.1, att_loss_type='lse', hinge_margin=0.4, lse_alpha=20):
        super().__init__(softmax_temperature, att_loss_type, hinge_margin, lse_alpha)

    def calc_attention_loss(self, cross_attn_on_video, moment_localization_labels, video_mask):
        bsz = len(video_mask)
        pos_mask = moment_localization_labels == 1
        neg_mask = (moment_localization_labels == 0) * video_mask
        mean_attn = cross_attn_on_video.mean(dim=(1, 2, 3))  # mean over heads, layers and decoder positions
        pos_scores = torch.stack([
            mean_attn[b, pos_mask[b]].sum()
            for b in range(bsz)
        ])
        neg_scores = torch.stack([
            mean_attn[b, neg_mask[b]].sum()
            for b in range(bsz)
        ])
        return self.ranking_loss(pos_scores, neg_scores)


# This is copied & modified from https://github.com/jayleicn/TVQAplus
class SamplingAttentionTransformerMomentLocLoss(TransformerMomentLocalizationLossModule):

    def __init__(self, num_negatives=2, use_hard_negatives=False, drop_topk=0,
                 negative_pool_size=0, num_hard=2,
                 softmax_temperature=0.1, att_loss_type='lse', hinge_margin=0.4, lse_alpha=20) -> None:
        super().__init__(softmax_temperature, att_loss_type, hinge_margin, lse_alpha)
        self.num_hard = num_hard
        self.negative_pool_size = negative_pool_size
        self.drop_topk = drop_topk
        self.use_hard_negatives = use_hard_negatives
        self.num_negatives = num_negatives

    def calc_attention_loss(self, cross_attn_on_video, att_labels, video_mask):
        # take max head, mean over layers and decoder positions
        scores = cross_attn_on_video.max(dim=2).values.mean(dim=(1, 2))

        # att_labels : Batch x VideoSequence. 0 for non-target, 1 for target
        # scores : Batch x VideoSequence. Between 0 and 1 as given by softmax
        # rescale to [-1, 1]
        scores = scores * 2 - 1

        pos_container = []  # contains tuples of 2 elements, which are (batch_i, img_i)
        neg_container = []
        bsz = len(att_labels)
        for b in range(bsz):
            pos_indices = att_labels[b].nonzero()  # N_pos x 1
            neg_indices = ((1 - att_labels[b]) * video_mask[b]).nonzero()  # N_neg x 1

            sampled_pos_indices, sampled_neg_indices = self._sample_negatives(scores[b], pos_indices, neg_indices)

            base_indices = torch.full((sampled_pos_indices.shape[0], 1), b, dtype=torch.long, device=pos_indices.device)
            pos_container.append(torch.cat([base_indices, sampled_pos_indices], dim=1))
            neg_container.append(torch.cat([base_indices, sampled_neg_indices], dim=1))

        pos_container = torch.cat(pos_container, dim=0)
        neg_container = torch.cat(neg_container, dim=0)

        pos_scores = scores[pos_container[:, 0], pos_container[:, 1]]
        neg_scores = scores[neg_container[:, 0], neg_container[:, 1]]

        att_loss = self.ranking_loss(pos_scores, neg_scores).mean(dim=-1)
        return att_loss

    def _sample_negatives(self, pred_score, pos_indices, neg_indices):
        """ Sample negatives from a set of indices. Several sampling strategies are supported:
        1, random; 2, hard negatives; 3, drop_topk hard negatives; 4, mix easy and hard negatives
        5, sampling within a pool of hard negatives; 6, sample across images of the same video.
        Args:
            pred_score: (num_img)
            pos_indices: (N_pos, 1)
            neg_indices: (N_neg, 1)
        Returns:

        """
        num_unique_pos = len(pos_indices)
        sampled_pos_indices = torch.cat([pos_indices] * self.num_negatives, dim=0)
        if self.use_hard_negatives:
            # print("using use_hard_negatives")
            neg_scores = pred_score[neg_indices[:, 0]]
            max_indices = torch.sort(neg_scores, descending=True)[1].tolist()
            if self.negative_pool_size > self.num_negatives:  # sample from a pool of hard negatives
                hard_pool = max_indices[self.drop_topk:self.drop_topk + self.negative_pool_size]
                hard_pool_indices = neg_indices[hard_pool]
                num_hard_negs = self.num_negatives
                sampled_easy_neg_indices = []
                if self.num_hard < self.num_negatives:
                    easy_pool = max_indices[self.drop_topk + self.negative_pool_size:]
                    easy_pool_indices = neg_indices[easy_pool]
                    num_hard_negs = self.num_hard
                    num_easy_negs = self.num_negatives - num_hard_negs
                    sampled_easy_neg_indices = easy_pool_indices[
                        torch.randint(low=0, high=len(easy_pool_indices),
                                      size=(num_easy_negs * num_unique_pos,), dtype=torch.long)
                    ]
                sampled_hard_neg_indices = hard_pool_indices[
                    torch.randint(low=0, high=len(hard_pool_indices),
                                  size=(num_hard_negs * num_unique_pos,), dtype=torch.long)
                ]

                if len(sampled_easy_neg_indices) != 0:
                    sampled_neg_indices = torch.cat([sampled_hard_neg_indices, sampled_easy_neg_indices], dim=0)
                else:
                    sampled_neg_indices = sampled_hard_neg_indices

            else:  # directly take the top negatives
                sampled_neg_indices = neg_indices[max_indices[self.drop_topk:self.drop_topk + len(sampled_pos_indices)]]
        else:
            sampled_neg_indices = neg_indices[
                torch.randint(low=0, high=len(neg_indices), size=(len(sampled_pos_indices),), dtype=torch.long)
            ]
        return sampled_pos_indices, sampled_neg_indices
