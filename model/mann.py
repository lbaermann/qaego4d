from typing import Literal, Tuple, Dict

import torch
from dnc import DNC
from torch import nn

from model.base import MemoryAugmentedTransformerEmqaModel
from model.external.stm import STM
from model.moment_loc import SeqMomentLocalizationLossModule


class SegmentationMemoryAugmentedTransformerEmqaModel(MemoryAugmentedTransformerEmqaModel):
    def __init__(self, pretrained_enc_dec: str,
                 segmentation_method: Literal['flat', 'avg'],
                 segment_length: int,
                 input_size: int,  # this is downscaled to dnc_input_size for each time step separately
                 mem_input_size: int,  # Actual input to Memory depends on segmentation_method
                 moment_loc: SeqMomentLocalizationLossModule = None
                 ) -> None:
        super().__init__(pretrained_enc_dec)
        self.segment_length = segment_length
        self.segmentation_method = segmentation_method
        self.input_downscale = (nn.Identity() if input_size == mem_input_size
                                else nn.Linear(input_size, mem_input_size, bias=False))
        self.actual_mem_input_size = (segment_length * mem_input_size
                                      if segmentation_method == 'flat' else mem_input_size)
        self.moment_loc = moment_loc

    def forward_memory(self, video_features, video_mask,
                       moment_localization_labels,
                       question_encoding) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # B: batch, H: downscaled hidden dim, L: segment_length, N: num_segments, S: input seq length
        seq_downscaled: torch.Tensor = self.input_downscale(video_features)  # B x S x H
        if self.segmentation_method == 'flat':
            mem_in, mem_in_mask = self._segment_flat(seq_downscaled, video_mask)
        elif self.segmentation_method == 'avg':
            mem_in, mem_in_mask = self._segment_avg(seq_downscaled, video_mask)
        else:
            raise ValueError(self.segmentation_method)
        memory, item_output = self.forward_segmented_memory(mem_in)

        aux_loss = self._calc_aux_loss(item_output, mem_in_mask, moment_localization_labels, question_encoding)
        return memory, aux_loss

    def _calc_aux_loss(self, item_output, mem_in_mask, moment_localization_labels, question_encoding):
        has_moment_loc = (item_output is not None
                          and self.moment_loc is not None
                          and moment_localization_labels is not None)
        if has_moment_loc:
            # have: B x input_seq
            # need: B x N
            segments = list(moment_localization_labels.split(self.segment_length, dim=1))  # N tensors of B x ( ≤L )
            segments = [segment_labels.sum(dim=1).clip(0, 1) for segment_labels in segments]  # N tensors of B
            moment_localization_labels = torch.stack(segments, dim=1)  # B x N
        aux_loss = {} if not has_moment_loc else {
            'moment_localization': self.moment_loc(item_output, mem_in_mask,
                                                   moment_localization_labels, question_encoding)
        }
        return aux_loss

    def _segment_flat(self, seq_downscaled, mask):
        bsz, _, h = seq_downscaled.shape
        segments = list(seq_downscaled.split(self.segment_length, dim=1))  # N tensors of B x ( ≤L ) x H
        mask_segments = list(mask.split(self.segment_length, dim=1))
        last_segment_length = segments[-1].shape[1]
        if last_segment_length != self.segment_length:
            # Zero-pad to segment length so that dnc_in can be constructed correctly
            segments[-1] = torch.cat([
                segments[-1],
                torch.zeros(bsz, self.segment_length - last_segment_length, h,
                            device=seq_downscaled.device, dtype=seq_downscaled.dtype)
            ], dim=1)
            mask_segments[-1] = torch.cat([
                mask_segments[-1],
                torch.zeros(bsz, self.segment_length - last_segment_length,
                            device=mask.device, dtype=mask.dtype)
            ])
        segments = [s.view(bsz, -1) for s in segments]  # N tensors of B x LH
        mem_in = torch.stack(segments, dim=1)  # B x N x LH
        mem_in_mask = torch.stack(mask_segments, dim=1)  # B x N x L
        mem_in_mask = mem_in_mask.any(dim=-1)  # B x N
        return mem_in, mem_in_mask

    def _segment_avg(self, seq_downscaled, mask):
        bsz, _, h = seq_downscaled.shape
        segments = list(seq_downscaled.split(self.segment_length, dim=1))  # N tensors of B x ( ≤L ) x H
        avg_segments = [x.mean(dim=1) for x in segments]  # N tensors of B x H
        mem_in = torch.stack(avg_segments, dim=1)  # B x N x H
        mask_segments = list(mask.split(self.segment_length, dim=1))  # N tensors of B x ( ≤L )
        mask_segments = [x.any(dim=1) for x in mask_segments]  # N tensors of B
        segment_mask = torch.stack(mask_segments, dim=1)  # B x N
        return mem_in, segment_mask

    def forward_segmented_memory(self, mem_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # mem_in: B x N x LH
        raise NotImplementedError


class DncEmqaModel(SegmentationMemoryAugmentedTransformerEmqaModel):

    def __init__(self, pretrained_enc_dec: str,
                 segmentation_method: Literal['flat', 'avg'],
                 segment_length: int,
                 input_size: int,  # this is downscaled to dnc_input_size for each time step separately
                 dnc_input_size: int,  # Actual input to DNC is segment_length * dnc_input_size
                 rnn_hidden_size: int,
                 num_dnc_layers=1,
                 num_rnn_hidden_layers=2,
                 num_mem_cells=5,
                 mem_hidden_size=10,
                 moment_loc: SeqMomentLocalizationLossModule = None
                 ) -> None:
        super().__init__(pretrained_enc_dec, segmentation_method, segment_length,
                         input_size, dnc_input_size, moment_loc)
        self.dnc = DNC(self.actual_mem_input_size,
                       rnn_hidden_size, num_layers=num_dnc_layers,
                       num_hidden_layers=num_rnn_hidden_layers,
                       nr_cells=num_mem_cells, cell_size=mem_hidden_size)

    def forward_segmented_memory(self, mem_in: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        self._hack_dnc_gpu_ids(mem_in.device)
        output, (controller_hidden, memory, read_vectors) = self.dnc(mem_in, reset_experience=True)
        # memory['memory']: Batch x NumMemCells x MemHiddenSize
        return memory['memory'], output

    def _hack_dnc_gpu_ids(self, device):
        gpu_id = -1 if device.type == 'cpu' else device.index
        if self.dnc.gpu_id == gpu_id:
            return
        # This is a hack because DNC is programmed not in a pytorch idiomatic way...
        self.dnc.gpu_id = gpu_id
        for m in self.dnc.memories:
            m.gpu_id = gpu_id
            m.I = m.I.to(device=device)


class StmEmqaModel(SegmentationMemoryAugmentedTransformerEmqaModel):

    def __init__(self, pretrained_enc_dec: str,
                 segmentation_method: Literal['flat', 'avg'],
                 segment_length: int,
                 input_size: int,  # this is downscaled to dnc_input_size for each time step separately
                 stm_input_size: int,  # Actual input to DNC is segment_length * dnc_input_size
                 mem_hidden_size=10,
                 stm_step=1,
                 stm_num_slot=8,
                 stm_mlp_size=128,
                 stm_slot_size=96,
                 stm_rel_size=96,
                 stm_out_att_size=64
                 ) -> None:
        super().__init__(pretrained_enc_dec, segmentation_method, segment_length, input_size, stm_input_size)
        self.stm = STM(self.actual_mem_input_size, mem_hidden_size,
                       stm_step, stm_num_slot, stm_mlp_size, stm_slot_size,
                       stm_rel_size, stm_out_att_size)

    def forward_segmented_memory(self, mem_in: torch.Tensor) -> Tuple[torch.Tensor, None]:
        mem_in = mem_in.transpose(0, 1)  # switch batch and sequence dim for STM
        output, (read_heads, item_memory_state, rel_memory_state) = self.stm(mem_in)
        # should return Batch x NumMemCells x MemHiddenSize
        # output is Batch x MemHiddenSize
        return output[:, None, :], None
