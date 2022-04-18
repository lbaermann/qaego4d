from itertools import chain
from typing import Tuple, Dict

import torch
from torch import nn

from .base import MemoryAugmentedTransformerEmqaModel
from .external.compressive_transformer import CompressiveTransformer


class CompressiveTransformerEmqaModel(MemoryAugmentedTransformerEmqaModel):

    def __init__(
            self,
            pretrained_enc_dec: str,
            input_dim: int,
            hidden_dim: int = 512,
            num_layers: int = 10,
            heads: int = 8,
            block_length: int = 16,
            mem_length: int = 32,
            cmem_lengths=None,
            compression_factors=4,
            use_ltmem: bool = True,
            memory_layers=None,
            dropout: float = 0.1
    ):
        super().__init__(pretrained_enc_dec)
        self.mem_transformer = CompressiveTransformer(
            num_tokens=1,  # Embedding is skipped (video features input)
            # However, we set emb_dim to automatically use CompressiveTransformer.to_model_dim
            emb_dim=input_dim,
            dim=hidden_dim, depth=num_layers, heads=heads,
            seq_len=block_length, mem_len=mem_length,
            cmem_lengths=cmem_lengths,
            cmem_ratios=compression_factors,
            use_ltmem=use_ltmem, memory_layers=memory_layers,
            attn_layer_dropout=dropout, ff_dropout=dropout, attn_dropout=dropout,
            reconstruction_attn_dropout=dropout,
            gru_gated_residual=False, mogrify_gru=False, ff_glu=False, one_kv_head=False
        )
        self.mem_transformer.token_emb = nn.Identity()
        self.mem_transformer.to_logits = nn.Identity()

    def forward_memory(self, video_features, video_mask,
                       moment_localization_labels,
                       question_encoding) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, seq_len = video_features.shape[0], video_features.shape[1]

        block_length = self.mem_transformer.seq_len
        num_blocks = seq_len // block_length + (0 if seq_len % block_length == 0 else 1)

        x = video_features
        memories = (None, None, None)
        aux_loss_sum = torch.zeros(1, device=x.device)
        for i in range(num_blocks):
            current_slice = slice(i * block_length, (i + 1) * block_length)
            input_block = x[:, current_slice, :]
            out, memories, aux_loss = self.mem_transformer(input_block,
                                                           memories=memories,
                                                           mask=video_mask[:, current_slice])
            aux_loss_sum += aux_loss

        if num_blocks > 0:
            mem, cmems, ltmem = memories
            # memories is a tuple with three items. mem and ltmem are tensors, cmems a list of tensors, all of size
            #   (num_memory_layers x batch x memory_seq_length x hidden)
            #   ltmem always has memory_seq_length of either 0 or 1
            # out is (batch x sequence x hidden) from transformers last layer.
            # concatenate at sequence level (dim=1), but treat each mem layer as it's own vector
            # Also, treat all compression layers the same and simply concatenate
            memories = mem, *cmems, ltmem
            # noinspection PyUnboundLocalVariable
            complete_em = torch.cat([out] + [layer_mem for layer_mem in chain(*memories)], dim=1)  # B x S x H
        else:
            complete_em = torch.empty(bsz, 0, self.output_size,
                                      device=x.device, dtype=x.dtype)

        return complete_em, {'ct_aux_loss': aux_loss_sum}
