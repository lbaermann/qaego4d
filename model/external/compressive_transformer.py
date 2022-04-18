# Modified from https://github.com/lucidrains/compressive-transformer-pytorch

import math
import sys
from collections import namedtuple
from functools import partial
from inspect import isfunction
from typing import Type, Tuple, List, Union

import torch
import torch.nn.functional as F
from torch import nn, Tensor

# structs

Memory: Type[Tuple[Tensor, List[Tensor], Tensor]] = namedtuple('Memory', ['mem', 'compressed_mem', 'lt_mem'])


# helper functions

def to(t):
    return {'dtype': t.dtype, 'device': t.device}


def cast_tuple(el):
    return el if isinstance(el, tuple) else (el,)


def default(x, val):
    if x is not None:
        return x
    return val if not isfunction(val) else val()


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def reshape_dim(t, dim, split_dims):
    shape = list(t.shape)
    num_dims = len(shape)
    dim = (dim + num_dims) % num_dims
    shape[dim:dim + 1] = split_dims
    return t.reshape(shape)


def split_at_index(dim, index, t):
    pre_slices = (slice(None),) * dim
    l = (*pre_slices, slice(None, index))
    r = (*pre_slices, slice(index, None))
    return t[l], t[r]


def queue_fifo(*args, length, dim=-2):
    queue = torch.cat(args, dim=dim)
    if length > 0:
        return split_at_index(dim, -length, queue)

    device = queue.device
    shape = list(queue.shape)
    shape[dim] = 0
    return queue, torch.empty(shape, device=device)


def shift(x):
    *_, i, j = x.shape
    zero_pad = torch.zeros((*_, i, i), **to(x))
    x = torch.cat([x, zero_pad], -1)
    l = i + j - 1
    x = x.view(*_, -1)
    zero_pad = torch.zeros(*_, -x.size(-1) % l, **to(x))
    shifted = torch.cat([x, zero_pad], -1).view(*_, -1, l)
    return shifted[..., :i, i - 1:]


def iterate_tensor(t):
    length = t.shape[0]
    for ind in range(length):
        yield t[ind]


# full attention for calculating auxiliary reconstruction loss

def full_attn(q, k, v, dropout_fn=None):
    *_, dim = q.shape
    dots = torch.einsum('bhid,bhjd->bhij', q, k) * (dim ** -0.5)
    attn = dots.softmax(dim=-1)
    if dropout_fn is not None:
        attn = dropout_fn(attn)
    return torch.einsum('bhij,bhjd->bhid', attn, v)


# helper classes

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        out = cast_tuple(out)
        ret = (out[0] + x), *out[1:]
        return ret


class GRUGating(nn.Module):
    def __init__(self, dim, fn, mogrify=False):
        super().__init__()
        self.dim = dim
        self.fn = fn
        self.gru = nn.GRUCell(dim, dim)
        if mogrify:
            try:
                # noinspection PyPackageRequirements
                from mogrifier import Mogrifier
                self.mogrify = Mogrifier(dim, factorize_k=dim // 4) if mogrify else None
            except ImportError:
                print('!! mogrify is set, but mogrifier library not available!'
                      ' Run "pip install mogrifier" to fix.', file=sys.stderr)

    def forward(self, x, **kwargs):
        batch, dim = x.shape[0], self.dim
        out = self.fn(x, **kwargs)
        (y, *rest) = cast_tuple(out)

        if self.mogrify is not None:
            y, x = self.mogrify(y, x)

        gated_output = self.gru(
            y.reshape(-1, dim),
            x.reshape(-1, dim)
        )

        gated_output = gated_output.reshape(batch, -1, dim)
        ret = gated_output, *rest
        return ret


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)


class ConvCompress(nn.Module):
    def __init__(self, dim, ratio=4):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, ratio, stride=ratio)

    def forward(self, mem):
        mem = mem.transpose(1, 2)
        compressed_mem = self.conv(mem)
        return compressed_mem.transpose(1, 2)


class DetachedConvCompress(nn.Module):
    def __init__(self, reference: ConvCompress):
        super().__init__()
        self.reference = reference

    def forward(self, mem):
        weight = self.reference.conv.weight.detach()
        bias = self.reference.conv.bias.detach()

        mem = mem.transpose(1, 2)
        compressed_mem = F.conv1d(mem, weight, bias, self.reference.conv.stride,
                                  self.reference.conv.padding, self.reference.conv.dilation,
                                  self.reference.conv.groups)
        return compressed_mem.transpose(1, 2)


# feedforward

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_


class FeedForward(nn.Module):
    def __init__(self, dim, ff_dim, dropout=0., activation=None, glu=False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, ff_dim * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(ff_dim, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x


class CompressionStage(nn.Module):

    def __init__(self, dim, cmem_ratio, cmem_len, attn_heads, attn_dim_heads, reconstruction_attn_dropout,
                 prev_lvl_mem_start_index, prev_lvl_mem_len) -> None:
        super().__init__()
        self.attn_heads = attn_heads  # of the containing SelfAttention object
        self.attn_dim_heads = attn_dim_heads
        self.mem_len_this_lvl = cmem_len
        self.prev_lvl_mem_start_index = prev_lvl_mem_start_index
        self.prev_lvl_mem_len = prev_lvl_mem_len

        assert prev_lvl_mem_len % cmem_ratio == 0, \
            f'mem length of previous level ({prev_lvl_mem_len}) must be divisble by compression ratio ({cmem_ratio})'

        self.reconstruction_attn_dropout = nn.Dropout(reconstruction_attn_dropout)
        self.compress_mem_fn = ConvCompress(dim, cmem_ratio)
        self.compress_mem_fn_without_grad = DetachedConvCompress(self.compress_mem_fn)

    def forward(self, prev_cmem_this_lvl, old_mem_prev_lvl, prev_lvl_mem_len, q, k, v, to_kv_weight):
        compressed_mem = self.compress_mem_fn_without_grad(old_mem_prev_lvl)
        old_cmem, new_cmem = split_at_index(1, -self.mem_len_this_lvl,
                                            torch.cat((prev_cmem_this_lvl, compressed_mem), dim=1))
        aux_loss = torch.zeros(1, requires_grad=True, **to(prev_cmem_this_lvl))

        if not self.training:
            return old_cmem, new_cmem, aux_loss

        # calculate compressed memory auxiliary loss if training
        merge_heads = lambda x: reshape_dim(x, -1, (-1, self.attn_dim_heads)).transpose(1, 2)

        compressed_mem = self.compress_mem_fn(old_mem_prev_lvl.detach())
        cmem_k, cmem_v = F.linear(compressed_mem, to_kv_weight.detach()).chunk(2, dim=-1)
        cmem_k, cmem_v = map(merge_heads, (cmem_k, cmem_v))
        cmem_k, cmem_v = map(lambda x: x.expand(-1, self.attn_heads, -1, -1), (cmem_k, cmem_v))

        old_mem_range = slice(- min(prev_lvl_mem_len, self.prev_lvl_mem_len) - self.prev_lvl_mem_start_index,
                              -self.prev_lvl_mem_start_index)
        old_mem_k, old_mem_v = map(lambda x: x[:, :, old_mem_range].clone(), (k, v))

        q, old_mem_k, old_mem_v = map(torch.detach, (q, old_mem_k, old_mem_v))

        attn_fn = partial(full_attn, dropout_fn=self.reconstruction_attn_dropout)

        aux_loss = F.mse_loss(
            attn_fn(q, old_mem_k, old_mem_v),
            attn_fn(q, cmem_k, cmem_v)
        )

        return old_cmem, new_cmem, aux_loss


# attention.

class SelfAttention(nn.Module):

    @staticmethod
    def validate_cmem_parameters(seq_len: int, mem_len: int,
                                 cmem_lengths: List[int], cmem_ratios: Union[List[int], int]):
        assert len(cmem_lengths) == len(cmem_ratios), f'{cmem_lengths}, {cmem_ratios} should have same length!'
        compression_levels = len(cmem_lengths)
        # compression stage 0 is mem -> cmem
        one_input_block_size = seq_len
        for i in range(compression_levels):
            assert one_input_block_size >= cmem_ratios[i], \
                f'At compression level {i}, one input block of {seq_len} tokens is already reduced to ' \
                f'{one_input_block_size} compressed tokens, cannot be compressed again with ratio {cmem_ratios[i]}'
            assert cmem_lengths[i] >= (one_input_block_size // cmem_ratios[i]), \
                f'length of compressed memory at level {i + 1} should be at least the compressed input block length ' \
                f'at level {i} ({one_input_block_size}) divided by the compression ratio {cmem_ratios[i]}, ' \
                f'i.e. at least {int(one_input_block_size // cmem_ratios[i])}'
            one_input_block_size //= cmem_ratios[i]

        # simulate information flow
        log = ''
        mem = 0
        cmems = [0] * compression_levels
        while True:  # simulate until lt mem would be filled. then, sizes do not change anymore (everything full)
            mem += seq_len
            log += f'i={seq_len} -> '
            if mem <= mem_len:
                log += f'm={mem}\n'
                continue
            old_mem = mem - mem_len
            mem = mem_len
            log += f'm={mem} -> {old_mem}'
            for lvl in range(compression_levels):
                log += f' --/{cmem_ratios[lvl]}--> c{lvl}='
                assert old_mem % cmem_ratios[lvl] == 0, \
                    f'mem length {old_mem} from previous layer not divisible by compression ratio {cmem_ratios[lvl]} ' \
                    f'at compression level {lvl}. Log:\n{log}'
                cmems[lvl] += old_mem // cmem_ratios[lvl]
                if cmems[lvl] <= cmem_lengths[lvl]:
                    log += f'{cmems[lvl]}'
                    old_mem = 0
                    break
                old_mem = cmems[lvl] - cmem_lengths[lvl]
                cmems[lvl] = cmem_lengths[lvl]
                log += f'{cmems[lvl]} -> {old_mem}'
            log += '\n'
            if old_mem > 0:
                break

    def __init__(self, dim, seq_len, mem_len: int,
                 cmem_lengths: List[int], cmem_ratios: Union[List[int], int],
                 use_ltmem=True,
                 heads=8, attn_dropout=0., dropout=0.,
                 reconstruction_attn_dropout=0., one_kv_head=False):
        super().__init__()
        assert (dim % heads) == 0, 'dimension must be divisible by the number of heads'
        if isinstance(cmem_ratios, int):
            cmem_ratios = [cmem_ratios] * len(cmem_lengths)
        SelfAttention.validate_cmem_parameters(seq_len, mem_len, cmem_lengths, cmem_ratios)

        self.heads = heads
        self.dim_head = dim // heads
        self.seq_len = seq_len
        self.mem_len = mem_len
        self.num_cmem_stages = len(cmem_lengths)
        self.cmem_lengths = cmem_lengths
        self.cmem_ratios = cmem_ratios
        self.use_ltmem = use_ltmem
        self.scale = self.dim_head ** (-0.5)

        self.compression_stages = nn.ModuleList()
        running_start_index = self.seq_len
        prev_length = mem_len
        for i in range(self.num_cmem_stages):
            self.compression_stages.append(CompressionStage(
                dim, cmem_ratios[i], cmem_lengths[i], heads, self.dim_head,
                reconstruction_attn_dropout,
                prev_lvl_mem_start_index=running_start_index,
                prev_lvl_mem_len=prev_length))
            prev_length = cmem_lengths[i]
            running_start_index += prev_length

        if self.use_ltmem:
            self.cmem_to_ltmem_query = nn.Parameter(torch.zeros(dim), requires_grad=True)
            self.ltmem_tokv = nn.Linear(dim, dim * 2, bias=False)
            self.recurrence = nn.GRUCell(dim, dim, bias=False)

        self.to_q = nn.Linear(dim, dim, bias=False)

        kv_dim = self.dim_head if one_kv_head else dim
        self.to_kv = nn.Linear(dim, kv_dim * 2, bias=False)
        self.to_out = nn.Linear(dim, dim)

        self.attn_dropout = nn.Dropout(attn_dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, memories=None, pos_emb=None, input_mask=None, calc_memory=True, **kwargs):
        b, t, e, h, dim_h = *x.shape, self.heads, self.dim_head

        memories: Memory = default(memories, (None, None, None))
        mem, cmems, ltmem = memories

        init_empty_mem = lambda: torch.empty(b, 0, e, **to(x))
        mem = default(mem, init_empty_mem)
        cmems = default(cmems, lambda: [init_empty_mem() for i in range(self.num_cmem_stages)])
        ltmem = default(ltmem, init_empty_mem)

        mem_len = mem.shape[1]
        cmem_len_sum = sum(cmem.shape[1] for cmem in cmems)
        ltmem_len = ltmem.shape[1]
        assert 0 <= ltmem_len <= 1, str(ltmem)

        q = self.to_q(x)

        if self.num_cmem_stages == 0:
            kv_input = torch.cat((ltmem, mem, x), dim=1)
        else:
            kv_input = torch.cat((ltmem, *cmems, mem, x), dim=1)
        kv_len = kv_input.shape[1]
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        merge_heads = lambda x: reshape_dim(x, -1, (-1, dim_h)).transpose(1, 2)
        q, k, v = map(merge_heads, (q, k, v))

        k, v = map(lambda x: x.expand(-1, h, -1, -1), (k, v))

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
        mask_value = max_neg_value(dots)

        if pos_emb is not None:
            pos_emb = pos_emb[:, -kv_len:].type(q.dtype)
            pos_dots = torch.einsum('bhid,hjd->bhij', q, pos_emb) * self.scale
            pos_dots = shift(pos_dots)
            dots = dots + pos_dots

        if input_mask is not None:
            mask = input_mask[:, None, :, None] * input_mask[:, None, None, :]
            mask = F.pad(mask, [mem_len + cmem_len_sum + ltmem_len, 0], value=True)
            dots.masked_fill_(~mask, mask_value)

        total_mem_len = mem_len + cmem_len_sum + ltmem_len
        mask = torch.ones(t, t + total_mem_len, **to(x)).triu_(diagonal=1 + total_mem_len).bool()
        dots.masked_fill_(mask[None, None, ...], mask_value)

        attn = dots.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = out.transpose(1, 2).reshape(b, t, -1)
        logits = self.to_out(out)
        logits = self.dropout(logits)

        new_mem = mem
        new_cmems = cmems
        new_ltmem = ltmem
        aux_loss = torch.zeros(1, requires_grad=False, **to(q))

        if self.seq_len > t or not calc_memory:
            return logits, Memory(new_mem, new_cmems, new_ltmem), aux_loss

        # calculate memory and compressed memory

        old_mem, new_mem = queue_fifo(mem, x, length=self.mem_len, dim=1)
        old_mem_padding = old_mem.shape[1] % self.cmem_ratios[0]

        if old_mem_padding != 0:
            old_mem = F.pad(old_mem, [0, 0, old_mem_padding, 0], value=0.)

        if old_mem.shape[1] == 0 or self.num_cmem_stages <= 0:
            return logits, Memory(new_mem, new_cmems, new_ltmem), aux_loss

        prev_mem_len = mem_len
        old_mem_prev_lvl = old_mem
        for i in range(self.num_cmem_stages):
            if old_mem_prev_lvl.size(1) == 0:
                break
            old_mem_prev_lvl, new_cmems[i], lvl_aux_loss = self.compression_stages[i](
                prev_cmem_this_lvl=cmems[i],
                old_mem_prev_lvl=old_mem_prev_lvl,
                prev_lvl_mem_len=prev_mem_len,
                q=q, k=k, v=v,
                to_kv_weight=self.to_kv.weight
            )
            aux_loss += lvl_aux_loss
            prev_mem_len = cmems[i].size(1)

        if old_mem_prev_lvl.size(1) > 0 and self.use_ltmem:
            old_cmem_k, old_cmem_v = (self.ltmem_tokv(old_mem_prev_lvl)
                                      .unsqueeze(dim=1)  # Insert fake head dimension
                                      .chunk(2, dim=-1))
            to_ltmem_query = self.cmem_to_ltmem_query.expand(b, 1, 1, e)  # b x 1(=h) x 1(=seq) x e
            ltmem_update = full_attn(to_ltmem_query, old_cmem_k, old_cmem_v)
            if ltmem_len > 0:
                new_ltmem = self.recurrence(ltmem_update.view(b, e), ltmem.squeeze(dim=1)).unsqueeze(dim=1)
            else:
                new_ltmem = ltmem_update.squeeze(dim=1)  # Remove heads dimension

        return logits, Memory(new_mem, new_cmems, new_ltmem), aux_loss


# transformer

class CompressiveTransformer(nn.Module):
    def __init__(self, num_tokens, dim, seq_len, depth, emb_dim=None,
                 memory_layers=None, mem_len=None,
                 cmem_lengths: List[int] = None, cmem_ratios: Union[int, List[int]] = 4,
                 use_ltmem=True,
                 heads=8, gru_gated_residual=True, mogrify_gru=False, attn_dropout=0.,
                 ff_glu=False, ff_dim=None, ff_dropout=0.,
                 attn_layer_dropout=0., reconstruction_attn_dropout=0., reconstruction_loss_weight=1.,
                 one_kv_head=False):
        super().__init__()
        if isinstance(cmem_ratios, int):
            if cmem_lengths is None:
                cmem_ratios = [cmem_ratios]
            else:
                cmem_ratios = [cmem_ratios] * len(cmem_lengths)
        else:
            assert cmem_lengths is not None
            assert len(cmem_lengths) == len(cmem_ratios)

        ff_dim = default(ff_dim, dim * 4)
        emb_dim = default(emb_dim, dim)
        mem_len = default(mem_len, seq_len)
        cmem_lengths = default(cmem_lengths, [mem_len // cmem_ratios[0]])
        memory_layers = default(memory_layers, list(range(1, depth + 1)))

        assert mem_len >= seq_len, 'length of memory should be at least the sequence length'
        assert all(
            [0 < layer <= depth for layer in memory_layers]), 'one of the indicated memory layers is invalid'

        self.seq_len = seq_len

        self.depth = depth
        self.memory_layers = list(memory_layers)
        self.num_cmem_stages = len(cmem_lengths)

        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.to_model_dim = nn.Identity() if emb_dim == dim else nn.Linear(emb_dim, dim)

        seq_and_mem_len = seq_len + mem_len + sum(cmem_lengths) + (1 if use_ltmem else 0)  # + 1 for LT Memory
        self.pos_emb = nn.Parameter(torch.zeros(heads, seq_and_mem_len, dim // heads), requires_grad=True)

        self.to_logits = nn.Sequential(
            nn.Identity() if emb_dim == dim else nn.Linear(dim, emb_dim),
            nn.Linear(emb_dim, num_tokens)
        )

        wrapper = partial(GRUGating, dim, mogrify=mogrify_gru) if gru_gated_residual else Residual

        self.attn_layers = nn.ModuleList([
            wrapper(PreNorm(dim, SelfAttention(
                dim, seq_len, mem_len,
                cmem_lengths if (i + 1) in memory_layers else [],
                cmem_ratios if (i + 1) in memory_layers else [],
                use_ltmem and (i + 1) in memory_layers,
                heads, dropout=attn_layer_dropout,
                attn_dropout=attn_dropout,
                reconstruction_attn_dropout=reconstruction_attn_dropout,
                one_kv_head=one_kv_head
            ))) for i in range(depth)])
        self.ff_layers = nn.ModuleList(
            [wrapper(PreNorm(dim, FeedForward(dim, ff_dim, dropout=ff_dropout, glu=ff_glu))) for _ in range(depth)])

        self.reconstruction_loss_weight = reconstruction_loss_weight

    def forward(self, x, memories=None, mask=None):
        input_device = x.device
        x = self.token_emb(x)
        x = self.to_model_dim(x)
        b, t, d = x.shape

        assert t <= self.seq_len, f'input contains a sequence length {t} that is greater than the designated maximum ' \
                                  f'sequence length {self.seq_len} '

        memories = default(memories, (None, None, None))
        mem, cmems, ltmem = memories

        num_memory_layers = len(self.memory_layers)
        init_empty_mem = lambda: torch.empty(num_memory_layers, b, 0, d, **to(x))
        mem = default(mem, init_empty_mem)
        cmems = default(cmems, lambda: [init_empty_mem() for i in range(self.num_cmem_stages)])
        ltmem = default(ltmem, init_empty_mem)

        total_len = mem.shape[2] + sum(cmem.shape[2] for cmem in cmems) + ltmem.shape[2] + self.seq_len
        pos_emb = self.pos_emb[:, (self.seq_len - t):total_len]

        # Lists of {c,lt,}mem per transformer layer
        next_mem = []
        next_cmems = []
        next_ltmem = []
        aux_loss = torch.tensor(0., requires_grad=True, **to(x))

        mem_iter, ltmem_iter = map(iterate_tensor, (mem, ltmem))
        cmems_iter = ([cmem[i] for cmem in cmems] for i in range(num_memory_layers))

        for ind in range(self.depth):
            x, mem_out, cmems_out, ltmem_out, layer_aux_loss \
                = self._pass_through_layer(ind, mem_iter, cmems_iter, ltmem_iter, mask, pos_emb, x)
            aux_loss = aux_loss + layer_aux_loss

            if (ind + 1) not in self.memory_layers:
                continue

            next_mem.append(mem_out)
            next_cmems.append(cmems_out)
            next_ltmem.append(ltmem_out)

        out = self.to_logits(x)

        next_mem, next_ltmem = map(torch.stack, (next_mem, next_ltmem))
        next_cmems = [torch.stack([next_cmems[layer][cstage] for layer in range(num_memory_layers)])
                      for cstage in range(self.num_cmem_stages)]

        aux_loss = aux_loss * self.reconstruction_loss_weight / num_memory_layers
        out = out.to(device=input_device)
        return out, Memory(mem=next_mem, compressed_mem=next_cmems, lt_mem=next_ltmem), aux_loss

    def _pass_through_layer(self, ind, mem_iter, cmems_iter, ltmem_iter, mask, pos_emb, x):
        attn = self.attn_layers[ind]
        ff = self.ff_layers[ind]

        layer_num = ind + 1
        use_memory = layer_num in self.memory_layers
        memories = (next(mem_iter), next(cmems_iter), next(ltmem_iter)) if use_memory else None
        _dev = lambda t: t.to(device=x.device)
        memories = (_dev(memories[0]), [_dev(m) for m in memories[1]], _dev(memories[2])) if memories else None

        x, (mem_out, cmems_out, ltmem_out), layer_aux_loss = attn(x, memories=memories, calc_memory=use_memory,
                                                                  input_mask=mask, pos_emb=pos_emb)
        x, = ff(x)

        return x, mem_out, cmems_out, ltmem_out, layer_aux_loss


class MultiDeviceCompressiveTransformer(CompressiveTransformer):
    """
    CompressiveTransformer with model parallelism.
    Note: Start fairseq-train with
    --distributed-no-spawn
    --distributed-world-size 1
    to prevent data parallelism
    """

    def __init__(self, num_tokens, dim, seq_len, depth, emb_dim=None, memory_layers=None, mem_len=None,
                 cmem_lengths: List[int] = None, cmem_ratios: Union[int, List[int]] = 4, use_ltmem=True, heads=8,
                 gru_gated_residual=True, mogrify_gru=False, attn_dropout=0., ff_glu=False, ff_dim=None, ff_dropout=0.,
                 attn_layer_dropout=0., reconstruction_attn_dropout=0., reconstruction_loss_weight=1.,
                 one_kv_head=False,
                 layers_to_gpus=None):
        super().__init__(num_tokens, dim, seq_len, depth, emb_dim, memory_layers, mem_len, cmem_lengths, cmem_ratios,
                         use_ltmem, heads, gru_gated_residual, mogrify_gru, attn_dropout, ff_glu, ff_dim, ff_dropout,
                         attn_layer_dropout, reconstruction_attn_dropout, reconstruction_loss_weight, one_kv_head)

        gpus = torch.cuda.device_count()
        layers_to_gpus = default(layers_to_gpus, [int(i / self.depth * gpus) for i in range(self.depth)])
        assert len(layers_to_gpus) == self.depth
        assert all(0 <= x < gpus for x in layers_to_gpus)
        self.layers_to_gpus = layers_to_gpus

    def cuda(self, device=None):
        # pos_emb, token_emb, to_model_dim and to_logits always stays on device 0
        self.pos_emb = nn.Parameter(self.pos_emb.cuda(), requires_grad=True)
        self.token_emb.to(device=0)
        self.to_model_dim.to(device=0)
        self.to_logits.to(device=torch.cuda.device_count() - 1)
        for i in range(self.depth):
            self.attn_layers[i].to(device=self.layers_to_gpus[i])
            self.ff_layers[i].to(device=self.layers_to_gpus[i])
        return self

    def _apply(self, fn):
        fake = torch.empty(0)
        if fn(fake).device.type == 'cuda' and fn(fake).device != fake.device:
            return self.cuda()
        else:
            # noinspection PyProtectedMember
            return super()._apply(fn)

    def _pass_through_layer(self, ind, mem_iter, cmems_iter, ltmem_iter, mask, pos_emb, x):
        gpu = self.layers_to_gpus[ind]

        x = x.to(device=gpu)
        pos_emb = pos_emb.to(device=gpu)
        mask = mask.to(device=gpu) if mask else None
        x, mem_out, cmems_out, ltmem_out, layer_aux_loss = super()._pass_through_layer(
            ind, mem_iter, cmems_iter, ltmem_iter, mask, pos_emb, x)

        mem_out = mem_out.to(device=0) if mem_out is not None else None
        cmems_out = [m.to(device=0) for m in cmems_out] if cmems_out is not None else None
        ltmem_out = ltmem_out.to(device=0) if ltmem_out is not None else None
        layer_aux_loss = layer_aux_loss.to(device=0) if layer_aux_loss is not None else None

        return x, mem_out, cmems_out, ltmem_out, layer_aux_loss
