"""Substituto do flash-attn 2 via SDPA do PyTorch.

O flash-attn 2.7.4 não tem pacote para ARM + Blackwell (GB10). O LongCat usa
flash_attn_func (layout [B, S, H, D]) e flash_attn_varlen_func (layout [total, H, D]
com cu_seqlens), sempre sem dropout no inference.
"""
import torch
import torch.nn.functional as F


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **_):
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # -> [B, H, S, D]
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
    return out.transpose(1, 2)  # -> [B, S, H, D]


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None,
                           dropout_p=0.0, softmax_scale=None, causal=False, **_):
    cq, ck = cu_seqlens_q.tolist(), cu_seqlens_k.tolist()
    saidas = []
    for i in range(len(cq) - 1):
        qi = q[cq[i]:cq[i + 1]].unsqueeze(0)  # [1, Sq, H, D]
        ki = k[ck[i]:ck[i + 1]].unsqueeze(0)
        vi = v[ck[i]:ck[i + 1]].unsqueeze(0)
        saidas.append(flash_attn_func(qi, ki, vi, dropout_p, softmax_scale, causal)[0])
    return torch.cat(saidas, dim=0)  # [total_q, H, D]
