"""Substituto do flash-attn 2 via SDPA do PyTorch.

O flash-attn 2.7.4 não tem pacote para ARM + Blackwell (GB10). O LongCat só usa
flash_attn_func(q, k, v, dropout_p, softmax_scale) com layout [B, S, H, D].
"""
import torch.nn.functional as F


def flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=None, causal=False, **_):
    q, k, v = (t.transpose(1, 2) for t in (q, k, v))  # -> [B, H, S, D]
    out = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=causal, scale=softmax_scale)
    return out.transpose(1, 2)  # -> [B, S, H, D]
