"""Substituto do flash-attn 3 (devolve tupla como o original: out, lse)."""
from flash_attn import flash_attn_func as _fa2


def flash_attn_func(q, k, v, softmax_scale=None, causal=False, **_):
    return _fa2(q, k, v, softmax_scale=softmax_scale, causal=causal), None
