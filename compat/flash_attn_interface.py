"""Substituto do flash-attn 3 (devolve tupla como o original: out, lse)."""
from flash_attn import flash_attn_func as _fa2, flash_attn_varlen_func as _fa2_varlen


def flash_attn_func(q, k, v, softmax_scale=None, causal=False, **_):
    return _fa2(q, k, v, softmax_scale=softmax_scale, causal=causal), None


def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q=None, max_seqlen_k=None,
                           softmax_scale=None, causal=False, **_):
    return _fa2_varlen(q, k, v, cu_seqlens_q, cu_seqlens_k, softmax_scale=softmax_scale, causal=causal), None
