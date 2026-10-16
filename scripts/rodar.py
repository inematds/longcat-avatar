#!/usr/bin/env python3
"""Roda o demo de avatar do upstream com remendos de memória para o GB10 (memória unificada).

Uso: python scripts/rodar.py <script-do-upstream.py> [args do script...]  (chamado pelo avatar.sh via torchrun)

Remendo 1 — load_quantized_dit: o original monta o DiT inteiro em fp32 (nn.Linear cheios, ~50 GB)
só para trocar os Linear por QuantizedLinear logo depois, e carrega os pesos na CPU antes de
mandar para a GPU. Aqui o esqueleto nasce no device "meta" (0 bytes) e os pesos int8 vão direto
para a GPU com load_state_dict(assign=True). Todos os buffers do modelo são persistentes e o
load é strict, então nada fica em "meta".
"""
import json
import os
import runpy
import sys

import torch
import torch.nn as nn
from safetensors.torch import load_file

import longcat_video.modules.quantization as q


def load_quantized_dit_enxuto(checkpoint_dir, subfolder="base_model_int8", **kwargs):
    from longcat_video.modules.avatar.longcat_video_dit_avatar import LongCatVideoAvatarTransformer3DModel

    pasta = os.path.join(checkpoint_dir, subfolder)
    config = json.load(open(os.path.join(pasta, "config.json")))
    for k in ("_class_name", "architectures", "_diffusers_version", "model_max_length"):
        config.pop(k, None)
    config.update(kwargs)

    with torch.device("meta"):
        model = LongCatVideoAvatarTransformer3DModel(**config)
        trocar = {}
        for nome, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and not any(p in nome for p in q.DEFAULT_SKIP_PATTERNS):
                trocar[nome] = q.QuantizedLinear(mod.in_features, mod.out_features, bias=mod.bias is not None)
    for nome, ql in trocar.items():
        pai = model
        *caminho, folha = nome.split(".")
        for parte in caminho:
            pai = getattr(pai, parte)
        setattr(pai, folha, ql)

    destino = f"cuda:{torch.cuda.current_device()}"
    indice = os.path.join(pasta, "quantized_model.safetensors.index.json")
    if os.path.exists(indice):
        arquivos = sorted(set(json.load(open(indice))["weight_map"].values()))
    else:
        arquivos = sorted(f for f in os.listdir(pasta) if f.endswith(".safetensors") and "index" not in f)
    estado = {}
    for f in arquivos:
        estado.update(load_file(os.path.join(pasta, f), device=destino))

    model.load_state_dict(estado, strict=True, assign=True)
    del estado
    model.eval()

    for nome, mod in model.named_modules():
        if isinstance(mod, q.QuantizedLinear):
            continue
        for _, p in mod.named_parameters(recurse=False):
            if p.dtype == torch.float32:
                p.data = p.data.to(torch.bfloat16)

    restos = [n for n, t in list(model.named_parameters()) + list(model.named_buffers()) if t.is_meta]
    if restos:
        raise RuntimeError(f"tensores ainda em meta: {restos[:5]}")
    torch.cuda.empty_cache()
    print(f"[longcat-avatar] DiT int8 carregado direto na GPU ({torch.cuda.memory_allocated() / 2**30:.1f} GiB alocados)", flush=True)
    return model


q.load_quantized_dit = load_quantized_dit_enxuto

script = sys.argv[1]
sys.argv = sys.argv[1:]
sys.path.insert(0, os.path.dirname(os.path.abspath(script)))
runpy.run_path(script, run_name="__main__")
