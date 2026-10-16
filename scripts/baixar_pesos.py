#!/usr/bin/env python3
"""Baixa só o que o Avatar 1.5 (int8 + distill) usa: ~43 GB em vez de ~158 GB.

Do LongCat-Video base: tokenizer, text_encoder, vae, scheduler (o DiT de 54 GB não é usado).
Do Avatar-1.5: base_model_int8, lora, whisper-large-v3 (só model.safetensors), vocal_separator.
Retoma se for interrompido.
"""
import os
import sys
from huggingface_hub import snapshot_download

DESTINO = os.environ.get("LONGCAT_PESOS", "/mnt/hd8t/longcat-weights")

ALVOS = [
    ("meituan-longcat/LongCat-Video", "LongCat-Video",
     ["tokenizer/*", "text_encoder/*", "vae/*", "scheduler/*", "config.json", "model_index.json"]),
    ("meituan-longcat/LongCat-Video-Avatar-1.5", "LongCat-Video-Avatar-1.5",
     ["base_model_int8/*", "lora/*", "scheduler/*", "vocal_separator/*", "config.json", "model_index.json",
      "whisper-large-v3/*.json", "whisper-large-v3/*.txt", "whisper-large-v3/model.safetensors"]),
]

for repo, pasta, padroes in ALVOS:
    print(f"== {repo} -> {DESTINO}/{pasta}", flush=True)
    snapshot_download(repo, local_dir=os.path.join(DESTINO, pasta), allow_patterns=padroes, max_workers=4)

print("BAIXOU-TUDO", flush=True)
sys.exit(0)
