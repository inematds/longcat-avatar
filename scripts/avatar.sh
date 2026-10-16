#!/bin/bash
# Foto + áudio -> vídeo falante (LongCat-Video-Avatar 1.5, int8 + distill, 1 GPU).
# Uso: scripts/avatar.sh <entrada.json> [segmentos=1] [480p|720p] [nome-da-saida]
#   1 segmento = 93 quadros = 3,7 s a 25 fps; cada segmento extra soma 3,2 s.
# O json segue o formato do upstream: {"prompt": ..., "cond_image": ..., "cond_audio": {"person1": ...}}
# (caminhos relativos ao próprio json).
set -e
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
LONGCAT_DIR="${LONGCAT_DIR:-$HOME/projetos/LongCat-Video}"
PESOS="${LONGCAT_PESOS:-/mnt/hd8t/longcat-weights}"
TETO="${LONGCAT_MEMORIA:-90G}"

ENTRADA="$(realpath "$1")"; SEG="${2:-1}"; RES="${3:-480p}"
NOME="${4:-$(basename "$ENTRADA" .json)-$(date +%Y%m%d-%H%M%S)}"
SAIDA="$RAIZ/saidas/$NOME"; mkdir -p "$SAIDA"

# o upstream lê caminhos relativos ao cwd: reescreve o json com caminhos absolutos
"$RAIZ/.venv/bin/python" - "$ENTRADA" "$SAIDA/entrada.json" <<'EOF'
import json, os, sys
src, dst = sys.argv[1], sys.argv[2]
d = json.load(open(src, encoding="utf-8")); base = os.path.dirname(src)
ab = lambda p: p if os.path.isabs(p) else os.path.join(base, p)
d["cond_image"] = ab(d["cond_image"])
d["cond_audio"] = {k: ab(v) for k, v in d["cond_audio"].items()}
json.dump(d, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
EOF

EXTRA=()
[ "$SEG" -gt 1 ] && EXTRA=(--num_segments="$SEG" --ref_img_index=10 --mask_frame_range=3)

cd "$SAIDA"
INICIO=$(date +%s)
systemd-run --user --scope -q -p MemoryMax="$TETO" -p MemorySwapMax=0 \
  "$RAIZ/.venv/bin/torchrun" --nproc_per_node=1 "$LONGCAT_DIR/run_demo_avatar_single_audio_to_video.py" \
    --context_parallel_size=1 --checkpoint_dir="$PESOS/LongCat-Video-Avatar-1.5" \
    --stage_1=ai2v --input_json="$SAIDA/entrada.json" --output_dir="$SAIDA" --resolution="$RES" \
    --use_distill --model_type avatar-v1.5 --use_int8 "${EXTRA[@]}" 2>&1 | tee "$SAIDA/log.txt"
echo "tempo_total_s=$(( $(date +%s) - INICIO ))" | tee -a "$SAIDA/log.txt"
ls -la "$SAIDA"/*.mp4
