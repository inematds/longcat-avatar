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
# GB10 = memória unificada: o que o CUDA aloca não entra no MemoryMax do cgroup.
# Trava real: mata a geração se o sistema ficar com menos de LONGCAT_LIVRE_MIN_MB livres.
LIVRE_MIN="${LONGCAT_LIVRE_MIN_MB:-10240}"
(
  while sleep 3; do
    PID=$(pgrep -f "run_demo_avatar_single_audio_to_video.py --context" | head -1)
    LIVRE=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
    RSS=$( [ -n "$PID" ] && awk '/VmRSS/{print int($2/1024)}' /proc/$PID/status 2>/dev/null )
    echo "$(( $(date +%s) - INICIO )) livre_mb=$LIVRE rss_mb=${RSS:-0}" >> "$SAIDA/memoria.txt"
    if [ "$LIVRE" -lt "$LIVRE_MIN" ]; then
      echo "ABORTADO: memória livre ${LIVRE} MB < ${LIVRE_MIN} MB" | tee -a "$SAIDA/log.txt"
      pkill -9 -f "run_demo_avatar_single_audio_to_video.py --context"; break
    fi
  done
) &
GUARDA=$!
trap 'kill $GUARDA 2>/dev/null' EXIT
MALLOC_ARENA_MAX=2 systemd-run --user --scope -q -p MemoryMax="$TETO" -p MemorySwapMax=0 \
  "$RAIZ/.venv/bin/torchrun" --nproc_per_node=1 "$RAIZ/scripts/rodar.py" "$LONGCAT_DIR/run_demo_avatar_single_audio_to_video.py" \
    --context_parallel_size=1 --checkpoint_dir="$PESOS/LongCat-Video-Avatar-1.5" \
    --stage_1=ai2v --input_json="$SAIDA/entrada.json" --output_dir="$SAIDA" --resolution="$RES" \
    --use_distill --model_type avatar-v1.5 --use_int8 "${EXTRA[@]}" 2>&1 | tee "$SAIDA/log.txt"
echo "tempo_total_s=$(( $(date +%s) - INICIO ))" | tee -a "$SAIDA/log.txt"
echo "pico: $(sort -t= -k2 -n "$SAIDA/memoria.txt" | head -1)  rss_max_mb=$(grep -o 'rss_mb=[0-9]*' "$SAIDA/memoria.txt" | cut -d= -f2 | sort -n | tail -1)" | tee -a "$SAIDA/log.txt"
ls -la "$SAIDA"/*.mp4
