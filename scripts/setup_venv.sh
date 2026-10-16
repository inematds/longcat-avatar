#!/bin/bash
# venv do longcat-avatar: reusa o torch cu130 do sistema e põe compat/ (shim do flash-attn) no caminho.
set -e
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
LONGCAT_DIR="${LONGCAT_DIR:-$HOME/projetos/LongCat-Video}"
cd "$RAIZ"
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -q "transformers==4.51.3" "diffusers==0.35.1" einops ftfy loguru av \
  opencv-python-headless imageio imageio-ffmpeg librosa soundfile soxr pyloudnorm "audio-separator==0.30.2"
SITE=$(.venv/bin/python -c "import site;print(site.getsitepackages()[0])")
printf '%s\n%s\n' "$RAIZ/compat" "$LONGCAT_DIR" > "$SITE/longcat_avatar.pth"
.venv/bin/python -c "import torch, flash_attn, longcat_video; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
