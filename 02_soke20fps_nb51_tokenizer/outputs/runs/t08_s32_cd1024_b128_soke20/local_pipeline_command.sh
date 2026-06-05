#!/usr/bin/env bash
set -euo pipefail
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:${LD_LIBRARY_PATH:-}
BASE=/home/cem/tez/exp/new_data/02_soke20fps_nb51_tokenizer
OUT="$BASE/outputs"
PRE="$OUT/preprocess_soke20"
RUN="$OUT/runs/t08_s32_cd1024_b128_soke20"
DATA=/home/cem/tez/exp/DATA/SOKE_DATA
INDEX="$BASE/combined_clip_index.csv"
cd "$BASE"
if [ ! -s "$PRE/train_manifest.csv" ] || [ ! -s "$PRE/val_manifest.csv" ]; then
  echo "[pipeline] starting preprocess $(date -Is)"
  python scripts/soke20_preprocess.py \
    --data-root "$DATA" \
    --out-root "$PRE" \
    --index-csv "$INDEX" \
    --target-fps 20 \
    --window-size 64 \
    --max-duration-sec 30 \
    --progress-every 250
  echo "[pipeline] preprocess done $(date -Is)"
else
  echo "[pipeline] preprocess manifests already exist; skipping $(date -Is)"
fi

echo "[pipeline] starting training $(date -Is)"
python scripts/train_nb51_soke20.py \
  --preprocess-root "$PRE" \
  --run-root "$RUN" \
  --epochs 300 \
  --batch-size 32 \
  --grad-accum 1 \
  --num-workers 2 \
  --num-bins 128 \
  --window-size 64 \
  --d-model 256 \
  --num-heads 8 \
  --num-kv-heads 4 \
  --num-temporal-latents 8 \
  --num-spatial-latents 32 \
  --code-dim 1024 \
  --code-num 512
