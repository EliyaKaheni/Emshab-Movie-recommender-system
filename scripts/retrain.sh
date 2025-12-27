#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/eliya/Documents/codes/Emshab"
PY="/home/eliya/anaconda3/bin/python"

mkdir -p "$PROJECT_DIR/logs"
cd "$PROJECT_DIR"

# جلوگیری از اجرای همزمان
flock -n /tmp/emshab_retrain.lock \
  "$PY" train_svd_model.py >> logs/retrain.log 2>&1

# به اپ علامت بده مدل جدید لود شود
touch models/reload.flag
