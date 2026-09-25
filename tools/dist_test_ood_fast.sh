#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 CONFIG CHECKPOINT 1 [OOD evaluation options...]" >&2
    exit 2
fi
CONFIG=$1
CHECKPOINT=$2
GPUS=$3
shift 3
if [[ "$GPUS" != 1 ]]; then
    echo "Fast OOD evaluation supports one GPU. Set CUDA_VISIBLE_DEVICES to select it." >&2
    exit 2
fi
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
exec "${PYTHON_BIN:-python}" "$REPO_DIR/tools/test_ood_fast.py" \
    "$CONFIG" "$CHECKPOINT" "$@"
