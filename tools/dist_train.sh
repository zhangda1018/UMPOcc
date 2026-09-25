#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 CONFIG GPUS [training options...]" >&2
    exit 2
fi
CONFIG=$1
GPUS=$2
shift 2
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
exec "${PYTHON_BIN:-python}" -m torch.distributed.launch \
    --nproc_per_node="$GPUS" --master_port="${PORT:-28509}" \
    "$REPO_DIR/tools/train.py" "$CONFIG" --launcher pytorch \
    --deterministic "$@"
