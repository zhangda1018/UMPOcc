#!/usr/bin/env bash
set -euo pipefail
if [[ $# -lt 3 ]]; then
    echo "Usage: $0 CONFIG CHECKPOINT GPUS [evaluation options...]" >&2
    exit 2
fi
CONFIG=$1
CHECKPOINT=$2
GPUS=$3
shift 3
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$REPO_DIR:${PYTHONPATH:-}"
exec "${PYTHON_BIN:-python}" -m torch.distributed.launch \
    --nproc_per_node="$GPUS" --master_port="${PORT:-29503}" \
    "$REPO_DIR/tools/test.py" "$CONFIG" "$CHECKPOINT" \
    --launcher pytorch --eval bbox "$@"
