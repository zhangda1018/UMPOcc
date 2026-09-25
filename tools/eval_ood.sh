#!/usr/bin/env bash
# UMPOcc PPSC preset. Run from the repository root; supply your own checkpoint.
set -euo pipefail
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 CONFIG CHECKPOINT [OOD evaluation options...]" >&2
    exit 2
fi
CONFIG=$1
CHECKPOINT=$2
shift 2
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/dist_test_ood_fast.sh" "$CONFIG" "$CHECKPOINT" 1 \
    --score-mode echoood_ppsc_v3 \
    --tadc-spatial-kernel 7 --tadc-spatial-alpha 0.75 --tadc-peak-gamma 2.0 \
    --ppsc-v2-boost 0.25 --ppsc-v2-suppress 0.0 \
    --ppsc-v3-min-support 0.0 --ppsc-v3-support-scale 0.2 \
    --ppsc-v3-support-power 1.0 \
    --ood-dilation-radius 6 --ood-dilation-radii 4 5 6 \
    --metric-histogram-bins 65536 --workers 2 --chunk-size 32768 \
    --deterministic "$@"
