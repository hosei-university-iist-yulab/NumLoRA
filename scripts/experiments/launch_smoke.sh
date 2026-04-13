#!/bin/bash
# =============================================================================
# NumLoRA — Smoke-tier experiments (validation only)
#
# Purpose: Verify pipeline runs end-to-end without errors.
# Scope:   3 pilot datasets x 1 MR x 1 seed x {LoRA, NumLoRA} = 6 runs
# Time:    ~10-15 min on RTX 3090
# GPUs:    CUDA 4-7 (4 concurrent GPUs, shared server)
#
# Usage:   bash scripts/experiments/launch_smoke.sh
# =============================================================================
set -e

DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$DIR"

source /home/Aboya_25R9803/anaconda3/etc/profile.d/conda.sh
conda activate llms

TIER="smoke"
EPOCHS=5
LOGDIR="$DIR/results/$TIER/logs"
OUTDIR="$DIR/results/$TIER"
mkdir -p "$LOGDIR" "$OUTDIR"

PILOT_DATASETS="physionet2012 solar_alabama exchange_rate"
METHODS="lora_r8 numlora_full"
MR="0.3"
SEED="42"
GPU=4

echo "============================================================"
echo "NumLoRA Smoke-Tier — $EPOCHS epochs, 1 seed, 1 MR"
echo "GPU: CUDA $GPU"
echo "============================================================"

for ds in $PILOT_DATASETS; do
    for method in $METHODS; do
        OUT="$OUTDIR/${method}_${ds}_mr${MR}_seed${SEED}.json"
        LOG="$LOGDIR/${method}_${ds}_mr${MR}_seed${SEED}.log"

        if [ -f "$OUT" ]; then
            echo "[SKIP] $OUT already exists"
            continue
        fi

        echo "[RUN] method=$method dataset=$ds mr=$MR seed=$SEED"
        # TODO: uncomment when train.py is implemented
        # CUDA_VISIBLE_DEVICES=$GPU python "$DIR/scripts/experiments/train.py" \
        #     --method "$method" \
        #     --dataset "$ds" \
        #     --missing-rate "$MR" \
        #     --seed "$SEED" \
        #     --epochs "$EPOCHS" \
        #     --tier "$TIER" \
        #     --output "$OUT" > "$LOG" 2>&1
    done
done

echo ""
echo "Smoke tier complete. Results: $(ls $OUTDIR/*.json 2>/dev/null | wc -l) files in $OUTDIR/"
