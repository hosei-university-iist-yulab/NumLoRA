#!/bin/bash
# =============================================================================
# NumLoRA — Ablation studies (3 dimensions)
#
# 1. Component ablation: 7 subsets of {SSR, MAI, CTGS}
#    3 datasets x 3 MRs x 3 seeds x 7 variants = 189 runs
#
# 2. Rank ablation: r = {2, 4, 8, 16, 32}
#    3 datasets x MR=0.3 x 3 seeds x 5 ranks x 2 methods = 90 runs
#
# 3. LR ratio ablation: lr_ssr/lr = {1x, 3x, 10x}
#    3 datasets x MR=0.3 x 3 seeds x 3 ratios = 27 runs
#
# Total: ~306 runs. GPUs 4-7, 3 concurrent per GPU.
# ETA: ~60-90 min
#
# Usage:
#   bash scripts/experiments/launch_ablation.sh component
#   bash scripts/experiments/launch_ablation.sh rank
#   bash scripts/experiments/launch_ablation.sh lr
#   bash scripts/experiments/launch_ablation.sh all
# =============================================================================
set -e

DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$DIR"

source /home/Aboya_25R9803/anaconda3/etc/profile.d/conda.sh
conda activate llms

PHASE="${1:-all}"
BACKBONE="smollm_360m"
EPOCHS=100
PATIENCE=20
LOGDIR="$DIR/results/full/logs"
OUTDIR="$DIR/results/full/ablation"
mkdir -p "$LOGDIR" "$OUTDIR"

ABL_DATASETS="ett_h1 exchange weather"
SEEDS="42 123 456"
GPUS=(4 5 6 7)
NUM_GPUS=${#GPUS[@]}

dispatch_jobs() {
    local jobfile=$1 parallel=$2 label=$3
    local n=$(wc -l < "$jobfile")
    echo "$label: $n jobs, $parallel concurrent/GPU"
    for ((i=0; i<NUM_GPUS; i++)); do
        local gpu=${GPUS[$i]}
        awk "NR % $NUM_GPUS == $i" "$jobfile" > "/tmp/numlora_abl_gpu${gpu}.txt"
        cat "/tmp/numlora_abl_gpu${gpu}.txt" | xargs -P "$parallel" -I {} bash -c "CUDA_VISIBLE_DEVICES=$gpu {}" &
    done
    wait
}

# ── Component ablation ──
run_component() {
    echo "=== Component Ablation ==="
    JOBFILE="/tmp/numlora_abl_component.txt"
    > "$JOBFILE"

    VARIANTS="numlora_mai_only numlora_ssr_only numlora_ctgs_only numlora_mai_ssr numlora_mai_ctgs numlora_ssr_ctgs numlora_full"

    for ds in $ABL_DATASETS; do
        for mr in 0.1 0.3 0.5; do
            for seed in $SEEDS; do
                for method in $VARIANTS; do
                    EXTRA=""
                    [ "$ds" = "ili" ] && EXTRA="--window-size 24 --patch-size 8"

                    OUT="$OUTDIR/${method}_${ds}_mr${mr}_seed${seed}.json"
                    LOG="$LOGDIR/abl_${method}_${ds}_mr${mr}_seed${seed}.log"
                    [ -f "$OUT" ] && continue

                    echo "python $DIR/scripts/experiments/train.py --method $method --backbone $BACKBONE --dataset $ds --epochs $EPOCHS --seed $seed --missing-rate $mr --patience $PATIENCE $EXTRA --output $OUT > $LOG 2>&1" >> "$JOBFILE"
                done
            done
        done
    done

    dispatch_jobs "$JOBFILE" 3 "Component ablation"
}

# ── Rank ablation ──
run_rank() {
    echo "=== Rank Ablation ==="
    JOBFILE="/tmp/numlora_abl_rank.txt"
    > "$JOBFILE"

    for ds in $ABL_DATASETS; do
        for seed in $SEEDS; do
            for rank in 2 4 8 16 32; do
                for method_base in lora numlora; do
                    EXTRA=""
                    [ "$ds" = "ili" ] && EXTRA="--window-size 24 --patch-size 8"

                    if [ "$method_base" = "lora" ]; then
                        METHOD="lora_r8"  # train.py needs method name; we override rank via config
                    else
                        METHOD="numlora_full"
                    fi

                    OUT="$OUTDIR/rank_${method_base}_r${rank}_${ds}_mr0.3_seed${seed}.json"
                    LOG="$LOGDIR/abl_rank_${method_base}_r${rank}_${ds}_seed${seed}.log"
                    [ -f "$OUT" ] && continue

                    # TODO: train.py needs --rank flag to override default r=8
                    echo "echo '[TODO] rank ablation: $method_base r=$rank $ds seed=$seed'" >> "$JOBFILE"
                done
            done
        done
    done

    dispatch_jobs "$JOBFILE" 3 "Rank ablation"
}

# ── LR ratio ablation ──
run_lr() {
    echo "=== LR Ratio Ablation ==="
    JOBFILE="/tmp/numlora_abl_lr.txt"
    > "$JOBFILE"

    for ds in $ABL_DATASETS; do
        for seed in $SEEDS; do
            for lr_ssr in 1e-3 3e-3 1e-2; do
                EXTRA=""
                [ "$ds" = "ili" ] && EXTRA="--window-size 24 --patch-size 8"

                OUT="$OUTDIR/lr_numlora_lrssr${lr_ssr}_${ds}_mr0.3_seed${seed}.json"
                LOG="$LOGDIR/abl_lr_lrssr${lr_ssr}_${ds}_seed${seed}.log"
                [ -f "$OUT" ] && continue

                echo "python $DIR/scripts/experiments/train.py --method numlora_full --backbone $BACKBONE --dataset $ds --epochs $EPOCHS --seed $seed --missing-rate 0.3 --patience $PATIENCE --lr-ssr $lr_ssr $EXTRA --output $OUT > $LOG 2>&1" >> "$JOBFILE"
            done
        done
    done

    dispatch_jobs "$JOBFILE" 3 "LR ratio ablation"
}

case "$PHASE" in
    component) run_component ;;
    rank)      run_rank ;;
    lr)        run_lr ;;
    all)
        run_component
        run_rank
        run_lr
        ;;
    *)
        echo "Usage: bash scripts/experiments/launch_ablation.sh [component|rank|lr|all]"
        exit 1
        ;;
esac

echo ""
echo "Ablation complete. Results in $OUTDIR/"
echo "Files: $(find $OUTDIR -name '*.json' | wc -l)"
