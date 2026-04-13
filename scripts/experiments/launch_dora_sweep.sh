#!/bin/bash
# =============================================================================
# NumLoRA — DoRA baseline sweep
#
# 5 datasets x 3 MRs x 3 seeds x {DoRA r=8} = 45 runs
# GPUs 4-7, 3 concurrent per GPU
# ETA: ~15-20 min
# =============================================================================
set -e

DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$DIR"

source /home/Aboya_25R9803/anaconda3/etc/profile.d/conda.sh
conda activate llms

TIER="full"
EPOCHS=100
PATIENCE=20
BACKBONE="smollm_360m"
LOGDIR="$DIR/results/$TIER/logs"
OUTDIR="$DIR/results/$TIER"
mkdir -p "$LOGDIR"

DATASETS="ett_h1 weather exchange traffic ili"
SEEDS="42 123 456"
MRS="0.1 0.3 0.5"
GPUS=(4 5 6 7)

JOBFILE="/tmp/numlora_dora_jobs.txt"
> "$JOBFILE"

for ds in $DATASETS; do
    for mr in $MRS; do
        for seed in $SEEDS; do
            EXTRA=""
            [ "$ds" = "ili" ] && EXTRA="--window-size 24 --patch-size 8"

            OUT="$OUTDIR/dora_r8_${ds}_mr${mr}_seed${seed}.json"
            LOG="$LOGDIR/dora_r8_${ds}_mr${mr}_seed${seed}.log"
            [ -f "$OUT" ] && continue

            echo "python $DIR/scripts/experiments/train.py --method dora_r8 --backbone $BACKBONE --dataset $ds --epochs $EPOCHS --seed $seed --missing-rate $mr --patience $PATIENCE $EXTRA --output $OUT > $LOG 2>&1" >> "$JOBFILE"
        done
    done
done

NJOBS=$(wc -l < "$JOBFILE")
echo "DoRA sweep: $NJOBS jobs across GPUs ${GPUS[*]}"

NUM_GPUS=${#GPUS[@]}
for ((i=0; i<NUM_GPUS; i++)); do
    GPU=${GPUS[$i]}
    awk "NR % $NUM_GPUS == $i" "$JOBFILE" > "/tmp/numlora_dora_gpu${GPU}.txt"
    N=$(wc -l < "/tmp/numlora_dora_gpu${GPU}.txt")
    echo "  GPU $GPU: $N jobs, 3 concurrent"
    cat "/tmp/numlora_dora_gpu${GPU}.txt" | xargs -P 3 -I {} bash -c "CUDA_VISIBLE_DEVICES=$GPU {}" &
done

wait
echo "DoRA sweep complete. Results: $(find $OUTDIR -name 'dora_*.json' | wc -l) files"
