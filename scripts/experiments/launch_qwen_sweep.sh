#!/bin/bash
# =============================================================================
# NumLoRA — Qwen2.5-0.5B backbone sweep (scale validation)
#
# 5 datasets x 3 MRs x 3 seeds x {LoRA, NumLoRA} = 90 runs
# GPUs 4-7, 2 concurrent per GPU (Qwen is larger than SmolLM)
# =============================================================================
set -e

DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$DIR"

source /home/Aboya_25R9803/anaconda3/etc/profile.d/conda.sh
conda activate llms

TIER="full"
EPOCHS=100
PATIENCE=20
BACKBONE="qwen_0.5b"
LOGDIR="$DIR/results/$TIER/logs"
OUTDIR="$DIR/results/$TIER"
mkdir -p "$LOGDIR"

DATASETS="ett_h1 weather exchange traffic ili"
SEEDS="42 123 456"
MRS="0.1 0.3 0.5"
GPUS=(4 5 6 7)

JOBFILE="/tmp/numlora_qwen_jobs.txt"
> "$JOBFILE"

for ds in $DATASETS; do
    for mr in $MRS; do
        for seed in $SEEDS; do
            for method in lora_r8 numlora_full; do
                EXTRA=""
                [ "$ds" = "ili" ] && EXTRA="--window-size 24 --patch-size 8"

                OUT="$OUTDIR/${method}_${BACKBONE}_${ds}_mr${mr}_seed${seed}.json"
                LOG="$LOGDIR/${method}_${BACKBONE}_${ds}_mr${mr}_seed${seed}.log"
                [ -f "$OUT" ] && continue

                echo "python $DIR/scripts/experiments/train.py --method $method --backbone $BACKBONE --dataset $ds --epochs $EPOCHS --seed $seed --missing-rate $mr --patience $PATIENCE $EXTRA --output $OUT > $LOG 2>&1" >> "$JOBFILE"
            done
        done
    done
done

NJOBS=$(wc -l < "$JOBFILE")
echo "Qwen sweep: $NJOBS jobs across GPUs ${GPUS[*]}"

NUM_GPUS=${#GPUS[@]}
for ((i=0; i<NUM_GPUS; i++)); do
    GPU=${GPUS[$i]}
    awk "NR % $NUM_GPUS == $i" "$JOBFILE" > "/tmp/numlora_qwen_gpu${GPU}.txt"
    N=$(wc -l < "/tmp/numlora_qwen_gpu${GPU}.txt")
    echo "  GPU $GPU: $N jobs, 2 concurrent"
    cat "/tmp/numlora_qwen_gpu${GPU}.txt" | xargs -P 2 -I {} bash -c "CUDA_VISIBLE_DEVICES=$GPU {}" &
done

wait
echo "Qwen sweep complete. Results: $(find $OUTDIR -name "*${BACKBONE}*.json" | wc -l) files"
