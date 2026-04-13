#!/bin/bash
# =============================================================================
# NumLoRA — Quick-tier experiments (reliability check)
#
# Purpose: Validate that NumLoRA beats LoRA on pilot datasets before full sweep.
# Scope:   3 datasets x 3 MRs x 2 seeds x {LoRA, LoRA-r9, NumLoRA, DoRA} = 72 runs
# Time:    ~2-3 hours on 4x RTX 3090
# GPUs:    CUDA 4-7 (4 concurrent GPUs, shared server)
#
# Usage:   bash scripts/experiments/launch_quick.sh
# =============================================================================
set -e

DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$DIR"

source /home/Aboya_25R9803/anaconda3/etc/profile.d/conda.sh
conda activate llms

TIER="quick"
EPOCHS=50
LOGDIR="$DIR/results/$TIER/logs"
OUTDIR="$DIR/results/$TIER"
mkdir -p "$LOGDIR" "$OUTDIR"

DATASETS="physionet2012 solar_alabama exchange_rate"
METHODS="lora_r8 lora_r9 dora_r8 numlora_full"
MRS="0.1 0.3 0.5"
SEEDS="42 123"
GPUS=(4 5 6 7)

echo "============================================================"
echo "NumLoRA Quick-Tier — $EPOCHS epochs, ${#GPUS[@]} GPUs"
echo "GPUs: ${GPUS[*]}"
echo "============================================================"

# Build job list
JOBFILE="/tmp/numlora_quick_jobs.txt"
> "$JOBFILE"

for ds in $DATASETS; do
    for mr in $MRS; do
        for seed in $SEEDS; do
            for method in $METHODS; do
                OUT="$OUTDIR/${method}_${ds}_mr${mr}_seed${seed}.json"
                LOG="$LOGDIR/${method}_${ds}_mr${mr}_seed${seed}.log"

                if [ -f "$OUT" ]; then
                    continue
                fi

                # TODO: uncomment when train.py is implemented
                # echo "python $DIR/scripts/experiments/train.py --method $method --dataset $ds --missing-rate $mr --seed $seed --epochs $EPOCHS --tier $TIER --output $OUT > $LOG 2>&1" >> "$JOBFILE"
                echo "echo '[RUN] method=$method dataset=$ds mr=$mr seed=$seed'" >> "$JOBFILE"
            done
        done
    done
done

NJOBS=$(wc -l < "$JOBFILE")
echo "Total jobs: $NJOBS"

# Round-robin across GPUs
NUM_GPUS=${#GPUS[@]}
for ((i=0; i<NUM_GPUS; i++)); do
    GPU=${GPUS[$i]}
    # Extract this GPU's share of jobs
    awk "NR % $NUM_GPUS == $i" "$JOBFILE" > "/tmp/numlora_quick_gpu${GPU}.txt"
    GCOUNT=$(wc -l < "/tmp/numlora_quick_gpu${GPU}.txt")
    echo "GPU $GPU: $GCOUNT jobs, 2 concurrent"
    cat "/tmp/numlora_quick_gpu${GPU}.txt" | xargs -P 2 -I {} bash -c "CUDA_VISIBLE_DEVICES=$GPU {}" &
done

wait
echo ""
echo "Quick tier complete. Results: $(ls $OUTDIR/*.json 2>/dev/null | wc -l) files in $OUTDIR/"
