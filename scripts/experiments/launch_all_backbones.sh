#!/bin/bash
# =============================================================================
# NumLoRA — Full sweep across ALL backbones and ALL datasets
#
# 4 backbones x 8 datasets x 3 MRs x 3 seeds x 2 methods = 1,152 total runs
# Plus component ablation (189) + LR ablation (27) = 1,368 total
#
# GPUs 4-7, concurrency adapted per backbone size:
#   SmolLM (360M):     3 concurrent/GPU
#   Qwen (494M):       2 concurrent/GPU
#   TinyLlama (1.1B):  2 concurrent/GPU
#   Phi-3-mini (3.8B): 1 concurrent/GPU
#
# Usage:
#   bash scripts/experiments/launch_all_backbones.sh [backbone]
#   backbone: smollm | qwen | tinyllama | phi3 | ablation | all
# =============================================================================
set -e

DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$DIR"

source /home/Aboya_25R9803/anaconda3/etc/profile.d/conda.sh
conda activate llms

PHASE="${1:-all}"
EPOCHS=100
PATIENCE=20
LOGDIR="$DIR/results/full/logs"
mkdir -p "$LOGDIR" "$DIR/results/full/ablation"

ALL_DATASETS="ett_h1 ett_h2 ett_m1 ett_m2 weather exchange traffic ili"
SEEDS="42 123 456"
MRS="0.1 0.3 0.5"
GPUS=(4 5 6 7)
NUM_GPUS=4

dispatch() {
    local jobfile=$1 parallel=$2 label=$3
    local n=$(wc -l < "$jobfile")
    echo "$label: $n jobs, $parallel concurrent/GPU across GPUs ${GPUS[*]}"
    for ((i=0; i<NUM_GPUS; i++)); do
        local gpu=${GPUS[$i]}
        awk "NR % $NUM_GPUS == $i" "$jobfile" > "/tmp/numlora_${label}_gpu${gpu}.txt"
        cat "/tmp/numlora_${label}_gpu${gpu}.txt" | xargs -P "$parallel" -I {} bash -c "CUDA_VISIBLE_DEVICES=$gpu {}" &
    done
    wait
    echo "$label: DONE ($(find $DIR/results/full -name '*.json' | wc -l) total files)"
}

build_sweep_jobs() {
    local backbone=$1 backbone_key=$2 jobfile=$3
    > "$jobfile"
    for ds in $ALL_DATASETS; do
        for mr in $MRS; do
            for seed in $SEEDS; do
                for method in lora_r8 numlora_full; do
                    EXTRA=""
                    [ "$ds" = "ili" ] && EXTRA="--window-size 24 --patch-size 8"
                    local OUT="$DIR/results/full/${method}_${backbone_key}_${ds}_mr${mr}_seed${seed}.json"
                    local LOG="$LOGDIR/${method}_${backbone_key}_${ds}_mr${mr}_seed${seed}.log"
                    [ -f "$OUT" ] && continue
                    echo "python $DIR/scripts/experiments/train.py --method $method --backbone $backbone --dataset $ds --epochs $EPOCHS --seed $seed --missing-rate $mr --patience $PATIENCE $EXTRA --output $OUT > $LOG 2>&1" >> "$jobfile"
                done
            done
        done
    done
}

run_smollm() {
    # SmolLM already has 5-dataset results; backfill ETT h2/m1/m2
    local JF="/tmp/numlora_smollm_backfill.txt"
    > "$JF"
    for ds in ett_h2 ett_m1 ett_m2; do
        for mr in $MRS; do
            for seed in $SEEDS; do
                for method in lora_r8 numlora_full; do
                    local OUT="$DIR/results/full/${method}_${ds}_mr${mr}_seed${seed}.json"
                    local LOG="$LOGDIR/${method}_${ds}_mr${mr}_seed${seed}.log"
                    [ -f "$OUT" ] && continue
                    echo "python $DIR/scripts/experiments/train.py --method $method --backbone smollm_360m --dataset $ds --epochs $EPOCHS --seed $seed --missing-rate $mr --patience $PATIENCE --output $OUT > $LOG 2>&1" >> "$JF"
                done
            done
        done
    done
    dispatch "$JF" 3 "SmolLM-backfill"
}

run_qwen() {
    local JF="/tmp/numlora_qwen_full.txt"
    build_sweep_jobs "qwen_0.5b" "qwen_0.5b" "$JF"
    dispatch "$JF" 2 "Qwen-0.5B"
}

run_tinyllama() {
    local JF="/tmp/numlora_tinyllama.txt"
    build_sweep_jobs "tinyllama_1.1b" "tinyllama_1.1b" "$JF"
    dispatch "$JF" 2 "TinyLlama-1.1B"
}

run_phi3() {
    local JF="/tmp/numlora_phi3.txt"
    build_sweep_jobs "phi3_mini" "phi3_mini" "$JF"
    dispatch "$JF" 1 "Phi3-mini"
}

run_ablation() {
    local JF="/tmp/numlora_ablation_full.txt"
    > "$JF"
    VARIANTS="numlora_mai_only numlora_ssr_only numlora_ctgs_only numlora_mai_ssr numlora_mai_ctgs numlora_ssr_ctgs numlora_full"
    for ds in ett_h1 exchange weather; do
        for mr in $MRS; do
            for seed in $SEEDS; do
                for method in $VARIANTS; do
                    local OUT="$DIR/results/full/ablation/${method}_${ds}_mr${mr}_seed${seed}.json"
                    local LOG="$LOGDIR/abl_${method}_${ds}_mr${mr}_seed${seed}.log"
                    [ -f "$OUT" ] && continue
                    echo "python $DIR/scripts/experiments/train.py --method $method --backbone smollm_360m --dataset $ds --epochs $EPOCHS --seed $seed --missing-rate $mr --patience $PATIENCE --output $OUT > $LOG 2>&1" >> "$JF"
                done
            done
        done
    done
    # LR ratio ablation
    for ds in ett_h1 exchange weather; do
        for seed in $SEEDS; do
            for lr_ssr in 1e-3 3e-3 1e-2; do
                local OUT="$DIR/results/full/ablation/lr_numlora_lrssr${lr_ssr}_${ds}_mr0.3_seed${seed}.json"
                local LOG="$LOGDIR/abl_lr_lrssr${lr_ssr}_${ds}_seed${seed}.log"
                [ -f "$OUT" ] && continue
                echo "python $DIR/scripts/experiments/train.py --method numlora_full --backbone smollm_360m --dataset $ds --epochs $EPOCHS --seed $seed --missing-rate 0.3 --patience $PATIENCE --lr-ssr $lr_ssr --output $OUT > $LOG 2>&1" >> "$JF"
            done
        done
    done
    dispatch "$JF" 3 "Ablation"
}

case "$PHASE" in
    smollm)    run_smollm ;;
    qwen)      run_qwen ;;
    tinyllama) run_tinyllama ;;
    phi3)      run_phi3 ;;
    ablation)  run_ablation ;;
    all)
        echo "============================================================"
        echo "NumLoRA FULL EXPERIMENT SUITE"
        echo "4 backbones x 8 datasets x 3 MRs x 3 seeds x 2 methods"
        echo "============================================================"
        run_smollm
        # Qwen already running separately; skip if results exist
        run_qwen
        run_tinyllama
        run_phi3
        run_ablation
        echo ""
        echo "ALL EXPERIMENTS COMPLETE"
        echo "Total files: $(find $DIR/results/full -name '*.json' | wc -l)"
        ;;
    *)
        echo "Usage: bash $0 [smollm|qwen|tinyllama|phi3|ablation|all]"
        exit 1
        ;;
esac
