#!/bin/bash
# =============================================================================
# NumLoRA — Full-tier experiments (final results for paper)
#
# Purpose:  Complete experimental protocol for all methods, datasets, seeds.
# Scope:    Selectable phases (numlora, baselines, ablation, scale, non_ts)
# Time:     Phase-dependent (see below)
# GPUs:     CUDA 4-7 (4 concurrent GPUs, shared server), RTX 3090 24GB
#
# Usage:
#   bash scripts/experiments/launch_full.sh numlora     # ~6 hours on 4 GPUs
#   bash scripts/experiments/launch_full.sh baselines    # ~12 hours on 4 GPUs
#   bash scripts/experiments/launch_full.sh ablation     # ~4 hours on 4 GPUs
#   bash scripts/experiments/launch_full.sh scale        # ~10 hours on 4 GPUs
#   bash scripts/experiments/launch_full.sh non_ts       # ~8 hours on 4 GPUs
#   bash scripts/experiments/launch_full.sh all          # all phases sequentially
# =============================================================================
set -e

DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$DIR"

source /home/Aboya_25R9803/anaconda3/etc/profile.d/conda.sh
conda activate llms

PHASE="${1:-numlora}"
TIER="full"
EPOCHS=200
LOGDIR="$DIR/results/$TIER/logs"
mkdir -p "$LOGDIR"

# ── Shared definitions ──
TS_DATASETS="physionet2012 beijing_aq italy_aq solar_alabama ett_h1 ett_h2 ett_m1 ett_m2 electricity traffic weather exchange_rate ili"
PILOT_DATASETS="physionet2012 solar_alabama exchange_rate"
NON_TS_DATASETS="uci_regression openml_cc18 qm9 pdebench higgs"

SEEDS="42 123 456"
MRS="0.1 0.2 0.3 0.4 0.5"
ABLATION_MRS="0.1 0.3 0.5"

PEFT_BASELINES="frozen lora_r8 lora_r9 dora_r8 pissa_r8 vera"
REPROG_BASELINES="llm4imp time_llm gpt4ts"
IMPUTATION_BASELINES="saits tslanet tefn timemixer_pp"
NUMLORA_ABLATIONS="numlora_mai_only numlora_ssr_only numlora_ctgs_only numlora_mai_ssr numlora_mai_ctgs numlora_ssr_ctgs numlora_full"

GPUS=(4 5 6 7)
NUM_GPUS=${#GPUS[@]}

# ── Job dispatch helper ──
run_gpu_jobs() {
    local jobfile=$1 parallel=$2
    local total=$(wc -l < "$jobfile")
    echo "  Total jobs: $total, distributing across ${GPUS[*]} ($parallel concurrent/GPU)"

    for ((i=0; i<NUM_GPUS; i++)); do
        local gpu=${GPUS[$i]}
        awk "NR % $NUM_GPUS == $i" "$jobfile" > "/tmp/numlora_${PHASE}_gpu${gpu}.txt"
        local n=$(wc -l < "/tmp/numlora_${PHASE}_gpu${gpu}.txt")
        echo "  GPU $gpu: $n jobs"
        cat "/tmp/numlora_${PHASE}_gpu${gpu}.txt" | xargs -P "$parallel" -I {} bash -c "CUDA_VISIBLE_DEVICES=$gpu {}" &
    done
    wait
}

build_job() {
    local method=$1 ds=$2 mr=$3 seed=$4 backbone=${5:-gpt2_small}
    local subdir="$PHASE"
    local outdir="$DIR/results/$TIER/$subdir"
    mkdir -p "$outdir"

    local out="$outdir/${method}_${ds}_mr${mr}_seed${seed}_${backbone}.json"
    local log="$LOGDIR/${PHASE}_${method}_${ds}_mr${mr}_seed${seed}.log"

    if [ -f "$out" ]; then
        return  # skip completed
    fi

    # TODO: uncomment when train.py is implemented
    # echo "python $DIR/scripts/experiments/train.py --method $method --dataset $ds --missing-rate $mr --seed $seed --epochs $EPOCHS --backbone $backbone --tier $TIER --output $out > $log 2>&1"
    echo "echo '[RUN] $method $ds mr=$mr s=$seed bb=$backbone'"
}

echo "============================================================"
echo "NumLoRA Full-Tier — Phase: $PHASE"
echo "GPUs: ${GPUS[*]} (RTX 3090, CUDA 4-7)"
echo "============================================================"

case "$PHASE" in
    numlora)
        echo "NumLoRA full TS sweep: 13 datasets x 5 MRs x 3 seeds"
        JOBFILE="/tmp/numlora_numlora_jobs.txt"
        > "$JOBFILE"
        for ds in $TS_DATASETS; do
            for mr in $MRS; do
                for seed in $SEEDS; do
                    build_job "numlora_full" "$ds" "$mr" "$seed" >> "$JOBFILE"
                done
            done
        done
        run_gpu_jobs "$JOBFILE" 3
        ;;

    baselines)
        echo "All baselines full TS sweep: 13 methods x 13 datasets x 5 MRs x 3 seeds"
        JOBFILE="/tmp/numlora_baselines_jobs.txt"
        > "$JOBFILE"
        for ds in $TS_DATASETS; do
            for mr in $MRS; do
                for seed in $SEEDS; do
                    for method in $PEFT_BASELINES $REPROG_BASELINES $IMPUTATION_BASELINES; do
                        build_job "$method" "$ds" "$mr" "$seed" >> "$JOBFILE"
                    done
                done
            done
        done
        run_gpu_jobs "$JOBFILE" 2
        ;;

    ablation)
        echo "Ablation study: 8 variants x 3 datasets x 3 MRs x 3 seeds"
        JOBFILE="/tmp/numlora_ablation_jobs.txt"
        > "$JOBFILE"
        for ds in $PILOT_DATASETS; do
            for mr in $ABLATION_MRS; do
                for seed in $SEEDS; do
                    for method in lora_r8 $NUMLORA_ABLATIONS; do
                        build_job "$method" "$ds" "$mr" "$seed" >> "$JOBFILE"
                    done
                done
            done
        done
        run_gpu_jobs "$JOBFILE" 3
        ;;

    scale)
        echo "Backbone scale validation: 3 backbones x 3 datasets x 5 MRs x 3 seeds"
        JOBFILE="/tmp/numlora_scale_jobs.txt"
        > "$JOBFILE"
        SCALE_BACKBONES="gpt2_small gpt2_medium llama3_8b"
        for backbone in $SCALE_BACKBONES; do
            for ds in $PILOT_DATASETS; do
                for mr in $MRS; do
                    for seed in $SEEDS; do
                        for method in lora_r8 numlora_full; do
                            build_job "$method" "$ds" "$mr" "$seed" "$backbone" >> "$JOBFILE"
                        done
                    done
                done
            done
        done
        # 1 concurrent for 7-8B models (24GB GPU memory limit)
        run_gpu_jobs "$JOBFILE" 1
        ;;

    non_ts)
        echo "Non-TS generalisation: 7 methods x 5 datasets x 3 seeds"
        JOBFILE="/tmp/numlora_nonts_jobs.txt"
        > "$JOBFILE"
        for ds in $NON_TS_DATASETS; do
            for seed in $SEEDS; do
                for method in lora_r8 lora_r9 dora_r8 pissa_r8 vera llm4imp numlora_full; do
                    build_job "$method" "$ds" "0.0" "$seed" >> "$JOBFILE"
                done
            done
        done
        run_gpu_jobs "$JOBFILE" 3
        ;;

    all)
        echo "Full protocol — all phases sequentially"
        for p in numlora baselines ablation scale non_ts; do
            bash "$0" "$p"
        done
        ;;

    *)
        echo "Unknown phase: $PHASE"
        echo "Usage: bash scripts/experiments/launch_full.sh [numlora|baselines|ablation|scale|non_ts|all]"
        exit 1
        ;;
esac

echo ""
echo "Phase $PHASE complete."
echo "Results: $(find $DIR/results/$TIER/$PHASE -name '*.json' 2>/dev/null | wc -l) JSON files"
