#!/bin/bash
# =============================================================================
# NumLoRA — CLEAN FULL SWEEP (production run)
#
# Phase 1: SmolLM (GPUs 4-5, 3/GPU) + Qwen (GPUs 6-7, 2/GPU)     ~3.5h
# Phase 2: TinyLlama (GPUs 4-5, 3/GPU) + Phi-3 (GPUs 6-7, 1/GPU) ~13h
# Total: ~16-17 hours
#
# Tasks: imputation (8 ds × 3 MR) + forecasting (9 configs)
# Methods: lora_r8, numlora_ctgs_only
# Seeds: 42, 123, 456
# Metrics: MAE, MSE, MRE
# Ablation: 7 component variants on SmolLM (ETT-h1, Exchange, Weather)
#
# Output: results/full/{imputation,forecasting,ablation}/
# =============================================================================
set -e

DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$DIR"

source /home/Aboya_25R9803/anaconda3/etc/profile.d/conda.sh
conda activate llms

EPOCHS=100
PATIENCE=20
SEEDS="42 123 456"
LOGDIR="$DIR/results/full/logs"
mkdir -p "$LOGDIR"

IMP_DATASETS="ett_h1 ett_h2 ett_m1 ett_m2 weather exchange traffic ili"
IMP_MRS="0.1 0.3 0.5"
FC_DATASETS="ett_h1_96 ett_h1_192 ett_h1_336 ett_h2_96 ett_h2_192 ett_h2_336 ett_m1_96 ett_m1_192 ett_m1_336"
METHODS="lora_r8 numlora_ctgs_only"
ABL_VARIANTS="numlora_mai_only numlora_ssr_only numlora_ctgs_only numlora_mai_ssr numlora_mai_ctgs numlora_ssr_ctgs numlora_full"

build_jobs() {
    local backbone=$1 jobfile=$2
    > "$jobfile"

    # Imputation
    for ds in $IMP_DATASETS; do
        for mr in $IMP_MRS; do
            for seed in $SEEDS; do
                for method in $METHODS; do
                    EXTRA=""
                    [ "$ds" = "ili" ] && EXTRA="--window-size 24 --patch-size 8"
                    local OUT="$DIR/results/full/imputation/${method}_${backbone}_${ds}_mr${mr}_seed${seed}.json"
                    local LOG="$LOGDIR/imp_${method}_${backbone}_${ds}_mr${mr}_s${seed}.log"
                    [ -f "$OUT" ] && continue
                    echo "python $DIR/scripts/experiments/train.py --task imputation --method $method --backbone $backbone --dataset $ds --epochs $EPOCHS --seed $seed --missing-rate $mr --patience $PATIENCE $EXTRA --output $OUT > $LOG 2>&1" >> "$jobfile"
                done
            done
        done
    done

    # Forecasting
    for ds in $FC_DATASETS; do
        for seed in $SEEDS; do
            for method in $METHODS; do
                local OUT="$DIR/results/full/forecasting/${method}_${backbone}_${ds}_seed${seed}.json"
                local LOG="$LOGDIR/fc_${method}_${backbone}_${ds}_s${seed}.log"
                [ -f "$OUT" ] && continue
                echo "python $DIR/scripts/experiments/train.py --task forecasting --method $method --backbone $backbone --dataset $ds --epochs $EPOCHS --seed $seed --patience $PATIENCE --output $OUT > $LOG 2>&1" >> "$jobfile"
            done
        done
    done
}

build_ablation_jobs() {
    local backbone=$1 jobfile=$2
    # Component ablation: imputation only, 3 datasets, 3 MRs
    for ds in ett_h1 exchange weather; do
        for mr in $IMP_MRS; do
            for seed in $SEEDS; do
                for method in $ABL_VARIANTS; do
                    local OUT="$DIR/results/full/ablation/${method}_${backbone}_${ds}_mr${mr}_seed${seed}.json"
                    local LOG="$LOGDIR/abl_${method}_${backbone}_${ds}_mr${mr}_s${seed}.log"
                    [ -f "$OUT" ] && continue
                    echo "python $DIR/scripts/experiments/train.py --task imputation --method $method --backbone $backbone --dataset $ds --epochs $EPOCHS --seed $seed --missing-rate $mr --patience $PATIENCE --output $OUT > $LOG 2>&1" >> "$jobfile"
                done
            done
        done
    done
}

dispatch() {
    local jobfile=$1 gpu=$2 parallel=$3 label=$4
    local n=$(wc -l < "$jobfile")
    echo "[$label] GPU $gpu: $n jobs, $parallel concurrent"
    cat "$jobfile" | xargs -P "$parallel" -I {} bash -c "CUDA_VISIBLE_DEVICES=$gpu {}"
    echo "[$label] GPU $gpu: DONE"
}

dispatch_split() {
    local jobfile=$1 gpu1=$2 gpu2=$3 parallel=$4 label=$5
    local n=$(wc -l < "$jobfile")
    echo "[$label] $n jobs across GPUs $gpu1,$gpu2 ($parallel concurrent each)"
    awk "NR % 2 == 0" "$jobfile" > "${jobfile}_g1"
    awk "NR % 2 == 1" "$jobfile" > "${jobfile}_g2"
    cat "${jobfile}_g1" | xargs -P "$parallel" -I {} bash -c "CUDA_VISIBLE_DEVICES=$gpu1 {}" &
    cat "${jobfile}_g2" | xargs -P "$parallel" -I {} bash -c "CUDA_VISIBLE_DEVICES=$gpu2 {}" &
    wait
    echo "[$label] DONE"
}

echo "============================================================"
echo "NumLoRA CLEAN FULL SWEEP — $(date)"
echo "============================================================"

# ── Phase 1: SmolLM (GPUs 4-5) + Qwen (GPUs 6-7) ──
echo ""
echo ">>> PHASE 1: SmolLM + Qwen"

build_jobs "smollm_360m" "/tmp/smollm_jobs.txt"
build_ablation_jobs "smollm_360m" "/tmp/smollm_abl.txt"
cat /tmp/smollm_abl.txt >> /tmp/smollm_jobs.txt
echo "SmolLM: $(wc -l < /tmp/smollm_jobs.txt) total jobs (main + ablation)"

build_jobs "qwen_0.5b" "/tmp/qwen_jobs.txt"
echo "Qwen: $(wc -l < /tmp/qwen_jobs.txt) jobs"

dispatch_split "/tmp/smollm_jobs.txt" 4 5 3 "SmolLM" &
dispatch_split "/tmp/qwen_jobs.txt" 6 7 2 "Qwen" &
wait

echo ""
echo ">>> PHASE 1 COMPLETE — $(date)"
echo "SmolLM results: $(find results/full -name '*smollm*' -name '*.json' | wc -l)"
echo "Qwen results: $(find results/full -name '*qwen*' -name '*.json' | wc -l)"

# ── Phase 2: TinyLlama (GPUs 4-5) + Phi-3 (GPUs 6-7) ──
echo ""
echo ">>> PHASE 2: TinyLlama + Phi-3"

build_jobs "tinyllama_1.1b" "/tmp/tl_jobs.txt"
build_ablation_jobs "tinyllama_1.1b" "/tmp/tl_abl.txt"
cat /tmp/tl_abl.txt >> /tmp/tl_jobs.txt
echo "TinyLlama: $(wc -l < /tmp/tl_jobs.txt) total jobs (main + ablation)"

build_jobs "phi3_mini" "/tmp/phi3_jobs.txt"
echo "Phi-3: $(wc -l < /tmp/phi3_jobs.txt) jobs"

dispatch_split "/tmp/tl_jobs.txt" 4 5 3 "TinyLlama" &
dispatch_split "/tmp/phi3_jobs.txt" 6 7 1 "Phi-3" &
wait

echo ""
echo ">>> PHASE 2 COMPLETE — $(date)"

# ── Generate tables and figures ──
echo ""
echo ">>> Generating tables and figures..."
python "$DIR/scripts/analysis/generate_tables.py" --results-dir results/full

echo ""
echo "============================================================"
echo "ALL DONE — $(date)"
echo "Total results: $(find results/full -name '*.json' | wc -l)"
echo "  Imputation: $(ls results/full/imputation/*.json 2>/dev/null | wc -l)"
echo "  Forecasting: $(ls results/full/forecasting/*.json 2>/dev/null | wc -l)"
echo "  Ablation: $(ls results/full/ablation/*.json 2>/dev/null | wc -l)"
echo "============================================================"
