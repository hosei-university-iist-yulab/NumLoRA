#!/bin/bash
# NumLoRA — Result Aggregation & LaTeX Table Generator
# Usage: bash scripts/generate_tables.sh [results_dir]
#
# Reads JSON results from the sweep, computes statistics,
# and generates LaTeX tables for the paper.

set -euo pipefail

RESULTS_DIR="${1:-results}"

echo "============================================"
echo "NumLoRA Table Generator"
echo "Results directory: $RESULTS_DIR"
echo "============================================"

# TODO: implement in Python
# python scripts/generate_tables.py \
#     --results_dir "$RESULTS_DIR" \
#     --output_dir "paper/tables/" \
#     --tables "main,mr_sweep,ablation,non_ts,efficiency,backbone_scale" \
#     --seeds 42,123,456 \
#     --bootstrap_samples 10000 \
#     --correction "holm-bonferroni" \
#     --format "latex"

echo ""
echo "Tables to generate:"
echo "  Table 1: Main results (14 methods x 10 datasets at MR=0.3)"
echo "  Table 2: MR sensitivity (6 methods x 4 datasets x 5 MRs)"
echo "  Table 3: Ablation (8 variants x 3 datasets x 3 MRs)"
echo "  Table 4: Non-TS generalisation (8 methods x 5 datasets)"
echo "  Table 5: Efficiency (params, training time, throughput)"
echo "  Table 6: Backbone scale (3 backbones x 3 datasets)"
echo ""
echo "Statistics:"
echo "  - Mean +/- std across 3 seeds"
echo "  - Paired bootstrap 95% CIs (10k resamples)"
echo "  - Holm-Bonferroni corrected p-values"
echo "  - Bold = best, Underline = second best"
echo "  - Significance markers: * (p<0.05), ** (p<0.01)"
echo ""
echo "TODO: implement scripts/generate_tables.py"
