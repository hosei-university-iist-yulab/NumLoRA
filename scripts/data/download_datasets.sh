#!/bin/bash
# NumLoRA — Dataset Download Script
# Run on server before experiments: bash scripts/download_datasets.sh
#
# Prerequisites: wget, unzip, git, python3 (for gdown if needed)
# Estimated total download: ~8 GB

set -euo pipefail

DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR/ts" "$DATA_DIR/non_ts"

echo "============================================"
echo "NumLoRA Dataset Downloader"
echo "Target directory: $DATA_DIR"
echo "============================================"

# ============================================================
# TIME-SERIES DATASETS (10)
# ============================================================

echo ""
echo "--- Time-Series Datasets ---"

# 1. PhysioNet 2012
echo "[1/15] PhysioNet 2012 Challenge..."
mkdir -p "$DATA_DIR/ts/physionet2012"
# wget -q -P "$DATA_DIR/ts/physionet2012/" https://physionet.org/files/challenge-2012/1.0.0/
echo "  -> TODO: download from physionet.org/content/challenge-2012/1.0.0/"
echo "     Requires PhysioNet credentials. Use: wget -r -N -c -np https://physionet.org/files/challenge-2012/1.0.0/"

# 2. Beijing Air Quality
echo "[2/15] Beijing Air Quality..."
mkdir -p "$DATA_DIR/ts/beijing_aq"
echo "  -> TODO: download from archive.ics.uci.edu/dataset/501"

# 3. Italy Air Quality
echo "[3/15] Italy Air Quality..."
mkdir -p "$DATA_DIR/ts/italy_aq"
echo "  -> TODO: download from archive.ics.uci.edu/dataset/360"

# 4. Solar Alabama
echo "[4/15] Solar Alabama..."
mkdir -p "$DATA_DIR/ts/solar_alabama"
echo "  -> TODO: download from nrel.gov/grid/solar-power-data.html"

# 5-8. ETT (h1, h2, m1, m2)
echo "[5-8/15] ETT datasets..."
mkdir -p "$DATA_DIR/ts/ett"
if [ ! -d "$DATA_DIR/ts/ett/ETDataset" ]; then
    git clone --depth 1 https://github.com/zhouhaoyi/ETDataset.git "$DATA_DIR/ts/ett/ETDataset" 2>/dev/null || \
        echo "  -> TODO: git clone https://github.com/zhouhaoyi/ETDataset"
fi

# 9. Electricity
echo "[9/15] Electricity (UCI)..."
mkdir -p "$DATA_DIR/ts/electricity"
echo "  -> TODO: download from archive.ics.uci.edu/dataset/321"

# 10. Traffic (PeMS)
echo "[10/15] Traffic (PeMS)..."
mkdir -p "$DATA_DIR/ts/traffic"
echo "  -> TODO: download from github.com/laiguokun/multivariate-time-series-data"

# 11. Weather (Jena)
echo "[11/15] Weather (Max Planck Jena)..."
mkdir -p "$DATA_DIR/ts/weather"
echo "  -> TODO: download from kaggle (requires kaggle API key)"
echo "     kaggle datasets download -d stytch16/jena-climate-2009-2016 -p $DATA_DIR/ts/weather/"

# 12. Exchange Rate
echo "[12/15] Exchange Rate..."
mkdir -p "$DATA_DIR/ts/exchange"
echo "  -> TODO: download from github.com/laiguokun/multivariate-time-series-data"

# 13. ILI (CDC)
echo "[13/15] ILI (CDC)..."
mkdir -p "$DATA_DIR/ts/ili"
echo "  -> TODO: download from cdc.gov/flu/weekly (or use Time-Series-Library preprocessed version)"

# ============================================================
# NON-TIME-SERIES DATASETS (5)
# ============================================================

echo ""
echo "--- Non-Time-Series Datasets ---"

# 1. UCI Regression Suite
echo "[14/15] UCI Regression Suite..."
mkdir -p "$DATA_DIR/non_ts/uci_regression"
echo "  -> TODO: download 8 datasets (Boston, Energy, Yacht, Kin8nm, Power, Protein, Wine, Naval)"
echo "     Consider using bayesian-benchmarks or uci_datasets Python package"

# 2. OpenML-CC18
echo "[15/15] OpenML-CC18..."
mkdir -p "$DATA_DIR/non_ts/openml_cc18"
echo "  -> TODO: pip install openml; then use openml.study.get_study(99) in Python"

# 3. QM9
echo "[bonus] QM9..."
mkdir -p "$DATA_DIR/non_ts/qm9"
echo "  -> TODO: pip install torch-geometric; use torch_geometric.datasets.QM9"

# 4. PDEBench
echo "[bonus] PDEBench..."
mkdir -p "$DATA_DIR/non_ts/pdebench"
echo "  -> TODO: git clone https://github.com/pdebench/PDEBench; follow their download script"

# 5. Higgs
echo "[bonus] Higgs (100k subset)..."
mkdir -p "$DATA_DIR/non_ts/higgs"
echo "  -> TODO: download from archive.ics.uci.edu/dataset/280; subsample to 100k rows"

echo ""
echo "============================================"
echo "Directory structure created. Fill in TODO items above."
echo "Total disk space needed: ~8 GB"
echo "============================================"
