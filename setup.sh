#!/usr/bin/env bash
# One-command setup for PoisonGuard: install requirements, download the
# corrected CICIDS2017 data (Lanvin et al., Exp_RP) and prepare the splits.
# Usage (from the repository folder): bash setup.sh
set -e
cd "$(dirname "$0")"

echo "1/3 Installing requirements..."
pip install -r requirements.txt

RAW_DIR="data/raw/lanvin_exp_rp"
if ls "$RAW_DIR"/*.csv >/dev/null 2>&1; then
    echo "2/3 Dataset already in $RAW_DIR, skipping download."
else
    echo "2/3 Downloading the corrected CICIDS2017 dataset (this can take a while)..."
    TMP_DIR="$(mktemp -d)"
    git clone --depth 1 https://gitlab.inria.fr/mlanvin/crisis2022.git "$TMP_DIR/crisis2022"
    mkdir -p "$RAW_DIR"
    cp "$TMP_DIR"/crisis2022/Labels/Exp_RP/*.csv "$RAW_DIR"/
    rm -rf "$TMP_DIR"
fi

echo "3/3 Cleaning, sampling, splitting and scaling the data..."
python -m src.data

echo "Setup complete. Run 'pytest' to check everything works."