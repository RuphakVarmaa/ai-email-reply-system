#!/usr/bin/env bash
# Download and extract the Twitter Customer Support dataset from Kaggle
# Requires: kaggle CLI (pip install kaggle) OR manual download
set -euo pipefail

cd "$(dirname "$0")/.."
mkdir -p data/raw

if command -v kaggle &>/dev/null; then
    echo "Downloading via Kaggle CLI..."
    kaggle datasets download thoughtvector/customer-support-on-twitter -p /tmp/twitter_cs --unzip
    echo "Extracting AppleSupport pairs..."
    python3 scripts/extract_apple.py /tmp/twitter_cs/twcs/twcs.csv data/raw/apple_pairs.jsonl
else
    echo "Kaggle CLI not found. Please:"
    echo "1. Download from: https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter"
    echo "2. Unzip and run: python3 scripts/extract_apple.py path/to/twcs.csv data/raw/apple_pairs.jsonl"
    exit 1
fi

echo "Running preprocessing..."
python3 scripts/preprocess.py
echo "Done! data/processed/apple_clean.jsonl is ready."
