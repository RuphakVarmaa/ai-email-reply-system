#!/usr/bin/env bash
# Full pipeline: generate replies for test set + evaluate + report
set -euo pipefail
cd "$(dirname "$0")/.."

MODE=${1:-rag}
N=${2:-}

args=(--mode "$MODE")
if [ -n "$N" ]; then
    args+=(--n "$N")
fi

echo "Running pipeline: mode=$MODE"
PYTHONPATH=src .venv/bin/python -m email_reply run "${args[@]}"
echo "Done. Reports in reports/"
