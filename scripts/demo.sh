#!/usr/bin/env bash
# Quick demo: single email → suggested reply → evaluation score
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=src .venv/bin/python -m email_reply demo
