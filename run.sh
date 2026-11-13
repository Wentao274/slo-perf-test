#!/usr/bin/env bash
# Long-context full-load SLO test launcher.
# Usage:
#   ./run.sh                                    # default cases, dry-run
#   ./run.sh --url http://... --model ...       # specify target server
#   ./run.sh --config config.json               # use config file
#   ./run.sh --scenario 64K:600 --fixed-rate 54 # single case at fixed rate
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SCRIPT="$SCRIPT_DIR/longcontext_slo.py"

# ── Default connection (override with CLI args or config file) ──────────
URL="${URL:-}"
MODEL="${MODEL:-}"
TOKENIZER_PATH="${TOKENIZER_PATH:-}"

# ── Build command ────────────────────────────────────────────────────────
CMD=(python3 "$SCRIPT")

if [[ -n "$URL" ]]; then
    CMD+=(--url "$URL")
fi
if [[ -n "$MODEL" ]]; then
    CMD+=(--model "$MODEL")
fi
if [[ -n "$TOKENIZER_PATH" ]]; then
    CMD+=(--tokenizer-path "$TOKENIZER_PATH")
fi

# Pass through all user arguments
CMD+=("$@")

echo "Running: ${CMD[*]}"
exec "${CMD[@]}"
