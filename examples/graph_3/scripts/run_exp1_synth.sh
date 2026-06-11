#!/usr/bin/env bash
# Exp 1 — adaptive-hint self-study synthesis (terminal 2).
#
#   bash examples/graph_3/scripts/run_exp1_synth.sh
#   LIMIT=32 bash examples/graph_3/scripts/run_exp1_synth.sh   # quick smoke
#
# Needs a running Tokasaurus server (it owns the GPU; this script only sends
# HTTP requests). Start the server first, e.g. in a separate terminal:
#   CUDA_VISIBLE_DEVICES=0 tksrs model=Qwen/Qwen3-1.7B \
#       kv_cache_num_tokens='(512*1024)' max_top_logprobs=20
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
cd "$REPO_ROOT"

if ! curl -fsS --max-time 10 "$CARTRIDGES_TOKASAURUS_URL/v1/models" >/dev/null 2>&1; then
  echo "ERROR: Tokasaurus not reachable at $CARTRIDGES_TOKASAURUS_URL"
  echo "Start it first (separate terminal):"
  echo "  CUDA_VISIBLE_DEVICES=0 tksrs model=$HANDSHAKE_SERVER_MODEL kv_cache_num_tokens='(512*1024)' max_top_logprobs=20"
  exit 1
fi

OUT="$CARTRIDGES_OUTPUT_DIR_GRAPH3/exp1_adaptive"
mkdir -p "$OUT"

LIMIT_ARGS=()
[ -n "${LIMIT:-}" ] && LIMIT_ARGS=(--limit "$LIMIT")
THINK_ARGS=()
[ "${THINKING:-1}" = "1" ] && THINK_ARGS=(--thinking)

echo "Exp 1 synthesis → server $CARTRIDGES_TOKASAURUS_URL · thinking=${THINKING:-1} · log → $OUT/synth.log"
python -m examples.graph_3.synthesis.exp1_synthesize \
  --output-dir "$CARTRIDGES_OUTPUT_DIR_GRAPH3" \
  "${THINK_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
  2>&1 | tee "$OUT/synth.log"
