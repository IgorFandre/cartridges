#!/usr/bin/env bash
# Exp 2 — plain self-study synthesis, no filter (terminal 3).
#
#   bash examples/graph_3/scripts/run_exp2_synth.sh
#   LIMIT=32 bash examples/graph_3/scripts/run_exp2_synth.sh             # smoke
#   SAMPLES_PER_Q=2 bash examples/graph_3/scripts/run_exp2_synth.sh     # scale
#
# Needs a running Tokasaurus server (shared with exp1 — both can run at once;
# the server batches requests). Start it first, e.g.:
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

OUT="$CARTRIDGES_OUTPUT_DIR_GRAPH3/exp2_plain"
mkdir -p "$OUT"

LIMIT_ARGS=()
[ -n "${LIMIT:-}" ] && LIMIT_ARGS=(--limit "$LIMIT")
THINK_ARGS=()
[ "${THINKING:-1}" = "1" ] && THINK_ARGS=(--thinking)
SBS_ARGS=()
[ "${STEPBYSTEP:-0}" = "1" ] && SBS_ARGS=(--stepbystep)   # → dataset_stepbystep.parquet

echo "Exp 2 synthesis → server $CARTRIDGES_TOKASAURUS_URL · thinking=${THINKING:-1} · stepbystep=${STEPBYSTEP:-0} · log → $OUT/synth.log"
python -m examples.graph_3.synthesis.exp2_synthesize \
  --output-dir "$CARTRIDGES_OUTPUT_DIR_GRAPH3" \
  --samples-per-question "${SAMPLES_PER_Q:-1}" \
  "${THINK_ARGS[@]}" \
  "${SBS_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
  2>&1 | tee "$OUT/synth.log"
