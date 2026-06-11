#!/usr/bin/env bash
# Exp 0 — ICL baseline eval (terminal 1).
#
#   bash examples/graph_3/scripts/run_exp0_icl.sh
#   GPU=1 bash examples/graph_3/scripts/run_exp0_icl.sh       # other GPU
#   LIMIT=20 bash examples/graph_3/scripts/run_exp0_icl.sh    # quick smoke
#
# Loads Qwen3-1.7B via HF directly on the GPU (no Tokasaurus needed).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
cd "$REPO_ROOT"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
OUT="$CARTRIDGES_OUTPUT_DIR_GRAPH3/exp0_icl"
mkdir -p "$OUT"

LIMIT_ARGS=()
[ -n "${LIMIT:-}" ] && LIMIT_ARGS=(--limit "$LIMIT")
THINK_ARGS=()
[ "${THINKING:-1}" = "1" ] && THINK_ARGS=(--thinking)

echo "Exp 0 ICL → GPU $GPU · thinking=${THINKING:-1} · results → $OUT/results.json · log → $OUT/run.log"
CUDA_VISIBLE_DEVICES="$GPU" \
python -m examples.graph_3.evaluation.eval \
  --mode icl \
  --output "$OUT/results.json" \
  "${THINK_ARGS[@]}" \
  "${LIMIT_ARGS[@]}" \
  2>&1 | tee "$OUT/run.log"

python -m examples.graph_3.evaluation.analyze "$OUT/results.json"
