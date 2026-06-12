#!/usr/bin/env bash
# Full evaluation: Exp0 ICL + Exp1/Exp2/Exp3 cartridges → comparison table.
#
# Assumes the base forest + QA exist (data/base/) and that the training runs have
# produced checkpoints under
# outputs_graph3/{exp1_adaptive,exp2_plain,exp3_rejection}/train.
#
# Thinking is ON by default (THINKING=1) to match synthesis/training-eval.
#
# Usage (from repo root):
#   CUDA_VISIBLE_DEVICES=0 bash examples/graph_3/scripts/run_eval.sh
#
# Overrides:
#   SKIP_exp0=1   skip ICL baseline
#   SKIP_exp1=1   skip Exp 1 cartridge eval
#   SKIP_exp2=1   skip Exp 2 cartridge eval
#   SKIP_exp3=1   skip Exp 3 cartridge eval
#   THINKING=0    disable Qwen3 <think> mode
#   MODEL=qwen4b  model key (default qwen1.7b)
#   LIMIT=N       evaluate only first N test questions (fast check)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
cd "$REPO_ROOT"

GPU="${CUDA_VISIBLE_DEVICES:-0}"
OUT="$CARTRIDGES_OUTPUT_DIR_GRAPH3"
MODEL="${MODEL:-qwen1.7b}"
LIMIT_ARGS=(); [ -n "${LIMIT:-}" ] && LIMIT_ARGS=(--limit "$LIMIT")
THINK_ARGS=(); [ "${THINKING:-1}" = "1" ] && THINK_ARGS=(--thinking)

find_ckpt() { find "$1" -name cache_last.pt 2>/dev/null | xargs ls -t 2>/dev/null | head -1 || true; }

DATA_DIR="$REPO_ROOT/examples/graph_3/data/base"
if [ ! -f "$DATA_DIR/corpus.txt" ] || [ ! -f "$DATA_DIR/test_meta.json" ]; then
  echo "ERROR: data not found under $DATA_DIR"
  echo "Run data generation first:"
  echo "  python -m examples.graph_3.data_gen.generate_forest"
  echo "  python -m examples.graph_3.data_gen.qagen"
  exit 1
fi
echo "Data: $DATA_DIR ✓ · thinking=${THINKING:-1}"

# ── Exp 0 — ICL baseline ──────────────────────────────────────────────────────
if [ "${SKIP_exp0:-0}" = "1" ]; then
  echo "── SKIP exp0 ──"
else
  echo ""; echo "══ exp0: ICL baseline (graph in context) ══"
  CUDA_VISIBLE_DEVICES="$GPU" \
  python -m examples.graph_3.evaluation.eval \
    --mode icl --model "$MODEL" \
    "${THINK_ARGS[@]}" "${LIMIT_ARGS[@]}" \
    --output "$OUT/exp0_icl/results.json"
fi

# ── Exp 1 — adaptive-hint cartridge ───────────────────────────────────────────
EXP1_CKPT="$(find_ckpt "$OUT/exp1_adaptive/train")"
if [ "${SKIP_exp1:-0}" = "1" ]; then
  echo "── SKIP exp1 ──"
elif [ -z "$EXP1_CKPT" ]; then
  echo "!! exp1: no checkpoint under $OUT/exp1_adaptive/train — skipping"
else
  echo ""; echo "══ exp1: cartridge eval (adaptive hint) ══"
  echo "   checkpoint: $EXP1_CKPT"
  CUDA_VISIBLE_DEVICES="$GPU" \
  python -m examples.graph_3.evaluation.eval \
    --mode cartridge --checkpoint "$EXP1_CKPT" --model "$MODEL" \
    "${THINK_ARGS[@]}" "${LIMIT_ARGS[@]}" \
    --output "$OUT/exp1_adaptive/eval/results.json"
fi

# ── Exp 2 — plain self-study cartridge ────────────────────────────────────────
EXP2_CKPT="$(find_ckpt "$OUT/exp2_plain/train")"
if [ "${SKIP_exp2:-0}" = "1" ]; then
  echo "── SKIP exp2 ──"
elif [ -z "$EXP2_CKPT" ]; then
  echo "!! exp2: no checkpoint under $OUT/exp2_plain/train — skipping"
else
  echo ""; echo "══ exp2: cartridge eval (plain self-study) ══"
  echo "   checkpoint: $EXP2_CKPT"
  CUDA_VISIBLE_DEVICES="$GPU" \
  python -m examples.graph_3.evaluation.eval \
    --mode cartridge --checkpoint "$EXP2_CKPT" --model "$MODEL" \
    "${THINK_ARGS[@]}" "${LIMIT_ARGS[@]}" \
    --output "$OUT/exp2_plain/eval/results.json"
fi

# ── Exp 3 — rejection-sampled cartridge ───────────────────────────────────────
EXP3_CKPT="$(find_ckpt "$OUT/exp3_rejection/train")"
if [ "${SKIP_exp3:-0}" = "1" ]; then
  echo "── SKIP exp3 ──"
elif [ -z "$EXP3_CKPT" ]; then
  echo "!! exp3: no checkpoint under $OUT/exp3_rejection/train — skipping"
else
  echo ""; echo "══ exp3: cartridge eval (rejection sampling) ══"
  echo "   checkpoint: $EXP3_CKPT"
  CUDA_VISIBLE_DEVICES="$GPU" \
  python -m examples.graph_3.evaluation.eval \
    --mode cartridge --checkpoint "$EXP3_CKPT" --model "$MODEL" \
    "${THINK_ARGS[@]}" "${LIMIT_ARGS[@]}" \
    --output "$OUT/exp3_rejection/eval/results.json"
fi

# ── Comparison ────────────────────────────────────────────────────────────────
echo ""; echo "══ comparison ══"
FILES=()
[ -f "$OUT/exp0_icl/results.json" ]          && FILES+=("$OUT/exp0_icl/results.json")
[ -f "$OUT/exp1_adaptive/eval/results.json" ] && FILES+=("$OUT/exp1_adaptive/eval/results.json")
[ -f "$OUT/exp2_plain/eval/results.json" ]   && FILES+=("$OUT/exp2_plain/eval/results.json")
[ -f "$OUT/exp3_rejection/eval/results.json" ] && FILES+=("$OUT/exp3_rejection/eval/results.json")

if [ "${#FILES[@]}" -ge 1 ]; then
  python -m examples.graph_3.evaluation.analyze "${FILES[@]}"
else
  echo "No result files found yet."
fi

echo ""; echo "Done. Results in $OUT"
