#!/usr/bin/env bash
# Full evaluation: Exp0 ICL + Exp1 cartridge + Exp2 cartridge → comparison table.
#
# Assumes tree + QA already generated (run_all.sh data stages done),
# and training checkpoints exist under outputs_graph2/.
#
# Usage (from repo root):
#   source examples/graph_2/scripts/env.sh
#   CUDA_VISIBLE_DEVICES=0 bash examples/graph_2/scripts/run_eval.sh
#
# Overrides:
#   SKIP_exp0=1   skip ICL baseline
#   SKIP_exp1=1   skip Exp 1 cartridge eval
#   SKIP_exp2=1   skip Exp 2 cartridge eval
#   MODEL=qwen4b  model key (default qwen1.7b)
#   LIMIT=N       evaluate only first N test questions (fast check)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

GPU="${CUDA_VISIBLE_DEVICES:-0}"
OUT="${CARTRIDGES_OUTPUT_DIR_GRAPH2:-$REPO_ROOT/outputs_graph2}"
MODEL="${MODEL:-qwen1.7b}"
LIMIT_ARG="${LIMIT:+--limit $LIMIT}"

# ── Helpers ───────────────────────────────────────────────────────────────────
run() {
  local stage="$1" desc="$2"; shift 2
  local skip_var="SKIP_${stage}"
  if [ "${!skip_var:-0}" = "1" ]; then
    echo "── SKIP $stage ──"; return
  fi
  echo ""; echo "══ $stage: $desc ══"
  "$@"
}

find_ckpt() {
  find "$1" -name cache_last.pt 2>/dev/null \
    | xargs ls -t 2>/dev/null \
    | head -1 || true
}

# ── Check data exists ─────────────────────────────────────────────────────────
DATA_DIR="$REPO_ROOT/examples/graph_2/data/base"
if [ ! -f "$DATA_DIR/family_tree_corpus.txt" ] || [ ! -f "$DATA_DIR/test_lineage.parquet" ]; then
  echo "ERROR: data not found under $DATA_DIR"
  echo "Run data generation first:"
  echo "  python -m examples.graph_2.data_gen.generate_tree"
  echo "  python -m examples.graph_2.data_gen.lineage_qagen"
  exit 1
fi
echo "Data: $DATA_DIR ✓"

# ── Exp 0 — ICL baseline ──────────────────────────────────────────────────────
run exp0 "ICL baseline (graph in context, free CoT)" \
  env CUDA_VISIBLE_DEVICES="$GPU" \
  python -m examples.graph_2.evaluation.lineage_eval \
    --mode icl \
    --model "$MODEL" \
    --max-new-tokens 1024 \
    $LIMIT_ARG \
    --output "$OUT/exp0_icl/results.json"

# ── Exp 1 — cartridge (BFS-path self-study) ───────────────────────────────────
EXP1_CKPT="$(find_ckpt "$OUT/exp1_selfstudy")"
if [ "${SKIP_exp1:-0}" = "1" ]; then
  echo "── SKIP exp1 ──"
elif [ -z "$EXP1_CKPT" ]; then
  echo "!! exp1: no checkpoint found under $OUT/exp1_selfstudy — skipping"
else
  echo ""; echo "══ exp1: cartridge eval (BFS-path self-study) ══"
  echo "   checkpoint: $EXP1_CKPT"
  CUDA_VISIBLE_DEVICES="$GPU" \
  python -m examples.graph_2.evaluation.lineage_eval \
    --mode cartridge \
    --checkpoint "$EXP1_CKPT" \
    --model "$MODEL" \
    --max-new-tokens 1024 \
    $LIMIT_ARG \
    --output "$OUT/exp1_selfstudy/eval/results.json"
fi

# ── Exp 2 — cartridge (STaR) ──────────────────────────────────────────────────
EXP2_CKPT="$(find_ckpt "$OUT/exp2_star")"
if [ "${SKIP_exp2:-0}" = "1" ]; then
  echo "── SKIP exp2 ──"
elif [ -z "$EXP2_CKPT" ]; then
  echo "!! exp2: no checkpoint found under $OUT/exp2_star — skipping"
else
  echo ""; echo "══ exp2: cartridge eval (STaR) ══"
  echo "   checkpoint: $EXP2_CKPT"
  CUDA_VISIBLE_DEVICES="$GPU" \
  python -m examples.graph_2.evaluation.lineage_eval \
    --mode cartridge \
    --checkpoint "$EXP2_CKPT" \
    --model "$MODEL" \
    --max-new-tokens 1024 \
    $LIMIT_ARG \
    --output "$OUT/exp2_star/eval/results.json"
fi

# ── Comparison ────────────────────────────────────────────────────────────────
echo ""; echo "══ comparison ══"
FILES=()
[ -f "$OUT/exp0_icl/results.json" ]          && FILES+=("$OUT/exp0_icl/results.json")
[ -f "$OUT/exp1_selfstudy/eval/results.json" ] && FILES+=("$OUT/exp1_selfstudy/eval/results.json")
[ -f "$OUT/exp2_star/eval/results.json" ]    && FILES+=("$OUT/exp2_star/eval/results.json")

if [ "${#FILES[@]}" -ge 1 ]; then
  python -m examples.graph_2.evaluation.analyze "${FILES[@]}"
else
  echo "No result files found yet."
fi

echo ""; echo "Done. Results in $OUT"
