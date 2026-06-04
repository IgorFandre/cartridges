#!/usr/bin/env bash
# Full pipeline: tree → QA → Exp0 → Exp1 → Exp2 → compare.
#
# Source env.sh first:
#   source examples/graph_2/scripts/env.sh
#   CUDA_VISIBLE_DEVICES=0 bash examples/graph_2/scripts/run_all.sh
#
# Skip individual stages with SKIP_<stage>=1, e.g.:
#   SKIP_tree=1 SKIP_qa=1 bash examples/graph_2/scripts/run_all.sh
#
# For multi-GPU training set NPROC=2 (uses torchrun).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"
OUT="${CARTRIDGES_OUTPUT_DIR_GRAPH2:-$REPO_ROOT/outputs_graph2}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
NPROC="${NPROC:-1}"

if [ "$NPROC" -gt 1 ]; then
  TRAIN_CMD="torchrun --standalone --nproc_per_node=$NPROC"
else
  TRAIN_CMD="python"
fi

run() {
  local stage="$1"; local desc="$2"; shift 2
  local skip_var="SKIP_${stage}"
  if [ "${!skip_var:-0}" = "1" ]; then
    echo "── SKIP $stage ──────────────────────────────────────────"
    return
  fi
  echo ""
  echo "══ $stage: $desc ══════════════════════════════════════════"
  "$@"
}

# ── 0. Tree ───────────────────────────────────────────────────────────────────
run tree "Generate family tree (depth>=8, ~100 people)" \
  python -m examples.graph_2.data_gen.generate_tree \
    --depth 8 --n-people 100 --seed 42

# ── 1. QA ─────────────────────────────────────────────────────────────────────
run qa "Generate lineage Yes/No QA + train/test split" \
  python -m examples.graph_2.data_gen.lineage_qagen \
    --test-frac 0.2 --seed 42

# ── 2. Exp 0 — ICL baseline ───────────────────────────────────────────────────
run exp0_icl "Exp 0: ICL baseline eval (needs GPU)" \
  env CUDA_VISIBLE_DEVICES="$GPU" \
  python -m examples.graph_2.evaluation.lineage_eval \
    --mode icl \
    --output "$OUT/exp0_icl/results.json"

run exp0_analyze "Exp 0: analyze results" \
  python -m examples.graph_2.evaluation.analyze \
    "$OUT/exp0_icl/results.json"

# ── 3. Exp 1 — BFS-path self-study synthesis ─────────────────────────────────
run exp1_syn "Exp 1: self-study synthesis (needs server + GPU not required)" \
  python -m examples.graph_2.synthesis.lineage_synthesize

# ── 4. Exp 1 — Train ─────────────────────────────────────────────────────────
run exp1_train "Exp 1: train cartridge (EXP=exp1, needs GPU)" \
  env CUDA_VISIBLE_DEVICES="$GPU" EXP=exp1 CARTRIDGE_TOKENS=512 \
  $TRAIN_CMD -m examples.graph_2.training.lineage_train

# ── 5. Exp 1 — Eval ──────────────────────────────────────────────────────────
EXP1_CKPT="$(find "$OUT/exp1_selfstudy/train" -name cache_last.pt 2>/dev/null | head -1 || true)"
if [ -n "$EXP1_CKPT" ]; then
  run exp1_eval "Exp 1: cartridge eval" \
    env CUDA_VISIBLE_DEVICES="$GPU" \
    python -m examples.graph_2.evaluation.lineage_eval \
      --mode cartridge \
      --checkpoint "$EXP1_CKPT" \
      --output "$OUT/exp1_selfstudy/train/eval/results.json"
else
  echo "── exp1_eval skipped: no checkpoint found under $OUT/exp1_selfstudy/train ──"
fi

# ── 6. Exp 2 — STaR synthesis ────────────────────────────────────────────────
run exp2_syn "Exp 2: STaR rejection-sampling synthesis (needs server)" \
  python -m examples.graph_2.synthesis.star_synthesize \
    --output-dir "$OUT"

# ── 7. Exp 2 — Train ─────────────────────────────────────────────────────────
run exp2_train "Exp 2: train cartridge (EXP=exp2, needs GPU)" \
  env CUDA_VISIBLE_DEVICES="$GPU" EXP=exp2 CARTRIDGE_TOKENS=512 \
  $TRAIN_CMD -m examples.graph_2.training.lineage_train

# ── 8. Exp 2 — Eval ──────────────────────────────────────────────────────────
EXP2_CKPT="$(find "$OUT/exp2_star/train" -name cache_last.pt 2>/dev/null | head -1 || true)"
if [ -n "$EXP2_CKPT" ]; then
  run exp2_eval "Exp 2: cartridge eval" \
    env CUDA_VISIBLE_DEVICES="$GPU" \
    python -m examples.graph_2.evaluation.lineage_eval \
      --mode cartridge \
      --checkpoint "$EXP2_CKPT" \
      --output "$OUT/exp2_star/train/eval/results.json"
else
  echo "── exp2_eval skipped: no checkpoint found under $OUT/exp2_star/train ──"
fi

# ── 9. Compare ────────────────────────────────────────────────────────────────
RESULT_FILES=()
[ -f "$OUT/exp0_icl/results.json" ]                && RESULT_FILES+=("$OUT/exp0_icl/results.json")
[ -f "$OUT/exp1_selfstudy/train/eval/results.json" ] && RESULT_FILES+=("$OUT/exp1_selfstudy/train/eval/results.json")
[ -f "$OUT/exp2_star/train/eval/results.json" ]    && RESULT_FILES+=("$OUT/exp2_star/train/eval/results.json")

if [ "${#RESULT_FILES[@]}" -ge 2 ]; then
  run compare "Cross-experiment comparison" \
    python -m examples.graph_2.evaluation.analyze "${RESULT_FILES[@]}"
else
  echo "── compare skipped: fewer than 2 result files available ──"
fi

echo ""
echo "Pipeline complete.  Results in $OUT"
