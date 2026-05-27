#!/usr/bin/env bash
# Full pipeline for kinship cartridge experiments.
#
# Stages (each can be skipped via SKIP_<STAGE>=1):
#   1. tree         — generate base family_tree.json
#   2. qa           — generate base QA (train_mc / test_mc parquets)
#   3. variants     — generate 4 swapped-name variants + their parquets
#   4. narrative    — generate narrative corpora (base + variants) for ICL
#   5. exp1_init    — compare INIT-state KV caches across variants (no training)
#   6. exp2_train   — train 4 cartridges (shared Alex init, masked-letter loss)
#   7. exp2_compare — compare trained KV caches across variants
#   8. exp3_icl     — ICL baselines (structured corpus vs narrative)
#
# Required env:
#   CARTRIDGES_DIR              (root of repo)
#   CARTRIDGES_OUTPUT_DIR       (checkpoints + outputs)
#   CARTRIDGES_WANDB_PROJECT    (optional but expected by configs)
#   CARTRIDGES_WANDB_ENTITY     (optional but expected by configs)
#
# Optional:
#   N_PEOPLE=45  MAX_DEPTH=5  N_VERIF_PER_REL=12
#   VARIANT_ONLY=alex          # for exp2_train run single variant
#   NPROC=1                    # set >1 to use torchrun for training
#   SKIP_<STAGE>=1             # skip specific stage
#
# Usage:
#   bash examples/graph/run_all.sh
#   SKIP_tree=1 SKIP_qa=1 bash examples/graph/run_all.sh
#   NPROC=2 bash examples/graph/run_all.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
cd "$ROOT"

: "${CARTRIDGES_DIR:=$ROOT}"
: "${CARTRIDGES_OUTPUT_DIR:=$ROOT/outputs}"
: "${CARTRIDGES_WANDB_PROJECT:=cartridges-graph}"
: "${CARTRIDGES_WANDB_ENTITY:=local}"
export CARTRIDGES_DIR CARTRIDGES_OUTPUT_DIR CARTRIDGES_WANDB_PROJECT CARTRIDGES_WANDB_ENTITY

N_PEOPLE="${N_PEOPLE:-45}"
MAX_DEPTH="${MAX_DEPTH:-5}"
N_VERIF_PER_REL="${N_VERIF_PER_REL:-12}"
NPROC="${NPROC:-1}"
VARIANTS=(alex ben carl dan)

GRAPH="$HERE"

run() {
  local stage="$1"; shift
  local skip_var="SKIP_${stage}"
  if [ "${!skip_var:-0}" = "1" ]; then
    echo "── [SKIP] $stage ──"
    return
  fi
  echo
  echo "════════════════════════════════════════"
  echo "── $stage"
  echo "════════════════════════════════════════"
  "$@"
}

train_cmd() {
  # Single-GPU vs torchrun
  if [ "$NPROC" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$NPROC" "$@"
  else
    python "$@"
  fi
}

# ── 1. tree ──────────────────────────────────────────────────────────────────
stage_tree() {
  python "$GRAPH/generate_tree.py" \
    --n-people "$N_PEOPLE" \
    --max-depth "$MAX_DEPTH" \
    --min-kids 1 --max-kids 2
}

# ── 2. qa (base tree) ────────────────────────────────────────────────────────
stage_qa() {
  python "$GRAPH/graph_qagen.py" \
    --test-frac 0.2 \
    --n-verif-per-rel "$N_VERIF_PER_REL"
}

# ── 3. variants ──────────────────────────────────────────────────────────────
stage_variants() {
  python "$GRAPH/generate_tree_variants.py" \
    --new-names "$(IFS=,; echo "${VARIANTS[*]^}")" \
    --test-frac 0.2 \
    --n-verif-per-rel "$N_VERIF_PER_REL"
}

# ── 4. narrative ─────────────────────────────────────────────────────────────
stage_narrative() {
  python "$GRAPH/generate_narrative_corpus.py"
  for v in "${VARIANTS[@]}"; do
    python "$GRAPH/generate_narrative_corpus.py" \
      --tree "$GRAPH/variants/$v/family_tree.json"
  done
}

# ── 5. exp1: init KV compare ─────────────────────────────────────────────────
stage_exp1_init() {
  python "$GRAPH/compare_init_kv.py" \
    --variants "$(IFS=,; echo "${VARIANTS[*]}")" \
    --top-k 30
}

# ── 6. exp2: train 4 cartridges (shared Alex init) ───────────────────────────
stage_exp2_train() {
  if [ -n "${VARIANT_ONLY:-}" ]; then
    VARIANT_ONLY="$VARIANT_ONLY" train_cmd "$GRAPH/graph_train_variants.py"
  else
    # Sequential — pydrantic accepts a list of configs.
    train_cmd "$GRAPH/graph_train_variants.py"
  fi
}

# ── 7. exp2: trained KV compare ──────────────────────────────────────────────
stage_exp2_compare() {
  if [ ! -f "$GRAPH/compare_kv.py" ]; then
    echo "compare_kv.py missing — skipping"
    return
  fi
  # compare_kv.py walks checkpoints under examples/graph/checkpoints_variants/
  python "$GRAPH/compare_kv.py" \
    --variants "$(IFS=,; echo "${VARIANTS[*]}")" || \
    echo "compare_kv.py failed — check checkpoint paths"
}

# ── 8. exp3: ICL baselines (corpus vs narrative) ─────────────────────────────
stage_exp3_icl() {
  # Base tree
  bash "$GRAPH/run_icl_baseline.sh"
  # Per-variant (optional — uncomment to run)
  # for v in "${VARIANTS[@]}"; do
  #   VARIANT="$v" bash "$GRAPH/run_icl_baseline.sh"
  # done
}

# ── Run all ──────────────────────────────────────────────────────────────────
run tree         stage_tree
run qa           stage_qa
run variants     stage_variants
run narrative    stage_narrative
run exp1_init    stage_exp1_init
run exp2_train   stage_exp2_train
run exp2_compare stage_exp2_compare
run exp3_icl     stage_exp3_icl

echo
echo "════════════════════════════════════════"
echo "Done. Outputs:"
echo "  $GRAPH/family_tree.json, train_mc.parquet, test_mc.parquet"
echo "  $GRAPH/variants/{alex,ben,carl,dan}/"
echo "  $GRAPH/kv_init_compare_out/                  (exp1)"
echo "  $CARTRIDGES_OUTPUT_DIR  + $GRAPH/checkpoints_variants/  (exp2)"
echo "  $GRAPH/icl_results_*_corpus.json, *_narrative.json     (exp3)"
echo "════════════════════════════════════════"
