#!/usr/bin/env bash
# Full pipeline for the kinship cartridge experiments.
#
# Code lives in examples/graph/{data_gen,training,evaluation,comparison}/ and is
# invoked as modules (python -m examples.graph.*). Data is generated under
# examples/graph/data/{base,variants}/. Results go to $OUT (default
# repo-root/outputs_graph), routed via $CARTRIDGES_OUTPUT_DIR_GRAPH:
#   exp1_init_kv/      init-KV compare + heatmaps
#   exp2_train/<v>/    trained cartridges + eval
#   exp2_train/compare/  trained-KV compare
#   exp3_icl/...       ICL baselines
#   exp4_stability/    (see run_exp4_stability.sh)
#
# Stages (skip with SKIP_<stage>=1):
#   tree qa variants narrative exp1_init exp2_train exp2_compare dynamics exp3_icl cart_eval
# Data is rebalanced + uniform-letter + hops by default; training defaults to
# MAX_STEPS=100 / SAVE_EVERY=20 (override via env). cart_eval also emits a
# cartridge-vs-ICL per-hop comparison when exp3 ICL results exist.
#
# Usage:
#   bash examples/graph/scripts/run_all.sh
#   SKIP_tree=1 SKIP_qa=1 SKIP_variants=1 SKIP_narrative=1 bash examples/graph/scripts/run_all.sh
#   NPROC=2 bash examples/graph/scripts/run_all.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"      # examples/graph/scripts
GRAPH="$(cd "$HERE/.." && pwd)"            # examples/graph
ROOT="$(cd "$GRAPH/../.." && pwd)"         # repo root
cd "$ROOT"

# ── Env ──────────────────────────────────────────────────────────────────────
: "${CARTRIDGES_DIR:=$ROOT}"
: "${OUT:=$ROOT/outputs_graph}"
export CARTRIDGES_DIR
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$OUT}"   # required by cartridges/__init__
export CARTRIDGES_OUTPUT_DIR_GRAPH="$OUT"   # paths.py routes train output here
: "${CARTRIDGES_WANDB_PROJECT:=cartridges-graph}"
: "${CARTRIDGES_WANDB_ENTITY:=local}"
export CARTRIDGES_WANDB_PROJECT CARTRIDGES_WANDB_ENTITY

N_PEOPLE="${N_PEOPLE:-45}"
MAX_DEPTH="${MAX_DEPTH:-6}"
FOUNDERS="${FOUNDERS:-1}"
SPOUSE_PROB="${SPOUSE_PROB:-0.95}"
N_VERIF_PER_REL="${N_VERIF_PER_REL:-12}"
NPROC="${NPROC:-1}"
VARIANTS=(alex ben carl dan)
DATA="$GRAPH/data"
BASE="$DATA/base"
VDIR="$DATA/variants"

mkdir -p "$OUT/exp1_init_kv" "$OUT/exp2_train/compare" \
         "$OUT/exp3_icl/base/corpus" "$OUT/exp3_icl/base/narrative"
for v in "${VARIANTS[@]}"; do
  mkdir -p "$OUT/exp2_train/$v/eval" "$OUT/exp2_train/$v/dynamics" \
           "$OUT/exp3_icl/$v/corpus" "$OUT/exp3_icl/$v/narrative"
done
echo "output dirs ready under: $OUT"

# ── Stage helpers ────────────────────────────────────────────────────────────
run() {
  local stage="$1"; shift
  local skip_var="SKIP_${stage}"
  if [ "${!skip_var:-0}" = "1" ]; then echo "── [SKIP] $stage ──"; return; fi
  echo; echo "════════════════════════════════════════"
  echo "── $stage"; echo "════════════════════════════════════════"
  "$@"
}

py()  { python -m "$@"; }
train_cmd() {
  if [ "$NPROC" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$NPROC" -m "$@"
  else
    python -m "$@"
  fi
}
names_csv() { local IFS=,; echo "${VARIANTS[*]}"; }
names_cap() { echo "Alex,Ben,Carl,Dan"; }

# ── 1. tree ──────────────────────────────────────────────────────────────────
stage_tree() {
  py examples.graph.data_gen.generate_tree \
    --n-people "$N_PEOPLE" --max-depth "$MAX_DEPTH" \
    --min-kids 1 --max-kids 2 --founders "$FOUNDERS" --spouse-prob "$SPOUSE_PROB"
}

# ── 2. qa (base tree) ────────────────────────────────────────────────────────
stage_qa() {
  py examples.graph.data_gen.qagen --test-frac 0.2 --n-verif-per-rel "$N_VERIF_PER_REL"
}

# ── 3. variants ──────────────────────────────────────────────────────────────
stage_variants() {
  py examples.graph.data_gen.generate_variants \
    --new-names "$(names_cap)" --test-frac 0.2 --n-verif-per-rel "$N_VERIF_PER_REL"
}

# ── 4. narrative ─────────────────────────────────────────────────────────────
stage_narrative() {
  py examples.graph.data_gen.generate_narrative
  for v in "${VARIANTS[@]}"; do
    py examples.graph.data_gen.generate_narrative --tree "$VDIR/$v/family_tree.json"
  done
}

# ── 5. exp1: init KV compare ─────────────────────────────────────────────────
stage_exp1_init() {
  py examples.graph.comparison.compare --source init \
    --names "$(names_csv)" --top-k 30 --spectra \
    --out-dir "$OUT/exp1_init_kv" \
    2>&1 | tee "$OUT/exp1_init_kv/run.log"
}

# ── 6. exp2: train ───────────────────────────────────────────────────────────
stage_exp2_train() {
  if [ -n "${VARIANT_ONLY:-}" ]; then
    MODE=variants VARIANT_ONLY="$VARIANT_ONLY" train_cmd examples.graph.training.train \
      2>&1 | tee "$OUT/exp2_train/$VARIANT_ONLY/run.log"
  else
    MODE=variants train_cmd examples.graph.training.train 2>&1 | tee "$OUT/exp2_train/run.log"
  fi
}

# ── 7. exp2: trained KV compare ──────────────────────────────────────────────
stage_exp2_compare() {
  py examples.graph.comparison.compare --source trained \
    --ckpt-root "$OUT/exp2_train" --names "$(names_csv)" \
    --init-corpus "$VDIR/alex/family_tree_corpus.txt" --localize-names "$(names_cap)" --spectra \
    --out-dir "$OUT/exp2_train/compare" \
    2>&1 | tee "$OUT/exp2_train/compare/run.log" \
    || echo "compare failed — check checkpoints exist"
}

# ── 7b. training dynamics: how K/V rotate during training ─────────────────────
stage_dynamics() {
  for v in "${VARIANTS[@]}"; do
    if ! find "$OUT/exp2_train/$v" -name 'cache-step*.pt' 2>/dev/null | grep -q .; then
      echo "no cache-step*.pt for $v — skip (train with low SAVE_EVERY)"; continue
    fi
    py examples.graph.comparison.dynamics \
      --run-dir "$OUT/exp2_train/$v" \
      --init-corpus "$VDIR/$v/family_tree_corpus.txt" --spectra \
      --out-dir "$OUT/exp2_train/$v/dynamics" \
      2>&1 | tee "$OUT/exp2_train/$v/dynamics/run.log"
  done
}

# ── 8. exp3: ICL (corpus vs narrative) ───────────────────────────────────────
icl_one() {  # <corpus> <variant-dir-or-empty> <out>
  local corpus="$1" vd="$2" out="$3"
  local extra=""; [ -n "$vd" ] && extra="--variant-dir $vd"
  py examples.graph.evaluation.eval --mode icl-cot --n-shot 0 \
    --corpus-path "$corpus" $extra --output "$out/results.json" \
    2>&1 | tee "$out/run.log"
}
stage_exp3_icl() {
  icl_one "$BASE/family_tree_corpus.txt"    "" "$OUT/exp3_icl/base/corpus"
  icl_one "$BASE/family_tree_narrative.txt" "" "$OUT/exp3_icl/base/narrative"
  py examples.graph.evaluation.analyze \
    "$OUT/exp3_icl/base/corpus/results.json" \
    "$OUT/exp3_icl/base/narrative/results.json" > "$OUT/exp3_icl/base/compare.txt"
  for v in "${VARIANTS[@]}"; do
    icl_one "$VDIR/$v/family_tree_corpus.txt"    "$VDIR/$v" "$OUT/exp3_icl/$v/corpus"
    icl_one "$VDIR/$v/family_tree_narrative.txt" "$VDIR/$v" "$OUT/exp3_icl/$v/narrative"
    py examples.graph.evaluation.analyze \
      "$OUT/exp3_icl/$v/corpus/results.json" \
      "$OUT/exp3_icl/$v/narrative/results.json" > "$OUT/exp3_icl/$v/compare.txt"
  done
}

# ── 9. cartridge eval (after training) ───────────────────────────────────────
stage_cart_eval() {
  for v in "${VARIANTS[@]}"; do
    local ckpt; ckpt="$(find "$OUT/exp2_train/$v" -name cache_last.pt 2>/dev/null | head -1)"
    if [ -z "$ckpt" ]; then echo "no checkpoint for $v — skip"; continue; fi
    py examples.graph.evaluation.eval --mode cartridge-cot \
      --checkpoint "$ckpt" --variant-dir "$VDIR/$v" \
      --output "$OUT/exp2_train/$v/eval/results.json" \
      2>&1 | tee "$OUT/exp2_train/$v/eval/run.log"
    py examples.graph.evaluation.analyze "$OUT/exp2_train/$v/eval/results.json" \
      > "$OUT/exp2_train/$v/eval/summary.txt"
    # cartridge vs ICL, per hop-class + per category (if ICL results exist)
    local icl="$OUT/exp3_icl/$v/corpus/results.json"
    if [ -f "$icl" ]; then
      py examples.graph.evaluation.analyze --no-detail \
        "$OUT/exp2_train/$v/eval/results.json" "$icl" \
        > "$OUT/exp2_train/$v/eval/vs_icl_by_hop.txt"
    fi
  done
}

# ── Run ──────────────────────────────────────────────────────────────────────
run tree         stage_tree
run qa           stage_qa
run variants     stage_variants
run narrative    stage_narrative
run exp1_init    stage_exp1_init
run exp2_train   stage_exp2_train
run exp2_compare stage_exp2_compare
run dynamics     stage_dynamics
run exp3_icl     stage_exp3_icl
run cart_eval    stage_cart_eval

echo; echo "Done. Results under $OUT"
