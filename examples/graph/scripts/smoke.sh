#!/usr/bin/env bash
# Fast end-to-end smoke of the CARTRIDGE pipeline. Trains two tiny cartridges
# (alex, ben) and runs every downstream step, printing PASS/FAIL per step and
# CONTINUING past failures so all errors surface in one run.
#
#   CUDA_VISIBLE_DEVICES=0 bash examples/graph/scripts/smoke.sh
#
# Writes to outputs_graph/_smoke (override SMOKE_OUT). W&B is offline by default.
# rm -rf the smoke dir when done. Exit code = number of failed steps.
set -uo pipefail   # deliberately NOT -e: we want to run every step

HERE="$(cd "$(dirname "$0")" && pwd)"   # examples/graph/scripts
GRAPH="$(cd "$HERE/.." && pwd)"         # examples/graph
ROOT="$(cd "$GRAPH/../.." && pwd)"      # repo root
cd "$ROOT"

export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$ROOT}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$ROOT/outputs}"   # required by cartridges/__init__
SMOKE_OUT="${SMOKE_OUT:-$ROOT/outputs_graph/_smoke}"
export CARTRIDGES_OUTPUT_DIR_GRAPH="$SMOKE_OUT"
export WANDB_MODE="${WANDB_MODE:-offline}"               # don't pollute W&B for a smoke
export CARTRIDGES_WANDB_PROJECT="${CARTRIDGES_WANDB_PROJECT:-cartridges-graph}"
export CARTRIDGES_WANDB_ENTITY="${CARTRIDGES_WANDB_ENTITY:-local}"

# tiny training: 6 steps, ckpt every 2 (→ dynamics has 3 checkpoints), eval 16 samples
export MAX_STEPS="${MAX_STEPS:-6}" SAVE_EVERY="${SAVE_EVERY:-2}" EVAL_LIMIT="${EVAL_LIMIT:-16}"

VD="$GRAPH/data/variants"
rm -rf "$SMOKE_OUT"; mkdir -p "$SMOKE_OUT"
echo "smoke → $SMOKE_OUT  ·  GPU=${CUDA_VISIBLE_DEVICES:-<unset>}  ·  WANDB_MODE=$WANDB_MODE"

pass=0; fail=0; failed=()
step() {
  local name="$1"; shift
  echo; echo "────── $name"
  if "$@"; then echo "✅ $name"; pass=$((pass+1))
  else local rc=$?; echo "❌ $name (exit $rc)"; fail=$((fail+1)); failed+=("$name"); fi
}
ck() { find "$SMOKE_OUT/exp2_train/$1" -name cache_last.pt 2>/dev/null | head -1; }

# ── train two tiny cartridges ─────────────────────────────────────────────────
step "train alex" env MODE=variants VARIANT_ONLY=alex python -m examples.graph.training.train
step "train ben"  env MODE=variants VARIANT_ONLY=ben  python -m examples.graph.training.train

# ── comparison: init + trained + dynamics ─────────────────────────────────────
step "exp1 init compare" python -m examples.graph.comparison.compare --source init \
    --names alex,ben --spectra --out-dir "$SMOKE_OUT/exp1_init_kv"

step "exp2 trained compare" python -m examples.graph.comparison.compare --source trained \
    --ckpt-root "$SMOKE_OUT/exp2_train" --names alex,ben \
    --init-corpus "$VD/alex/family_tree_corpus.txt" --localize-names Alex,Ben --spectra \
    --out-dir "$SMOKE_OUT/exp2_train/compare"

step "dynamics alex" python -m examples.graph.comparison.dynamics \
    --run-dir "$SMOKE_OUT/exp2_train/alex" \
    --init-corpus "$VD/alex/family_tree_corpus.txt" --spectra \
    --out-dir "$SMOKE_OUT/exp2_train/alex/dynamics"

# ── eval: own accuracy + analyze + cross-variant + K/V ablation ───────────────
CKA="$(ck alex)"; CKB="$(ck ben)"
echo "checkpoints: alex=[$CKA] ben=[$CKB]"

step "eval alex (own)" python -m examples.graph.evaluation.eval --mode cartridge-cot \
    --checkpoint "$CKA" --variant-dir "$VD/alex" --limit 8 --max-new-tokens 128 \
    --output "$SMOKE_OUT/exp2_train/alex/eval/results.json"

step "analyze alex" python -m examples.graph.evaluation.analyze \
    "$SMOKE_OUT/exp2_train/alex/eval/results.json"

step "cross eval alex→ben" python -m examples.graph.evaluation.eval --mode cartridge-cot \
    --checkpoint "$CKA" --variant-dir "$VD/ben" --limit 8 --max-new-tokens 128 \
    --output "$SMOKE_OUT/exp2_train/alex/eval_on_ben/results.json"

step "ablate alex keys←ben" python -m examples.graph.evaluation.eval --mode cartridge-cot \
    --checkpoint "$CKA" --ablate-keys-from "$CKB" --variant-dir "$VD/alex" \
    --limit 8 --max-new-tokens 128 \
    --output "$SMOKE_OUT/exp2_train/alex/eval_ablate_keys/results.json"

# ── summary ───────────────────────────────────────────────────────────────────
echo; echo "════════ smoke: $pass passed, $fail failed ════════"
if [ "$fail" -eq 0 ]; then
  echo "✅ cartridge pipeline OK  →  $SMOKE_OUT"
else
  printf '❌ failed: %s\n' "${failed[@]}"
fi
echo "clean up:  rm -rf $SMOKE_OUT"
exit "$fail"
