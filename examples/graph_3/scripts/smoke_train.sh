#!/usr/bin/env bash
# Training + eval smoke test — NO Tokasaurus server required.
#
# Verifies the full train + generation-eval + cartridge-eval wiring by:
#   1. generating a tiny forest + QA
#   2. fabricating a small self-study parquet from QA (no server)
#   3. training a cartridge for a few steps (TARGETS=tokens, plain CE)
#   4. running cartridge eval on the held-out test set
#
# This does NOT test the logprob-distillation path (needs a server) — it proves
# the training loop, intermediate generation evals, checkpointing, and cartridge
# eval all run end-to-end on a GPU.
#
# Usage (from repo root):
#   CUDA_VISIBLE_DEVICES=0 bash examples/graph_3/scripts/smoke_train.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
cd "$REPO_ROOT"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
OUT="$CARTRIDGES_OUTPUT_DIR_GRAPH3/smoke_train"
DATA="$OUT/data"
SMOKE_PARQUET="$OUT/artifact/dataset.parquet"

echo "=== graph_3 TRAIN+EVAL smoke (no server) ==="
echo "output: $OUT"
mkdir -p "$DATA"

# ── 1. Tiny forest ────────────────────────────────────────────────────────────
echo ""; echo "── forest (2 components × 20 people) ──"
python -m examples.graph_3.data_gen.generate_forest \
  --components 2 --component-size 20 --seed 7 --out "$DATA/forest.json"

# ── 2. QA ─────────────────────────────────────────────────────────────────────
echo ""; echo "── QA ──"
python -m examples.graph_3.data_gen.qagen \
  --forest "$DATA/forest.json" --out-dir "$DATA" \
  --test-per-hop 5 --train-per-hop 10 --seed 7

# ── 3. Fabricate self-study parquet (no server) ───────────────────────────────
echo ""; echo "── fabricate smoke self-study dataset ──"
python -m examples.graph_3.synthesis.make_smoke_dataset \
  --train-meta "$DATA/train_meta.json" \
  --out "$SMOKE_PARQUET" \
  --limit 96

# ── 4. Train (tokens CE, few steps, ~2 intermediate evals) ────────────────────
echo ""; echo "── train cartridge (TARGETS=tokens, MAX_STEPS=6) — needs GPU $GPU ──"
CUDA_VISIBLE_DEVICES="$GPU" \
EXP=exp1 \
TARGETS=tokens \
TRAIN_PARQUET="$SMOKE_PARQUET" \
INIT_CORPUS="$DATA/corpus.txt" \
TEST_PARQUET="$DATA/test_handshake.parquet" \
CARTRIDGE_TOKENS=256 \
MAX_STEPS=6 \
N_EVALS=2 \
EVAL_LIMIT=16 \
EPOCHS=4 \
CARTRIDGES_OUTPUT_DIR_GRAPH3="$OUT" \
python -m examples.graph_3.training.lineage_train

# ── 5. Cartridge eval ─────────────────────────────────────────────────────────
CKPT="$(find "$OUT" -name cache_last.pt 2>/dev/null | xargs ls -t 2>/dev/null | head -1 || true)"
if [ -z "$CKPT" ]; then
  echo "!! no checkpoint produced — training smoke FAILED"
  exit 1
fi
echo ""; echo "── cartridge eval (limit 16) — checkpoint: $CKPT ──"
CUDA_VISIBLE_DEVICES="$GPU" \
python -m examples.graph_3.evaluation.eval \
  --mode cartridge \
  --checkpoint "$CKPT" \
  --forest    "$DATA/forest.json" \
  --test-meta "$DATA/test_meta.json" \
  --limit 16 \
  --max-new-tokens 512 \
  --output "$OUT/eval/results.json"

# ── 6. Analyze ────────────────────────────────────────────────────────────────
echo ""; echo "── analyze ──"
python -m examples.graph_3.evaluation.analyze "$OUT/eval/results.json"

echo ""; echo "=== TRAIN+EVAL smoke PASSED ==="
