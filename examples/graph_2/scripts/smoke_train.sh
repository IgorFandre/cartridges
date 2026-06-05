#!/usr/bin/env bash
# Training + eval smoke test — NO Tokasaurus server required.
#
# Verifies the full train + generation-eval + cartridge-eval wiring by:
#   1. generating a tiny tree + QA
#   2. fabricating a small self-study parquet from QA (no server)
#   3. training a cartridge for a few steps (TARGETS=tokens, plain CE)
#   4. running cartridge eval on the held-out test set
#
# This does NOT test the logprob-distillation path (needs a server) — it proves
# the training loop, intermediate generation evals, checkpointing, and cartridge
# eval all run end-to-end on a GPU.
#
# Usage (from repo root, after sourcing env.sh):
#   CUDA_VISIBLE_DEVICES=0 bash examples/graph_2/scripts/smoke_train.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
OUT="${CARTRIDGES_OUTPUT_DIR_GRAPH2:-$REPO_ROOT/outputs_graph2}/smoke_train"
DATA="$OUT/data"
SMOKE_PARQUET="$OUT/artifact/dataset.parquet"

echo "=== graph_2 TRAIN+EVAL smoke (no server) ==="
echo "output: $OUT"

# ── 1. Tiny tree ──────────────────────────────────────────────────────────────
echo ""; echo "── tree (depth 8, 40 people) ──"
python -m examples.graph_2.data_gen.generate_tree \
  --depth 8 --n-people 40 --seed 7 --out "$DATA/family_tree.json"

# ── 2. QA ─────────────────────────────────────────────────────────────────────
echo ""; echo "── QA ──"
python -m examples.graph_2.data_gen.lineage_qagen \
  --tree "$DATA/family_tree.json" --out-dir "$DATA" --test-frac 0.2 --seed 7

# ── 3. Fabricate self-study parquet (no server) ───────────────────────────────
echo ""; echo "── fabricate smoke self-study dataset ──"
python -m examples.graph_2.synthesis.make_smoke_dataset \
  --train-meta "$DATA/train_meta.json" \
  --out "$SMOKE_PARQUET" \
  --limit 96

# ── 4. Train (tokens CE, few steps, ~3 intermediate evals) ────────────────────
echo ""; echo "── train cartridge (TARGETS=tokens, MAX_STEPS=6) — needs GPU $GPU ──"
CUDA_VISIBLE_DEVICES="$GPU" \
EXP=exp1 \
TARGETS=tokens \
TRAIN_PARQUET="$SMOKE_PARQUET" \
INIT_CORPUS="$DATA/family_tree_corpus.txt" \
TEST_PARQUET="$DATA/test_lineage.parquet" \
CARTRIDGE_TOKENS=256 \
MAX_STEPS=6 \
N_EVALS=2 \
EVAL_LIMIT=16 \
EPOCHS=4 \
CARTRIDGES_OUTPUT_DIR_GRAPH2="$OUT" \
python -m examples.graph_2.training.lineage_train

# ── 5. Cartridge eval ─────────────────────────────────────────────────────────
CKPT="$(find "$OUT" -name cache_last.pt 2>/dev/null | xargs ls -t 2>/dev/null | head -1 || true)"
if [ -z "$CKPT" ]; then
  echo "!! no checkpoint produced — training smoke FAILED"
  exit 1
fi
echo ""; echo "── cartridge eval (limit 16) — checkpoint: $CKPT ──"
CUDA_VISIBLE_DEVICES="$GPU" \
python -m examples.graph_2.evaluation.lineage_eval \
  --mode cartridge \
  --checkpoint "$CKPT" \
  --test-parquet "$DATA/test_lineage.parquet" \
  --test-meta    "$DATA/test_meta.json" \
  --limit 16 \
  --max-new-tokens 512 \
  --output "$OUT/eval/results.json"

# ── 6. Analyze ────────────────────────────────────────────────────────────────
echo ""; echo "── analyze ──"
python -m examples.graph_2.evaluation.analyze "$OUT/eval/results.json"

echo ""; echo "=== TRAIN+EVAL smoke PASSED ==="
