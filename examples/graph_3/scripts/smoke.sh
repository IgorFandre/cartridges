#!/usr/bin/env bash
# Fast end-to-end wiring check — no large GPU jobs, tiny data.
#
# Does NOT need a Tokasaurus server (synthesis stages are optional).
# Tests: forest gen, QA gen, Exp-0 ICL (small subset), analyze.
#
# Usage (from repo root, after sourcing env.sh):
#   CUDA_VISIBLE_DEVICES=0 bash examples/graph_3/scripts/smoke.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
OUT="${CARTRIDGES_OUTPUT_DIR_GRAPH3:-$REPO_ROOT/outputs_graph3}/smoke"

echo "=== graph_3 smoke test ==="
echo "output dir: $OUT"
mkdir -p "$OUT"

# ── 1. Forest (tiny) ──────────────────────────────────────────────────────────
echo ""
echo "── forest: 2 components × 20 people ──"
python -m examples.graph_3.data_gen.generate_forest \
  --components 2 --component-size 20 --seed 7 \
  --out "$OUT/forest.json"

# ── 2. QA ─────────────────────────────────────────────────────────────────────
echo ""
echo "── QA gen ──"
python -m examples.graph_3.data_gen.qagen \
  --forest "$OUT/forest.json" \
  --out-dir "$OUT" \
  --test-per-hop 5 --train-per-hop 10 --seed 7

# ── 3. Exp 0 ICL (limit=16) ───────────────────────────────────────────────────
echo ""
echo "── exp0 ICL (limit=16) — needs GPU $GPU ──"
CUDA_VISIBLE_DEVICES="$GPU" \
python -m examples.graph_3.evaluation.eval \
  --mode icl \
  --forest   "$OUT/forest.json" \
  --test-meta "$OUT/test_meta.json" \
  --corpus-path "$OUT/corpus.txt" \
  --limit 16 \
  --output  "$OUT/exp0_smoke/results.json"

# ── 4. Analyze ────────────────────────────────────────────────────────────────
echo ""
echo "── analyze ──"
python -m examples.graph_3.evaluation.analyze \
  "$OUT/exp0_smoke/results.json"

echo ""
echo "=== smoke PASSED ==="
