#!/usr/bin/env bash
# Fast end-to-end wiring check — no large GPU jobs, tiny data.
#
# Does NOT need a Tokasaurus server (synthesis stages are optional).
# Tests: tree gen, QA gen, Exp-0 ICL (small subset), analyze.
#
# Usage (from repo root, after sourcing env.sh):
#   CUDA_VISIBLE_DEVICES=0 bash examples/graph_2/scripts/smoke.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
OUT="${CARTRIDGES_OUTPUT_DIR_GRAPH2:-$REPO_ROOT/outputs_graph2}/smoke"

echo "=== graph_2 smoke test ==="
echo "output dir: $OUT"

# ── 1. Tree (small) ───────────────────────────────────────────────────────────
echo ""
echo "── tree: 8-generation spine + 40 people ──"
python -m examples.graph_2.data_gen.generate_tree \
  --depth 8 --n-people 40 --seed 7 \
  --out "$OUT/family_tree.json"

# ── 2. QA ─────────────────────────────────────────────────────────────────────
echo ""
echo "── QA gen ──"
python -m examples.graph_2.data_gen.lineage_qagen \
  --tree "$OUT/family_tree.json" \
  --out-dir "$OUT" \
  --test-frac 0.2 --seed 7

# ── 3. Exp 0 ICL (limit=16) ───────────────────────────────────────────────────
echo ""
echo "── exp0 ICL (limit=16) — needs GPU $GPU ──"
CUDA_VISIBLE_DEVICES="$GPU" \
python -m examples.graph_2.evaluation.lineage_eval \
  --mode icl \
  --test-parquet "$OUT/test_lineage.parquet" \
  --test-meta    "$OUT/test_meta.json" \
  --limit 16 \
  --corpus-path  "$OUT/family_tree_corpus.txt" \
  --output       "$OUT/exp0_smoke/results.json"

# ── 4. Analyze ────────────────────────────────────────────────────────────────
echo ""
echo "── analyze ──"
python -m examples.graph_2.evaluation.analyze \
  "$OUT/exp0_smoke/results.json"

echo ""
echo "=== smoke PASSED ==="
