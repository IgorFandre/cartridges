#!/usr/bin/env bash
# Exp 3: ICL baselines for kinship MC — structured corpus vs prose narrative,
# same test set, no few-shot.
#
# Usage:
#   bash examples/graph/scripts/run_icl_baseline.sh             # base tree
#   VARIANT=alex bash examples/graph/scripts/run_icl_baseline.sh   # a variant tree

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"      # examples/graph/scripts
GRAPH="$(cd "$HERE/.." && pwd)"            # examples/graph
ROOT="$(cd "$GRAPH/../.." && pwd)"         # repo root
cd "$ROOT"

VARIANT="${VARIANT:-}"
MODEL="${MODEL:-qwen1.7b}"
COT="${COT:-1}"
OUT_DIR="${OUT_DIR:-$ROOT/outputs_graph/exp3_icl}"

if [ -n "$VARIANT" ]; then
  DIR="$GRAPH/data/variants/$VARIANT"
  TEST_ARG="--variant-dir $DIR"
  TAG="$VARIANT"
else
  DIR="$GRAPH/data/base"
  TEST_ARG=""
  TAG="base"
fi
CORPUS="$DIR/family_tree_corpus.txt"
NARRATIVE="$DIR/family_tree_narrative.txt"

if [ ! -f "$NARRATIVE" ]; then
  echo "Generating narrative corpus…"
  python -m examples.graph.data_gen.generate_narrative --tree "$DIR/family_tree.json"
fi

MODE="icl"; [ "$COT" = "1" ] && MODE="icl-cot"
mkdir -p "$OUT_DIR/$TAG/corpus" "$OUT_DIR/$TAG/narrative"

echo "── corpus form: structured ─────────────────────────"
python -m examples.graph.evaluation.eval --mode "$MODE" --model "$MODEL" --n-shot 0 \
  --corpus-path "$CORPUS" $TEST_ARG --output "$OUT_DIR/$TAG/corpus/results.json"

echo "── corpus form: narrative ──────────────────────────"
python -m examples.graph.evaluation.eval --mode "$MODE" --model "$MODEL" --n-shot 0 \
  --corpus-path "$NARRATIVE" $TEST_ARG --output "$OUT_DIR/$TAG/narrative/results.json"

python -m examples.graph.evaluation.analyze \
  "$OUT_DIR/$TAG/corpus/results.json" \
  "$OUT_DIR/$TAG/narrative/results.json" | tee "$OUT_DIR/$TAG/compare.txt"
