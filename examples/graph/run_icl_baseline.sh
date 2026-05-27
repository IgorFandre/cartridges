#!/usr/bin/env bash
# Exp 3: ICL baselines for kinship MC.
# Compares two corpus formats on the same test set, no few-shot.
#
# Outputs:
#   icl_results_corpus.json
#   icl_results_narrative.json
#
# Usage:
#   bash examples/graph/run_icl_baseline.sh           # base tree
#   VARIANT=alex bash examples/graph/run_icl_baseline.sh   # variant tree

set -euo pipefail

cd "$(dirname "$0")"

VARIANT="${VARIANT:-}"
MODEL="${MODEL:-qwen1.7b}"
COT="${COT:-1}"

if [ -n "$VARIANT" ]; then
  DIR="variants/$VARIANT"
  CORPUS="$DIR/family_tree_corpus.txt"
  NARRATIVE="$DIR/family_tree_narrative.txt"
  TEST_ARG="--variant-dir $DIR"
  TAG="$VARIANT"
else
  CORPUS="family_tree_corpus.txt"
  NARRATIVE="family_tree_narrative.txt"
  TEST_ARG=""
  TAG="base"
fi

if [ ! -f "$NARRATIVE" ]; then
  echo "Generating narrative corpus..."
  if [ -n "$VARIANT" ]; then
    python generate_narrative_corpus.py --tree "variants/$VARIANT/family_tree.json"
  else
    python generate_narrative_corpus.py
  fi
fi

if [ "$COT" = "1" ]; then
  MODE="icl-cot"
else
  MODE="icl"
fi

echo "── corpus form: structured ─────────────────────────"
python graph_eval.py --mode "$MODE" --model "$MODEL" --n-shot 0 \
  --corpus-path "$CORPUS" $TEST_ARG \
  --output "icl_results_${TAG}_corpus.json"

echo "── corpus form: narrative ──────────────────────────"
python graph_eval.py --mode "$MODE" --model "$MODEL" --n-shot 0 \
  --corpus-path "$NARRATIVE" $TEST_ARG \
  --output "icl_results_${TAG}_narrative.json"

echo
echo "Done. Compare with: python analyze_results.py icl_results_${TAG}_corpus.json icl_results_${TAG}_narrative.json"
