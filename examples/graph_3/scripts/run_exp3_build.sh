#!/usr/bin/env bash
# Exp 3 — build the rejection-sampled dataset from exp2 (NO server).
#
# Pure filter: keeps only the exp2_plain traces whose `correct` flag is True.
# Run any time after exp2 synthesis — no model queries.
#
#   bash examples/graph_3/scripts/run_exp3_build.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
cd "$REPO_ROOT"

SBS_ARGS=(); DSNAME="dataset.parquet"
if [ "${STEPBYSTEP:-0}" = "1" ]; then SBS_ARGS=(--stepbystep); DSNAME="dataset_stepbystep.parquet"; fi

SOURCE="$CARTRIDGES_OUTPUT_DIR_GRAPH3/exp2_plain/artifact/$DSNAME"
if [ ! -f "$SOURCE" ]; then
  echo "ERROR: exp2 dataset not found: $SOURCE"
  echo "Run exp2 synthesis first:  STEPBYSTEP=${STEPBYSTEP:-0} bash $SCRIPT_DIR/run_exp2_synth.sh"
  exit 1
fi

OUT="$CARTRIDGES_OUTPUT_DIR_GRAPH3/exp3_rejection"
mkdir -p "$OUT"

echo "Exp 3 build · filter $SOURCE by correct=True · stepbystep=${STEPBYSTEP:-0} · log → $OUT/build.log"
python -m examples.graph_3.synthesis.exp3_build \
  --source "$SOURCE" \
  --output-dir "$CARTRIDGES_OUTPUT_DIR_GRAPH3" \
  "${SBS_ARGS[@]}" \
  2>&1 | tee "$OUT/build.log"
