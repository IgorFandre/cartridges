#!/usr/bin/env bash
# Exp 3 — train the rejection-sampled cartridge (terminal 6).
#
# Distills the exp3_rejection dataset (exp2 filtered to correct traces) into a
# TrainableCache. No server needed. Build the dataset first with run_exp3_build.sh.
#
#   CUDA_VISIBLE_DEVICES=0 bash examples/graph_3/scripts/run_exp3_train.sh
#   MAX_STEPS=400 CARTRIDGE_TOKENS=512 bash examples/graph_3/scripts/run_exp3_train.sh
#   NPROC=2 bash examples/graph_3/scripts/run_exp3_train.sh   # multi-GPU (torchrun)
#
# Tunables (env, with defaults):
#   CARTRIDGE_TOKENS=512   cartridge compression budget (empty = full corpus)
#   MAX_STEPS=200          optimizer steps
#   N_EVALS=4              intermediate generation evals during training
#   EVAL_LIMIT=            cap test set for faster intermediate evals
#   NPROC=1                GPUs for data-parallel torchrun
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/env.sh"
cd "$REPO_ROOT"

DSNAME="dataset.parquet"; [ "${STEPBYSTEP:-0}" = "1" ] && DSNAME="dataset_stepbystep.parquet"
DATASET="$CARTRIDGES_OUTPUT_DIR_GRAPH3/exp3_rejection/artifact/$DSNAME"
if [ ! -f "$DATASET" ]; then
  echo "ERROR: exp3 dataset not found: $DATASET"
  echo "Build it first:  STEPBYSTEP=${STEPBYSTEP:-0} bash $SCRIPT_DIR/run_exp3_build.sh"
  exit 1
fi

NPROC="${NPROC:-1}"
if [ "$NPROC" -gt 1 ]; then
  TRAIN_CMD=(torchrun --standalone --nproc_per_node="$NPROC")
else
  TRAIN_CMD=(python)
fi

TRAINSUB="train"; [ "${STEPBYSTEP:-0}" = "1" ] && TRAINSUB="train_stepbystep"
OUT="$CARTRIDGES_OUTPUT_DIR_GRAPH3/exp3_rejection/$TRAINSUB"
mkdir -p "$OUT"

echo "Exp 3 train · dataset=$DATASET · tokens=${CARTRIDGE_TOKENS:-512} · steps=${MAX_STEPS:-200} · stepbystep=${STEPBYSTEP:-0} · log → $OUT/train.log"
EXP=exp3 \
CARTRIDGE_TOKENS="${CARTRIDGE_TOKENS:-512}" \
MAX_STEPS="${MAX_STEPS:-200}" \
N_EVALS="${N_EVALS:-4}" \
STEPBYSTEP="${STEPBYSTEP:-0}" \
"${TRAIN_CMD[@]}" -m examples.graph_3.training.lineage_train \
  2>&1 | tee "$OUT/train.log"
