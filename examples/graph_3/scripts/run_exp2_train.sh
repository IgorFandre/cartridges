#!/usr/bin/env bash
# Exp 2 — train the plain (unfiltered) self-study cartridge (terminal 5).
#
# Distills the exp2_plain self-study dataset (KL on top-20 logprobs) into a
# TrainableCache. No server needed — this owns the GPU for training.
#
#   CUDA_VISIBLE_DEVICES=0 bash examples/graph_3/scripts/run_exp2_train.sh
#   MAX_STEPS=400 CARTRIDGE_TOKENS=512 bash examples/graph_3/scripts/run_exp2_train.sh
#   NPROC=2 bash examples/graph_3/scripts/run_exp2_train.sh   # multi-GPU (torchrun)
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

DATASET="$CARTRIDGES_OUTPUT_DIR_GRAPH3/exp2_plain/artifact/dataset.parquet"
if [ ! -f "$DATASET" ]; then
  echo "ERROR: exp2 dataset not found: $DATASET"
  echo "Run synthesis first:  bash $SCRIPT_DIR/run_exp2_synth.sh"
  exit 1
fi

NPROC="${NPROC:-1}"
if [ "$NPROC" -gt 1 ]; then
  TRAIN_CMD=(torchrun --standalone --nproc_per_node="$NPROC")
else
  TRAIN_CMD=(python)
fi

OUT="$CARTRIDGES_OUTPUT_DIR_GRAPH3/exp2_plain/train"
mkdir -p "$OUT"

echo "Exp 2 train · dataset=$DATASET · tokens=${CARTRIDGE_TOKENS:-512} · steps=${MAX_STEPS:-200} · log → $OUT/train.log"
EXP=exp2 \
CARTRIDGE_TOKENS="${CARTRIDGE_TOKENS:-512}" \
MAX_STEPS="${MAX_STEPS:-200}" \
N_EVALS="${N_EVALS:-4}" \
"${TRAIN_CMD[@]}" -m examples.graph_3.training.lineage_train \
  2>&1 | tee "$OUT/train.log"
