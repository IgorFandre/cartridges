#!/usr/bin/env bash
# Exp 4: training-stability runs. Train the same (init, data) pair N times with
# different seeds to measure run-to-run noise in the trained cartridge — the
# floor against which exp1/exp2 divergences are judged.
#
# Default plan = two groups of 5 seeds each:
#   ben_on_ben  (init=ben, data=ben)  — same-data stability (pure noise floor)
#   ben_on_alex (init=ben, data=alex) — init≠data control
#
# Trains in parallel across GPUs, then runs the unified compare on each group.
#
# Usage:
#   bash examples/graph/scripts/run_exp4_stability.sh
#   GPUS="0 1 2 3 4 0 1 2 3 4" bash examples/graph/scripts/run_exp4_stability.sh
#   TASKS="ben ben 1;ben ben 2;..." GPUS="0 1 ..." bash ...   # custom plan
#   SKIP_TRAIN=1 bash examples/graph/scripts/run_exp4_stability.sh   # compare only

set -u

HERE="$(cd "$(dirname "$0")" && pwd)"
GRAPH="$(cd "$HERE/.." && pwd)"
ROOT="$(cd "$GRAPH/../.." && pwd)"
cd "$ROOT"

export CARTRIDGES_DIR="$ROOT"
: "${OUT:=$ROOT/outputs_graph}"
export CARTRIDGES_OUTPUT_DIR_GRAPH="$OUT"
: "${CARTRIDGES_WANDB_PROJECT:=cartridges-graph}"
: "${CARTRIDGES_WANDB_ENTITY:=local}"
export CARTRIDGES_WANDB_PROJECT CARTRIDGES_WANDB_ENTITY
export TORCHDYNAMO_DISABLE=1

N_RUNS="${N_RUNS:-5}"
EXP4="$OUT/exp4_stability"
mkdir -p "$EXP4"

# Tasks: "<INIT> <DATA> <run_idx>" separated by ';'. Default = 2 groups × 5 seeds.
DEFAULT_TASKS="ben ben 1;ben ben 2;ben ben 3;ben ben 4;ben ben 5;ben alex 1;ben alex 2;ben alex 3;ben alex 4;ben alex 5"
IFS=';' read -r -a TASKS <<< "${TASKS:-$DEFAULT_TASKS}"
read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 0 1 2 3 4}"

ts(){ date '+%H:%M:%S'; }
log(){ echo "===[$(ts)] $*==="; }

# ── Train ─────────────────────────────────────────────────────────────────────
if [ "${SKIP_TRAIN:-0}" != "1" ]; then
  if [ "${#GPUS[@]}" -ne "${#TASKS[@]}" ]; then
    echo "ERROR: GPUS (${#GPUS[@]}) must match TASKS (${#TASKS[@]})"; exit 1
  fi
  log "launching ${#TASKS[@]} stability runs"
  PIDS=()
  for i in "${!TASKS[@]}"; do
    read -r INIT DATA RUN <<< "${TASKS[$i]}"
    gpu="${GPUS[$i]}"
    tag="${INIT}_on_${DATA}_run${RUN}"
    log "  task $i: INIT=$INIT DATA=$DATA run=$RUN GPU=$gpu → $tag"
    (
      CUDA_VISIBLE_DEVICES="$gpu" TORCHDYNAMO_DISABLE=1 \
      MODE=stability INIT_VARIANT="$INIT" DATA_VARIANT="$DATA" \
      N_RUNS="$N_RUNS" RUN_ONLY="$RUN" \
        python -m examples.graph.training.train \
        2>&1 | tee "$EXP4/${tag}.log"
    ) &
    PIDS+=($!)
  done
  log "waiting on PIDs: ${PIDS[*]}"
  wait
  log "all runs done"
fi

# ── Compare each group (unique INIT_on_DATA prefixes) ────────────────────────
declare -A SEEN
for t in "${TASKS[@]}"; do
  read -r INIT DATA _ <<< "$t"
  prefix="${INIT}_on_${DATA}"
  [ -n "${SEEN[$prefix]:-}" ] && continue
  SEEN[$prefix]=1
  log "compare group $prefix"
  python -m examples.graph.comparison.compare --source trained \
    --ckpt-root "$EXP4" --run-prefix "$prefix" --n-runs "$N_RUNS" \
    --name-slots off \
    --out-dir "$EXP4/${prefix}_compare" \
    2>&1 | tee "$EXP4/${prefix}_compare/run.log" \
    || echo "compare $prefix failed — check checkpoints"
done

log "exp4 done → $EXP4"
