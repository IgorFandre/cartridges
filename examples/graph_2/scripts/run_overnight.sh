#!/usr/bin/env bash
# Overnight pipeline: collect BOTH self-study datasets, train BOTH cartridges,
# evaluate (Exp0 ICL + Exp1 + Exp2), and compare.  Requires a running Tokasaurus
# server (Qwen3-1.7B) for the two synthesis stages.
#
# Robust by design: each stage logs to its own file, a failed stage is recorded
# but does NOT abort the run, and a summary is printed at the end.
#
# Usage (from repo root):
#   source examples/graph_2/scripts/env.sh
#   # server on one card, train/eval on another:
#   CUDA_VISIBLE_DEVICES=1 tksrs model=Qwen/Qwen3-1.7B kv_cache_num_tokens='(512 * 1024)' max_top_logprobs=20 &
#   GPU=0 nohup bash examples/graph_2/scripts/run_overnight.sh > overnight.out 2>&1 &
#
# Tunables (env, with defaults):
#   GPU=0              card for training/eval (server uses its own card)
#   N_SAMPLES=4096     exp1 self-study conversations
#   SYN_BATCH=16       synthesis batch size
#   ATTEMPTS=3         exp2 STaR attempts (1 initial + retries)
#   CARTRIDGE_TOKENS=512   cartridge compression budget
#   MAX_STEPS=400      optimizer steps per training run
#   N_EVALS=4          intermediate generation evals during training
#   SKIP_data=1 / SKIP_exp1_syn=1 / ...   skip individual stages
# ─────────────────────────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
cd "$REPO_ROOT"

GPU="${GPU:-${CUDA_VISIBLE_DEVICES:-0}}"
OUT="${CARTRIDGES_OUTPUT_DIR_GRAPH2:-$REPO_ROOT/outputs_graph2}"

# Tunables
N_SAMPLES="${N_SAMPLES:-4096}"
SYN_BATCH="${SYN_BATCH:-16}"
ATTEMPTS="${ATTEMPTS:-3}"
CARTRIDGE_TOKENS="${CARTRIDGE_TOKENS:-512}"
MAX_STEPS="${MAX_STEPS:-400}"
N_EVALS="${N_EVALS:-4}"

# Stamp without Date.now in-script: use shell date (allowed here)
STAMP="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$OUT/overnight_$STAMP/logs"
mkdir -p "$LOGDIR"

echo "════════════════════════════════════════════════════════════════"
echo " graph_2 OVERNIGHT run · $STAMP"
echo "   GPU(train/eval)=$GPU   server=$CARTRIDGES_TOKASAURUS_URL ($LINEAGE_SERVER_MODEL)"
echo "   N_SAMPLES=$N_SAMPLES  ATTEMPTS=$ATTEMPTS  CARTRIDGE_TOKENS=$CARTRIDGE_TOKENS  MAX_STEPS=$MAX_STEPS"
echo "   logs → $LOGDIR"
echo "════════════════════════════════════════════════════════════════"

declare -a STAGE_NAMES STAGE_STATUS
declare -A STATUS         # name → OK|FAIL|SKIP  (for dependency checks)

_record() { STAGE_NAMES+=("$1"); STAGE_STATUS+=("$2"); STATUS["$1"]="${2%% *}"; }

# run_stage <name> [--needs DEP] <command...>  — logs, times, records status,
# never aborts the run.  With --needs DEP, skips if DEP didn't finish OK.
run_stage() {
  local name="$1"; shift
  local need=""
  if [ "${1:-}" = "--needs" ]; then need="$2"; shift 2; fi

  local skip_var="SKIP_${name}"
  if [ "${!skip_var:-0}" = "1" ]; then
    echo "── SKIP $name (SKIP_$name=1) ──"; _record "$name" "SKIP"; return 0
  fi
  if [ -n "$need" ] && [ "${STATUS[$need]:-}" != "OK" ]; then
    echo "── SKIP $name (depends on $need, which is ${STATUS[$need]:-missing}) ──"
    _record "$name" "SKIP dep:$need"; return 0
  fi

  local log="$LOGDIR/$name.log"
  echo ""
  echo "▶ [$name]  $(date +%H:%M:%S)  → $log"
  local t0=$SECONDS
  if "$@" > "$log" 2>&1; then
    local dt=$((SECONDS - t0))
    echo "✓ [$name]  done in ${dt}s"; _record "$name" "OK ${dt}s"
  else
    local rc=$?; local dt=$((SECONDS - t0))
    echo "✗ [$name]  FAILED rc=$rc after ${dt}s  (see $log)"
    echo "   ── tail of $name.log ──"
    tail -n 15 "$log" | sed 's/^/   /'
    _record "$name" "FAIL rc=$rc"
  fi
}

find_ckpt() { find "$1" -name cache_last.pt 2>/dev/null | xargs ls -t 2>/dev/null | head -1; }

# ── Server preflight ─────────────────────────────────────────────────────────
# Synthesis needs a reachable Tokasaurus server whose served model matches
# LINEAGE_SERVER_MODEL.  Check once; if it's down, the synth stages are skipped
# (and the train stages that depend on them), so the night isn't wasted on the
# parts that can't possibly work.
SERVER_OK=0
if curl -fsS --max-time 10 "$CARTRIDGES_TOKASAURUS_URL/v1/models" >/tmp/g2_models.$$ 2>/dev/null; then
  SERVER_OK=1
  echo "Server reachable at $CARTRIDGES_TOKASAURUS_URL · /v1/models:"
  sed 's/^/   /' /tmp/g2_models.$$ | head -3
  if ! grep -q "$LINEAGE_SERVER_MODEL" /tmp/g2_models.$$ 2>/dev/null; then
    echo "   ⚠ WARNING: served model does not look like LINEAGE_SERVER_MODEL=$LINEAGE_SERVER_MODEL"
    echo "     The TokasaurusClient will reject a mismatch — make them equal."
  fi
  rm -f /tmp/g2_models.$$
else
  echo "⚠ Server NOT reachable at $CARTRIDGES_TOKASAURUS_URL — synthesis stages will be skipped."
  echo "  Start it on its own card, e.g.:"
  echo "    CUDA_VISIBLE_DEVICES=1 tksrs model=$LINEAGE_SERVER_MODEL kv_cache_num_tokens='(512 * 1024)' max_top_logprobs=20"
fi
# Force-skip synth if the server is down (unless explicitly already skipped).
if [ "$SERVER_OK" != "1" ]; then
  export SKIP_exp1_syn="${SKIP_exp1_syn:-1}"
  export SKIP_exp2_syn="${SKIP_exp2_syn:-1}"
fi

# ── 1. Data ───────────────────────────────────────────────────────────────────
run_stage data bash -c "
  python -m examples.graph_2.data_gen.generate_tree --depth 8 --n-people 100 --seed 42 &&
  python -m examples.graph_2.data_gen.lineage_qagen --test-frac 0.2 --seed 42
"

# ── 2. Exp 1 synthesis (graph + BFS path) — needs server ─────────────────────
run_stage exp1_syn env N_SAMPLES="$N_SAMPLES" BATCH_SIZE="$SYN_BATCH" \
  python -m examples.graph_2.synthesis.lineage_synthesize

# ── 3. Exp 2 synthesis (STaR + honest ICL accuracy) — needs server ───────────
run_stage exp2_syn env ATTEMPTS="$ATTEMPTS" BATCH_SIZE=32 \
  python -m examples.graph_2.synthesis.star_synthesize --output-dir "$OUT"

# ── 4. Exp 1 train (needs exp1_syn) ──────────────────────────────────────────
run_stage exp1_train --needs exp1_syn env CUDA_VISIBLE_DEVICES="$GPU" \
  EXP=exp1 CARTRIDGE_TOKENS="$CARTRIDGE_TOKENS" MAX_STEPS="$MAX_STEPS" N_EVALS="$N_EVALS" \
  python -m examples.graph_2.training.lineage_train

# ── 5. Exp 2 train (needs exp2_syn) ──────────────────────────────────────────
run_stage exp2_train --needs exp2_syn env CUDA_VISIBLE_DEVICES="$GPU" \
  EXP=exp2 CARTRIDGE_TOKENS="$CARTRIDGE_TOKENS" MAX_STEPS="$MAX_STEPS" N_EVALS="$N_EVALS" \
  python -m examples.graph_2.training.lineage_train

# ── 6. Exp 0 ICL baseline eval ───────────────────────────────────────────────
run_stage exp0_eval env CUDA_VISIBLE_DEVICES="$GPU" \
  python -m examples.graph_2.evaluation.lineage_eval --mode icl \
    --max-new-tokens 1024 --output "$OUT/exp0_icl/results.json"

# ── 7. Exp 1 cartridge eval ──────────────────────────────────────────────────
EXP1_CKPT="$(find_ckpt "$OUT/exp1_selfstudy/train")"
if [ "${STATUS[exp1_train]:-}" != "OK" ]; then
  echo "── SKIP exp1_eval (exp1_train is ${STATUS[exp1_train]:-missing}) ──"
  _record exp1_eval "SKIP dep:exp1_train"
elif [ -n "$EXP1_CKPT" ]; then
  run_stage exp1_eval env CUDA_VISIBLE_DEVICES="$GPU" \
    python -m examples.graph_2.evaluation.lineage_eval --mode cartridge \
      --checkpoint "$EXP1_CKPT" --max-new-tokens 1024 \
      --output "$OUT/exp1_selfstudy/eval/results.json"
else
  echo "── SKIP exp1_eval (no checkpoint under $OUT/exp1_selfstudy/train) ──"
  _record exp1_eval "SKIP no-ckpt"
fi

# ── 8. Exp 2 cartridge eval ──────────────────────────────────────────────────
EXP2_CKPT="$(find_ckpt "$OUT/exp2_star/train")"
if [ "${STATUS[exp2_train]:-}" != "OK" ]; then
  echo "── SKIP exp2_eval (exp2_train is ${STATUS[exp2_train]:-missing}) ──"
  _record exp2_eval "SKIP dep:exp2_train"
elif [ -n "$EXP2_CKPT" ]; then
  run_stage exp2_eval env CUDA_VISIBLE_DEVICES="$GPU" \
    python -m examples.graph_2.evaluation.lineage_eval --mode cartridge \
      --checkpoint "$EXP2_CKPT" --max-new-tokens 1024 \
      --output "$OUT/exp2_star/eval/results.json"
else
  echo "── SKIP exp2_eval (no checkpoint under $OUT/exp2_star/train) ──"
  _record exp2_eval "SKIP no-ckpt"
fi

# ── 9. Comparison ────────────────────────────────────────────────────────────
CMP_FILES=()
[ -f "$OUT/exp0_icl/results.json" ]            && CMP_FILES+=("$OUT/exp0_icl/results.json")
[ -f "$OUT/exp1_selfstudy/eval/results.json" ] && CMP_FILES+=("$OUT/exp1_selfstudy/eval/results.json")
[ -f "$OUT/exp2_star/eval/results.json" ]      && CMP_FILES+=("$OUT/exp2_star/eval/results.json")
if [ "${#CMP_FILES[@]}" -ge 1 ]; then
  run_stage compare python -m examples.graph_2.evaluation.analyze "${CMP_FILES[@]}"
fi

# ── Summary ───────────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo " OVERNIGHT SUMMARY · $STAMP → finished $(date +%H:%M:%S)"
echo "════════════════════════════════════════════════════════════════"
for i in "${!STAGE_NAMES[@]}"; do
  printf "  %-12s %s\n" "${STAGE_NAMES[$i]}" "${STAGE_STATUS[$i]}"
done
echo ""
echo "Datasets:"
echo "  exp1: $(find "$OUT/exp1_selfstudy" -name dataset.parquet 2>/dev/null | head -1)"
echo "  exp2: $OUT/exp2_star/artifact/dataset.parquet"
echo "Reports:"
echo "  exp2 ICL acc:  $OUT/exp2_star/icl_accuracy.json"
echo "  exp2 survival: $OUT/exp2_star/star_survival.json"
echo "  comparison:    $LOGDIR/compare.log"
echo "All logs: $LOGDIR"
