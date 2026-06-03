#!/usr/bin/env bash
# Compare already-trained stability cartridges with the unified compare CLI.
#
# Works against an existing results dir of trained runs (default exp4_stability/:
# alex_run{1..5}/, ben_on_alex_run1/, ben_on_ben_run1/, each holding
# <timestamp>/<uuid>/cache_last.pt). find_cache_last rglobs, so the pydrantic
# nesting is handled automatically.
#
# Produces, under $RES:
#   alex_stability_compare/   — 5 alex seeds, same data → run-to-run NOISE FLOOR
#   type_compare/             — alex vs ben_on_alex vs ben_on_ben pairwise;
#                               top-diverging slots = WHERE the entities live
#
# Usage:
#   bash examples/graph/scripts/compare_results.sh
#   RES=outputs_graph/exp4_stability bash examples/graph/scripts/compare_results.sh   # (default)
#   ALEX_RUNS=5 bash examples/graph/scripts/compare_results.sh

set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"      # examples/graph/scripts
GRAPH="$(cd "$HERE/.." && pwd)"            # examples/graph
ROOT="$(cd "$GRAPH/../.." && pwd)"         # repo root
cd "$ROOT"

export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$ROOT}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$ROOT/outputs_graph}"  # cartridges/__init__ requires this
export CARTRIDGES_OUTPUT_DIR_GRAPH="${CARTRIDGES_OUTPUT_DIR_GRAPH:-$ROOT/outputs_graph}"
RES="${RES:-$ROOT/outputs_graph/exp4_stability}"
ALEX_RUNS="${ALEX_RUNS:-5}"

[ -d "$RES" ] || { echo "results dir not found: $RES (set RES=...)"; exit 1; }
cmp() { python -m examples.graph.comparison.compare "$@"; }

# ── 1. Alex stability: 5 seeds, identical data → noise floor ──────────────────
echo "════ 1. alex stability (×$ALEX_RUNS seeds) ════"
cmp --source trained --ckpt-root "$RES" --run-prefix alex --n-runs "$ALEX_RUNS" \
    --name-slots off \
    --out-dir "$RES/alex_stability_compare" \
    2>&1 | tee "$RES/alex_stability_compare.log"

# ── 2. Pairwise across cartridge types → where entities live ──────────────────
# alex        : init=alex data=alex
# ben_on_alex : init=ben  data=alex   (vs alex → isolates INIT/name effect)
# ben_on_ben  : init=ben  data=ben    (vs ben_on_alex → isolates DATA effect)
echo "════ 2. type compare (alex | ben_on_alex | ben_on_ben) ════"
cmp --source trained \
    --inputs "alex=$RES/alex_run1,ben_on_alex=$RES/ben_on_alex_run1,ben_on_ben=$RES/ben_on_ben_run1" \
    --name-slots off --top-k 40 \
    --out-dir "$RES/type_compare" \
    2>&1 | tee "$RES/type_compare.log"

echo
echo "Done."
echo "  noise floor → $RES/alex_stability_compare/stability.json"
echo "  entity loc  → $RES/type_compare/localization.json  (per-pair top slots)"
echo "  read top-slot angles in type_compare against the alex_stability mean angle."
