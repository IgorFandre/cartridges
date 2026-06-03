#!/usr/bin/env bash
# Graph-experiment environment. Source me from the repo root:
#
#   source examples/graph/scripts/env.sh
#
# Sets the CARTRIDGES_* vars + TORCHDYNAMO_DISABLE and activates .venv.
# Pick the GPU per run yourself (1 card each), e.g.:
#
#   CUDA_VISIBLE_DEVICES=0 python -m examples.graph.training.train
#
# See README.md ("Running experiments") for each experiment's module + args.
# ─────────────────────────────────────────────────────────────────────────────

# locate repo root from this file (works when sourced under bash AND zsh)
if [ -n "${BASH_SOURCE:-}" ]; then _graph_self="${BASH_SOURCE[0]}"
else _graph_self="${(%):-%x}"; fi                       # zsh
REPO_ROOT="$(cd "$(dirname "$_graph_self")/../../.." && pwd)"

# activate the project venv if present and not already active
if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi

# env vars — only set if you haven't already exported them
export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$REPO_ROOT}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$REPO_ROOT/outputs}"             # required by cartridges/__init__
export CARTRIDGES_OUTPUT_DIR_GRAPH="${CARTRIDGES_OUTPUT_DIR_GRAPH:-$REPO_ROOT/outputs_graph}"
export CARTRIDGES_WANDB_PROJECT="${CARTRIDGES_WANDB_PROJECT:-cartridges-graph}"
export CARTRIDGES_WANDB_ENTITY="${CARTRIDGES_WANDB_ENTITY:-shalygin-04}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

echo "graph env ready · root=$CARTRIDGES_DIR · out=$CARTRIDGES_OUTPUT_DIR_GRAPH"
echo "  prefix each run with CUDA_VISIBLE_DEVICES=<n> (1 GPU each)"
