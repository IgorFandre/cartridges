#!/usr/bin/env bash
# graph_2-experiment environment (lineage cartridges).  Source from repo root:
#
#   source examples/graph_2/scripts/env.sh
#
# Sets CARTRIDGES_* vars, server URL vars, TORCHDYNAMO_DISABLE, activates .venv.
# GPU selection is per-run: prefix each training/eval command with CUDA_VISIBLE_DEVICES=<n>.
#
# Synthesis (Exp 1 & 2) also requires a running Tokasaurus server:
#   - Local:  tksrs model=Qwen/Qwen3-4b kv_cache_num_tokens='(512*1024)' max_top_logprobs=20
#   - Modal:  modal deploy infra/modal_deploy_tokasaurus.py → set CARTRIDGES_TOKASAURUS_URL
# ─────────────────────────────────────────────────────────────────────────────

if [ -n "${BASH_SOURCE:-}" ]; then _self="${BASH_SOURCE[0]}"
else _self="${(%):-%x}"; fi
REPO_ROOT="$(cd "$(dirname "$_self")/../../.." && pwd)"

if [ -z "${VIRTUAL_ENV:-}" ] && [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.venv/bin/activate"
fi

export CARTRIDGES_DIR="${CARTRIDGES_DIR:-$REPO_ROOT}"
export CARTRIDGES_OUTPUT_DIR="${CARTRIDGES_OUTPUT_DIR:-$REPO_ROOT/outputs}"
export CARTRIDGES_OUTPUT_DIR_GRAPH2="${CARTRIDGES_OUTPUT_DIR_GRAPH2:-$REPO_ROOT/outputs_graph2}"
export CARTRIDGES_WANDB_PROJECT="${CARTRIDGES_WANDB_PROJECT:-cartridges-graph2}"
export CARTRIDGES_WANDB_ENTITY="${CARTRIDGES_WANDB_ENTITY:-shalygin-04}"
export TORCHDYNAMO_DISABLE="${TORCHDYNAMO_DISABLE:-1}"

# Tokasaurus server for self-study synthesis
export CARTRIDGES_TOKASAURUS_URL="${CARTRIDGES_TOKASAURUS_URL:-http://localhost:8000}"
export LINEAGE_SERVER_MODEL="${LINEAGE_SERVER_MODEL:-Qwen/Qwen3-4b}"

echo "graph_2 env ready · root=$CARTRIDGES_DIR · out=$CARTRIDGES_OUTPUT_DIR_GRAPH2"
echo "  server=$CARTRIDGES_TOKASAURUS_URL model=$LINEAGE_SERVER_MODEL"
echo "  prefix training/eval runs with CUDA_VISIBLE_DEVICES=<n>"
