"""
Exp 1 — BFS-path-guided self-study synthesis.

Runs the standard A/B self-study pipeline (SelfStudySynthesizer) with
LineageGraphResource supplying context = graph text + lineal path.
Bot B's answers carry top-k logprobs for KL-distillation training.

Requires a running Tokasaurus server (or Modal deployment).
Set $CARTRIDGES_TOKASAURUS_URL and $LINEAGE_SERVER_MODEL before running.

Output: {CARTRIDGES_OUTPUT_DIR_GRAPH2}/exp1_selfstudy/<run>/artifact/dataset.parquet

Usage:
    python -m examples.graph_2.synthesis.lineage_synthesize
    N_SAMPLES=2048 BATCH_SIZE=16 python -m examples.graph_2.synthesis.lineage_synthesize
"""
import os

import pydrantic
from pydrantic.variables import FormatStringVariable

from cartridges.synthesize import SynthesizeConfig
from cartridges.synthesizers.self_study import SelfStudySynthesizer
from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.utils.wandb import WandBConfig
from examples.graph_2 import paths
from examples.graph_2.synthesis.lineage_resource import LineageGraphResource

# ── Env knobs ─────────────────────────────────────────────────────────────────
SERVER_URL    = os.environ.get("CARTRIDGES_TOKASAURUS_URL",  "http://localhost:8000")
SERVER_MODEL  = os.environ.get("LINEAGE_SERVER_MODEL",       "Qwen/Qwen3-4b")
N_SAMPLES     = int(os.environ.get("N_SAMPLES",  "1024"))
BATCH_SIZE    = int(os.environ.get("BATCH_SIZE", "8"))
PARALLEL      = int(os.environ.get("PARALLEL",   "32"))
# Write under EXP1_DIR so lineage_train._train_parquet_for("exp1") finds it.
OUTPUT_DIR    = str(paths.EXP1_DIR)

# ── Config ─────────────────────────────────────────────────────────────────────
client = TokasaurusClient.Config(
    url=SERVER_URL,
    model_name=SERVER_MODEL,
)

config = SynthesizeConfig(
    synthesizer=SelfStudySynthesizer.Config(
        client=client,
        max_rounds=1,
        prob_thinking=0.3,
        temperature_a=0.7,
        temperature_b=0.0,
        max_completion_tokens_a=256,
        max_completion_tokens_b=1024,
        num_top_logprobs=20,
        min_prob_mass=0.99,
        tools=[],
        resources=[
            LineageGraphResource.Config(
                tree_path=str(paths.BASE_TREE_JSON),
                corpus_path=str(paths.BASE_CORPUS),
            )
        ],
    ),
    num_samples=N_SAMPLES,
    batch_size=BATCH_SIZE,
    max_num_batches_in_parallel=PARALLEL,
    name=FormatStringVariable("lineage_exp1_selfstudy_n{num_samples}"),
    output_dir=OUTPUT_DIR,
    upload_to_wandb=False,
    save_wandb_preview=True,
    wandb=WandBConfig(tags=["lineage", "exp1", "selfstudy"]),
)

if __name__ == "__main__":
    pydrantic.main([config])
