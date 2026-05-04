"""
Soundness experiment: does the cartridge actually compress the graph?

Key design — train and eval use IDENTICAL format:
  Train: [system: family tree] [user: MC question with options] [assistant: <think>reasoning</think> A]
  Eval:  [user: same MC question, SAME options] — NO system_prompt, cartridge only

Fixed options per pair stored in parquet → zero format drift between train/eval.

Requires:
    python examples/graph/graph_generate_mc.py --with-context
    # → train_dataset_mc_ctx.parquet, val_dataset_mc_ctx.parquet

Three W&B runs to compare:
    1. This script (cartridge, no context at eval)        → graph_soundness/hop_N_score
    2. Exp 3 upper bound (cartridge + context at eval)    → graph_accuracy_with_ctx/hop_N_score
    3. baseline_graph.py ICL (no cartridge, context)      → hop_N_score

Usage:
    python examples/graph/graph_train_soundness.py
"""
import os
from pathlib import Path

import pydrantic

from cartridges.initialization.text import KVFromText
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.train import TrainConfig, GenerationEvalConfig
from cartridges.datasets import DataSource, TrainDataset
from cartridges.utils.wandb import WandBConfig

from examples.graph.graph_evals import GraphRelationshipMCEvalDataset

GRAPH_DIR = Path(__file__).parent
TRAIN_DATASET_PATH = str(GRAPH_DIR / "train_dataset_mc_ctx.parquet")
VAL_DATASET_PATH   = str(GRAPH_DIR / "val_dataset_mc_ctx.parquet")
TREE_TEXT_PATH     = str(GRAPH_DIR / "family_tree.txt")

NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "1024"))
LR         = float(os.environ.get("LR", "2e-2"))
EPOCHS     = int(os.environ.get("EPOCHS", "20"))

config = TrainConfig(
    name=f"graph_soundness_kvtext_{NUM_TOKENS}tok_lr{LR}_e{EPOCHS}",
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),

    model=HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-1.7B",
        model_cls=FlexQwen3ForCausalLM,
    ),
    kv_cache_initializer=KVFromText.Config(
        max_tokens=NUM_TOKENS,
        text_source=TREE_TEXT_PATH,
    ),

    dataset=TrainDataset.Config(
        data_sources=[
            DataSource(path=TRAIN_DATASET_PATH, type="local"),
        ],
        targets="tokens",
        top_k_logits=0,
        packed_seq_length=1024,
        packing_mode="truncate",
    ),

    lr=LR,
    epochs=EPOCHS,
    global_batch_size=8,

    generate_eval_every_n_steps=100,
    generate_before_training=True,
    generate_evals=[
        GenerationEvalConfig(
            # Same val parquet, same fixed options, NO system_prompt → cartridge only
            dataset=GraphRelationshipMCEvalDataset.Config(
                val_path=VAL_DATASET_PATH,
                use_fixed_options=True,
                include_system_prompt=False,
            ),
            name_for_wandb="graph_soundness",
            generate_max_new_tokens=1024,
            temperature=0.0,
            batch_size=4,
            num_samples=1,
        ),
    ],

    save_every_n_steps=200,
    save_after_training=True,
    save_to_wandb=True,
    wandb=WandBConfig(tags=["train", "graph", "soundness"]),

    distributed_backend="gloo",
    seed=42,
)


if __name__ == "__main__":
    pydrantic.main(config)
