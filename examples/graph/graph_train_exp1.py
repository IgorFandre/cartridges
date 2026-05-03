"""
Experiment 1: large cartridge (4096 tokens) + more epochs (100).

Hypothesis: pure Q&A memorization without graph in context
can work if cartridge has enough capacity.

No graph text in context — only Q&A pairs.
Random KV init.

Usage:
    python examples/graph/graph_train_exp1.py
"""
import os
from pathlib import Path

import pydrantic

from cartridges.initialization.random import KVFromRandomVectors
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.train import TrainConfig, GenerationEvalConfig
from cartridges.datasets import DataSource, TrainDataset
from cartridges.utils.wandb import WandBConfig

from examples.graph.graph_evals import GraphRelationshipMCEvalDataset

GRAPH_DIR = Path(__file__).parent
TRAIN_DATASET_PATH = str(GRAPH_DIR / "train_dataset.parquet")
VAL_DATASET_PATH = str(GRAPH_DIR / "val_dataset.parquet")

NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "4096"))
LR = float(os.environ.get("LR", "1e-3"))
EPOCHS = int(os.environ.get("EPOCHS", "100"))

config = TrainConfig(
    name=f"graph_exp1_random_{NUM_TOKENS}tok_lr{LR}_e{EPOCHS}",
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),

    model=HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-1.7B",
        model_cls=FlexQwen3ForCausalLM,
    ),
    kv_cache_initializer=KVFromRandomVectors.Config(
        max_tokens=NUM_TOKENS,
        num_frozen_tokens=1,
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

    generate_eval_every_n_steps=200,
    generate_before_training=True,
    generate_evals=[
        GenerationEvalConfig(
            dataset=GraphRelationshipMCEvalDataset.Config(
                val_path=VAL_DATASET_PATH,
            ),
            name_for_wandb="graph_accuracy",
            generate_max_new_tokens=1024,
            temperature=0.0,
            batch_size=4,
            num_samples=1,
        ),
    ],

    save_every_n_steps=500,
    save_after_training=True,
    save_to_wandb=True,
    wandb=WandBConfig(tags=["train", "graph", "exp1_large_cartridge"]),

    distributed_backend="gloo",
    seed=42,
)


if __name__ == "__main__":
    pydrantic.main(config)
