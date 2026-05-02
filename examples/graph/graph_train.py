"""
Trains a cartridge on family tree Q&A pairs.

The cartridge (KV-cache) is initialized with random vectors and trained
to encode the family tree structure purely through Q&A supervision.
No graph text is ever shown in the context — only questions and path-based answers.

Eval: accuracy broken down by chain length (hop_1, hop_2, hop_3, ...).

Usage (single GPU):
    python examples/graph/graph_train.py

Usage (multi-GPU):
    torchrun --standalone --nproc_per_node=2 examples/graph/graph_train.py
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

config = TrainConfig(
    name="graph_family_tree",
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),

    model=HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-1.7B",
        model_cls=FlexQwen3ForCausalLM,
    ),
    kv_cache_initializer=KVFromRandomVectors.Config(
        max_tokens=512,
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

    lr=1e-3,
    epochs=20,
    global_batch_size=8,

    generate_eval_every_n_steps=50,
    generate_before_training=True,
    generate_evals=[
        GenerationEvalConfig(
            dataset=GraphRelationshipMCEvalDataset.Config(
                val_path=VAL_DATASET_PATH,
            ),
            name_for_wandb="graph_accuracy",
            generate_max_new_tokens=1024,   # space for <think>...</think>
            temperature=0.0,
            batch_size=4,
            num_samples=1,
        ),
    ],

    save_every_n_steps=200,
    save_after_training=True,
    save_to_wandb=True,
    wandb=WandBConfig(tags=["train", "graph"]),

    distributed_backend="gloo",
    seed=42,
)


if __name__ == "__main__":
    pydrantic.main(config)
