"""
Experiment 2: graph text in system_prompt + KVFromText init.

Canonical cartridge paradigm applied to family tree:
  - Train: system_prompt = full family tree text, user/assistant = Q&A
  - KV init: forward pass on family tree text (KVFromText)
  - Eval: NO system_prompt — only cartridge

This tests whether the cartridge can compress the graph structure
the same way LongHealth/NIAH cartridges compress their corpora.

Requires: python examples/graph/graph_generate.py --with-context
          (generates train/val_dataset_ctx.parquet + family_tree.txt)

Usage:
    python examples/graph/graph_train_exp2.py
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
TRAIN_DATASET_PATH = str(GRAPH_DIR / "train_dataset_ctx.parquet")
VAL_DATASET_PATH = str(GRAPH_DIR / "val_dataset_ctx.parquet")
TREE_TEXT_PATH = str(GRAPH_DIR / "family_tree.txt")

NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "1024"))
LR = float(os.environ.get("LR", "2e-2"))
EPOCHS = int(os.environ.get("EPOCHS", "20"))

config = TrainConfig(
    name=f"graph_exp2_kvtext_{NUM_TOKENS}tok_lr{LR}_e{EPOCHS}",
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
            # val dataset WITHOUT context — tests if cartridge alone is sufficient
            dataset=GraphRelationshipMCEvalDataset.Config(
                val_path=str(GRAPH_DIR / "val_dataset.parquet"),
            ),
            name_for_wandb="graph_accuracy_no_ctx",
            generate_max_new_tokens=1024,
            temperature=0.0,
            batch_size=4,
            num_samples=1,
        ),
    ],

    save_every_n_steps=200,
    save_after_training=True,
    save_to_wandb=True,
    wandb=WandBConfig(tags=["train", "graph", "exp2_with_context"]),

    distributed_backend="gloo",
    seed=42,
)


if __name__ == "__main__":
    pydrantic.main(config)
