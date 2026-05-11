"""
Train a kinship cartridge on MC CoT QA pairs.
Target = "<think>BFS reasoning</think>\\n\\n<letter>." (NTP loss).

Run graph_qagen_cot.py first to generate train_cot_mc.parquet.

Usage (single GPU):
    python examples/graph/graph_train_cot.py

Usage (multi-GPU):
    torchrun --standalone --nproc_per_node=2 examples/graph/graph_train_cot.py
"""
import os
from pathlib import Path

import pydrantic

from cartridges.initialization import KVFromText
from cartridges.train import TrainConfig, GenerationEvalConfig
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.datasets import DataSource, TrainDataset
from examples.graph.graph_mc_eval import GraphMCEvalDataset

OUTPUT_DIR = Path(__file__).parent

config = TrainConfig(
    model=HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-1.7B",
        model_cls=FlexQwen3ForCausalLM,
    ),
    kv_cache_initializer=KVFromText.Config(
        text_source=str(OUTPUT_DIR / "family_tree_corpus.txt"),
        max_tokens=None,
    ),

    lr=2e-2,
    epochs=20,
    global_batch_size=32,

    dataset=TrainDataset.Config(
        data_sources=[DataSource(path=str(OUTPUT_DIR / "train_cot_mc.parquet"), type="local")],
        targets="tokens",
        packed_seq_length=512,   # longer — CoT reasoning is bigger
        packing_mode="pad",
    ),

    generate_eval_every_n_steps=100,
    generate_evals=[
        GenerationEvalConfig(
            dataset=GraphMCEvalDataset.Config(
                data_source=DataSource(
                    path=str(OUTPUT_DIR / "test_cot_mc.parquet"),
                    type="local",
                ),
                cot=True,
            ),
            name_for_wandb="kinship_cot_mc_test",
            generate_max_new_tokens=256,
            batch_size=8,
            temperature=0.0,
        )
    ],

    save_every_n_steps=200,
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", str(OUTPUT_DIR / "checkpoints")),
    name="graph-kinship-mc-cot",

    distributed_backend="gloo",
)

if __name__ == "__main__":
    pydrantic.main(config)
