"""
Train a kinship cartridge on family tree MC QA pairs (NTP loss, letter target).

Run graph_qagen.py first to generate train_mc.parquet and family_tree_corpus.txt.

Usage (single GPU):
    python examples/graph/graph_train.py

Usage (multi-GPU):
    torchrun --standalone --nproc_per_node=2 examples/graph/graph_train.py

Override any config field on CLI:
    python examples/graph/graph_train.py lr=1e-2 epochs=20
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
    # Cartridge length = size of family tree text (all tokens)
    kv_cache_initializer=KVFromText.Config(
        text_source=str(OUTPUT_DIR / "family_tree_corpus.txt"),
        max_tokens=None,
    ),

    lr=2e-2,
    epochs=20,
    global_batch_size=32,

    dataset=TrainDataset.Config(
        data_sources=[DataSource(path=str(OUTPUT_DIR / "train_mc.parquet"), type="local")],
        targets="tokens",        # NTP loss on MC letter answer
        packed_seq_length=256,
        packing_mode="pad",
    ),

    generate_eval_every_n_steps=100,
    generate_evals=[
        GenerationEvalConfig(
            dataset=GraphMCEvalDataset.Config(
                data_source=DataSource(
                    path=str(OUTPUT_DIR / "test_mc.parquet"),
                    type="local",
                ),
                cot=False,
            ),
            name_for_wandb="kinship_mc_test",
            generate_max_new_tokens=8,
            batch_size=16,
            temperature=0.0,
        )
    ],

    save_every_n_steps=200,
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", str(OUTPUT_DIR / "checkpoints")),
    name="graph-kinship-mc-ntp",

    distributed_backend="gloo",
)

if __name__ == "__main__":
    pydrantic.main(config)
