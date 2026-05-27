"""
Train kinship cartridge with masked-letter loss.

Data: train_mc.parquet — assistant content = just "X." (no reasoning).
Loss: applied only on the final letter token via MaskedAnswerTrainDataset.
Init: KVFromText on family_tree_corpus.txt.
Eval: cot=True — model generates its own reasoning at eval time and is
      scored by letter-extraction from the output.

Run graph_qagen.py first to generate train_mc.parquet.

Single GPU:
    python examples/graph/graph_train.py
Multi-GPU:
    torchrun --standalone --nproc_per_node=2 examples/graph/graph_train.py
Override:
    python examples/graph/graph_train.py lr=1e-2 epochs=20
"""
import os
from pathlib import Path

import pydrantic

from cartridges.initialization import KVFromText
from cartridges.train import TrainConfig, GenerationEvalConfig
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.datasets import DataSource
from examples.graph.masked_answer_dataset import MaskedAnswerTrainDataset
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

    dataset=MaskedAnswerTrainDataset.Config(
        data_sources=[DataSource(path=str(OUTPUT_DIR / "train_mc.parquet"), type="local")],
        targets="tokens",
        packed_seq_length=1024,
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
                cot=True,
            ),
            name_for_wandb="kinship_mc_test",
            generate_max_new_tokens=256,
            batch_size=16,
            temperature=0.0,
        )
    ],

    save_every_n_steps=200,
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", str(OUTPUT_DIR / "checkpoints")),
    name="graph-kinship-masked",

    distributed_backend="gloo",
)

if __name__ == "__main__":
    pydrantic.main(config)
