"""
Exp 2: train 4 cartridges on variants {alex, ben, carl, dan}.

Shared across all 4 runs:
  - KV init = variants/alex/family_tree_corpus.txt   (anchor: Alex)
  - seed, lr, batch size, epochs, model
  - masked-reasoning loss (loss only on final letter token)

Per-variant differences: train data (variants/<v>/train_mc.parquet) and
eval data (variants/<v>/test_mc.parquet).

Run `examples/graph/generate_tree_variants.py` first.

Usage:
    python examples/graph/graph_train_variants.py            # all 4 sequentially
    VARIANT_ONLY=alex python examples/graph/graph_train_variants.py
    VARIANT_ONLY=ben  torchrun --standalone --nproc_per_node=2 examples/graph/graph_train_variants.py
"""
import json
import os
from pathlib import Path

import pydrantic

from cartridges.initialization import KVFromText
from cartridges.train import TrainConfig, GenerationEvalConfig
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.datasets import DataSource
from examples.graph.masked_answer_dataset import MaskedAnswerTrainDataset
from examples.graph.graph_mc_eval import GraphMCEvalDataset

VARIANTS_ROOT = Path(__file__).parent / "variants"
META_PATH = VARIANTS_ROOT / "variants_meta.json"

if not META_PATH.exists():
    raise FileNotFoundError(
        f"{META_PATH} missing — run examples/graph/generate_tree_variants.py first."
    )

meta = json.loads(META_PATH.read_text())
VARIANTS = [v.lower() for v in meta["variants"]]
ANCHOR = "alex"
assert ANCHOR in VARIANTS, f"anchor {ANCHOR!r} not in variants {VARIANTS}"

# Shared KV init: Alex corpus, identical for all 4 runs.
INIT_CORPUS = VARIANTS_ROOT / ANCHOR / "family_tree_corpus.txt"
if not INIT_CORPUS.exists():
    raise FileNotFoundError(f"{INIT_CORPUS} missing — generate variants first.")

OUTPUT_BASE = Path(
    os.environ.get(
        "CARTRIDGES_OUTPUT_DIR_VARIANTS",
        str(Path(__file__).parent / "checkpoints_variants"),
    )
)

SEED = 42


def build_config(variant: str) -> TrainConfig:
    variant_dir = VARIANTS_ROOT / variant
    return TrainConfig(
        model=HFModelConfig(
            pretrained_model_name_or_path="Qwen/Qwen3-1.7B",
            model_cls=FlexQwen3ForCausalLM,
        ),
        kv_cache_initializer=KVFromText.Config(
            text_source=str(INIT_CORPUS),
            max_tokens=None,
        ),
        seed=SEED,
        lr=2e-2,
        epochs=10,
        global_batch_size=32,
        dataset=MaskedAnswerTrainDataset.Config(
            data_sources=[DataSource(path=str(variant_dir / "train_mc.parquet"), type="local")],
            targets="tokens",
            packed_seq_length=1024,
            packing_mode="pad",
        ),
        generate_eval_every_n_steps=150,
        generate_evals=[
            GenerationEvalConfig(
                dataset=GraphMCEvalDataset.Config(
                    data_source=DataSource(
                        path=str(variant_dir / "test_mc.parquet"),
                        type="local",
                    ),
                    cot=True,
                ),
                name_for_wandb=f"kinship_mc_test_{variant}",
                generate_max_new_tokens=256,
                batch_size=8,
                temperature=0.0,
            )
        ],
        save_every_n_steps=200,
        output_dir=str(OUTPUT_BASE / variant),
        name=f"graph-variant-masked-{variant}",
        distributed_backend="gloo",
    )


only = os.environ.get("VARIANT_ONLY")
if only:
    if only.lower() not in VARIANTS:
        raise ValueError(f"VARIANT_ONLY={only!r} not in {VARIANTS}")
    configs = build_config(only.lower())
else:
    configs = [build_config(v) for v in VARIANTS]

if __name__ == "__main__":
    pydrantic.main(configs)
