"""
Exp 2-stability: train Alex variant 5 times with different seeds to measure
training noise (run-to-run variance in trained KV cache).

Shared across all 5 runs:
  - variant = alex (corpus, train data, test data — all from variants/alex/)
  - KV init = variants/alex/family_tree_corpus.txt
  - cartridge size = 40 tokens (compressed from full 282)
  - hyperparams (lr, epochs, batch size)

Per-run differences: seed only (42, 43, 44, 45, 46) → produces independent
trained caches alex_run{1..5}/cache_last.pt for pairwise comparison.

Usage:
    python examples/graph/graph_train_alex_stability.py            # all 5 sequentially
    RUN_ONLY=1 python examples/graph/graph_train_alex_stability.py # single run
    RUN_ONLY=3 torchrun --standalone --nproc_per_node=2 examples/graph/graph_train_alex_stability.py
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

VARIANTS_ROOT = Path(__file__).parent / "variants"
ANCHOR = "alex"
ALEX_DIR = VARIANTS_ROOT / ANCHOR
INIT_CORPUS = ALEX_DIR / "family_tree_corpus.txt"

OUTPUT_BASE = Path(
    os.environ.get(
        "CARTRIDGES_OUTPUT_DIR_STABILITY",
        str(Path(__file__).parent / "checkpoints_stability"),
    )
)

CARTRIDGE_TOKENS = 40
N_RUNS = 5
SEEDS = [42, 43, 44, 45, 46]


def build_config(run_idx: int) -> TrainConfig:
    seed = SEEDS[run_idx]
    run_name = f"alex_run{run_idx + 1}"
    return TrainConfig(
        model=HFModelConfig(
            pretrained_model_name_or_path="Qwen/Qwen3-1.7B",
            model_cls=FlexQwen3ForCausalLM,
        ),
        kv_cache_initializer=KVFromText.Config(
            text_source=str(INIT_CORPUS),
            max_tokens=CARTRIDGE_TOKENS,
        ),
        seed=seed,
        lr=2e-2,
        epochs=10,
        global_batch_size=32,
        dataset=MaskedAnswerTrainDataset.Config(
            data_sources=[DataSource(path=str(ALEX_DIR / "train_mc.parquet"), type="local")],
            targets="tokens",
            packed_seq_length=1024,
            packing_mode="pad",
        ),
        generate_eval_every_n_steps=150,
        generate_evals=[
            GenerationEvalConfig(
                dataset=GraphMCEvalDataset.Config(
                    data_source=DataSource(
                        path=str(ALEX_DIR / "test_mc.parquet"),
                        type="local",
                    ),
                    cot=True,
                ),
                name_for_wandb=f"kinship_mc_test_{run_name}",
                generate_max_new_tokens=256,
                batch_size=8,
                temperature=0.0,
            )
        ],
        save_every_n_steps=200,
        output_dir=str(OUTPUT_BASE / run_name),
        name=f"graph-alex-stability-{run_name}-tok{CARTRIDGE_TOKENS}",
        distributed_backend="gloo",
    )


only = os.environ.get("RUN_ONLY")
if only:
    idx = int(only) - 1
    if not (0 <= idx < N_RUNS):
        raise ValueError(f"RUN_ONLY={only!r} must be 1..{N_RUNS}")
    configs = build_config(idx)
else:
    configs = [build_config(i) for i in range(N_RUNS)]

if __name__ == "__main__":
    pydrantic.main(configs)
