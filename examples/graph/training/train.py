"""
Unified trainer for all kinship-cartridge experiments.

Replaces graph_train.py, graph_train_variants.py, graph_train_alex_stability.py
and graph_train_stability_pair.py — one config-driven entrypoint selected by
$MODE.

All runs share: Qwen3-1.7B frozen base, KVFromText init, masked-letter loss
(loss only on the final answer letter), cot=True generation eval.

────────────────────────────────────────────────────────────────────────────
MODE=variants   (exp2)  — 4 cartridges from a shared anchor init, one per data
                          variant. Init corpus = anchor (Alex) for ALL runs, so
                          post-train differences come only from the train data.
    env: ANCHOR=alex  VARIANTS=alex,ben,carl,dan  VARIANT_ONLY=ben
         CARTRIDGE_TOKENS=  (empty = full corpus)
    out: $outputs/exp2_train/<variant>/

MODE=stability  (exp4)  — train <INIT> init on <DATA> data N times with
                          different seeds → measure run-to-run noise.
    env: INIT_VARIANT=ben  DATA_VARIANT=alex  N_RUNS=5  SEED_BASE=42
         CARTRIDGE_TOKENS=40  RUN_ONLY=3
    out: $outputs/exp4_stability/<init>_on_<data>_run{i}/

MODE=single             — one cartridge on the base (un-swapped) tree.
    env: SEED_BASE=42  CARTRIDGE_TOKENS=
    out: $outputs/exp2_train/base/

Common env: EPOCHS (default 10), LR (default 2e-2),
            MAX_STEPS (default 100, -1 = unlimited / full EPOCHS),
            SAVE_EVERY (default 20 → checkpoints at 20/40/60/80/100 for dynamics).
Results root: $CARTRIDGES_OUTPUT_DIR_GRAPH (default <repo>/outputs_graph).

Usage:
    MODE=variants VARIANT_ONLY=alex python -m examples.graph.training.train
    MODE=stability INIT_VARIANT=ben DATA_VARIANT=ben RUN_ONLY=1 \
        python -m examples.graph.training.train
    torchrun --standalone --nproc_per_node=2 -m examples.graph.training.train
"""
import os
from pathlib import Path

import pydrantic

from cartridges.initialization import KVFromText
from cartridges.train import TrainConfig, GenerationEvalConfig
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.datasets import DataSource

from examples.graph.training.masked_answer_dataset import MaskedAnswerTrainDataset
from examples.graph.training.mc_eval import GraphMCEvalDataset
from examples.graph import paths


# ── Shared hyperparameters ───────────────────────────────────────────────────
EPOCHS = int(os.environ.get("EPOCHS", "10"))
LR = float(os.environ.get("LR", "2e-2"))
# Hard cap on optimizer steps (-1 = unlimited, run full EPOCHS). Default 100 →
# short runs whose K/V rotation is traced by comparison/dynamics.py.
MAX_STEPS = int(os.environ.get("MAX_STEPS", "100"))
# Checkpoint cadence. Default 20 → with MAX_STEPS=100 yields checkpoints at
# steps 20/40/60/80/100 (5 points) for the rotation analysis.
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", "20"))


def _cartridge_tokens() -> int | None:
    raw = os.environ.get("CARTRIDGE_TOKENS", "").strip()
    return int(raw) if raw else None


def build_config(
    *,
    init_corpus: Path,
    train_parquet: Path,
    test_parquet: Path,
    seed: int,
    cartridge_tokens: int | None,
    output_dir: Path,
    run_name: str,
    wandb_eval_name: str,
) -> TrainConfig:
    """One TrainConfig — identical skeleton for every experiment."""
    for path, label in [(init_corpus, "init corpus"),
                        (train_parquet, "train data"),
                        (test_parquet, "test data")]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} missing: {path}")

    return TrainConfig(
        model=HFModelConfig(
            pretrained_model_name_or_path="Qwen/Qwen3-1.7B",
            model_cls=FlexQwen3ForCausalLM,
        ),
        kv_cache_initializer=KVFromText.Config(
            text_source=str(init_corpus),
            max_tokens=cartridge_tokens,
        ),
        seed=seed,
        lr=LR,
        epochs=EPOCHS,
        max_optimizer_steps=MAX_STEPS,
        global_batch_size=32,
        dataset=MaskedAnswerTrainDataset.Config(
            data_sources=[DataSource(path=str(train_parquet), type="local")],
            targets="tokens",
            packed_seq_length=1024,
            packing_mode="pad",
        ),
        generate_eval_every_n_steps=150,
        generate_evals=[
            GenerationEvalConfig(
                dataset=GraphMCEvalDataset.Config(
                    data_source=DataSource(path=str(test_parquet), type="local"),
                    cot=True,
                ),
                name_for_wandb=wandb_eval_name,
                generate_max_new_tokens=256,
                batch_size=8,
                temperature=0.0,
            )
        ],
        save_every_n_steps=SAVE_EVERY,
        output_dir=str(output_dir),
        name=run_name,
        distributed_backend="gloo",
    )


# ── Mode: variants (exp2) ────────────────────────────────────────────────────
def configs_variants() -> list[TrainConfig]:
    import json

    anchor = os.environ.get("ANCHOR", "alex").lower()
    if os.environ.get("VARIANTS"):
        variants = [v.strip().lower() for v in os.environ["VARIANTS"].split(",") if v.strip()]
    elif paths.VARIANTS_META.exists():
        meta = json.loads(paths.VARIANTS_META.read_text())
        variants = [v.lower() for v in meta["variants"]]
    else:
        raise FileNotFoundError(
            f"{paths.VARIANTS_META} missing — run generate_variants.py, or set $VARIANTS."
        )

    only = os.environ.get("VARIANT_ONLY")
    if only:
        only = only.lower()
        if only not in variants:
            raise ValueError(f"VARIANT_ONLY={only!r} not in {variants}")
        variants = [only]

    init_corpus = paths.variant_corpus(anchor)  # shared anchor init for all
    tokens = _cartridge_tokens()
    seed = int(os.environ.get("SEED_BASE", "42"))

    return [
        build_config(
            init_corpus=init_corpus,
            train_parquet=paths.variant_train(v),
            test_parquet=paths.variant_test(v),
            seed=seed,
            cartridge_tokens=tokens,
            output_dir=paths.EXP2_DIR / v,
            run_name=f"graph-variant-masked-{v}",
            wandb_eval_name=f"kinship_mc_test_{v}",
        )
        for v in variants
    ]


# ── Mode: stability (exp4) ───────────────────────────────────────────────────
def configs_stability() -> list[TrainConfig]:
    init_v = os.environ.get("INIT_VARIANT", "alex").lower()
    data_v = os.environ.get("DATA_VARIANT", "alex").lower()
    n_runs = int(os.environ.get("N_RUNS", "5"))
    seed_base = int(os.environ.get("SEED_BASE", "42"))
    tokens = _cartridge_tokens()
    if tokens is None:
        tokens = 40  # stability default: compressed cartridge
    tag = f"{init_v}_on_{data_v}"

    def one(i: int) -> TrainConfig:
        run_name = f"{tag}_run{i + 1}"
        return build_config(
            init_corpus=paths.variant_corpus(init_v),
            train_parquet=paths.variant_train(data_v),
            test_parquet=paths.variant_test(data_v),
            seed=seed_base + i,
            cartridge_tokens=tokens,
            output_dir=paths.EXP4_DIR / run_name,
            run_name=f"graph-stability-{run_name}-tok{tokens}",
            wandb_eval_name=f"kinship_mc_test_{run_name}",
        )

    only = os.environ.get("RUN_ONLY")
    if only:
        idx = int(only) - 1
        if not (0 <= idx < n_runs):
            raise ValueError(f"RUN_ONLY={only!r} must be 1..{n_runs}")
        return [one(idx)]
    return [one(i) for i in range(n_runs)]


# ── Mode: single (base tree) ─────────────────────────────────────────────────
def configs_single() -> list[TrainConfig]:
    return [
        build_config(
            init_corpus=paths.BASE_CORPUS,
            train_parquet=paths.BASE_TRAIN_PARQUET,
            test_parquet=paths.BASE_TEST_PARQUET,
            seed=int(os.environ.get("SEED_BASE", "42")),
            cartridge_tokens=_cartridge_tokens(),
            output_dir=paths.EXP2_DIR / "base",
            run_name="graph-kinship-masked",
            wandb_eval_name="kinship_mc_test",
        )
    ]


MODES = {
    "variants": configs_variants,
    "stability": configs_stability,
    "single": configs_single,
}

mode = os.environ.get("MODE", "variants").lower()
if mode not in MODES:
    raise ValueError(f"MODE={mode!r} not in {list(MODES)}")
configs = MODES[mode]()
configs = configs[0] if len(configs) == 1 else configs


if __name__ == "__main__":
    pydrantic.main(configs)
