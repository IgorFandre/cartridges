"""
Unified trainer for lineage-cartridge experiments.

Select experiment with $EXP (exp1|exp2). Compression budget via $CARTRIDGE_TOKENS.
Trains with KL logprob distillation (targets="logits").

Env vars:
  EXP                exp1 or exp2 (default exp1)
  CARTRIDGE_TOKENS   token budget for KVFromText init (default 512; empty=full)
  EPOCHS             (default 10)
  LR                 (default 2e-2)
  MAX_STEPS          (default 200; -1=full EPOCHS)
  SAVE_EVERY         (default MAX_STEPS, i.e. one checkpoint)
  N_EVALS            number of intermediate generation evals (default 4)
  EVAL_LIMIT         cap the test set for fast eval (default: full)
  SEED               (default 42)

Usage:
    EXP=exp1 CARTRIDGE_TOKENS=512 python -m examples.graph_2.training.lineage_train
    EXP=exp2 CARTRIDGE_TOKENS=256 MAX_STEPS=300 python -m examples.graph_2.training.lineage_train
"""
import os
from pathlib import Path

import pydrantic

from cartridges.initialization import KVFromText
from cartridges.train import TrainConfig, GenerationEvalConfig
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM
from cartridges.datasets import DataSource, TrainDataset
from cartridges.utils.wandb import WandBConfig
from examples.graph_2 import paths
from examples.graph_2.training.lineage_eval_dataset import LineageYesNoEvalDataset


# ── Env knobs ─────────────────────────────────────────────────────────────────
EXP        = os.environ.get("EXP",        "exp1").lower()
EPOCHS     = int(os.environ.get("EPOCHS", "10"))
LR         = float(os.environ.get("LR",   "2e-2"))
MAX_STEPS  = int(os.environ.get("MAX_STEPS", "200"))
SAVE_EVERY = int(os.environ.get("SAVE_EVERY", str(max(1, MAX_STEPS))))
N_EVALS    = int(os.environ.get("N_EVALS", "4"))
SEED       = int(os.environ.get("SEED",    "42"))

_eval_limit_raw = os.environ.get("EVAL_LIMIT", "").strip()
EVAL_LIMIT = int(_eval_limit_raw) if _eval_limit_raw else None

# Training target: "logits" = KL distillation (needs top_logprobs in parquet,
# i.e. server-synthesized data); "tokens" = plain CE on the assistant tokens
# (works on any parquet, used for the no-server training smoke test).
TARGETS = os.environ.get("TARGETS", "logits").strip().lower()
assert TARGETS in ("logits", "tokens"), f"TARGETS={TARGETS!r} must be logits|tokens"

# Explicit train parquet override (skips the artifact/ auto-discovery). Used by
# the smoke test to point at a fabricated dataset.
TRAIN_PARQUET_OVERRIDE = os.environ.get("TRAIN_PARQUET", "").strip()

# Explicit init-corpus / test-parquet overrides (default: the package data dir).
# The smoke test generates data into a temp dir, so it overrides these.
INIT_CORPUS_OVERRIDE  = os.environ.get("INIT_CORPUS", "").strip()
TEST_PARQUET_OVERRIDE = os.environ.get("TEST_PARQUET", "").strip()


def _cartridge_tokens() -> int | None:
    raw = os.environ.get("CARTRIDGE_TOKENS", "").strip()
    return int(raw) if raw else None


def _train_parquet_for(exp: str) -> Path:
    """Locate the synthesized dataset.parquet for exp1 or exp2."""
    if exp == "exp1":
        root = paths.EXP1_DIR
    elif exp == "exp2":
        root = paths.EXP2_DIR / "exp2_star"
    else:
        raise ValueError(f"Unknown EXP={exp!r}. Use 'exp1' or 'exp2'.")

    hits = list(root.rglob("artifact/dataset.parquet")) if root.exists() else []
    if not hits:
        raise FileNotFoundError(
            f"No artifact/dataset.parquet found under {root}. "
            f"Run {'lineage_synthesize' if exp=='exp1' else 'star_synthesize'}.py first."
        )
    # Use the most recently written
    return sorted(hits, key=lambda p: p.stat().st_mtime)[-1]


def build_config(
    *,
    exp: str,
    train_parquet: Path,
    test_parquet: Path,
    init_corpus: Path,
    cartridge_tokens: int | None,
    output_dir: Path,
    run_name: str,
) -> TrainConfig:
    for path, label in [
        (init_corpus,   "init corpus"),
        (train_parquet, "train data"),
        (test_parquet,  "test data"),
    ]:
        if not Path(path).exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    # Compute eval cadence: ~N_EVALS intermediate + 1 at step 0 (generate_before_training)
    eval_every = max(1, MAX_STEPS // N_EVALS) if MAX_STEPS > 0 else 50
    assert eval_every <= max(1, MAX_STEPS), (
        f"eval_every={eval_every} > MAX_STEPS={MAX_STEPS}. "
        "Increase MAX_STEPS or decrease N_EVALS."
    )

    return TrainConfig(
        model=HFModelConfig(
            pretrained_model_name_or_path="Qwen/Qwen3-1.7B",
            model_cls=FlexQwen3ForCausalLM,
        ),
        kv_cache_initializer=KVFromText.Config(
            text_source=str(init_corpus),
            max_tokens=cartridge_tokens,
        ),
        seed=SEED,
        lr=LR,
        epochs=EPOCHS,
        max_optimizer_steps=MAX_STEPS,
        global_batch_size=32,
        dataset=TrainDataset.Config(
            data_sources=[DataSource(path=str(train_parquet), type="local")],
            top_k_logits=20,
            targets=TARGETS,            # "logits" KL-distill (server data) | "tokens" CE (smoke)
            packed_seq_length=2048,
            packing_mode="pad",
        ),
        generate_before_training=True,
        generate_eval_every_n_steps=eval_every,
        generate_evals=[
            GenerationEvalConfig(
                dataset=LineageYesNoEvalDataset.Config(
                    data_source=DataSource(
                        path=str(test_parquet),
                        type="local",
                        limit=EVAL_LIMIT,
                    ),
                    cot=True,
                ),
                name_for_wandb=f"lineage_{exp}_test",
                generate_max_new_tokens=512,  # free-form CoT → Yes/No
                batch_size=8,
                temperature=0.0,
            )
        ],
        save_every_n_steps=SAVE_EVERY,
        output_dir=str(output_dir),
        name=run_name,
        distributed_backend="gloo",
        wandb=WandBConfig(tags=["lineage", exp, "train"]),
    )


# ── Build config(s) ───────────────────────────────────────────────────────────
if EXP not in ("exp1", "exp2"):
    raise ValueError(f"EXP={EXP!r} must be 'exp1' or 'exp2'.")

tokens = _cartridge_tokens()
tokens_tag = f"tok{tokens}" if tokens else "tokfull"

if TRAIN_PARQUET_OVERRIDE:
    train_parquet = Path(TRAIN_PARQUET_OVERRIDE)
    if not train_parquet.exists():
        raise FileNotFoundError(f"TRAIN_PARQUET not found: {train_parquet}")
else:
    train_parquet = _train_parquet_for(EXP)
test_parquet  = Path(TEST_PARQUET_OVERRIDE) if TEST_PARQUET_OVERRIDE else paths.BASE_TEST_PARQUET
init_corpus   = Path(INIT_CORPUS_OVERRIDE)  if INIT_CORPUS_OVERRIDE  else paths.BASE_CORPUS
output_dir    = (paths.EXP1_DIR if EXP == "exp1" else paths.EXP2_DIR) / "train"
run_name      = f"lineage-{EXP}-{tokens_tag}"

config = build_config(
    exp=EXP,
    train_parquet=train_parquet,
    test_parquet=test_parquet,
    init_corpus=init_corpus,
    cartridge_tokens=tokens,
    output_dir=output_dir,
    run_name=run_name,
)

if __name__ == "__main__":
    pydrantic.main(config)
