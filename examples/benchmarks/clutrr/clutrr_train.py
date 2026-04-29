import os
from pathlib import Path

import pydrantic

from cartridges.initialization import KVFromText, KVFromAttnMatching, KVFromRandomVectors, KVFromSampledChunks, KVFromTextPerturbed
from cartridges.train import GenerationEvalConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import TrainDataset, DataSource
from cartridges.data.clutrr.evals import CLUTRRRelationGenerateDataset
from cartridges.data.clutrr.resources import CLUTRRResource
from cartridges.utils.wandb import WandBConfig


NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "1024"))
ATTN_MATCHING_CKPT = os.environ.get("ATTN_MATCHING_CKPT", None)
SCI_CHUNK_SIZE = int(os.environ.get("SCI_CHUNK_SIZE", "64"))
KEY_SCALE = float(os.environ.get("KEY_SCALE", "2.0"))
KEY_NOISE_STD = float(os.environ.get("KEY_NOISE_STD", "0.05"))
NOISE_SEED = int(os.environ.get("NOISE_SEED", "42"))
INIT_MODE = os.environ.get("INIT_MODE", "attn_match" if ATTN_MATCHING_CKPT else "text")

MODEL = os.environ.get("MODEL", "qwen1.7b")
if MODEL == "llama":
    from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
    data_sources = []  # fill in after running synthesis
    model = HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_cls=FlexLlamaForCausalLM,
    )
elif MODEL == "qwen4b":
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
    data_sources = []  # fill in after running synthesis
    model = HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-4b",
        model_cls=FlexQwen3ForCausalLM,
    )
elif MODEL == "qwen1.7b":
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
    data_sources = []  # fill in after running synthesis
    model = HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-1.7B",
        model_cls=FlexQwen3ForCausalLM,
    )
else:
    raise ValueError(f"Invalid model: {MODEL}")

LR = 2e-2
if INIT_MODE == "attn_match":
    init_tag = f"attn_match_{Path(ATTN_MATCHING_CKPT).stem}"
elif INIT_MODE == "random":
    init_tag = f"random_{NUM_TOKENS}toks"
elif INIT_MODE == "sci":
    init_tag = f"sci_{NUM_TOKENS}toks_c{SCI_CHUNK_SIZE}"
elif INIT_MODE == "text_scaled":
    init_tag = f"text_scaled_{NUM_TOKENS}toks_a{KEY_SCALE}"
elif INIT_MODE == "text_noisy":
    init_tag = f"text_noisy_{NUM_TOKENS}toks_s{KEY_NOISE_STD}"
elif INIT_MODE == "text_scaled_noisy":
    init_tag = f"text_scaled_noisy_{NUM_TOKENS}toks_a{KEY_SCALE}_s{KEY_NOISE_STD}"
else:
    init_tag = f"text_{NUM_TOKENS}toks"

RUN_NAME = f"clutrr_{MODEL}_{init_tag}_lr{LR}"

_corpus_text = CLUTRRResource(CLUTRRResource.Config()).to_string()

if INIT_MODE == "sci":
    _kv_initializer = KVFromSampledChunks.Config(
        max_tokens=NUM_TOKENS,
        chunk_size=SCI_CHUNK_SIZE,
        text=_corpus_text,
    )
elif INIT_MODE == "attn_match":
    _kv_initializer = KVFromAttnMatching.Config(path=ATTN_MATCHING_CKPT)
elif INIT_MODE == "random":
    _kv_initializer = KVFromRandomVectors.Config(max_tokens=NUM_TOKENS)
elif INIT_MODE == "text_scaled":
    _kv_initializer = KVFromTextPerturbed.Config(
        max_tokens=NUM_TOKENS,
        key_scale=KEY_SCALE,
    )
elif INIT_MODE == "text_noisy":
    _kv_initializer = KVFromTextPerturbed.Config(
        max_tokens=NUM_TOKENS,
        key_noise_std=KEY_NOISE_STD,
        noise_seed=NOISE_SEED,
    )
elif INIT_MODE == "text_scaled_noisy":
    _kv_initializer = KVFromTextPerturbed.Config(
        max_tokens=NUM_TOKENS,
        key_scale=KEY_SCALE,
        key_noise_std=KEY_NOISE_STD,
        noise_seed=NOISE_SEED,
    )
else:
    _kv_initializer = KVFromText.Config(max_tokens=NUM_TOKENS)

config = TrainConfig(
    model=model,
    kv_cache_initializer=_kv_initializer,

    lr=LR,
    epochs=5,
    global_batch_size=128,

    dataset=TrainDataset.Config(
        data_sources=[DataSource(path=source, type="hf") for source in data_sources],
        top_k_logits=20,
        packed_seq_length=1024,
        packing_mode="truncate",
    ),

    save_every_n_steps=512,
    generate_eval_every_n_steps=128,
    generate_evals=[
        GenerationEvalConfig(
            dataset=CLUTRRRelationGenerateDataset.Config(
                config="gen_train23_test2to10",
                split="test",
                noise_types=[1],
            ),
            name_for_wandb="clutrr_test",
            generate_max_new_tokens=256,
            batch_size=32,
            temperature=0.0,
        )
    ],
    distributed_backend="nccl",

    wandb=WandBConfig(tags=["train", "clutrr"]),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=RUN_NAME,
)

if __name__ == "__main__":
    pydrantic.main(config)
