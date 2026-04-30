import os
from pathlib import Path
import socket

import pydrantic

from cartridges.initialization import KVFromText, KVFromAttnMatching, KVFromRandomVectors, KVFromSampledChunks, KVFromTextPerturbed
from cartridges.train import GenerationEvalConfig, TrainConfig
from cartridges.models.config import HFModelConfig
from cartridges.datasets import TrainDataset, DataSource
from cartridges.data.longhealth.evals import LongHealthMultipleChoiceGenerateDataset
from cartridges.data.longhealth.resources import LongHealthResource
from cartridges.utils.wandb import WandBConfig


NUM_PATIENTS = 1
patient_idxs = list(range(1, NUM_PATIENTS + 1))
patients_str = f"p{NUM_PATIENTS}"
patient_ids = [f"patient_{idx:02d}" for idx in patient_idxs]

NUM_TOKENS = int(os.environ.get("NUM_TOKENS", "1024"))
ZERO_LAYERS_FROM_START = int(os.environ.get("ZERO_LAYERS_FROM_START", "0"))
ZERO_LAYERS_FROM_END = int(os.environ.get("ZERO_LAYERS_FROM_END", "0"))
ATTN_MATCHING_CKPT = os.environ.get("ATTN_MATCHING_CKPT", None)
SCI_CHUNK_SIZE = int(os.environ.get("SCI_CHUNK_SIZE", "64"))
KEY_SCALE = float(os.environ.get("KEY_SCALE", "2.0"))
KEY_NOISE_STD = float(os.environ.get("KEY_NOISE_STD", "0.05"))
NOISE_SEED = int(os.environ.get("NOISE_SEED", "42"))
# INIT_MODE: "text" | "random" | "attn_match" (requires ATTN_MATCHING_CKPT) | "sci"
#          | "text_scaled" | "text_noisy" | "text_scaled_noisy"
INIT_MODE = os.environ.get("INIT_MODE", "attn_match" if ATTN_MATCHING_CKPT else "text")

MODEL = os.environ.get("MODEL", "qwen1.7b")
if MODEL == "llama":
    from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
    data_sources = [
        "hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-0",
        "hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-1",
        "hazyresearch/m07d11_longhealth_synthesize_llama-3.2-3b_p10_n65536-2"
    ]
    model = HFModelConfig(
        pretrained_model_name_or_path="meta-llama/Llama-3.2-3B-Instruct",
        model_cls=FlexLlamaForCausalLM,
    )
elif MODEL == "qwen4b":
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
    data_sources = [
        "hazyresearch/m07d11_longhealth_synthesize_qwen3-4b_p10_n65536-0",
        "hazyresearch/m07d11_longhealth_synthesize_qwen3-4b_p10_n65536-1"
    ]
    model=HFModelConfig(
        pretrained_model_name_or_path="Qwen/Qwen3-4b",
        model_cls=FlexQwen3ForCausalLM,
    )
elif MODEL == "qwen1.7b":
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
    data_sources = [
        "hazyresearch/m07d11_longhealth_synthesize_qwen3-4b_p10_n65536-0",
        "hazyresearch/m07d11_longhealth_synthesize_qwen3-4b_p10_n65536-1"
    ]
    model=HFModelConfig(
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

_zero_tag = ""
if ZERO_LAYERS_FROM_START > 0:
    _zero_tag += f"_zs{ZERO_LAYERS_FROM_START}"
if ZERO_LAYERS_FROM_END > 0:
    _zero_tag += f"_ze{ZERO_LAYERS_FROM_END}"

RUN_NAME = f"longhealth_{MODEL}_{patients_str}_{init_tag}{_zero_tag}_lr{LR}"

if INIT_MODE == "sci":
    _corpus_text = LongHealthResource(LongHealthResource.Config(patient_ids=patient_ids)).to_string()
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
            dataset=LongHealthMultipleChoiceGenerateDataset.Config(
                patient_ids=patient_ids,
            ),
            name_for_wandb=f"longhealth_{patients_str}",
            generate_max_new_tokens=2048,
            batch_size=32,
            temperature=0.0,
        )
    ],
    zero_layers_from_start=ZERO_LAYERS_FROM_START,
    zero_layers_from_end=ZERO_LAYERS_FROM_END,
    distributed_backend="nccl",

    wandb=WandBConfig(tags=["train", "longhealth"]),
    output_dir=os.environ.get("CARTRIDGES_OUTPUT_DIR", "."),
    name=RUN_NAME,
)


if __name__ == "__main__":
    pydrantic.main(config)