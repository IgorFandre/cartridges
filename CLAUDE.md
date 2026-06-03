# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo does

Trains **cartridges** — small KV caches that represent large text corpora — using a test-time training recipe called **self-study**. Two-step pipeline:
1. **Synthesize** — generate synthetic Q&A conversations about a corpus using an LLM inference server (Tokasaurus or SGLang)
2. **Train** — run context-distillation on the synthesized data to train the KV cache

Paper: [Cartridges: Lightweight and general-purpose long context representations via self-study](https://arxiv.org/abs/2506.06266)

## Setup

```bash
pip install uv
uv pip install -e .
```

Required environment variables:
```bash
export CARTRIDGES_DIR=/path/to/cartridges
export CARTRIDGES_OUTPUT_DIR=/path/to/outputs
export CARTRIDGES_WANDB_PROJECT=your-wandb-project
export CARTRIDGES_WANDB_ENTITY=your-wandb-entity
```

## Commands

**Run synthesis:**
```bash
python examples/arxiv/arxiv_synthesize.py
# Override config fields on CLI:
python examples/arxiv/arxiv_synthesize.py num_samples=1024
```

**Run training (single GPU):**
```bash
python examples/arxiv/arxiv_train.py
```

**Run training (multi-GPU, data parallel):**
```bash
torchrun --standalone --nproc_per_node=2 examples/arxiv/arxiv_train.py
# If NCCL timeout occurs: set distributed_backend="gloo" in TrainConfig
```

**Run tests:**
```bash
python -m pytest cartridges/tests/
# Single test file:
python -m pytest cartridges/tests/models/test_qwen.py
```

**Chat with a trained cartridge:**
```bash
python -m cartridges.utils.chat <wandb_entity>/<wandb_project>/<run_id>
```

**Local inference server (Tokasaurus):**
```bash
tksrs model=Qwen/Qwen3-4b kv_cache_num_tokens='(512 * 1024)' max_top_logprobs=20
```

## Architecture

### Configuration system
All configs use [Pydantic](https://docs.pydantic.dev/) + [`pydrantic`](https://github.com/seyuboglu/pydrantic). `RunConfig` subclasses have a `.run()` method and are launched with `pydrantic.main([config])`, which also enables CLI field overrides. Nested configs use `ObjectConfig` with an `.instantiate()` method.

### Core pipeline flow

**Synthesis** (`cartridges/synthesize.py`): `SynthesizeConfig.run()` orchestrates async batch processing. Each batch calls `SelfStudySynthesizer.sample_convos()`, which runs two LLM agents (A asks questions, B answers with context). Output: `dataset.parquet` of `Conversation` objects.

**Training** (`cartridges/train.py`): `TrainConfig.run()` → `train()`. Freezes base model weights, trains only `TrainableCache` (the cartridge). Uses `CacheAndModel` wrapper for the forward pass. Supports DDP via `torchrun`. Checkpoints saved as `cache-step{N}.pt`; `cache_last.pt` symlink always points to latest. All metrics logged to W&B.

**TrainableCache** (`cartridges/cache.py`): `nn.Module` holding per-layer key/value tensors. Uses `CARTRIDGE_SEQ_ID = -1` to mark cartridge tokens so FlexAttention knows they attend to all sequences. Separate frozen/trainable token splits.

**Models** (`cartridges/models/`): Custom FlexAttention implementations of Qwen3 (`modeling_qwen3.py`) and LLaMA (`modeling_llama.py`) that accept `seq_ids` and `past_key_values=TrainableCache`. Use `HFModelConfig` with `model_cls=FlexQwen3ForCausalLM` (or LLaMA equivalent) to load. Two tuning methods: `custom_prefix` (trains KV cache, default) and `peft` (LoRA etc.).

**Datasets** (`cartridges/datasets.py`): `CartridgeTrainDataset` packs tokenized conversations into fixed-length sequences. Reads `.parquet` files of `Conversation` objects. Training uses top-k logprob distillation (KL divergence against teacher logprobs stored in the parquet).

**Data / Resources** (`cartridges/data/`): `Resource` subclasses feed chunked text + seed prompts to the synthesizer. Available: `TextFileResource`, `JSONResource`, `LaTeXResource` (with arxiv ID support), `SlackResource`, `GMailResource`. Chunking via `TokenChunker` in `data/chunkers.py`.

**Clients** (`cartridges/clients/`): Async HTTP clients wrapping Tokasaurus and SGLang inference servers. Return `FlatTopLogprobs` (sparse top-k logprob dicts). Tokasaurus is the recommended client for synthesis.

**Initialization** (`cartridges/initialization/`): Controls how cartridge KV cache is initialized before training. `KVFromRandomText` (default, best), `KVFromRandomVectors`, `KVCacheFromPretrained`. `max_tokens` sets cartridge size ($p$ in paper).

### Key data structures
- `Conversation` / `Conversation.Message` (`structs.py`): synthesized training examples with token IDs and `FlatTopLogprobs`
- `DatasetBatch` (`datasets.py`): packed batch with `input_ids`, `element_ids` (seq membership), `position_ids`, `topk_token_ids`, `topk_logprobs` for distillation loss

### Benchmark examples
`examples/benchmarks/` has complete synthesize+train scripts for: LongHealth (medical QA), MTOB (translation), NIAH (needle-in-a-haystack). Each follows the same pattern as the arxiv example.

### Infrastructure
`infra/` has Modal deployment scripts for Tokasaurus and SGLang inference servers. Recommended for synthesis due to serverless horizontal scaling.

### Visualization
`viz/` is a Vite/React app for exploring synthesized datasets. See `viz/README.md`.
