# Kinship-graph cartridge experiments

Do trainable KV-cache **cartridges** capture the relational structure of a family
tree? We train cartridges on a 45-person / 6-generation tree over a frozen
Qwen3-1.7B and probe them with controlled name-swap variants.

- **How cartridges are compared** → [`METHODOLOGY.md`](METHODOLOGY.md)
- **Where every output/checkpoint lands** → [`OUTPUTS.md`](OUTPUTS.md)
- **Findings** → [`REPORT.md`](REPORT.md) — ⚠️ all prior results are **stale** (computed
  on a dataset with a fixed MC-option bug); experiments are being recomputed.

## Layout

```
examples/graph/
├── paths.py            central data/result paths + checkpoint discovery (import, don't hardcode)
├── data_gen/           data generation
│   ├── generate_tree.py      random family tree
│   ├── family_tree.py        FamilyTree (BFS kinship reasoning) + REL_HOPS/hops_for
│   ├── qagen.py              9-category MC QA, category rebalance, uniform letters → parquets
│   ├── generate_variants.py  4 name-swap variants {alex,ben,carl,dan}
│   └── generate_narrative.py prose corpus (exp3 ICL)
├── training/
│   ├── train.py              unified trainer (MODE=variants|stability|single)
│   ├── masked_answer_dataset.py   loss only on the answer letter
│   └── mc_eval.py            generation-time MC eval dataset
├── evaluation/
│   ├── eval.py               ICL / cartridge eval (+ --ablate-keys/values-from)
│   └── analyze.py            per-category / per-hop / cross-file accuracy stats
├── comparison/               cartridge comparison engine (see METHODOLOGY.md)
│   ├── kv_compare.py         shared primitives (directions, spectra, plots)
│   ├── compare.py            unified compare CLI (+ --spectra)
│   └── dynamics.py           K/V rotation across training checkpoints
├── scripts/
│   ├── run_all.sh            full pipeline (data → exp1/2/3 + dynamics)
│   ├── run_exp4_stability.sh seed-variation stability runs + compare
│   └── run_icl_baseline.sh   exp3 corpus-vs-narrative ICL
└── data/                     generated artifacts
    ├── base/                 base tree, corpora, base parquets
    └── variants/{alex,ben,carl,dan}/
```

Results live **outside** the package, under repo-root `outputs_graph/` (override
with `$CARTRIDGES_OUTPUT_DIR_GRAPH`): `exp1_init_kv/ exp2_train/ exp3_icl/
exp4_stability/`. Full map + `paths.py` helpers → [`OUTPUTS.md`](OUTPUTS.md).

## Dataset (current)

9-category multiple-choice kinship QA, **rebalanced** to a target mixture (default
on): reasoning categories ≈ natural proportions, verification+existence ≈ 25%
combined (raw is 88%). Each record carries `hops` (1=adjacent, 2=grand/uncle/aunt,
3=cousin/distant) for per-hop eval. Correct-answer letters are **uniform** (5-opt:
20% A–E; 3-opt: 33% A–C). Train/test split is **stratified by category**, random
**by question** by default (`--split-mode person` holds out whole people for a
structural-generalization test). Loss = masked answer-letter only.

## Experiments

| # | Question | Driver |
|---|----------|--------|
| 1 | What does the *init* cache encode? (no training) | `compare.py --source init` |
| 2 | 4 cartridges, shared Alex init, different data | `train.py MODE=variants` → `compare.py --source trained` |
| 3 | ICL baseline: structured corpus vs prose (per-hop vs cartridge) | `eval.py --mode icl-cot` |
| 4 | Training stability across seeds (noise floor) | `train.py MODE=stability` → `compare.py --run-prefix` |
| — | How do K/V vectors rotate during training? | `dynamics.py` (reads `cache-step*.pt`) |

## Setup

```bash
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_OUTPUT_DIR=$PWD/outputs            # required by cartridges/__init__
export CARTRIDGES_OUTPUT_DIR_GRAPH=$PWD/outputs_graph # graph results root
export CARTRIDGES_WANDB_PROJECT=cartridges-graph CARTRIDGES_WANDB_ENTITY=<you>
```

## Recompute everything (commands)

The one-shot pipeline (data → exp1 init compare → exp2 train+compare → exp3 ICL →
cartridge eval). Skip stages with `SKIP_<stage>=1`; multi-GPU with `NPROC=N`:

```bash
bash examples/graph/scripts/run_all.sh
bash examples/graph/scripts/run_exp4_stability.sh   # exp4 (edit GPUS to hardware)
```

Or step by step (from repo root; all modules run as `python -m examples.graph.…`):

```bash
# 1. data — base tree, QA (rebalanced, hops, uniform letters), 4 variants, narratives
python -m examples.graph.data_gen.generate_tree --n-people 45 --max-depth 6 \
    --min-kids 1 --max-kids 2 --founders 1 --spouse-prob 0.95
python -m examples.graph.data_gen.qagen                       # base; --split-mode person for structural split
python -m examples.graph.data_gen.generate_variants --new-names Alex,Ben,Carl,Dan
python -m examples.graph.data_gen.generate_narrative
for v in alex ben carl dan; do
  python -m examples.graph.data_gen.generate_narrative --tree examples/graph/data/variants/$v/family_tree.json
done

# 2. exp1 — init-KV compare across variants (needs the model)
python -m examples.graph.comparison.compare --source init \
    --names alex,ben,carl,dan --spectra --out-dir outputs_graph/exp1_init_kv

# 3. exp2 — train 4 cartridges (shared Alex init). Defaults: 100 steps, ckpt every 20.
MODE=variants python -m examples.graph.training.train
#   single GPU per variant:  MODE=variants VARIANT_ONLY=alex python -m examples.graph.training.train
#   multi-GPU:                MODE=variants torchrun --standalone --nproc_per_node=2 -m examples.graph.training.train
#   longer run:               MAX_STEPS=-1 EPOCHS=10 SAVE_EVERY=200 MODE=variants python -m examples.graph.training.train

# 3b. exp2 — trained-KV compare (shared init → diffs come only from train data)
python -m examples.graph.comparison.compare --source trained \
    --ckpt-root outputs_graph/exp2_train --names alex,ben,carl,dan \
    --init-corpus examples/graph/data/variants/alex/family_tree_corpus.txt \
    --localize-names Alex,Ben,Carl,Dan --spectra \
    --out-dir outputs_graph/exp2_train/compare

# 3c. training dynamics — how K/V vectors rotate (uses the cache-step*.pt series)
python -m examples.graph.comparison.dynamics \
    --run-dir outputs_graph/exp2_train/alex \
    --init-corpus examples/graph/data/variants/alex/family_tree_corpus.txt --spectra \
    --out-dir outputs_graph/exp2_train/alex/dynamics

# 4. cartridge accuracy eval (CoT) + per-hop analysis
python -m examples.graph.evaluation.eval --mode cartridge-cot \
    --checkpoint "$(find outputs_graph/exp2_train/alex -name cache_last.pt | head -1)" \
    --variant-dir examples/graph/data/variants/alex \
    --output outputs_graph/exp2_train/alex/eval/results.json

# 5. exp3 — ICL baseline + compare cartridge vs ICL PER HOP
python -m examples.graph.evaluation.eval --mode icl-cot --n-shot 0 \
    --corpus-path examples/graph/data/variants/alex/family_tree_corpus.txt \
    --variant-dir examples/graph/data/variants/alex \
    --output outputs_graph/exp3_icl/alex/corpus/results.json
python -m examples.graph.evaluation.analyze \
    outputs_graph/exp2_train/alex/eval/results.json \
    outputs_graph/exp3_icl/alex/corpus/results.json    # → accuracy-by-hop & by-category tables

# 4b/exp4 — cross-variant eval (structure vs surface): run alex cartridge on ben test
python -m examples.graph.evaluation.eval --mode cartridge-cot \
    --checkpoint "$(find outputs_graph/exp2_train/alex -name cache_last.pt | head -1)" \
    --variant-dir examples/graph/data/variants/ben \
    --output outputs_graph/exp2_train/alex/eval_on_ben/results.json

# Paper-2 ablation: swap learned K (or V) vectors in from another cartridge
python -m examples.graph.evaluation.eval --mode cartridge-cot \
    --checkpoint outputs_graph/exp2_train/alex/.../cache_last.pt \
    --ablate-keys-from outputs_graph/exp2_train/ben/.../cache_last.pt \
    --variant-dir examples/graph/data/variants/alex \
    --output outputs_graph/exp2_train/alex/eval_ablate_keys/results.json
```

All code is invoked as `python -m examples.graph.<pkg>.<module>` from the repo root.
