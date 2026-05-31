# Kinship-graph cartridge experiments

Do trainable KV-cache **cartridges** capture the relational structure of a family
tree? We train cartridges on a 45-person / 6-generation tree over a frozen
Qwen3-1.7B and probe them with controlled name-swap variants. See `REPORT.md`
for findings and `METHODOLOGY.md` for how cartridges are compared.

## Layout

```
examples/graph/
├── paths.py            central data/result paths (import, don't hardcode)
├── data_gen/           data generation
│   ├── generate_tree.py      random family tree
│   ├── family_tree.py        FamilyTree class (BFS kinship reasoning)
│   ├── qagen.py              8-category MC QA → parquets
│   ├── generate_variants.py  4 name-swap variants {alex,ben,carl,dan}
│   └── generate_narrative.py prose corpus (exp3 ICL)
├── training/
│   ├── train.py              unified trainer (MODE=variants|stability|single)
│   ├── masked_answer_dataset.py   loss only on the answer letter
│   └── mc_eval.py            generation-time MC eval dataset
├── evaluation/
│   ├── eval.py               ICL / cartridge eval (icl, icl-cot, cartridge[-cot])
│   └── analyze.py            per-category accuracy stats
├── comparison/               cartridge comparison engine (see METHODOLOGY.md)
│   ├── kv_compare.py         shared primitives
│   └── compare.py            unified CLI → standard output files
├── scripts/
│   ├── run_all.sh            full pipeline (data → exp1/2/3)
│   ├── run_exp4_stability.sh seed-variation stability runs + compare
│   └── run_icl_baseline.sh   exp3 corpus-vs-narrative ICL
└── data/                     generated artifacts
    ├── base/                 base tree, corpora, base parquets
    └── variants/{alex,ben,carl,dan}/
```

Results live **outside** the package, under repo-root `outputs_graph/` (override
with `$CARTRIDGES_OUTPUT_DIR_GRAPH`):
`exp1_init_kv/ exp2_train/ exp3_icl/ exp4_stability/`.

## Experiments

| # | Question | Driver |
|---|----------|--------|
| 1 | What does the *init* cache encode? (no training) | `compare.py --source init` |
| 2 | 4 cartridges, shared Alex init, different data | `train.py MODE=variants` → `compare.py --source trained` |
| 3 | ICL baseline: structured corpus vs prose | `eval.py --mode icl-cot` |
| 4 | Training stability across seeds (noise floor) | `train.py MODE=stability` → `compare.py --run-prefix` |

## Quick start

```bash
# env
export CARTRIDGES_DIR=$PWD
export CARTRIDGES_WANDB_PROJECT=cartridges-graph CARTRIDGES_WANDB_ENTITY=<you>

# generate data + run exp1/2/3
bash examples/graph/scripts/run_all.sh

# stability (exp4) — edit GPUS to your hardware
bash examples/graph/scripts/run_exp4_stability.sh
```

Run any module directly, e.g.:

```bash
MODE=variants VARIANT_ONLY=alex python -m examples.graph.training.train
python -m examples.graph.evaluation.eval --mode cartridge-cot \
    --checkpoint <run>/cache_last.pt --variant-dir examples/graph/data/variants/alex
```

All code is invoked as `python -m examples.graph.<pkg>.<module>` from the repo
root (relies on the package import path).
