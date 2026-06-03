# Outputs & checkpoints — canonical map

Single source of truth for **where every artifact in the graph experiments is
written**. All path math lives in [`paths.py`](paths.py) — import it, don't
hard-code. This file documents the layout that `paths.py` produces.

---

## Roots & env vars

| Env var | Used by | Default |
|---|---|---|
| `CARTRIDGES_OUTPUT_DIR_GRAPH` | `paths.py` → routes ALL graph results | `<repo>/outputs_graph` |
| `CARTRIDGES_OUTPUT_DIR` | required by `cartridges/__init__` (core) | — (scripts set it to the same root) |
| `CARTRIDGES_DIR` | core | repo root |

Results root: **`outputs_graph/`** (override via `CARTRIDGES_OUTPUT_DIR_GRAPH`).
Scripts export both `CARTRIDGES_OUTPUT_DIR` and `CARTRIDGES_OUTPUT_DIR_GRAPH`.

---

## Inputs — data (in-package, NOT under outputs_graph)

```
examples/graph/data/
├── base/                              # single un-swapped tree
│   ├── family_tree.json
│   ├── family_tree_corpus.txt         # structured corpus (ICL + KVFromText init)
│   ├── family_tree_narrative.txt      # prose corpus (exp3 ICL)
│   ├── {train,test}_mc.parquet        # MC QA (assistant = letter only)
│   ├── {train,test}_meta_mc.json      # per-question metadata
│   └── split_meta.json
└── variants/{alex,ben,carl,dan}/      # isomorphic trees, one name swapped
    ├── (same files as base/)
    └── ../variants_meta.json
```
Accessors: `paths.BASE_*`, `paths.variant_{dir,corpus,train,test}(name)`.

---

## Outputs root layout

```
outputs_graph/                         # = CARTRIDGES_OUTPUT_DIR_GRAPH
├── exp1_init_kv/                      # exp1: init-KV compare (no training)
├── exp2_train/                        # exp2: trained cartridges + compare + dynamics
│   ├── <variant>/                     #   training output_dir (alex|ben|carl|dan|base)
│   │   └── <launch_id>/<run_id>/      #   ← pydrantic-nested run_dir (checkpoints here)
│   │   ├── eval/                      #   per-variant accuracy eval (results.json)
│   │   └── dynamics/                  #   comparison/dynamics.py rotation analysis
│   └── compare/                       #   exp2 trained-KV compare (compare.py)
├── exp3_icl/                          # exp3: ICL baselines (no cartridge)
│   ├── base/  <variant>/              #   each: corpus/ + narrative/ + compare.txt
└── exp4_stability/                    # exp4: stability / noise floor
    ├── <init>_on_<data>_run{i}/       #   training output_dir per stability run
    │   └── <launch_id>/<run_id>/      #   ← checkpoints here
    ├── alex_stability_compare/        #   noise-floor compare (5 alex seeds)
    └── type_compare/                  #   identity-vs-init compare
```

> **Note (2026-06-02):** stability outputs were consolidated from the legacy
> `exp2_stability/` name to **`exp4_stability/`** (matches the "Exp 4" numbering
> in REPORT.md). `paths.EXP4_DIR`, `compare_results.sh` (`RES` default) and
> REPORT.md all point here now.

---

## Checkpoints (training) — the pydrantic nesting

`train.py` sets `TrainConfig.output_dir`; pydrantic then creates the real run
directory **two levels below** it:

```
<output_dir>/<launch_id>/<run_id>/
   launch_id = <YYYY-MM-DD-HH-MM-SS>-<script_id>
   run_id    = <uuid>
```

`output_dir` per MODE (`paths` helper → result):

| MODE | helper | output_dir |
|---|---|---|
| `variants` (exp2) | `paths.exp2_variant_dir(v)` | `exp2_train/<variant>/` |
| `single` | `paths.exp2_variant_dir('base')` | `exp2_train/base/` |
| `stability` (exp4) | `paths.exp4_run_dir(init,data,i)` | `exp4_stability/<init>_on_<data>_run{i}/` |

Inside each `run_dir`:

| File | What | Cadence |
|---|---|---|
| `cache-step{N}.pt` | trainable KV checkpoint | every `SAVE_EVERY` (env, default 20) up to `MAX_STEPS` (env, default 100) |
| `cache_last.pt` | symlink → latest `cache-step` | updated each save |
| `config.yaml` | resolved run config | once |
| `peft_model*/` | (only if PEFT tuning) | — |

**Discovery — never hard-code the timestamp/uuid:**
- `paths.latest_checkpoint(output_dir)` → newest `cache_last.pt` (rglob, mtime).
- `paths.run_checkpoints(output_dir)` → sorted `[(step, path)]` of all `cache-step*.pt`.

`kv_compare.find_cache_last` and `dynamics.find_step_checkpoints` delegate to
these, so there is one discovery implementation.

---

## Comparison outputs (`comparison/compare.py`, `--out-dir`)

| Exp | canonical out-dir | helper |
|---|---|---|
| exp1 | `exp1_init_kv/` | `paths.EXP1_DIR` |
| exp2 | `exp2_train/compare/` | `paths.exp2_compare_dir()` |
| exp4 | `exp4_stability/<group>_compare/` | (under `paths.EXP4_DIR`) |

Files written (identical schema for any source / N inputs):
- `compare_summary.json` — per-pair K/V cos·angle·rel_l2·norm_ratio + `overall`
- `localization.json` — per-pair top-K diverging slots + name-slot ratio
- `stability.json` — overall mean angle + std-across-pairs (noise floor)
- `heatmap_{K,V}_angle.png` — mean (layer × slot) angle over all pairs
- `slot_<a>_<b>_{K,V}_angle.png` — per-pair (layer × slot), red vlines = name slots
- `spectra.json`, `spectra_{K,V}.png` — **only with `--spectra`** (singular value spectra)

---

## Dynamics outputs (`comparison/dynamics.py`, `--out-dir`)

Canonical: `paths.dynamics_dir(variant)` → `exp2_train/<variant>/dynamics/`.

- `dynamics.json` — steps + per-step overall/per-layer K&V rotation angles
- `rotation_incremental.png` — mean `angle(Z_t, Z_{t-1})` vs step, K vs V
- `rotation_cumulative.png` — mean `angle(Z_t, Z_0)` vs step, K vs V
- `heatmap_{K,V}_{incremental,cumulative}.png` — (layer × step) angle
- `spectra.json`, `spectra_{K,V}.png` — **only with `--spectra`** (init vs final)

---

## ICL outputs (`evaluation/eval.py` → exp3, via `run_icl_baseline.sh`)

```
exp3_icl/<tag>/
├── corpus/results.json        # ICL on structured corpus
├── narrative/results.json     # ICL on prose corpus
└── compare.txt                # analyze.py side-by-side
```
`<tag>` = `base` or a variant name.

---

## W&B

Each training run logs to project/entity from `CARTRIDGES_WANDB_*`. Run name =
`TrainConfig.name` (e.g. `graph-variant-masked-alex`, `graph-stability-…`).
Inline MC eval logs `acc`; checkpoints are mirrored to W&B when `save_to_wandb`.

---

## paths.py API summary

```python
# data
BASE_*                              # base-tree files
variant_dir/corpus/train/test(name) # variant data files
# output dir anchors
EXP1_DIR EXP2_DIR EXP3_DIR EXP4_DIR
exp2_variant_dir(v)  exp2_compare_dir()  exp4_run_dir(init,data,i)  dynamics_dir(v)
# checkpoint discovery (handle pydrantic nesting)
run_checkpoints(output_dir) -> [(step, Path)]
latest_checkpoint(output_dir) -> Path
```
