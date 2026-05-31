# Cartridge comparison methodology

How we compare kinship cartridges (trainable KV caches) against each other, and
what files each comparison emits. One tool — `comparison/compare.py` — produces
the same standard outputs for any set of **1..N** cartridges, so every
experiment is read the same way.

There are two questions we ask of a set of cartridges:

| Goal | Compare… | Read from |
|------|----------|-----------|
| **(a) Where are entities encoded?** | *different* cartridges (different data / different swapped name) | `localization.json`, `slot_*` heatmaps |
| **(b) How stable is training?** | *similar* cartridges (same data, different seed) | `stability.json` |

(b) gives the **noise floor**. A divergence in (a) only means "the entity moved
this slot" if it is larger than the seed-to-seed divergence from (b) on the same
slots. Always read (a) against (b).

---

## 1. What a cartridge is reduced to

A cartridge holds, per transformer layer, a key tensor and a value tensor of
shape `(1, H, T, D)` — `H` heads, `T` cartridge slots (positions), `D` head dim.

We collapse heads to get **one direction per (layer, slot)**:

```
K[layer] : (1, H, T, D)  --mean over H-->  (T, D)     # stack layers --> (L, T, D)
```

Heads are averaged (not concatenated) so the comparison is about *where the slot
points*, robust to per-head scale. Keys (`K`) and values (`V`) are compared
separately throughout — they encode different things and often diverge
differently.

## 2. Metrics (per layer, per slot)

For two cartridges A, B and each slot direction `dA, dB ∈ R^D`:

| Metric | Formula | Reads as |
|--------|---------|----------|
| `cos` | ⟨dA,dB⟩ / (‖dA‖‖dB‖) | direction agreement, `[-1, 1]` |
| `angle_deg` | `acos(cos)` | 0° = identical; **primary metric** |
| `l2_shift` | ‖dA − dB‖ | absolute movement |
| `rel_l2` | ‖dA − dB‖ / ‖dA‖ | movement relative to magnitude |
| `norm_ratio` | ‖dA‖ / ‖dB‖ | magnitude inflation/deflation |

`angle_deg` is the headline number: scale-invariant, additive across slots,
and directly comparable between (a) and (b).

## 3. Aggregation

- **Per pair** → mean over all (layer, slot) for each metric (`compare_summary.json`).
- **Per slot** → mean over layers → ranks the slots where divergence concentrates
  (`localization.json`, top-K).
- **Name-slot mask** → average angle *inside* the slots that hold the swapped
  name vs *outside* them. `name_slot_ratio = inside / outside`. A ratio ≫ 1 means
  the entity change is localized to its name slots (goal **a**).
- **Overall** → mean angle across all pairs + std across pairs. For same-data
  seed runs this std is the stability floor (goal **b**).

Name slots are found by tokenizing the init corpus and locating positions where
the swapped name is a single token (`kv_compare.find_name_slots`).

---

## 4. Standard output files

Every `compare.py` run writes to `--out-dir`:

| File | Content | Used for |
|------|---------|----------|
| `compare_summary.json` | per-pair scalar metrics (K/V cos, angle, rel_l2, norm_ratio) + `overall` | headline comparison |
| `localization.json` | per-pair top-K diverging slots + name-slot-vs-other ratio | **(a)** entity localization |
| `stability.json` | overall mean angle + std-across-pairs | **(b)** noise floor |
| `heatmap_K_angle.png`, `heatmap_V_angle.png` | mean (layer × slot) angle over all pairs | overview map |
| `slot_<a>_<b>_K_angle.png`, `…_V_angle.png` | per-pair (layer × slot) angle; red vlines = name slots | **(a)** see *where* on the cartridge |

The schema is identical regardless of source (`init` vs `trained`) or count of
inputs, so results compose across experiments.

---

## 5. Sources

`compare.py --source` chooses how the N caches are obtained:

- `--source init` — build *untrained* caches by running each corpus through
  `KVFromText` (needs the model). Isolates what the **initialization** alone
  encodes. Each variant localizes its own swapped name in its own corpus.
- `--source trained` — load `cache_last.pt` checkpoints. For exp2 (shared Alex
  init) pass `--init-corpus` + `--localize-names` so the name-slot mask is the
  anchor's name positions, shared across all pairs.

---

## 6. The three comparison experiments

**Exp 1 — init divergence** (different init per variant):
```
python -m examples.graph.comparison.compare --source init \
    --names alex,ben,carl,dan --out-dir outputs_graph/exp1_init_kv
```
Measures the effect of swapping one name *before* any training. Each variant's
name slots are auto-detected.

**Exp 2 — trained divergence** (shared Alex init, different train data):
```
python -m examples.graph.comparison.compare --source trained \
    --ckpt-root outputs_graph/exp2_train --names alex,ben,carl,dan \
    --init-corpus data/variants/alex/family_tree_corpus.txt \
    --localize-names Alex,Ben,Carl,Dan \
    --out-dir outputs_graph/exp2_train/compare
```
Same init for all 4 → post-train differences come *only* from train data. Tells
us where the structural/name signal lands after learning.

**Exp 4 — stability** (same data, different seed):
```
python -m examples.graph.comparison.compare --source trained \
    --ckpt-root outputs_graph/exp4_stability \
    --run-prefix ben_on_ben --n-runs 5 --name-slots off \
    --out-dir outputs_graph/exp4_stability/ben_on_ben_compare
```
`stability.json.K_angle_mean` here is the floor: any exp1/exp2 slot angle below
it is indistinguishable from training noise.

---

## 7. How to read a result

1. Open `stability.json` for the relevant noise floor (e.g. mean K angle ≈ θ₀).
2. Open `localization.json` for the comparison of interest. A slot whose
   `angle_deg` ≫ θ₀ carries real signal; near θ₀ is noise.
3. If `name_slot_ratio ≫ 1`, the swapped entity is localized to its name slots.
   If ≈ 1, the change is spread across the cartridge (entangled, not localized).
4. Use `slot_*` heatmaps to see *which layers* concentrate the change.
