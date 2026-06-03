"""
Unified cartridge comparison CLI.

One tool to compare 1..N kinship cartridges and emit a standard set of output
files. Replaces compare_kv.py, compare_init_kv.py and compare_alex_stability.py.

Two source modes:
  --source init      build untrained caches from variant corpora (exp1)
  --source trained   load trained cache_last.pt checkpoints     (exp2 / exp4)

It answers the two methodology questions (see METHODOLOGY.md):
  (a) WHERE are entities encoded?  → localization.json + slot_* heatmaps
      (per-slot divergence; name-slot-vs-other ratio)
  (b) HOW stable is training?      → stability.json
      (run-to-run angle = the noise floor you compare (a) against)

Standard outputs (always under --out-dir):
  compare_summary.json   pairwise per-layer cos/angle/rel_l2/norm_ratio + overall
  localization.json      per-slot top divergence + name-slot ratio (if names known)
  stability.json         overall angle mean + std-across-pairs
  heatmap_*.png          pair × layer maps
  slot_<a>_<b>_*.png     per-pair layer × slot maps (red vlines = name slots)

Examples:
  # exp1: init caches across 4 variants, localize each variant's swapped name
  python -m examples.graph.comparison.compare --source init \
      --names alex,ben,carl,dan --out-dir outputs_graph/exp1_init_kv

  # exp2: trained cartridges across variants, shared Alex init corpus
  python -m examples.graph.comparison.compare --source trained \
      --ckpt-root outputs_graph/exp2_train --names alex,ben,carl,dan \
      --init-corpus data/variants/alex/family_tree_corpus.txt \
      --localize-names Alex,Ben,Carl,Dan \
      --out-dir outputs_graph/exp2_train/compare

  # exp4: stability of one group (5 seeds, same data) → noise floor
  python -m examples.graph.comparison.compare --source trained \
      --ckpt-root outputs_graph/exp4_stability \
      --run-prefix ben_on_ben --n-runs 5 \
      --out-dir outputs_graph/exp4_stability/ben_on_ben_compare
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from examples.graph import paths
from examples.graph.comparison import kv_compare as kvc


# ── Input resolution ─────────────────────────────────────────────────────────
def parse_inputs(spec: str) -> dict[str, str]:
    """'a=path1,b=path2' → {'a': 'path1', 'b': 'path2'}."""
    out = {}
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        label, _, ref = part.partition("=")
        out[label.strip()] = ref.strip()
    return out


def resolve_labels(args) -> list[str]:
    if args.inputs:
        return list(parse_inputs(args.inputs).keys())
    if args.run_prefix:
        return [f"{args.run_prefix}_run{i + 1}" for i in range(args.n_runs)]
    if args.names:
        return [n.strip().lower() for n in args.names.split(",") if n.strip()]
    raise ValueError("provide one of --inputs / --run-prefix / --names")


def resolve_trained_ckpt(label: str, args) -> Path:
    if args.inputs:
        ref = Path(parse_inputs(args.inputs)[label])
        return ref if ref.is_file() else kvc.find_cache_last(ref)
    return kvc.find_cache_last(Path(args.ckpt_root) / label)


def resolve_init_corpus(label: str, args) -> Path:
    if args.inputs:
        return Path(parse_inputs(args.inputs)[label])
    return paths.variant_corpus(label)


# ── Load directions for every label ──────────────────────────────────────────
def load_all(args, labels: list[str], collect_raw: bool = False):
    """Returns (K_dirs, V_dirs[, K_raw, V_raw]) dicts: label → tensors.

    K_dirs/V_dirs hold (L, T, D) head-mean directions. When `collect_raw`, also
    returns per-layer raw [(1, H, T, D)] lists (needed for singular value spectra).
    """
    K_dirs, V_dirs, K_raw, V_raw = {}, {}, {}, {}
    if args.source == "init":
        from transformers import AutoTokenizer
        from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM

        print(f"loading {args.model} on {args.device} for init builds …")
        model = HFModelConfig(
            pretrained_model_name_or_path=args.model,
            model_cls=FlexQwen3ForCausalLM,
        ).instantiate()
        model.to(args.device).eval()
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        for label in labels:
            corpus = resolve_init_corpus(label, args)
            K, V = kvc.build_init_cache(model, tokenizer, corpus, args.max_tokens)
            K_dirs[label] = kvc.stack_directions(K)
            V_dirs[label] = kvc.stack_directions(V)
            if collect_raw:
                K_raw[label], V_raw[label] = K, V
            print(f"  {label}: {tuple(K_dirs[label].shape)}  ({corpus})")
    else:
        for label in labels:
            pt = resolve_trained_ckpt(label, args)
            K, V = kvc.load_trained_cache(pt)
            K_dirs[label] = kvc.stack_directions(K)
            V_dirs[label] = kvc.stack_directions(V)
            if collect_raw:
                K_raw[label], V_raw[label] = K, V
            print(f"  {label}: {tuple(K_dirs[label].shape)}  ({pt})")
    if collect_raw:
        return K_dirs, V_dirs, K_raw, V_raw
    return K_dirs, V_dirs


# ── Name-slot resolution → callable pair_slots(a, b) → list[int] ─────────────
def build_pair_slots(args, labels: list[str]):
    if args.name_slots == "off":
        return (lambda a, b: []), {}

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    per_label: dict[str, list[int]] = {}

    if args.source == "init":
        # each variant localizes its own swapped name in its own corpus
        meta = (json.loads(paths.VARIANTS_META.read_text())
                if paths.VARIANTS_META.exists() else {"variants": []})
        cap = {v.lower(): v for v in meta.get("variants", [])}
        for label in labels:
            name = cap.get(label, label.capitalize())
            corpus = resolve_init_corpus(label, args)
            per_label[label] = kvc.find_name_slots(tokenizer, corpus, name, args.max_tokens)
            print(f"  name slots {label} ({name}): {per_label[label][:6]}"
                  f"{'…' if len(per_label[label]) > 6 else ''}")
        # pair mask = union of the two variants' name slots
        return (lambda a, b: sorted(set(per_label.get(a, []) + per_label.get(b, [])))), per_label

    # trained: shared init corpus + explicit names → global slot mask for all pairs
    if not (args.init_corpus and args.localize_names):
        print("  (trained source without --init-corpus/--localize-names → no localization)")
        return (lambda a, b: []), {}
    corpus = Path(args.init_corpus)
    names = [n.strip() for n in args.localize_names.split(",") if n.strip()]
    slots = sorted(set().union(*[
        set(kvc.find_name_slots(tokenizer, corpus, n, args.max_tokens)) for n in names
    ]))
    print(f"  global name slots ({names}) in {corpus.name}: {slots[:8]}"
          f"{'…' if len(slots) > 8 else ''}")
    return (lambda a, b: slots), {"_global": slots}


# ── Core comparison + output writing ─────────────────────────────────────────
def run(K_dirs, V_dirs, labels, pair_slots, out_dir: Path, top_k: int, no_plots: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = list(itertools.combinations(labels, 2))
    n_layers, n_tokens, _ = K_dirs[labels[0]].shape

    summary = {
        "labels": labels, "n_pairs": len(pairs),
        "n_layers": n_layers, "n_tokens": n_tokens, "pairs": {},
    }
    localization = {"n_tokens": n_tokens, "pairs": {}}
    all_K_angle, all_V_angle = [], []

    print(f"\n=== pairwise ({len(pairs)} pairs) ===")
    print(f"{'pair':<24} {'K angle°':>9} {'V angle°':>9} "
          f"{'K rel_l2':>9} {'V rel_l2':>9} {'K cos':>7} {'V cos':>7}")

    for a, b in pairs:
        mK = kvc.directional_metrics(K_dirs[a], K_dirs[b])
        mV = kvc.directional_metrics(V_dirs[a], V_dirs[b])
        tag = f"{a}|{b}"
        slots = pair_slots(a, b)

        summary["pairs"][tag] = kvc.pair_summary(mK, mV)
        localization["pairs"][tag] = kvc.slot_localization(mK, mV, slots, n_tokens, top_k)
        all_K_angle.append(mK["angle_deg"])
        all_V_angle.append(mV["angle_deg"])

        s = summary["pairs"][tag]
        print(f"{tag:<24} {s['K_angle_mean']:>9.3f} {s['V_angle_mean']:>9.3f} "
              f"{s['K_rel_l2_mean']:>9.4f} {s['V_rel_l2_mean']:>9.4f} "
              f"{s['K_cos_mean']:>7.4f} {s['V_cos_mean']:>7.4f}")
        loc = localization["pairs"][tag]
        if "K_name_slot_ratio" in loc:
            print(f"    K name-slot angle={loc['K_name_slot_angle']:.2f}  "
                  f"other={loc['K_other_slot_angle']:.2f}  "
                  f"ratio={loc['K_name_slot_ratio']:.2f}")

        if not no_plots:
            for kv_tag, m in (("K", mK), ("V", mV)):
                kvc.heatmap(
                    m["angle_deg"], title=f"{a} vs {b}  {kv_tag} angle°",
                    out_path=out_dir / f"slot_{a}_{b}_{kv_tag}_angle.png",
                    highlight=slots, cmap="viridis", vmin=0.0)

    # ── overall / stability aggregate ────────────────────────────────────────
    K_all = np.stack(all_K_angle, axis=0)   # (n_pairs, L, T)
    V_all = np.stack(all_V_angle, axis=0)
    overall = {
        "K_angle_mean": float(K_all.mean()),
        "K_angle_std_across_pairs": float(K_all.mean(axis=(1, 2)).std()),
        "V_angle_mean": float(V_all.mean()),
        "V_angle_std_across_pairs": float(V_all.mean(axis=(1, 2)).std()),
    }
    summary["overall"] = overall
    (out_dir / "compare_summary.json").write_text(json.dumps(summary, indent=2))
    (out_dir / "localization.json").write_text(json.dumps(localization, indent=2))
    (out_dir / "stability.json").write_text(json.dumps({
        "labels": labels, "n_pairs": len(pairs),
        "note": "Interpret as a noise floor only when all inputs share the same "
                "train data (e.g. seed-variation runs).",
        **overall,
    }, indent=2))

    print(f"\n=== overall ({len(pairs)} pairs) ===")
    print(f"K angle: mean={overall['K_angle_mean']:.3f}°  "
          f"std/pairs={overall['K_angle_std_across_pairs']:.3f}°")
    print(f"V angle: mean={overall['V_angle_mean']:.3f}°  "
          f"std/pairs={overall['V_angle_std_across_pairs']:.3f}°")

    if not no_plots:
        kvc.heatmap(K_all.mean(axis=0), title=f"mean K angle° ({len(pairs)} pairs)",
                    out_path=out_dir / "heatmap_K_angle.png", cmap="viridis", vmin=0)
        kvc.heatmap(V_all.mean(axis=0), title=f"mean V angle° ({len(pairs)} pairs)",
                    out_path=out_dir / "heatmap_V_angle.png", cmap="viridis", vmin=0)

    print(f"\nsaved → {out_dir}/  (compare_summary.json, localization.json, stability.json)")


def write_spectra(K_raw, V_raw, labels, out_dir: Path, no_plots: bool):
    """Per-label singular value spectra (Paper 2): keys flat, values decay."""
    out_dir.mkdir(parents=True, exist_ok=True)
    spectra = {"labels": labels, "K_participation": {}, "V_participation": {}}
    K_mean, V_mean = {}, {}
    for label in labels:
        spK = kvc.singular_value_spectra(K_raw[label])
        spV = kvc.singular_value_spectra(V_raw[label])
        spectra["K_participation"][label] = spK["participation"].tolist()
        spectra["V_participation"][label] = spV["participation"].tolist()
        K_mean[label] = spK["mean_spectrum"]
        V_mean[label] = spV["mean_spectrum"]
    (out_dir / "spectra.json").write_text(json.dumps(spectra, indent=2))
    if not no_plots:
        kvc.plot_spectra(K_mean, title="Key singular value spectra",
                         out_path=out_dir / "spectra_K.png")
        kvc.plot_spectra(V_mean, title="Value singular value spectra",
                         out_path=out_dir / "spectra_V.png")
    print(f"  spectra → {out_dir}/spectra.json (+ spectra_K/V.png)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, choices=["init", "trained"])
    # input selection (one of)
    p.add_argument("--names", default=None, help="CSV of labels/variants")
    p.add_argument("--inputs", default=None, help="explicit 'label=path,...'")
    p.add_argument("--run-prefix", default=None, help="stability group prefix → <prefix>_run<i>")
    p.add_argument("--n-runs", type=int, default=5)
    p.add_argument("--ckpt-root", default=None, help="(trained) root holding <label>/ dirs")
    # init source
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # localization
    p.add_argument("--name-slots", choices=["auto", "off"], default="auto")
    p.add_argument("--init-corpus", default=None,
                   help="(trained) corpus the caches were initialized from (for name slots)")
    p.add_argument("--localize-names", default=None,
                   help="(trained) CSV of names to mark in --init-corpus")
    # output
    p.add_argument("--out-dir", required=True)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--top-k", type=int, default=30)
    p.add_argument("--spectra", action="store_true",
                   help="also emit per-label singular value spectra (Paper 2)")
    args = p.parse_args()

    labels = resolve_labels(args)
    print(f"source={args.source}  labels={labels}")

    loaded = load_all(args, labels, collect_raw=args.spectra)
    if args.spectra:
        K_dirs, V_dirs, K_raw, V_raw = loaded
    else:
        K_dirs, V_dirs = loaded
    # sanity: identical shapes
    ref = K_dirs[labels[0]].shape
    for label in labels:
        assert K_dirs[label].shape == ref, f"{label} shape {K_dirs[label].shape} != {ref}"

    pair_slots, _ = build_pair_slots(args, labels)
    run(K_dirs, V_dirs, labels, pair_slots, Path(args.out_dir), args.top_k, args.no_plots)
    if args.spectra:
        write_spectra(K_raw, V_raw, labels, Path(args.out_dir), args.no_plots)


if __name__ == "__main__":
    main()
