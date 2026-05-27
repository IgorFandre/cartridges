"""
Compare INIT-state KV caches across variants (no training), with heads
mixed (mean-aggregated) into a single direction per (layer, slot).

For each variant we build the init cache via KVFromText on its corpus.
We then:
  1. Collapse the head dim by averaging:  K[layer] : (1, H, T, D)
     →  K_dir[layer]  : (T, D)                # one direction per slot
  2. Pairwise compare variants per (layer, slot):
        - cosine       <dA, dB> / (|dA||dB|)
        - angle (deg)  arccos(cos)
        - L2 shift     |dA - dB|
        - norm ratio   |dA| / |dB|
  3. Render heatmaps (layer × slot) and report top-K most-deviating slots.
     Slots holding the swapped name token are highlighted as red vlines.

Usage:
    python examples/graph/compare_init_kv.py --variants alex,ben,steven
    python examples/graph/compare_init_kv.py --no-plots --top-k 30
"""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from transformers import AutoTokenizer

from cartridges.cache import AttnConfig
from cartridges.initialization import KVFromText
from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM

VARIANTS_ROOT = Path(__file__).parent / "variants"


# ── Build init cache ─────────────────────────────────────────────────────────
def build_init_cache(model, tokenizer, corpus_path: Path, max_tokens=None):
    cfg = KVFromText.Config(text_source=str(corpus_path), max_tokens=max_tokens)
    factory = cfg.instantiate()
    attn = AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=getattr(model.config, "head_dim",
                         model.config.hidden_size // model.config.num_attention_heads),
    )
    cache = factory.initialize_kv_cache(tokenizer, model, attn)
    keys   = [t.detach().float().cpu() for t in cache.trainable_keys]
    values = [t.detach().float().cpu() for t in cache.trainable_values]
    return keys, values


# ── Head-averaged direction extraction ──────────────────────────────────────
def head_mean_direction(layer_tensor: torch.Tensor) -> torch.Tensor:
    """(1, H, T, D)  →  (T, D)  via mean over heads."""
    return layer_tensor.squeeze(0).mean(dim=0)  # (T, D)


def stack_directions(per_layer_list: list[torch.Tensor]) -> torch.Tensor:
    """Stack [(T, D)] across layers into (L, T, D)."""
    return torch.stack([head_mean_direction(x) for x in per_layer_list], dim=0)


# ── Pairwise direction-deviation metrics ────────────────────────────────────
def directional_metrics(A: torch.Tensor, B: torch.Tensor):
    """A, B: (L, T, D). Returns dict of (L, T) numpy arrays.

    - cos        cosine similarity
    - angle_deg  acos(cos) in degrees
    - l2_shift   |A - B|         (Frobenius over D)
    - norm_a     |A|
    - norm_b     |B|
    - norm_ratio |A| / |B|
    """
    eps = 1e-12
    dot = (A * B).sum(dim=-1)                        # (L, T)
    nA  = A.norm(dim=-1).clamp_min(eps)
    nB  = B.norm(dim=-1).clamp_min(eps)
    cos = (dot / (nA * nB)).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cos))
    l2_shift = (A - B).norm(dim=-1)
    return {
        "cos":        cos.numpy(),
        "angle_deg":  angle.numpy(),
        "l2_shift":   l2_shift.numpy(),
        "norm_a":     nA.numpy(),
        "norm_b":     nB.numpy(),
        "norm_ratio": (nA / nB).numpy(),
    }


# ── Name-token slot detection (used as visual marker) ───────────────────────
def find_name_slots(tokenizer, corpus_path: Path, name: str) -> list[int]:
    text = corpus_path.read_text()
    ids = tokenizer.encode(text, add_special_tokens=False)
    candidates = [name, " " + name]
    name_id_sets = [tokenizer.encode(c, add_special_tokens=False) for c in candidates]
    positions = []
    for i, tid in enumerate(ids):
        for nid_set in name_id_sets:
            if len(nid_set) == 1 and tid == nid_set[0]:
                positions.append(i)
    return positions


# ── Plot helpers ────────────────────────────────────────────────────────────
def heatmap(arr, *, title: str, out_path: Path, highlight=(),
            cmap="viridis", vmin=None, vmax=None):
    L, T = arr.shape
    fig, ax = plt.subplots(figsize=(max(8, T * 0.04), max(4, L * 0.2)))
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    for s in highlight:
        ax.axvline(s, color="red", linewidth=0.6, alpha=0.7)
    ax.set_xlabel("slot")
    ax.set_ylabel("layer")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variants",      type=str, default=None,
                   help="Comma-separated subset. Default = all in variants_meta.json")
    p.add_argument("--variants-root", type=str, default=str(VARIANTS_ROOT))
    p.add_argument("--model",         type=str, default="Qwen/Qwen3-1.7B")
    p.add_argument("--max-tokens",    type=int, default=None)
    p.add_argument("--device",        type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out-dir",       type=str,
                   default=str(Path(__file__).parent / "kv_init_compare_out"))
    p.add_argument("--no-plots",      action="store_true")
    p.add_argument("--top-k",         type=int, default=20,
                   help="Report top-K slots with largest mean-over-layers angle")
    args = p.parse_args()

    root = Path(args.variants_root)
    meta = json.loads((root / "variants_meta.json").read_text())
    variants = (
        [v.strip().lower() for v in args.variants.split(",")] if args.variants
        else [v.lower() for v in meta["variants"]]
    )
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # one shared model+tokenizer
    print(f"loading {args.model} on {args.device} …")
    hf_cfg = HFModelConfig(
        pretrained_model_name_or_path=args.model,
        model_cls=FlexQwen3ForCausalLM,
    )
    model = hf_cfg.instantiate()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.to(args.device).eval()

    print(f"variants: {variants}")
    K_dirs: dict[str, torch.Tensor] = {}   # (L, T, D)
    V_dirs: dict[str, torch.Tensor] = {}
    name_slots: dict[str, list[int]] = {}
    for v in variants:
        corpus = root / v / "family_tree_corpus.txt"
        K, V = build_init_cache(model, tokenizer, corpus, args.max_tokens)
        K_dirs[v] = stack_directions(K)
        V_dirs[v] = stack_directions(V)
        name = next(n for n in meta["variants"] if n.lower() == v)
        name_slots[v] = find_name_slots(tokenizer, corpus, name)
        L, T, D = K_dirs[v].shape
        print(f"  {v}: L={L}  T={T}  D={D}  name='{name}' slots={name_slots[v][:6]}"
              f"{'…' if len(name_slots[v]) > 6 else ''}")

    n_layers, n_tokens, _ = K_dirs[variants[0]].shape
    pairs = list(itertools.combinations(variants, 2))

    summary = {
        "variants":     variants,
        "name_slots":   name_slots,
        "n_layers":     n_layers,
        "n_tokens":     n_tokens,
        "pairs":        {},
    }

    print(f"\n=== directional deviation (heads averaged) ===")
    print(f"{'pair':<22} {'K angle (mean)':>15} {'V angle (mean)':>15} "
          f"{'K l2_shift':>11} {'V l2_shift':>11}")

    for a, b in pairs:
        mK = directional_metrics(K_dirs[a], K_dirs[b])
        mV = directional_metrics(V_dirs[a], V_dirs[b])
        tag = f"{a}|{b}"

        # ── aggregate stats ────────────────────────────────────────────────
        per_pair = {
            "K_angle_mean":     float(mK["angle_deg"].mean()),
            "V_angle_mean":     float(mV["angle_deg"].mean()),
            "K_l2_shift_mean":  float(mK["l2_shift"].mean()),
            "V_l2_shift_mean":  float(mV["l2_shift"].mean()),
            "K_norm_ratio_med": float(np.median(mK["norm_ratio"])),
            "V_norm_ratio_med": float(np.median(mV["norm_ratio"])),
        }

        # name-slot vs other-slot analysis
        slots_set = sorted(set(name_slots.get(a, []) + name_slots.get(b, [])))
        if slots_set:
            mask = np.zeros(n_tokens, dtype=bool); mask[slots_set] = True
            for key, arr in [("K_angle_deg", mK["angle_deg"]),
                             ("V_angle_deg", mV["angle_deg"]),
                             ("K_l2_shift",  mK["l2_shift"]),
                             ("V_l2_shift",  mV["l2_shift"])]:
                per_pair[f"{key}_name_slots"]  = float(arr[:, mask].mean())
                per_pair[f"{key}_other_slots"] = float(arr[:, ~mask].mean()) if (~mask).any() else 0.0
                per_pair[f"{key}_ratio"]       = (
                    per_pair[f"{key}_name_slots"] / (per_pair[f"{key}_other_slots"] + 1e-12)
                )

        # top-K slots ranked by mean-over-layers angle (where divergence concentrates)
        K_slot_angle = mK["angle_deg"].mean(axis=0)   # (T,)
        V_slot_angle = mV["angle_deg"].mean(axis=0)
        topK_idx = np.argsort(-K_slot_angle)[: args.top_k].tolist()
        topV_idx = np.argsort(-V_slot_angle)[: args.top_k].tolist()
        per_pair["top_slots_K_angle"] = [
            {"slot": int(s), "angle_deg": float(K_slot_angle[s])} for s in topK_idx
        ]
        per_pair["top_slots_V_angle"] = [
            {"slot": int(s), "angle_deg": float(V_slot_angle[s])} for s in topV_idx
        ]

        summary["pairs"][tag] = per_pair

        print(f"{tag:<22} {per_pair['K_angle_mean']:>15.3f} "
              f"{per_pair['V_angle_mean']:>15.3f} "
              f"{per_pair['K_l2_shift_mean']:>11.4f} "
              f"{per_pair['V_l2_shift_mean']:>11.4f}")
        if slots_set:
            print(f"    K angle  name-slots={per_pair['K_angle_deg_name_slots']:.3f}  "
                  f"other={per_pair['K_angle_deg_other_slots']:.3f}  "
                  f"ratio={per_pair['K_angle_deg_ratio']:.2f}")
            print(f"    V angle  name-slots={per_pair['V_angle_deg_name_slots']:.3f}  "
                  f"other={per_pair['V_angle_deg_other_slots']:.3f}  "
                  f"ratio={per_pair['V_angle_deg_ratio']:.2f}")

        # ── heatmaps ───────────────────────────────────────────────────────
        if not args.no_plots:
            for label, m in [("K", mK), ("V", mV)]:
                heatmap(m["cos"],
                        title=f"INIT {a} vs {b}  {label} cos (head-mean)",
                        out_path=out_dir / f"init_{a}_{b}_{label}_cos.png",
                        highlight=slots_set, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
                heatmap(m["angle_deg"],
                        title=f"INIT {a} vs {b}  {label} angle deg (head-mean)",
                        out_path=out_dir / f"init_{a}_{b}_{label}_angle.png",
                        highlight=slots_set, cmap="viridis", vmin=0.0)
                heatmap(m["l2_shift"],
                        title=f"INIT {a} vs {b}  {label} |A-B|  (head-mean)",
                        out_path=out_dir / f"init_{a}_{b}_{label}_l2.png",
                        highlight=slots_set, cmap="viridis", vmin=0.0)
                heatmap(m["norm_ratio"],
                        title=f"INIT {a} vs {b}  {label} |A|/|B|  (head-mean)",
                        out_path=out_dir / f"init_{a}_{b}_{label}_normratio.png",
                        highlight=slots_set, cmap="RdBu_r", vmin=0.5, vmax=1.5)

    (out_dir / "init_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsummary → {out_dir / 'init_summary.json'}")
    if not args.no_plots:
        print(f"heatmaps → {out_dir}/")


if __name__ == "__main__":
    main()
