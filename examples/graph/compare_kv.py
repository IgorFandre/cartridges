"""
Compare trained KV caches from `graph_train_variants.py` runs.

Loads cache_last.pt from each variant checkpoint dir, computes pairwise
per-layer distance metrics (cosine similarity, L2 norm, max abs diff) for
both keys and values, plus delta-from-init.

Usage:
    python examples/graph/compare_kv.py
    python examples/graph/compare_kv.py --ckpt-root path/to/checkpoints_variants
"""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_cache(pt_path: Path) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    ckpt = torch.load(str(pt_path), map_location="cpu", weights_only=False)
    keys   = [t.detach().float() for t in ckpt["trainable_keys"]]
    values = [t.detach().float() for t in ckpt["trainable_values"]]
    return keys, values


def find_cache_last(variant_dir: Path) -> Path:
    candidates = list(variant_dir.rglob("cache_last.pt"))
    if not candidates:
        raise FileNotFoundError(f"no cache_last.pt under {variant_dir}")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def pair_metrics(a: torch.Tensor, b: torch.Tensor) -> dict:
    af = a.flatten()
    bf = b.flatten()
    diff = af - bf
    cos = torch.nn.functional.cosine_similarity(af, bf, dim=0).item()
    l2  = diff.norm().item()
    max_abs = diff.abs().max().item()
    rel = l2 / (af.norm().item() + 1e-12)
    return {"cos": cos, "l2": l2, "max_abs": max_abs, "rel_l2": rel}


def per_slot_metrics(a: torch.Tensor, b: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """
    a, b: (1, n_heads, n_tokens, head_dim).
    Pool heads + head_dim into a single vector per slot, then compute
    cosine sim and relative-L2 per slot. Returns (cos[T], rel_l2[T]).
    """
    # flatten heads & head_dim per slot → (n_tokens, n_heads * head_dim)
    a2 = a.squeeze(0).permute(1, 0, 2).reshape(a.shape[2], -1)
    b2 = b.squeeze(0).permute(1, 0, 2).reshape(b.shape[2], -1)
    cos = torch.nn.functional.cosine_similarity(a2, b2, dim=-1).cpu().numpy()
    diff = (a2 - b2).norm(dim=-1)
    norm_a = a2.norm(dim=-1).clamp_min(1e-12)
    rel = (diff / norm_a).cpu().numpy()
    return cos, rel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", type=str,
                   default=str(Path(__file__).parent / "checkpoints_variants"))
    p.add_argument("--subdir", type=str, default="cot",
                   help="Subdir under ckpt-root: 'cot' or 'ntp'")
    p.add_argument("--variants", type=str, default=None,
                   help="Comma-separated subset of variants (override meta).")
    p.add_argument("--variants-root", type=str,
                   default=str(Path(__file__).parent / "variants"))
    p.add_argument("--out-dir", type=str,
                   default=str(Path(__file__).parent / "kv_compare_out"))
    p.add_argument("--per-slot", action="store_true",
                   help="Also produce per-(layer, slot) heatmaps and stats.")
    p.add_argument("--name-positions", type=str, default=None,
                   help="JSON list of slot indices that hold the swapped name in the init corpus. "
                        "If given, prints stats inside vs outside these slots.")
    args = p.parse_args()

    ckpt_root = Path(args.ckpt_root)
    variants_meta = Path(args.variants_root) / "variants_meta.json"
    if not variants_meta.exists():
        raise FileNotFoundError(f"{variants_meta} missing")
    meta = json.loads(variants_meta.read_text())
    if args.variants:
        variants = [v.strip().lower() for v in args.variants.split(",") if v.strip()]
    else:
        variants = [v.lower() for v in meta["variants"]]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"variants: {variants}")
    print(f"loading caches from {ckpt_root}")

    caches: dict[str, tuple[list[torch.Tensor], list[torch.Tensor]]] = {}
    for v in variants:
        # prefer ckpt_root/<subdir>/<variant>, fall back to ckpt_root/<variant>
        vdir = ckpt_root / args.subdir / v
        if not vdir.exists():
            vdir = ckpt_root / v
        pt = find_cache_last(vdir)
        caches[v] = load_cache(pt)
        K, V = caches[v]
        print(f"  {v}: {pt}  layers={len(K)}  K[0]={tuple(K[0].shape)}")

    # sanity: all same n_layers / shape
    n_layers = len(caches[variants[0]][0])
    for v in variants:
        K, V = caches[v]
        assert len(K) == len(V) == n_layers
        assert K[0].shape == caches[variants[0]][0][0].shape

    pairs = list(itertools.combinations(variants, 2))
    metric_names = ["cos", "l2", "max_abs", "rel_l2"]
    # results[kv]["cos"] shape (n_pairs, n_layers)
    results: dict[str, dict[str, np.ndarray]] = {
        "K": {m: np.zeros((len(pairs), n_layers)) for m in metric_names},
        "V": {m: np.zeros((len(pairs), n_layers)) for m in metric_names},
    }

    for pi, (a, b) in enumerate(pairs):
        Ka, Va = caches[a]
        Kb, Vb = caches[b]
        for li in range(n_layers):
            mk = pair_metrics(Ka[li], Kb[li])
            mv = pair_metrics(Va[li], Vb[li])
            for m in metric_names:
                results["K"][m][pi, li] = mk[m]
                results["V"][m][pi, li] = mv[m]

    # mean across layers per pair
    print("\n=== pairwise (mean over layers) ===")
    print(f"{'pair':<20} {'K.cos':>8} {'K.relL2':>9} {'V.cos':>8} {'V.relL2':>9}")
    rows = []
    for pi, (a, b) in enumerate(pairs):
        kcos = results["K"]["cos"][pi].mean()
        krel = results["K"]["rel_l2"][pi].mean()
        vcos = results["V"]["cos"][pi].mean()
        vrel = results["V"]["rel_l2"][pi].mean()
        label = f"{a} vs {b}"
        print(f"{label:<20} {kcos:>8.4f} {krel:>9.4f} {vcos:>8.4f} {vrel:>9.4f}")
        rows.append({
            "a": a, "b": b,
            "K_cos_mean": float(kcos), "K_rel_l2_mean": float(krel),
            "V_cos_mean": float(vcos), "V_rel_l2_mean": float(vrel),
        })

    summary = {
        "variants": variants,
        "pairs": [{"a": a, "b": b} for a, b in pairs],
        "n_layers": n_layers,
        "per_pair_mean": rows,
        "per_layer": {
            kv: {m: results[kv][m].tolist() for m in metric_names}
            for kv in ("K", "V")
        },
    }
    (out_dir / "compare_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nsaved → {out_dir / 'compare_summary.json'}")

    # ── per-slot × layer analysis ──────────────────────────────────────────
    if args.per_slot:
        n_tokens = caches[variants[0]][0][0].shape[2]
        print(f"\n=== per-slot × layer  (n_tokens={n_tokens}) ===")
        name_pos = None
        if args.name_positions:
            name_pos = json.loads(args.name_positions)
            print(f"name slots: {name_pos}")

        per_slot_summary: dict[str, dict] = {}
        for pi, (a, b) in enumerate(pairs):
            Ka, Va = caches[a]
            Kb, Vb = caches[b]
            # (n_layers, n_tokens) for cos and rel_l2, K and V
            K_cos = np.zeros((n_layers, n_tokens))
            K_rel = np.zeros((n_layers, n_tokens))
            V_cos = np.zeros((n_layers, n_tokens))
            V_rel = np.zeros((n_layers, n_tokens))
            for li in range(n_layers):
                kc, kr = per_slot_metrics(Ka[li], Kb[li])
                vc, vr = per_slot_metrics(Va[li], Vb[li])
                K_cos[li] = kc; K_rel[li] = kr
                V_cos[li] = vc; V_rel[li] = vr

            per_slot_summary[f"{a}|{b}"] = {
                "K_cos_mean": float(K_cos.mean()),
                "K_rel_l2_mean": float(K_rel.mean()),
                "V_cos_mean": float(V_cos.mean()),
                "V_rel_l2_mean": float(V_rel.mean()),
            }

            if name_pos is not None:
                mask = np.zeros(n_tokens, dtype=bool)
                mask[name_pos] = True
                for tag, arr in [("V_rel", V_rel), ("K_rel", K_rel)]:
                    inside = arr[:, mask].mean() if mask.any() else 0.0
                    outside = arr[:, ~mask].mean() if (~mask).any() else 0.0
                    ratio = inside / (outside + 1e-12)
                    print(f"  {a}|{b}  {tag}: name_slots={inside:.4f}  other={outside:.4f}  ratio={ratio:.2f}")

            # heatmaps per-pair: layer × slot
            for kv_tag, arr, m in [
                ("V_cos", V_cos, "cos"), ("V_rel_l2", V_rel, "rel_l2"),
                ("K_cos", K_cos, "cos"), ("K_rel_l2", K_rel, "rel_l2"),
            ]:
                fig, ax = plt.subplots(figsize=(max(8, n_tokens * 0.04), max(4, n_layers * 0.2)))
                im = ax.imshow(arr, aspect="auto",
                               cmap=("viridis" if m == "rel_l2" else "RdBu_r"),
                               vmin=(0.0 if m == "rel_l2" else None),
                               vmax=(None if m == "rel_l2" else 1.0))
                if name_pos is not None:
                    for s in name_pos:
                        ax.axvline(s, color="red", linewidth=0.5, alpha=0.5)
                ax.set_xlabel("slot (cartridge position)")
                ax.set_ylabel("layer")
                ax.set_title(f"{a} vs {b}  {kv_tag}")
                fig.colorbar(im, ax=ax)
                fig.tight_layout()
                out = out_dir / f"slot_{a}_{b}_{kv_tag}.png"
                fig.savefig(out, dpi=120)
                plt.close(fig)
                print(f"  per-slot heatmap → {out}")

        (out_dir / "per_slot_summary.json").write_text(
            json.dumps(per_slot_summary, indent=2)
        )

    # heatmaps: pairs × layers
    pair_labels = [f"{a}|{b}" for a, b in pairs]
    for kv in ("K", "V"):
        for m in ("cos", "rel_l2"):
            arr = results[kv][m]
            fig, ax = plt.subplots(figsize=(max(8, n_layers * 0.4), max(3, len(pairs) * 0.5)))
            im = ax.imshow(arr, aspect="auto",
                           cmap=("viridis" if m == "rel_l2" else "RdBu_r"),
                           vmin=(0.0 if m == "rel_l2" else None),
                           vmax=(None if m == "rel_l2" else 1.0))
            ax.set_yticks(range(len(pairs)))
            ax.set_yticklabels(pair_labels)
            ax.set_xticks(range(n_layers))
            ax.set_xlabel("layer")
            ax.set_title(f"{kv} {m}")
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            out = out_dir / f"heatmap_{kv}_{m}.png"
            fig.savefig(out, dpi=120)
            plt.close(fig)
            print(f"  heatmap → {out}")


if __name__ == "__main__":
    main()
