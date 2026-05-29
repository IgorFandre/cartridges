"""
Compare 5 alex stability runs pairwise to measure training noise.

Loads cache_last.pt from each of alex_run{1..5}/, computes pairwise
directional diff (head-mean, same as compare_init_kv) per (layer, slot),
and reports the variance budget.

Outputs:
  - stability_summary.json — pairwise + mean stats
  - stability_heatmap_K_angle.png, ..._V_angle.png — mean angle over all 10 pairs

Usage:
    python examples/graph/compare_alex_stability.py
    python examples/graph/compare_alex_stability.py --ckpt-root outputs_graph/exp2_stability
    python examples/graph/compare_alex_stability.py --no-plots
"""
import argparse
import itertools
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from cartridges.cache import TrainableCache


def load_cache(pt_path: Path) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    cache = TrainableCache.from_pretrained(str(pt_path))
    keys   = [t.detach().float().cpu() for t in cache.trainable_keys]
    values = [t.detach().float().cpu() for t in cache.trainable_values]
    return keys, values


def find_cache_last(run_dir: Path) -> Path:
    hits = list(run_dir.rglob("cache_last.pt"))
    if not hits:
        raise FileNotFoundError(f"no cache_last.pt under {run_dir}")
    return hits[0]


def head_mean_direction(layer_tensor: torch.Tensor) -> torch.Tensor:
    """(1, H, T, D) -> (T, D)."""
    return layer_tensor.squeeze(0).mean(dim=0)


def stack_directions(layers: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack([head_mean_direction(x) for x in layers], dim=0)


def directional_metrics(A: torch.Tensor, B: torch.Tensor):
    eps = 1e-12
    dot = (A * B).sum(dim=-1)
    nA = A.norm(dim=-1).clamp_min(eps)
    nB = B.norm(dim=-1).clamp_min(eps)
    cos = (dot / (nA * nB)).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cos))
    l2_shift = (A - B).norm(dim=-1)
    rel_l2 = (l2_shift / nA).numpy()
    return {
        "cos": cos.numpy(),
        "angle_deg": angle.numpy(),
        "l2_shift": l2_shift.numpy(),
        "rel_l2": rel_l2,
        "norm_a": nA.numpy(),
        "norm_b": nB.numpy(),
        "norm_ratio": (nA / nB).numpy(),
    }


def heatmap(arr, *, title: str, out_path: Path, cmap="viridis", vmin=None, vmax=None):
    L, T = arr.shape
    fig, ax = plt.subplots(figsize=(max(6, T * 0.12), max(4, L * 0.2)))
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xlabel("slot")
    ax.set_ylabel("layer")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt-root", type=str,
                   default=str(Path(__file__).parent / "checkpoints_stability"))
    p.add_argument("--n-runs", type=int, default=5)
    p.add_argument("--out-dir", type=str,
                   default=str(Path(__file__).parent / "stability_compare_out"))
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    ckpt_root = Path(args.ckpt_root)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    runs = [f"alex_run{i+1}" for i in range(args.n_runs)]

    print(f"loading {len(runs)} caches from {ckpt_root}")
    K_dirs: dict[str, torch.Tensor] = {}
    V_dirs: dict[str, torch.Tensor] = {}
    for r in runs:
        run_dir = ckpt_root / r
        if not run_dir.exists():
            print(f"  {r}: MISSING — skip")
            continue
        pt = find_cache_last(run_dir)
        K, V = load_cache(pt)
        K_dirs[r] = stack_directions(K)
        V_dirs[r] = stack_directions(V)
        L, T, D = K_dirs[r].shape
        print(f"  {r}: L={L} T={T} D={D}  ({pt})")

    present = list(K_dirs.keys())
    if len(present) < 2:
        print("need at least 2 trained runs — abort"); return

    pairs = list(itertools.combinations(present, 2))
    n_layers, n_tokens, _ = K_dirs[present[0]].shape

    summary = {
        "runs":     present,
        "n_runs":   len(present),
        "n_layers": n_layers,
        "n_tokens": n_tokens,
        "pairs":    {},
    }

    print(f"\n=== pairwise stability ({len(pairs)} pairs, lower angle = more stable) ===")
    print(f"{'pair':<22} {'K angle°':>10} {'V angle°':>10} "
          f"{'K rel_l2':>10} {'V rel_l2':>10} {'K cos':>8} {'V cos':>8}")

    # accumulate per-pair metric arrays to compute mean across pairs
    all_K_angle = []
    all_V_angle = []

    for a, b in pairs:
        mK = directional_metrics(K_dirs[a], K_dirs[b])
        mV = directional_metrics(V_dirs[a], V_dirs[b])
        tag = f"{a}|{b}"
        summary["pairs"][tag] = {
            "K_angle_mean":    float(mK["angle_deg"].mean()),
            "V_angle_mean":    float(mV["angle_deg"].mean()),
            "K_cos_mean":      float(mK["cos"].mean()),
            "V_cos_mean":      float(mV["cos"].mean()),
            "K_rel_l2_mean":   float(mK["rel_l2"].mean()),
            "V_rel_l2_mean":   float(mV["rel_l2"].mean()),
            "K_norm_ratio_med": float(np.median(mK["norm_ratio"])),
            "V_norm_ratio_med": float(np.median(mV["norm_ratio"])),
        }
        all_K_angle.append(mK["angle_deg"])
        all_V_angle.append(mV["angle_deg"])
        s = summary["pairs"][tag]
        print(f"{tag:<22} {s['K_angle_mean']:>10.3f} {s['V_angle_mean']:>10.3f} "
              f"{s['K_rel_l2_mean']:>10.4f} {s['V_rel_l2_mean']:>10.4f} "
              f"{s['K_cos_mean']:>8.4f} {s['V_cos_mean']:>8.4f}")

    # aggregate over all pairs
    K_angle_all = np.stack(all_K_angle, axis=0)   # (n_pairs, L, T)
    V_angle_all = np.stack(all_V_angle, axis=0)
    summary["overall"] = {
        "K_angle_mean":  float(K_angle_all.mean()),
        "K_angle_std":   float(K_angle_all.mean(axis=(1,2)).std()),
        "V_angle_mean":  float(V_angle_all.mean()),
        "V_angle_std":   float(V_angle_all.mean(axis=(1,2)).std()),
    }
    print(f"\n=== overall (across {len(pairs)} pairs) ===")
    print(f"K angle: mean={summary['overall']['K_angle_mean']:.3f}°  "
          f"std across pairs={summary['overall']['K_angle_std']:.3f}°")
    print(f"V angle: mean={summary['overall']['V_angle_mean']:.3f}°  "
          f"std across pairs={summary['overall']['V_angle_std']:.3f}°")

    if not args.no_plots:
        K_mean_map = K_angle_all.mean(axis=0)   # (L, T)
        V_mean_map = V_angle_all.mean(axis=0)
        heatmap(K_mean_map, title=f"alex stability — K mean angle° ({len(pairs)} pairs)",
                out_path=out_dir / "stability_K_angle.png", cmap="viridis", vmin=0)
        heatmap(V_mean_map, title=f"alex stability — V mean angle° ({len(pairs)} pairs)",
                out_path=out_dir / "stability_V_angle.png", cmap="viridis", vmin=0)
        print(f"heatmaps → {out_dir}")

    (out_dir / "stability_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"summary → {out_dir / 'stability_summary.json'}")


if __name__ == "__main__":
    main()
