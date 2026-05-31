"""
Shared library for comparing kinship cartridges (KV caches).

One place for every primitive the comparison CLI needs, so compare.py stays a
thin orchestrator. Used by both:
  * init compare  (build caches from corpora, no training)   — exp1
  * trained compare (load cache_last.pt checkpoints)          — exp2 / exp4

A "cartridge" here is reduced to a per-(layer, slot) DIRECTION by averaging
over heads:  K[layer] (1, H, T, D)  →  (T, D).  Stacking layers gives a
(L, T, D) tensor we compare slot-by-slot.

Metrics (see METHODOLOGY.md):
  cos        directional agreement in [-1, 1]
  angle_deg  acos(cos), 0° = identical direction
  l2_shift   |A - B|              (absolute movement)
  rel_l2     |A - B| / |A|        (movement relative to magnitude)
  norm_ratio |A| / |B|            (magnitude inflation/deflation)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


# ── Loading ──────────────────────────────────────────────────────────────────
def load_trained_cache(pt_path: Path) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Load a trained cartridge checkpoint → (keys, values), each a list of
    per-layer (1, H, T, D) float CPU tensors."""
    from cartridges.cache import TrainableCache

    cache = TrainableCache.from_pretrained(str(pt_path))
    keys = [t.detach().float().cpu() for t in cache.trainable_keys]
    values = [t.detach().float().cpu() for t in cache.trainable_values]
    return keys, values


def build_init_cache(
    model, tokenizer, corpus_path: Path, max_tokens: int | None = None
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Build an *untrained* cartridge by running a corpus through KVFromText."""
    from cartridges.cache import AttnConfig
    from cartridges.initialization import KVFromText

    cfg = KVFromText.Config(text_source=str(corpus_path), max_tokens=max_tokens)
    factory = cfg.instantiate()
    attn = AttnConfig(
        n_layers=model.config.num_hidden_layers,
        n_heads=model.config.num_key_value_heads,
        head_dim=getattr(
            model.config, "head_dim",
            model.config.hidden_size // model.config.num_attention_heads,
        ),
    )
    cache = factory.initialize_kv_cache(tokenizer, model, attn)
    keys = [t.detach().float().cpu() for t in cache.trainable_keys]
    values = [t.detach().float().cpu() for t in cache.trainable_values]
    return keys, values


def find_cache_last(run_dir: Path) -> Path:
    """Find the (most recent) cache_last.pt anywhere under run_dir.

    Pydrantic nests each run under <timestamp>-<uuid>/, so the checkpoint is
    usually a couple levels down.
    """
    hits = list(Path(run_dir).rglob("cache_last.pt"))
    if not hits:
        raise FileNotFoundError(f"no cache_last.pt under {run_dir}")
    hits.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return hits[0]


# ── Direction extraction ─────────────────────────────────────────────────────
def head_mean_direction(layer_tensor: torch.Tensor) -> torch.Tensor:
    """(1, H, T, D) → (T, D) by averaging over heads."""
    return layer_tensor.squeeze(0).mean(dim=0)


def stack_directions(layers: list[torch.Tensor]) -> torch.Tensor:
    """[(1, H, T, D)] across layers → (L, T, D) head-mean directions."""
    return torch.stack([head_mean_direction(x) for x in layers], dim=0)


# ── Metrics ──────────────────────────────────────────────────────────────────
def directional_metrics(A: torch.Tensor, B: torch.Tensor) -> dict[str, np.ndarray]:
    """A, B: (L, T, D). Returns dict of (L, T) numpy arrays."""
    eps = 1e-12
    dot = (A * B).sum(dim=-1)                 # (L, T)
    nA = A.norm(dim=-1).clamp_min(eps)
    nB = B.norm(dim=-1).clamp_min(eps)
    cos = (dot / (nA * nB)).clamp(-1.0, 1.0)
    angle = torch.rad2deg(torch.acos(cos))
    l2_shift = (A - B).norm(dim=-1)
    return {
        "cos": cos.numpy(),
        "angle_deg": angle.numpy(),
        "l2_shift": l2_shift.numpy(),
        "rel_l2": (l2_shift / nA).numpy(),
        "norm_ratio": (nA / nB).numpy(),
    }


def pair_summary(m_K: dict, m_V: dict) -> dict:
    """Collapse per-(layer, slot) metric arrays to scalars for one pair."""
    out = {}
    for tag, m in (("K", m_K), ("V", m_V)):
        out[f"{tag}_cos_mean"] = float(m["cos"].mean())
        out[f"{tag}_angle_mean"] = float(m["angle_deg"].mean())
        out[f"{tag}_rel_l2_mean"] = float(m["rel_l2"].mean())
        out[f"{tag}_l2_shift_mean"] = float(m["l2_shift"].mean())
        out[f"{tag}_norm_ratio_med"] = float(np.median(m["norm_ratio"]))
    return out


def slot_localization(m_K: dict, m_V: dict, name_slots: list[int], n_tokens: int,
                      top_k: int = 30) -> dict:
    """Where on the cartridge does divergence concentrate?

    Returns per-slot mean-over-layers angle, the top-K slots, and (if name
    slots are given) the name-slot-vs-other ratio — the signal that tells you
    a swapped entity is localized to specific cartridge positions.
    """
    out: dict = {}
    for tag, m in (("K", m_K), ("V", m_V)):
        slot_angle = m["angle_deg"].mean(axis=0)        # (T,)
        top_idx = np.argsort(-slot_angle)[:top_k].tolist()
        out[f"{tag}_top_slots"] = [
            {"slot": int(s), "angle_deg": float(slot_angle[s])} for s in top_idx
        ]
        if name_slots:
            mask = np.zeros(n_tokens, dtype=bool)
            mask[name_slots] = True
            inside = float(m["angle_deg"][:, mask].mean())
            outside = float(m["angle_deg"][:, ~mask].mean()) if (~mask).any() else 0.0
            out[f"{tag}_name_slot_angle"] = inside
            out[f"{tag}_other_slot_angle"] = outside
            out[f"{tag}_name_slot_ratio"] = inside / (outside + 1e-12)
    return out


# ── Name-token slot detection ────────────────────────────────────────────────
def find_name_slots(tokenizer, corpus_path: Path, name: str) -> list[int]:
    """Slot indices in the (tokenized) corpus where `name` appears as a single
    token. Used to mark/localize the swapped entity."""
    text = Path(corpus_path).read_text()
    ids = tokenizer.encode(text, add_special_tokens=False)
    name_id_sets = [
        tokenizer.encode(c, add_special_tokens=False) for c in (name, " " + name)
    ]
    positions = []
    for i, tid in enumerate(ids):
        for nid in name_id_sets:
            if len(nid) == 1 and tid == nid[0]:
                positions.append(i)
    return positions


# ── Plotting ─────────────────────────────────────────────────────────────────
def heatmap(arr: np.ndarray, *, title: str, out_path: Path,
            highlight: list[int] = (), cmap: str = "viridis",
            vmin=None, vmax=None) -> None:
    """Render a (layer × slot) heatmap; red vlines mark highlighted slots."""
    L, T = arr.shape
    fig, ax = plt.subplots(figsize=(max(8, T * 0.05), max(4, L * 0.2)))
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    for s in highlight:
        ax.axvline(s, color="red", linewidth=0.6, alpha=0.7)
    ax.set_xlabel("slot (cartridge position)")
    ax.set_ylabel("layer")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
