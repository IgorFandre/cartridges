"""
Training-dynamics analyzer: how do cartridge K/V vectors ROTATE during training?

Reads the `cache-step{N}.pt` checkpoint series that train.py already writes
(set a low SAVE_EVERY for dense sampling) and measures, with the SAME head-mean
direction + angle_deg reduction as compare.py, two rotations per checkpoint:

  incremental  angle(Z_t, Z_{t-1})  — where/when the cartridge is still moving
  cumulative   angle(Z_t, Z_0)      — how far it has turned from its start

Keys (K) and values (V) are tracked separately. This reproduces Paper 2's
finding shape (value rotations ≫ key rotations, and continue late into training)
on the kinship cartridges, in units directly comparable to the Exp1/2/4 noise
floor (degrees, head-mean direction per layer×slot).

Baseline Z_0:
  * default — the earliest checkpoint found.
  * --init-corpus PATH — build the TRUE untrained init via KVFromText (loads the
    model, exactly like compare.py --source init) and prepend it as step 0.

Optionally (--spectra) also dumps init-vs-final singular value spectra (Paper 2:
keys stay flat, values gain a decaying spectrum).

Outputs (under --out-dir):
  dynamics.json                 steps + per-step overall/per-layer K&V angles
  rotation_incremental.png      mean angle(Z_t,Z_{t-1}) vs step, K vs V
  rotation_cumulative.png       mean angle(Z_t,Z_0)     vs step, K vs V
  heatmap_{K,V}_incremental.png (layer × step) incremental angle
  heatmap_{K,V}_cumulative.png  (layer × step) cumulative angle
  spectra_{K,V}.png             (only with --spectra) init vs final spectra

Example:
  python -m examples.graph.comparison.dynamics \
      --run-dir outputs_graph/exp2_train/alex \
      --init-corpus examples/graph/data/variants/alex/family_tree_corpus.txt \
      --spectra --out-dir outputs_graph/exp2_train/alex/dynamics
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from examples.graph import paths
from examples.graph.comparison import kv_compare as kvc


# ── Checkpoint discovery ──────────────────────────────────────────────────────
def find_step_checkpoints(run_dir: Path) -> list[tuple[int, Path]]:
    """All cache-step{N}.pt under run_dir, sorted by step. Ignores cache_last.pt."""
    hits = paths.run_checkpoints(run_dir)
    if not hits:
        raise FileNotFoundError(
            f"no cache-step*.pt under {run_dir} — train with a low SAVE_EVERY "
            f"(e.g. SAVE_EVERY=50) so dynamics has enough checkpoints"
        )
    return hits


def build_init_directions(corpus: Path, model_name: str, device: str,
                          max_tokens: int | None):
    """True step-0 baseline: head-mean K/V directions from the untrained init."""
    from transformers import AutoTokenizer
    from cartridges.models import HFModelConfig, FlexQwen3ForCausalLM

    print(f"loading {model_name} on {device} for true-init baseline …")
    model = HFModelConfig(
        pretrained_model_name_or_path=model_name,
        model_cls=FlexQwen3ForCausalLM,
    ).instantiate()
    model.to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    K, V = kvc.build_init_cache(model, tokenizer, corpus, max_tokens)
    return kvc.stack_directions(K), kvc.stack_directions(V)


# ── Core ──────────────────────────────────────────────────────────────────────
def load_directions(pt: Path):
    K, V = kvc.load_trained_cache(pt)
    return kvc.stack_directions(K), kvc.stack_directions(V)


def angle_reductions(A: torch.Tensor, B: torch.Tensor) -> dict:
    """A,B: (L,T,D) head-mean directions → angle reductions for one transition."""
    angle = kvc.directional_metrics(A, B)["angle_deg"]   # (L, T)
    return {
        "overall": float(angle.mean()),
        "per_layer": angle.mean(axis=1),                 # (L,)
        "per_slot": angle.mean(axis=0),                  # (T,)
    }


def run(run_dir: Path, out_dir: Path, *, init_corpus: Path | None,
        model_name: str, device: str, max_tokens: int | None,
        spectra: bool, no_plots: bool):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpts = find_step_checkpoints(run_dir)
    print(f"found {len(ckpts)} checkpoints: steps {[s for s, _ in ckpts]}")

    # Build the ordered (step, K_dir, V_dir) sequence; optional true init at step 0.
    steps: list[int] = []
    K_seq: list[torch.Tensor] = []
    V_seq: list[torch.Tensor] = []
    K_layers_first = V_layers_first = None   # raw layer lists for spectra

    if init_corpus is not None:
        K0, V0 = build_init_directions(init_corpus, model_name, device, max_tokens)
        steps.append(0); K_seq.append(K0); V_seq.append(V0)

    for step, pt in ckpts:
        Kd, Vd = load_directions(pt)
        steps.append(step); K_seq.append(Kd); V_seq.append(Vd)
        print(f"  step {step}: {tuple(Kd.shape)}  ({pt})")

    n_layers, n_tokens, _ = K_seq[0].shape
    for d in (*K_seq, *V_seq):
        assert d.shape == (n_layers, n_tokens, d.shape[-1]), "checkpoint shapes differ"

    # ── incremental (t-1 → t) and cumulative (0 → t) ──────────────────────────
    inc_steps = steps[1:]
    inc_K_overall, inc_V_overall = [], []
    cum_K_overall, cum_V_overall = [], []
    inc_K_layer, inc_V_layer = [], []
    cum_K_layer, cum_V_layer = [], []

    for i in range(1, len(steps)):
        rk = angle_reductions(K_seq[i - 1], K_seq[i])
        rv = angle_reductions(V_seq[i - 1], V_seq[i])
        inc_K_overall.append(rk["overall"]); inc_V_overall.append(rv["overall"])
        inc_K_layer.append(rk["per_layer"]); inc_V_layer.append(rv["per_layer"])

        ck = angle_reductions(K_seq[0], K_seq[i])
        cv = angle_reductions(V_seq[0], V_seq[i])
        cum_K_overall.append(ck["overall"]); cum_V_overall.append(cv["overall"])
        cum_K_layer.append(ck["per_layer"]); cum_V_layer.append(cv["per_layer"])

    dynamics = {
        "run_dir": str(run_dir),
        "n_layers": n_layers, "n_tokens": n_tokens,
        "steps": steps,
        "baseline": "true-init (step 0)" if init_corpus else f"first checkpoint (step {steps[0]})",
        "incremental": {
            "steps": inc_steps,
            "K_angle_overall": inc_K_overall, "V_angle_overall": inc_V_overall,
        },
        "cumulative": {
            "steps": inc_steps,
            "K_angle_overall": cum_K_overall, "V_angle_overall": cum_V_overall,
        },
    }
    (out_dir / "dynamics.json").write_text(json.dumps(dynamics, indent=2))

    print("\n=== rotation per step (mean over layer×slot, degrees) ===")
    print(f"{'step':>8} {'ΔK inc':>8} {'ΔV inc':>8} {'K cum':>8} {'V cum':>8}")
    for j, s in enumerate(inc_steps):
        print(f"{s:>8} {inc_K_overall[j]:>8.3f} {inc_V_overall[j]:>8.3f} "
              f"{cum_K_overall[j]:>8.3f} {cum_V_overall[j]:>8.3f}")

    if not no_plots:
        kvc.rotation_curve(
            inc_steps, inc_K_overall, inc_V_overall,
            title="Incremental rotation  angle(Z_t, Z_{t-1})",
            ylabel="mean angle° (per step)",
            out_path=out_dir / "rotation_incremental.png")
        kvc.rotation_curve(
            inc_steps, cum_K_overall, cum_V_overall,
            title="Cumulative rotation  angle(Z_t, Z_0)",
            ylabel="mean angle° (from baseline)",
            out_path=out_dir / "rotation_cumulative.png")

        for tag, layer_series in (("K", inc_K_layer), ("V", inc_V_layer)):
            kvc.heatmap_xy(
                np.stack(layer_series, axis=1), xticklabels=inc_steps,
                xlabel="optimizer step", title=f"{tag} incremental angle° (layer × step)",
                out_path=out_dir / f"heatmap_{tag}_incremental.png", vmin=0)
        for tag, layer_series in (("K", cum_K_layer), ("V", cum_V_layer)):
            kvc.heatmap_xy(
                np.stack(layer_series, axis=1), xticklabels=inc_steps,
                xlabel="optimizer step", title=f"{tag} cumulative angle° (layer × step)",
                out_path=out_dir / f"heatmap_{tag}_cumulative.png", vmin=0)

    # ── optional spectra: init (or first) vs final ────────────────────────────
    if spectra:
        first_K, first_V = kvc.load_trained_cache(ckpts[0][1])
        last_K, last_V = kvc.load_trained_cache(ckpts[-1][1])
        sp_first_K = kvc.singular_value_spectra(first_K)
        sp_last_K = kvc.singular_value_spectra(last_K)
        sp_first_V = kvc.singular_value_spectra(first_V)
        sp_last_V = kvc.singular_value_spectra(last_V)
        spectra_json = {
            "K_participation_first": sp_first_K["participation"].tolist(),
            "K_participation_last": sp_last_K["participation"].tolist(),
            "V_participation_first": sp_first_V["participation"].tolist(),
            "V_participation_last": sp_last_V["participation"].tolist(),
        }
        (out_dir / "spectra.json").write_text(json.dumps(spectra_json, indent=2))
        if not no_plots:
            kvc.plot_spectra(
                {f"K @ step{ckpts[0][0]}": sp_first_K["mean_spectrum"],
                 f"K @ step{ckpts[-1][0]}": sp_last_K["mean_spectrum"]},
                title="Key singular value spectra (init vs final)",
                out_path=out_dir / "spectra_K.png")
            kvc.plot_spectra(
                {f"V @ step{ckpts[0][0]}": sp_first_V["mean_spectrum"],
                 f"V @ step{ckpts[-1][0]}": sp_last_V["mean_spectrum"]},
                title="Value singular value spectra (init vs final)",
                out_path=out_dir / "spectra_V.png")

    print(f"\nsaved → {out_dir}/")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True,
                   help="run directory holding cache-step*.pt (pydrantic nests under a timestamp subdir)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--init-corpus", default=None,
                   help="build the TRUE untrained init as step 0 (loads the model)")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--spectra", action="store_true",
                   help="also dump init-vs-final singular value spectra")
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args()

    run(Path(args.run_dir), Path(args.out_dir),
        init_corpus=Path(args.init_corpus) if args.init_corpus else None,
        model_name=args.model, device=args.device, max_tokens=args.max_tokens,
        spectra=args.spectra, no_plots=args.no_plots)


if __name__ == "__main__":
    main()
