"""
Quick per-hop sample eval — dump RAW cartridge generations (no parsing, no
scoring) for a SHARED, fixed set of test questions, into one JSON per cartridge.
Each record also carries the gold answer + gold path pulled from the test set,
so you can eyeball whether the generation is right by hand.

Built for "I won't have time for the full eval before the call" — it samples
`--per-hop` questions from EACH hop bucket (1..8 + `none`) with a FIXED seed, so:
  • the same questions are asked of every cartridge in a single run, AND
  • the same questions are asked across SEPARATE runs of this script
    (you can eval cartridges in batches and the JSONs stay comparable).

The base Qwen3 model is loaded ONCE and reused; only the TrainableCache is
swapped per cartridge. Thinking is OFF by default (PLAN.md §3 — the push/pop
scratchpad IS the visible reasoning); pass --thinking to enable Qwen3 <think>.

Usage:
    # Two cartridges on GPU 0 (default: 2 questions/hop = 18 questions):
    python -m examples.graph_3.evaluation.quick_sample_eval --cuda 0 \\
        outputs_graph3/exp1_adaptive/train/.../cache_last.pt \\
        outputs_graph3/exp2_plain/train/.../cache_last.pt

    # Later, a second batch on the SAME questions (seed is fixed):
    python -m examples.graph_3.evaluation.quick_sample_eval --cuda 1 \\
        outputs_graph3/exp3_rejection/train/.../cache-step79.pt

A checkpoint arg may be a .pt file OR a directory (then cache_last.pt under it).

Output (in --out-dir, default outputs_graph3/quick_eval/):
    <label>.json              {cartridge, checkpoint, ..., generations: [...]}
                              each generation has the raw model text + gold
                              answer/path from the dataset (NO parsing/scoring).
    selected_questions.json   the chosen questions (written every run; identical).
"""
from __future__ import annotations

# NOTE: argparse + CUDA_VISIBLE_DEVICES must be handled BEFORE importing torch,
# so heavy imports are deferred into functions below.
import argparse
import json
import os
import random
import re
from collections import defaultdict
from pathlib import Path


SEED_DEFAULT = 42
PER_HOP_DEFAULT = 2


# ── Shared question selection (deterministic across runs) ──────────────────────
def select_questions(meta: list[dict], per_hop: int, seed: int) -> list[dict]:
    """Pick `per_hop` questions from each hop bucket with a fixed RNG.

    Buckets are visited in a stable order and rows are sorted by question text
    before sampling, so the selection depends ONLY on (meta file, per_hop, seed)
    — never on dict/file ordering. Same args → same questions, every run.
    """
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for m in meta:
        by_bucket[str(m["n_bucket"])].append(m)

    rng = random.Random(seed)
    chosen: list[dict] = []
    for bucket in sorted(by_bucket, key=lambda b: (b == "none", b)):
        rows = sorted(by_bucket[bucket], key=lambda r: r["question"])
        k = min(per_hop, len(rows))
        chosen.extend(rng.sample(rows, k))
    return chosen


def cartridge_label(ckpt_path: Path) -> str:
    """Short comparable label like `exp3_rejection-step79` from a checkpoint path."""
    parts = ckpt_path.parts
    exp = next((s for s in parts if s.startswith("exp")), None)
    m = re.search(r"cache-step(\d+)\.pt$", ckpt_path.name)
    if m:
        step = f"step{m.group(1)}"
    elif "last" in ckpt_path.name:
        step = "last"
    else:
        step = ckpt_path.stem
    base = exp or ckpt_path.parent.name
    return f"{base}-{step}"


def resolve_checkpoint(raw: str) -> Path:
    """Accept a .pt file or a directory (→ its cache_last.pt)."""
    p = Path(raw)
    if p.is_dir():
        from examples.graph_3 import paths
        return paths.latest_checkpoint(p)
    return p


# ── Per-cartridge generation (raw, no scoring) ────────────────────────────────
def generate_cartridge(ckpt: Path, tokenizer, model, device: str,
                       questions: list[dict], args) -> list[dict]:
    """Generate the shared questions for one cartridge; return raw records.

    No answer extraction / scoring — each record just holds the raw model text
    plus the gold answer + gold path copied straight from the test meta.
    """
    import torch
    from cartridges.cache import TrainableCache
    from cartridges.generation import flex_generate
    from examples.graph.evaluation.eval import build_inputs

    cache: TrainableCache = TrainableCache.from_pretrained(str(ckpt), device=device).to(device)

    records: list[dict] = []
    for i in range(0, len(questions), args.batch_size):
        batch_meta = questions[i : i + args.batch_size]
        qs = [m["question"] for m in batch_meta]

        input_ids, seq_ids, position_ids = build_inputs(
            qs, tokenizer, system_prompt=None, device=device,
            enable_thinking=args.thinking,
        )
        with torch.no_grad():
            pred_ids = flex_generate(
                model=model, tokenizer=tokenizer,
                input_ids=input_ids, seq_ids=seq_ids, position_ids=position_ids,
                cache=cache, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, show_progress=False,
            )
        for m, ids in zip(batch_meta, pred_ids):
            generation = tokenizer.decode(ids, skip_special_tokens=True)
            records.append({
                "n_bucket":      str(m["n_bucket"]),
                "true_distance": m.get("true_distance"),
                "x":             m.get("x"),
                "y":             m.get("y"),
                "question":      m["question"],
                "gold_answer":   m.get("answer"),     # "3" | "not connected"
                "gold_path":     m.get("path"),       # list[str] | None
                "generation":    generation,          # RAW model output
            })

    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return records


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("checkpoints", nargs="+",
                    help="One or more cartridge .pt files (or dirs → cache_last.pt).")
    ap.add_argument("--cuda", default=None,
                    help="CUDA_VISIBLE_DEVICES value (e.g. '0' or '0,1'). "
                         "Set here so it applies before torch initializes.")
    ap.add_argument("--per-hop", type=int, default=PER_HOP_DEFAULT,
                    help="Questions sampled per hop bucket (default 2 → 18 total).")
    ap.add_argument("--seed", type=int, default=SEED_DEFAULT,
                    help="Fixed sampling seed — keep constant across runs to "
                         "compare cartridges on the SAME questions.")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B",
                    help="Base model (must match what the cartridge was trained on).")
    ap.add_argument("--test-meta", default=None, help="Override test_meta.json path.")
    ap.add_argument("--out-dir", default=None,
                    help="Where JSONs go (default: outputs_graph3/quick_eval).")
    ap.add_argument("--max-new-tokens", type=int, default=4096,
                    help="Push/pop scratchpads reach ~2.7k tokens — keep headroom.")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--thinking", action="store_true",
                    help="Enable Qwen3 <think> (off by default, PLAN.md §3).")
    args = ap.parse_args()

    # Must precede any torch import.
    if args.cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    import torch
    from examples.graph_3 import paths
    from transformers import AutoTokenizer
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Shared question set (fixed across runs) ──
    test_meta = Path(args.test_meta) if args.test_meta else paths.BASE_TEST_META
    if not test_meta.exists():
        raise FileNotFoundError(
            f"test meta not found: {test_meta}\n"
            "Run: python -m examples.graph_3.data_gen.generate_forest && "
            "python -m examples.graph_3.data_gen.qagen"
        )
    meta = json.loads(test_meta.read_text())
    questions = select_questions(meta, args.per_hop, args.seed)

    out_dir = Path(args.out_dir) if args.out_dir else (paths.OUTPUTS_DIR / "quick_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Record the shared selection so cross-run JSONs can be cross-checked.
    (out_dir / "selected_questions.json").write_text(
        json.dumps(
            [{"n_bucket": str(m["n_bucket"]), "question": m["question"],
              "gold_answer": m.get("answer"), "gold_path": m.get("path")}
             for m in questions],
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"Device: {device}  (CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')})")
    print(f"Selected {len(questions)} questions "
          f"({args.per_hop}/hop, seed={args.seed}) → {out_dir}/selected_questions.json")

    # ── Load base model ONCE, swap cartridge per checkpoint ──
    print(f"Loading base model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = FlexQwen3ForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    for raw in args.checkpoints:
        ckpt = resolve_checkpoint(raw)
        if not ckpt.exists():
            print(f"SKIP (not found): {raw}")
            continue
        label = cartridge_label(ckpt)
        print(f"\n>>> Generating: {label}  ({ckpt})")
        records = generate_cartridge(ckpt, tokenizer, model, device, questions, args)

        payload = {
            "cartridge":      label,
            "checkpoint":     str(ckpt),
            "model":          args.model,
            "thinking":       args.thinking,
            "max_new_tokens": args.max_new_tokens,
            "temperature":    args.temperature,
            "per_hop":        args.per_hop,
            "seed":           args.seed,
            "generations":    records,
        }
        out_path = out_dir / f"{label}.json"
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        print(f"    {len(records)} generations  →  {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
