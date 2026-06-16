"""
Quick per-hop sample eval — generate answers for a SHARED, fixed set of test
questions across one or more cartridges and dump the full reasoning + answers +
correctness to a .log per cartridge.

Built for "I won't have time for the full eval before the call" — it samples
`--per-hop` questions from EACH hop bucket (1..8 + `none`) with a FIXED seed, so:
  • the same questions are asked of every cartridge in a single run, AND
  • the same questions are asked across SEPARATE runs of this script
    (you can eval cartridges in batches and the logs stay comparable).

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

Logs land in --out-dir (default outputs_graph3/quick_eval/):
    <label>.log          full reasoning + per-question correctness + summary
    selected_questions.json   the 18 chosen questions (written every run; identical)
    _summary.log              appended cross-cartridge acc-per-hop table per run
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


# ── Per-cartridge evaluation ──────────────────────────────────────────────────
def eval_cartridge(ckpt: Path, tokenizer, model, device: str,
                   questions: list[dict], index, args) -> list[dict]:
    """Generate + score the shared questions for a single cartridge."""
    import torch
    from cartridges.cache import TrainableCache
    from cartridges.generation import flex_generate
    from examples.graph.evaluation.eval import build_inputs
    from examples.graph_3.evaluation.eval import score_row

    cache: TrainableCache = TrainableCache.from_pretrained(str(ckpt), device=device).to(device)

    rows: list[dict] = []
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
            pred_text = tokenizer.decode(ids, skip_special_tokens=True)
            rows.append(score_row(m, pred_text, index))

    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


# ── Log formatting ────────────────────────────────────────────────────────────
def _bucket_order(buckets) -> list[str]:
    return sorted(buckets, key=lambda b: (b == "none", b))


def write_cartridge_log(out_path: Path, label: str, ckpt: Path,
                        rows: list[dict], args) -> dict[str, tuple[int, int]]:
    """Write the full reasoning + correctness log; return per-bucket (ok, n)."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append(f"CARTRIDGE: {label}")
    lines.append(f"checkpoint: {ckpt}")
    lines.append(
        f"model: {args.model}  thinking={args.thinking}  temp={args.temperature}  "
        f"max_new_tokens={args.max_new_tokens}"
    )
    lines.append(
        f"questions: {len(rows)} ({args.per_hop} per hop, seed={args.seed})"
    )
    lines.append("=" * 80)
    lines.append("")

    per_bucket: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        b = str(r["n_bucket"])
        per_bucket[b].append(bool(r["correct"]))

        hop = "not-connected" if b == "none" else f"{b} hop(s)"
        lines.append(f"[hop {b}] ({hop})")
        lines.append(f"  Q: {r['question']}")
        lines.append(f"  gold_answer: {r['gold_answer']}   gold_path: {r['gold_path']}")
        lines.append(
            f"  predicted_answer: {r['predicted_answer']}   "
            f"correct: {r['correct']}   "
            f"path_valid: {r['path_valid']}   abs_err: {r['abs_err']}"
        )
        lines.append("  --- reasoning / scratchpad ---")
        # indent the raw model output for readability
        body = r["predicted"] or "(empty)"
        lines.extend("  " + ln for ln in body.splitlines() or ["(empty)"])
        lines.append("  --- end ---")
        lines.append("")

    # per-cartridge summary table
    total_ok = sum(r["correct"] for r in rows)
    lines.append("-" * 48)
    lines.append(f"SUMMARY  {label}")
    lines.append(f"{'bucket':>8} {'N':>4} {'acc':>7}")
    summary: dict[str, tuple[int, int]] = {}
    for b in _bucket_order(per_bucket):
        oks = per_bucket[b]
        ok, n = sum(oks), len(oks)
        summary[b] = (ok, n)
        lines.append(f"{b:>8} {n:>4} {f'{ok}/{n}':>7}")
    lines.append(f"{'OVERALL':>8} {len(rows):>4} {f'{total_ok}/{len(rows)}':>7}")
    lines.append("")

    out_path.write_text("\n".join(lines), encoding="utf-8")
    return summary


def print_and_append_run_summary(out_dir: Path, results: dict[str, dict],
                                 buckets: list[str], stamp: str):
    """Cross-cartridge acc-per-hop table → stdout + appended to _summary.log."""
    labels = list(results)
    header = f"{'hop':>6} " + " ".join(f"{lab[:16]:>16}" for lab in labels)
    rows_txt = [f"run {stamp}", header, "-" * len(header)]
    for b in buckets:
        cells = []
        for lab in labels:
            ok, n = results[lab].get(b, (0, 0))
            cells.append(f"{f'{ok}/{n}':>16}" if n else f"{'-':>16}")
        rows_txt.append(f"{b:>6} " + " ".join(cells))
    # overall row
    cells = []
    for lab in labels:
        ok = sum(v[0] for v in results[lab].values())
        n = sum(v[1] for v in results[lab].values())
        cells.append(f"{f'{ok}/{n}':>16}")
    rows_txt.append(f"{'TOTAL':>6} " + " ".join(cells))
    block = "\n".join(rows_txt)

    print("\n" + block + "\n")
    with (out_dir / "_summary.log").open("a", encoding="utf-8") as f:
        f.write(block + "\n\n")


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
    ap.add_argument("--forest", default=None, help="Override forest.json (path_valid).")
    ap.add_argument("--out-dir", default=None,
                    help="Where logs go (default: outputs_graph3/quick_eval).")
    ap.add_argument("--max-new-tokens", type=int, default=4096)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--thinking", action="store_true",
                    help="Enable Qwen3 <think> (off by default, PLAN.md §3).")
    args = ap.parse_args()

    # Must precede any torch import.
    if args.cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    import torch
    from datetime import datetime
    from examples.graph_3 import paths
    from examples.graph_3.data_gen.graph_index import GraphIndex
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

    forest_path = Path(args.forest) if args.forest else paths.FOREST_JSON
    index = GraphIndex.load(forest_path) if forest_path.exists() else None
    if index is None:
        print(f"WARNING: forest not found at {forest_path} — path_valid will be null.")

    out_dir = Path(args.out_dir) if args.out_dir else (paths.OUTPUTS_DIR / "quick_eval")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Record the shared selection so cross-run logs can be cross-checked.
    (out_dir / "selected_questions.json").write_text(
        json.dumps(
            [{"n_bucket": str(m["n_bucket"]), "question": m["question"],
              "answer": m["answer"]} for m in questions],
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

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    all_buckets = _bucket_order({str(m["n_bucket"]) for m in questions})
    run_summaries: dict[str, dict] = {}

    for raw in args.checkpoints:
        ckpt = resolve_checkpoint(raw)
        if not ckpt.exists():
            print(f"SKIP (not found): {raw}")
            continue
        label = cartridge_label(ckpt)
        print(f"\n>>> Eval cartridge: {label}  ({ckpt})")
        rows = eval_cartridge(ckpt, tokenizer, model, device, questions, index, args)

        log_path = out_dir / f"{label}.log"
        summary = write_cartridge_log(log_path, label, ckpt, rows, args)
        run_summaries[label] = summary

        ok = sum(r["correct"] for r in rows)
        print(f"    acc {ok}/{len(rows)}  →  {log_path}")

    if run_summaries:
        print_and_append_run_summary(out_dir, run_summaries, all_buckets, stamp)
    print("Done.")


if __name__ == "__main__":
    main()
