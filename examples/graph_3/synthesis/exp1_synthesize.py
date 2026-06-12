"""
Exp 1 — adaptive-hint self-study.  PLAN.md §4, Alex's rule (09.06 13:39):
use the model's own correct unaided answers; add the hint ONLY where it errs.

Two phases over the train questions:
  1. NO-HINT pass (greedy, corpus-only context — identical to exp2's pass 0).
     Correct traces enter the dataset as-is (source="no_hint").
  2. HINT pass for the questions the model got wrong: the system prompt
     additionally contains the worked push/pop BFS scratchpad for this exact
     pair (generated deterministically by GraphIndex.scratchpad). The
     regenerated trace enters the dataset regardless of correctness
     (source="with_hint"; residual errors are recorded, not filtered —
     filtering is Exp 3's job).

Every train question therefore contributes EXACTLY ONE trace — the dataset is
size-matched with exp2 (--samples-per-question 1), removing graph_2's
"method × data volume" confound (REPORT §3).

adaptive_report.json: per bucket — how many questions needed the hint and how
many the hint actually fixed (the survival-style honesty report, PLAN §9).

Requires a running Tokasaurus server:
  $CARTRIDGES_TOKASAURUS_URL, $HANDSHAKE_SERVER_MODEL

Output:
  {outputs_graph3}/exp1_adaptive/artifact/dataset.parquet
  {outputs_graph3}/exp1_adaptive/adaptive_report.json

Usage:
    python -m examples.graph_3.synthesis.exp1_synthesize
    python -m examples.graph_3.synthesis.exp1_synthesize --limit 32   # smoke
    python -m examples.graph_3.synthesis.exp1_synthesize --dry-run
"""
from __future__ import annotations
import argparse
import asyncio
import random
from collections import defaultdict
from pathlib import Path

from examples.graph_3 import paths
from examples.graph_3.data_gen.graph_index import GraphIndex
from examples.graph_3.synthesis.common import (
    BATCH_SIZE, MAX_NEW_TOKENS, STEPBYSTEP_DIRECTIVE,
    artifact_parquet, batched, check_logprobs, load_train_meta, make_convo,
    report_path, save_report, system_graph_prompt,
)

_HINT_TMPL = (
    "{base}\n\n"
    "A worked breadth-first search for this exact question:\n"
    "{scratchpad}\n\n"
    "Use it to answer the question correctly, showing the search the same way."
)

# Stronger instruction for the stepbystep variant: force the model to REPRODUCE
# the whole search rather than copy the answer from the scratchpad tail.
_HINT_TMPL_STEPBYSTEP = (
    "{base}\n\n"
    "A worked breadth-first search for this exact question:\n"
    "{scratchpad}\n\n"
    "Re-run this search yourself, step by step. In your reply you MUST reproduce "
    "every queue pop and push IN ORDER, exactly as shown above, narrating the "
    "frontier as it grows. Do NOT skip ahead, and do NOT state the path or the "
    "final number until you have walked the entire search to the target. Only "
    'after the full trace, end with the "path:" line and the "Answer:" line.'
)


def hint_prompt(
    base_prompt: str, index: GraphIndex, m: dict, seed: int, stepbystep: bool = False
) -> str:
    """System prompt with the corpus AND the worked scratchpad for this pair.

    The rng is seeded per-question so reruns are reproducible while
    within-level neighbor order still varies across questions. With
    ``stepbystep`` the instruction forces full reproduction of the search.
    """
    rng = random.Random(f"{seed}:{m['x']}:{m['y']}")
    scratchpad = index.scratchpad(m["x"], m["y"], rng)
    tmpl = _HINT_TMPL_STEPBYSTEP if stepbystep else _HINT_TMPL
    return tmpl.format(base=base_prompt, scratchpad=scratchpad)


async def run_exp1(
    meta_list: list[dict],
    base_prompt: str,
    index: GraphIndex,
    client,
    *,
    batch_size: int,
    max_new_tokens: int,
    seed: int,
    enable_thinking: bool,
    output_dir: Path,
    stepbystep: bool = False,
):
    from cartridges.structs import write_conversations
    from examples.graph_3.evaluation.eval import extract_answer

    # In stepbystep mode the no-hint phase is also asked to narrate the full
    # search (the hint phase uses the stronger reproduce-the-scratchpad template).
    phase1_prompt = base_prompt + STEPBYSTEP_DIRECTIVE if stepbystep else base_prompt

    kept = []
    report: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "no_hint_correct": 0, "hint_fixed": 0, "still_wrong": 0}
    )
    for m in meta_list:
        report[str(m["n_bucket"])]["total"] += 1

    # ── Phase 1: no hint (greedy) ─────────────────────────────────────────────
    print(f"Phase 1 (no hint, greedy): {len(meta_list)} questions")
    need_hint: list[dict] = []
    for batch_meta in batched(meta_list, batch_size):
        chats = [
            [
                {"role": "system", "content": phase1_prompt},
                {"role": "user",   "content": m["question"]},
            ]
            for m in batch_meta
        ]
        resp = await client.chat(
            chats,
            max_completion_tokens=max_new_tokens,
            temperature=0.0,
            top_logprobs=20,
            enable_thinking=enable_thinking,
        )
        for m, sample in zip(batch_meta, resp.samples):
            if extract_answer(sample.text) == m["answer"]:
                report[str(m["n_bucket"])]["no_hint_correct"] += 1
                kept.append(make_convo(m, sample, source="no_hint",
                                       correct=True, temp=0.0))
            else:
                need_hint.append(m)

    print(f"Phase 1 done: {len(kept)} correct without hint, "
          f"{len(need_hint)} go to the hint phase")

    # ── Phase 2: regenerate the failures with the scratchpad hint ─────────────
    print(f"Phase 2 (with scratchpad hint, greedy): {len(need_hint)} questions")
    for batch_meta in batched(need_hint, batch_size):
        chats = [
            [
                {"role": "system", "content": hint_prompt(base_prompt, index, m, seed, stepbystep)},
                {"role": "user",   "content": m["question"]},
            ]
            for m in batch_meta
        ]
        resp = await client.chat(
            chats,
            max_completion_tokens=max_new_tokens,
            temperature=0.0,
            top_logprobs=20,
            enable_thinking=enable_thinking,
        )
        for m, sample in zip(batch_meta, resp.samples):
            correct = extract_answer(sample.text) == m["answer"]
            b = report[str(m["n_bucket"])]
            b["hint_fixed" if correct else "still_wrong"] += 1
            # Kept regardless of correctness — exp1 adds the hint, not a filter.
            kept.append(make_convo(m, sample, source="with_hint",
                                   correct=correct, temp=0.0))

    n_fixed = sum(b["hint_fixed"] for b in report.values())
    n_bad   = sum(b["still_wrong"] for b in report.values())
    print(f"Done. {len(kept)} traces total "
          f"(= {len(meta_list)} questions): "
          f"{len(kept) - len(need_hint)} no-hint correct, "
          f"{n_fixed} fixed by hint, {n_bad} still wrong (kept, flagged).")
    check_logprobs(kept)

    parquet_path = artifact_parquet(output_dir, "exp1_adaptive", stepbystep)
    write_conversations(kept, str(parquet_path))
    print(f"Saved → {parquet_path}")

    report_out = {}
    for bucket, c in sorted(report.items(), key=lambda x: (x[0] == "none", x[0])):
        report_out[bucket] = {
            **c,
            "no_hint_rate":    round(c["no_hint_correct"] / max(1, c["total"]), 3),
            "still_wrong_rate": round(c["still_wrong"] / max(1, c["total"]), 3),
        }
    save_report(report_out, report_path(output_dir, "exp1_adaptive", "adaptive_report", stepbystep))
    print("\nPer bucket: correct without hint / fixed by hint / still wrong (of total):")
    for bucket, s in report_out.items():
        print(f"  n={bucket}: {s['no_hint_correct']}/{s['hint_fixed']}/{s['still_wrong']} "
              f"(of {s['total']})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-meta", type=str, default=str(paths.BASE_TRAIN_META))
    ap.add_argument("--corpus",     type=str, default=str(paths.CORPUS_TXT))
    ap.add_argument("--forest",     type=str, default=str(paths.FOREST_JSON))
    ap.add_argument("--output-dir", type=str, default=str(paths.OUTPUTS_DIR))
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--seed",       type=int, default=42, help="Scratchpad rng seed")
    ap.add_argument("--limit",      type=int, default=None, help="Smoke test cap")
    ap.add_argument("--thinking",   action="store_true",
                    help="Enable Qwen3 <think> mode for the synthesis traces")
    ap.add_argument("--dry-run",    action="store_true",
                    help="Print one assembled hint prompt and exit (no server needed)")
    ap.add_argument("--stepbystep", action="store_true",
                    help="Force full step-by-step search reproduction; write to "
                         "dataset_stepbystep.parquet (does not overwrite the default)")
    args = ap.parse_args()

    meta_list = load_train_meta(args.train_meta, args.limit)
    base_prompt = system_graph_prompt(Path(args.corpus).read_text())
    index = GraphIndex.load(args.forest)
    print(f"Questions: {len(meta_list)}  ·  base prompt: {len(base_prompt)} chars")

    if args.dry_run:
        m = next(mm for mm in meta_list if mm["n_bucket"] not in ("none", "1"))
        full = hint_prompt(base_prompt, index, m, args.seed, args.stepbystep)
        print(f"\n--- hint system prompt (stepbystep={args.stepbystep}, corpus truncated) ---")
        head, _, tail = full.partition("A worked breadth-first search")
        print(head[:300] + " ...\n")
        print("A worked breadth-first search" + tail)
        print("\n--- user ---")
        print(m["question"])
        print(f"\ngold: {m['answer']}")
        return

    from examples.graph_3.synthesis.common import make_client

    client = make_client()

    asyncio.run(run_exp1(
        meta_list=meta_list,
        base_prompt=base_prompt,
        index=index,
        client=client,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        enable_thinking=args.thinking,
        output_dir=Path(args.output_dir),
        stepbystep=args.stepbystep,
    ))


if __name__ == "__main__":
    main()
