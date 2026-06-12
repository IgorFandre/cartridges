"""
Exp 2 — plain self-study (no hint, no filter).  PLAN.md §4.

For every train question: context = friendship corpus only (the same system
prompt as the ICL eval), the model answers free-form, and the trace enters the
training set UNCONDITIONALLY — wrong answers included.  This isolates "what
does distilling the model's own unaided behavior into a cartridge give"
without the selection bias that sank graph_2's Exp 2 (rejection sampling
collapsed negatives to 1.8% and deep hops to single digits).

The first sample per question is greedy; extra samples (--samples-per-question
> 1) are drawn at --extra-temp for trace diversity (SCoTD: diverse rationales
are the key distillation knob for <2B students).

Correctness per trace is recorded in metadata (and a per-bucket report) — it is
the train-set ICL accuracy proxy, NOT a filter.  Exp 3 (rejection sampling) =
this dataset filtered by `correct`, so it can be built later without re-running
the server.

Requires a running Tokasaurus server:
  $CARTRIDGES_TOKASAURUS_URL, $HANDSHAKE_SERVER_MODEL

Output:
  {outputs_graph3}/exp2_plain/artifact/dataset.parquet
  {outputs_graph3}/exp2_plain/collection_report.json

Usage:
    python -m examples.graph_3.synthesis.exp2_synthesize
    python -m examples.graph_3.synthesis.exp2_synthesize --samples-per-question 2
    python -m examples.graph_3.synthesis.exp2_synthesize --dry-run
"""
from __future__ import annotations
import argparse
import asyncio
from collections import defaultdict
from pathlib import Path

from examples.graph_3 import paths
from examples.graph_3.synthesis.common import (
    BATCH_SIZE, MAX_NEW_TOKENS, STEPBYSTEP_DIRECTIVE,
    artifact_parquet, batched, check_logprobs, load_train_meta, make_convo,
    report_path, save_report, system_graph_prompt,
)


async def run_exp2(
    meta_list: list[dict],
    system_prompt: str,
    client,
    *,
    samples_per_question: int,
    extra_temp: float,
    batch_size: int,
    max_new_tokens: int,
    enable_thinking: bool,
    output_dir: Path,
    stepbystep: bool = False,
):
    from cartridges.structs import write_conversations
    from examples.graph_3.evaluation.eval import extract_answer

    # system_prompt already carries STEPBYSTEP_DIRECTIVE when stepbystep is set
    # (appended by main); `stepbystep` here only selects the output filename.
    prompt = system_prompt

    kept = []
    report: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "correct": 0})

    # Pass 0 = greedy; passes 1.. = sampled at extra_temp for diversity.
    for pass_idx in range(samples_per_question):
        temp = 0.0 if pass_idx == 0 else extra_temp
        print(f"Pass {pass_idx + 1}/{samples_per_question} "
              f"(temp={temp}): {len(meta_list)} questions")
        for batch_meta in batched(meta_list, batch_size):
            chats = [
                [
                    {"role": "system", "content": prompt},
                    {"role": "user",   "content": m["question"]},
                ]
                for m in batch_meta
            ]
            resp = await client.chat(
                chats,
                max_completion_tokens=max_new_tokens,
                temperature=temp,
                top_logprobs=20,
                enable_thinking=enable_thinking,
            )
            for m, sample in zip(batch_meta, resp.samples):
                correct = extract_answer(sample.text) == m["answer"]
                # No filter: every trace enters the dataset (exp2 = plain self-study).
                kept.append(make_convo(m, sample, source="no_hint",
                                       correct=correct, temp=temp))
                b = report[str(m["n_bucket"])]
                b["total"] += 1
                b["correct"] += int(correct)

    n_correct = sum(b["correct"] for b in report.values())
    n_total   = sum(b["total"] for b in report.values())
    print(f"Done. Kept ALL {len(kept)} traces "
          f"({n_correct}/{n_total} = {n_correct / max(1, n_total):.1%} correct — recorded, not filtered).")
    check_logprobs(kept)

    parquet_path = artifact_parquet(output_dir, "exp2_plain", stepbystep)
    write_conversations(kept, str(parquet_path))
    print(f"Saved → {parquet_path}")

    report_out = {
        bucket: {**counts, "correct_rate": round(counts["correct"] / max(1, counts["total"]), 3)}
        for bucket, counts in sorted(report.items(), key=lambda x: (x[0] == "none", x[0]))
    }
    save_report(report_out, report_path(output_dir, "exp2_plain", "collection_report", stepbystep))
    print("\nTrain-set correctness per bucket (proxy ICL accuracy, NOT a filter):")
    for bucket, s in report_out.items():
        print(f"  n={bucket}: {s['correct']}/{s['total']} = {s['correct_rate']:.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-meta", type=str, default=str(paths.BASE_TRAIN_META))
    ap.add_argument("--corpus",     type=str, default=str(paths.CORPUS_TXT))
    ap.add_argument("--output-dir", type=str, default=str(paths.OUTPUTS_DIR))
    ap.add_argument("--samples-per-question", type=int, default=1)
    ap.add_argument("--extra-temp", type=float, default=0.7,
                    help="Temperature for passes beyond the first (greedy) one")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--limit",      type=int, default=None, help="Smoke test cap")
    ap.add_argument("--thinking",   action="store_true",
                    help="Enable Qwen3 <think> mode for the synthesis traces")
    ap.add_argument("--dry-run",    action="store_true",
                    help="Print one assembled prompt and exit (no server needed)")
    ap.add_argument("--stepbystep", action="store_true",
                    help="Ask for full step-by-step search; write to "
                         "dataset_stepbystep.parquet (does not overwrite the default)")
    args = ap.parse_args()

    meta_list = load_train_meta(args.train_meta, args.limit)
    system_prompt = system_graph_prompt(Path(args.corpus).read_text())
    if args.stepbystep:
        system_prompt += STEPBYSTEP_DIRECTIVE
    print(f"Questions: {len(meta_list)}  ·  system prompt: {len(system_prompt)} chars"
          f"  ·  stepbystep={args.stepbystep}")

    if args.dry_run:
        m = meta_list[0]
        print("\n--- system (truncated) ---")
        print(system_prompt[:400] + " ...")
        print("\n--- user ---")
        print(m["question"])
        print(f"\ngold: {m['answer']}")
        return

    from examples.graph_3.synthesis.common import make_client

    client = make_client()

    asyncio.run(run_exp2(
        meta_list=meta_list,
        system_prompt=system_prompt,
        client=client,
        samples_per_question=args.samples_per_question,
        extra_temp=args.extra_temp,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.thinking,
        output_dir=Path(args.output_dir),
        stepbystep=args.stepbystep,
    ))


if __name__ == "__main__":
    main()
