"""
Exp 2 — self-study via rejection sampling.

The questions are sampled (X, Y, n, direction) pairs with controlled per-n-hop
coverage (read from train_meta.json, produced by lineage_qagen).  Only the GRAPH
is in context — no BFS-path hint (that's Exp 1).  The model reasons freely and we
keep its trace ONLY when it actually reaches the correct answer; the resulting
self-study dataset is the model's own correct reasoning, distilled into the
cartridge.

For each sampled question:
  1. Attempt 0 (greedy): the model answers with free CoT.
  2. Extract Yes/No; if it matches the gold label, keep the trace (with top-k
     logprobs) as a training Conversation.
  3. If wrong, retry (temperature rising) up to `--attempts-1` more times.  A
     trace enters the dataset as soon as the model gets the answer right.

No accuracy is reported here — the honest ICL baseline lives in Exp 0
(evaluation/lineage_eval.py --mode icl).  We only report the per-n-hop
collection yield (how many sampled questions produced a correct trace).

Requires a running Tokasaurus server.  Set:
  $CARTRIDGES_TOKASAURUS_URL and $LINEAGE_SERVER_MODEL

Output:
  {CARTRIDGES_OUTPUT_DIR_GRAPH2}/exp2_star/artifact/dataset.parquet
  {CARTRIDGES_OUTPUT_DIR_GRAPH2}/exp2_star/star_survival.json   (per-n-hop yield)

Usage:
    python -m examples.graph_2.synthesis.star_synthesize              # 3 attempts (2 retries)
    ATTEMPTS=2 python -m examples.graph_2.synthesis.star_synthesize   # 1 retry
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterator

from cartridges.clients.tokasaurus import TokasaurusClient
from cartridges.structs import Conversation, read_conversations, write_conversations
from examples.graph_2 import paths
from examples.graph_2.evaluation.lineage_eval import extract_yes_no


# ── Constants/env ─────────────────────────────────────────────────────────────
SERVER_URL   = os.environ.get("CARTRIDGES_TOKASAURUS_URL", "http://localhost:8000")
SERVER_MODEL = os.environ.get("LINEAGE_SERVER_MODEL",      "Qwen/Qwen3-1.7B")
# Total attempts per question = 1 initial + retries.  Default 3 ⇒ up to 2 retries.
ATTEMPTS     = int(os.environ.get("ATTEMPTS", os.environ.get("K_RETRIES", "3")))
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", "32"))
OUTPUT_DIR   = os.environ.get("CARTRIDGES_OUTPUT_DIR_GRAPH2", str(paths.OUTPUTS_DIR))
# Attempt 0 = greedy (temp 0) — this is the HONEST ICL answer that scores the
# reported accuracy.  Retries sample with rising temperature to recover more
# training data (these do NOT count toward the accuracy).
TEMPERATURES = [0.0, 0.7, 0.9, 1.0]


# Same instruction as evaluation/lineage_eval (graph in context, free CoT → Yes/No).
_SYSTEM_TMPL = (
    "Use the family tree below to answer the lineage question. "
    "Reason step by step, then end with exactly 'Yes.' or 'No.'  "
    "Do not output anything after the final Yes/No."
    "\n\nFamily tree:\n{corpus}"
)


# ── Batching helper ───────────────────────────────────────────────────────────
def _batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


# ── Conversation builder ──────────────────────────────────────────────────────
def _make_convo(meta: dict, sample, attempt: int, temp: float) -> Conversation:
    flat = None
    if sample.top_logprobs is not None:
        flat = sample.top_logprobs.flatten(threshold=0.99)

    return Conversation(
        system_prompt="",  # graph lives in the cartridge; empty here for training
        type="lineage_star",
        metadata={
            "question_text":  meta["question_text"],
            "label":          meta["label"],
            "true_distance":  meta["true_distance"],
            "claimed_n":      meta["claimed_n"],
            "n_bucket":       meta["n_bucket"],
            "direction":      meta["direction"],
            "x": meta.get("x"), "y": meta.get("y"),
            "attempt":        attempt,
            "temp":           temp,
        },
        messages=[
            Conversation.Message(
                role="user",
                content=meta["question_text"],
                token_ids=None,
                top_logprobs=None,
            ),
            Conversation.Message(
                role="assistant",
                content=sample.text,
                token_ids=sample.token_ids,
                top_logprobs=flat,
            ),
        ],
    )


# ── Main async loop ───────────────────────────────────────────────────────────
async def run_star(
    meta_list: list[dict],
    corpus_text: str,
    client: TokasaurusClient,
    attempts: int,
    batch_size: int,
    output_dir: Path,
):
    system_prompt = _SYSTEM_TMPL.format(corpus=corpus_text)

    pending   = list(meta_list)
    kept:     list[Conversation] = []
    # Per n-hop collection yield: how many sampled questions produced a correct
    # self-study trace that entered the training set (after up to `attempts`).
    survival: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "kept": 0})

    for m in pending:
        survival[str(m["n_bucket"])]["total"] += 1

    for attempt in range(attempts):
        if not pending:
            break
        temp = TEMPERATURES[min(attempt, len(TEMPERATURES) - 1)]
        tag = "greedy" if attempt == 0 else f"retry temp={temp}"
        print(f"Attempt {attempt+1}/{attempts} [{tag}]: {len(pending)} pending")

        next_pending: list[dict] = []
        for batch_meta in _batched(pending, batch_size):
            chats = [
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": m["question_text"]},
                ]
                for m in batch_meta
            ]
            resp = await client.chat(
                chats,
                max_completion_tokens=1024,
                temperature=temp,
                top_logprobs=20,
                enable_thinking=True,   # free-form CoT reasoning
            )
            for m, sample in zip(batch_meta, resp.samples):
                pred = extract_yes_no(sample.text)
                correct = pred is not None and pred == m["label"]
                # Self-study rejection sampling: keep the trace iff the model
                # actually reached the correct answer (on this or a prior retry).
                if correct:
                    survival[str(m["n_bucket"])]["kept"] += 1
                    kept.append(_make_convo(m, sample, attempt, temp))
                else:
                    next_pending.append(m)

        pending = next_pending

    print(f"Done.  Kept {len(kept)} / {len(meta_list)} "
          f"({len(kept)/max(1,len(meta_list)):.1%}) correct self-study traces.")

    # Verify logprobs present for KL training
    n_no_logprobs = sum(1 for c in kept if c.messages[-1].top_logprobs is None)
    if n_no_logprobs:
        print(
            f"WARNING: {n_no_logprobs}/{len(kept)} kept conversations have "
            "no top_logprobs.  Check server configuration.  "
            "Set targets='tokens' in train config if server doesn't emit logprobs."
        )

    artifact_dir = output_dir / "exp2_star" / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = artifact_dir / "dataset.parquet"
    write_conversations(kept, str(parquet_path))
    print(f"Saved → {parquet_path}")

    # Survival report
    survival_path = output_dir / "exp2_star" / "star_survival.json"
    survival_dict = {}
    for bucket, counts in sorted(survival.items(), key=lambda x: str(x[0])):
        total = counts["total"]
        kept_n = counts["kept"]
        survival_dict[bucket] = {
            "total": total, "kept": kept_n,
            "survival_rate": round(kept_n / max(1, total), 3),
        }
    survival_path.write_text(json.dumps(survival_dict, indent=2))
    print(f"Yield report → {survival_path}")
    print("\nCollection yield per n_bucket (correct traces / sampled questions):")
    for bucket, s in survival_dict.items():
        print(f"  n_bucket={bucket}: {s['kept']}/{s['total']} = {s['survival_rate']:.1%}")

    return parquet_path


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-meta", type=str, default=str(paths.BASE_TRAIN_META))
    ap.add_argument("--corpus",     type=str, default=str(paths.BASE_CORPUS))
    ap.add_argument("--output-dir", type=str, default=OUTPUT_DIR)
    ap.add_argument("--attempts",   type=int, default=ATTEMPTS,
                    help="Total attempts per question = 1 initial + retries (default 3)")
    ap.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    ap.add_argument("--limit",      type=int, default=None,
                    help="Limit number of questions (smoke test)")
    args = ap.parse_args()

    meta_path = Path(args.train_meta)
    if not meta_path.exists():
        raise FileNotFoundError(f"Train meta not found: {meta_path}.  Run lineage_qagen.py first.")

    meta_list: list[dict] = json.loads(meta_path.read_text())
    if args.limit:
        meta_list = meta_list[: args.limit]

    # train_meta.json records use key "question" and may carry an int n_bucket;
    # normalise to the names the loop expects (question_text, str n_bucket).
    for m in meta_list:
        if "question_text" not in m:
            m["question_text"] = m.get("question", "")
        m["n_bucket"] = str(m.get("n_bucket", "none"))

    print(f"Questions to attempt: {len(meta_list)}")

    corpus_text = Path(args.corpus).read_text()

    client = TokasaurusClient.Config(
        url=SERVER_URL,
        model_name=SERVER_MODEL,
    ).instantiate()

    asyncio.run(run_star(
        meta_list=meta_list,
        corpus_text=corpus_text,
        client=client,
        attempts=args.attempts,
        batch_size=args.batch_size,
        output_dir=Path(args.output_dir),
    ))


if __name__ == "__main__":
    main()
