"""
Fabricate a tiny self-study dataset from the handshake QA train set — NO SERVER.

Builds (user=question, assistant="<think>brief reasoning</think>\\n\\nAnswer: N")
conversations from the ground-truth labels.  No top-k logprobs are produced, so
the resulting parquet must be trained with TARGETS=tokens (plain CE), not the
KL-distillation path.

This exists ONLY to smoke-test the training + generation-eval wiring without a
Tokasaurus server.  The real Exp-1 / Exp-2 datasets come from
exp1_synthesize.py / exp2_synthesize.py.

Usage:
    python -m examples.graph_3.synthesis.make_smoke_dataset
    python -m examples.graph_3.synthesis.make_smoke_dataset --limit 64 \\
        --train-meta examples/graph_3/data/base/train_meta.json \\
        --out outputs_graph3/smoke/artifact/dataset.parquet
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from cartridges.structs import Conversation, write_conversations
from examples.graph_3 import paths


def _reasoning_for(rec: dict) -> str:
    """Brief gold-grounded reasoning ending with the required Answer: line."""
    x, y = rec.get("x"), rec.get("y")
    answer = rec["answer"]
    path = rec.get("path")

    if answer == "not connected":
        body = f"{x} and {y} are in different friend groups with no shared connections."
    elif path:
        chain = " → ".join(path)
        body = f"Tracing the shortest path: {chain}. That is {answer} step(s)."
    else:
        body = f"The shortest path between {x} and {y} is {answer} step(s)."

    return (
        f"<think>{body}</think>\n\n"
        f"path: {' - '.join(path) if path else 'n/a'}\n"
        f"Answer: {answer}"
    )


def make_smoke_conversation(rec: dict) -> Conversation:
    return Conversation(
        system_prompt="",
        type="handshake_smoke",
        metadata={
            "question":      rec["question"],
            "answer":        rec["answer"],
            "true_distance": rec["true_distance"],
            "n_bucket":      str(rec["n_bucket"]),
            "x": rec.get("x"), "y": rec.get("y"),
        },
        messages=[
            Conversation.Message(
                role="user", content=rec["question"],
                token_ids=None, top_logprobs=None,
            ),
            Conversation.Message(
                role="assistant", content=_reasoning_for(rec),
                token_ids=None, top_logprobs=None,
            ),
        ],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-meta", type=str, default=str(paths.BASE_TRAIN_META))
    ap.add_argument("--out",        type=str,
                    default=str(paths.OUTPUTS_DIR / "smoke" / "artifact" / "dataset.parquet"))
    ap.add_argument("--limit",      type=int, default=128,
                    help="Number of QA records to fabricate (default 128)")
    args = ap.parse_args()

    meta_path = Path(args.train_meta)
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Train meta not found: {meta_path}.  Run qagen.py first."
        )

    recs = json.loads(meta_path.read_text())
    if args.limit:
        recs = recs[: args.limit]

    convos = [make_smoke_conversation(r) for r in recs]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_conversations(convos, str(out))
    print(f"Fabricated {len(convos)} smoke conversations → {out}")
    print("Train with: TARGETS=tokens TRAIN_PARQUET=%s ..." % out)


if __name__ == "__main__":
    main()
