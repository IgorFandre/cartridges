"""
Fabricate a tiny self-study dataset from the lineage QA train set — NO SERVER.

Builds (user=question, assistant="<think>brief reasoning</think>\\n\\nYes."/"No.")
conversations from the ground-truth labels.  No top-k logprobs are produced, so
the resulting parquet must be trained with TARGETS=tokens (plain CE), not the
KL-distillation path.

This exists ONLY to smoke-test the training + generation-eval wiring without a
Tokasaurus server.  The real Exp-1 / Exp-2 datasets come from
lineage_synthesize.py / star_synthesize.py.

Usage:
    python -m examples.graph_2.synthesis.make_smoke_dataset
    python -m examples.graph_2.synthesis.make_smoke_dataset --limit 64 \\
        --train-meta examples/graph_2/data/base/train_meta.json \\
        --out outputs_graph2/smoke/artifact/dataset.parquet
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from cartridges.structs import Conversation, write_conversations
from examples.graph_2 import paths


def _reasoning_for(rec: dict) -> str:
    """A short, gold-grounded reasoning string ending with the answer."""
    label = rec["label"]
    direction = rec.get("direction", "ancestor")
    claimed_n = rec.get("claimed_n")
    true_d = rec.get("true_distance")
    x, y = rec.get("x"), rec.get("y")

    if true_d is None:
        body = f"{x} and {y} are not in a direct ancestor/descendant line."
    else:
        rel = "above" if direction == "ancestor" else "below"
        body = (
            f"{x} is {true_d} generation(s) {rel} {y}; "
            f"the question claims {claimed_n}."
        )
    return f"<think>{body}</think>\n\n{label}."


def make_smoke_conversation(rec: dict) -> Conversation:
    return Conversation(
        system_prompt="",
        type="lineage_smoke",
        metadata={
            "question_text": rec["question"],
            "label":          rec["label"],
            "true_distance":  rec["true_distance"],
            "claimed_n":      rec["claimed_n"],
            "n_bucket":       str(rec["n_bucket"]),
            "direction":      rec["direction"],
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
            f"Train meta not found: {meta_path}.  Run lineage_qagen.py first."
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
