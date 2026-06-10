"""
Open-ended handshake-distance QA generator (PLAN.md §2).

One question template everywhere (train and test):
    How many handshakes apart are {X} and {Y}? ...

Gold label: the unique distance (forest ⇒ unique path) or "not connected".

Split design (no leakage, even test buckets):
  - Split happens at the UNORDERED PAIR level, never at the question level,
    so a pair never appears in both train and test (in any order).
  - Hops 7..8 are TEST-ONLY (depth-generalization holdout, per Alex 09.06).
  - Test: --test-per-hop pairs per hop 1..8 + the same number of non-connected
    pairs; ONE randomly-ordered question per pair → even buckets.
  - Train: hops 1..6 from the remaining pairs, BOTH orders per pair, capped at
    --train-per-hop per bucket; plus ~--train-nonrel-frac "not connected".

Output: train_handshake.parquet / test_handshake.parquet (Conversation objects),
        train_meta.json / test_meta.json / split_meta.json.

Usage:
    python -m examples.graph_3.data_gen.qagen
    python -m examples.graph_3.data_gen.qagen --test-per-hop 100 --train-per-hop 300
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

from cartridges.structs import Conversation, write_conversations
from examples.graph_3.data_gen.graph_index import GraphIndex


# ── Question / gold templates ─────────────────────────────────────────────────
def make_question(x: str, y: str) -> str:
    return (
        f"How many handshakes apart are {x} and {y}? "
        "If they are not connected, say so. "
        f'End your reply with "path: {x} - ... - {y}" and "Answer: <number>", '
        'or "Answer: not connected".'
    )


def gold_answer_text(path: list[str] | None, distance: int | None) -> str:
    if distance is None:
        return "Answer: not connected"
    return "path: " + " - ".join(path) + f"\nAnswer: {distance}"


def make_record(index: GraphIndex, x: str, y: str) -> dict:
    """One ordered question record; gold derived from the index (never the model)."""
    d = index.distance(x, y)
    path = index.path(x, y)
    return dict(
        question=make_question(x, y),
        answer=str(d) if d is not None else "not connected",
        true_distance=d,
        n_bucket=str(d) if d is not None else "none",
        x=x, y=y,
        path=path,
    )


def record_to_conversation(rec: dict) -> Conversation:
    return Conversation(
        system_prompt="",
        type="handshake_qa",
        metadata={k: rec[k] for k in
                  ("question", "answer", "true_distance", "n_bucket", "x", "y", "path")},
        messages=[
            Conversation.Message(
                role="user", content=rec["question"],
                token_ids=None, top_logprobs=None,
            ),
            Conversation.Message(
                role="assistant",
                content=gold_answer_text(rec["path"], rec["true_distance"]),
                token_ids=None, top_logprobs=None,
            ),
        ],
    )


# ── Split logic ───────────────────────────────────────────────────────────────
def split_pairs(
    index: GraphIndex,
    rng: random.Random,
    *,
    max_hop: int,
    train_max_hop: int,
    test_per_hop: int,
    train_per_hop: int,
    train_nonrel_frac: float,
) -> tuple[list[dict], list[dict], dict]:
    by_d = index.pairs_by_distance(max_distance=max_hop)
    test_recs:  list[dict] = []
    train_recs: list[dict] = []
    report: dict = {}

    for d in range(1, max_hop + 1):
        pairs = list(by_d.get(d, []))
        rng.shuffle(pairs)
        if len(pairs) < test_per_hop:
            raise ValueError(
                f"hop {d}: only {len(pairs)} pairs < test_per_hop={test_per_hop}. "
                "Regenerate the forest with higher minimums."
            )
        test_pairs  = pairs[:test_per_hop]
        train_pairs = pairs[test_per_hop:] if d <= train_max_hop else []

        for (a, b) in test_pairs:
            x, y = (a, b) if rng.random() < 0.5 else (b, a)
            test_recs.append(make_record(index, x, y))

        train_qs = [q for (a, b) in train_pairs for q in
                    (make_record(index, a, b), make_record(index, b, a))]
        rng.shuffle(train_qs)
        train_qs = train_qs[:train_per_hop]
        train_recs.extend(train_qs)

        report[str(d)] = {
            "pairs": len(pairs),
            "test_questions": len(test_pairs),
            "train_questions": len(train_qs),
            "unused_pairs": max(0, len(pairs) - test_per_hop - len(train_pairs)),
        }

    # Non-connected negatives: disjoint pair sets for test and train
    n_train_pos = len(train_recs)
    n_train_neg = round(n_train_pos * train_nonrel_frac / (1 - train_nonrel_frac))
    nonrel = index.non_connected_pairs(test_per_hop + n_train_neg, rng)
    rng.shuffle(nonrel)
    if len(nonrel) < test_per_hop + n_train_neg:
        raise ValueError(
            f"Only {len(nonrel)} non-connected pairs available, "
            f"need {test_per_hop + n_train_neg}."
        )
    for (a, b) in nonrel[:test_per_hop]:
        x, y = (a, b) if rng.random() < 0.5 else (b, a)
        test_recs.append(make_record(index, x, y))
    for (a, b) in nonrel[test_per_hop : test_per_hop + n_train_neg]:
        x, y = (a, b) if rng.random() < 0.5 else (b, a)
        train_recs.append(make_record(index, x, y))

    report["none"] = {
        "pairs": len(nonrel),
        "test_questions": test_per_hop,
        "train_questions": n_train_neg,
    }

    rng.shuffle(train_recs)
    rng.shuffle(test_recs)
    return train_recs, test_recs, report


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    from examples.graph_3 import paths

    ap = argparse.ArgumentParser()
    ap.add_argument("--forest",  type=str, default=str(paths.FOREST_JSON))
    ap.add_argument("--out-dir", type=str, default=str(paths.BASE_DIR))
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--max-hop",       type=int, default=8)
    ap.add_argument("--train-max-hop", type=int, default=6,
                    help="Hops above this are test-only (generalization holdout)")
    ap.add_argument("--test-per-hop",  type=int, default=100,
                    help="Test questions per hop bucket (and for the none bucket)")
    ap.add_argument("--train-per-hop", type=int, default=300,
                    help="Cap on train questions per hop bucket")
    ap.add_argument("--train-nonrel-frac", type=float, default=0.12,
                    help="Share of 'not connected' questions in train")
    args = ap.parse_args()

    index = GraphIndex.load(args.forest)
    print(f"Forest: {len(index.people)} people, "
          f"{len(index.components)} components, max distance {index.max_distance()}")

    rng = random.Random(args.seed)
    train_recs, test_recs, report = split_pairs(
        index, rng,
        max_hop=args.max_hop,
        train_max_hop=args.train_max_hop,
        test_per_hop=args.test_per_hop,
        train_per_hop=args.train_per_hop,
        train_nonrel_frac=args.train_nonrel_frac,
    )

    print(f"\nTrain: {len(train_recs)}  Test: {len(test_recs)}")
    print(f"{'bucket':>8} {'pairs':>6} {'train_q':>8} {'test_q':>7}")
    for b, r in report.items():
        low = " ← <120" if (b not in ("7", "8", "none") and r["train_questions"] < 120) else ""
        print(f"{b:>8} {r['pairs']:>6} {r['train_questions']:>8} {r['test_questions']:>7}{low}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_conversations(
        [record_to_conversation(r) for r in train_recs],
        str(out_dir / "train_handshake.parquet"),
    )
    write_conversations(
        [record_to_conversation(r) for r in test_recs],
        str(out_dir / "test_handshake.parquet"),
    )
    (out_dir / "train_meta.json").write_text(json.dumps(train_recs, indent=2))
    (out_dir / "test_meta.json").write_text(json.dumps(test_recs, indent=2))
    (out_dir / "split_meta.json").write_text(json.dumps({
        "split_mode":        "unordered_pair",
        "seed":              args.seed,
        "max_hop":           args.max_hop,
        "train_max_hop":     args.train_max_hop,
        "test_per_hop":      args.test_per_hop,
        "train_per_hop":     args.train_per_hop,
        "train_nonrel_frac": args.train_nonrel_frac,
        "n_train":           len(train_recs),
        "n_test":            len(test_recs),
        "buckets":           report,
    }, indent=2))
    print(f"\n→ {out_dir}/train_handshake.parquet ({len(train_recs)})")
    print(f"→ {out_dir}/test_handshake.parquet  ({len(test_recs)})")


if __name__ == "__main__":
    main()
