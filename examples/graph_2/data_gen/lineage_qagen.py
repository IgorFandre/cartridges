"""
Yes/No lineage QA generator.

All questions: "Is {X} an ancestor of {Y}, {n} generation(s) above? Answer Yes or No."
(and the descendant variant).

Gold label: Yes iff LineageIndex.ancestor_distance(X, Y) == claimed n.

Per-n bucketing:
  n_bucket = true_distance (for true-n and wrong-n rows)
  n_bucket = "none"        (for non-relative rows)

Balance: per real lineal pair at true distance t, emit:
  - 1 true positive  (claimed_n = t → Yes)
  - 1 wrong-n negative (claimed_n = t±1 → No)
  - ~1 non-relative negative (random n → No)
Then cap the majority class to 50/50 overall.

Output: train_lineage.parquet / test_lineage.parquet (Conversation objects),
        train_meta.json / test_meta.json / split_meta.json.

Usage:
    python -m examples.graph_2.data_gen.lineage_qagen
    python -m examples.graph_2.data_gen.lineage_qagen --test-frac 0.2 --seed 42
"""
from __future__ import annotations
import argparse
import json
import math
import random
from pathlib import Path

from cartridges.structs import Conversation, write_conversations
from examples.graph.data_gen.family_tree import FamilyTree
from examples.graph_2.data_gen.lineage_index import LineageIndex


# ── Question templates ─────────────────────────────────────────────────────────
def _ancestor_q(x: str, y: str, n: int) -> str:
    gen = "generation" if n == 1 else "generations"
    return f"Is {x} an ancestor of {y}, {n} {gen} above? Answer Yes or No."


def _descendant_q(x: str, y: str, n: int) -> str:
    gen = "generation" if n == 1 else "generations"
    return f"Is {x} a descendant of {y}, {n} {gen} below? Answer Yes or No."


# ── Record builder ─────────────────────────────────────────────────────────────
def _make_record(
    *,
    direction: str,    # "ancestor" | "descendant"
    x: str, y: str,    # subject and anchor
    claimed_n: int,
    label: str,        # "Yes" | "No"
    true_distance: int | None,   # actual lineal distance (or None for non-relative)
    n_bucket: int | str,         # true_distance or "none"
) -> dict:
    if direction == "ancestor":
        question = _ancestor_q(x, y, claimed_n)
    else:
        question = _descendant_q(x, y, claimed_n)
    return dict(
        question=question,
        label=label,
        direction=direction,
        x=x, y=y,
        claimed_n=claimed_n,
        true_distance=true_distance,
        n_bucket=n_bucket,
    )


# ── Core generator ────────────────────────────────────────────────────────────
def generate_qa(
    index: LineageIndex,
    rng: random.Random,
    per_pair_wrong_n: int = 1,
    nonrel_frac: float = 0.5,
) -> list[dict]:
    """Generate raw Yes/No QA records before splitting/rebalancing."""
    max_d = index.max_distance()
    all_pairs_by_d = index.by_distance()
    records: list[dict] = []

    for dist, pairs in sorted(all_pairs_by_d.items()):
        for (ancestor, descendant) in pairs:
            # True positive — ancestor direction
            records.append(_make_record(
                direction="ancestor", x=ancestor, y=descendant,
                claimed_n=dist, label="Yes",
                true_distance=dist, n_bucket=dist,
            ))
            # True positive — descendant direction
            records.append(_make_record(
                direction="descendant", x=descendant, y=ancestor,
                claimed_n=dist, label="Yes",
                true_distance=dist, n_bucket=dist,
            ))

            # Wrong-n negatives (per_pair_wrong_n per pair)
            for _ in range(per_pair_wrong_n):
                # Prefer near miss; if at boundary, go the other way
                candidates = []
                if dist > 1:
                    candidates.append(dist - 1)
                if dist < max_d:
                    candidates.append(dist + 1)
                if not candidates:
                    candidates = list(range(1, max(2, max_d + 1)))
                wrong_n = rng.choice(candidates)
                for direction, sx, sy in [
                    ("ancestor",   ancestor,   descendant),
                    ("descendant", descendant, ancestor),
                ]:
                    records.append(_make_record(
                        direction=direction, x=sx, y=sy,
                        claimed_n=wrong_n, label="No",
                        true_distance=dist, n_bucket=dist,
                    ))

    # Non-relative negatives
    n_lineal_pairs = len(index.triples())
    n_nonrel = max(4, int(n_lineal_pairs * nonrel_frac))
    nonrel = index.non_lineal_pairs(n_nonrel * 2, rng)
    rng.shuffle(nonrel)
    for (a, b) in nonrel[:n_nonrel]:
        claimed_n = rng.randint(1, max(max_d, 1))
        direction = rng.choice(["ancestor", "descendant"])
        if direction == "ancestor":
            records.append(_make_record(
                direction="ancestor", x=a, y=b,
                claimed_n=claimed_n, label="No",
                true_distance=None, n_bucket="none",
            ))
        else:
            records.append(_make_record(
                direction="descendant", x=b, y=a,
                claimed_n=claimed_n, label="No",
                true_distance=None, n_bucket="none",
            ))

    rng.shuffle(records)
    return records


# ── Rebalance Yes/No ──────────────────────────────────────────────────────────
def rebalance_labels(records: list[dict], seed: int) -> list[dict]:
    """Cap the majority label class to match the minority, keeping the shuffle."""
    yes = [r for r in records if r["label"] == "Yes"]
    no  = [r for r in records if r["label"] == "No"]
    rng = random.Random(seed)
    n = min(len(yes), len(no))
    rng.shuffle(yes); rng.shuffle(no)
    out = yes[:n] + no[:n]
    rng.shuffle(out)
    return out


# ── Train/test split ──────────────────────────────────────────────────────────
def split_by_question(
    records: list[dict], test_frac: float, seed: int
) -> tuple[list[dict], list[dict]]:
    """Stratified hold-out by n_bucket so every bucket appears in both splits."""
    rng = random.Random(seed)
    by_bucket: dict[str | int, list[dict]] = {}
    for r in records:
        by_bucket.setdefault(r["n_bucket"], []).append(r)

    train, test = [], []
    for bucket in sorted(by_bucket, key=str):
        items = by_bucket[bucket][:]
        rng.shuffle(items)
        n_test = max(1, math.ceil(len(items) * test_frac)) if len(items) > 1 else 0
        test.extend(items[:n_test])
        train.extend(items[n_test:])
    rng.shuffle(train)
    rng.shuffle(test)
    return train, test


# ── Conversation builder ──────────────────────────────────────────────────────
def record_to_conversation(rec: dict) -> Conversation:
    return Conversation(
        system_prompt="",
        type="lineage_qa",
        metadata={
            "question_text": rec["question"],
            "label":          rec["label"],
            "true_distance":  rec["true_distance"],   # int or None
            "claimed_n":      rec["claimed_n"],
            "n_bucket":       str(rec["n_bucket"]),   # always str ("1".."8" or "none")
            "direction":      rec["direction"],
            "x": rec["x"], "y": rec["y"],
        },
        messages=[
            Conversation.Message(
                role="user", content=rec["question"],
                token_ids=None, top_logprobs=None,
            ),
            Conversation.Message(
                role="assistant", content=rec["label"] + ".",
                token_ids=None, top_logprobs=None,
            ),
        ],
    )


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    from examples.graph_2 import paths

    ap = argparse.ArgumentParser()
    ap.add_argument("--tree",            type=str, default=str(paths.BASE_TREE_JSON))
    ap.add_argument("--out-dir",         type=str, default=str(paths.BASE_DIR))
    ap.add_argument("--test-frac",       type=float, default=0.2)
    ap.add_argument("--seed",            type=int, default=42)
    ap.add_argument("--per-pair-wrong-n", type=int, default=1)
    ap.add_argument("--nonrel-frac",     type=float, default=0.5)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ft = FamilyTree.load(args.tree)
    print(f"Tree: {len(ft.people)} people")
    index = LineageIndex.from_tree(ft)
    print(f"Max lineal depth: {index.max_distance()}")
    by_d = index.by_distance()
    print("Lineal pairs per distance:")
    for d in sorted(by_d):
        print(f"  n={d}: {len(by_d[d])} pairs")

    rng = random.Random(args.seed)
    records = generate_qa(
        index, rng,
        per_pair_wrong_n=args.per_pair_wrong_n,
        nonrel_frac=args.nonrel_frac,
    )
    print(f"Raw records: {len(records)}")

    records = rebalance_labels(records, seed=args.seed)
    print(f"After rebalance: {len(records)}")

    yes_ct = sum(1 for r in records if r["label"] == "Yes")
    no_ct  = sum(1 for r in records if r["label"] == "No")
    print(f"  Yes: {yes_ct} ({yes_ct/len(records):.1%})  No: {no_ct} ({no_ct/len(records):.1%})")

    by_bucket: dict = {}
    for r in records:
        by_bucket.setdefault(r["n_bucket"], 0)
        by_bucket[r["n_bucket"]] += 1
    for k in sorted(by_bucket, key=str):
        print(f"  n_bucket={k}: {by_bucket[k]}")

    train_recs, test_recs = split_by_question(records, test_frac=args.test_frac, seed=args.seed)
    print(f"Train: {len(train_recs)}  Test: {len(test_recs)}")

    write_conversations(
        [record_to_conversation(r) for r in train_recs],
        str(out_dir / "train_lineage.parquet"),
    )
    write_conversations(
        [record_to_conversation(r) for r in test_recs],
        str(out_dir / "test_lineage.parquet"),
    )
    (out_dir / "train_meta.json").write_text(json.dumps(train_recs, indent=2))
    (out_dir / "test_meta.json").write_text( json.dumps(test_recs,  indent=2))
    (out_dir / "split_meta.json").write_text(json.dumps({
        "split_mode":    "question",
        "test_frac":     args.test_frac,
        "seed":          args.seed,
        "n_train":       len(train_recs),
        "n_test":        len(test_recs),
        "n_total":       len(records),
        "max_distance":  index.max_distance(),
    }, indent=2))
    print(f"→ {out_dir}/train_lineage.parquet ({len(train_recs)})")
    print(f"→ {out_dir}/test_lineage.parquet  ({len(test_recs)})")


if __name__ == "__main__":
    main()
