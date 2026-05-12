"""
Generate 4 variants of family_tree.json + corpus + MC parquets.
All variants identical except ONE person's name (target) is swapped.

Output: variants/{variant_name}/
  - family_tree.json
  - family_tree_corpus.txt
  - train_mc.parquet
  - test_mc.parquet
  - test_meta_mc.json
  - train_meta_mc.json

Usage:
    python examples/graph/generate_tree_variants.py
    python examples/graph/generate_tree_variants.py --target-name James --new-names Alex,Ben,Carl,Dan
"""
import argparse
import json
import random
from pathlib import Path

from cartridges.structs import write_conversations
from examples.graph.family_tree import FamilyTree
from examples.graph.graph_qagen import (
    build_rel_lookup, generate_qa_pairs, build_mc_record,
    qa_to_mc_conversation,
)
from examples.graph.graph_qagen_cot import (
    generate_cot_qa_pairs, build_mc_cot_record, qa_to_mc_cot_conversation,
)

BASE_TREE = Path(__file__).parent / "family_tree.json"
OUT_ROOT = Path(__file__).parent / "variants"


def swap_name_in_tree(tree_data: dict, old: str, new: str) -> dict:
    out = json.loads(json.dumps(tree_data))
    if old == new:
        return out  # identity: keep original tree
    found = False
    for p in out["people"]:
        if p["name"] == old:
            p["name"] = new
            found = True
    for e in out["parent_child"]:
        if e["parent"] == old:
            e["parent"] = new
        if e["child"] == old:
            e["child"] = new
    for s in out["spouses"]:
        if s["a"] == old:
            s["a"] = new
        if s["b"] == old:
            s["b"] = new
    if not found:
        raise ValueError(f"name {old!r} not present in tree")
    if any(p["name"] == old for p in out["people"]):
        raise ValueError(f"swap incomplete for {old!r}")
    if any(p["name"] == new for p in tree_data["people"] if p["name"] != old):
        raise ValueError(f"new name {new!r} collides with existing person")
    return out


def build_variant(tree_data: dict, out_dir: Path, neg_frac: float, train_frac: float, seed: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "family_tree.json").write_text(json.dumps(tree_data, indent=2))

    tree = FamilyTree(tree_data)

    corpus_text = tree.to_text()
    (out_dir / "family_tree_corpus.txt").write_text(corpus_text)

    lookup = build_rel_lookup(tree)
    qa_pairs = generate_qa_pairs(tree, lookup)

    rng = random.Random(seed)

    def is_negative(q): return q["answer"].strip() in ("None.", "0.")
    pos_qa = [q for q in qa_pairs if not is_negative(q)]
    neg_qa = [q for q in qa_pairs if is_negative(q)]
    if neg_frac < 1.0:
        target_neg = min(int(len(pos_qa) * neg_frac), len(neg_qa))
        rng.shuffle(neg_qa)
        neg_qa = neg_qa[:target_neg]
    qa_pairs = pos_qa + neg_qa

    rng.shuffle(qa_pairs)
    n_train = int(len(qa_pairs) * train_frac)
    train_qa = qa_pairs[:n_train]
    test_qa  = qa_pairs[n_train:]

    all_names = [p["name"] for p in tree.people]
    mc_train = [build_mc_record(q, all_names) for q in train_qa]
    mc_test  = [build_mc_record(q, all_names) for q in test_qa]

    write_conversations([qa_to_mc_conversation(r) for r in mc_train], str(out_dir / "train_mc.parquet"))
    write_conversations([qa_to_mc_conversation(r) for r in mc_test],  str(out_dir / "test_mc.parquet"))
    (out_dir / "train_meta_mc.json").write_text(json.dumps(mc_train, indent=2))
    (out_dir / "test_meta_mc.json").write_text(json.dumps(mc_test, indent=2))

    # ── CoT variants ───────────────────────────────────────────────────────
    cot_qa = generate_cot_qa_pairs(tree, lookup)
    cot_index = {(q["person"], q["rel"]): q for q in cot_qa}
    # match the same train/test split via (person, rel) keys
    train_keys = {(q["person"], q["rel"]) for q in train_qa}
    test_keys  = {(q["person"], q["rel"]) for q in test_qa}
    cot_train_qa = [cot_index[k] for k in train_keys if k in cot_index]
    cot_test_qa  = [cot_index[k] for k in test_keys  if k in cot_index]

    mc_cot_train = [build_mc_cot_record(q, all_names) for q in cot_train_qa]
    mc_cot_test  = [build_mc_cot_record(q, all_names) for q in cot_test_qa]
    write_conversations([qa_to_mc_cot_conversation(r) for r in mc_cot_train],
                        str(out_dir / "train_cot_mc.parquet"))
    write_conversations([qa_to_mc_cot_conversation(r) for r in mc_cot_test],
                        str(out_dir / "test_cot_mc.parquet"))
    (out_dir / "train_meta_cot_mc.json").write_text(json.dumps(mc_cot_train, indent=2))
    (out_dir / "test_meta_cot_mc.json").write_text(json.dumps(mc_cot_test, indent=2))

    print(f"  {out_dir.name}: corpus {len(corpus_text.split())} words, "
          f"train {len(mc_train)} (cot {len(mc_cot_train)}) / "
          f"test {len(mc_test)} (cot {len(mc_cot_test)})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-name", type=str, default=None,
                   help="Person whose name to swap. Default = first founder.")
    p.add_argument("--new-names", type=str, default="Alex,Ben,Carl,Dan",
                   help="Comma-separated 4 names to swap in.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-frac", type=float, default=0.8)
    p.add_argument("--neg-frac", type=float, default=0.3)
    args = p.parse_args()

    if not BASE_TREE.exists():
        raise FileNotFoundError(
            f"{BASE_TREE} missing — run examples/graph/generate_tree.py first."
        )
    base = json.loads(BASE_TREE.read_text())

    target = args.target_name or base["people"][0]["name"]
    new_names = [n.strip() for n in args.new_names.split(",")]
    if len(new_names) != 4:
        raise ValueError("--new-names must have exactly 4 comma-separated names")

    print(f"target person: {target!r}")
    print(f"new names: {new_names}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for new in new_names:
        variant_tree = swap_name_in_tree(base, target, new)
        out_dir = OUT_ROOT / new.lower()
        build_variant(
            variant_tree, out_dir,
            neg_frac=args.neg_frac,
            train_frac=args.train_frac,
            seed=args.seed,
        )

    meta = {
        "target_original_name": target,
        "variants": new_names,
        "seed": args.seed,
        "train_frac": args.train_frac,
        "neg_frac": args.neg_frac,
    }
    (OUT_ROOT / "variants_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"meta → {OUT_ROOT / 'variants_meta.json'}")


if __name__ == "__main__":
    main()
