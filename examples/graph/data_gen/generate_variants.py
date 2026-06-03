"""
Generate 4 family-tree variants that differ in exactly ONE swapped name.

For each variant emit:
  variants/<name>/family_tree.json
  variants/<name>/family_tree_corpus.txt
  variants/<name>/train_mc.parquet
  variants/<name>/test_mc.parquet
  variants/<name>/train_meta_mc.json
  variants/<name>/test_meta_mc.json
  variants/<name>/split_meta.json

The split (--split-mode, default "question") is identical across variants by
reusing the same seed: question generation is deterministic and a name swap
doesn't change ordering, so the same questions/people are held out in every
variant → cross-variant eval stays comparable.

Run examples/graph/generate_tree.py first.

Usage:
    python examples/graph/generate_tree_variants.py
    python examples/graph/generate_tree_variants.py --new-names Alex,Ben,Carl,Dan
"""
import argparse
import json
from pathlib import Path

from cartridges.structs import write_conversations
from examples.graph.data_gen.family_tree import FamilyTree
from examples.graph.data_gen.qagen import (
    build_rel_lookup, build_all_qa, build_mc_record,
    qa_to_mc_conversation, split_by_person, split_by_question,
    rebalance, balance_letters, MIX_DEFAULT,
)
from examples.graph.paths import BASE_TREE_JSON, VARIANTS_DIR

BASE_TREE = BASE_TREE_JSON
OUT_ROOT = VARIANTS_DIR


def swap_name_in_tree(tree_data: dict, old: str, new: str) -> dict:
    out = json.loads(json.dumps(tree_data))
    if old == new:
        return out
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


def build_variant(tree_data: dict, out_dir: Path, test_frac: float, seed: int,
                  n_verif_per_rel: int, split_mode: str = "question",
                  do_rebalance: bool = True):
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "family_tree.json").write_text(json.dumps(tree_data, indent=2))

    tree = FamilyTree(tree_data)
    corpus_text = tree.to_text()
    (out_dir / "family_tree_corpus.txt").write_text(corpus_text)

    lookup = build_rel_lookup(tree)
    qa = build_all_qa(tree, lookup, n_verif_per_rel=n_verif_per_rel, seed=seed)
    if do_rebalance:
        qa = rebalance(qa, MIX_DEFAULT, seed=seed)
    if split_mode == "person":
        train_qa, test_qa, train_people, test_people = split_by_person(
            tree, qa, test_frac=test_frac, seed=seed,
        )
    else:
        train_qa, test_qa = split_by_question(qa, test_frac=test_frac, seed=seed)
        train_people, test_people = None, None

    all_names = [p["name"] for p in tree.people]
    mc_train = [build_mc_record(q, all_names) for q in train_qa]
    mc_test  = [build_mc_record(q, all_names) for q in test_qa]
    balance_letters(mc_train, seed=seed)
    balance_letters(mc_test,  seed=seed + 1)

    write_conversations([qa_to_mc_conversation(r) for r in mc_train], str(out_dir / "train_mc.parquet"))
    write_conversations([qa_to_mc_conversation(r) for r in mc_test],  str(out_dir / "test_mc.parquet"))
    (out_dir / "train_meta_mc.json").write_text(json.dumps(mc_train, indent=2))
    (out_dir / "test_meta_mc.json").write_text(json.dumps(mc_test,  indent=2))
    (out_dir / "split_meta.json").write_text(json.dumps({
        "split_mode":   split_mode,
        "train_people": train_people,   # null for split_mode=question
        "test_people":  test_people,    # null for split_mode=question
        "n_train_qa":   len(train_qa),
        "n_test_qa":    len(test_qa),
        "test_frac":    test_frac,
        "seed":         seed,
        "rebalanced":   do_rebalance,
        "n_verif_per_rel": n_verif_per_rel,
    }, indent=2))
    print(f"  {out_dir.name}: corpus {len(corpus_text.split())} words, "
          f"train {len(mc_train)} / test {len(mc_test)} (split={split_mode}, rebalance={do_rebalance})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target-name", type=str, default=None,
                   help="Person whose name to swap. Default = first founder.")
    p.add_argument("--new-names", type=str, default="Alex,Ben,Carl,Dan",
                   help="Comma-separated 4 names. Alex = canonical anchor.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-frac", type=float, default=0.2)
    p.add_argument("--split-mode", choices=["person", "question"], default="question",
                   help="person=hold out whole people; question=random hold-out (default)")
    p.add_argument("--rebalance", dest="rebalance", action="store_true", default=True,
                   help="Downsample categories to MIX_DEFAULT target shares (default on)")
    p.add_argument("--no-rebalance", dest="rebalance", action="store_false",
                   help="Keep all raw categories at natural counts")
    p.add_argument("--n-verif-per-rel", type=int, default=12)
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
    print(f"new names:     {new_names}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    for new in new_names:
        variant_tree = swap_name_in_tree(base, target, new)
        build_variant(
            variant_tree, OUT_ROOT / new.lower(),
            test_frac=args.test_frac, seed=args.seed,
            n_verif_per_rel=args.n_verif_per_rel, split_mode=args.split_mode,
            do_rebalance=args.rebalance,
        )

    (OUT_ROOT / "variants_meta.json").write_text(json.dumps({
        "target_original_name": target,
        "variants": new_names,
        "seed": args.seed,
        "test_frac": args.test_frac,
        "split_mode": args.split_mode,
        "n_verif_per_rel": args.n_verif_per_rel,
        "note": "Alex is the canonical/anchor variant. Init cache for exp2 uses Alex corpus for all variants.",
    }, indent=2))
    print(f"meta → {OUT_ROOT / 'variants_meta.json'}")


if __name__ == "__main__":
    main()
