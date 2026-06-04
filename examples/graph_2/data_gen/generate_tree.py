"""
Generate a family tree with guaranteed depth >= `--depth` (default 8) generations.

Strategy: build a deterministic lineal SPINE of (depth+1) people
(`Anchor0 → ... → Anchor{depth}`), each with a spouse, so depth is guaranteed.
Then optionally merge a breadth-component from the stock generate_family_tree
for extra people and more non-relative pairs.

The spine also adds 1 sibling sub-branch per generation to populate deep
bucket pairs beyond the single spine chain.

Usage:
    python -m examples.graph_2.data_gen.generate_tree
    python -m examples.graph_2.data_gen.generate_tree --depth 8 --n-people 100 --seed 42
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

# Reuse breadth growth from the existing generator (REUSE, unchanged).
from examples.graph.data_gen.generate_tree import generate_family_tree
from examples.graph.data_gen.family_tree import FamilyTree


# ── Depth measurement ─────────────────────────────────────────────────────────
def tree_depth(parent_child: list[dict]) -> int:
    """Longest directed parent→child chain (number of edges = generations-1)."""
    children: dict[str, list[str]] = defaultdict(list)
    nodes: set[str] = set()
    for e in parent_child:
        children[e["parent"]].append(e["child"])
        nodes.update([e["parent"], e["child"]])

    @lru_cache(maxsize=None)
    def longest_from(n: str) -> int:
        kids = children[n]
        return 0 if not kids else 1 + max(longest_from(c) for c in kids)

    return max((longest_from(n) for n in nodes), default=0)


# ── Spine builder ─────────────────────────────────────────────────────────────
def build_spine(depth: int = 8, with_siblings: bool = True) -> dict:
    """Build a deterministic lineal spine of (depth+1) people.

    Names: AnchorM{i}/AnchorF{i} (main chain), AnchorSpouseM{i}/AnchorSpouseF{i},
    AnchorSibM{i}/AnchorSibF{i} (one sibling per generation when with_siblings=True).
    None of these appear in MALE_NAMES/FEMALE_NAMES (they start with 'Anchor'),
    so they are safe to merge with the stock generator output.
    """
    people: list[dict] = []
    parent_child: list[dict] = []
    spouses: list[dict] = []

    # Generation 0 founders
    main_chain: list[str] = []
    for i in range(depth + 1):
        gender = "male" if i % 2 == 0 else "female"
        name = f"AnchorM{i}" if gender == "male" else f"AnchorF{i}"
        people.append({"name": name, "gender": gender})
        main_chain.append(name)

        # Spouse (opposite gender)
        sp_gender = "female" if gender == "male" else "male"
        sp_name = f"AnchorSpouseF{i}" if sp_gender == "female" else f"AnchorSpouseM{i}"
        people.append({"name": sp_name, "gender": sp_gender})
        spouses.append({"a": name, "b": sp_name})

    # Parent→child edges along the spine
    for i in range(depth):
        parent = main_chain[i]
        child  = main_chain[i + 1]
        # Both main_chain[i] and their spouse are parents of main_chain[i+1]
        sp_name = f"AnchorSpouseF{i}" if main_chain[i].startswith("AnchorM") else f"AnchorSpouseM{i}"
        parent_child.append({"parent": parent,  "child": child})
        parent_child.append({"parent": sp_name, "child": child})

    # Optional: add one sibling per interior generation to enrich deep pairs
    if with_siblings:
        for i in range(depth):
            parent = main_chain[i]
            sp_name = f"AnchorSpouseF{i}" if main_chain[i].startswith("AnchorM") else f"AnchorSpouseM{i}"
            sib_gender = "female" if i % 2 == 0 else "male"
            sib_name = f"AnchorSibF{i}" if sib_gender == "female" else f"AnchorSibM{i}"
            people.append({"name": sib_name, "gender": sib_gender})
            parent_child.append({"parent": parent,  "child": sib_name})
            parent_child.append({"parent": sp_name, "child": sib_name})

    return {
        "people":       people,
        "parent_child": parent_child,
        "spouses":      spouses,
    }


# ── Merge helper ──────────────────────────────────────────────────────────────
def merge_trees(*trees: dict) -> dict:
    """Merge multiple tree dicts (dedup by name; assert all names unique)."""
    all_names: set[str] = set()
    people:       list[dict] = []
    parent_child: list[dict] = []
    spouses:      list[dict] = []

    for tree in trees:
        for p in tree.get("people", []):
            if p["name"] not in all_names:
                all_names.add(p["name"])
                people.append(p)
        for e in tree.get("parent_child", []):
            parent_child.append(e)
        for s in tree.get("spouses", []):
            spouses.append(s)

    # Dedup edges
    seen_pc: set[tuple] = set()
    deduped_pc: list[dict] = []
    for e in parent_child:
        key = (e["parent"], e["child"])
        if key not in seen_pc:
            seen_pc.add(key)
            deduped_pc.append(e)

    seen_sp: set[frozenset] = set()
    deduped_sp: list[dict] = []
    for s in spouses:
        key = frozenset([s["a"], s["b"]])
        if key not in seen_sp:
            seen_sp.add(key)
            deduped_sp.append(s)

    return {"people": people, "parent_child": deduped_pc, "spouses": deduped_sp}


# ── Top-level builder ─────────────────────────────────────────────────────────
def build_lineage_tree(depth: int = 8, n_people: int = 100, seed: int = 42) -> dict:
    """Build a tree with guaranteed depth and ~n_people total people.

    Combines a deterministic spine (guarantees depth) with a breadth component
    from generate_family_tree for volume and variety.
    """
    spine = build_spine(depth=depth, with_siblings=True)

    # Breadth component via existing generator
    n_breadth = max(n_people - len(spine["people"]), 20)
    breadth = generate_family_tree(
        n_people=n_breadth,
        seed=seed,
        max_depth=5,
        min_kids=1,
        max_kids=2,
        founders=2,
        spouse_prob=0.9,
    )

    merged = merge_trees(spine, breadth)

    # Validate depth
    d = tree_depth(merged["parent_child"])
    assert d >= depth, (
        f"tree_depth={d} < required {depth}. "
        "Increase depth or check spine construction."
    )

    merged["_meta"] = {
        "n_people":    len(merged["people"]),
        "depth":       d,
        "n_edges":     len(merged["parent_child"]),
        "n_spouses":   len(merged["spouses"]),
        "seed":        seed,
        "spine_depth": depth,
    }
    return merged


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    from examples.graph_2 import paths

    ap = argparse.ArgumentParser()
    ap.add_argument("--depth",    type=int, default=8,  help="Required genealogical depth (generations-1)")
    ap.add_argument("--n-people", type=int, default=100)
    ap.add_argument("--seed",     type=int, default=42)
    ap.add_argument("--out",      type=str, default=str(paths.BASE_TREE_JSON))
    args = ap.parse_args()

    tree = build_lineage_tree(depth=args.depth, n_people=args.n_people, seed=args.seed)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(tree, f, indent=2)

    m = tree["_meta"]
    print(
        f"Generated {m['n_people']} people, "
        f"depth={m['depth']}, "
        f"{m['n_edges']} parent-child edges, "
        f"{m['n_spouses']} spouse pairs"
    )

    # Also write corpus text
    ft = FamilyTree(tree)
    corpus_path = out_path.parent / "family_tree_corpus.txt"
    corpus_path.write_text(ft.to_text())
    print(f"Saved tree    → {out_path}")
    print(f"Saved corpus  → {corpus_path}")


if __name__ == "__main__":
    main()
