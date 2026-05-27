"""
Narrative-prose corpus generator. Produces family_tree_narrative.txt next to
the supplied --tree (or in --out-dir).

Used by ICL baseline (exp 3) to compare structured-list corpus vs prose
corpus on the same tree.

Usage:
    python examples/graph/generate_narrative_corpus.py
    python examples/graph/generate_narrative_corpus.py --tree variants/ben/family_tree.json
"""
import argparse
import json
import random
from pathlib import Path

from examples.graph.family_tree import FamilyTree


def to_narrative(tree: FamilyTree, seed: int = 0) -> str:
    rng = random.Random(seed)
    children_of: dict[str, list[str]] = {}
    for e in tree.parent_child:
        children_of.setdefault(e["parent"], []).append(e["child"])

    paragraphs = ["This is the story of a family."]

    described = set()
    couple_intros = [
        "{a} and {b} were married.",
        "{a} married {b}.",
        "{a} and {b} built a life together.",
        "{a} took {b} as a spouse.",
    ]
    family_lines = [
        "Together, {a} and {b} had {kids_phrase}.",
        "{a} and {b} raised {kids_phrase}.",
        "The couple had {kids_phrase}.",
        "Their family grew to include {kids_phrase}.",
    ]
    single_lines = [
        "{a} also became a parent. {a}'s child was {kids}.",
        "{a} had a child named {kids}.",
    ]

    def kid_phrase(kids: list[str]) -> str:
        if len(kids) == 1:
            return f"one child, {kids[0]}"
        return f"{len(kids)} children — {', '.join(kids[:-1])}, and {kids[-1]}"

    for pair in tree.spouses:
        a, b = pair["a"], pair["b"]
        kids_a = set(children_of.get(a, []))
        kids_b = set(children_of.get(b, []))
        shared = sorted(kids_a & kids_b)
        sents = [rng.choice(couple_intros).format(a=a, b=b)]
        if shared:
            sents.append(rng.choice(family_lines).format(
                a=a, b=b, kids_phrase=kid_phrase(shared),
            ))
        paragraphs.append(" ".join(sents))
        described.update([a, b])

    for edge in tree.parent_child:
        p = edge["parent"]
        if p in described:
            continue
        kids = sorted(children_of.get(p, []))
        if not kids:
            continue
        paragraphs.append(rng.choice(single_lines).format(
            a=p, kids=", ".join(kids),
        ))
        described.add(p)

    return "\n\n".join(paragraphs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", type=str,
                    default=str(Path(__file__).parent / "family_tree.json"))
    ap.add_argument("--out-dir", type=str, default=None,
                    help="Default: directory of --tree.")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tree_path = Path(args.tree)
    data = json.loads(tree_path.read_text())
    tree = FamilyTree(data)
    out_dir = Path(args.out_dir) if args.out_dir else tree_path.parent

    narr = to_narrative(tree, seed=args.seed)
    (out_dir / "family_tree_narrative.txt").write_text(narr)
    print(f"narrative ({len(narr.split())} words) → {out_dir}/family_tree_narrative.txt")


if __name__ == "__main__":
    main()
