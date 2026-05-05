"""
Generate kinship QA dataset from family tree.
Saves train.parquet, test.parquet, test_meta.json, family_tree_corpus.txt.

Usage:
    python examples/graph/graph_qagen.py [--n-people 40] [--seed 42] [--train-frac 0.8]
"""
import argparse
import json
import random
from pathlib import Path

from cartridges.structs import Conversation, write_conversations
from examples.graph.family_tree import FamilyTree
from examples.graph.generate_tree import generate_family_tree

TREE_PATH = Path(__file__).parent / "family_tree.json"
OUTPUT_DIR = Path(__file__).parent

# Cat 1: 1-hop, single answer
CAT1_SINGLE = [
    ("father",  "Who is {name}'s father?"),
    ("mother",  "Who is {name}'s mother?"),
    ("husband", "Who is {name}'s husband?"),
    ("wife",    "Who is {name}'s wife?"),
]

# Cat 1: 1-hop, multi-answer
CAT1_MULTI = [
    ("son",      "Who are {name}'s sons?"),
    ("daughter", "Who are {name}'s daughters?"),
]

# Cat 2: multi-hop, single answer
CAT2_SINGLE = [
    ("grandfather", "Who is {name}'s grandfather?"),
    ("grandmother", "Who is {name}'s grandmother?"),
]

# Cat 2: multi-hop, multi-answer
CAT2_MULTI = [
    ("brother",      "Who are {name}'s brothers?"),
    ("sister",       "Who are {name}'s sisters?"),
    ("uncle",        "Who are {name}'s uncles?"),
    ("aunt",         "Who are {name}'s aunts?"),
    ("nephew",       "Who are {name}'s nephews?"),
    ("niece",        "Who are {name}'s nieces?"),
    ("grandson",     "Who are {name}'s grandsons?"),
    ("granddaughter","Who are {name}'s granddaughters?"),
    ("cousin",       "Who are {name}'s cousins?"),
]

# Cat 3: counting
CAT3 = [
    (["son", "daughter"], "How many children does {name} have?"),
    (["son"],             "How many sons does {name} have?"),
    (["daughter"],        "How many daughters does {name} have?"),
    (["brother", "sister"], "How many siblings does {name} have?"),
    (["grandson", "granddaughter"], "How many grandchildren does {name} have?"),
]


def build_rel_lookup(tree: FamilyTree) -> dict[str, dict[str, list[str]]]:
    """
    Returns lookup[person][rel] = [people who ARE person's rel].
    E.g. lookup["Alice"]["father"] = ["Bob"] means Bob is Alice's father.

    Uses BFS (find_path_reasoning) over all ordered pairs.
    find_path_reasoning(a, b) returns (_, rel, _) where a is b's rel.
    So b's [rel] includes a.
    """
    names = [p["name"] for p in tree.people]
    lookup: dict[str, dict[str, list[str]]] = {n: {} for n in names}

    for a in names:
        for b in names:
            if a == b:
                continue
            result = tree.find_path_reasoning(a, b)
            if result is None:
                continue
            _, rel, _ = result
            if rel == "distant relative":
                continue
            lookup[b].setdefault(rel, [])
            if a not in lookup[b][rel]:
                lookup[b][rel].append(a)

    return lookup


def format_answer(names: list[str]) -> str:
    if not names:
        return "None."
    return ", ".join(sorted(names)) + "."


def generate_qa_pairs(tree: FamilyTree, lookup: dict) -> list[dict]:
    people = [p["name"] for p in tree.people]
    qa: list[dict] = []

    for name in people:
        for rel, template in CAT1_SINGLE:
            qa.append({
                "question": template.format(name=name),
                "answer":   format_answer(lookup[name].get(rel, [])),
                "category": 1,
                "rel":      rel,
                "person":   name,
            })

        for rel, template in CAT1_MULTI:
            qa.append({
                "question": template.format(name=name),
                "answer":   format_answer(lookup[name].get(rel, [])),
                "category": 1,
                "rel":      rel,
                "person":   name,
            })

        for rel, template in CAT2_SINGLE:
            qa.append({
                "question": template.format(name=name),
                "answer":   format_answer(lookup[name].get(rel, [])),
                "category": 2,
                "rel":      rel,
                "person":   name,
            })

        for rel, template in CAT2_MULTI:
            qa.append({
                "question": template.format(name=name),
                "answer":   format_answer(lookup[name].get(rel, [])),
                "category": 2,
                "rel":      rel,
                "person":   name,
            })

        for rel_list, template in CAT3:
            count = sum(len(lookup[name].get(r, [])) for r in rel_list)
            qa.append({
                "question": template.format(name=name),
                "answer":   f"{count}.",
                "category": 3,
                "rel":      "+".join(rel_list),
                "person":   name,
            })

    return qa


def qa_to_conversation(q: dict) -> Conversation:
    return Conversation(messages=[
        Conversation.Message(role="user",      content=q["question"], token_ids=None, top_logprobs=None),
        Conversation.Message(role="assistant", content=q["answer"],   token_ids=None, top_logprobs=None),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-people",   type=int,   default=40)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    args = parser.parse_args()

    # Load or generate tree
    if TREE_PATH.exists():
        with open(TREE_PATH) as f:
            data = json.load(f)
        tree = FamilyTree(data)
        print(f"Loaded tree: {len(tree.people)} people")
    else:
        data = generate_family_tree(n_people=args.n_people, seed=args.seed)
        with open(TREE_PATH, "w") as f:
            json.dump(data, f, indent=2)
        tree = FamilyTree(data)
        print(f"Generated tree: {len(tree.people)} people")

    # Save corpus text
    corpus_text = tree.to_text()
    corpus_path = OUTPUT_DIR / "family_tree_corpus.txt"
    corpus_path.write_text(corpus_text)
    print(f"Corpus: {len(corpus_text.split())} words → {corpus_path}")

    # Build relationship lookup via BFS over all pairs
    print("Building relationship lookup (BFS over all pairs)...")
    lookup = build_rel_lookup(tree)

    # Generate QA pairs
    qa_pairs = generate_qa_pairs(tree, lookup)
    print(f"Generated {len(qa_pairs)} QA pairs")
    for cat in [1, 2, 3]:
        n = sum(1 for q in qa_pairs if q["category"] == cat)
        print(f"  Cat {cat}: {n}")

    # Train / test split
    rng = random.Random(args.seed)
    rng.shuffle(qa_pairs)
    n_train = int(len(qa_pairs) * args.train_frac)
    train_qa = qa_pairs[:n_train]
    test_qa  = qa_pairs[n_train:]

    # Save parquets
    train_path = OUTPUT_DIR / "train.parquet"
    test_path  = OUTPUT_DIR / "test.parquet"
    write_conversations([qa_to_conversation(q) for q in train_qa], str(train_path))
    write_conversations([qa_to_conversation(q) for q in test_qa],  str(test_path))
    print(f"Train: {len(train_qa)} → {train_path}")
    print(f"Test:  {len(test_qa)} → {test_path}")

    # Save test metadata for eval scoring
    test_meta_path = OUTPUT_DIR / "test_meta.json"
    test_meta_path.write_text(json.dumps(test_qa, indent=2))
    print(f"Test metadata → {test_meta_path}")


if __name__ == "__main__":
    main()
