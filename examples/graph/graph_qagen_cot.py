"""
Generate kinship QA dataset with chain-of-thought reasoning answers.
Saves train_cot.parquet, test_cot.parquet, test_meta_cot.json.

CoT format:
  - Single-answer (cat1/cat2): BFS reasoning path + "Answer: X."
  - Multi-answer  (cat1/cat2): reasoning per person + final list + "Answer: ..."
  - Counting      (cat3):      list items + count + "Answer: N."

Usage:
    python examples/graph/graph_qagen_cot.py [--n-people 40] [--seed 42] [--train-frac 0.8]
"""
import argparse
import json
import random
from pathlib import Path

from cartridges.structs import Conversation, write_conversations
from examples.graph.family_tree import FamilyTree
from examples.graph.generate_tree import generate_family_tree
from examples.graph.graph_qagen import (
    CAT1_SINGLE, CAT1_MULTI, CAT2_SINGLE, CAT2_MULTI, CAT3,
    build_rel_lookup, format_answer,
)

TREE_PATH = Path(__file__).parent / "family_tree.json"
OUTPUT_DIR = Path(__file__).parent

REL_PLURAL = {
    "son": "sons", "daughter": "daughters",
    "brother": "brothers", "sister": "sisters",
    "uncle": "uncles", "aunt": "aunts",
    "nephew": "nephews", "niece": "nieces",
    "grandson": "grandsons", "granddaughter": "granddaughters",
    "cousin": "cousins",
    "father": "father", "mother": "mother",
    "husband": "husband", "wife": "wife",
    "grandfather": "grandfather", "grandmother": "grandmother",
}

CAT3_LABELS = {
    ("son", "daughter"): "children",
    ("son",): "sons",
    ("daughter",): "daughters",
    ("brother", "sister"): "siblings",
    ("grandson", "granddaughter"): "grandchildren",
}


def cot_single(name: str, rel: str, lookup: dict, tree: FamilyTree) -> str:
    """CoT for single-answer relationship questions."""
    answers = lookup[name].get(rel, [])
    if not answers:
        return f"{name} has no {REL_PLURAL[rel]}. Answer: None."
    a = answers[0]
    result = tree.find_path_reasoning(a, name)
    if result:
        reasoning, _, _ = result
        return f"{reasoning} Answer: {a}."
    return f"Answer: {a}."


def cot_multi(name: str, rel: str, lookup: dict, tree: FamilyTree) -> str:
    """CoT for multi-answer relationship questions."""
    answers = sorted(lookup[name].get(rel, []))
    label = REL_PLURAL[rel]
    if not answers:
        return f"{name} has no {label}. Answer: None."

    lines = []
    for a in answers:
        result = tree.find_path_reasoning(a, name)
        if result:
            reasoning, _, _ = result
            lines.append(f"- {a}: {reasoning}")
        else:
            lines.append(f"- {a}.")

    lines.append(f"{name}'s {label}: {', '.join(answers)}. Answer: {format_answer(answers)}")
    return "\n".join(lines)


def cot_counting(name: str, rel_list: list[str], lookup: dict) -> str:
    """CoT for counting questions."""
    label = CAT3_LABELS[tuple(rel_list)]
    items: list[str] = []
    for r in rel_list:
        items.extend(lookup[name].get(r, []))
    items = sorted(items)
    count = len(items)
    if not items:
        return f"{name} has no {label}. Count: 0. Answer: 0."
    return f"{name}'s {label}: {', '.join(items)}. Count: {count}. Answer: {count}."


def generate_cot_qa_pairs(tree: FamilyTree, lookup: dict) -> list[dict]:
    people = [p["name"] for p in tree.people]
    qa: list[dict] = []

    for name in people:
        for rel, template in CAT1_SINGLE:
            qa.append({
                "question": template.format(name=name),
                "answer":   cot_single(name, rel, lookup, tree),
                "category": 1,
                "rel":      rel,
                "person":   name,
            })

        for rel, template in CAT1_MULTI:
            qa.append({
                "question": template.format(name=name),
                "answer":   cot_multi(name, rel, lookup, tree),
                "category": 1,
                "rel":      rel,
                "person":   name,
            })

        for rel, template in CAT2_SINGLE:
            qa.append({
                "question": template.format(name=name),
                "answer":   cot_single(name, rel, lookup, tree),
                "category": 2,
                "rel":      rel,
                "person":   name,
            })

        for rel, template in CAT2_MULTI:
            qa.append({
                "question": template.format(name=name),
                "answer":   cot_multi(name, rel, lookup, tree),
                "category": 2,
                "rel":      rel,
                "person":   name,
            })

        for rel_list, template in CAT3:
            qa.append({
                "question": template.format(name=name),
                "answer":   cot_counting(name, rel_list, lookup),
                "category": 3,
                "rel":      "+".join(rel_list),
                "person":   name,
            })

    return qa


def qa_to_conversation(q: dict) -> Conversation:
    return Conversation(
        system_prompt="",
        metadata={"category": q["category"], "rel": q["rel"], "person": q["person"]},
        messages=[
            Conversation.Message(role="user",      content=q["question"], token_ids=None, top_logprobs=None),
            Conversation.Message(role="assistant", content=q["answer"],   token_ids=None, top_logprobs=None),
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-people",   type=int,   default=40)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    args = parser.parse_args()

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

    print("Building relationship lookup (BFS)...")
    lookup = build_rel_lookup(tree)

    qa_pairs = generate_cot_qa_pairs(tree, lookup)
    print(f"Generated {len(qa_pairs)} CoT QA pairs")
    for cat in [1, 2, 3]:
        n = sum(1 for q in qa_pairs if q["category"] == cat)
        print(f"  Cat {cat}: {n}")

    # Show sample
    for q in qa_pairs[:2]:
        print(f"\nQ: {q['question']}")
        print(f"A: {q['answer']}")

    rng = random.Random(args.seed)
    rng.shuffle(qa_pairs)
    n_train = int(len(qa_pairs) * args.train_frac)
    train_qa = qa_pairs[:n_train]
    test_qa  = qa_pairs[n_train:]

    train_path = OUTPUT_DIR / "train_cot.parquet"
    test_path  = OUTPUT_DIR / "test_cot.parquet"
    write_conversations([qa_to_conversation(q) for q in train_qa], str(train_path))
    write_conversations([qa_to_conversation(q) for q in test_qa],  str(test_path))
    print(f"Train: {len(train_qa)} → {train_path}")
    print(f"Test:  {len(test_qa)} → {test_path}")

    test_meta_path = OUTPUT_DIR / "test_meta_cot.json"
    test_meta_path.write_text(json.dumps(test_qa, indent=2))
    print(f"Test metadata → {test_meta_path}")


if __name__ == "__main__":
    main()
