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
    make_mc_options, format_mc_question, mc_seed, OPTION_LETTERS,
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


def _plain_answer_single(name: str, rel: str, lookup: dict) -> str:
    answers = lookup[name].get(rel, [])
    return format_answer(answers[:1])


def _plain_answer_multi(name: str, rel: str, lookup: dict) -> str:
    return format_answer(lookup[name].get(rel, []))


def _plain_answer_count(name: str, rel_list: list[str], lookup: dict) -> str:
    count = sum(len(lookup[name].get(r, [])) for r in rel_list)
    return f"{count}."


def generate_cot_qa_pairs(tree: FamilyTree, lookup: dict) -> list[dict]:
    people = [p["name"] for p in tree.people]
    qa: list[dict] = []

    def push(question, cot_text, plain, category, rel, name):
        qa.append({
            "question":     question,
            "answer":       cot_text,
            "plain_answer": plain,
            "category":     category,
            "rel":          rel,
            "person":       name,
        })

    for name in people:
        for rel, template in CAT1_SINGLE:
            push(template.format(name=name), cot_single(name, rel, lookup, tree),
                 _plain_answer_single(name, rel, lookup), 1, rel, name)

        for rel, template in CAT1_MULTI:
            push(template.format(name=name), cot_multi(name, rel, lookup, tree),
                 _plain_answer_multi(name, rel, lookup), 1, rel, name)

        for rel, template in CAT2_SINGLE:
            push(template.format(name=name), cot_single(name, rel, lookup, tree),
                 _plain_answer_single(name, rel, lookup), 2, rel, name)

        for rel, template in CAT2_MULTI:
            push(template.format(name=name), cot_multi(name, rel, lookup, tree),
                 _plain_answer_multi(name, rel, lookup), 2, rel, name)

        for rel_list, template in CAT3:
            push(template.format(name=name), cot_counting(name, rel_list, lookup),
                 _plain_answer_count(name, rel_list, lookup), 3, "+".join(rel_list), name)

    return qa


def _strip_answer_tail(cot_text: str) -> str:
    """Drop trailing 'Answer: ...' from CoT text — keep only reasoning."""
    if "Answer:" in cot_text:
        return cot_text.rsplit("Answer:", 1)[0].rstrip()
    return cot_text.rstrip()


def build_mc_cot_record(q: dict, all_names: list[str]) -> dict:
    opts, idx = make_mc_options(q["plain_answer"], q["category"], all_names, mc_seed(q))
    reasoning = _strip_answer_tail(q["answer"])
    letter = OPTION_LETTERS[idx]
    cot_mc_text = f"<think>\n{reasoning}\n</think>\n\n{letter}."
    return {
        **q,
        "options": opts,
        "correct_idx": idx,
        "correct_letter": letter,
        "mc_question": format_mc_question(q["question"], opts),
        "mc_answer":   cot_mc_text,
    }


def qa_to_mc_cot_conversation(rec: dict) -> Conversation:
    return Conversation(
        system_prompt="",
        metadata={
            "category": rec["category"], "rel": rec["rel"], "person": rec["person"],
            "options": rec["options"],
            "correct_idx": rec["correct_idx"],
            "correct_letter": rec["correct_letter"],
            "original_answer": rec["plain_answer"],
            "question_text": rec["question"],
        },
        messages=[
            Conversation.Message(role="user",      content=rec["mc_question"], token_ids=None, top_logprobs=None),
            Conversation.Message(role="assistant", content=rec["mc_answer"],   token_ids=None, top_logprobs=None),
        ],
    )


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
    parser.add_argument("--neg-frac",   type=float, default=0.3,
                        help="Subsample negative QA (Answer: None./0.) to this fraction of positives. Set 1.0 to keep all.")
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
    print(f"Generated {len(qa_pairs)} CoT QA pairs (raw)")
    for cat in [1, 2, 3]:
        n = sum(1 for q in qa_pairs if q["category"] == cat)
        print(f"  Cat {cat}: {n}")

    # Show sample
    for q in qa_pairs[:2]:
        print(f"\nQ: {q['question']}")
        print(f"A: {q['answer']}")

    rng = random.Random(args.seed)

    # Subsample negative-answer questions (CoT answers end with "Answer: None." / "Answer: 0.")
    def is_negative(q: dict) -> bool:
        tail = q["answer"].rstrip().rsplit("Answer:", 1)[-1].strip().rstrip(".").strip()
        return tail in ("None", "0")

    pos_qa = [q for q in qa_pairs if not is_negative(q)]
    neg_qa = [q for q in qa_pairs if is_negative(q)]
    print(f"Positive: {len(pos_qa)}  Negative: {len(neg_qa)}  (neg-frac={args.neg_frac})")

    if args.neg_frac < 1.0:
        target_neg = int(len(pos_qa) * args.neg_frac)
        target_neg = min(target_neg, len(neg_qa))
        rng.shuffle(neg_qa)
        neg_qa = neg_qa[:target_neg]
        print(f"Subsampled negatives → {len(neg_qa)}")

    qa_pairs = pos_qa + neg_qa
    print(f"Final QA pairs: {len(qa_pairs)} ({len(pos_qa)} pos + {len(neg_qa)} neg)")

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

    # ── MC + CoT (BFS reasoning inside <think>, then letter) ────────────────
    all_names = [p["name"] for p in tree.people]
    mc_train = [build_mc_cot_record(q, all_names) for q in train_qa]
    mc_test  = [build_mc_cot_record(q, all_names) for q in test_qa]

    train_mc_path = OUTPUT_DIR / "train_cot_mc.parquet"
    test_mc_path  = OUTPUT_DIR / "test_cot_mc.parquet"
    write_conversations([qa_to_mc_cot_conversation(r) for r in mc_train], str(train_mc_path))
    write_conversations([qa_to_mc_cot_conversation(r) for r in mc_test],  str(test_mc_path))
    print(f"Train CoT-MC: {len(mc_train)} → {train_mc_path}")
    print(f"Test  CoT-MC: {len(mc_test)}  → {test_mc_path}")

    test_meta_mc_path  = OUTPUT_DIR / "test_meta_cot_mc.json"
    train_meta_mc_path = OUTPUT_DIR / "train_meta_cot_mc.json"
    test_meta_mc_path.write_text(json.dumps(mc_test,  indent=2))
    train_meta_mc_path.write_text(json.dumps(mc_train, indent=2))
    print(f"CoT-MC metadata → {test_meta_mc_path}, {train_meta_mc_path}")


if __name__ == "__main__":
    main()
