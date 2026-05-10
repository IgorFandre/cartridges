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
    return Conversation(
        system_prompt="",
        metadata={"category": q["category"], "rel": q["rel"], "person": q["person"]},
        messages=[
            Conversation.Message(role="user",      content=q["question"], token_ids=None, top_logprobs=None),
            Conversation.Message(role="assistant", content=q["answer"],   token_ids=None, top_logprobs=None),
        ],
    )


# ── Multiple-choice (5 options A-E) ──────────────────────────────────────────
OPTION_LETTERS = "ABCDE"


def make_mc_options(answer: str, category: int, all_names: list[str], seed: int) -> tuple[list[str], int]:
    """Return (options[5], correct_idx). Deterministic by seed."""
    rng = random.Random(seed)

    if category == 3:
        # counting → integer distractors
        correct_n = int(answer.rstrip("."))
        upper = max(correct_n + 5, 8)
        pool = [n for n in range(0, upper) if n != correct_n]
        rng.shuffle(pool)
        distractors = [f"{n}." for n in pool[:4]]
    elif answer == "None.":
        picks = rng.sample(all_names, 4)
        distractors = [f"{n}." for n in picks]
    elif "," not in answer:
        # single name
        correct_name = answer.rstrip(".")
        pool = [n for n in all_names if n != correct_name]
        picks = rng.sample(pool, 3)
        distractors = [f"{p}." for p in picks] + ["None."]
    else:
        # multi name set
        names_in = [n.strip() for n in answer.rstrip(".").split(",")]
        pool_other = [n for n in all_names if n not in names_in]
        candidates: list[list[str]] = []
        if len(names_in) > 1:
            candidates.append(sorted(names_in[:-1]))
        if pool_other:
            candidates.append(sorted(names_in + [rng.choice(pool_other)]))
            replaced = names_in[:-1] + [rng.choice(pool_other)]
            candidates.append(sorted(replaced))
            k = min(len(names_in), len(pool_other))
            candidates.append(sorted(rng.sample(pool_other, k)))
        candidates.append([])  # None
        seen = {tuple(sorted(names_in))}
        distractors = []
        for c in candidates:
            t = tuple(c)
            if t in seen:
                continue
            seen.add(t)
            distractors.append(", ".join(c) + "." if c else "None.")
            if len(distractors) == 4:
                break
        # pad with random single names if needed
        while len(distractors) < 4 and pool_other:
            pick = rng.choice(pool_other)
            cand = f"{pick}."
            if cand not in distractors and cand != answer:
                distractors.append(cand)
        while len(distractors) < 4:
            distractors.append("None.")  # last resort

    options = distractors[:4] + [answer]
    rng.shuffle(options)
    correct_idx = options.index(answer)
    return options, correct_idx


def format_mc_question(question: str, options: list[str]) -> str:
    lines = [question]
    for i, opt in enumerate(options):
        lines.append(f"{OPTION_LETTERS[i]}. {opt}")
    return "\n".join(lines)


def mc_seed(q: dict) -> int:
    return abs(hash(("mc", q["person"], q["rel"], q["question"]))) & 0xFFFFFFFF


def build_mc_record(q: dict, all_names: list[str]) -> dict:
    opts, idx = make_mc_options(q["answer"], q["category"], all_names, mc_seed(q))
    return {
        **q,
        "options": opts,
        "correct_idx": idx,
        "correct_letter": OPTION_LETTERS[idx],
        "mc_question": format_mc_question(q["question"], opts),
    }


def qa_to_mc_conversation(rec: dict) -> Conversation:
    return Conversation(
        system_prompt="",
        metadata={
            "category": rec["category"], "rel": rec["rel"], "person": rec["person"],
            "options": rec["options"],
            "correct_idx": rec["correct_idx"],
            "correct_letter": rec["correct_letter"],
            "original_answer": rec["answer"],
            "question_text": rec["question"],
        },
        messages=[
            Conversation.Message(role="user",      content=rec["mc_question"],          token_ids=None, top_logprobs=None),
            Conversation.Message(role="assistant", content=f"{rec['correct_letter']}.", token_ids=None, top_logprobs=None),
        ],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-people",   type=int,   default=40)
    parser.add_argument("--seed",       type=int,   default=42)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--neg-frac",   type=float, default=0.3,
                        help="Subsample negative QA (answer 'None.' / '0.') to this fraction of positives. Set 1.0 to keep all.")
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
    print(f"Generated {len(qa_pairs)} QA pairs (raw)")
    for cat in [1, 2, 3]:
        n = sum(1 for q in qa_pairs if q["category"] == cat)
        print(f"  Cat {cat}: {n}")

    rng = random.Random(args.seed)

    # Subsample negative-answer questions
    def is_negative(q: dict) -> bool:
        return q["answer"].strip() in ("None.", "0.")

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

    # Train / test split
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

    # ── MC variants (5 options A-E, letter answer) ──────────────────────────
    all_names = [p["name"] for p in tree.people]
    mc_train = [build_mc_record(q, all_names) for q in train_qa]
    mc_test  = [build_mc_record(q, all_names) for q in test_qa]

    train_mc_path = OUTPUT_DIR / "train_mc.parquet"
    test_mc_path  = OUTPUT_DIR / "test_mc.parquet"
    write_conversations([qa_to_mc_conversation(r) for r in mc_train], str(train_mc_path))
    write_conversations([qa_to_mc_conversation(r) for r in mc_test],  str(test_mc_path))
    print(f"Train MC: {len(mc_train)} → {train_mc_path}")
    print(f"Test  MC: {len(mc_test)}  → {test_mc_path}")

    test_meta_mc_path  = OUTPUT_DIR / "test_meta_mc.json"
    train_meta_mc_path = OUTPUT_DIR / "train_meta_mc.json"
    test_meta_mc_path.write_text(json.dumps(mc_test,  indent=2))
    train_meta_mc_path.write_text(json.dumps(mc_train, indent=2))
    print(f"MC metadata → {test_meta_mc_path}, {train_meta_mc_path}")


if __name__ == "__main__":
    main()
