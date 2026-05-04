"""
Generates MC-format training data with FIXED options per pair.

Training assistant format:
    <think>Alice is Bob's sister. Bob is Tom's mother. So Alice is Tom's aunt.</think> A

Options are pre-computed and stored in metadata so eval uses identical options
(no random sampling at eval time → perfect format alignment with training).

Modes:
  --no-context   (default) system_prompt=""  → Exp soundness
  --with-context            system_prompt=family_tree_text

Usage:
    python examples/graph/graph_generate_mc.py              # → train/val_dataset_mc.parquet
    python examples/graph/graph_generate_mc.py --with-context  # → train/val_dataset_mc_ctx.parquet
"""
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from examples.graph.family_tree import FamilyTree
from examples.graph.graph_evals import LETTERS, DONT_KNOW, FORMAT_INSTRUCTION, _build_mc_prompt
from cartridges.structs import Conversation, write_conversations

SEED = 42


def _build_options(correct_rel: str, all_rels: list[str], rng: random.Random) -> tuple[list[str], str]:
    """Sample 3 distractors + correct + DONT_KNOW, shuffle, return (options, correct_letter)."""
    pool = [r for r in all_rels if r != correct_rel and r != DONT_KNOW]
    distractors = rng.sample(pool, min(3, len(pool)))
    options = distractors + [correct_rel, DONT_KNOW]
    rng.shuffle(options)
    correct_letter = LETTERS[options.index(correct_rel)]
    return options, correct_letter


def build_mc_dataset(tree: FamilyTree, system_prompt: str = "", seed: int = SEED) -> list[Conversation]:
    rng = random.Random(seed)

    # First pass: collect all pairs and relationships
    pairs = []
    for a_name, b_name in tree.all_pairs():
        result = tree.find_path_reasoning(a_name, b_name)
        if result is None:
            continue
        reasoning_str, final_rel, chain_length = result
        pairs.append((a_name, b_name, reasoning_str, final_rel, chain_length))

    all_rels = sorted(set(p[3] for p in pairs))

    conversations = []
    for a_name, b_name, reasoning_str, final_rel, chain_length in pairs:
        options, correct_letter = _build_options(final_rel, all_rels, rng)

        question = f"What is the relationship between {a_name} and {b_name}?"
        user_content = _build_mc_prompt(question, options)

        # Assistant: <think>reasoning chain</think> correct_letter
        assistant_content = f"<think>{reasoning_str}</think> {correct_letter}"

        conv = Conversation(
            messages=[
                Conversation.Message(
                    role="user",
                    content=user_content,
                    token_ids=None,
                    top_logprobs=None,
                ),
                Conversation.Message(
                    role="assistant",
                    content=assistant_content,
                    token_ids=None,
                    top_logprobs=None,
                ),
            ],
            system_prompt=system_prompt,
            metadata={
                "person_a": a_name,
                "person_b": b_name,
                "final_rel": final_rel,
                "chain_length": chain_length,
                # Fixed options stored for eval alignment
                "options_json": json.dumps(options),
                "correct_letter": correct_letter,
            },
            type="graph_qa_mc",
        )
        conversations.append(conv)

    print(f"Generated {len(conversations)} MC conversations ({len(all_rels)} unique relationships)")
    return conversations


if __name__ == "__main__":
    with_context = "--with-context" in sys.argv

    tree_path = Path(__file__).parent / "family_tree.json"
    if not tree_path.exists():
        print("family_tree.json not found. Run generate_tree.py first.")
        sys.exit(1)

    tree = FamilyTree.load(tree_path)
    print(f"Loaded {len(tree.people)} people from {tree_path}")

    system_prompt = ""
    suffix = "_mc"
    if with_context:
        txt_path = Path(__file__).parent / "family_tree.txt"
        if not txt_path.exists():
            txt_path.write_text(tree.to_text())
        system_prompt = txt_path.read_text()
        suffix = "_mc_ctx"
        print(f"Graph context: {len(system_prompt)} chars")

    conversations = build_mc_dataset(tree, system_prompt=system_prompt)

    split = int(len(conversations) * 0.8)
    train_convs = conversations[:split]
    val_convs = conversations[split:]

    chain_counts = Counter(c.metadata["chain_length"] for c in conversations)
    print(f"Chain length distribution: {dict(sorted(chain_counts.items()))}")

    print("\nSample:")
    c = conversations[0]
    print(f"  [user]\n{c.messages[0].content}")
    print(f"  [assistant]\n{c.messages[1].content[:120]}...")
    print()

    out_dir = Path(__file__).parent
    train_path = out_dir / f"train_dataset{suffix}.parquet"
    val_path = out_dir / f"val_dataset{suffix}.parquet"

    write_conversations(train_convs, str(train_path))
    write_conversations(val_convs, str(val_path))
    print(f"Train: {len(train_convs)} → {train_path}")
    print(f"Val:   {len(val_convs)} → {val_path}")
