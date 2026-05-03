"""
Generates train/val parquets from family_tree.json.

Two modes:
  --no-context  (default) system_prompt=""  — Exp 1: cartridge memorizes Q&A
  --with-context           system_prompt=family_tree_text — Exp 2: cartridge distills corpus

Usage:
    python examples/graph/graph_generate.py              # → train/val_dataset.parquet
    python examples/graph/graph_generate.py --with-context  # → train/val_dataset_ctx.parquet
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from examples.graph.family_tree import FamilyTree
from cartridges.structs import Conversation, write_conversations


def build_dataset(tree: FamilyTree, system_prompt: str = "") -> list[Conversation]:
    conversations = []
    skipped = 0

    for a_name, b_name in tree.all_pairs():
        result = tree.find_path_reasoning(a_name, b_name)
        if result is None:
            skipped += 1
            continue

        reasoning_str, final_rel, chain_length = result

        conv = Conversation(
            messages=[
                Conversation.Message(
                    role="user",
                    content=f"What is the relationship between {a_name} and {b_name}?",
                    token_ids=None,
                    top_logprobs=None,
                ),
                Conversation.Message(
                    role="assistant",
                    content=reasoning_str,
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
            },
            type="graph_qa",
        )
        conversations.append(conv)

    print(f"Generated {len(conversations)} conversations, skipped {skipped} (no path)")
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
    suffix = ""
    if with_context:
        system_prompt = tree.to_text()
        suffix = "_ctx"
        txt_path = Path(__file__).parent / "family_tree.txt"
        txt_path.write_text(system_prompt)
        print(f"Saved family tree text ({len(system_prompt)} chars) → {txt_path}")
        print("\nFamily tree text preview:")
        print("\n".join(system_prompt.splitlines()[:5]) + "\n...")

    conversations = build_dataset(tree, system_prompt=system_prompt)

    split = int(len(conversations) * 0.8)
    train_convs = conversations[:split]
    val_convs = conversations[split:]

    chain_counts = Counter(c.metadata["chain_length"] for c in conversations)
    print(f"Chain length distribution: {dict(sorted(chain_counts.items()))}")

    print("\nSample:")
    for conv in conversations[:2]:
        print(f"  Q: {conv.messages[0].content}")
        print(f"  A: {conv.messages[1].content}")
        if conv.system_prompt:
            print(f"  sys: {conv.system_prompt[:60]}...")
        print()

    out_dir = Path(__file__).parent
    train_path = out_dir / f"train_dataset{suffix}.parquet"
    val_path = out_dir / f"val_dataset{suffix}.parquet"

    write_conversations(train_convs, str(train_path))
    write_conversations(val_convs, str(val_path))
    print(f"Train: {len(train_convs)} → {train_path}")
    print(f"Val:   {len(val_convs)} → {val_path}")
