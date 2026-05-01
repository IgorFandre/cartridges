"""
Generates train_dataset.parquet and val_dataset.parquet from family_tree.json.

Each pair (A, B) where a path exists becomes one Conversation:
  user:      "What is the relationship between Alice and Tom?"
  assistant: "Alice is Valery's sister. Valery is Tom's mother. So Alice is Tom's aunt."

Metadata per conversation: person_a, person_b, final_rel, chain_length.

Split: 80% train / 20% val (deterministic, by index).

Usage:
    python examples/graph/graph_generate.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2]))

from examples.graph.family_tree import FamilyTree
from cartridges.structs import Conversation, write_conversations


def build_dataset(tree: FamilyTree) -> list[Conversation]:
    conversations = []
    pairs = tree.all_pairs()
    skipped = 0

    for a_name, b_name in pairs:
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
            system_prompt="",
            metadata={
                "person_a": a_name,
                "person_b": b_name,
                "final_rel": final_rel,
                "chain_length": chain_length,
            },
            type="graph_qa",
        )
        conversations.append(conv)

    print(f"Generated {len(conversations)} conversations, skipped {skipped} pairs (no path)")
    return conversations


if __name__ == "__main__":
    tree_path = Path(__file__).parent / "family_tree.json"
    if not tree_path.exists():
        print("family_tree.json not found. Run generate_tree.py first.")
        sys.exit(1)

    tree = FamilyTree.load(tree_path)
    print(f"Loaded {len(tree.people)} people from {tree_path}")

    conversations = build_dataset(tree)

    # 80/20 split
    split = int(len(conversations) * 0.8)
    train_convs = conversations[:split]
    val_convs = conversations[split:]

    # Print chain length distribution
    from collections import Counter
    chain_counts = Counter(c.metadata["chain_length"] for c in conversations)
    print(f"\nChain length distribution: {dict(sorted(chain_counts.items()))}")

    # Print a few samples
    print("\nSample conversations:")
    for conv in conversations[:3]:
        print(f"\n  Q: {conv.messages[0].content}")
        print(f"  A: {conv.messages[1].content}")
        print(f"  chain_length: {conv.metadata['chain_length']}")

    out_dir = Path(__file__).parent
    train_path = out_dir / "train_dataset.parquet"
    val_path = out_dir / "val_dataset.parquet"

    write_conversations(train_convs, str(train_path))
    write_conversations(val_convs, str(val_path))
    print(f"\nTrain: {len(train_convs)} conversations → {train_path}")
    print(f"Val:   {len(val_convs)} conversations → {val_path}")
