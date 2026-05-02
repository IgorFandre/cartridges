"""
ICL baselines for the family tree cartridge experiment.

Two baselines, same val questions/options as GraphRelationshipMCEvalDataset (seed=42):

  zero_shot  — no family tree context. Controls for model prior (~20% random expected).
  icl_edges  — system prompt = raw edges from family_tree.json (parent_child + spouses).
               Model must reason over the raw graph to answer. This is the honest ICL upper bound:
               it gets the exact same facts the cartridge was trained on, just in-context.

Outputs accuracy by chain length + saves to baseline_results.json.

Usage:
    python examples/graph/baseline_graph.py
    python examples/graph/baseline_graph.py --mode zero_shot
    python examples/graph/baseline_graph.py --mode icl_edges
    python examples/graph/baseline_graph.py --batch_size 4
"""
import argparse
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parents[2]))

from examples.graph.family_tree import FamilyTree
from examples.graph.graph_evals import (
    DONT_KNOW,
    FORMAT_INSTRUCTION,
    LETTERS,
    _build_mc_prompt,
    _extract_answer_letter,
)
from cartridges.structs import read_conversations

GRAPH_DIR = Path(__file__).parent
TREE_PATH = GRAPH_DIR / "family_tree.json"
VAL_PATH = GRAPH_DIR / "val_dataset.parquet"
MODEL_NAME = "Qwen/Qwen3-1.7B"
RESULTS_PATH = GRAPH_DIR / "baseline_results.json"


# ── context builders ──────────────────────────────────────────────────────────

def build_icl_context(tree: FamilyTree) -> str:
    """
    Build a text listing all raw edges (parent_child + spouses).
    Only primary facts — same information the cartridge was trained on.
    Model must derive multi-hop relationships itself.
    """
    lines = ["Family relationships:"]
    for edge in tree.parent_child:
        p, c = edge["parent"], edge["child"]
        p_gender = tree._gender.get(p, "male")
        c_gender = tree._gender.get(c, "male")
        p_label = "father" if p_gender == "male" else "mother"
        c_label = "son" if c_gender == "male" else "daughter"
        lines.append(f"{p} is {c}'s {p_label}.")
        lines.append(f"{c} is {p}'s {c_label}.")
    for pair in tree.spouses:
        a, b = pair["a"], pair["b"]
        a_label = "husband" if tree._gender.get(a, "male") == "male" else "wife"
        b_label = "husband" if tree._gender.get(b, "male") == "male" else "wife"
        lines.append(f"{a} is {b}'s {a_label}.")
        lines.append(f"{b} is {a}'s {b_label}.")
    return "\n".join(lines)


# ── question builder (mirrors GraphRelationshipMCEvalDataset.__init__) ────────

def build_val_items(val_path: Path, seed: int = 42) -> list[dict]:
    """
    Reproduce the same MC options as GraphRelationshipMCEvalDataset with seed=42.
    Returns list of dicts with: question, options, correct_letter, correct_rel, chain_length.
    """
    convs = read_conversations(str(val_path))
    all_rels = sorted(set(c.metadata["final_rel"] for c in convs))
    rng = random.Random(seed)

    items = []
    for conv in convs:
        correct_rel = conv.metadata["final_rel"]
        # same distractor sampling as GraphRelationshipMCEvalDataset
        pool = [r for r in all_rels if r != correct_rel and r != DONT_KNOW]
        distractors = rng.sample(pool, min(3, len(pool)))
        options = distractors + [correct_rel, DONT_KNOW]
        rng.shuffle(options)
        correct_letter = LETTERS[options.index(correct_rel)]
        items.append({
            "person_a": conv.metadata["person_a"],
            "person_b": conv.metadata["person_b"],
            "options": options,
            "correct_letter": correct_letter,
            "correct_rel": correct_rel,
            "chain_length": conv.metadata["chain_length"],
        })
    return items


# ── inference ─────────────────────────────────────────────────────────────────

@torch.inference_mode()
def run_baseline(
    items: list[dict],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    system_prompt: str | None,
    batch_size: int = 1,
    max_new_tokens: int = 512,
) -> list[dict]:
    """
    Run inference on all val items.
    system_prompt=None → zero_shot, system_prompt=str → ICL.
    Returns list of result dicts.
    """
    device = next(model.parameters()).device
    results = []

    for i in tqdm(range(0, len(items), batch_size), desc="Inference"):
        batch = items[i : i + batch_size]
        input_ids_list = []

        for item in batch:
            question = f"What is the relationship between {item['person_a']} and {item['person_b']}?"
            prompt_text = _build_mc_prompt(question, item["options"])

            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt_text})

            input_ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=True,   # consistent with cartridge eval
            )
            input_ids_list.append(input_ids[0])

        # Pad to same length
        max_len = max(t.shape[0] for t in input_ids_list)
        padded = torch.stack([
            torch.nn.functional.pad(t, (max_len - t.shape[0], 0), value=tokenizer.pad_token_id or 0)
            for t in input_ids_list
        ]).to(device)
        attention_mask = (padded != (tokenizer.pad_token_id or 0)).long()

        out = model.generate(
            padded,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,         # greedy
            pad_token_id=tokenizer.eos_token_id,
        )

        for j, (item, inp) in enumerate(zip(batch, input_ids_list)):
            gen_ids = out[j][max_len:]    # strip prompt
            pred = tokenizer.decode(gen_ids, skip_special_tokens=True)
            pred_letter = _extract_answer_letter(pred)
            correct = pred_letter == item["correct_letter"]

            results.append({
                "person_a": item["person_a"],
                "person_b": item["person_b"],
                "chain_length": item["chain_length"],
                "correct_rel": item["correct_rel"],
                "correct_letter": item["correct_letter"],
                "pred_letter": pred_letter,
                "pred": pred[:200],
                "correct": correct,
            })

    return results


# ── scoring ───────────────────────────────────────────────────────────────────

def compute_accuracy(results: list[dict]) -> dict:
    by_hop: dict[int, list[float]] = defaultdict(list)
    for r in results:
        by_hop[r["chain_length"]].append(float(r["correct"]))
    overall = sum(r["correct"] for r in results) / len(results)
    per_hop = {f"hop_{k}": sum(v) / len(v) for k, v in sorted(by_hop.items())}
    return {"overall": overall, **per_hop, "n": len(results)}


def print_accuracy(name: str, acc: dict):
    print(f"\n{'─'*50}")
    print(f"Baseline: {name}")
    print(f"  overall: {acc['overall']:.3f}  (n={acc['n']})")
    for k, v in acc.items():
        if k.startswith("hop_"):
            n_hop = sum(1 for r in [] if r["chain_length"] == int(k.split("_")[1]))
            print(f"  {k}: {v:.3f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="both", choices=["zero_shot", "icl_edges", "both"])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    print(f"Loading val data: {VAL_PATH}")
    items = build_val_items(VAL_PATH, seed=42)
    print(f"  {len(items)} val items")

    tree = FamilyTree.load(TREE_PATH)
    icl_context = build_icl_context(tree)
    icl_token_count = len(tokenizer.encode(icl_context))
    print(f"  ICL context: {len(icl_context.splitlines())} lines, ~{icl_token_count} tokens")

    all_results = {}
    modes = ["zero_shot", "icl_edges"] if args.mode == "both" else [args.mode]

    for mode in modes:
        print(f"\nRunning: {mode}")
        sys_prompt = None if mode == "zero_shot" else icl_context
        results = run_baseline(
            items, model, tokenizer, sys_prompt,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        acc = compute_accuracy(results)
        print_accuracy(mode, acc)
        all_results[mode] = {"accuracy": acc, "samples": results}

    # Save
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved to {RESULTS_PATH}")

    # Summary table
    print(f"\n{'═'*50}")
    print("SUMMARY (overall accuracy)")
    for mode, data in all_results.items():
        print(f"  {mode:15s}: {data['accuracy']['overall']:.3f}")
    print()
    print("Per-hop accuracy:")
    hop_keys = sorted(set(
        k for data in all_results.values()
        for k in data["accuracy"] if k.startswith("hop_")
    ))
    header = f"{'hop':>8}" + "".join(f"  {m:>12}" for m in modes)
    print(header)
    for hop in hop_keys:
        row = f"{hop:>8}"
        for mode in modes:
            v = all_results[mode]["accuracy"].get(hop, float("nan"))
            row += f"  {v:>12.3f}"
        print(row)


if __name__ == "__main__":
    main()
