"""
ICL baselines for the family tree cartridge experiment.

Two baselines, same val questions/options as GraphRelationshipMCEvalDataset (seed=42):

  zero_shot  — no family tree context. Controls for model prior (~20% random expected).
  icl_edges  — system prompt = raw edges from family_tree.json (parent_child + spouses).
               Model must reason over the raw graph to answer. Honest ICL upper bound:
               same facts the cartridge was trained on, just in-context.

W&B logging mirrors cartridge eval exactly:
  generate_graph_accuracy/hop_1_score
  generate_graph_accuracy/hop_2_score
  ...
  generate_graph_accuracy/table
  train/optimizer_step = 0

Each baseline mode = one W&B run for direct comparison with cartridge training runs.

Usage:
    python examples/graph/baseline_graph.py
    python examples/graph/baseline_graph.py --mode zero_shot
    python examples/graph/baseline_graph.py --mode icl_edges --batch_size 4
    python examples/graph/baseline_graph.py --no_wandb
"""
import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parents[2]))

from examples.graph.family_tree import FamilyTree
from examples.graph.graph_evals import (
    DONT_KNOW,
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
WANDB_NAME_FOR_WANDB = "graph_accuracy"   # default; override via --name_for_wandb


# ── context builders ──────────────────────────────────────────────────────────

def build_icl_context(tree: FamilyTree) -> str:
    """
    Raw edges from family_tree.json (parent_child + spouses) as readable text.
    Sibling relationships deliberately excluded — model must derive them itself,
    same as the cartridge did during training.
    """
    lines = ["Family relationships:"]
    for edge in tree.parent_child:
        p, c = edge["parent"], edge["child"]
        p_label = "father" if tree._gender.get(p, "male") == "male" else "mother"
        c_label = "son" if tree._gender.get(c, "male") == "male" else "daughter"
        lines.append(f"{p} is {c}'s {p_label}.")
        lines.append(f"{c} is {p}'s {c_label}.")
    for pair in tree.spouses:
        a, b = pair["a"], pair["b"]
        a_label = "husband" if tree._gender.get(a, "male") == "male" else "wife"
        b_label = "husband" if tree._gender.get(b, "male") == "male" else "wife"
        lines.append(f"{a} is {b}'s {a_label}.")
        lines.append(f"{b} is {a}'s {b_label}.")
    return "\n".join(lines)


# ── val item builder (mirrors GraphRelationshipMCEvalDataset.__init__) ─────────

def build_val_items(val_path: Path, seed: int = 42) -> list[dict]:
    """
    Reproduce same MC options as GraphRelationshipMCEvalDataset(seed=42).
    Same distractors → results are directly comparable to cartridge eval.
    """
    convs = read_conversations(str(val_path))
    all_rels = sorted(set(c.metadata["final_rel"] for c in convs))
    rng = random.Random(seed)

    items = []
    for conv in convs:
        correct_rel = conv.metadata["final_rel"]
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
    device = next(model.parameters()).device
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    results = []

    for i in tqdm(range(0, len(items), batch_size), desc="Inference"):
        batch = items[i : i + batch_size]
        input_ids_list = []

        for item in batch:
            question = (
                f"What is the relationship between "
                f"{item['person_a']} and {item['person_b']}?"
            )
            prompt_text = _build_mc_prompt(question, item["options"])
            messages = []
            if system_prompt is not None:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt_text})

            ids = tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_tensors="pt",
                enable_thinking=True,   # same as cartridge eval
            )
            input_ids_list.append(ids[0])

        max_len = max(t.shape[0] for t in input_ids_list)
        padded = torch.stack([
            torch.nn.functional.pad(t, (max_len - t.shape[0], 0), value=pad_id)
            for t in input_ids_list
        ]).to(device)
        attention_mask = (padded != pad_id).long()

        out = model.generate(
            padded,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )

        for j, item in enumerate(batch):
            gen_ids = out[j][max_len:]
            pred = tokenizer.decode(gen_ids, skip_special_tokens=True)
            pred_letter = _extract_answer_letter(pred)
            correct = pred_letter == item["correct_letter"]

            results.append({
                "index": i + j,
                "person_a": item["person_a"],
                "person_b": item["person_b"],
                "chain_length": item["chain_length"],
                "correct_rel": item["correct_rel"],
                "correct_letter": item["correct_letter"],
                "pred_letter": pred_letter,
                "pred": pred[:300],
                # score column — same naming as cartridge eval (endswith "score")
                f"hop_{item['chain_length']}_score": float(correct),
            })

    return results


# ── wandb logging ─────────────────────────────────────────────────────────────

def log_to_wandb(mode: str, results: list[dict], name_for_wandb: str, icl_token_count: int | None = None):
    """
    Log metrics mirroring cartridge evaluate_generations() output exactly:
      generate_graph_accuracy/hop_N_score  (avg accuracy per chain length)
      generate_graph_accuracy/table         (full predictions dataframe)
      train/optimizer_step = 0              (baseline = step 0 reference)
    """
    import wandb

    project = os.environ.get("CARTRIDGES_WANDB_PROJECT", "cartridges")
    entity = os.environ.get("CARTRIDGES_WANDB_ENTITY", None)
    prefix = f"generate_{name_for_wandb}"

    run_config = {
        "baseline_mode": mode,
        "model": MODEL_NAME,
        "val_path": str(VAL_PATH),
        "num_val_items": len(results),
    }
    if icl_token_count is not None:
        run_config["icl_context_tokens"] = icl_token_count

    wandb.init(
        project=project,
        entity=entity,
        name=f"graph_baseline_{mode}_{name_for_wandb}",
        tags=["baseline", "graph", mode],
        config=run_config,
    )

    df = pd.DataFrame(results)
    score_cols = [c for c in df.columns if c.endswith("_score")]
    avg_scores = {f"{prefix}/{col}": df[col].mean() for col in score_cols}

    log_dict = {
        **avg_scores,
        f"{prefix}/table": wandb.Table(dataframe=df),
        f"{prefix}/num_system_and_user_tokens": 0,
        "train/optimizer_step": 0,
    }
    wandb.log(log_dict, step=0)
    wandb.finish()

    return avg_scores


# ── scoring summary ───────────────────────────────────────────────────────────

def summarize(mode: str, results: list[dict]):
    by_hop: dict[int, list[float]] = defaultdict(list)
    for r in results:
        hop = r["chain_length"]
        score = r.get(f"hop_{hop}_score", 0.0)
        by_hop[hop].append(score)

    overall = sum(
        r.get(f"hop_{r['chain_length']}_score", 0.0) for r in results
    ) / len(results)

    print(f"\n{'─'*50}")
    print(f"Baseline: {mode}  (n={len(results)}, overall={overall:.3f})")
    for hop, scores in sorted(by_hop.items()):
        print(f"  hop_{hop}: {sum(scores)/len(scores):.3f}  (n={len(scores)})")

    return {
        "overall": overall,
        **{
            f"hop_{hop}": sum(scores) / len(scores)
            for hop, scores in sorted(by_hop.items())
        },
        "n": len(results),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", default="both",
        choices=["zero_shot", "icl_edges", "both"],
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--no_wandb", action="store_true")
    parser.add_argument(
        "--name_for_wandb", default=WANDB_NAME_FOR_WANDB,
        help="W&B metric prefix — must match name_for_wandb in the experiment's GenerationEvalConfig",
    )
    parser.add_argument(
        "--val_path", default=str(VAL_PATH),
        help="Path to val parquet (use val_dataset_ctx.parquet for Exp 3)",
    )
    args = parser.parse_args()

    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    val_path = Path(args.val_path)
    print(f"Loading val items from {val_path}")
    items = build_val_items(val_path, seed=42)
    print(f"  {len(items)} items")

    tree = FamilyTree.load(TREE_PATH)
    icl_context = build_icl_context(tree)
    icl_token_count = len(tokenizer.encode(icl_context))
    print(f"  ICL context: {icl_token_count} tokens")

    modes = ["zero_shot", "icl_edges"] if args.mode == "both" else [args.mode]
    all_results: dict[str, dict] = {}

    for mode in modes:
        sys_prompt = None if mode == "zero_shot" else icl_context
        print(f"\nRunning baseline: {mode}")
        results = run_baseline(
            items, model, tokenizer, sys_prompt,
            batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens,
        )
        acc = summarize(mode, results)
        all_results[mode] = {"accuracy": acc, "samples": results}

        if not args.no_wandb:
            avg = log_to_wandb(
                mode, results,
                name_for_wandb=args.name_for_wandb,
                icl_token_count=icl_token_count if mode == "icl_edges" else None,
            )
            print(f"  W&B logged: {avg}")

    # Save to JSON
    save_data = {
        m: {"accuracy": d["accuracy"], "n_samples": len(d["samples"])}
        for m, d in all_results.items()
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved summary to {RESULTS_PATH}")

    # Comparison table
    print(f"\n{'═'*55}")
    print("SUMMARY — compare to cartridge W&B runs")
    print(f"  W&B metric prefix: generate_{args.name_for_wandb}/")
    hop_keys = sorted(set(
        k for d in all_results.values()
        for k in d["accuracy"] if k.startswith("hop_")
    ))
    header = f"{'':>8}" + "".join(f"  {m:>14}" for m in modes)
    print(header)
    print(f"{'overall':>8}" + "".join(
        f"  {all_results[m]['accuracy']['overall']:>14.3f}" for m in modes
    ))
    for h in hop_keys:
        row = f"{h:>8}"
        for m in modes:
            v = all_results[m]["accuracy"].get(h, float("nan"))
            row += f"  {v:>14.3f}"
        print(row)


if __name__ == "__main__":
    main()
