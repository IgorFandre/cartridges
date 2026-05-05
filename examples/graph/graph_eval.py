"""
Evaluate trained cartridge or ICL baseline on kinship QA test set.

Run graph_qagen.py first to generate test.parquet and test_meta.json.

Usage:
    # Cartridge (trained):
    python examples/graph/graph_eval.py --mode cartridge --checkpoint /path/to/cache-last.pt

    # ICL baseline (family tree text in system prompt):
    python examples/graph/graph_eval.py --mode icl
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from cartridges.structs import read_conversations

OUTPUT_DIR = Path(__file__).parent
CORPUS_PATH = OUTPUT_DIR / "family_tree_corpus.txt"
TEST_PARQUET = OUTPUT_DIR / "test.parquet"
TEST_META    = OUTPUT_DIR / "test_meta.json"

MODEL_CONFIGS = {
    "qwen1.7b": "Qwen/Qwen3-1.7B",
    "qwen4b":   "Qwen/Qwen3-4b",
}


def score_answer(pred: str, expected: str) -> bool:
    pred     = pred.strip().rstrip(".").strip().lower()
    expected = expected.strip().rstrip(".").strip().lower()
    return pred == expected


def build_inputs(questions: List[str], tokenizer, system_prompt: str | None, device: str):
    """Build batched input tensors for flex_generate (packed sequences)."""
    from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE, MODELS_WITH_THINKING

    kwargs = {}
    if tokenizer.name_or_path in MODELS_WITH_THINKING:
        kwargs["enable_thinking"] = False

    input_ids_list = []
    for q in questions:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": q})

        ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=MODEL_TO_CHAT_TEMPLATE.get(tokenizer.name_or_path),
            **kwargs,
        )
        input_ids_list.append(ids)

    input_ids   = torch.cat([ids[0] for ids in input_ids_list]).to(device)
    seq_ids     = torch.cat([
        torch.full((ids.shape[1],), j, dtype=torch.long, device=device)
        for j, ids in enumerate(input_ids_list)
    ])
    position_ids = torch.cat([
        torch.arange(ids.shape[1], device=device)
        for ids in input_ids_list
    ])
    return input_ids, seq_ids, position_ids


def run_cartridge_eval(args) -> List[dict]:
    from cartridges.cache import TrainableCache
    from cartridges.generation import flex_generate
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

    model_name = MODEL_CONFIGS[args.model]
    device = args.device

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = FlexQwen3ForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    cache: TrainableCache = TrainableCache.from_pretrained(args.checkpoint, device=device)
    cache = cache.to(device)

    convos = read_conversations(str(TEST_PARQUET))
    meta   = json.loads(TEST_META.read_text())
    assert len(convos) == len(meta), "test.parquet and test_meta.json out of sync"

    results = []
    for i in tqdm(range(0, len(convos), args.batch_size), desc="cartridge eval"):
        batch_convos = convos[i : i + args.batch_size]
        batch_meta   = meta[i : i + args.batch_size]

        questions = [c.messages[0].content for c in batch_convos]
        expected  = [c.messages[1].content for c in batch_convos]

        input_ids, seq_ids, position_ids = build_inputs(
            questions, tokenizer, system_prompt=None, device=device
        )

        with torch.no_grad():
            pred_ids: Dict[int, List[int]] = flex_generate(
                model=model,
                tokenizer=tokenizer,
                input_ids=input_ids,
                seq_ids=seq_ids,
                position_ids=position_ids,
                cache=cache,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                show_progress=False,
            )

        for j, (exp, m) in enumerate(zip(expected, batch_meta)):
            pred_text = tokenizer.decode(pred_ids[j], skip_special_tokens=True)
            results.append({
                "category":  m["category"],
                "rel":       m["rel"],
                "person":    m["person"],
                "question":  m["question"],
                "expected":  exp,
                "predicted": pred_text,
                "correct":   score_answer(pred_text, exp),
            })

    return results


def run_icl_eval(args) -> List[dict]:
    from transformers import AutoModelForCausalLM
    from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE, MODELS_WITH_THINKING

    model_name = MODEL_CONFIGS[args.model]
    device = args.device

    corpus_text   = CORPUS_PATH.read_text()
    system_prompt = (
        "Use the following family tree to answer questions.\n\n"
        + corpus_text
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    convos = read_conversations(str(TEST_PARQUET))
    meta   = json.loads(TEST_META.read_text())
    assert len(convos) == len(meta), "test.parquet and test_meta.json out of sync"

    kwargs = {}
    if model_name in MODELS_WITH_THINKING:
        kwargs["enable_thinking"] = False

    results = []
    for i in tqdm(range(len(convos)), desc="ICL eval"):
        question = convos[i].messages[0].content
        expected = convos[i].messages[1].content
        m        = meta[i]

        input_ids = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": question},
            ],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=MODEL_TO_CHAT_TEMPLATE.get(model_name),
            **kwargs,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
            )

        pred_text = tokenizer.decode(
            output_ids[0][input_ids.shape[1]:], skip_special_tokens=True
        )
        results.append({
            "category":  m["category"],
            "rel":       m["rel"],
            "person":    m["person"],
            "question":  m["question"],
            "expected":  expected,
            "predicted": pred_text,
            "correct":   score_answer(pred_text, expected),
        })

    return results


def print_results(results: List[dict], mode: str):
    total     = len(results)
    n_correct = sum(r["correct"] for r in results)
    print(f"\n=== {mode.upper()} ===")
    print(f"Overall  {n_correct:4d}/{total} = {n_correct/total:.1%}")

    for cat in [1, 2, 3]:
        cat_r = [r for r in results if r["category"] == cat]
        if not cat_r:
            continue
        nc = sum(r["correct"] for r in cat_r)
        print(f"  Cat {cat}  {nc:4d}/{len(cat_r)} = {nc/len(cat_r):.1%}")

    rels = sorted(set(r["rel"] for r in results))
    print("\nPer relation:")
    for rel in rels:
        rel_r = [r for r in results if r["rel"] == rel]
        nc    = sum(r["correct"] for r in rel_r)
        print(f"  {rel:20s} {nc:4d}/{len(rel_r)} = {nc/len(rel_r):.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",           required=True, choices=["cartridge", "icl"])
    parser.add_argument("--checkpoint",     default=None,  help="Path to .pt cache (cartridge mode)")
    parser.add_argument("--model",          default="qwen1.7b", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size",     type=int, default=8)
    parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output",         default=None, help="Save per-question results JSON")
    args = parser.parse_args()

    if args.mode == "cartridge":
        assert args.checkpoint, "--checkpoint required for cartridge mode"
        results = run_cartridge_eval(args)
    else:
        results = run_icl_eval(args)

    print_results(results, args.mode)

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
