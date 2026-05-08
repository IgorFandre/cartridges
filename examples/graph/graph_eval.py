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
CORPUS_PATH      = OUTPUT_DIR / "family_tree_corpus.txt"
CORPUS_JSON_PATH = OUTPUT_DIR / "family_tree.json"
TEST_PARQUET     = OUTPUT_DIR / "test.parquet"
TEST_META        = OUTPUT_DIR / "test_meta.json"
TEST_PARQUET_COT = OUTPUT_DIR / "test_cot.parquet"
TEST_META_COT    = OUTPUT_DIR / "test_meta_cot.json"
TRAIN_PARQUET    = OUTPUT_DIR / "train.parquet"
TRAIN_META       = OUTPUT_DIR / "train_meta.json"

MODEL_CONFIGS = {
    "qwen1.7b": "Qwen/Qwen3-1.7B",
    "qwen4b":   "Qwen/Qwen3-4b",
}


def score_answer(pred: str, expected: str) -> bool:
    pred     = pred.strip().rstrip(".").strip().lower()
    expected = expected.strip().rstrip(".").strip().lower()
    return pred == expected


def extract_final_answer(text: str) -> str:
    """Extract 'X' from CoT text ending with 'Answer: X.'"""
    import re
    m = re.search(r'[Aa]nswer:\s*(.+?)\.?\s*$', text.strip(), re.DOTALL)
    if m:
        return m.group(1).strip()
    return text.strip()


def score_cot_answer(pred: str, expected: str) -> bool:
    pred_ans = extract_final_answer(pred)
    exp_ans  = extract_final_answer(expected)
    return score_answer(pred_ans, exp_ans)


def build_inputs(questions: List[str], tokenizer, system_prompt: str | None, device: str):
    """Build batched input tensors for flex_generate (packed sequences)."""
    from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE, MODELS_WITH_THINKING

    is_thinking_model = tokenizer.name_or_path.lower() in {m.lower() for m in MODELS_WITH_THINKING}
    kwargs = {}
    if is_thinking_model:
        kwargs["enable_thinking"] = False

    no_think_ids = (
        tokenizer.encode("<think>\n</think>\n", add_special_tokens=False)
        if is_thinking_model else []
    )

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
        if no_think_ids:
            ids = torch.cat([ids, torch.tensor([no_think_ids])], dim=1)
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

    convos = read_conversations(str(args._test_parquet))
    meta   = json.loads(Path(args._test_meta).read_text())
    assert len(convos) == len(meta), "parquet and meta out of sync"
    scorer = args._scorer

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
                temperature=args.temperature,
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
                "correct":   scorer(pred_text, exp),
            })

    return results


def _find_sys_prefix_len(tokenizer, system_prompt: str, chat_template, kwargs: dict) -> int:
    """
    Find token count of the system prompt prefix (before user content starts).
    Uses a sentinel string to locate the exact split point.
    """
    SENTINEL = "KINSHIP_EVAL_SENTINEL_XYZ_99999"
    sentinel_ids = tokenizer.encode(SENTINEL, add_special_tokens=False)
    assert len(sentinel_ids) > 0, "Sentinel tokenized to empty — pick a different string"

    full_ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": SENTINEL}],
        add_generation_prompt=False,
        return_tensors=None,
        chat_template=chat_template,
        **kwargs,
    )

    # Find first occurrence of sentinel token sequence
    n = len(sentinel_ids)
    for j in range(len(full_ids) - n + 1):
        if full_ids[j:j + n] == sentinel_ids:
            assert j > 0, f"Sentinel found at position 0 — system prefix is empty"
            return j
    raise ValueError("Could not locate sentinel in tokenized template output")


def run_icl_eval(args) -> List[dict]:
    from transformers import AutoModelForCausalLM
    from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE, MODELS_WITH_THINKING

    model_name = MODEL_CONFIGS[args.model]
    device = args.device

    if args.icl_format == "json":
        corpus_text = CORPUS_JSON_PATH.read_text()
    else:
        corpus_text = CORPUS_PATH.read_text()
    few_shot_block = ""
    if args.n_shot > 0 and TRAIN_PARQUET.exists():
        import random
        train_convos = read_conversations(str(TRAIN_PARQUET))
        k = min(args.n_shot, len(train_convos))
        samples = random.sample(train_convos, k)
        lines = []
        for s in samples:
            q = s.messages[0].content.strip()
            a = s.messages[1].content.strip()
            if args.cot:
                lines.append(f"Q: {q}\nA: {a}")
            else:
                lines.append(f"Q: {q}\nA: {a}")
        few_shot_block = "\n\nExamples:\n" + "\n\n".join(lines) + "\n\nNow answer:"

    if args.cot:
        system_prompt = (
            "Use the following family tree to answer questions.\n\n"
            + corpus_text
            + "\n\nReason step by step using the family tree, then end your answer with 'Answer: <answer>.' where <answer> is the name(s) comma-separated or a number."
            + few_shot_block
        )
    else:
        system_prompt = (
            "Use the following family tree to answer questions.\n\n"
            + corpus_text
            + "\n\nAnswer concisely. For name questions output only the name(s) comma-separated followed by a period (e.g. 'Alice, Bob.'). For counting questions output only the number followed by a period (e.g. '3.'). No explanation."
            + few_shot_block
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    convos = read_conversations(str(args._test_parquet))
    meta   = json.loads(Path(args._test_meta).read_text())
    assert len(convos) == len(meta), "parquet and meta out of sync"
    scorer = args._scorer

    kwargs = {}
    if model_name.lower() in {m.lower() for m in MODELS_WITH_THINKING}:
        kwargs["enable_thinking"] = False

    chat_template = MODEL_TO_CHAT_TEMPLATE.get(model_name)

    # Cache system prompt KV once — O(1) prefill instead of O(N)
    sys_prefix_len = _find_sys_prefix_len(tokenizer, system_prompt, chat_template, kwargs)
    sys_input_ids  = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": ""}],
        add_generation_prompt=False,
        return_tensors="pt",
        chat_template=chat_template,
        **kwargs,
    )[:, :sys_prefix_len].to(device)

    with torch.no_grad():
        sys_past_kv = model(sys_input_ids, use_cache=True).past_key_values
    print(f"System prefix cached: {sys_prefix_len} tokens")

    import copy

    results = []
    for i in tqdm(range(len(convos)), desc="ICL eval"):
        question = convos[i].messages[0].content
        expected = convos[i].messages[1].content
        m        = meta[i]

        full_ids = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": question},
            ],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=chat_template,
            **kwargs,
        ).to(device)

        # Force no-thinking: append empty <think></think> block so model skips reasoning
        if kwargs.get("enable_thinking") is False:
            no_think_ids = torch.tensor(
                [tokenizer.encode("<think>\n</think>\n", add_special_tokens=False)],
                device=device,
            )
            full_ids = torch.cat([full_ids, no_think_ids], dim=1)

        do_sample = args.temperature > 0
        with torch.no_grad():
            output_ids = model.generate(
                full_ids,
                past_key_values=copy.deepcopy(sys_past_kv),
                max_new_tokens=args.max_new_tokens,
                do_sample=do_sample,
                temperature=args.temperature if do_sample else None,
                top_p=None,
            )

        pred_text = tokenizer.decode(
            output_ids[0][full_ids.shape[1]:], skip_special_tokens=True
        )
        if i < 3:
            print(f"\n[DEBUG Q{i}] {convos[i].messages[0].content}")
            print(f"[DEBUG expected] {expected!r}")
            print(f"[DEBUG predicted] {pred_text!r}")
        results.append({
            "category":  m["category"],
            "rel":       m["rel"],
            "person":    m["person"],
            "question":  m["question"],
            "expected":  expected,
            "predicted": pred_text,
            "correct":   scorer(pred_text, expected),
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


def print_stability(all_results: List[List[dict]], mode: str):
    import numpy as np
    accs = [sum(r["correct"] for r in run) / len(run) for run in all_results]
    print(f"\n=== {mode.upper()} STABILITY ({len(all_results)} runs) ===")
    print(f"Accuracy per run: {[f'{a:.1%}' for a in accs]}")
    print(f"Mean: {np.mean(accs):.1%}  Std: {np.std(accs):.1%}  Min: {min(accs):.1%}  Max: {max(accs):.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",           required=True, choices=["cartridge", "icl"])
    parser.add_argument("--checkpoint",     default=None,  help="Path to .pt cache (cartridge mode)")
    parser.add_argument("--model",          default="qwen1.7b", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--batch-size",     type=int, default=8)
    parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output",         default=None, help="Save per-question results JSON")
    parser.add_argument("--cot",            action="store_true", help="Use CoT test set (test_cot.parquet)")
    parser.add_argument("--temperature",    type=float, default=0.0)
    parser.add_argument("--n-runs",         type=int,   default=1, help="Repeat eval N times (stability check)")
    parser.add_argument("--icl-format",     default="text", choices=["text", "json"], help="ICL context format")
    parser.add_argument("--n-shot",         type=int,   default=0, help="Few-shot examples from train.parquet in ICL system prompt")
    args = parser.parse_args()

    # Route to correct parquet/meta/scorer
    if args.cot:
        args._test_parquet = TEST_PARQUET_COT
        args._test_meta    = TEST_META_COT
        args._scorer       = score_cot_answer
        if args.max_new_tokens == 64:
            args.max_new_tokens = 256
    else:
        args._test_parquet = TEST_PARQUET
        args._test_meta    = TEST_META
        args._scorer       = score_answer

    run_fn = run_cartridge_eval if args.mode == "cartridge" else run_icl_eval
    if args.mode == "cartridge":
        assert args.checkpoint, "--checkpoint required for cartridge mode"

    all_results = []
    for run_idx in range(args.n_runs):
        if args.n_runs > 1:
            print(f"\n--- Run {run_idx + 1}/{args.n_runs} ---")
        results = run_fn(args)
        all_results.append(results)
        print_results(results, args.mode)

    if args.n_runs > 1:
        print_stability(all_results, args.mode)

    if args.output:
        out = all_results[0] if args.n_runs == 1 else all_results
        Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False))
        print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
