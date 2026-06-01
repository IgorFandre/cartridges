"""
Evaluate trained cartridge or ICL baseline on kinship MC (5-option A-E) test set.

Run graph_qagen.py and/or graph_qagen_cot.py first to generate:
    test_mc.parquet, test_meta_mc.json          (NTP letter target)
    test_cot_mc.parquet, test_meta_cot_mc.json  (CoT then letter)

Four eval modes:
    icl       : ICL, strict letter answer (single token)
    icl-cot   : ICL, <think>...</think> then letter
    cartridge     : trained cartridge, NTP letter (use --checkpoint from graph_train.py)
    cartridge-cot : trained cartridge, CoT then letter (use --checkpoint from graph_train_cot.py)

Usage:
    python examples/graph/graph_eval.py --mode icl
    python examples/graph/graph_eval.py --mode icl-cot
    python examples/graph/graph_eval.py --mode cartridge     --checkpoint /path/to/cache-last.pt
    python examples/graph/graph_eval.py --mode cartridge-cot --checkpoint /path/to/cache-last.pt
"""
import argparse
import copy
import json
import re
from pathlib import Path
from typing import Dict, List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from cartridges.structs import read_conversations

from examples.graph.paths import (
    BASE_CORPUS, BASE_TREE_JSON, BASE_NARRATIVE,
    BASE_TEST_PARQUET, BASE_TEST_META,
    BASE_TRAIN_PARQUET, BASE_TRAIN_META,
)

# Base-tree defaults (override on the CLI for variants).
CORPUS_PATH      = BASE_CORPUS
CORPUS_JSON_PATH = BASE_TREE_JSON
NARRATIVE_PATH   = BASE_NARRATIVE
TEST_MC_PARQUET  = BASE_TEST_PARQUET
TEST_MC_META     = BASE_TEST_META
TRAIN_MC_PARQUET = BASE_TRAIN_PARQUET
TRAIN_MC_META    = BASE_TRAIN_META

MODEL_CONFIGS = {
    "qwen1.7b": "Qwen/Qwen3-1.7B",
    "qwen4b":   "Qwen/Qwen3-4b",
}

VALID_LETTERS = ("A", "B", "C", "D", "E")


# ── Letter extraction ────────────────────────────────────────────────────────
def extract_letter(text: str, n_options: int = 5) -> str:
    """Pull a valid option letter (A..A+n-1) from a model answer.

    For CoT outputs the answer follows the reasoning block, so we scan only the
    region after ``</think>``. An *unclosed* ``<think>`` means generation was
    truncated before any answer was emitted → return "" (scored wrong, surfaced
    as malformed by analyze.py). Reasoning prose is never scanned, so the
    indefinite article "a" can no longer be mistaken for option A.
    """
    t = text or ""
    if "</think>" in t:
        region = t.rsplit("</think>", 1)[1]      # answer lives after the think block
    elif "<think>" in t:
        return ""                                 # truncated mid-reasoning, no answer
    else:
        region = t                                # non-CoT: output is the answer itself
    s = region.strip()
    valid = "ABCDE"[:n_options]
    m = re.search(rf"(?:answer\s*[:\-]?\s*)?\b([{valid}])\b\.?", s, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m2 = re.search(f"[{valid}{valid.lower()}]", s)
    return m2.group(0).upper() if m2 else ""


def score_letter(pred: str, expected_letter: str, n_options: int = 5) -> bool:
    return extract_letter(pred, n_options=n_options) == expected_letter.upper()


# ── Input building (cartridge mode) ──────────────────────────────────────────
def build_inputs(questions: List[str], tokenizer, system_prompt: str | None, device: str,
                 enable_thinking: bool):
    """Build batched packed input tensors for flex_generate."""
    from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE, MODELS_WITH_THINKING

    is_thinking_model = tokenizer.name_or_path.lower() in {m.lower() for m in MODELS_WITH_THINKING}
    kwargs = {}
    if is_thinking_model:
        kwargs["enable_thinking"] = enable_thinking

    # Force-empty <think> block for non-CoT NTP cartridge eval
    no_think_ids = []
    if is_thinking_model and not enable_thinking:
        no_think_ids = tokenizer.encode("<think>\n</think>\n", add_special_tokens=False)

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


# ── Cartridge eval ───────────────────────────────────────────────────────────
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

    results = []
    for i in tqdm(range(0, len(convos), args.batch_size), desc=f"{args.mode} eval"):
        batch_convos = convos[i : i + args.batch_size]
        batch_meta   = meta[i : i + args.batch_size]

        questions = [c.messages[0].content for c in batch_convos]

        input_ids, seq_ids, position_ids = build_inputs(
            questions, tokenizer, system_prompt=None, device=device,
            enable_thinking=args._cot,
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

        for j, m in enumerate(batch_meta):
            pred_text = tokenizer.decode(pred_ids[j], skip_special_tokens=True)
            results.append({
                "category":        m["category"],
                "rel":             m["rel"],
                "person":          m["person"],
                "question":        m.get("question_text", m.get("question", "")),
                "options":         m["options"],
                "n_options":       m.get("n_options", len(m["options"])),
                "correct_letter":  m["correct_letter"],
                "predicted":       pred_text,
                "predicted_letter": extract_letter(pred_text, n_options=m.get("n_options", len(m["options"]))),
                "correct":         score_letter(pred_text, m["correct_letter"], n_options=m.get("n_options", len(m["options"]))),
            })

    return results


# ── ICL eval ─────────────────────────────────────────────────────────────────
def _find_sys_prefix_len(tokenizer, system_prompt: str, chat_template, kwargs: dict) -> int:
    SENTINEL = "KINSHIP_EVAL_SENTINEL_XYZ_99999"
    sentinel_ids = tokenizer.encode(SENTINEL, add_special_tokens=False)
    assert len(sentinel_ids) > 0

    full_ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt}, {"role": "user", "content": SENTINEL}],
        add_generation_prompt=False,
        return_tensors=None,
        chat_template=chat_template,
        **kwargs,
    )
    n = len(sentinel_ids)
    for j in range(len(full_ids) - n + 1):
        if full_ids[j:j + n] == sentinel_ids:
            assert j > 0
            return j
    raise ValueError("Sentinel not found")


def run_icl_eval(args) -> List[dict]:
    from transformers import AutoModelForCausalLM
    from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE, MODELS_WITH_THINKING

    model_name = MODEL_CONFIGS[args.model]
    device = args.device

    if args.corpus_path:
        corpus_text = Path(args.corpus_path).read_text()
    elif args.icl_format == "json":
        corpus_text = CORPUS_JSON_PATH.read_text()
    elif args.icl_format == "narrative":
        corpus_text = NARRATIVE_PATH.read_text()
    else:
        corpus_text = CORPUS_PATH.read_text()

    # Few-shot block (drawn from corresponding MC train set)
    few_shot_block = ""
    train_pq        = TRAIN_MC_PARQUET
    train_meta_path = TRAIN_MC_META
    if args.n_shot > 0 and train_pq.exists() and train_meta_path.exists():
        import random
        rng = random.Random(args.n_shot_seed)
        train_convos = read_conversations(str(train_pq))
        train_meta   = json.loads(train_meta_path.read_text())
        print(f"Few-shot source: {train_pq.name}")

        by_rel: dict[str, list[int]] = {}
        for idx, m in enumerate(train_meta):
            by_rel.setdefault(m["rel"], []).append(idx)
        rel_keys = sorted(by_rel.keys())
        one_per_rel = [rng.choice(sorted(by_rel[r])) for r in rel_keys]
        rng.shuffle(one_per_rel)
        selected = one_per_rel[: args.n_shot]

        lines = []
        for idx in selected:
            q = train_convos[idx].messages[0].content.strip()
            a = train_convos[idx].messages[1].content.strip()
            lines.append(f"Q: {q}\nA: {a}")
        few_shot_block = "\n\nExamples:\n" + "\n\n".join(lines) + "\n\nNow answer:"
        print(f"Few-shot: {len(selected)} examples, seed={args.n_shot_seed}")

    if args._cot:
        instruction = (
            "Use the family tree below to answer the multiple-choice question. "
            "Reason step-by-step inside <think>...</think>, then output exactly one letter A/B/C/D/E "
            "followed by a period (e.g. 'B.'). Output nothing after the letter."
        )
    else:
        instruction = (
            "Use the family tree below to answer the multiple-choice question. "
            "Output exactly one letter A/B/C/D/E followed by a period (e.g. 'B.'). "
            "No explanation, no extra tokens."
        )

    system_prompt = (
        instruction
        + "\n\nFamily tree:\n"
        + corpus_text
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

    is_thinking_model = model_name.lower() in {m.lower() for m in MODELS_WITH_THINKING}
    kwargs = {}
    if is_thinking_model:
        kwargs["enable_thinking"] = bool(args._cot)
    chat_template = MODEL_TO_CHAT_TEMPLATE.get(model_name)

    if args.print_prompt:
        print("=" * 80)
        head, _, tail = system_prompt.partition(corpus_text)
        print(head + f"[<corpus omitted: {len(corpus_text)} chars>]" + tail)
        print("=" * 80)

    # Cache system prefix once
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

    results = []
    for i in tqdm(range(len(convos)), desc=f"{args.mode} eval"):
        question = convos[i].messages[0].content
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

        # Force empty think block for strict-letter (non-CoT) eval on a thinking model
        if is_thinking_model and not args._cot:
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
            print(f"\n[DEBUG Q{i}] {question}")
            print(f"[DEBUG expected letter] {m['correct_letter']}")
            print(f"[DEBUG predicted] {pred_text!r}")

        n_opt = m.get("n_options", len(m["options"]))
        results.append({
            "category":        m["category"],
            "rel":             m["rel"],
            "person":          m["person"],
            "question":        m.get("question_text", m.get("question", "")),
            "options":         m["options"],
            "n_options":       n_opt,
            "correct_letter":  m["correct_letter"],
            "predicted":       pred_text,
            "predicted_letter": extract_letter(pred_text, n_options=n_opt),
            "correct":         score_letter(pred_text, m["correct_letter"], n_options=n_opt),
        })

    return results


# ── Reporting ────────────────────────────────────────────────────────────────
def print_results(results: List[dict], mode: str):
    total     = len(results)
    n_correct = sum(r["correct"] for r in results)
    print(f"\n=== {mode.upper()} ===")
    print(f"Overall  {n_correct:4d}/{total} = {n_correct/total:.1%}")

    cats = sorted({str(r["category"]) for r in results}, key=lambda c: (c[0], c))
    for cat in cats:
        cat_r = [r for r in results if str(r["category"]) == cat]
        if not cat_r:
            continue
        nc = sum(r["correct"] for r in cat_r)
        print(f"  Cat {cat:>3}  {nc:4d}/{len(cat_r)} = {nc/len(cat_r):.1%}")

    rels = sorted(set(r["rel"] for r in results))
    print("\nPer relation:")
    for rel in rels:
        rel_r = [r for r in results if r["rel"] == rel]
        nc    = sum(r["correct"] for r in rel_r)
        print(f"  {rel:20s} {nc:4d}/{len(rel_r)} = {nc/len(rel_r):.1%}")

    # Letter distribution sanity check
    from collections import Counter
    pred_letters = Counter(r["predicted_letter"] or "?" for r in results)
    exp_letters  = Counter(r["correct_letter"] for r in results)
    print(f"\nPredicted letter dist: {dict(pred_letters)}")
    print(f"Correct  letter dist: {dict(exp_letters)}")


def print_stability(all_results: List[List[dict]], mode: str):
    import numpy as np
    accs = [sum(r["correct"] for r in run) / len(run) for run in all_results]
    print(f"\n=== {mode.upper()} STABILITY ({len(all_results)} runs) ===")
    print(f"Per run: {[f'{a:.1%}' for a in accs]}")
    print(f"Mean: {np.mean(accs):.1%}  Std: {np.std(accs):.1%}  Min: {min(accs):.1%}  Max: {max(accs):.1%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True,
                        choices=["icl", "icl-cot", "cartridge", "cartridge-cot"])
    parser.add_argument("--checkpoint",     default=None, help="Path to .pt cache (cartridge modes)")
    parser.add_argument("--model",          default="qwen1.7b", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--max-new-tokens", type=int, default=None,
                        help="Default: 8 for letter-only modes, 1024 for CoT modes")
    parser.add_argument("--batch-size",     type=int, default=8)
    parser.add_argument("--device",         default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output",         default=None, help="Save per-question results JSON")
    parser.add_argument("--temperature",    type=float, default=0.0)
    parser.add_argument("--n-runs",         type=int,   default=1)
    parser.add_argument("--icl-format",     default="text",
                        choices=["text", "json", "narrative"],
                        help="ICL corpus form: text=family_tree_corpus.txt, json=family_tree.json, narrative=family_tree_narrative.txt")
    parser.add_argument("--corpus-path",    default=None,
                        help="Override ICL corpus path (e.g. variants/alex/family_tree_corpus.txt)")
    parser.add_argument("--n-shot",         type=int,   default=0)
    parser.add_argument("--n-shot-seed",    type=int,   default=42)
    parser.add_argument("--print-prompt",   action="store_true")
    parser.add_argument("--test-parquet",   default=None,
                        help="Override test parquet path (e.g. variants/ben/test_mc.parquet)")
    parser.add_argument("--test-meta",      default=None,
                        help="Override test meta json path. Defaults to <test-parquet stem>_meta.json")
    parser.add_argument("--variant-dir",    default=None,
                        help="Shortcut: variants/<name>/. Picks test_mc.parquet or test_cot_mc.parquet by mode.")
    args = parser.parse_args()

    cot_mode = args.mode in ("icl-cot", "cartridge-cot")
    args._cot = cot_mode

    if args.variant_dir:
        vd = Path(args.variant_dir)
        # Pipeline now produces only test_mc.parquet (assistant = letter only,
        # reasoning is generated at eval time via cot=True chat template).
        args._test_parquet = vd / "test_mc.parquet"
        args._test_meta    = vd / "test_meta_mc.json"
    elif args.test_parquet:
        args._test_parquet = Path(args.test_parquet)
        if args.test_meta:
            args._test_meta = Path(args.test_meta)
        else:
            stem = args._test_parquet.stem
            args._test_meta = args._test_parquet.with_name(
                stem.replace("test_", "test_meta_") + ".json"
            )
    elif cot_mode:
        args._test_parquet = TEST_MC_PARQUET
        args._test_meta    = TEST_MC_META
    else:
        args._test_parquet = TEST_MC_PARQUET
        args._test_meta    = TEST_MC_META

    if args.max_new_tokens is None:
        # CoT kinship traces routinely exceed 256 tokens; 256 truncated ~97% of
        # generations mid-reasoning (no answer emitted). 1024 lets nearly all finish.
        args.max_new_tokens = 1024 if cot_mode else 8

    if args.mode in ("cartridge", "cartridge-cot"):
        assert args.checkpoint, f"--checkpoint required for {args.mode}"
        run_fn = run_cartridge_eval
    else:
        run_fn = run_icl_eval

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
