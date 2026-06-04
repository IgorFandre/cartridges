"""
Exp 0 baseline + cartridge evaluation for lineage Yes/No questions.

Two modes:
  icl        — graph in system prompt (ICL), free CoT then Yes/No.
  cartridge  — trained cartridge, graph NOT in prompt, CoT then Yes/No.

Both modes use free-form generation (large max_new_tokens) with thinking enabled.

`extract_yes_no` is defined here and imported by other modules.

Usage:
    # Exp 0 — ICL baseline
    python -m examples.graph_2.evaluation.lineage_eval --mode icl \\
        --output outputs_graph2/exp0_icl/results.json

    # Cartridge eval
    python -m examples.graph_2.evaluation.lineage_eval --mode cartridge \\
        --checkpoint /path/to/cache_last.pt \\
        --output outputs_graph2/exp1_selfstudy/train/eval/results.json
"""
from __future__ import annotations
import argparse
import json
import re
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from cartridges.structs import read_conversations
from examples.graph_2 import paths


MODEL_CONFIGS = {
    "qwen1.7b": "Qwen/Qwen3-1.7B",
    "qwen4b":   "Qwen/Qwen3-4b",
}


# ── Yes/No extraction ─────────────────────────────────────────────────────────
def extract_yes_no(text: str) -> str | None:
    """Extract 'Yes' or 'No' from a model response.

    Scans only after </think> (if present).  An unclosed <think> means the
    generation was truncated before any answer → return None.
    Prefers the LAST Yes/No occurrence in the answer region.
    """
    t = text or ""
    if "</think>" in t:
        region = t.rsplit("</think>", 1)[1]
    elif "<think>" in t:
        return None   # truncated mid-reasoning
    else:
        region = t

    # Find all yes/no occurrences and return the last
    matches = re.findall(r"\b(yes|no)\b\.?", region.strip(), flags=re.IGNORECASE)
    if matches:
        return matches[-1].capitalize()
    return None


# ── Input building ─────────────────────────────────────────────────────────────
def _build_icl_inputs(
    question: str,
    system_prompt: str,
    tokenizer,
    device: str,
    enable_thinking: bool,
):
    """Build full_ids tensor for ICL eval (system+user combined)."""
    from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE, MODELS_WITH_THINKING

    kwargs: dict = {}
    is_thinking = tokenizer.name_or_path.lower() in {m.lower() for m in MODELS_WITH_THINKING}
    if is_thinking:
        kwargs["enable_thinking"] = enable_thinking

    full_ids = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": question},
        ],
        add_generation_prompt=True,
        return_tensors="pt",
        chat_template=MODEL_TO_CHAT_TEMPLATE.get(tokenizer.name_or_path),
        **kwargs,
    ).to(device)

    if is_thinking and not enable_thinking:
        no_think = tokenizer.encode("<think>\n</think>\n", add_special_tokens=False)
        full_ids = torch.cat(
            [full_ids, torch.tensor([no_think], device=device)], dim=1
        )
    return full_ids


def _cache_sys_prefix(system_prompt: str, tokenizer, model, device: str, enable_thinking: bool):
    """Cache the system prefix KV once, then reuse per question (ICL eval)."""
    from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE, MODELS_WITH_THINKING
    import copy

    kwargs: dict = {}
    is_thinking = tokenizer.name_or_path.lower() in {m.lower() for m in MODELS_WITH_THINKING}
    if is_thinking:
        kwargs["enable_thinking"] = enable_thinking

    SENTINEL = "__LINEAGE_SENTINEL_XYZ__"
    sentinel_ids = tokenizer.encode(SENTINEL, add_special_tokens=False)

    full_ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user",   "content": SENTINEL}],
        add_generation_prompt=False,
        return_tensors=None,
        chat_template=MODEL_TO_CHAT_TEMPLATE.get(tokenizer.name_or_path),
        **kwargs,
    )
    n = len(sentinel_ids)
    sys_prefix_len = None
    for j in range(len(full_ids) - n + 1):
        if full_ids[j:j+n] == sentinel_ids:
            sys_prefix_len = j
            break
    if sys_prefix_len is None:
        raise ValueError("Sentinel not found in tokenized prompt.")

    sys_ids = tokenizer.apply_chat_template(
        [{"role": "system", "content": system_prompt},
         {"role": "user",   "content": ""}],
        add_generation_prompt=False,
        return_tensors="pt",
        chat_template=MODEL_TO_CHAT_TEMPLATE.get(tokenizer.name_or_path),
        **kwargs,
    )[:, :sys_prefix_len].to(device)

    with torch.no_grad():
        sys_past_kv = model(sys_ids, use_cache=True).past_key_values
    return sys_past_kv, sys_prefix_len


# ── Shared result-row builder ──────────────────────────────────────────────────
def _result_row(m: dict, pred_text: str) -> dict:
    """Build a per-question result dict from a meta record + model output.

    Works whether `m` comes from test_meta.json (raw records, key='question')
    or from conversation metadata (key='question_text').  n_bucket is always
    normalised to str so analyze.py can sort it uniformly.
    """
    pred_yn = extract_yes_no(pred_text)
    label   = m.get("label", "")
    return {
        "direction":     m.get("direction"),
        "true_distance": m.get("true_distance"),
        "n_bucket":      str(m.get("n_bucket", "none")),
        "claimed_n":     m.get("claimed_n"),
        "label":         label,
        "x": m.get("x"), "y": m.get("y"),
        "question":      m.get("question_text", m.get("question", "")),
        "predicted":     pred_text,
        "predicted_yn":  pred_yn,
        "correct":       pred_yn is not None and pred_yn == label,
    }


# ── ICL eval ──────────────────────────────────────────────────────────────────
def run_icl_eval(args) -> List[dict]:
    import copy
    from transformers import AutoModelForCausalLM

    model_name = MODEL_CONFIGS[args.model]
    device     = args.device

    corpus = Path(args.corpus_path).read_text() if args.corpus_path else \
             paths.BASE_CORPUS.read_text()

    instruction = (
        "Use the family tree below to answer the lineage question. "
        "Reason step by step, then end with exactly 'Yes.' or 'No.'  "
        "Do not output anything after the final Yes/No."
    )
    system_prompt = instruction + "\n\nFamily tree:\n" + corpus

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    convos = read_conversations(str(args._test_parquet))
    meta   = json.loads(Path(args._test_meta).read_text())
    assert len(convos) == len(meta)

    if args.limit:
        convos = convos[:args.limit]; meta = meta[:args.limit]

    sys_past_kv, _ = _cache_sys_prefix(
        system_prompt, tokenizer, model, device, enable_thinking=True
    )
    print(f"System prefix cached ({len(corpus)} chars corpus).")

    from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE, MODELS_WITH_THINKING

    is_thinking = model_name.lower() in {m.lower() for m in MODELS_WITH_THINKING}
    kwargs = {"enable_thinking": True} if is_thinking else {}
    chat_template = MODEL_TO_CHAT_TEMPLATE.get(model_name)

    results = []
    for i in tqdm(range(len(convos)), desc="icl eval"):
        question = convos[i].messages[0].content
        m        = meta[i]

        full_ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user",   "content": question}],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=chat_template,
            **kwargs,
        ).to(device)

        with torch.no_grad():
            output_ids = model.generate(
                full_ids,
                past_key_values=copy.deepcopy(sys_past_kv),
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
            )

        pred_text = tokenizer.decode(
            output_ids[0][full_ids.shape[1]:], skip_special_tokens=True
        )
        results.append(_result_row(m, pred_text))

    return results


# ── Cartridge eval ───────────────────────────────────────────────────────────── ─────────────────────────────────────────────────────────────
def run_cartridge_eval(args) -> List[dict]:
    from cartridges.cache import TrainableCache
    from cartridges.generation import flex_generate
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
    from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE, MODELS_WITH_THINKING

    model_name = MODEL_CONFIGS[args.model]
    device     = args.device

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = FlexQwen3ForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    cache: TrainableCache = TrainableCache.from_pretrained(args.checkpoint, device=device)
    cache = cache.to(device)

    convos = read_conversations(str(args._test_parquet))
    meta   = json.loads(Path(args._test_meta).read_text())
    assert len(convos) == len(meta)

    if args.limit:
        convos = convos[:args.limit]; meta = meta[:args.limit]

    is_thinking = model_name.lower() in {m.lower() for m in MODELS_WITH_THINKING}
    kwargs = {"enable_thinking": True} if is_thinking else {}

    results = []
    for i in tqdm(range(0, len(convos), args.batch_size), desc="cartridge eval"):
        batch_convos = convos[i : i + args.batch_size]
        batch_meta   = meta[i : i + args.batch_size]
        questions    = [c.messages[0].content for c in batch_convos]

        # Reuse the input building logic from the graph eval
        from examples.graph.evaluation.eval import build_inputs
        input_ids, seq_ids, position_ids = build_inputs(
            questions, tokenizer,
            system_prompt=None,
            device=device,
            enable_thinking=True,
        )

        with torch.no_grad():
            pred_ids = flex_generate(
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
            results.append(_result_row(m, pred_text))

    return results


# ── Reporting ─────────────────────────────────────────────────────────────────
def print_results(results: List[dict], mode: str):
    total = len(results)
    n_ok  = sum(r["correct"] for r in results)
    print(f"\n=== {mode.upper()} ===")
    print(f"Overall  {n_ok:4d}/{total} = {n_ok/total:.1%}")

    # Per n_bucket
    buckets = sorted({str(r["n_bucket"]) for r in results}, key=lambda x: (x=="none", x))
    print("\nPer n_bucket (true hop-distance):")
    for b in buckets:
        rows = [r for r in results if str(r["n_bucket"]) == b]
        nc = sum(r["correct"] for r in rows)
        print(f"  n={b:>4}: {nc:4d}/{len(rows):4d} = {nc/len(rows):.1%}")

    # Per direction
    for d in ("ancestor", "descendant"):
        rows = [r for r in results if r.get("direction") == d]
        if rows:
            nc = sum(r["correct"] for r in rows)
            print(f"  {d:12s}: {nc}/{len(rows)} = {nc/len(rows):.1%}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",          required=True, choices=["icl", "cartridge"])
    ap.add_argument("--checkpoint",    default=None,  help="Path to cache_last.pt")
    ap.add_argument("--corpus-path",   default=None,  help="Override ICL corpus path")
    ap.add_argument("--test-parquet",  default=None)
    ap.add_argument("--test-meta",     default=None)
    ap.add_argument("--model",         default="qwen1.7b", choices=list(MODEL_CONFIGS))
    ap.add_argument("--max-new-tokens",type=int,  default=1024)
    ap.add_argument("--batch-size",    type=int,  default=8)
    ap.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output",        default=None)
    ap.add_argument("--temperature",   type=float, default=0.0)
    ap.add_argument("--limit",         type=int,   default=None)
    args = ap.parse_args()

    args._test_parquet = Path(args.test_parquet) if args.test_parquet else paths.BASE_TEST_PARQUET
    args._test_meta    = Path(args.test_meta)    if args.test_meta    else paths.BASE_TEST_META

    if args.mode == "cartridge":
        assert args.checkpoint, "--checkpoint required for cartridge mode"
        results = run_cartridge_eval(args)
    else:
        results = run_icl_eval(args)

    print_results(results, args.mode)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nSaved → {args.output}")


if __name__ == "__main__":
    main()
