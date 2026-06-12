"""
Exp 0 baseline + cartridge evaluation for handshake-distance questions.

Two modes:
  icl        — friendship corpus in the system prompt (ICL).
  cartridge  — trained cartridge, corpus NOT in the prompt.

The model answers free-form (the question itself asks for the `path:` /
`Answer:` ending); thinking is DISABLED by default (PLAN.md §3) — the
scratchpad is the visible reasoning. Enable Qwen3 thinking with --thinking.

Scoring is fully deterministic (PLAN.md §5):
  acc       — extracted answer == gold distance / "not connected"   (primary)
  fidelity  — answer correct AND predicted path == the unique gold path
  path_valid— predicted path is a real chain x→…→y in the forest (diagnostic)
  abs_err   — |pred − true| when both are numeric (MAE over these)

`extract_answer` / `extract_path` are defined here and imported by the
synthesis (exp1/exp2) and training-eval modules.

Usage:
    # Exp 0 — ICL baseline
    python -m examples.graph_3.evaluation.eval --mode icl \\
        --output outputs_graph3/exp0_icl/results.json

    # Cartridge eval
    python -m examples.graph_3.evaluation.eval --mode cartridge \\
        --checkpoint /path/to/cache_last.pt \\
        --output outputs_graph3/exp1_adaptive/eval/results.json
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

from examples.graph_3 import paths
from examples.graph_3.data_gen.graph_index import GraphIndex


MODEL_CONFIGS = {
    "qwen1.7b": "Qwen/Qwen3-1.7B",
    "qwen4b":   "Qwen/Qwen3-4b",
}

_SYSTEM_INSTRUCTION = (
    "You will be asked how many handshakes apart two people are. "
    "Use the friendship list below. Two people are 1 handshake apart if they "
    "know each other directly; people in different friend groups are not "
    "connected.\n\nFriendships:\n{corpus}"
)


# ── Answer / path extraction ──────────────────────────────────────────────────
def _answer_region(text: str) -> str | None:
    """The part of the response that may contain the final answer.

    Scans only after </think> (if present). An unclosed <think> means the
    generation was truncated before any answer → None.
    """
    t = text or ""
    if "</think>" in t:
        return t.rsplit("</think>", 1)[1]
    if "<think>" in t:
        return None
    return t


def extract_answer(text: str) -> str | None:
    """'3'-style digit string, 'not connected', or None if unparseable.

    Takes the LAST `Answer:` occurrence in the answer region.
    """
    region = _answer_region(text)
    if region is None:
        return None
    matches = re.findall(
        r"answer\s*:\s*\**\s*(not\s+connected|\d+)", region, flags=re.IGNORECASE
    )
    if not matches:
        return None
    last = matches[-1].lower()
    return "not connected" if last.startswith("not") else str(int(last))


def extract_path(text: str) -> list[str] | None:
    """Name list from the LAST `path:` line in the answer region, or None."""
    region = _answer_region(text)
    if region is None:
        return None
    matches = re.findall(
        r"^\s*\**\s*path\s*:\s*(.+)$", region, flags=re.IGNORECASE | re.MULTILINE
    )
    if not matches:
        return None
    names = [n.strip(" .*") for n in matches[-1].split("-")]
    names = [n for n in names if n]
    return names or None


# ── Scoring ───────────────────────────────────────────────────────────────────
def score_row(m: dict, pred_text: str, index: GraphIndex | None = None) -> dict:
    """Per-question result row from a qagen meta record + model output."""
    gold      = m["answer"]                      # "3" | "not connected"
    gold_path = m.get("path")                    # list | None
    pred_ans  = extract_answer(pred_text)
    pred_path = extract_path(pred_text)

    correct = pred_ans is not None and pred_ans == gold

    if gold == "not connected":
        fidelity = correct                       # no path required for negatives
    else:
        fidelity = correct and pred_path == gold_path

    path_valid = None
    if index is not None and pred_path is not None:
        path_valid = (
            len(pred_path) >= 2
            and pred_path[0] == m["x"]
            and pred_path[-1] == m["y"]
            and all(b in index.neighbors.get(a, []) for a, b in zip(pred_path, pred_path[1:]))
        )

    abs_err = None
    if pred_ans is not None and pred_ans != "not connected" and gold != "not connected":
        abs_err = abs(int(pred_ans) - int(gold))

    return {
        "question":         m["question"],
        "x": m["x"], "y": m["y"],
        "true_distance":    m["true_distance"],
        "n_bucket":         str(m["n_bucket"]),
        "gold_answer":      gold,
        "gold_path":        gold_path,
        "predicted":        pred_text,
        "predicted_answer": pred_ans,
        "predicted_path":   pred_path,
        "correct":          correct,
        "fidelity":         fidelity,
        "path_valid":       path_valid,
        "abs_err":          abs_err,
    }


# ── Input building (shared with graph eval helpers) ───────────────────────────
def _chat_kwargs(tokenizer, enable_thinking: bool) -> dict:
    from cartridges.initialization.tokenization_utils import MODELS_WITH_THINKING

    if tokenizer.name_or_path.lower() in {m.lower() for m in MODELS_WITH_THINKING}:
        return {"enable_thinking": enable_thinking}
    return {}


def _no_think_ids(tokenizer, enable_thinking: bool) -> list[int]:
    """Force-empty <think> block when thinking is disabled on a thinking model."""
    if _chat_kwargs(tokenizer, enable_thinking) and not enable_thinking:
        return tokenizer.encode("<think>\n</think>\n", add_special_tokens=False)
    return []


def _cache_sys_prefix(system_prompt: str, tokenizer, model, device: str, enable_thinking: bool):
    """Cache the system prefix KV once, then reuse per question (ICL eval)."""
    from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE

    kwargs = _chat_kwargs(tokenizer, enable_thinking)
    SENTINEL = "__HANDSHAKE_SENTINEL_XYZ__"
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
        if full_ids[j:j + n] == sentinel_ids:
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


def _apply_question_filter(args, meta: list) -> list:
    """Keep only records whose question is in args._question_filter (if set)."""
    keep_set = getattr(args, "_question_filter", None)
    if keep_set is None:
        return meta
    out = [m for m in meta if m["question"] in keep_set]
    print(f"rerun filter: {len(out)}/{len(meta)} questions selected")
    return out


def _load_meta(args) -> list[dict]:
    meta = json.loads(Path(args._test_meta).read_text())
    if args.limit:
        meta = meta[: args.limit]
    return _apply_question_filter(args, meta)


# ── ICL eval ──────────────────────────────────────────────────────────────────
def run_icl_eval(args, index: GraphIndex) -> List[dict]:
    import copy
    from transformers import AutoModelForCausalLM
    from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE

    model_name = MODEL_CONFIGS[args.model]
    device     = args.device

    corpus_path = Path(args.corpus_path) if args.corpus_path else paths.CORPUS_TXT
    if not corpus_path.exists():
        raise FileNotFoundError(
            f"Corpus not found: {corpus_path}\n"
            "Run data generation first:\n"
            "  python -m examples.graph_3.data_gen.generate_forest\n"
            "  python -m examples.graph_3.data_gen.qagen"
        )
    system_prompt = _SYSTEM_INSTRUCTION.format(corpus=corpus_path.read_text())
    if getattr(args, "stepbystep", False):
        # Same directive as the stepbystep synthesis datasets, so the ICL
        # baseline can be compared on equal footing.
        from examples.graph_3.synthesis.common import STEPBYSTEP_DIRECTIVE
        system_prompt += STEPBYSTEP_DIRECTIVE

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    meta = _load_meta(args)

    sys_past_kv, _ = _cache_sys_prefix(
        system_prompt, tokenizer, model, device, enable_thinking=args.thinking
    )
    print(f"System prefix cached ({len(system_prompt)} chars).")

    kwargs        = _chat_kwargs(tokenizer, args.thinking)
    no_think      = _no_think_ids(tokenizer, args.thinking)
    chat_template = MODEL_TO_CHAT_TEMPLATE.get(model_name)

    results = []
    for m in tqdm(meta, desc="icl eval"):
        full_ids = tokenizer.apply_chat_template(
            [{"role": "system", "content": system_prompt},
             {"role": "user",   "content": m["question"]}],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=chat_template,
            **kwargs,
        ).to(device)
        if no_think:
            full_ids = torch.cat(
                [full_ids, torch.tensor([no_think], device=device)], dim=1
            )

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
        results.append(score_row(m, pred_text, index))

    return results


# ── Cartridge eval ────────────────────────────────────────────────────────────
def run_cartridge_eval(args, index: GraphIndex) -> List[dict]:
    from cartridges.cache import TrainableCache
    from cartridges.generation import flex_generate
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
    from examples.graph.evaluation.eval import build_inputs

    model_name = MODEL_CONFIGS[args.model]
    device     = args.device

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = FlexQwen3ForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()

    cache: TrainableCache = TrainableCache.from_pretrained(args.checkpoint, device=device)
    cache = cache.to(device)

    meta = _load_meta(args)

    results = []
    for i in tqdm(range(0, len(meta), args.batch_size), desc="cartridge eval"):
        batch_meta = meta[i : i + args.batch_size]
        questions  = [m["question"] for m in batch_meta]

        input_ids, seq_ids, position_ids = build_inputs(
            questions, tokenizer,
            system_prompt=None,
            device=device,
            enable_thinking=args.thinking,
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

        for m, ids in zip(batch_meta, pred_ids):
            pred_text = tokenizer.decode(ids, skip_special_tokens=True)
            results.append(score_row(m, pred_text, index))

    return results


# ── Reporting ─────────────────────────────────────────────────────────────────
def _pct(v: float) -> str:
    return "  n/a" if v != v else f"{v:.1%}"


def _mae(rows: list[dict]) -> float:
    errs = [r["abs_err"] for r in rows if r["abs_err"] is not None]
    return sum(errs) / len(errs) if errs else float("nan")


def print_results(results: List[dict], mode: str):
    total = len(results)
    n_ok  = sum(r["correct"] for r in results)
    n_fid = sum(r["fidelity"] for r in results)
    n_unp = sum(1 for r in results if r["predicted_answer"] is None)
    print(f"\n=== {mode.upper()} ===")
    print(f"Overall   acc {n_ok:4d}/{total} = {_pct(n_ok/total)}"
          f"   fidelity {_pct(n_fid/total)}"
          f"   unparsed {_pct(n_unp/total)}"
          f"   MAE {_mae(results):.2f}")

    buckets = sorted({r["n_bucket"] for r in results}, key=lambda x: (x == "none", x))
    print(f"\n{'bucket':>7} {'N':>5} {'acc':>7} {'fidelity':>9} {'unparsed':>9} {'MAE':>6}")
    print("-" * 48)
    for b in buckets:
        rows = [r for r in results if r["n_bucket"] == b]
        acc  = sum(r["correct"] for r in rows) / len(rows)
        fid  = sum(r["fidelity"] for r in rows) / len(rows)
        unp  = sum(1 for r in rows if r["predicted_answer"] is None) / len(rows)
        mae  = _mae(rows)
        mae_s = f"{mae:6.2f}" if mae == mae else "   n/a"
        print(f"{b:>7} {len(rows):>5} {_pct(acc):>7} {_pct(fid):>9} {_pct(unp):>9} {mae_s}")

    gen = [r for r in results if r["n_bucket"] in ("7", "8")]
    if gen:
        acc = sum(r["correct"] for r in gen) / len(gen)
        print(f"\nGeneralization (hops 7-8, unseen in train): {_pct(acc)}")

    # Error typology across the connected/not-connected boundary
    false_nc = sum(
        1 for r in results
        if r["gold_answer"] != "not connected" and r["predicted_answer"] == "not connected"
    )
    false_conn = sum(
        1 for r in results
        if r["gold_answer"] == "not connected"
        and r["predicted_answer"] not in (None, "not connected")
    )
    print(f"false 'not connected' (gold has a path): {false_nc}")
    print(f"false connection (gold not connected, predicted a number): {false_conn}")


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode",          required=True, choices=["icl", "cartridge"])
    ap.add_argument("--checkpoint",    default=None,  help="Path to cache_last.pt")
    ap.add_argument("--corpus-path",   default=None,  help="Override ICL corpus path")
    ap.add_argument("--test-meta",     default=None)
    ap.add_argument("--forest",        default=None,  help="Override forest.json (path validity)")
    ap.add_argument("--model",         default="qwen1.7b", choices=list(MODEL_CONFIGS))
    ap.add_argument("--max-new-tokens", type=int, default=4096,
                    help="Push/pop scratchpads reach ~2.7k tokens — keep headroom")
    ap.add_argument("--batch-size",    type=int,  default=8)
    ap.add_argument("--device",        default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--output",        default=None)
    ap.add_argument("--temperature",   type=float, default=0.0)
    ap.add_argument("--limit",         type=int,   default=None)
    ap.add_argument("--thinking",      action="store_true",
                    help="Enable Qwen3 <think> mode (off by default, PLAN.md §3)")
    ap.add_argument("--stepbystep",    action="store_true",
                    help="ICL mode: append the step-by-step search directive to the "
                         "system prompt (same wording as the stepbystep datasets)")
    ap.add_argument("--rerun-unparsed", default=None,
                    help="Path to a prior results.json: re-evaluate ONLY the questions "
                         "whose predicted_answer is null, merge the fresh answers back "
                         "and write to --output (default: in place).")
    args = ap.parse_args()

    args._test_meta = Path(args.test_meta) if args.test_meta else paths.BASE_TEST_META
    if not args._test_meta.exists():
        raise FileNotFoundError(
            f"test meta not found: {args._test_meta}\n"
            "Run data generation first:\n"
            "  python -m examples.graph_3.data_gen.generate_forest\n"
            "  python -m examples.graph_3.data_gen.qagen"
        )

    forest_path = Path(args.forest) if args.forest else paths.FOREST_JSON
    index = GraphIndex.load(forest_path) if forest_path.exists() else None
    if index is None:
        print(f"WARNING: forest not found at {forest_path} — path_valid will be null.")

    # --rerun-unparsed: load prior results, target only the unanswered ones
    prior_results: list[dict] | None = None
    if args.rerun_unparsed:
        prior_path = Path(args.rerun_unparsed)
        if not prior_path.exists():
            raise FileNotFoundError(f"--rerun-unparsed results not found: {prior_path}")
        prior_results = json.loads(prior_path.read_text())
        unparsed_qs = {
            r["question"] for r in prior_results if r.get("predicted_answer") is None
        }
        if not unparsed_qs:
            print("No unparsed answers in the prior results — nothing to rerun.")
            return
        print(f"Re-running {len(unparsed_qs)} unparsed question(s) "
              f"from {prior_path} with max_new_tokens={args.max_new_tokens}")
        args._question_filter = unparsed_qs

    if args.mode == "cartridge":
        assert args.checkpoint, "--checkpoint required for cartridge mode"
        results = run_cartridge_eval(args, index)
    else:
        results = run_icl_eval(args, index)

    # Merge fresh answers back into the prior results (keyed by question text)
    if prior_results is not None:
        fresh = {r["question"]: r for r in results}
        merged = [fresh.get(r["question"], r) for r in prior_results]
        n_recovered = (
            sum(1 for r in merged if r.get("predicted_answer") is not None)
            - sum(1 for r in prior_results if r.get("predicted_answer") is not None)
        )
        print(f"Recovered {n_recovered} previously-unparsed answer(s) after rerun.")
        results = merged

    print_results(results, args.mode)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nSaved → {args.output}")
    elif args.rerun_unparsed:
        Path(args.rerun_unparsed).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nSaved (in place) → {args.rerun_unparsed}")


if __name__ == "__main__":
    main()
