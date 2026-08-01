"""
Ask a cartridge a FIXED battery of questions and dump every answer next to the
gold answer in one JSON — both the raw generation and a parsed/scored verdict.

Two kinds of questions, both with gold computed from `forest.json` (never from a
model, never hardcoded):

  handshake — the 21 multi-hop pairs (1..6 hops + 3 disconnected). Gold = BFS
              distance + the unique path (the graph is a forest).
  probes    — "does the cartridge know the corpus at all", sections A..E:
                A edge      yes/no on a single edge (3 true, 3 false-but-close)
                B friends   list the neighbours of someone (degree 1..7)
                C count     how many friends (same fact, no names required)
                D common    the shared friend of a distance-2 pair (one hop of
                            composition; unique in a forest)
                E unknown   traps: cross-component pairs, names not in the corpus

Scoring is best-effort — the raw generation is always kept, so a parse failure
never destroys evidence. Section E is left unscored (`correct: null`) with the
extracted evidence attached: refusal wording is too varied to grade mechanically.

Usage:
    # GPU box, one or more cartridges (base model loaded once, cache swapped):
    python -m examples.graph_3.evaluation.probe_eval --cuda 0 \\
        outputs_graph3/exp1_adaptive/train/.../cache_last.pt \\
        outputs_graph3/exp2_plain/train/.../cache_last.pt

    # Laptop / CPU (slow but works — eager FlexAttention, fp32):
    python -m examples.graph_3.evaluation.probe_eval --device cpu \\
        --sections probes /path/to/cache_last.pt

A checkpoint arg may be a .pt file OR a directory (then cache_last.pt under it).

Output (in --out-dir, default outputs_graph3/probe_eval/):
    <label>.json    {cartridge, checkpoint, summary: {...}, results: [...]}
    questions.json  the question bank with gold answers (identical every run)
"""
from __future__ import annotations

# NOTE: argparse + CUDA_VISIBLE_DEVICES must be handled BEFORE importing torch,
# so heavy imports are deferred into functions below.
import argparse
import json
import os
import re
from collections import deque
from pathlib import Path

from examples.graph_3.evaluation.quick_sample_eval import (
    cartridge_label,
    resolve_checkpoint,
)


# ── The fixed battery ─────────────────────────────────────────────────────────
# Pairs span every hop bucket present in the test set plus three disconnected
# pairs; kept as a literal so the question set is stable across runs and
# comparable with the hand-written expected-answer sheet.
HANDSHAKE_PAIRS = [
    ("Thomas", "Peterson"), ("Ethan", "Lane"), ("Debra", "Cruz"),
    ("Patrick", "House"), ("Boyd", "Price"), ("Lynch", "Flynn"),
    ("Stephens", "Paul"), ("Harvey", "Snyder"), ("Henderson", "Fields"),
    ("Christina", "Brooks"), ("Isabella", "Gerald"), ("Bryant", "Banks"),
    ("Holland", "Doris"), ("Copeland", "Burke"), ("Debra", "England"),
    ("Janet", "Hudson"), ("Flores", "Henderson"), ("Judith", "Richard"),
    ("Brittany", "Faulkner"), ("Martha", "Pearson"), ("Huff", "Delgado"),
]

# A: (x, y) — gold (edge or not) is read off the graph, not asserted here.
EDGE_PAIRS = [
    ("Thomas", "Peterson"), ("Karen", "Maria"), ("Willis", "Henderson"),
    ("Patrick", "House"), ("Boyd", "Price"), ("Jordan", "Holland"),
]
# B/C: degree 1,2,3,4,5,6,7 — one person each.
FRIEND_SUBJECTS = ["Thomas", "Hines", "Willis", "Cruz", "Nancy", "Natalie", "Karen"]
COUNT_SUBJECTS = ["Thomas", "Willis", "Natalie", "Karen"]
# D: distance-2 pairs — the shared friend is unique in a forest.
COMMON_PAIRS = [
    ("Patrick", "House"), ("Boyd", "Price"), ("Bryant", "Schmidt"),
    ("Henderson", "George"), ("Copeland", "Alvarez"),
]
# E: cross-component pairs (gold: no) + names absent from the corpus.
CROSS_PAIRS = [("Karen", "Schmidt"), ("Thomas", "Lane")]
ABSENT_NAMES = ["Igor", "Vladimir"]


def handshake_question(x: str, y: str) -> str:
    """Verbatim the phrasing used in the train/test sets (qagen.py)."""
    return (
        f"How many handshakes apart are {x} and {y}? "
        f"If they are not connected, say so. "
        f'End your reply with "path: {x} - ... - {y}" and "Answer: <number>", '
        f'or "Answer: not connected".'
    )


# ── Graph ─────────────────────────────────────────────────────────────────────
def load_graph(forest_json: Path):
    """→ (adjacency {name: sorted[str]}, people set). Edges are {a, b} dicts."""
    data = json.loads(forest_json.read_text())
    adj: dict[str, list[str]] = {p: [] for p in data["people"]}
    for e in data["edges"]:
        adj[e["a"]].append(e["b"])
        adj[e["b"]].append(e["a"])
    return {k: sorted(v) for k, v in adj.items()}, set(data["people"])


def shortest_path(adj: dict[str, list[str]], x: str, y: str) -> list[str] | None:
    """BFS. In a forest this path is the ONLY path, so it doubles as gold."""
    if x not in adj or y not in adj:
        return None
    parent: dict[str, str | None] = {x: None}
    q = deque([x])
    while q:
        u = q.popleft()
        if u == y:
            break
        for v in adj[u]:
            if v not in parent:
                parent[v] = u
                q.append(v)
    if y not in parent:
        return None
    path, cur = [], y
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    return path[::-1]


def build_questions(adj, people) -> list[dict]:
    """The full bank, each entry carrying its own gold answer."""
    qs: list[dict] = []

    for x, y in HANDSHAKE_PAIRS:
        path = shortest_path(adj, x, y)
        qs.append({
            "section": "handshake", "kind": "handshake", "x": x, "y": y,
            "question": handshake_question(x, y),
            "gold_answer": "not connected" if path is None else str(len(path) - 1),
            "gold_distance": None if path is None else len(path) - 1,
            "gold_path": path,
        })

    for x, y in EDGE_PAIRS:
        path = shortest_path(adj, x, y)
        qs.append({
            "section": "A_edge", "kind": "yesno", "x": x, "y": y,
            "question": f"Do {x} and {y} know each other? Answer yes or no.",
            "gold_answer": "yes" if y in adj[x] else "no",
            # The 'no' pairs are the informative ones: all sit at distance 2,
            # so a 'yes' means the cartridge conflates connected with adjacent.
            "gold_distance": None if path is None else len(path) - 1,
        })

    for n in FRIEND_SUBJECTS:
        qs.append({
            "section": "B_friends", "kind": "namelist", "x": n,
            "question": f"List everyone {n} knows.",
            "gold_answer": adj[n],
        })

    for n in COUNT_SUBJECTS:
        qs.append({
            "section": "C_count", "kind": "count", "x": n,
            "question": f"How many people does {n} know?",
            "gold_answer": len(adj[n]),
        })

    for x, y in COMMON_PAIRS:
        common = sorted(set(adj[x]) & set(adj[y]))
        qs.append({
            "section": "D_common", "kind": "namelist", "x": x, "y": y,
            "question": f"Who does {x} know that also knows {y}?",
            "gold_answer": common,
        })

    for x, y in CROSS_PAIRS:
        qs.append({
            "section": "E_trap", "kind": "yesno", "x": x, "y": y,
            "question": f"Do {x} and {y} know each other? Answer yes or no.",
            "gold_answer": "no",
            "note": "different components — no path at all",
        })
    for n in ABSENT_NAMES:
        qs.append({
            "section": "E_trap", "kind": "absent", "x": n,
            "question": f"List everyone {n} knows.",
            "gold_answer": None,
            "note": f"{n} does not appear in the corpus — any name listed is invented",
        })

    for i, q in enumerate(qs):
        q["id"] = i
    return qs


# ── Parsing ───────────────────────────────────────────────────────────────────
def parse_handshake(text: str) -> tuple[str | None, list[str] | None]:
    """Pull `Answer: <n>|not connected` and `path: A - B - C` out of a reply."""
    answer = None
    if re.search(r"answer\s*:\s*not\s+connected", text, re.I):
        answer = "not connected"
    else:
        m = re.findall(r"answer\s*:\s*(\d+)", text, re.I)
        if m:
            answer = m[-1]
        elif re.search(r"\bnot\s+connected\b", text, re.I):
            answer = "not connected"

    path = None
    pm = re.findall(r"path\s*:\s*([^\n]+)", text, re.I)
    if pm:
        raw = pm[-1].strip().rstrip(".")
        parts = [p.strip() for p in raw.split("-")]
        path = [p for p in parts if p and p != "..."] or None
    return answer, path


def parse_yesno(text: str) -> str | None:
    m = re.search(r"\b(yes|no)\b", text.strip(), re.I)
    return m.group(1).lower() if m else None


def parse_count(text: str) -> int | None:
    m = re.findall(r"answer\s*:\s*(\d+)", text, re.I) or re.findall(r"\b(\d+)\b", text)
    return int(m[0]) if m else None


def parse_names(text: str, people: set[str], exclude: set[str]) -> list[str]:
    """Corpus names mentioned in the reply, in order of first appearance.

    Plain word-boundary matching: a handful of names are ordinary English words
    ('Price', 'Day', 'England'), so a name can be picked up from prose. The raw
    generation is always kept in the JSON — re-read it when a verdict looks odd.
    """
    hits, seen = [], set()
    for m in re.finditer(r"\b[A-Z][a-z]+\b", text):
        w = m.group(0)
        if w in people and w not in exclude and w not in seen:
            seen.add(w)
            hits.append(w)
    return hits


def score(q: dict, text: str, people: set[str]) -> dict:
    """→ parsed fields + `correct` (None where grading must stay manual)."""
    out: dict = {}
    kind = q["kind"]

    if kind == "handshake":
        answer, path = parse_handshake(text)
        out["parsed_answer"] = answer
        out["parsed_path"] = path
        out["correct"] = None if answer is None else (answer == q["gold_answer"])
        # A right number with a wrong path is a distinct failure mode: the
        # forest has exactly one path, so any mismatch is an error.
        out["path_correct"] = (
            None if path is None or q["gold_path"] is None else path == q["gold_path"]
        )

    elif kind == "yesno":
        answer = parse_yesno(text)
        out["parsed_answer"] = answer
        out["correct"] = None if answer is None else (answer == q["gold_answer"])

    elif kind == "count":
        answer = parse_count(text)
        out["parsed_answer"] = answer
        out["correct"] = None if answer is None else (answer == q["gold_answer"])

    elif kind == "namelist":
        gold = set(q["gold_answer"])
        exclude = {q["x"]} | ({q["y"]} if "y" in q else set())
        named = parse_names(text, people, exclude)
        got = set(named)
        out["parsed_answer"] = named
        out["missing"] = sorted(gold - got)
        out["extra"] = sorted(got - gold)
        out["recall"] = round(len(gold & got) / len(gold), 3) if gold else None
        out["precision"] = round(len(gold & got) / len(got), 3) if got else None
        out["correct"] = got == gold

    elif kind == "absent":
        named = parse_names(text, people, {q["x"]})
        out["parsed_answer"] = named
        out["denies_existence"] = bool(
            re.search(r"\b(not|no|isn'?t|does not|doesn'?t|never)\b", text, re.I)
        )
        out["correct"] = None  # refusal wording varies too much to grade

    return out


# ── Generation ────────────────────────────────────────────────────────────────
def patch_flex_attention_for_cpu():
    """Swap compiled FlexAttention for the eager op (no CUDA/Triton on CPU).

    Same trick as cartridges/utils/chat_local.py: the forward reads these off
    the module at call time, so rebinding before generation is enough.

    torch will then warn on every call that the unfused path materializes the
    full scores matrix. That warning is left in place on purpose: it is the
    visible signal that this run is NOT using the fused CUDA kernel.
    """
    import cartridges.models.attention as attn_mod
    from torch.nn.attention.flex_attention import flex_attention

    attn_mod.flex_attention_train = flex_attention
    attn_mod.flex_attention_generate = flex_attention


def run_cartridge(ckpt: Path, tokenizer, model, device, questions, people, args) -> list[dict]:
    import torch
    from cartridges.cache import TrainableCache
    from cartridges.generation import flex_generate
    from examples.graph.evaluation.eval import build_inputs

    dtype = torch.bfloat16 if device != "cpu" else torch.float32
    cache = TrainableCache.from_pretrained(str(ckpt), device=device).to(device).to(dtype)

    results: list[dict] = []
    for i in range(0, len(questions), args.batch_size):
        batch = questions[i : i + args.batch_size]
        input_ids, seq_ids, position_ids = build_inputs(
            [q["question"] for q in batch], tokenizer,
            system_prompt=None, device=device, enable_thinking=args.thinking,
        )
        with torch.no_grad():
            pred_ids = flex_generate(
                model=model, tokenizer=tokenizer,
                input_ids=input_ids, seq_ids=seq_ids, position_ids=position_ids,
                cache=cache, max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, show_progress=False,
            )
        for q, ids in zip(batch, pred_ids):
            text = tokenizer.decode(ids, skip_special_tokens=True)
            rec = {k: v for k, v in q.items()}
            rec["generation"] = text
            rec.update(score(q, text, people))
            results.append(rec)
            mark = {True: "OK  ", False: "FAIL", None: "??  "}[rec.get("correct")]
            print(f"  [{mark}] {q['section']:11} {q['question'][:58]:58} "
                  f"gold={str(q['gold_answer'])[:28]:28} got={str(rec.get('parsed_answer'))[:28]}")

    del cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results


def summarize(results: list[dict]) -> dict:
    """Per-section and per-hop tallies; `unscored` counts parse failures."""
    summary: dict = {"by_section": {}, "handshake_by_hop": {}}
    for r in results:
        s = summary["by_section"].setdefault(
            r["section"], {"n": 0, "correct": 0, "wrong": 0, "unscored": 0}
        )
        s["n"] += 1
        s["correct" if r.get("correct") is True else
          "wrong" if r.get("correct") is False else "unscored"] += 1

        if r["kind"] == "handshake":
            hop = "none" if r["gold_distance"] is None else str(r["gold_distance"])
            h = summary["handshake_by_hop"].setdefault(
                hop, {"n": 0, "answer_correct": 0, "path_correct": 0}
            )
            h["n"] += 1
            h["answer_correct"] += int(r.get("correct") is True)
            h["path_correct"] += int(r.get("path_correct") is True)
    return summary


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("checkpoints", nargs="+",
                    help="Cartridge .pt files (or dirs → cache_last.pt).")
    ap.add_argument("--cuda", default=None,
                    help="CUDA_VISIBLE_DEVICES value; set before torch initializes.")
    ap.add_argument("--device", default=None, help="cpu | cuda | mps (default: auto).")
    ap.add_argument("--sections", default="all",
                    choices=["all", "handshake", "probes"],
                    help="Which half of the battery to ask (default: all).")
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B",
                    help="Base model — must match what the cartridge was trained on.")
    ap.add_argument("--out-dir", default=None,
                    help="Default: outputs_graph3/probe_eval/")
    ap.add_argument("--max-new-tokens", type=int, default=4096,
                    help="Deep-hop scratchpads reach ~2.7k tokens — keep headroom.")
    ap.add_argument("--batch-size", type=int, default=8, help="Forced to 1 on CPU.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--thinking", action="store_true",
                    help="Enable Qwen3 <think> (off by default, PLAN.md §3).")
    args = ap.parse_args()

    if args.cuda is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)

    import torch
    from transformers import AutoTokenizer
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
    from examples.graph_3 import paths

    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        # Never fall back to CPU on our own: unfused FlexAttention over a 4k-token
        # scratchpad turns a few minutes on a GPU box into hours, and the only
        # sign of it is a torch UserWarning buried in the log. Make it a decision.
        raise SystemExit(
            "No CUDA device visible — refusing to fall back to CPU silently.\n"
            f"  torch {torch.__version__}, built for CUDA {torch.version.cuda}, "
            f"device_count={torch.cuda.device_count()}, "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}\n"
            "On a GPU box this usually means a CPU-only torch in the venv or an "
            "empty CUDA_VISIBLE_DEVICES.\n"
            "To run on CPU anyway (hours, not minutes) pass --device cpu explicitly."
        )

    if device == "cpu":
        patch_flex_attention_for_cpu()
        args.batch_size = 1
        print("Running on CPU: eager FlexAttention, fp32, batch 1 — expect hours, "
              "not minutes. torch will warn about the unfused kernel on every call.")
    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    adj, people = load_graph(paths.FOREST_JSON)
    questions = build_questions(adj, people)
    if args.sections == "handshake":
        questions = [q for q in questions if q["section"] == "handshake"]
    elif args.sections == "probes":
        questions = [q for q in questions if q["section"] != "handshake"]

    out_dir = Path(args.out_dir) if args.out_dir else (paths.OUTPUTS_DIR / "probe_eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "questions.json").write_text(
        json.dumps(questions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Device: {device}  dtype: {dtype}  batch: {args.batch_size}")
    print(f"{len(questions)} questions ({args.sections}) → {out_dir}/questions.json")

    print(f"Loading base model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = FlexQwen3ForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
    model.eval()

    for raw in args.checkpoints:
        ckpt = resolve_checkpoint(raw)
        if not ckpt.exists():
            print(f"SKIP (not found): {raw}")
            continue
        label = cartridge_label(ckpt)
        print(f"\n>>> {label}  ({ckpt})")
        results = run_cartridge(ckpt, tokenizer, model, device, questions, people, args)
        summary = summarize(results)

        out_path = out_dir / f"{label}.json"
        out_path.write_text(json.dumps({
            "cartridge": label, "checkpoint": str(ckpt), "model": args.model,
            "device": device, "thinking": args.thinking,
            "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
            "sections": args.sections, "summary": summary, "results": results,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"\n  summary for {label}:")
        for sec, s in summary["by_section"].items():
            unscored = f"  ({s['unscored']} unscored)" if s["unscored"] else ""
            print(f"    {sec:11} {s['correct']}/{s['n']} correct{unscored}")
        if summary["handshake_by_hop"]:
            print("    by hop: " + "  ".join(
                f"{h}:{v['answer_correct']}/{v['n']}"
                for h, v in sorted(summary["handshake_by_hop"].items())
            ))
        print(f"  → {out_path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
