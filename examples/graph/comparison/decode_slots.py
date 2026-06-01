"""
Decode cartridge slot indices → corpus tokens, for reading a compare run.

`compare.py` writes localization.json with per-pair `K_top_slots` /
`V_top_slots` (slot index + angle_deg) but no token text. This maps each slot
back to the token it holds, using the SAME chat-template tokenization that
KVFromText uses to build the cache — so slot i here is exactly cartridge slot i
(the `<|im_start|>system\\n` prefix offset is included, not guessed).

It also prints K-vs-V side by side so you can see where Keys diverge most vs
where Values diverge most (they localize differently).

Usage:
    python -m examples.graph.comparison.decode_slots \
        --localization outputs_graph/exp1_init_kv/localization.json \
        --corpus examples/graph/data/variants/alex/family_tree_corpus.txt \
        --pair alex|ben --top 20
"""
import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from cartridges.initialization.tokenization_utils import (
    MODEL_TO_SYSTEM_PROMPT_TOKENIZER,
)


def cartridge_token_ids(tokenizer, corpus_path: Path, max_tokens=None) -> list[int]:
    """Reproduce KVFromText's exact tokenization → slot i == cartridge slot i."""
    content = Path(corpus_path).read_text()
    fn = MODEL_TO_SYSTEM_PROMPT_TOKENIZER[tokenizer.name_or_path.lower()]
    ids = fn(tokenizer=tokenizer, content=content, max_tokens=max_tokens).squeeze(0)
    return ids.tolist()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--localization", required=True, help="localization.json path")
    p.add_argument("--corpus", required=True, help="init corpus the cache came from")
    p.add_argument("--model", default="Qwen/Qwen3-1.7B")
    p.add_argument("--pair", default=None, help="which pair, e.g. alex|ben (default: first)")
    p.add_argument("--max-tokens", type=int, default=None)
    p.add_argument("--top", type=int, default=20)
    args = p.parse_args()

    loc = json.loads(Path(args.localization).read_text())
    pairs = loc["pairs"]
    pair = args.pair or next(iter(pairs))
    if pair not in pairs:
        raise SystemExit(f"pair {pair!r} not in {list(pairs)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ids = cartridge_token_ids(tokenizer, Path(args.corpus), args.max_tokens)
    n = len(ids)

    def tok(slot: int) -> str:
        return repr(tokenizer.decode([ids[slot]])) if 0 <= slot < n else "<OOR>"

    entry = pairs[pair]
    print(f"pair={pair}   cartridge slots={n}   corpus={Path(args.corpus).name}\n")

    for tag in ("K", "V"):
        rows = entry.get(f"{tag}_top_slots", [])[: args.top]
        nm = entry.get(f"{tag}_name_slot_angle")
        ot = entry.get(f"{tag}_other_slot_angle")
        rt = entry.get(f"{tag}_name_slot_ratio")
        hdr = f"── {tag} top {len(rows)} diverging slots "
        if rt is not None:
            hdr += f"(name-slot angle={nm:.2f} other={ot:.2f} ratio={rt:.2f}) "
        print(hdr + "─" * max(0, 60 - len(hdr)))
        print(f"  {'slot':>5} {'angle°':>8}   token")
        for r in rows:
            s = r["slot"]
            print(f"  {s:>5} {r['angle_deg']:>8.2f}   {tok(s)}")
        print()

    # where do K and V disagree on the ranking?
    ks = [r["slot"] for r in entry.get("K_top_slots", [])[: args.top]]
    vs = [r["slot"] for r in entry.get("V_top_slots", [])[: args.top]]
    only_k = [s for s in ks if s not in vs]
    only_v = [s for s in vs if s not in ks]
    print("── K-vs-V divergence in ranking " + "─" * 28)
    print(f"  top-{args.top} in K only: " + ", ".join(f"{s}({tok(s)})" for s in only_k))
    print(f"  top-{args.top} in V only: " + ", ".join(f"{s}({tok(s)})" for s in only_v))


if __name__ == "__main__":
    main()
