"""
Per-n-hop accuracy analysis for lineage cartridge experiments.

Reads one or more results.json files (output of lineage_eval.py) and prints:
  - Overall accuracy per file
  - Per-n_bucket table (columns: n_bucket | N | correct | acc%)
  - Per-direction (ancestor vs descendant)
  - Yes/No confusion matrix
  - Comparison summary table (rows = files, cols = n_bucket)

Usage:
    python -m examples.graph_2.evaluation.analyze results.json
    python -m examples.graph_2.evaluation.analyze \\
        outputs_graph2/exp0_icl/results.json \\
        outputs_graph2/exp1_selfstudy/train/eval/results.json \\
        outputs_graph2/exp2_star/train/eval/results.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


_NULL = (None, "?", "")  # predicted_yn values that mean "no parseable answer"


def _load(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def _acc(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    return sum(r["correct"] for r in rows) / len(rows)


def _is_parsed(r: dict) -> bool:
    return r.get("predicted_yn") not in _NULL


def _n_unparsed(rows: list[dict]) -> int:
    return sum(1 for r in rows if not _is_parsed(r))


def _acc_parsed(rows: list[dict]) -> float:
    """Accuracy among rows that produced a Yes/No (excludes truncated answers)."""
    parsed = [r for r in rows if _is_parsed(r)]
    if not parsed:
        return float("nan")
    return sum(r["correct"] for r in parsed) / len(parsed)


def _frac_unparsed(rows: list[dict]) -> float:
    return _n_unparsed(rows) / len(rows) if rows else float("nan")


def _label(path: str) -> str:
    """Short experiment label from a results path (the 'exp*' dir if present)."""
    for part in Path(path).parts:
        if part.startswith("exp"):
            return part
    return Path(path).parent.name


def _pct(v: float) -> str:
    if v != v:  # nan
        return "  n/a"
    return f"{v:.1%}"


def analyze_run(results: list[dict], label: str = "") -> dict:
    """Print a per-n-bucket report and return the per-bucket acc dict."""
    total  = len(results)
    n_ok   = sum(r["correct"] for r in results)
    n_unp  = _n_unparsed(results)
    header = f"=== {label or 'RESULTS'} ==="
    print(f"\n{header}")
    print(f"Overall  {n_ok:4d}/{total} = {n_ok/total:.1%}")
    print(f"Unparsed {n_unp:4d}/{total} = {_pct(_frac_unparsed(results))}"
          f"   ·   acc(parsed only) = {_pct(_acc_parsed(results))}")

    # ── Per n_bucket ──────────────────────────────────────────────────────────
    buckets_raw = sorted(
        {r["n_bucket"] for r in results},
        key=lambda x: (x is None or x == "none", x),
    )
    bucket_accs: dict[str, float] = {}
    print(f"\n{'n_bucket':>10} {'N':>6} {'correct':>8} {'acc':>7} {'none':>6} {'acc_par':>8}")
    print("-" * 50)
    for b in buckets_raw:
        rows = [r for r in results if r["n_bucket"] == b]
        nc   = sum(r["correct"] for r in rows)
        acc  = nc / len(rows)
        bucket_accs[str(b)] = acc
        print(f"{str(b):>10} {len(rows):>6} {nc:>8} {_pct(acc):>7} "
              f"{_n_unparsed(rows):>6} {_pct(_acc_parsed(rows)):>8}")
    print("-" * 50)
    print(f"{'total':>10} {total:>6} {n_ok:>8} {_pct(n_ok/total):>7} "
          f"{n_unp:>6} {_pct(_acc_parsed(results)):>8}")

    # ── Per direction ─────────────────────────────────────────────────────────
    print("\nPer direction:")
    for d in ("ancestor", "descendant"):
        rows = [r for r in results if r.get("direction") == d]
        if rows:
            nc = sum(r["correct"] for r in rows)
            print(f"  {d:12s}: {nc}/{len(rows)} = {_pct(nc/len(rows))}")

    # ── Yes/No confusion ──────────────────────────────────────────────────────
    from collections import Counter
    print("\nConfusion (gold → pred):")
    conf: Counter = Counter()
    for r in results:
        gold = r["label"]
        pred = r.get("predicted_yn") or "?"
        conf[(gold, pred)] += 1
    for (g, p), n in sorted(conf.items()):
        print(f"  gold={g:3s} pred={p:3s}: {n}")

    # ── Sample errors ─────────────────────────────────────────────────────────
    errors = [r for r in results if not r["correct"]][:3]
    if errors:
        print("\nSample errors:")
        for e in errors:
            print(f"  [{e.get('n_bucket')}] Q: {e['question']}")
            print(f"       gold={e['label']}  pred={e.get('predicted_yn')}  claimed_n={e.get('claimed_n')}")

    return bucket_accs


def compare_summary(files: list[str]) -> None:
    """Print a comparison table: rows = files, columns = n_bucket."""
    if len(files) < 2:
        return

    all_results = [(p, _load(p)) for p in files]
    # Collect all unique buckets
    all_buckets = sorted(
        {r["n_bucket"] for _, rs in all_results for r in rs},
        key=lambda x: (x is None or x == "none", x),
    )

    # Header: overall / acc among parsed / %unparsed, then per-n-hop acc
    col_w = 8
    b_headers = [f"n={str(b):>4}" for b in all_buckets]
    header = (f"{'experiment':<18} {'overall':>8} {'acc_par':>8} {'%none':>6}  "
              + " ".join(b_headers))
    print(f"\n{'='*len(header)}")
    print("COMPARISON SUMMARY  (overall = raw acc · acc_par = acc among answered · %none = unparsed)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for path, rs in all_results:
        row = (f"{_label(path):<18} {_pct(_acc(rs)):>8} {_pct(_acc_parsed(rs)):>8} "
               f"{_pct(_frac_unparsed(rs)):>6}  ")
        for b in all_buckets:
            rows_b = [r for r in rs if r["n_bucket"] == b]
            row += f" {_pct(_acc(rows_b)):>{col_w}}"
        print(row)
    print("=" * len(header))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="Path(s) to results.json")
    ap.add_argument("--no-detail", action="store_true", help="Skip per-file detail, show comparison only")
    args = ap.parse_args()

    bucket_accs_all: dict[str, dict] = {}
    for path in args.files:
        rs = _load(path)
        if not args.no_detail:
            bucket_accs_all[path] = analyze_run(rs, _label(path))

    if len(args.files) > 1:
        compare_summary(args.files)


if __name__ == "__main__":
    main()
