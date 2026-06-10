"""
Per-hop analysis for handshake-cartridge experiments.

Reads one or more results.json files (output of evaluation/eval.py) and prints:
  - Overall acc / fidelity / unparsed / MAE per file
  - Per-bucket table (hops 1..8 + none)
  - Generalization (hops 7-8) vs trained hops (1-6)
  - Connected/not-connected error typology
  - Comparison summary table across files (rows = experiments)

Usage:
    python -m examples.graph_3.evaluation.analyze results.json
    python -m examples.graph_3.evaluation.analyze \\
        outputs_graph3/exp0_icl/results.json \\
        outputs_graph3/exp1_adaptive/eval/results.json \\
        outputs_graph3/exp2_plain/eval/results.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def _load(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def _is_parsed(r: dict) -> bool:
    return r.get("predicted_answer") is not None


def _acc(rows: list[dict]) -> float:
    return sum(r["correct"] for r in rows) / len(rows) if rows else float("nan")


def _fid(rows: list[dict]) -> float:
    return sum(r["fidelity"] for r in rows) / len(rows) if rows else float("nan")


def _acc_parsed(rows: list[dict]) -> float:
    parsed = [r for r in rows if _is_parsed(r)]
    return _acc(parsed)


def _frac_unparsed(rows: list[dict]) -> float:
    return sum(1 for r in rows if not _is_parsed(r)) / len(rows) if rows else float("nan")


def _mae(rows: list[dict]) -> float:
    errs = [r["abs_err"] for r in rows if r.get("abs_err") is not None]
    return sum(errs) / len(errs) if errs else float("nan")


def _label(path: str) -> str:
    for part in Path(path).parts:
        if part.startswith("exp"):
            return part
    return Path(path).parent.name


def _pct(v: float) -> str:
    return "  n/a" if v != v else f"{v:.1%}"


def _num(v: float) -> str:
    return "  n/a" if v != v else f"{v:.2f}"


def _buckets(rows: list[dict]) -> list[str]:
    return sorted({r["n_bucket"] for r in rows}, key=lambda x: (x == "none", x))


def analyze_run(results: list[dict], label: str = "") -> None:
    total = len(results)
    print(f"\n=== {label or 'RESULTS'} ===")
    print(f"Overall  acc {_pct(_acc(results))}  fidelity {_pct(_fid(results))}  "
          f"unparsed {_pct(_frac_unparsed(results))}  "
          f"acc(parsed) {_pct(_acc_parsed(results))}  MAE {_num(_mae(results))}  N={total}")

    print(f"\n{'bucket':>7} {'N':>5} {'acc':>7} {'fidelity':>9} {'acc_par':>8} {'unparsed':>9} {'MAE':>6}")
    print("-" * 56)
    for b in _buckets(results):
        rows = [r for r in results if r["n_bucket"] == b]
        print(f"{b:>7} {len(rows):>5} {_pct(_acc(rows)):>7} {_pct(_fid(rows)):>9} "
              f"{_pct(_acc_parsed(rows)):>8} {_pct(_frac_unparsed(rows)):>9} {_num(_mae(rows)):>6}")

    trained = [r for r in results if r["n_bucket"] in tuple("123456")]
    gen     = [r for r in results if r["n_bucket"] in ("7", "8")]
    if trained and gen:
        print(f"\nTrained hops 1-6: acc {_pct(_acc(trained))}   ·   "
              f"Generalization hops 7-8: acc {_pct(_acc(gen))}")

    false_nc = sum(
        1 for r in results
        if r["gold_answer"] != "not connected" and r.get("predicted_answer") == "not connected"
    )
    false_conn = sum(
        1 for r in results
        if r["gold_answer"] == "not connected"
        and r.get("predicted_answer") not in (None, "not connected")
    )
    print(f"false 'not connected': {false_nc}   ·   false connection: {false_conn}")

    errors = [r for r in results if not r["correct"]][:3]
    if errors:
        print("\nSample errors:")
        for e in errors:
            print(f"  [{e['n_bucket']}] {e['x']} ↔ {e['y']}: "
                  f"gold={e['gold_answer']}  pred={e.get('predicted_answer')}")


def compare_summary(files: list[str]) -> None:
    if len(files) < 2:
        return
    all_results = [(p, _load(p)) for p in files]
    all_buckets = sorted(
        {r["n_bucket"] for _, rs in all_results for r in rs},
        key=lambda x: (x == "none", x),
    )

    b_headers = [f"n={b:>4}" for b in all_buckets]
    header = (f"{'experiment':<16} {'acc':>7} {'fid':>7} {'unp':>6} {'MAE':>6}  "
              + " ".join(b_headers))
    print(f"\n{'=' * len(header)}")
    print("COMPARISON  (acc = exact answer · fid = answer+path · unp = unparsed · per-bucket = acc)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for path, rs in all_results:
        row = (f"{_label(path):<16} {_pct(_acc(rs)):>7} {_pct(_fid(rs)):>7} "
               f"{_pct(_frac_unparsed(rs)):>6} {_num(_mae(rs)):>6}  ")
        for b in all_buckets:
            row += f" {_pct(_acc([r for r in rs if r['n_bucket'] == b])):>7}"
        print(row)
    print("=" * len(header))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="Path(s) to results.json")
    ap.add_argument("--no-detail", action="store_true",
                    help="Skip per-file detail, show comparison only")
    args = ap.parse_args()

    if not args.no_detail:
        for path in args.files:
            analyze_run(_load(path), _label(path))

    compare_summary(args.files)


if __name__ == "__main__":
    main()
