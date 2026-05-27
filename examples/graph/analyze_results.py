"""
Stats over result_*.json produced by graph_eval.py (--output).

Usage:
    python examples/graph/analyze_results.py path/to/result.json
    python examples/graph/analyze_results.py result1.json result2.json   # compare
    python examples/graph/analyze_results.py result.json --top-rel 30 --n-errors 20
    python examples/graph/analyze_results.py result.json --save-csv out.csv
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

LETTERS = list("ABCDE")


def load_results(path: Path) -> list[list[dict]]:
    """Return list of runs (each run = list of result dicts)."""
    data = json.loads(path.read_text())
    if not data:
        return []
    # Single-run = list[dict]; multi-run = list[list[dict]]
    if isinstance(data[0], dict):
        return [data]
    return data


def acc(rs: list[dict]) -> float:
    return sum(r["correct"] for r in rs) / len(rs) if rs else 0.0


def fmt_pct(n: int, total: int) -> str:
    return f"{n}/{total} = {100 * n / total:.2f}%" if total else "0/0"


def header(s: str):
    print(f"\n{'═' * 70}\n{s}\n{'═' * 70}")


def section(s: str):
    print(f"\n── {s} " + "─" * (66 - len(s)))


def analyze_run(results: list[dict], run_idx: int = 0, n_runs: int = 1,
                top_rel: int = 0, n_errors: int = 8, n_worst: int = 15):
    if not results:
        print("empty run")
        return

    n = len(results)
    n_corr = sum(r["correct"] for r in results)
    header(f"RUN {run_idx + 1}/{n_runs}  —  N={n}  accuracy={fmt_pct(n_corr, n)}")

    # ── per-category ────────────────────────────────────────────────────────
    section("per category")
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_cat[str(r["category"])].append(r)
    print(f"{'cat':<5}{'N':>6}{'correct':>10}{'acc':>10}")
    for cat in sorted(by_cat, key=lambda c: (c[0], c)):
        rs = by_cat[cat]
        c = sum(r["correct"] for r in rs)
        print(f"{cat:<5}{len(rs):>6}{c:>10}{acc(rs) * 100:>9.2f}%")

    # ── per-relation ────────────────────────────────────────────────────────
    section("per relation")
    by_rel: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_rel[r["rel"]].append(r)
    print(f"{'rel':<22}{'N':>6}{'correct':>10}{'acc':>10}")
    rows = []
    for rel, rs in by_rel.items():
        rows.append((rel, len(rs), sum(r["correct"] for r in rs), acc(rs)))
    rows.sort(key=lambda x: -x[3])
    shown = rows if top_rel <= 0 else rows[:top_rel]
    for rel, total, c, a in shown:
        print(f"{rel:<22}{total:>6}{c:>10}{a * 100:>9.2f}%")

    # ── letter distribution ────────────────────────────────────────────────
    section("letter distribution")
    pred = Counter(r["predicted_letter"] or "?" for r in results)
    gold = Counter(r["correct_letter"] for r in results)
    print(f"{'letter':<8}{'gold':>8}{'pred':>8}{'gold%':>8}{'pred%':>8}")
    keys = sorted(set(pred) | set(gold))
    for k in keys:
        g = gold.get(k, 0)
        p = pred.get(k, 0)
        print(f"{k:<8}{g:>8}{p:>8}{g / n * 100:>7.2f}%{p / n * 100:>7.2f}%")

    # ── confusion matrix gold → pred ────────────────────────────────────────
    section("confusion gold × pred (rows = gold, cols = pred, '.' = none)")
    cols = LETTERS + ["?"]
    cm: dict[str, Counter] = {g: Counter() for g in LETTERS}
    for r in results:
        g = r["correct_letter"]
        p = r["predicted_letter"] or "?"
        if g in cm:
            cm[g][p] += 1
    print(" " * 6 + "".join(f"{c:>6}" for c in cols))
    for g in LETTERS:
        row = cm[g]
        cells = "".join(f"{row.get(c, 0):>6}" for c in cols)
        total_g = sum(row.values())
        diag_pct = (row.get(g, 0) / total_g * 100) if total_g else 0.0
        print(f"{g:<6}{cells}   diag={diag_pct:.1f}%")

    # ── hardest persons ─────────────────────────────────────────────────────
    section("worst persons (acc < 50%, ≥3 questions)")
    by_person: dict[str, list[dict]] = defaultdict(list)
    for r in results:
        by_person[r["person"]].append(r)
    worst = []
    for p, rs in by_person.items():
        if len(rs) >= 3:
            a = acc(rs)
            if a < 0.5:
                worst.append((p, len(rs), sum(r["correct"] for r in rs), a))
    worst.sort(key=lambda x: x[3])
    if not worst:
        print("  (none)")
    else:
        for p, total, c, a in worst[:n_worst]:
            print(f"  {p:<20} {c}/{total}  {a * 100:>5.1f}%")

    # ── sample errors ──────────────────────────────────────────────────────
    section(f"sample errors (first {n_errors})")
    errs = [r for r in results if not r["correct"]]
    print(f"total errors: {len(errs)}  ({len(errs) / n * 100:.1f}%)")
    for r in errs[:n_errors]:
        q = r["question"][:80].replace("\n", " ")
        pl = r["predicted_letter"] or "?"
        print(f"  [{r['category']}|{r['rel']:<14}] {q}")
        print(f"      gold={r['correct_letter']}  pred={pl}  raw={r['predicted']!r:.120}")

    # ── empty / malformed predictions ──────────────────────────────────────
    section("malformed predictions")
    empty = [r for r in results if not r["predicted_letter"]]
    print(f"no letter extracted: {fmt_pct(len(empty), n)}")
    if empty:
        for r in empty[:5]:
            print(f"  [{r['rel']:<14}] raw={r['predicted']!r:.120}")


def stability(runs: list[list[dict]]):
    if len(runs) < 2:
        return
    accs = [acc(r) for r in runs]
    mean = sum(accs) / len(accs)
    var = sum((a - mean) ** 2 for a in accs) / len(accs)
    std = var ** 0.5
    header(f"STABILITY across {len(runs)} runs")
    print(f"acc per run: {[f'{a * 100:.2f}%' for a in accs]}")
    print(f"mean: {mean * 100:.2f}%   std: {std * 100:.2f}%   "
          f"min: {min(accs) * 100:.2f}%   max: {max(accs) * 100:.2f}%")


def per_rel_csv(all_files: dict[str, list[list[dict]]], out_path: Path):
    """Per-relation accuracy table across files. Row = rel, col = file."""
    rels: set[str] = set()
    flat: dict[str, list[dict]] = {}
    for label, runs in all_files.items():
        merged = [r for run in runs for r in run]
        flat[label] = merged
        rels.update(r["rel"] for r in merged)

    with out_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["rel"] + list(all_files.keys()))
        for rel in sorted(rels):
            row = [rel]
            for label, merged in flat.items():
                rs = [r for r in merged if r["rel"] == rel]
                row.append(f"{acc(rs) * 100:.2f}" if rs else "")
            w.writerow(row)
        # overall row
        w.writerow(["__overall__"] + [f"{acc(m) * 100:.2f}" for m in flat.values()])
    print(f"\ncsv → {out_path}")


def compare_summary(all_files: dict[str, list[list[dict]]]):
    header(f"COMPARE  ({len(all_files)} files)")
    print(f"{'file':<40}{'N':>8}{'acc':>10}")
    for label, runs in all_files.items():
        merged = [r for run in runs for r in run]
        print(f"{label:<40}{len(merged):>8}{acc(merged) * 100:>9.2f}%")


def main():
    ap = argparse.ArgumentParser(description="Stats over graph_eval.py result JSON")
    ap.add_argument("paths", nargs="+", type=Path, help="result JSON file(s)")
    ap.add_argument("--top-rel", type=int, default=0,
                    help="Show only top-N relations per run (0 = all)")
    ap.add_argument("--n-errors", type=int, default=8,
                    help="Number of sample errors to print per run")
    ap.add_argument("--n-worst", type=int, default=15,
                    help="Number of worst persons to print per run")
    ap.add_argument("--save-csv", type=Path, default=None,
                    help="Write per-relation accuracy CSV across input files")
    ap.add_argument("--no-detail", action="store_true",
                    help="Skip per-run detail; only print compare summary")
    args = ap.parse_args()

    all_files: dict[str, list[list[dict]]] = {}
    for path in args.paths:
        if not path.exists():
            raise SystemExit(f"file not found: {path}")
        runs = load_results(path)
        if not runs:
            print(f"!! empty: {path}")
            continue
        all_files[path.name] = runs

        if args.no_detail:
            continue

        print(f"\n████ {path} ████")
        for i, run in enumerate(runs):
            analyze_run(
                run, i, len(runs),
                top_rel=args.top_rel,
                n_errors=args.n_errors,
                n_worst=args.n_worst,
            )
        stability(runs)

    if len(all_files) > 1:
        compare_summary(all_files)

    if args.save_csv:
        per_rel_csv(all_files, args.save_csv)


if __name__ == "__main__":
    main()
