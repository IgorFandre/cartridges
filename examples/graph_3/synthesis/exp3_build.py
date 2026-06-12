"""
Exp 3 — rejection-sampling self-study.  PLAN.md §4.

Exp 3 = Exp 2 (plain self-study) with the wrong traces removed: keep only the
conversations whose recorded `correct` flag is True.  NO SERVER needed — this is
a pure filter over the exp2_plain artifact, so it can be built any time after
exp2 synthesis without re-querying the model (PLAN §4: "Exp 3 … добавляется
тривиально поверх Exp 2").

This isolates the classic STaR / rejection-sampling recipe — "train only on the
model's CORRECT unaided traces".  The per-bucket kept counts expose its known
failure mode: on hard questions the model is almost never right, so deep hops
collapse to near-zero training signal (graph_2 REPORT §3).  That collapse is the
result, not a bug — the filter is faithful and the report makes it explicit.

Input:
  {outputs_graph3}/exp2_plain/artifact/dataset.parquet   (carries `correct` meta)
Output:
  {outputs_graph3}/exp3_rejection/artifact/dataset.parquet
  {outputs_graph3}/exp3_rejection/filter_report.json

Usage:
    python -m examples.graph_3.synthesis.exp3_build
    python -m examples.graph_3.synthesis.exp3_build \\
        --source outputs_graph3/exp2_plain/artifact/dataset.parquet
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path

from examples.graph_3 import paths
from examples.graph_3.synthesis.common import save_report


def _is_correct(meta: dict) -> bool:
    """Robustly read the `correct` flag (bool, or stringified by parquet)."""
    v = (meta or {}).get("correct")
    return v in (True, 1, "True", "true", "1")


def build_exp3(source: Path, output_dir: Path) -> Path:
    from cartridges.structs import read_conversations, write_conversations

    convos = read_conversations(str(source))
    report: dict[str, dict[str, int]] = defaultdict(lambda: {"total": 0, "kept": 0})

    kept = []
    for c in convos:
        b = str((c.metadata or {}).get("n_bucket"))
        report[b]["total"] += 1
        if _is_correct(c.metadata):
            report[b]["kept"] += 1
            kept.append(c)

    if not kept:
        raise RuntimeError(
            f"No correct traces in {source} — exp3 would be empty. "
            "Check that exp2 synthesis recorded the `correct` metadata flag."
        )

    artifact_dir = output_dir / "exp3_rejection" / "artifact"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = artifact_dir / "dataset.parquet"
    write_conversations(kept, str(parquet_path))

    n_missing_lp = sum(1 for c in kept if c.messages[-1].top_logprobs is None)
    if n_missing_lp:
        print(
            f"WARNING: {n_missing_lp}/{len(kept)} kept traces have no top_logprobs "
            "— train them with TARGETS=tokens, not the default logit distillation."
        )

    report_out = {
        bucket: {**counts, "keep_rate": round(counts["kept"] / max(1, counts["total"]), 3)}
        for bucket, counts in sorted(report.items(), key=lambda x: (x[0] == "none", x[0]))
    }
    save_report(report_out, output_dir / "exp3_rejection" / "filter_report.json")

    print(f"\nKept {len(kept)}/{len(convos)} correct traces → {parquet_path}")
    print("Per bucket: kept / total (keep rate):")
    empty = []
    for bucket, s in report_out.items():
        flag = "  ← EMPTY" if s["kept"] == 0 else ""
        print(f"  n={bucket}: {s['kept']}/{s['total']} = {s['keep_rate']:.1%}{flag}")
        if s["kept"] == 0:
            empty.append(bucket)
    if empty:
        print(
            f"\nNOTE: buckets {empty} have ZERO correct traces — the rejection filter "
            "starves the deep hops. The exp3 cartridge will see no training signal "
            "there. This is the expected rejection-sampling collapse (PLAN §4)."
        )
    return parquet_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=str,
                    default=str(paths.EXP2_DIR / "artifact" / "dataset.parquet"),
                    help="exp2_plain dataset.parquet to filter")
    ap.add_argument("--output-dir", type=str, default=str(paths.OUTPUTS_DIR))
    args = ap.parse_args()

    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(
            f"exp2 dataset not found: {source}\n"
            "Run exp2 synthesis first:  python -m examples.graph_3.synthesis.exp2_synthesize"
        )
    build_exp3(source, Path(args.output_dir))


if __name__ == "__main__":
    main()
