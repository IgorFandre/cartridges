"""
Central path definitions for the graph_3 (handshake-graph cartridge) experiments.

All data/result paths are defined here. Import from this module — never
hardcode paths in other modules.

Layout:
    examples/graph_3/
    ├── paths.py                 ← this file
    ├── data/base/               generated forest, QA parquets
    └── ...

Results live under repo-root `outputs_graph3/`
(override with $CARTRIDGES_OUTPUT_DIR_GRAPH3).
"""
import os
import re
from pathlib import Path

GRAPH3_ROOT = Path(__file__).resolve().parent
REPO_ROOT   = GRAPH3_ROOT.parent.parent

# ── Data ─────────────────────────────────────────────────────────────────────
DATA_DIR = GRAPH3_ROOT / "data"
BASE_DIR = DATA_DIR / "base"

FOREST_JSON        = BASE_DIR / "forest.json"
CORPUS_TXT         = BASE_DIR / "corpus.txt"
BASE_TRAIN_PARQUET = BASE_DIR / "train_handshake.parquet"
BASE_TEST_PARQUET  = BASE_DIR / "test_handshake.parquet"
BASE_TRAIN_META    = BASE_DIR / "train_meta.json"
BASE_TEST_META     = BASE_DIR / "test_meta.json"
BASE_SPLIT_META    = BASE_DIR / "split_meta.json"

# ── Results ───────────────────────────────────────────────────────────────────
OUTPUTS_DIR = Path(
    os.environ.get("CARTRIDGES_OUTPUT_DIR_GRAPH3", str(REPO_ROOT / "outputs_graph3"))
)
EXP0_DIR = OUTPUTS_DIR / "exp0_icl"
EXP1_DIR = OUTPUTS_DIR / "exp1_adaptive"
EXP2_DIR = OUTPUTS_DIR / "exp2_plain"
EXP3_DIR = OUTPUTS_DIR / "exp3_rejection"


# ── Checkpoint discovery ──────────────────────────────────────────────────────
def run_checkpoints(output_dir) -> list[tuple[int, Path]]:
    """All cache-step{N}.pt under output_dir, sorted by step."""
    hits = []
    for p in Path(output_dir).rglob("cache-step*.pt"):
        m = re.search(r"cache-step(\d+)\.pt$", p.name)
        if m:
            hits.append((int(m.group(1)), p))
    hits.sort(key=lambda x: x[0])
    return hits


def latest_checkpoint(output_dir) -> Path:
    """Most-recently-written cache_last.pt under output_dir."""
    hits = sorted(
        Path(output_dir).rglob("cache_last.pt"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not hits:
        raise FileNotFoundError(f"no cache_last.pt under {output_dir}")
    return hits[0]
