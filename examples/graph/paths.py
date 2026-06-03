"""
Central path definitions for the graph (kinship cartridge) experiments.

Every module imports its data locations from here instead of computing them
from `Path(__file__).parent`, so moving code between subpackages never breaks
data routing.

Layout:
    examples/graph/
    ├── paths.py            ← this file (GRAPH_ROOT marker)
    ├── data/
    │   ├── base/           base tree, corpora, base parquets
    │   └── variants/       {alex,ben,carl,dan}/ + variants_meta.json
    └── ...

Results live OUTSIDE the package, under the repo-root `outputs_graph/`
(override with $CARTRIDGES_OUTPUT_DIR_GRAPH).
"""
import os
from pathlib import Path

# examples/graph/  (this file sits at the package root)
GRAPH_ROOT = Path(__file__).resolve().parent
REPO_ROOT = GRAPH_ROOT.parent.parent

# ── Data ─────────────────────────────────────────────────────────────────────
DATA_DIR = GRAPH_ROOT / "data"
BASE_DIR = DATA_DIR / "base"
VARIANTS_DIR = DATA_DIR / "variants"
VARIANTS_META = VARIANTS_DIR / "variants_meta.json"

# Base-tree artifacts (single, un-swapped tree)
BASE_TREE_JSON = BASE_DIR / "family_tree.json"
BASE_CORPUS = BASE_DIR / "family_tree_corpus.txt"
BASE_NARRATIVE = BASE_DIR / "family_tree_narrative.txt"
BASE_TRAIN_PARQUET = BASE_DIR / "train_mc.parquet"
BASE_TEST_PARQUET = BASE_DIR / "test_mc.parquet"
BASE_TRAIN_META = BASE_DIR / "train_meta_mc.json"
BASE_TEST_META = BASE_DIR / "test_meta_mc.json"
BASE_SPLIT_META = BASE_DIR / "split_meta.json"

# ── Results (outside the package) ────────────────────────────────────────────
OUTPUTS_DIR = Path(
    os.environ.get("CARTRIDGES_OUTPUT_DIR_GRAPH", str(REPO_ROOT / "outputs_graph"))
)
EXP1_DIR = OUTPUTS_DIR / "exp1_init_kv"
EXP2_DIR = OUTPUTS_DIR / "exp2_train"
EXP3_DIR = OUTPUTS_DIR / "exp3_icl"
EXP4_DIR = OUTPUTS_DIR / "exp4_stability"


# ── Variant helpers ──────────────────────────────────────────────────────────
def variant_dir(name: str) -> Path:
    """Data directory for a named variant, e.g. variant_dir('alex')."""
    return VARIANTS_DIR / name.lower()


def variant_corpus(name: str) -> Path:
    return variant_dir(name) / "family_tree_corpus.txt"


def variant_train(name: str) -> Path:
    return variant_dir(name) / "train_mc.parquet"


def variant_test(name: str) -> Path:
    return variant_dir(name) / "test_mc.parquet"


# ── Run output dirs (the `output_dir` BEFORE pydrantic adds <launch>/<run>/) ──
# train.py passes these as TrainConfig.output_dir; pydrantic then nests the
# actual run under <output_dir>/<launch_id>/<run_id>/, where the checkpoints
# (cache-step{N}.pt, cache_last.pt, config.yaml) live. Discovery helpers below
# rglob through that nesting so callers never hard-code the timestamp/uuid.
def exp2_variant_dir(variant: str) -> Path:
    """exp2 cartridge output_dir for a variant (or 'base')."""
    return EXP2_DIR / variant.lower()


def exp2_compare_dir() -> Path:
    """Canonical out-dir for the exp2 trained-KV comparison."""
    return EXP2_DIR / "compare"


def exp4_run_dir(init: str, data: str, run_idx: int) -> Path:
    """exp4 stability output_dir for one run (run_idx is 1-based, as train.py names it)."""
    return EXP4_DIR / f"{init.lower()}_on_{data.lower()}_run{run_idx}"


def dynamics_dir(variant: str) -> Path:
    """Canonical out-dir for comparison/dynamics.py rotation analysis of a variant."""
    return EXP2_DIR / variant.lower() / "dynamics"


# ── Checkpoint discovery (digs through pydrantic's <launch_id>/<run_id>/) ─────
def run_checkpoints(output_dir) -> list[tuple[int, Path]]:
    """All cache-step{N}.pt under output_dir, sorted by step (ascending)."""
    import re
    hits = []
    for p in Path(output_dir).rglob("cache-step*.pt"):
        m = re.search(r"cache-step(\d+)\.pt$", p.name)
        if m:
            hits.append((int(m.group(1)), p))
    hits.sort(key=lambda x: x[0])
    return hits


def latest_checkpoint(output_dir) -> Path:
    """Most-recently-written cache_last.pt under output_dir (newest mtime)."""
    hits = sorted(Path(output_dir).rglob("cache_last.pt"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    if not hits:
        raise FileNotFoundError(f"no cache_last.pt under {output_dir}")
    return hits[0]
