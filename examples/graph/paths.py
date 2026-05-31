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
