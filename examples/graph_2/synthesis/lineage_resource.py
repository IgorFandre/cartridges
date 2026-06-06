"""
LineageGraphResource — a custom Resource for Exp-1 self-study synthesis.

`sample_prompt(batch_size)` returns:
  ctx  = graph corpus text + a pre-computed lineal walk for a sampled pair
  seeds = lineage-flavored seed prompts (Bot A instruction)

Bot A (the question-asker) receives this as the system context
({subcorpus}) and is nudged to ask a lineage-shaped question.
Bot B (the answerer) replies with top-k logprobs for distillation.
"""
from __future__ import annotations
import random
from pathlib import Path
from typing import List

from cartridges.data.resources import Resource
from examples.graph.data_gen.family_tree import FamilyTree
from examples.graph_2.data_gen.lineage_index import LineageIndex

# ── Lineage seed prompts ───────────────────────────────────────────────────────
_SEED_PROMPTS = [
    "Generate a single yes/no question asking whether one person is an ancestor of another "
    "at a specific number of generations, using the names from the family tree above. "
    "Output only the question.",

    "Using the worked lineage path above as a hint, ask a yes/no question of the form "
    "'Is {X} an ancestor of {Y}, N generation(s) above?' "
    "Choose X and Y from the family tree. Output only the question.",

    "Ask a yes/no question of the form 'Is {X} a descendant of {Y}, N generation(s) below?' "
    "using names from the family tree. Output only the question.",

    "Generate a question about whether two people in the family tree are in a direct "
    "ancestor-descendant relationship at a given generational distance. "
    "The answer should be Yes or No. Output only the question.",

    "Based on the lineage path shown above, write a verification question: "
    "'Is [person] an ancestor/descendant of [person] at [n] generations?' "
    "Output only the question.",
]


def sample_lineage_seed_prompts(batch_size: int) -> List[str]:
    """Return `batch_size` lineage seed prompt strings."""
    return random.choices(_SEED_PROMPTS, k=batch_size)


# ── Resource subclass ─────────────────────────────────────────────────────────
class LineageGraphResource(Resource):
    """Resource that pairs the full graph corpus with a BFS lineal path."""

    class Config(Resource.Config):
        # MUST subclass Resource.Config: SelfStudySynthesizer.Config.resources is
        # typed List[Resource.Config], so a bare ObjectConfig fails validation.
        _pass_as_config = True
        tree_path: str
        corpus_path: str | None = None
        min_distance: int = 1
        max_distance: int | None = None

    def __init__(self, config: "LineageGraphResource.Config"):
        self.config = config
        self.tree:       FamilyTree | None = None
        self.index:      LineageIndex | None = None
        self.graph_text: str = ""
        self.pairs_by_d: dict[int, list[tuple[str, str]]] = {}

    async def setup(self):
        self.tree = FamilyTree.load(self.config.tree_path)
        self.index = LineageIndex.from_tree(self.tree)
        if self.config.corpus_path:
            self.graph_text = Path(self.config.corpus_path).read_text()
        else:
            self.graph_text = self.tree.to_text()

        max_d = self.config.max_distance or self.index.max_distance()
        all_pairs = self.index.by_distance()
        self.pairs_by_d = {
            d: pairs
            for d, pairs in all_pairs.items()
            if self.config.min_distance <= d <= max_d
        }
        if not self.pairs_by_d:
            raise ValueError(
                f"No lineal pairs in distance range "
                f"[{self.config.min_distance}, {max_d}]"
            )

    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        # Pick a random distance then a random pair at that distance
        d = random.choice(list(self.pairs_by_d.keys()))
        ancestor, descendant = random.choice(self.pairs_by_d[d])

        # Build the lineal walk narration
        walk = self.index.lineal_walk_text(ancestor, descendant) or (
            f"{ancestor} is {d} generation(s) above {descendant}."
        )

        ctx = (
            self.graph_text
            + f"\n\nWorked lineage path:\n{walk}\n"
            + f"({ancestor} is an ancestor of {descendant}, "
            f"{d} generation{'s' if d != 1 else ''} above; "
            f"equivalently {descendant} is a descendant of {ancestor}, "
            f"{d} generation{'s' if d != 1 else ''} below.)"
        )

        seeds = sample_lineage_seed_prompts(batch_size)
        return ctx, seeds
