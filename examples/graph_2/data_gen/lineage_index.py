"""
LineageIndex — authoritative source of lineal ancestor distances + paths.

Lineal-only: considers only directed parent→child chains, ignoring
siblings/spouses/in-laws. This is the gold-label source for the QA generator
and the BFS path provider for Exp-1 self-study.

Usage:
    from examples.graph_2.data_gen.lineage_index import LineageIndex
    from examples.graph.data_gen.family_tree import FamilyTree

    ft = FamilyTree.load("family_tree.json")
    idx = LineageIndex.from_tree(ft)
    print(idx.ancestor_distance("AnchorM0", "AnchorM8"))  # → 8
"""
from __future__ import annotations
from collections import defaultdict, deque


class LineageIndex:
    """Precomputed lineal ancestor/descendant distances for a family tree.

    Distances are the number of parent→child hops; X is an ancestor of Y
    at distance n iff there is a directed path of n edges from X down to Y.
    """

    def __init__(self, parent_child: list[dict]):
        self.children_of: dict[str, list[str]] = defaultdict(list)
        self.parents_of:  dict[str, list[str]] = defaultdict(list)

        for e in parent_child:
            p, c = e["parent"], e["child"]
            self.children_of[p].append(c)
            self.parents_of[c].append(p)

        self.people: list[str] = sorted(
            {e["parent"] for e in parent_child} | {e["child"] for e in parent_child}
        )

        # (ancestor, descendant) → min parent-hop distance
        self._anc_dist: dict[tuple[str, str], int] = {}
        # (ancestor, descendant) → predecessor in BFS (for path reconstruction)
        self._pred: dict[tuple[str, str], str | None] = {}
        self._build()

    @classmethod
    def from_tree(cls, tree) -> "LineageIndex":
        """Construct from a FamilyTree object (uses tree.parent_child)."""
        return cls(tree.parent_child)

    def _build(self) -> None:
        """BFS downward from each node to record all descendant distances."""
        for src in self.people:
            queue: deque[tuple[str, int, str | None]] = deque()
            queue.append((src, 0, None))
            visited: set[str] = {src}
            while queue:
                node, dist, pred = queue.popleft()
                if node != src:
                    key = (src, node)
                    # Keep the minimum distance (handles rare cousin-marriage paths)
                    if key not in self._anc_dist or dist < self._anc_dist[key]:
                        self._anc_dist[key] = dist
                        self._pred[key] = pred
                for child in self.children_of.get(node, []):
                    if child not in visited:
                        visited.add(child)
                        queue.append((child, dist + 1, node))

    # ── Query interface ───────────────────────────────────────────────────────
    def ancestor_distance(self, ancestor: str, descendant: str) -> int | None:
        """Number of parent-hops from `ancestor` down to `descendant`, or None."""
        return self._anc_dist.get((ancestor, descendant))

    def descendant_distance(self, descendant: str, ancestor: str) -> int | None:
        """Alias: distance FROM ancestor DOWN TO descendant (same as ancestor_distance)."""
        return self.ancestor_distance(ancestor, descendant)

    def path(self, ancestor: str, descendant: str) -> list[str] | None:
        """Reconstructed lineal chain [ancestor, ..., descendant], or None."""
        if (ancestor, descendant) not in self._anc_dist:
            return None
        chain: list[str] = [descendant]
        cur = descendant
        while cur != ancestor:
            prev = self._pred.get((ancestor, cur))
            if prev is None:
                return None
            chain.append(prev)
            cur = prev
        chain.reverse()
        return chain

    def triples(self) -> list[tuple[str, str, int]]:
        """All (ancestor, descendant, distance) triples with distance >= 1."""
        return [(a, d, dist) for (a, d), dist in self._anc_dist.items()]

    def by_distance(self) -> dict[int, list[tuple[str, str]]]:
        """Map distance n → list of (ancestor, descendant) lineal pairs."""
        out: dict[int, list[tuple[str, str]]] = defaultdict(list)
        for (a, d), dist in self._anc_dist.items():
            out[dist].append((a, d))
        return dict(out)

    def max_distance(self) -> int:
        """Longest lineal chain (should equal tree_depth)."""
        return max(self._anc_dist.values(), default=0)

    def non_lineal_pairs(self, k: int, rng) -> list[tuple[str, str]]:
        """k ordered (a, b) pairs where neither is a lineal ancestor of the other."""
        lineal: set[tuple[str, str]] = set(self._anc_dist.keys())
        # also exclude (a, a)
        pool: list[tuple[str, str]] = [
            (a, b)
            for a in self.people
            for b in self.people
            if a != b and (a, b) not in lineal and (b, a) not in lineal
        ]
        if len(pool) <= k:
            return pool
        return rng.sample(pool, k)

    def lineal_walk_text(self, ancestor: str, descendant: str) -> str | None:
        """Human-readable step-by-step path: 'A is a parent of B. B is a parent of C. ...'"""
        chain = self.path(ancestor, descendant)
        if chain is None:
            return None
        n = len(chain) - 1
        sentences = [
            f"{chain[i]} is a parent of {chain[i+1]}."
            for i in range(n)
        ]
        sentences.append(
            f"So {ancestor} is {n} generation{'s' if n != 1 else ''} above {descendant}."
        )
        return " ".join(sentences)
