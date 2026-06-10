"""
GraphIndex — authoritative source of handshake distances, unique paths and
BFS scratchpads for the friendship forest.

The graph is an undirected forest (each connected component is a tree), so the
path between any two connected people is UNIQUE — both the distance and the
chain are deterministic gold labels.

Also home of the push/pop BFS scratchpad generator (the Exp-1 hint format and
the canonical reasoning format of the dataset, see PLAN.md §3).

Usage:
    from examples.graph_3.data_gen.graph_index import GraphIndex

    idx = GraphIndex.load("forest.json")
    idx.distance("Alice", "Grace")     # → 3 or None
    idx.path("Alice", "Grace")         # → ["Alice", "Carol", "Eve", "Grace"]
    idx.scratchpad("Alice", "Grace")   # → push/pop BFS log ending in "Answer: 3"
"""
from __future__ import annotations
import json
import random
from collections import defaultdict, deque
from pathlib import Path


class GraphIndex:
    """Precomputed all-pairs distances for an undirected forest."""

    def __init__(self, edges: list[dict]):
        """edges: list of {"a": name, "b": name} undirected friendship edges."""
        self.neighbors: dict[str, list[str]] = defaultdict(list)
        for e in edges:
            a, b = e["a"], e["b"]
            self.neighbors[a].append(b)
            self.neighbors[b].append(a)

        self.people: list[str] = sorted(self.neighbors)

        # name → component id; component id → sorted member list
        self.component_of: dict[str, int] = {}
        self.components: list[list[str]] = []
        self._label_components()

        # (x, y) → distance, for x < y (undirected, stored once per pair)
        self._dist: dict[tuple[str, str], int] = {}
        self._build_distances()

    # ── Construction ───────────────────────────────────────────────────────────
    @classmethod
    def load(cls, forest_path: str | Path) -> "GraphIndex":
        forest = json.loads(Path(forest_path).read_text())
        return cls(forest["edges"])

    def _label_components(self) -> None:
        seen: set[str] = set()
        for start in self.people:
            if start in seen:
                continue
            cid = len(self.components)
            members = []
            queue = deque([start])
            seen.add(start)
            while queue:
                node = queue.popleft()
                members.append(node)
                self.component_of[node] = cid
                for nb in self.neighbors[node]:
                    if nb not in seen:
                        seen.add(nb)
                        queue.append(nb)
            self.components.append(sorted(members))

    def _build_distances(self) -> None:
        """BFS from each node; store each unordered pair once (x < y)."""
        for src in self.people:
            dist = {src: 0}
            queue = deque([src])
            while queue:
                node = queue.popleft()
                for nb in self.neighbors[node]:
                    if nb not in dist:
                        dist[nb] = dist[node] + 1
                        queue.append(nb)
            for node, d in dist.items():
                if src < node:
                    self._dist[(src, node)] = d

    # ── Query interface ────────────────────────────────────────────────────────
    def distance(self, x: str, y: str) -> int | None:
        """Handshake distance between x and y, or None if not connected."""
        if x == y:
            return 0
        return self._dist.get((x, y) if x < y else (y, x))

    def connected(self, x: str, y: str) -> bool:
        return self.component_of.get(x) == self.component_of.get(y)

    def path(self, x: str, y: str) -> list[str] | None:
        """The unique path [x, ..., y], or None if not connected."""
        if not self.connected(x, y):
            return None
        if x == y:
            return [x]
        pred: dict[str, str] = {}
        queue = deque([x])
        seen = {x}
        while queue:
            node = queue.popleft()
            if node == y:
                break
            for nb in self.neighbors[node]:
                if nb not in seen:
                    seen.add(nb)
                    pred[nb] = node
                    queue.append(nb)
        chain = [y]
        while chain[-1] != x:
            chain.append(pred[chain[-1]])
        chain.reverse()
        return chain

    def pairs_by_distance(self, max_distance: int | None = None) -> dict[int, list[tuple[str, str]]]:
        """Map distance d → list of unordered (x, y) pairs (x < y), d >= 1."""
        out: dict[int, list[tuple[str, str]]] = defaultdict(list)
        for (x, y), d in self._dist.items():
            if max_distance is None or d <= max_distance:
                out[d].append((x, y))
        return dict(out)

    def max_distance(self) -> int:
        return max(self._dist.values(), default=0)

    def non_connected_pairs(self, k: int, rng: random.Random) -> list[tuple[str, str]]:
        """k unordered (x, y) pairs from different components."""
        pool: list[tuple[str, str]] = [
            (x, y)
            for i, x in enumerate(self.people)
            for y in self.people[i + 1:]
            if self.component_of[x] != self.component_of[y]
        ]
        if len(pool) <= k:
            return pool
        return rng.sample(pool, k)

    # ── Scratchpad (push/pop BFS log) ─────────────────────────────────────────
    def scratchpad(self, x: str, y: str, rng: random.Random | None = None) -> str:
        """Push/pop BFS queue log from x searching for y (PLAN.md §3).

        Each queue element carries its distance: `Bob(1)`. Neighbor expansion
        order is randomized per call (pass a seeded rng for determinism) so the
        format can't be memorized as a fixed surface pattern. The search stops
        as soon as y is PUSHED (standard BFS early exit); if the component is
        exhausted without reaching y, the log ends with `Answer: not connected`.
        """
        rng = rng or random
        lines: list[str] = []
        queue: deque[tuple[str, int]] = deque([(x, 0)])
        seen: set[str] = {x}
        pred: dict[str, str] = {}
        lines.append(f"queue: [{x}(0)]")

        def fmt_queue() -> str:
            return "[" + ", ".join(f"{n}({d})" for n, d in queue) + "]"

        found_d: int | None = None
        while queue:
            node, d = queue.popleft()
            fresh = [nb for nb in self.neighbors[node] if nb not in seen]
            rng.shuffle(fresh)
            for nb in fresh:
                seen.add(nb)
                pred[nb] = node
                queue.append((nb, d + 1))
            if y in fresh:
                pushed = ", ".join(f"{nb}({d + 1})" for nb in fresh)
                lines.append(
                    f"pop {node}({d}) -> visit, push {pushed}"
                    f" -> target {y} reached at distance {d + 1}"
                )
                found_d = d + 1
                break
            if fresh:
                pushed = ", ".join(f"{nb}({d + 1})" for nb in fresh)
                lines.append(f"pop {node}({d}) -> visit, push {pushed} -> queue: {fmt_queue()}")
            else:
                lines.append(f"pop {node}({d}) -> visit -> queue: {fmt_queue()}")

        if found_d is None:
            lines.append(f"queue: [] -> frontier exhausted, {y} not reached")
            lines.append("Answer: not connected")
            return "\n".join(lines)

        chain = [y]
        while chain[-1] != x:
            chain.append(pred[chain[-1]])
        chain.reverse()
        lines.append("path: " + " - ".join(chain))
        lines.append(f"Answer: {found_d}")
        return "\n".join(lines)
