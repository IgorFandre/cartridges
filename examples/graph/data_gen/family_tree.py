"""
FamilyTree: loads family_tree.json and provides relationship path finding.

The JSON schema:
{
  "people": [{"name": str, "gender": "male"|"female"}, ...],
  "parent_child": [{"parent": str, "child": str}, ...],
  "spouses": [{"a": str, "b": str}, ...]
}
"""
import json
from collections import deque
from pathlib import Path


# Maps (from_rel, to_rel) -> composed relationship label
# Edge semantics: (A, rel, X) means "A is X's rel"
# from_rel: what A is to X  (e.g. "mother" = A is X's mother)
# to_rel:   what X is to B  (e.g. "father" = X is B's father)
# value: what A is to B
COMPOSITION: dict[tuple[str, str], str] = {
    # ── grandparent ──────────────────────────────────────────────────────────
    ("father", "father"): "grandfather",
    ("father", "mother"): "grandfather",
    ("mother", "father"): "grandmother",
    ("mother", "mother"): "grandmother",

    # ── grandchild ───────────────────────────────────────────────────────────
    ("son", "son"): "grandson",
    ("son", "daughter"): "grandson",
    ("daughter", "son"): "granddaughter",
    ("daughter", "daughter"): "granddaughter",

    # ── aunt / uncle ─────────────────────────────────────────────────────────
    ("sister", "father"): "aunt",
    ("sister", "mother"): "aunt",
    ("brother", "father"): "uncle",
    ("brother", "mother"): "uncle",

    # ── nephew / niece ───────────────────────────────────────────────────────
    ("son", "brother"): "nephew",
    ("son", "sister"): "nephew",
    ("daughter", "brother"): "niece",
    ("daughter", "sister"): "niece",

    # ── sibling + sibling = sibling ──────────────────────────────────────────
    ("brother", "brother"): "brother",
    ("brother", "sister"): "sister",
    ("sister", "brother"): "brother",
    ("sister", "sister"): "sister",

    # ── parent + sibling of child = parent (shared children) ─────────────────
    ("father", "brother"): "father",
    ("father", "sister"): "father",
    ("mother", "brother"): "mother",
    ("mother", "sister"): "mother",

    # ── child + parent of child = sibling ────────────────────────────────────
    ("son", "father"): "brother",
    ("son", "mother"): "brother",
    ("daughter", "father"): "sister",
    ("daughter", "mother"): "sister",

    # ── grandparent + sibling = grandparent (shared grandchildren) ───────────
    ("grandfather", "brother"): "grandfather",
    ("grandfather", "sister"): "grandfather",
    ("grandmother", "brother"): "grandmother",
    ("grandmother", "sister"): "grandmother",

    # ── cousin: nephew/niece's child or aunt/uncle's child ───────────────────
    ("nephew", "father"): "cousin",
    ("nephew", "mother"): "cousin",
    ("niece", "father"): "cousin",
    ("niece", "mother"): "cousin",
    ("uncle", "son"): "cousin",
    ("uncle", "daughter"): "cousin",
    ("aunt", "son"): "cousin",
    ("aunt", "daughter"): "cousin",

    # ── cousin + parent/spouse → cousin-once-removed ──────────────────────────
    ("cousin", "father"): "cousin",
    ("cousin", "mother"): "cousin",
}

# Gendered relationship labels
GENDER_REL: dict[tuple[str, str], str] = {
    # (base_rel, gender_of_target) -> display label
    ("parent", "male"): "father",
    ("parent", "female"): "mother",
    ("child", "male"): "son",
    ("child", "female"): "daughter",
    ("sibling", "male"): "brother",
    ("sibling", "female"): "sister",
    ("spouse", "male"): "husband",
    ("spouse", "female"): "wife",
}


# Hop-class of a relation = graph BFS distance bucket (validated against
# FamilyTree.find_path_reasoning chain lengths). Used for per-hop eval:
# "how far can the model walk the graph?". Siblings = 1 (direct sibling edge);
# cousin = 3 bucket (actual chain length spans 3-6 for cousins-once/twice-removed).
REL_HOPS: dict[str, int] = {
    "father": 1, "mother": 1, "son": 1, "daughter": 1, "husband": 1, "wife": 1,
    "brother": 1, "sister": 1,
    "grandfather": 2, "grandmother": 2, "grandson": 2, "granddaughter": 2,
    "uncle": 2, "aunt": 2, "nephew": 2, "niece": 2,
    "cousin": 3,
}


def hops_for(category, rel: str) -> int:
    """Relation hop-class (1/2/3) for a QA record, from its category and rel.

    1 = adjacent (parent/child/spouse/sibling), 2 = grandparent/grandchild/
    uncle/aunt, 3 = cousin (3+, distant). Cat-1w (whose-style) is the inverse of
    one edge → 1; Cat-3 counting takes the max hop over its summed relations.
    """
    cat = str(category)
    if cat == "1w":
        return 1
    if cat == "3":  # rel like "son+daughter" / "grandson+granddaughter"
        return max((REL_HOPS.get(p, 1) for p in rel.split("+")), default=1)
    base = rel.rsplit("_", 1)[-1]  # strip verify_/exist_/disambig_ prefixes
    return REL_HOPS.get(base, REL_HOPS.get(rel, 1))


class FamilyTree:
    def __init__(self, data: dict):
        self.people: list[dict] = data["people"]
        self.parent_child: list[dict] = data["parent_child"]
        self.spouses: list[dict] = data.get("spouses", [])

        self._gender: dict[str, str] = {p["name"]: p["gender"] for p in self.people}
        self._rel_graph: dict[str, list[tuple[str, str]]] = {}  # name -> [(neighbor, rel_label)]
        self._build_graph()

    @classmethod
    def load(cls, path: str | Path) -> "FamilyTree":
        with open(path) as f:
            return cls(json.load(f))

    def _rel(self, base: str, target_name: str) -> str:
        gender = self._gender.get(target_name, "male")
        return GENDER_REL.get((base, gender), base)

    def _add_edge(self, a: str, b: str, label: str):
        self._rel_graph.setdefault(a, []).append((b, label))

    def _build_graph(self):
        # Edge label = what SOURCE is to TARGET.
        # (A, rel, X) means "A is X's rel"
        children_of: dict[str, list[str]] = {}
        parents_of: dict[str, list[str]] = {}
        for edge in self.parent_child:
            p, c = edge["parent"], edge["child"]
            children_of.setdefault(p, []).append(c)
            parents_of.setdefault(c, []).append(p)
            # p is c's father/mother (use p's gender)
            self._add_edge(p, c, self._rel("parent", p))
            # c is p's son/daughter (use c's gender)
            self._add_edge(c, p, self._rel("child", c))

        # siblings: share at least one parent
        added_siblings: set[frozenset] = set()
        for c, parents in parents_of.items():
            siblings = set()
            for par in parents:
                siblings |= set(children_of.get(par, []))
            siblings.discard(c)
            for sib in siblings:
                key = frozenset([c, sib])
                if key not in added_siblings:
                    added_siblings.add(key)
                    # c is sib's brother/sister (use c's gender)
                    self._add_edge(c, sib, self._rel("sibling", c))
                    # sib is c's brother/sister (use sib's gender)
                    self._add_edge(sib, c, self._rel("sibling", sib))

        # spouses
        for pair in self.spouses:
            a, b = pair["a"], pair["b"]
            # a is b's husband/wife (use a's gender)
            self._add_edge(a, b, self._rel("spouse", a))
            # b is a's husband/wife (use b's gender)
            self._add_edge(b, a, self._rel("spouse", b))

    def find_path_reasoning(self, person_a: str, person_b: str) -> tuple[str, str, int] | None:
        """
        BFS shortest path from person_a to person_b.
        Returns (reasoning_string, final_relationship, chain_length) or None if unreachable.

        chain_length = number of hops (edges) in the path.

        Example:
            ("Alice is Valery's sister. Valery is Tom's mother. So Alice is Tom's aunt.", "aunt", 2)
        """
        if person_a not in self._rel_graph or person_b not in self._rel_graph:
            return None
        if person_a == person_b:
            return None

        # BFS: track path as list of (node, edge_label_to_next)
        # Each queue item: (current_node, path_so_far)
        # path_so_far: list of (from_node, rel_label, to_node)
        visited = {person_a}
        queue: deque[tuple[str, list[tuple[str, str, str]]]] = deque()
        queue.append((person_a, []))

        while queue:
            node, path = queue.popleft()
            for neighbor, rel in self._rel_graph.get(node, []):
                if neighbor in visited:
                    continue
                new_path = path + [(node, rel, neighbor)]
                if neighbor == person_b:
                    reasoning, final_rel = self._path_to_reasoning(new_path)
                    return reasoning, final_rel, len(new_path)
                visited.add(neighbor)
                queue.append((neighbor, new_path))

        return None

    def _path_to_reasoning(self, path: list[tuple[str, str, str]]) -> tuple[str, str]:
        """
        path: [(from, rel_of_to, to), ...]
        rel_of_to: what `from` is to `to` (e.g. "mother", "sister")

        Builds: "A is X's sister. X is B's mother. So A is B's aunt."
        Always ends with a "So ..." clause (fallback: "distant relative").
        """
        sentences = []
        for from_node, rel, to_node in path:
            sentences.append(f"{from_node} is {to_node}'s {rel}")

        reasoning = ". ".join(sentences) + "."

        final_rel = self._compose_relationship(path)
        a = path[0][0]
        b = path[-1][2]
        reasoning += f" So {a} is {b}'s {final_rel}."

        return reasoning, final_rel

    def _compose_relationship(self, path: list[tuple[str, str, str]]) -> str:
        if len(path) == 1:
            return path[0][1]  # direct relationship

        current_rel = path[0][1]
        for _, rel, _ in path[1:]:
            composed = COMPOSITION.get((current_rel, rel))
            if composed is None:
                return "distant relative"
            current_rel = composed
        return current_rel

    def to_text(self) -> str:
        """Human-readable description of the family tree for use as corpus text."""
        lines = ["Family tree:"]

        # Couples and their children
        children_of: dict[str, list[str]] = {}
        for edge in self.parent_child:
            children_of.setdefault(edge["parent"], []).append(edge["child"])

        described: set[str] = set()
        for pair in self.spouses:
            a, b = pair["a"], pair["b"]
            kids_a = set(children_of.get(a, []))
            kids_b = set(children_of.get(b, []))
            shared = sorted(kids_a & kids_b)
            lines.append(f"{a} is married to {b}.")
            if shared:
                lines.append(f"{a} and {b} have children: {', '.join(shared)}.")
            described.update([a, b])

        # Single parents or people not in spouse list
        for edge in self.parent_child:
            p = edge["parent"]
            if p not in described:
                kids = sorted(children_of.get(p, []))
                if kids:
                    lines.append(f"{p} is the parent of {', '.join(kids)}.")
                described.add(p)

        return "\n".join(lines)

    def all_pairs(self) -> list[tuple[str, str]]:
        names = [p["name"] for p in self.people]
        return [(a, b) for a in names for b in names if a != b]


if __name__ == "__main__":
    # Quick smoke test
    from examples.graph.paths import BASE_TREE_JSON as tree_path
    if not tree_path.exists():
        print("family_tree.json not found. Run generate_tree.py first.")
        exit(1)

    tree = FamilyTree.load(tree_path)
    print(f"Loaded {len(tree.people)} people")

    found = 0
    for a, b in tree.all_pairs():
        result = tree.find_path_reasoning(a, b)
        if result:
            reasoning, rel = result
            print(f"\n{a} -> {b}:")
            print(f"  {reasoning}")
            found += 1
            if found >= 10:
                break
