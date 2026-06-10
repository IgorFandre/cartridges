"""
Generate the friendship forest: several disjoint random trees ("handshake graph").

Each connected component is a tree, so the path between any two connected
people is unique. Components are deliberately disjoint — cross-component pairs
are the structural source of "not connected" negatives.

Per-hop coverage is validated at generation time: the forest must contain at
least --min-pairs-shallow unordered pairs at every distance 1..6 (train+test
hops) and --min-pairs-deep at distances 7..8 (test-only generalization hops).
If a seed fails, the next seed is tried (--tries).

Depth bias: new nodes attach to an existing node sampled with weight
(1 + depth)^alpha. alpha=0 → uniform random recursive tree (shallow);
larger alpha → deeper, path-like trees with more long-distance pairs.

Usage:
    python -m examples.graph_3.data_gen.generate_forest
    python -m examples.graph_3.data_gen.generate_forest --components 8 --component-size 50 --alpha 1.5
"""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path

from examples.graph_3.data_gen.graph_index import GraphIndex

# ── Name pool ─────────────────────────────────────────────────────────────────
# Unique single-word names: common US first names + surname-style names.
# Connections are random under the seed, so nothing here is memorizable from
# pretraining; real names just keep questions natural.
_FEMALE = """
Mary Patricia Jennifer Linda Barbara Susan Jessica Sarah Karen Lisa
Nancy Betty Sandra Margaret Ashley Kimberly Emily Donna Michelle Carol
Amanda Melissa Deborah Stephanie Rebecca Sharon Laura Cynthia Dorothy Amy
Kathleen Angela Shirley Emma Brenda Pamela Nicole Anna Samantha Katherine
Christine Debra Rachel Carolyn Janet Maria Olivia Heather Helen Catherine
Diane Julie Victoria Joyce Lauren Kelly Christina Ruth Joan Virginia
Judith Evelyn Hannah Andrea Megan Cheryl Jacqueline Madison Teresa Abigail
Sophia Martha Sara Gloria Janice Kathryn Ann Isabella Judy Charlotte
Julia Grace Amber Alice Jean Denise Frances Danielle Marilyn Natalie
Beverly Diana Brittany Theresa Kayla Alexis Doris Lori Tiffany Erin
""".split()

_MALE = """
James Robert John Michael David William Richard Joseph Thomas Charles
Christopher Daniel Matthew Anthony Mark Donald Steven Paul Andrew Joshua
Kenneth Kevin Brian George Timothy Ronald Edward Jason Jeffrey Ryan
Jacob Gary Nicholas Eric Jonathan Stephen Larry Justin Scott Brandon
Benjamin Samuel Gregory Alexander Frank Patrick Raymond Jack Dennis Jerry
Tyler Aaron Jose Adam Nathan Henry Douglas Zachary Peter Kyle
Ethan Walter Noah Jeremy Christian Keith Roger Terry Austin Sean
Gerald Carl Harold Dylan Arthur Lawrence Jordan Jesse Bryan Billy
Bruce Gabriel Joe Logan Alan Juan Albert Willie Elijah Wayne
Randy Vincent Mason Roy Ralph Bobby Russell Bradley Philip Eugene
""".split()

_SURNAME_STYLE = """
Walker Hayes Brooks Reed Cole Bennett Murphy Bailey Rivera Cooper
Richardson Ward Torres Peterson Ramirez Price Foster Morales Powell Sullivan
Ortiz Jenkins Gutierrez Perry Butler Barnes Fisher Henderson Coleman Simmons
Patterson Jordan Reynolds Hamilton Graham Kim Gonzales Alexander Ramos Wallace
Griffin West Stone Hawkins Dunn Perkins Hudson Spencer Gardner Stephens
Payne Pierce Berry Matthews Arnold Wagner Willis Ray Watkins Olson
Carroll Duncan Snyder Hart Cunningham Bradley Lane Andrews Ruiz Harper
Fox Riley Armstrong Carpenter Weaver Greene Lawrence Elliott Chavez Sims
Austin Peters Kelley Franklin Lawson Fields Gutman Ryan Schmidt Carr
Vasquez Castillo Wheeler Chapman Oliver Montgomery Richards Williamson Johnston Banks
Meyer Bishop Mccoy Howell Alvarez Morrison Hansen Fernandez Garza Harvey
Little Burton Stanley Nguyen George Jacobs Reid Fuller Lynch Dean
Gilbert Garrett Romero Welch Larson Frazier Burke Hanson Day Mendoza
Moreno Bowman Medina Fowler Brewer Hoffman Carlson Silva Pearson Holland
Abbott Acosta Adkins Aguilar Atkins Avery Barker Barrett Barton Bates
Beck Becker Benson Blair Blake Bond Boone Bowen Boyd Brady
Brennan Briggs Bryant Buchanan Burgess Burns Byrd Cain Calhoun Cameron
Campos Cannon Carey Carson Casey Chambers Chandler Chase Clarke Clayton
Cobb Cochran Cohen Collier Combs Compton Conley Conrad Conway Copeland
Cortez Craig Crane Crawford Cross Cruz Curry Curtis Dalton Daniels
Davenport Dawson Decker Delgado Dickson Dillon Dixon Donovan Dorsey Doyle
Drake Dudley Duffy Duke Durham Dyer Eaton Emerson England English
Erickson Espinoza Estes Everett Farley Farmer Farrell Faulkner Ferguson Finley
Fitzgerald Fleming Fletcher Flores Flynn Forbes Ford Franco Freeman French
Frost Gaines Gallagher Galloway Gentry Gibbs Gibson Glass Glenn Golden
Goodman Goodwin Gordon Grant Graves Greer Griffith Grimes Guerrero Guthrie
Hale Haley Hammond Hampton Hardy Harmon Harrington Hartman Hatfield Hebert
Hendricks Hess Hickman Hicks Hines Hodge Hogan Holman Holt Hood
Hooper Hoover Hopkins Horne House Houston Hubbard Huber Huff Hull
""".split()

NAME_POOL: list[str] = sorted(set(_FEMALE) | set(_MALE) | set(_SURNAME_STYLE))


# ── Tree builder ──────────────────────────────────────────────────────────────
def build_random_tree(
    names: list[str], rng: random.Random, alpha: float = 1.5
) -> list[dict]:
    """Random tree over `names`: each new node attaches to an existing node
    sampled with weight (1 + depth)^alpha. Returns undirected edges."""
    depth = {names[0]: 0}
    edges: list[dict] = []
    for name in names[1:]:
        existing = list(depth)
        weights = [(1 + depth[n]) ** alpha for n in existing]
        parent = rng.choices(existing, weights=weights, k=1)[0]
        depth[name] = depth[parent] + 1
        edges.append({"a": parent, "b": name})
    return edges


def build_forest(
    n_components: int,
    component_size: int,
    seed: int,
    alpha: float = 1.5,
) -> dict:
    n_people = n_components * component_size
    if n_people > len(NAME_POOL):
        raise ValueError(
            f"Need {n_people} unique names but pool has {len(NAME_POOL)}. "
            "Extend NAME_POOL or reduce graph size."
        )
    rng = random.Random(seed)
    names = rng.sample(NAME_POOL, n_people)

    edges: list[dict] = []
    components: list[list[str]] = []
    for c in range(n_components):
        member_names = names[c * component_size : (c + 1) * component_size]
        components.append(member_names)
        edges.extend(build_random_tree(member_names, rng, alpha=alpha))

    rng.shuffle(edges)  # corpus order must not encode component structure

    return {
        "people": sorted(names),
        "edges": edges,
        "_meta": {
            "n_people": n_people,
            "n_edges": len(edges),
            "n_components": n_components,
            "component_size": component_size,
            "alpha": alpha,
            "seed": seed,
        },
    }


# ── Validation ────────────────────────────────────────────────────────────────
def pair_counts(index: GraphIndex, max_hop: int = 8) -> dict[int, int]:
    by_d = index.pairs_by_distance(max_distance=max_hop)
    return {d: len(by_d.get(d, [])) for d in range(1, max_hop + 1)}

def is_valid(counts: dict[int, int], min_shallow: int, min_deep: int) -> bool:
    return all(counts[d] >= min_shallow for d in range(1, 7)) and all(
        counts[d] >= min_deep for d in (7, 8)
    )


# ── Corpus writer ─────────────────────────────────────────────────────────────
def corpus_text(forest: dict) -> str:
    return "\n".join(
        f"{e['a']} and {e['b']} know each other." for e in forest["edges"]
    )


def count_tokens(text: str) -> tuple[int, str]:
    """Qwen3 token count; falls back to a chars/4 estimate without the tokenizer."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
        return len(tok.encode(text)), "Qwen/Qwen3-1.7B"
    except Exception:
        return len(text) // 4, "estimate (chars/4)"


# ── Entry point ───────────────────────────────────────────────────────────────
def main():
    from examples.graph_3 import paths

    ap = argparse.ArgumentParser()
    ap.add_argument("--components",       type=int,   default=8)
    ap.add_argument("--component-size",   type=int,   default=50)
    ap.add_argument("--alpha",            type=float, default=1.5)
    ap.add_argument("--seed",             type=int,   default=42)
    ap.add_argument("--tries",            type=int,   default=50,
                    help="Seeds to try (seed, seed+1, ...) until per-hop minimums hold")
    ap.add_argument("--min-pairs-shallow", type=int,  default=160,
                    help="Min unordered pairs at each hop 1..6 (train+test)")
    ap.add_argument("--min-pairs-deep",    type=int,  default=100,
                    help="Min unordered pairs at hops 7..8 (test-only)")
    ap.add_argument("--out", type=str, default=str(paths.FOREST_JSON))
    args = ap.parse_args()

    forest, counts, used_seed = None, None, None
    best = None  # (n_satisfied_hops, forest, counts, seed) for the failure report
    for i in range(args.tries):
        s = args.seed + i
        cand = build_forest(args.components, args.component_size, seed=s, alpha=args.alpha)
        idx = GraphIndex(cand["edges"])
        cnt = pair_counts(idx)
        score = sum(
            cnt[d] >= (args.min_pairs_shallow if d <= 6 else args.min_pairs_deep)
            for d in range(1, 9)
        )
        if best is None or score > best[0]:
            best = (score, cand, cnt, s)
        if is_valid(cnt, args.min_pairs_shallow, args.min_pairs_deep):
            forest, counts, used_seed = cand, cnt, s
            break

    if forest is None:
        _, forest, counts, used_seed = best
        print(
            f"WARNING: no seed in {args.tries} tries met the minimums "
            f"(shallow>={args.min_pairs_shallow}, deep>={args.min_pairs_deep}). "
            f"Saving the best candidate (seed={used_seed}); consider larger "
            "--component-size / --components or higher --alpha."
        )

    text = corpus_text(forest)
    n_tokens, tokenizer_name = count_tokens(text)
    forest["_meta"].update({
        "used_seed":     used_seed,
        "pairs_per_hop": counts,
        "corpus_tokens": n_tokens,
        "tokenizer":     tokenizer_name,
        "cartridge_tokens_x5": n_tokens // 5,
    })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(forest, indent=2))
    corpus_path = out_path.parent / "corpus.txt"
    corpus_path.write_text(text)

    m = forest["_meta"]
    print(f"Forest: {m['n_people']} people, {m['n_edges']} edges, "
          f"{m['n_components']} components × {m['component_size']}, "
          f"alpha={m['alpha']}, seed={used_seed}")
    print("Unordered pairs per hop:")
    for d in range(1, 9):
        flag = "" if counts[d] >= (args.min_pairs_shallow if d <= 6 else args.min_pairs_deep) else "  ← below min"
        print(f"  d={d}: {counts[d]}{flag}")
    print(f"Corpus: {len(text)} chars = {n_tokens} tokens [{tokenizer_name}]")
    print(f"×5 compression → CARTRIDGE_TOKENS ≈ {n_tokens // 5}")
    print(f"Saved forest → {out_path}")
    print(f"Saved corpus → {corpus_path}")


if __name__ == "__main__":
    main()
