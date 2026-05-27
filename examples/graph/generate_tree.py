"""
One-time script: generates a random family tree and saves it to family_tree.json.

Configurable parameters:
  --n-people     target number of people (will stop adding once reached)
  --max-depth    max number of generations after founders (default: 5)
  --min-kids     min children per couple (default: 1)
  --max-kids     max children per couple (default: 2)
  --founders     number of founding couples in generation 0 (default: 2)
  --spouse-prob  probability that a child gets a new spouse (default: 0.7)

Usage:
    python examples/graph/generate_tree.py --n-people 45 --max-depth 5
"""
import json
import random
from pathlib import Path

MALE_NAMES = [
    "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph",
    "Thomas", "Charles", "Daniel", "Matthew", "Anthony", "Mark", "Donald", "Steven",
    "Paul", "Andrew", "Kenneth", "George", "Joshua", "Kevin", "Brian", "Edward",
    "Ronald", "Timothy", "Jason", "Jeffrey", "Ryan", "Jacob", "Gary", "Nicholas",
]
FEMALE_NAMES = [
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth", "Susan", "Jessica",
    "Sarah", "Karen", "Lisa", "Nancy", "Betty", "Dorothy", "Sandra", "Ashley",
    "Emily", "Donna", "Michelle", "Carol", "Amanda", "Melissa", "Deborah", "Stephanie",
    "Rebecca", "Laura", "Sharon", "Cynthia", "Kathleen", "Amy", "Shirley", "Angela",
]


def generate_family_tree(
    n_people: int = 45,
    seed: int = 42,
    max_depth: int = 5,
    min_kids: int = 1,
    max_kids: int = 2,
    founders: int = 2,
    spouse_prob: float = 0.7,
) -> dict:
    rng = random.Random(seed)

    male_pool = MALE_NAMES[:]
    female_pool = FEMALE_NAMES[:]
    rng.shuffle(male_pool)
    rng.shuffle(female_pool)

    people: list[dict] = []
    parent_child: list[dict] = []
    spouses: list[dict] = []

    def new_person(gender: str) -> str | None:
        pool = male_pool if gender == "male" else female_pool
        if not pool:
            return None
        name = pool.pop(0)
        people.append({"name": name, "gender": gender})
        return name

    # Generation 0: `founders` couples
    couples_g0: list[tuple[str, str]] = []
    for _ in range(founders):
        if len(people) >= n_people:
            break
        h = new_person("male"); w = new_person("female")
        if h is None or w is None:
            break
        couples_g0.append((h, w))
        spouses.append({"a": h, "b": w})

    current_couples = couples_g0
    depth = 0
    while current_couples and depth < max_depth and len(people) < n_people:
        next_couples: list[tuple[str, str]] = []

        for father, mother in current_couples:
            n_children = rng.randint(min_kids, max_kids)
            children = []
            for _ in range(n_children):
                if len(people) >= n_people:
                    break
                gender = rng.choice(["male", "female"])
                child = new_person(gender)
                if child is None:
                    break
                parent_child.append({"parent": father, "child": child})
                parent_child.append({"parent": mother, "child": child})
                children.append(child)

            for child in children:
                if len(people) >= n_people:
                    break
                child_gender = next(p["gender"] for p in people if p["name"] == child)
                spouse_gender = "female" if child_gender == "male" else "male"
                if rng.random() < spouse_prob:
                    spouse = new_person(spouse_gender)
                    if spouse is None:
                        continue
                    spouses.append({"a": child, "b": spouse})
                    next_couples.append(
                        (child, spouse) if child_gender == "male" else (spouse, child)
                    )

        current_couples = next_couples
        depth += 1

    return {
        "people": people,
        "parent_child": parent_child,
        "spouses": spouses,
        "_meta": {
            "n_people": len(people),
            "depth": depth,
            "founders": founders,
            "min_kids": min_kids,
            "max_kids": max_kids,
            "spouse_prob": spouse_prob,
            "seed": seed,
        },
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-people",    type=int,   default=45)
    ap.add_argument("--seed",        type=int,   default=42)
    ap.add_argument("--max-depth",   type=int,   default=5)
    ap.add_argument("--min-kids",    type=int,   default=1)
    ap.add_argument("--max-kids",    type=int,   default=2)
    ap.add_argument("--founders",    type=int,   default=2)
    ap.add_argument("--spouse-prob", type=float, default=0.7)
    ap.add_argument("--out",         type=str,
                    default=str(Path(__file__).parent / "family_tree.json"))
    args = ap.parse_args()

    tree = generate_family_tree(
        n_people=args.n_people, seed=args.seed,
        max_depth=args.max_depth, min_kids=args.min_kids, max_kids=args.max_kids,
        founders=args.founders, spouse_prob=args.spouse_prob,
    )

    out_path = Path(args.out)
    with open(out_path, "w") as f:
        json.dump(tree, f, indent=2)

    m = tree["_meta"]
    print(f"Generated {m['n_people']} people, depth={m['depth']}, "
          f"{len(tree['parent_child'])} parent-child edges, "
          f"{len(tree['spouses'])} spouse pairs")
    print(f"Saved to {out_path}")
    print("\nSample people:")
    for p in tree["people"][:8]:
        print(f"  {p['name']} ({p['gender']})")
