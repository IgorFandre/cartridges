"""
One-time script: generates a random family tree and saves it to family_tree.json.
Run once, then commit family_tree.json to the repo.

Usage:
    python examples/graph/generate_tree.py
"""
import json
import random
from pathlib import Path

MALE_NAMES = [
    "James", "John", "Robert", "Michael", "William", "David", "Richard", "Joseph",
    "Thomas", "Charles", "Daniel", "Matthew", "Anthony", "Mark", "Donald", "Steven",
    "Paul", "Andrew", "Kenneth", "George", "Joshua", "Kevin", "Brian", "Edward",
]
FEMALE_NAMES = [
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth", "Susan", "Jessica",
    "Sarah", "Karen", "Lisa", "Nancy", "Betty", "Dorothy", "Sandra", "Ashley",
    "Emily", "Donna", "Michelle", "Carol", "Amanda", "Melissa", "Deborah", "Stephanie",
]


def generate_family_tree(n_people: int = 40, seed: int = 42) -> dict:
    rng = random.Random(seed)

    male_pool = MALE_NAMES[:]
    female_pool = FEMALE_NAMES[:]
    rng.shuffle(male_pool)
    rng.shuffle(female_pool)

    people: list[dict] = []
    parent_child: list[dict] = []
    spouses: list[dict] = []

    def new_person(gender: str) -> str:
        if gender == "male":
            name = male_pool.pop(0)
        else:
            name = female_pool.pop(0)
        people.append({"name": name, "gender": gender})
        return name

    # Generation 0: 2 couples (4 founders)
    gen: list[list[str]] = []  # gen[i] = list of names in generation i

    couples_g0 = []
    for _ in range(2):
        husband = new_person("male")
        wife = new_person("female")
        couples_g0.append((husband, wife))
        spouses.append({"a": husband, "b": wife})

    gen.append([p for c in couples_g0 for p in c])

    # Build subsequent generations until we reach n_people
    current_couples = couples_g0
    while len(people) < n_people:
        next_gen_names: list[str] = []
        next_couples: list[tuple[str, str]] = []

        for father, mother in current_couples:
            n_children = rng.randint(2, 3)
            children = []
            for _ in range(n_children):
                if len(people) >= n_people:
                    break
                gender = rng.choice(["male", "female"])
                child = new_person(gender)
                parent_child.append({"parent": father, "child": child})
                parent_child.append({"parent": mother, "child": child})
                children.append(child)
                next_gen_names.append(child)

            # Pair some children with new spouses from opposite sex
            for child in children:
                if len(people) >= n_people:
                    break
                child_gender = next(p["gender"] for p in people if p["name"] == child)
                spouse_gender = "female" if child_gender == "male" else "male"
                # Only pair ~60% of children to keep the tree realistic
                if rng.random() < 0.6 and (male_pool if spouse_gender == "male" else female_pool):
                    spouse = new_person(spouse_gender)
                    next_gen_names.append(spouse)
                    spouses.append({"a": child, "b": spouse})
                    next_couples.append(
                        (child, spouse) if child_gender == "male" else (spouse, child)
                    )

        if not next_gen_names:
            break
        gen.append(next_gen_names)
        current_couples = next_couples
        if not current_couples:
            break

    return {
        "people": people,
        "parent_child": parent_child,
        "spouses": spouses,
    }


if __name__ == "__main__":
    out_path = Path(__file__).parent / "family_tree.json"
    tree = generate_family_tree(n_people=40, seed=42)
    with open(out_path, "w") as f:
        json.dump(tree, f, indent=2)

    n = len(tree["people"])
    n_pc = len(tree["parent_child"])
    n_sp = len(tree["spouses"])
    print(f"Generated {n} people, {n_pc} parent-child edges, {n_sp} spouse pairs")
    print(f"Saved to {out_path}")
    # Print a sample
    print("\nSample people:")
    for p in tree["people"][:8]:
        print(f"  {p['name']} ({p['gender']})")
    print("\nSample parent-child edges:")
    for e in tree["parent_child"][:6]:
        print(f"  {e['parent']} -> {e['child']}")
