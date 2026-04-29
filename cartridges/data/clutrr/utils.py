from typing import List, Optional, Tuple
from pydantic import BaseModel


CLUTRR_RELATIONS = [
    "aunt", "daughter-in-law", "grandfather", "grandson", "mother",
    "nephew", "son", "brother", "daughter", "grandmother", "granddaughter",
    "mother-in-law", "niece", "son-in-law", "uncle", "father",
    "father-in-law", "sister", "husband", "wife",
]


class CLUTRRStory(BaseModel):
    story_id: str
    story: str
    clean_story: str
    query: Tuple[str, str]      # (entity_a, entity_b)
    target: str                 # relation text: "aunt", "father", etc.
    f_comb: str                 # e.g. "father-sister"
    task_name: str              # e.g. "task_1.2"
    edge_types: List[str]       # explicit relations in story

    @property
    def chain_length(self) -> int:
        """Extract hop count from task_name, e.g. 'task_1.2' -> 2."""
        try:
            return int(self.task_name.split(".")[-1])
        except (ValueError, IndexError):
            return -1

    @property
    def noise_type(self) -> int:
        """Extract noise type from task_name, e.g. 'task_1.2' -> 1."""
        try:
            return int(self.task_name.split("_")[1].split(".")[0])
        except (ValueError, IndexError):
            return -1


_BASE_URL = "https://raw.githubusercontent.com/kliang5/CLUTRR_huggingface_dataset/main"

_SPLIT_MAP = {"train": "train", "validation": "validation", "test": "test"}


def load_clutrr_dataset(
    split: str = "train",
    config: str = "gen_train23_test2to10",
    max_chain_length: Optional[int] = None,
    noise_types: Optional[List[int]] = None,
) -> List[CLUTRRStory]:
    import ast
    import csv
    import io
    import requests

    url = f"{_BASE_URL}/{config}/{_SPLIT_MAP[split]}.csv"
    resp = requests.get(url)
    resp.raise_for_status()

    reader = csv.DictReader(io.StringIO(resp.text))
    stories = []
    for i, row in enumerate(reader):
        query_raw = row.get("query", "")
        try:
            parsed = ast.literal_eval(query_raw)
            query: Tuple[str, str] = (str(parsed[0]), str(parsed[1]))
        except Exception:
            import re
            matches = re.findall(r"'([^']+)'", query_raw)
            query = (matches[0], matches[1]) if len(matches) >= 2 else ("?", "?")

        edge_types_raw = row.get("edge_types", "[]")
        try:
            edge_types = list(ast.literal_eval(edge_types_raw))
        except Exception:
            edge_types = []

        story = CLUTRRStory(
            story_id=row.get("id", str(i)),
            story=row.get("story", ""),
            clean_story=row.get("clean_story", row.get("story", "")),
            query=query,
            target=row.get("target_text", ""),
            f_comb=row.get("f_comb", ""),
            task_name=row.get("task_name", ""),
            edge_types=edge_types,
        )

        if max_chain_length is not None and story.chain_length > max_chain_length:
            continue
        if noise_types is not None and story.noise_type not in noise_types:
            continue

        stories.append(story)

    return stories


def format_story_context(story: CLUTRRStory, use_clean: bool = True) -> str:
    text = story.clean_story if use_clean else story.story
    return f"The following is a short story describing family relationships:\n\n{text}"


def format_query_prompt(story: CLUTRRStory) -> str:
    entity_a, entity_b = story.query
    return (
        f"{format_story_context(story)}\n\n"
        f"Based on the story, what family relation is {entity_b} to {entity_a}? "
        f"Answer with a single relationship word (e.g. 'aunt', 'grandfather', 'nephew').\n\n"
        f"<answer>\n{{YOUR_ANSWER}}\n</answer>"
    )
