import random
from typing import List, Optional

from cartridges.data.resources import Resource, sample_seed_prompts, SEED_TYPES
from cartridges.data.clutrr.utils import CLUTRRStory, load_clutrr_dataset, format_story_context


CORPUS_HEADER = (
    "The following are stories about family relationships. "
    "Each story describes explicit kinship relations between family members.\n\n"
)


class CLUTRRResource(Resource):
    class Config(Resource.Config):
        config: str = "gen_train23_test2to10"
        split: str = "train"
        noise_types: Optional[List[int]] = [1]   # 1=clean only by default
        max_chain_length: Optional[int] = None
        stories_per_prompt: int = 5
        seed_prompts: List[SEED_TYPES] = ["question", "summarization", "generic"]

    def __init__(self, config: Config):
        self.config = config
        self.stories: List[CLUTRRStory] = load_clutrr_dataset(
            split=config.split,
            config=config.config,
            max_chain_length=config.max_chain_length,
            noise_types=config.noise_types,
        )

    async def sample_prompt(self, batch_size: int) -> tuple[str, List[str]]:
        batch = random.sample(self.stories, min(self.config.stories_per_prompt, len(self.stories)))
        stories_text = "\n\n".join(
            f"Story {i + 1}: {s.clean_story}" for i, s in enumerate(batch)
        )
        ctx = f"{CORPUS_HEADER}{stories_text}"
        seed_prompts = sample_seed_prompts(self.config.seed_prompts, batch_size)
        return ctx, seed_prompts

    def to_string(self) -> str:
        lines = [CORPUS_HEADER]
        for i, s in enumerate(self.stories):
            lines.append(f"Story {i + 1}: {s.clean_story}")
        return "\n\n".join(lines)
