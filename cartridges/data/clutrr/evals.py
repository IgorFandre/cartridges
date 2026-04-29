import re
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizerFast

from cartridges.datasets import GenerateEvalDataset, GenerateEvalDatasetElement
from cartridges.data.clutrr.utils import CLUTRRStory, load_clutrr_dataset, format_query_prompt
from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE, MODELS_WITH_THINKING


class CLUTRRRelationGenerateDataset(GenerateEvalDataset):
    class Config(GenerateEvalDataset.Config):
        _pass_as_config = True
        config: str = "gen_train23_test2to10"
        split: str = "test"
        noise_types: Optional[List[int]] = [1]   # clean stories only
        max_questions: Optional[int] = None
        cot: bool = True

    def __init__(self, config: Config, tokenizer: PreTrainedTokenizerFast, seed: int):
        self.config = config
        self.tokenizer = tokenizer

        stories = load_clutrr_dataset(
            split=config.split,
            config=config.config,
            noise_types=config.noise_types,
        )

        random.Random(seed).shuffle(stories)
        if config.max_questions is not None:
            stories = stories[: config.max_questions]

        self.stories = stories
        self.story_id_to_idx = {s.story_id: i for i, s in enumerate(stories)}

    def __len__(self) -> int:
        return len(self.stories)

    def __getitem__(self, index: int) -> GenerateEvalDatasetElement:
        story = self.stories[index]
        prompt = format_query_prompt(story)

        kwargs = {}
        if self.tokenizer.name_or_path in MODELS_WITH_THINKING:
            kwargs["enable_thinking"] = self.config.cot

        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=MODEL_TO_CHAT_TEMPLATE.get(self.tokenizer.name_or_path, None),
            **kwargs,
        )

        return GenerateEvalDatasetElement(
            input_ids=input_ids,
            prompt=prompt,
            answer=story.target,
            convo_id=story.story_id,
            metadata={
                "idx": index,
                "chain_length": story.chain_length,
                "f_comb": story.f_comb,
                "task_name": story.task_name,
            },
        )

    def score(
        self,
        pred: str,
        answer: str,
        convo_id: str,
    ) -> Tuple[bool, Dict[str, Optional[str]]]:
        story = self.stories[self.story_id_to_idx[convo_id]]

        extracted = _extract_answer(pred)
        correct = _normalize(extracted) == _normalize(answer)

        return correct, {
            "extracted_pred": extracted,
            "chain_length": story.chain_length,
            "task_name": story.task_name,
        }

    def aggregate_scores(
        self, scores: List[Tuple[bool, Dict]]
    ) -> Dict[str, float]:
        by_chain: Dict[int, List[bool]] = defaultdict(list)
        all_correct = []

        for correct, meta in scores:
            all_correct.append(correct)
            cl = meta.get("chain_length", -1)
            by_chain[cl].append(correct)

        result = {"accuracy": sum(all_correct) / len(all_correct) if all_correct else 0.0}
        for cl in sorted(by_chain):
            key = f"accuracy_hop{cl}"
            result[key] = sum(by_chain[cl]) / len(by_chain[cl])
        return result


def _extract_answer(pred: str) -> str:
    match = re.search(r"<answer>(.*?)</answer>", pred, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # fallback: last non-empty line
    lines = [l.strip() for l in pred.strip().splitlines() if l.strip()]
    return lines[-1] if lines else pred.strip()


def _normalize(text: str) -> str:
    return text.strip().lower().rstrip("s")  # rough lemmatize: aunt==aunt, aunts→aunt
