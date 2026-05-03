"""
GraphRelationshipMCEvalDataset: multiple-choice eval for family tree cartridge.

Format (user message):
    What is the relationship between Alice and Tom?

    Options:
    A. aunt
    B. grandmother
    C. sister
    D. cousin
    E. I don't know

    Think step by step inside <think>...</think>, then give your answer as a single letter.

Score: accuracy broken down by chain length (hop_1_score, hop_2_score, ...).
"""
import random
import re
from typing import Dict, List, Optional, Tuple

from transformers import PreTrainedTokenizerFast

from cartridges.datasets import GenerateEvalDataset, GenerateEvalDatasetElement
from cartridges.structs import Conversation, read_conversations
from cartridges.initialization.tokenization_utils import (
    MODEL_TO_CHAT_TEMPLATE,
    MODELS_WITH_THINKING,
)

LETTERS = ["A", "B", "C", "D", "E"]
DONT_KNOW = "I don't know"

FORMAT_INSTRUCTION = (
    "Think step by step inside <think>...</think>, "
    "then give your answer as a single letter (A/B/C/D/E)."
)


def _build_mc_prompt(question: str, options: list[str]) -> str:
    """Build the full user message with multiple choice options."""
    opts_str = "\n".join(f"{L}. {opt}" for L, opt in zip(LETTERS, options))
    return f"{question}\n\nOptions:\n{opts_str}\n\n{FORMAT_INSTRUCTION}"


def _extract_answer_letter(pred: str) -> str | None:
    """Extract the answer letter from model output after </think>."""
    # Strip thinking block if present
    after_think = re.sub(r"<think>.*?</think>", "", pred, flags=re.DOTALL).strip()
    # Look for a standalone letter A-E
    match = re.search(r"\b([A-E])\b", after_think)
    if match:
        return match.group(1)
    # Fallback: last capital letter in the whole output
    matches = re.findall(r"\b([A-E])\b", pred)
    return matches[-1] if matches else None


class GraphRelationshipMCEvalDataset(GenerateEvalDataset):
    class Config(GenerateEvalDataset.Config):
        _pass_as_config = True
        val_path: str           # path to val_dataset.parquet
        seed: int = 42
        include_system_prompt: bool = False  # if True, prepend system_prompt from parquet

    def __init__(
        self,
        config: "GraphRelationshipMCEvalDataset.Config",
        tokenizer: PreTrainedTokenizerFast,
        seed: int,
    ):
        self.config = config
        self.tokenizer = tokenizer

        self.data: list[Conversation] = read_conversations(config.val_path)
        self.rng = random.Random(config.seed)

        # Pool of all possible relationships for distractor sampling
        self._all_rels: list[str] = sorted(
            set(c.metadata["final_rel"] for c in self.data)
        )

        # Pre-build options for each item (deterministic)
        self._items: list[dict] = []
        for conv in self.data:
            correct_rel = conv.metadata["final_rel"]
            distractors = self._sample_distractors(correct_rel, n=3)
            options = distractors + [correct_rel, DONT_KNOW]
            self.rng.shuffle(options)
            correct_letter = LETTERS[options.index(correct_rel)]
            self._items.append({
                "options": options,
                "correct_letter": correct_letter,
                "correct_rel": correct_rel,
                "chain_length": conv.metadata["chain_length"],
                "person_a": conv.metadata["person_a"],
                "person_b": conv.metadata["person_b"],
            })

    def _sample_distractors(self, correct: str, n: int) -> list[str]:
        pool = [r for r in self._all_rels if r != correct and r != DONT_KNOW]
        return self.rng.sample(pool, min(n, len(pool)))

    def __getitem__(self, index: int) -> GenerateEvalDatasetElement:
        conv = self.data[index]
        item = self._items[index]

        a, b = item["person_a"], item["person_b"]
        question = f"What is the relationship between {a} and {b}?"
        prompt_text = _build_mc_prompt(question, item["options"])

        kwargs = {}
        if self.tokenizer.name_or_path in MODELS_WITH_THINKING:
            kwargs["enable_thinking"] = True

        prompt_messages = []
        if self.config.include_system_prompt and conv.system_prompt:
            prompt_messages.append({"role": "system", "content": conv.system_prompt})
        prompt_messages.append({"role": "user", "content": prompt_text})

        input_ids = self.tokenizer.apply_chat_template(
            prompt_messages,
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=MODEL_TO_CHAT_TEMPLATE.get(self.tokenizer.name_or_path, None),
            **kwargs,
        )

        return GenerateEvalDatasetElement(
            input_ids=input_ids,
            prompt=prompt_messages,
            answer=item["correct_letter"],   # gold = letter
            convo_id=str(index),
            metadata={
                "chain_length": item["chain_length"],
                "correct_rel": item["correct_rel"],
                "correct_letter": item["correct_letter"],
                "options": str(item["options"]),
            },
        )

    def __len__(self) -> int:
        return len(self.data)

    def score(
        self,
        pred: str,
        answer: str,        # correct letter
        convo_id: str,
    ) -> Tuple[Dict[str, float], Dict[str, object]]:
        item = self._items[int(convo_id)]
        chain_length = item["chain_length"]

        pred_letter = _extract_answer_letter(pred)
        correct = float(pred_letter == answer) if pred_letter is not None else 0.0

        metrics = {f"hop_{chain_length}": correct}
        extras = {
            "chain_length": chain_length,
            "pred_letter": pred_letter,
            "correct_letter": answer,
            "correct_rel": item["correct_rel"],
        }
        return metrics, extras
