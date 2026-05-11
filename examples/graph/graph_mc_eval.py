"""
Generate-time MC eval dataset for kinship cartridge training.

Scores by extracting the letter A-E from prediction and comparing to gold letter.
Gold letter sits in the assistant message of every MC parquet (either "X." or
"<think>...</think>\\n\\nX.").

Two configs:
    GraphMCEvalDataset.Config(cot=False)  → for graph_train.py     (NTP letter)
    GraphMCEvalDataset.Config(cot=True)   → for graph_train_cot.py (think + letter)
"""
import re
from typing import Optional, Tuple, Dict, Any

import torch
from transformers import PreTrainedTokenizerFast

from cartridges.datasets import GenerateEvalDataset, GenerateEvalDatasetElement
from cartridges.initialization.tokenization_utils import (
    MODEL_TO_CHAT_TEMPLATE,
    MODELS_WITH_THINKING,
)


_LETTER_RE = re.compile(r"(?:answer\s*[:\-]?\s*)?\b([A-E])\b\.?", re.IGNORECASE)
_THINK_RE  = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_letter(text: str) -> str:
    """Pull A-E. Strips <think>...</think> first, then matches first letter token."""
    s = _THINK_RE.sub("", text or "").strip()
    m = _LETTER_RE.search(s)
    if m:
        return m.group(1).upper()
    m2 = re.search(r"[A-Ea-e]", s)
    return m2.group(0).upper() if m2 else ""


class GraphMCEvalDataset(GenerateEvalDataset):
    """Generate-eval dataset that scores by letter extraction (A-E)."""

    class Config(GenerateEvalDataset.Config):
        _pass_as_config = True

    def __getitem__(self, index: int) -> GenerateEvalDatasetElement:
        convo = self.data[index]
        assert len(convo.messages) >= 2

        kwargs = {}
        if self.tokenizer.name_or_path.lower() in {m.lower() for m in MODELS_WITH_THINKING}:
            kwargs["enable_thinking"] = self.config.cot

        input_ids = self.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in convo.messages[:-1]],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=MODEL_TO_CHAT_TEMPLATE.get(self.tokenizer.name_or_path, None),
            **kwargs,
        )

        # NOTE: chat template with enable_thinking=False already injects an empty
        # <think>\n\n</think> block — no manual append needed.

        gold_letter = extract_letter(convo.messages[-1].content)

        return GenerateEvalDatasetElement(
            input_ids=input_ids,
            prompt=[{"role": m.role, "content": m.content} for m in convo.messages[:-1]],
            answer=gold_letter,
            convo_id=str(index),
            metadata={
                "idx": index,
                "gold_letter": gold_letter,
                "raw_target": convo.messages[-1].content,
                **(convo.metadata or {}),
            },
        )

    def score(
        self, pred: str, answer: str, convo_id: str
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        pred_letter = extract_letter(pred)
        gold_letter = (answer or "").strip().upper()
        correct = pred_letter == gold_letter and gold_letter in "ABCDE"
        return (
            {"acc": float(correct)},
            {"pred_letter": pred_letter, "gold_letter": gold_letter},
        )
