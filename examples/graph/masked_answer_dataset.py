"""
Train dataset with reasoning masked out of the loss.

Each CoT-MC conversation has assistant content like:
    "<think>\n{BFS reasoning}\n</think>\n\nC."

We tokenize the full message so the reasoning is teacher-forced (model
conditions on it during training), but we set `topk_token_idxs` to ONLY
the position of the final letter token (A-E) — so the loss is computed
exclusively over the answer letter.

Why: matches the experimental spec — "model generates whatever it wants
in reasoning, loss is only on the answer letter".

Use via `dataset=MaskedAnswerTrainDataset.Config(...)` in TrainConfig.
"""
from __future__ import annotations
from typing import Literal

import numpy as np
import torch
from transformers import PreTrainedTokenizerFast

from cartridges.datasets import (
    TrainDataset, DatasetElement, TokenCounts,
    MODEL_TO_MESSAGE_CONVERTER, _prepare_data_source,
)

LETTER_TOKENS_CACHE: dict[str, dict[str, list[int]]] = {}


def _letter_token_ids(tokenizer: PreTrainedTokenizerFast) -> dict[str, list[int]]:
    """Return {letter -> [candidate token-id encodings]} for A-E."""
    name = tokenizer.name_or_path
    if name in LETTER_TOKENS_CACHE:
        return LETTER_TOKENS_CACHE[name]
    out: dict[str, list[int]] = {}
    for L in "ABCDE":
        ids = set()
        for cand in (L, " " + L, L + ".", " " + L + ".", "\n" + L):
            enc = tokenizer.encode(cand, add_special_tokens=False)
            # take only single-token candidates so we can pinpoint exactly
            if len(enc) == 1:
                ids.add(enc[0])
            # also include the first id of multi-token sequences (often
            # the bare letter even when prefixed with newline)
            if enc:
                ids.add(enc[0])
        out[L] = sorted(ids)
    LETTER_TOKENS_CACHE[name] = out
    return out


def _find_letter_position(
    input_ids: list[int],
    assistant_start_idx: int,
    letter: str,
    tokenizer: PreTrainedTokenizerFast,
) -> int | None:
    """Find absolute index of the LAST occurrence of `letter` token inside
    the assistant span starting at `assistant_start_idx`.
    """
    cand_ids = set(_letter_token_ids(tokenizer)[letter])
    last = None
    for i in range(assistant_start_idx, len(input_ids)):
        if input_ids[i] in cand_ids:
            # verify by decoding: cheap sanity check that decoded token
            # contains the letter and is not embedded in another word
            tok = tokenizer.decode([input_ids[i]])
            if letter in tok and not any(c.isalpha() and c.upper() != letter for c in tok):
                last = i
    return last


class MaskedAnswerTrainDataset(TrainDataset):
    """TrainDataset that keeps loss only on the final answer letter."""

    class Config(TrainDataset.Config):
        _pass_as_config = True
        targets: Literal["tokens"] = "tokens"

    def _prepare_elements(self) -> list[DatasetElement]:
        data = []
        for source in self.config.data_sources:
            data.extend(_prepare_data_source(source))

        convert = MODEL_TO_MESSAGE_CONVERTER[self.tokenizer.name_or_path.lower()]

        elements: list[DatasetElement] = []
        for row in data:
            elem = convert(row.messages, retokenize=True, tokenizer=self.tokenizer)

            # extract gold letter from metadata (preferred) or from final message
            letter = None
            if row.metadata and "correct_letter" in row.metadata:
                letter = str(row.metadata["correct_letter"]).strip().upper()
            else:
                tail = (row.messages[-1].content or "").strip().rstrip(".").strip()
                if tail and tail[-1].upper() in "ABCDE":
                    letter = tail[-1].upper()
            if letter not in {"A", "B", "C", "D", "E"}:
                # skip malformed
                continue

            ids_list = elem.input_ids.tolist()
            # assistant span = anywhere after the last user message;
            # since topk_token_idxs in the base converter spans the assistant
            # message tokens, use min(topk_token_idxs) as the span start.
            asst_start = int(elem.topk_token_idxs.min().item())
            pos = _find_letter_position(ids_list, asst_start, letter, self.tokenizer)
            if pos is None:
                continue

            # one supervised token: the letter
            letter_id = ids_list[pos]
            elements.append(DatasetElement(
                input_ids=elem.input_ids,
                topk_token_ids=torch.tensor([letter_id], dtype=torch.long),
                topk_logprobs=torch.zeros(1, dtype=torch.float16),
                topk_token_idxs=torch.tensor([pos], dtype=torch.int32),
                metadata=[{"correct_letter": letter, "letter_pos": pos}],
                token_counts=TokenCounts(
                    num_system_and_user_tokens=asst_start,
                    num_assistant_tokens=len(ids_list) - asst_start,
                ),
            ))

        return elements
