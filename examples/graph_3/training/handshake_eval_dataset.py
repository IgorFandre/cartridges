"""
HandshakeEvalDataset — generation-eval dataset for handshake cartridge training.

Tokenizes the user-only prompt (no corpus in context; the cartridge supplies
it), scores by extracting the distance answer ("3" / "not connected") from the
model's free-form response with the same parser as evaluation/eval.py.

Thinking is off by default (cot=False): the push/pop scratchpad IS the visible
reasoning (PLAN.md §3).

Per-row metadata (n_bucket, true_distance, x, y) is attached to the
GenerateEvalDatasetElement so train.py logs it to the wandb table, enabling
offline per-hop analysis.
"""
from __future__ import annotations
from typing import Tuple, Dict, Any

from cartridges.datasets import GenerateEvalDataset, GenerateEvalDatasetElement
from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE, MODELS_WITH_THINKING


class HandshakeEvalDataset(GenerateEvalDataset):
    """Eval dataset for handshake-distance questions; scores by answer extraction."""

    class Config(GenerateEvalDataset.Config):
        _pass_as_config = True
        # inherits: data_source (DataSource | str), cot (bool)

    def __getitem__(self, index: int) -> GenerateEvalDatasetElement:
        convo = self.data[index]
        assert len(convo.messages) >= 2

        kwargs: dict = {}
        model_name = self.tokenizer.name_or_path
        if model_name.lower() in {m.lower() for m in MODELS_WITH_THINKING}:
            kwargs["enable_thinking"] = self.config.cot

        # User-only prompt (the corpus is in the cartridge, not in the context)
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in convo.messages[:-1]],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=MODEL_TO_CHAT_TEMPLATE.get(model_name),
            **kwargs,
        )

        md = dict(convo.metadata or {})
        gold = md.get("answer", "")

        # Stringify any non-scalar metadata fields for wandb table compatibility
        for key in list(md.keys()):
            if isinstance(md[key], (list, dict)):
                md[key] = str(md[key])

        return GenerateEvalDatasetElement(
            input_ids=input_ids,
            prompt=[{"role": m.role, "content": m.content} for m in convo.messages[:-1]],
            answer=gold,
            convo_id=str(index),
            metadata={
                "idx":           index,
                "n_bucket":      md.get("n_bucket"),
                "true_distance": md.get("true_distance"),
                "x":             md.get("x"),
                "y":             md.get("y"),
                "question":      md.get("question", ""),
            },
        )

    def score(
        self,
        pred: str,
        answer: str,
        convo_id: str,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        # Import here to avoid a circular import; the parser lives in evaluation
        from examples.graph_3.evaluation.eval import extract_answer

        pred_ans = extract_answer(pred)
        correct = pred_ans is not None and pred_ans == answer
        return (
            {"acc": float(correct)},
            {"pred": pred_ans or "?", "gold": answer},
        )
