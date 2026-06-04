"""
LineageYesNoEvalDataset — generation-eval dataset for lineage cartridge training.

Tokenizes the user-only prompt (no graph in context; the cartridge supplies it),
enables CoT thinking if requested, scores by extracting Yes/No from the model's
free-form response.

Per-row metadata (n_bucket, true_distance, direction) is attached to the
GenerateEvalDatasetElement so train.py logs it to the wandb table, enabling
offline per-n-hop analysis.
"""
from __future__ import annotations
from typing import Tuple, Dict, Any

from cartridges.datasets import GenerateEvalDataset, GenerateEvalDatasetElement
from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE, MODELS_WITH_THINKING


class LineageYesNoEvalDataset(GenerateEvalDataset):
    """Eval dataset for Yes/No lineage questions; scores by Yes/No extraction."""

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

        # User-only prompt (graph is in the cartridge, not in the context here)
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": m.role, "content": m.content} for m in convo.messages[:-1]],
            add_generation_prompt=True,
            return_tensors="pt",
            chat_template=MODEL_TO_CHAT_TEMPLATE.get(model_name),
            **kwargs,
        )

        md = dict(convo.metadata or {})
        gold_label = md.get("label", "")
        # Normalise: strip trailing period if present
        if gold_label.endswith("."):
            gold_label = gold_label[:-1]

        # Stringify any non-scalar metadata fields for wandb table compatibility
        for key in list(md.keys()):
            if isinstance(md[key], (list, dict)):
                md[key] = str(md[key])

        return GenerateEvalDatasetElement(
            input_ids=input_ids,
            prompt=[{"role": m.role, "content": m.content} for m in convo.messages[:-1]],
            answer=gold_label,
            convo_id=str(index),
            metadata={
                "idx":           index,
                "n_bucket":      md.get("n_bucket"),
                "true_distance": md.get("true_distance"),
                "claimed_n":     md.get("claimed_n"),
                "direction":     md.get("direction"),
                "label":         gold_label,
                "question_text": md.get("question_text", ""),
            },
        )

    def score(
        self,
        pred: str,
        answer: str,
        convo_id: str,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        # Import here to avoid circular import; extract_yes_no lives in evaluation
        from examples.graph_2.evaluation.lineage_eval import extract_yes_no

        pred_yn = extract_yes_no(pred)
        gold_yn = (answer or "").strip().capitalize()
        correct = pred_yn is not None and pred_yn == gold_yn and gold_yn in ("Yes", "No")
        return (
            {"acc": float(correct)},
            {"pred_yn": pred_yn or "?", "gold_yn": gold_yn},
        )
