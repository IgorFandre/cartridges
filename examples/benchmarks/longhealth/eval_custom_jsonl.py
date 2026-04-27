"""
Evaluate a trained cartridge checkpoint on a custom JSONL file of multiple-choice questions.

Usage:
    python examples/benchmarks/longhealth/eval_custom_jsonl.py \
        --checkpoint /path/to/cache-step1295.pt \
        --questions /path/to/generated_patient01_questions_60.jsonl \
        [--model qwen1.7b|qwen4b|llama] \
        [--max-new-tokens 512] \
        [--batch-size 8] \
        [--device cuda]
"""
import argparse
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from cartridges.cache import TrainableCache
from cartridges.data.longhealth.utils import LongHealthQuestion
from cartridges.generation import flex_generate
from cartridges.initialization.tokenization_utils import MODEL_TO_CHAT_TEMPLATE, MODELS_WITH_THINKING


MODEL_CONFIGS = {
    "qwen1.7b": "Qwen/Qwen3-1.7B",
    "qwen4b": "Qwen/Qwen3-4b",
    "llama": "meta-llama/Llama-3.2-3B-Instruct",
}


def load_questions(path: str) -> List[LongHealthQuestion]:
    questions = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            questions.append(LongHealthQuestion(
                question_id=data["question_id"],
                question=data["question"],
                correct=data["correct"],
                answer_a=data["answer_a"],
                answer_b=data["answer_b"],
                answer_c=data["answer_c"],
                answer_d=data["answer_d"],
                answer_e=data["answer_e"],
                answer_location=data.get("answer_location"),
            ))
    return questions


def wrap_question(question: LongHealthQuestion, tokenizer, cot: bool = True) -> str:
    options = (
        f"{question.answer_a}\n"
        f"{question.answer_b}\n"
        f"{question.answer_c}\n"
        f"{question.answer_d}\n"
        f"{question.answer_e}"
    )
    if cot and tokenizer.name_or_path not in MODELS_WITH_THINKING:
        cot_prompt = (
            "You should first think step by step. Then give your final answer exactly as it appears "
            "in the options. Your output should be in the following format: \n"
            "<thinking> {{YOUR_THOUGHT_PROCESS}} </thinking> "
        )
    else:
        cot_prompt = "Please provide your answer exactly as it appears in the options with the following format:"

    return (
        f"Please answer the question below.\n\n"
        f"<question>\n{question.question}\n</question>\n\n"
        f"<options>\n{options}\n</options>\n{cot_prompt}"
        f"\n\n<answer>\n{{YOUR_ANSWER}}\n</answer>"
    )


def score_prediction(pred: str, question: LongHealthQuestion) -> Tuple[bool, Optional[str]]:
    options = [
        question.answer_a.strip().lower(),
        question.answer_b.strip().lower(),
        question.answer_c.strip().lower(),
        question.answer_d.strip().lower(),
        question.answer_e.strip().lower(),
    ]

    match = re.search(r'<answer>(.*?)</answer>', pred, re.DOTALL)
    if match:
        extracted = match.group(1).strip().lower()
        closest = max(options, key=lambda x: SequenceMatcher(None, extracted, x).ratio())
        return closest == question.correct.strip().lower(), extracted
    else:
        return question.answer_a.strip().lower() == question.correct.strip().lower(), None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to .pt cache checkpoint")
    parser.add_argument("--questions", required=True, help="Path to JSONL questions file")
    parser.add_argument("--model", default="qwen1.7b", choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", default=None, help="Save per-question results to JSON")
    args = parser.parse_args()

    device = args.device
    model_name = MODEL_CONFIGS[args.model]

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(f"Loading model: {model_name}")
    if args.model in ("qwen1.7b", "qwen4b"):
        from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM
        model = FlexQwen3ForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    else:
        from cartridges.models.llama.modeling_llama import FlexLlamaForCausalLM
        model = FlexLlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    model.eval()

    print(f"Loading cache checkpoint: {args.checkpoint}")
    cache: TrainableCache = TrainableCache.from_pretrained(args.checkpoint, device=device)
    cache = cache.to(device)

    print(f"Loading questions: {args.questions}")
    questions = load_questions(args.questions)
    print(f"  {len(questions)} questions loaded")

    results = []
    correct_count = 0

    for batch_start in tqdm(range(0, len(questions), args.batch_size), desc="Evaluating"):
        batch = questions[batch_start: batch_start + args.batch_size]

        prompts = [wrap_question(q, tokenizer) for q in batch]

        kwargs = {}
        if tokenizer.name_or_path in MODELS_WITH_THINKING:
            kwargs["enable_thinking"] = True

        input_ids_list = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True,
                return_tensors="pt",
                chat_template=MODEL_TO_CHAT_TEMPLATE.get(tokenizer.name_or_path),
                **kwargs,
            )
            for p in prompts
        ]

        input_ids = torch.cat([ids[0] for ids in input_ids_list]).to(device)
        seq_ids = torch.cat([
            torch.full((ids.shape[1],), idx, dtype=torch.long, device=device)
            for idx, ids in enumerate(input_ids_list)
        ])
        position_ids = torch.cat([
            torch.arange(ids.shape[1], device=device) for ids in input_ids_list
        ])

        with torch.no_grad():
            pred_ids: Dict[int, List[int]] = flex_generate(
                input_ids=input_ids,
                seq_ids=seq_ids,
                position_ids=position_ids,
                cache=cache,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                show_progress=False,
            )

        for seq_id, token_ids in pred_ids.items():
            question = batch[seq_id]
            pred_text = tokenizer.decode(token_ids, skip_special_tokens=True)
            is_correct, extracted = score_prediction(pred_text, question)
            correct_count += int(is_correct)
            results.append({
                "question_id": question.question_id,
                "correct": is_correct,
                "extracted_pred": extracted,
                "expected": question.correct,
                "pred_text": pred_text,
            })

    total = len(results)
    score = correct_count / total if total > 0 else 0.0
    print(f"\nScore: {correct_count}/{total} = {score:.4f} ({score*100:.1f}%)")

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps({
            "score": score,
            "correct": correct_count,
            "total": total,
            "results": results,
        }, indent=2, ensure_ascii=False))
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
