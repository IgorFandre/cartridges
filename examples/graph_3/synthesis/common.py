"""
Shared helpers for the graph_3 self-study synthesis scripts (exp1/exp2).

Both experiments iterate over the SAME train questions (qagen's
train_meta.json), ask a Tokasaurus-served model to answer with the friendship
corpus in context, and store the model's own traces (with top-k logprobs) as
training Conversations. They differ only in the hint policy — see
exp1_synthesize.py / exp2_synthesize.py.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Iterator

from cartridges.structs import Conversation

# ── Env knobs (shared) ────────────────────────────────────────────────────────
SERVER_URL   = os.environ.get("CARTRIDGES_TOKASAURUS_URL", "http://localhost:8000")
SERVER_MODEL = os.environ.get("HANDSHAKE_SERVER_MODEL",    "Qwen/Qwen3-1.7B")
BATCH_SIZE   = int(os.environ.get("BATCH_SIZE", "32"))

# Push/pop scratchpads reach ~2.7k tokens — give generation headroom (PLAN §5).
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "4096"))

# Client request timeout (seconds) for the FIRST attempt; it grows ×1.5 per
# retry. The default of 90s is too short when long (4k-token) generations share
# the GPU with another synthesis — each request then decodes at ~35 tok/s and a
# 4k generation takes >100s, tripping a needless TimeoutError + retry (which
# re-does the work on the server). Bump it for parallel / long-trace runs.
CLIENT_BASE_TIMEOUT = int(os.environ.get("CLIENT_BASE_TIMEOUT", "300"))


def make_client(url: str = SERVER_URL, model: str = SERVER_MODEL):
    """Tokasaurus client with a longer first-attempt timeout (CLIENT_BASE_TIMEOUT)
    so long generations under shared GPU load don't trip needless retries."""
    from cartridges.clients.tokasaurus import TokasaurusClient

    return TokasaurusClient.Config(
        url=url,
        model_name=model,
        base_timeout=CLIENT_BASE_TIMEOUT,
    ).instantiate()


def batched(items: list, size: int) -> Iterator[list]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def load_train_meta(path: str | Path, limit: int | None = None) -> list[dict]:
    meta_path = Path(path)
    if not meta_path.exists():
        raise FileNotFoundError(
            f"Train meta not found: {meta_path}. Run data generation first:\n"
            "  python -m examples.graph_3.data_gen.generate_forest\n"
            "  python -m examples.graph_3.data_gen.qagen"
        )
    meta: list[dict] = json.loads(meta_path.read_text())
    return meta[:limit] if limit else meta


def system_graph_prompt(corpus_text: str) -> str:
    """Graph-in-context system prompt — same wording as the ICL eval, so the
    synthesis context matches what Exp 0 sees (no train/eval mismatch)."""
    from examples.graph_3.evaluation.eval import _SYSTEM_INSTRUCTION

    return _SYSTEM_INSTRUCTION.format(corpus=corpus_text)


# Step-by-step directive (the "stepbystep" dataset variant). Appended to the
# graph prompt so the model narrates the WHOLE search instead of jumping to the
# answer — addresses the shortcut traces seen in the default dataset (≈20% of
# deep-hop hint traces just copied `path:`/`Answer:` from the scratchpad tail).
STEPBYSTEP_DIRECTIVE = (
    "\n\nThink step by step and SHOW your full search. Starting from the first "
    "person, expand the friendship list one handshake at a time — list who is 1 "
    "handshake away, then 2, then 3, and so on, level by level — until you reach "
    "the second person. Write out every level of this search. Do NOT state the "
    'path or the final number until you have worked through the whole search, '
    'then finish with the "path:" line and the "Answer:" line.'
)


def artifact_parquet(output_dir: Path, exp_subdir: str, stepbystep: bool) -> Path:
    """Path to an experiment's dataset parquet; the stepbystep variant writes to
    a separate `dataset_stepbystep.parquet` so it never overwrites the default."""
    name = "dataset_stepbystep.parquet" if stepbystep else "dataset.parquet"
    d = Path(output_dir) / exp_subdir / "artifact"
    d.mkdir(parents=True, exist_ok=True)
    return d / name


def report_path(output_dir: Path, exp_subdir: str, base_name: str, stepbystep: bool) -> Path:
    """`<base>.json` or `<base>_stepbystep.json` under the experiment dir."""
    stem = base_name[:-5] if base_name.endswith(".json") else base_name
    suffix = "_stepbystep" if stepbystep else ""
    return Path(output_dir) / exp_subdir / f"{stem}{suffix}.json"


def make_convo(meta: dict, sample, *, source: str, correct: bool, temp: float) -> Conversation:
    """Training Conversation: empty system (the corpus lives in the cartridge),
    user = question, assistant = the model's own trace with top-k logprobs."""
    flat = None
    if sample.top_logprobs is not None:
        flat = sample.top_logprobs.flatten(threshold=0.99)

    return Conversation(
        system_prompt="",
        type="handshake_selfstudy",
        metadata={
            "question":      meta["question"],
            "answer":        meta["answer"],
            "true_distance": meta["true_distance"],
            "n_bucket":      str(meta["n_bucket"]),
            "x": meta["x"], "y": meta["y"],
            "source":        source,      # "no_hint" | "with_hint"
            "correct":       correct,     # the trace's final answer matches gold
            "temp":          temp,
        },
        messages=[
            Conversation.Message(
                role="user",
                content=meta["question"],
                token_ids=None,
                top_logprobs=None,
            ),
            Conversation.Message(
                role="assistant",
                content=sample.text,
                token_ids=sample.token_ids,
                top_logprobs=flat,
            ),
        ],
    )


def check_logprobs(kept: list[Conversation]) -> None:
    n_missing = sum(1 for c in kept if c.messages[-1].top_logprobs is None)
    if n_missing:
        print(
            f"WARNING: {n_missing}/{len(kept)} conversations have no top_logprobs. "
            "Check server configuration. Set targets='tokens' in the train config "
            "if the server doesn't emit logprobs."
        )


def save_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))
    print(f"Report → {path}")
