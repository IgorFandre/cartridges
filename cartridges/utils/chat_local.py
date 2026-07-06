#!/usr/bin/env python3
"""
Interactive CLI for chatting with a cartridge checkpoint file that lives on
disk (e.g. a `cache_last.pt` / `cache-step*.pt` copied from the GPU box you
trained on) — no wandb, no CUDA required.

Unlike `cartridges.utils.chat` (which pulls the checkpoint + model from a
wandb run and hardcodes `cuda`), this loads a local .pt file and runs on
whatever device you point it at, defaulting to CPU. `torch.compile` for
FlexAttention only supports CUDA, so on CPU we swap in the eager
implementation (materializes the full attention matrix — fine at these
sequence lengths, just slower than the compiled CUDA kernel).

Usage:
    python -m cartridges.utils.chat_local \
        outputs_graph3/exp1_adaptive/train/.../cache_last.pt \
        --model Qwen/Qwen3-1.7B
"""

import argparse
import sys
import readline
from typing import List, Dict

import torch
from transformers import AutoTokenizer


def _patch_flex_attention_for_cpu():
    """Swap the compiled FlexAttention kernels for the eager op.

    `cartridges.models.attention` compiles `flex_attention` at import time
    with `torch.compile(...)`, which only has a CUDA/Triton lowering.
    `flex_attention_forward` reads `flex_attention_train`/`flex_attention_generate`
    off the module at call time, so overwriting them here (before any
    generation happens) is enough — no need to touch the shared file that
    the GPU training/eval scripts depend on.
    """
    import cartridges.models.attention as attn_mod
    from torch.nn.attention.flex_attention import flex_attention

    attn_mod.flex_attention_train = flex_attention
    attn_mod.flex_attention_generate = flex_attention


class ChatSession:
    def __init__(self, model, tokenizer, cache, device: str):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache
        self.device = device
        self.conversation_history: List[Dict[str, str]] = []
        self.input_history: List[str] = []

    def add_message(self, role: str, content: str):
        self.conversation_history.append({"role": role, "content": content})

    def undo_last_message(self):
        if len(self.conversation_history) >= 2:
            self.conversation_history = self.conversation_history[:-2]
            return True
        elif len(self.conversation_history) == 1:
            self.conversation_history = []
            return True
        return False

    def clear_conversation(self):
        self.conversation_history = []

    def generate_response(
        self, user_input: str, enable_thinking: bool, max_new_tokens: int
    ) -> str:
        if user_input.strip() and user_input not in self.input_history:
            self.input_history.append(user_input)
            readline.add_history(user_input)

        self.add_message("user", user_input)

        input_ids = self.tokenizer.apply_chat_template(
            self.conversation_history,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=enable_thinking,
        ).to(self.device)

        flat_input_ids = input_ids.flatten()
        seq_ids = torch.zeros(flat_input_ids.shape[0], dtype=torch.long, device=self.device)
        position_ids = torch.arange(flat_input_ids.shape[0], device=self.device)

        from cartridges.generation import flex_generate

        output = flex_generate(
            model=self.model,
            input_ids=flat_input_ids,
            seq_ids=seq_ids,
            position_ids=position_ids,
            tokenizer=self.tokenizer,
            cache=self.cache,
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            show_progress=True,
        )

        if 0 in output and output[0]:
            response_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            self.add_message("assistant", response_text)
            return response_text
        return "I'm sorry, I couldn't generate a response."


def main():
    parser = argparse.ArgumentParser(description="Chat with a local cartridge checkpoint")
    parser.add_argument("checkpoint", help="Path to cache_last.pt / cache-step*.pt")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B", help="Base model the cartridge was trained on")
    parser.add_argument("--device", default=None, help="cpu | cuda | mps (default: auto-detect, falls back to cpu)")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable Qwen3 <think>")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Push/pop scratchpads can run long on deep hops")
    args = parser.parse_args()

    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    if device == "cpu":
        _patch_flex_attention_for_cpu()

    dtype = torch.bfloat16 if device != "cpu" else torch.float32

    print(f"Device: {device}  dtype: {dtype}")
    print(f"Loading base model {args.model} ...")

    from cartridges.cache import TrainableCache
    from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM

    try:
        model = FlexQwen3ForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device)
        model.eval()

        print(f"Loading cartridge from {args.checkpoint} ...")
        cache = TrainableCache.from_pretrained(args.checkpoint, device=device).to(device).to(dtype)

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        print("Model and cache loaded successfully!\n")
    except Exception as e:
        print(f"Error loading model and cache: {e}")
        sys.exit(1)

    chat = ChatSession(model, tokenizer, cache, device)
    readline.set_startup_hook(None)

    print("=== Chat with local cartridge ===")
    print("Commands:")
    print("  /undo  - Undo the last message exchange")
    print("  /clear - Clear the entire conversation")
    print("  /quit  - Exit the chat")
    print("  /help  - Show this help message")
    print("Arrow keys: ↑↓ for command history, ←→ for line editing")
    print("-" * 40)

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue

            if user_input == "/quit":
                print("Goodbye!")
                break
            elif user_input == "/help":
                print("\nCommands:")
                print("  /undo  - Undo the last message exchange")
                print("  /clear - Clear the entire conversation")
                print("  /quit  - Exit the chat")
                print("  /help  - Show this help message")
                continue
            elif user_input == "/undo":
                print("Last message exchange undone." if chat.undo_last_message() else "No messages to undo.")
                continue
            elif user_input == "/clear":
                chat.clear_conversation()
                print("Conversation cleared.")
                continue

            print("Assistant: ", end="", flush=True)
            response = chat.generate_response(
                user_input, enable_thinking=args.enable_thinking, max_new_tokens=args.max_new_tokens
            )
            print(response)

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError generating response: {e}")
            print("Type /help for available commands.")


if __name__ == "__main__":
    main()
