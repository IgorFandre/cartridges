import argparse
import torch
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.generation import flex_generate
from cartridges.models.qwen.modeling_qwen3 import FlexQwen3ForCausalLM


def load_cache_tensors(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    needed = ["trainable_keys", "trainable_values", "frozen_keys", "frozen_values"]
    for k in needed:
        if k not in ckpt:
            raise KeyError(f"Checkpoint {path} does not contain key: {k}")

    train_k = ckpt["trainable_keys"]
    train_v = ckpt["trainable_values"]
    froz_k = ckpt["frozen_keys"]
    froz_v = ckpt["frozen_values"]

    n_layers = len(train_k)
    if n_layers == 0:
        raise ValueError(f"Empty checkpoint: {path}")

    per_layer_k = []
    per_layer_v = []

    for i in range(n_layers):
        k_parts = []
        v_parts = []

        if len(froz_k) > 0 and froz_k[i] is not None and froz_k[i].numel() > 0:
            k_parts.append(froz_k[i].detach().to(torch.float32))
        if len(froz_v) > 0 and froz_v[i] is not None and froz_v[i].numel() > 0:
            v_parts.append(froz_v[i].detach().to(torch.float32))

        k_parts.append(train_k[i].detach().to(torch.float32))
        v_parts.append(train_v[i].detach().to(torch.float32))

        k_cat = torch.cat(k_parts, dim=2).contiguous() if len(k_parts) > 1 else k_parts[0].contiguous()
        v_cat = torch.cat(v_parts, dim=2).contiguous() if len(v_parts) > 1 else v_parts[0].contiguous()

        per_layer_k.append(k_cat)
        per_layer_v.append(v_cat)

    n_heads = per_layer_k[0].shape[1]
    head_dim = per_layer_k[0].shape[3]
    cfg = AttnConfig(n_layers=n_layers, n_heads=n_heads, head_dim=head_dim)
    num_tokens = per_layer_k[0].shape[2]

    return cfg, per_layer_k, per_layer_v, num_tokens


def compose_caches(cache_paths):
    loaded = [load_cache_tensors(p) for p in cache_paths]
    print(f'Загружено {len(loaded)} кешей')
    base_cfg = loaded[0][0]

    for idx, (cfg, _, _, _) in enumerate(loaded[1:], start=1):
        if cfg != base_cfg:
            raise ValueError(
                f"Cache at index {idx} has different attn config: {cfg} vs {base_cfg}"
            )

    n_layers = base_cfg.n_layers
    composed_keys = []
    composed_values = []

    token_counts = []
    for _, _, _, n_tok in loaded:
        token_counts.append(n_tok)

    for layer in range(n_layers):
        k_list = [item[1][layer] for item in loaded]
        v_list = [item[2][layer] for item in loaded]
        composed_keys.append(torch.cat(k_list, dim=2).contiguous())
        composed_values.append(torch.cat(v_list, dim=2).contiguous())

    total_tokens = composed_keys[0].shape[2]
    print(f"Loaded {len(cache_paths)} cartridges")
    print(f"Tokens per cartridge: {token_counts}")
    print(f"Total composed cartridge tokens: {total_tokens}")

    # num_frozen_tokens=0 is intentional for inference-only composed cache
    cache = TrainableCache(
        config=base_cfg,
        init_keys=composed_keys,
        init_values=composed_values,
        num_frozen_tokens=0,
    )
    return cache


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_paths", nargs="+", required=True)
    parser.add_argument("--model_name", default="Qwen/Qwen3-4b")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--enable_thinking", action="store_true")
    args = parser.parse_args()

    print("Loading model...")
    model = (
        FlexQwen3ForCausalLM
        .from_pretrained(args.model_name)
        .to("cuda")
        .to(torch.bfloat16)
        .eval()
    )

    print("Composing local caches...")
    cache = compose_caches(args.cache_paths).to("cuda").to(torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    history = []

    print("Chat started. Commands: /clear, /quit")
    while True:
        user_text = input("\nYou: ").strip()
        if not user_text:
            continue
        if user_text == "/quit":
            break
        if user_text == "/clear":
            history = []
            print("History cleared")
            continue

        history.append({"role": "user", "content": user_text})

        input_ids = tokenizer.apply_chat_template(
            history,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            enable_thinking=args.enable_thinking,
        ).to("cuda")

        flat_input_ids = input_ids.flatten()
        seq_ids = torch.zeros(flat_input_ids.shape[0], dtype=torch.long, device="cuda")
        position_ids = torch.arange(flat_input_ids.shape[0], dtype=torch.long, device="cuda")

        pred_ids = flex_generate(
            model=model,
            tokenizer=tokenizer,
            input_ids=flat_input_ids,
            seq_ids=seq_ids,
            position_ids=position_ids,
            cache=cache,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            show_progress=False,
        )

        tokens = pred_ids.get(0, [])
        answer = tokenizer.decode(tokens, skip_special_tokens=True) if tokens else "No response generated."
        history.append({"role": "assistant", "content": answer})
        print(f"Assistant: {answer}")


if __name__ == "__main__":
    main()