from pathlib import Path
from typing import Optional

import torch

from cartridges.cache import AttnConfig, KVCacheFactory, TrainableCache


class KVFromSampledChunks(KVCacheFactory):
    """Sampled Chunk Initialization (SCI) — Algorithm 2 from https://arxiv.org/abs/2508.17032.

    Instead of taking the first k tokens from a fixed text (KVFromText), SCI randomly
    samples nchunks=floor(p/c) non-overlapping windows of size c from the full corpus,
    concatenates them, and runs a forward pass to produce a diverse initialization.

    Paper reports faster convergence than First-k Token Initialization (p < 0.05).
    Recommended chunk_size=64 (midpoint of n-gram diversity vs. context length tradeoff).
    """

    class Config(KVCacheFactory.Config):
        max_tokens: int
        chunk_size: int = 64
        text_source: Optional[str] = None
        text: Optional[str] = None
        seed: Optional[int] = None

    def initialize_kv_cache(
        self,
        tokenizer,
        model,
        attn_config: AttnConfig,
    ) -> TrainableCache:
        if self.config.text is not None:
            content = self.config.text
        elif self.config.text_source is not None:
            content = Path(self.config.text_source).read_text()
        else:
            raise ValueError("KVFromSampledChunks requires either 'text' or 'text_source'")

        token_ids = tokenizer.encode(content, add_special_tokens=False)
        x = torch.tensor(token_ids, dtype=torch.long)
        n_total = len(x)

        p = self.config.max_tokens
        c = self.config.chunk_size

        if n_total < c:
            raise ValueError(
                f"Corpus has {n_total} tokens, which is less than chunk_size={c}. "
                "Use a larger corpus or a smaller chunk_size."
            )

        n_chunks = p // c

        generator = torch.Generator()
        if self.config.seed is not None:
            generator.manual_seed(self.config.seed)

        starts = torch.randint(0, n_total - c + 1, (n_chunks,), generator=generator)
        chunks = [x[s : s + c] for s in starts]
        x_init = torch.cat(chunks, dim=0)[:p]

        init_cache = TrainableCache(config=attn_config)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                input_ids = x_init.to(model.device)
                seq_ids = torch.zeros(x_init.shape[0], dtype=torch.long, device=model.device)
                position_ids = torch.arange(x_init.shape[0], dtype=torch.long, device=model.device)
                model(
                    input_ids=input_ids,
                    seq_ids=seq_ids,
                    position_ids=position_ids,
                    use_cache=True,
                    past_key_values=init_cache,
                    mode="generate",
                )

        return TrainableCache(
            config=attn_config,
            init_keys=init_cache._keys,
            init_values=init_cache._values,
            num_frozen_tokens=self.config.num_frozen_tokens,
        )
