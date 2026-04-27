from typing import Optional

import torch

from cartridges.cache import AttnConfig, TrainableCache
from cartridges.initialization.text import KVFromText


class KVFromTextPerturbed(KVFromText):
    class Config(KVFromText.Config):
        key_scale: float = 1.0
        key_noise_std: float = 0.0
        value_scale: float = 1.0
        noise_seed: Optional[int] = None

    def initialize_kv_cache(
        self,
        tokenizer,
        model,
        attn_config: AttnConfig,
    ) -> TrainableCache:
        cache = super().initialize_kv_cache(tokenizer, model, attn_config)

        gen = None
        if self.config.noise_seed is not None:
            gen = torch.Generator(device=cache.trainable_keys[0].device)
            gen.manual_seed(self.config.noise_seed)

        with torch.no_grad():
            for k in cache.trainable_keys:
                if self.config.key_scale != 1.0:
                    k.data.mul_(self.config.key_scale)
                if self.config.key_noise_std > 0.0:
                    noise = torch.empty_like(k.data).normal_(
                        mean=0.0, std=self.config.key_noise_std, generator=gen
                    )
                    k.data.add_(noise)
            if self.config.value_scale != 1.0:
                for v in cache.trainable_values:
                    v.data.mul_(self.config.value_scale)

        return cache
