from pathlib import Path
from typing import Optional

import torch
from transformers import AutoTokenizer

from cartridges.cache import AttnConfig, KVCacheFactory, TrainableCache
from cartridges.utils import get_logger

logger = get_logger(__name__)


class KVFromAttnMatching(KVCacheFactory):
    """Initialize a TrainableCache from an attention-matching checkpoint.

    The checkpoint format is a dict produced by torch.save with keys:
        format_version: int
        meta: dict
        layers: list of dicts, each with:
            c1: (B, KV, T, D)  — compact keys
            c2: (B, KV, T, D)  — compact values
            beta: (B, KV, T)   — bias; -inf marks padding positions

    Per-layer token counts may differ across layers, and per-head token counts
    may differ within a layer (indicated by -inf in beta).  This initializer:
    1. Uses beta as a validity mask to find real vs padded positions per head.
    2. Selects the minimum valid length across all heads in a layer as that
       layer's cartridge length (so all heads get the same real prefix).
    3. Pads shorter layers with zeros to the global maximum length.
    4. Stores per-layer valid lengths in the cache so that get_score_mod() can
       return a FlexAttention score_mod that masks the zero-padded positions.
    """

    class Config(KVCacheFactory.Config):
        path: str
        num_frozen_tokens: int = 0

    def __init__(self, config: Config):
        self.config = config

    def initialize_kv_cache(
        self,
        tokenizer: Optional[AutoTokenizer] = None,
        model: Optional[torch.nn.Module] = None,
        attn_config: Optional[AttnConfig] = None,
    ) -> TrainableCache:
        path = Path(self.config.path)
        logger.info(f"Loading attention-matching checkpoint from {path}")
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        layers = checkpoint["layers"]
        n_layers = len(layers)

        if attn_config is not None and attn_config.n_layers != n_layers:
            raise ValueError(
                f"Checkpoint has {n_layers} layers but attn_config expects "
                f"{attn_config.n_layers}"
            )

        per_layer_keys: list[torch.Tensor] = []
        per_layer_values: list[torch.Tensor] = []
        per_layer_valid_lengths: list[int] = []
        per_layer_beta_list: list[torch.Tensor] = []

        for layer_idx, layer_data in enumerate(layers):
            c1 = layer_data["c1"]   # (B, KV, T, D) or (KV, T, D)
            c2 = layer_data["c2"]
            beta = layer_data["beta"]  # (B, KV, T) or (KV, T)

            # Normalise to (KV, T, D) / (KV, T)
            if c1.dim() == 4:
                c1 = c1[0]
                c2 = c2[0]
                beta = beta[0]

            n_kv_heads, T, D = c1.shape

            if attn_config is not None and attn_config.n_heads != n_kv_heads:
                raise ValueError(
                    f"Layer {layer_idx}: checkpoint has {n_kv_heads} KV heads "
                    f"but attn_config expects {attn_config.n_heads}"
                )

            # Find minimum valid (non-padded) length across all heads.
            # beta[h, t] == -inf indicates position t is padding for head h.
            min_valid = T
            for h in range(n_kv_heads):
                valid = int(torch.isfinite(beta[h]).sum().item())
                if valid < min_valid:
                    min_valid = valid

            if min_valid == 0:
                raise ValueError(
                    f"Layer {layer_idx} has no valid tokens in at least one head"
                )

            # Truncate all heads to min_valid (drop padding).
            per_layer_keys.append(c1[:, :min_valid, :].unsqueeze(0).contiguous())   # (1, KV, T_l, D)
            per_layer_values.append(c2[:, :min_valid, :].unsqueeze(0).contiguous())
            per_layer_valid_lengths.append(min_valid)
            # Store finite beta values for valid positions — used as attention bias.
            per_layer_beta_list.append(beta[:, :min_valid].contiguous())  # (KV, T_l)

            logger.info(f"Layer {layer_idx}: T={T}, min_valid={min_valid}")

        global_max = max(per_layer_valid_lengths)
        logger.info(f"Global max valid length across layers: {global_max}")

        # Pad beta tensors to global_max and stack into (n_layers, n_kv_heads, global_max).
        n_kv_heads_global = per_layer_beta_list[0].shape[0]
        beta_dtype = per_layer_beta_list[0].dtype
        per_layer_beta_padded = torch.zeros(
            n_layers, n_kv_heads_global, global_max, dtype=beta_dtype
        )
        for l, b in enumerate(per_layer_beta_list):
            per_layer_beta_padded[l, :, :b.shape[1]] = b

        # Build attn_config from checkpoint data if not supplied.
        if attn_config is None:
            _, D = per_layer_keys[0].shape[2], per_layer_keys[0].shape[3]
            n_kv_heads = per_layer_keys[0].shape[1]
            attn_config = AttnConfig(
                n_layers=n_layers,
                n_heads=n_kv_heads,
                head_dim=D,
            )

        # Pad shorter layers to global_max with zeros.  Zero-padded positions
        # are masked with -inf via score_mod during attention so they don't
        # contribute to the output.
        init_keys: list[torch.Tensor] = []
        init_values: list[torch.Tensor] = []
        for l, (k, v) in enumerate(zip(per_layer_keys, per_layer_values)):
            T_l = k.shape[2]
            if T_l < global_max:
                pad_shape = (1, k.shape[1], global_max - T_l, k.shape[3])
                pad_k = torch.zeros(pad_shape, dtype=k.dtype)
                pad_v = torch.zeros(pad_shape, dtype=v.dtype)
                k = torch.cat([k, pad_k], dim=2)
                v = torch.cat([v, pad_v], dim=2)
            init_keys.append(k)
            init_values.append(v)

        num_frozen = self.config.num_frozen_tokens
        cache = TrainableCache(
            config=attn_config,
            init_keys=init_keys,
            init_values=init_values,
            num_frozen_tokens=num_frozen,
            per_layer_valid_lengths=per_layer_valid_lengths,
            per_layer_beta=per_layer_beta_padded,
        )

        logger.info(
            f"Initialized TrainableCache from attention-matching checkpoint: "
            f"{n_layers} layers, {global_max} max cartridge tokens, "
            f"valid lengths: {per_layer_valid_lengths}"
        )
        return cache
