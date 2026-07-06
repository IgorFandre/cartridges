"""
Muon optimizer for cartridge training.

Based on the original implementation by Keller Jordan:
https://github.com/KellerJordan/Muon

Muon applies Newton-Schulz orthogonalization to the EMA momentum gradient,
giving steepest descent in the spectral norm geometry rather than Euclidean.

KV cache tensors have shape (1, n_heads, n_tokens, head_dim). The canonical
Muon reshape is `view(len(G), -1)`, which works for conv weights (out, in, kH, kW)
but gives a degenerate 1×N matrix for our batch-size-1 KV tensors. Instead we
use `reshape(-1, head_dim)` so each token's KV vector is a row — a semantically
natural 2D structure for NS orthogonalization over the token subspace.
"""

import torch
from torch import Tensor
from torch.optim import Optimizer


def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """
    Newton-Schulz iteration to compute the nearest matrix with orthonormal
    columns to G (in Frobenius norm), scaled to have norm sqrt(min(m, n)).

    Operates in bfloat16 for numerical stability, returns in original dtype.
    """
    assert G.ndim == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X = X / (X.norm() + eps)
    if X.shape[0] > X.shape[1]:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.shape[0] > G.shape[1]:
        X = X.mT
    return X.to(G.dtype)


class Muon(Optimizer):
    """
    Muon — MomentUm Orthogonalized by Newton-Schulz.

    For parameters with ndim >= 2, applies NS orthogonalization to the
    Nesterov EMA momentum gradient. For 1-D parameters, applies plain
    SGD with momentum (no orthogonalization).

    Args:
        params: parameters to optimize (typically cache.parameters())
        lr: learning rate. Muon typically uses much higher lr than Adam
            (0.02 is the canonical default vs 1e-4 for Adam).
        momentum: EMA momentum coefficient β (default 0.95).
            Momentum buffer: m = β*m + (1-β)*g  (EMA form, not heavy-ball).
        weight_decay: decoupled weight decay (default 0.0).
        ns_steps: number of Newton-Schulz iterations (default 5).
    """

    def __init__(
        self,
        params,
        lr: float = 0.02,
        momentum: float = 0.95,
        weight_decay: float = 0.0,
        ns_steps: int = 5,
    ):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            beta = group["momentum"]
            weight_decay = group["weight_decay"]
            ns_steps = group["ns_steps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                state = self.state[p]

                if not state:
                    state["momentum_buffer"] = torch.zeros_like(g)

                buf = state["momentum_buffer"]

                # EMA momentum update: m = β*m + (1-β)*g
                buf.lerp_(g, 1 - beta)

                # Nesterov lookahead: interpolate between g and m
                # update = (1-β)*g + β*m  (equivalent to grad.lerp_(buf, beta))
                update = g.lerp(buf, beta)

                if update.ndim >= 2:
                    original_shape = update.shape
                    # Reshape to 2D with last dim as feature dim.
                    # For KV cache (1, n_heads, n_tokens, head_dim) → (n_heads*n_tokens, head_dim).
                    # (The canonical `view(len(G), -1)` gives a degenerate (1, huge) for our tensors.)
                    g_2d = update.reshape(-1, update.shape[-1])
                    g_orth = zeropower_via_newtonschulz5(g_2d, steps=ns_steps)
                    # Scale by aspect ratio: compensates for NS output norm being sqrt(min(m,n)).
                    scale = max(1.0, g_2d.shape[0] / g_2d.shape[1]) ** 0.5
                    update = g_orth.reshape(original_shape) * scale

                # Decoupled weight decay
                if weight_decay != 0.0:
                    p.mul_(1.0 - lr * weight_decay)

                p.add_(update, alpha=-lr)
