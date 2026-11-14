"""Depth-map degradation (brief §6.4, trick 2). Applied to the DEPTH MAP ONLY.

Each call samples fresh random magnitudes:
  1. small random affine (translate ±4%, scale U(0.95,1.05), rotate ±3°), edge-value fill
  2. down-then-up bilinear resample by a factor from {1, 2, 4, 8}
  3. Gaussian blur, sigma ~ U(0, 4) px (at 512 res)
`cfg.enabled = False` returns the map unchanged.
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F


def _affine(d: torch.Tensor, tx: float, ty: float, scale: float, rot_deg: float) -> torch.Tensor:
    """d: [B,C,H,W]. Translate by (tx,ty) fractions of W/H, scale, rotate; border (edge-value) fill."""
    a = math.radians(rot_deg)
    cos, sin = math.cos(a) / scale, math.sin(a) / scale
    # affine_grid maps output coords -> input coords (normalized [-1,1]); translation of +tx in
    # the image means sampling from x - 2*tx in normalized coords.
    theta = torch.tensor([[cos, -sin, -2 * tx], [sin, cos, -2 * ty]], dtype=d.dtype, device=d.device)
    theta = theta.unsqueeze(0).expand(d.shape[0], 2, 3)
    grid = F.affine_grid(theta, list(d.shape), align_corners=False)
    return F.grid_sample(d, grid, mode="bilinear", padding_mode="border", align_corners=False)


def _blur(d: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma < 0.3:
        return d
    r = int(math.ceil(3 * sigma))
    x = torch.arange(-r, r + 1, dtype=d.dtype, device=d.device)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    k = (k / k.sum()).view(1, 1, 1, -1)
    C = d.shape[1]
    d = F.pad(d, (r, r, r, r), mode="replicate")
    d = F.conv2d(d, k.expand(C, 1, 1, -1), groups=C)                      # horizontal
    d = F.conv2d(d, k.view(1, 1, -1, 1).expand(C, 1, -1, 1), groups=C)    # vertical
    return d


def degrade(depth: torch.Tensor, cfg: dict, rng: random.Random | None = None) -> tuple[torch.Tensor, dict]:
    """depth: [C,H,W] or [B,C,H,W] in [0,1]. Returns (degraded depth, sampled params)."""
    if not cfg.get("enabled", False):
        return depth, {}
    rng = rng or random
    squeeze = depth.dim() == 3
    d = depth.unsqueeze(0) if squeeze else depth
    H, W = d.shape[-2:]
    p = {
        "tx": rng.uniform(-cfg["translate_frac"], cfg["translate_frac"]),
        "ty": rng.uniform(-cfg["translate_frac"], cfg["translate_frac"]),
        "scale": rng.uniform(*cfg["scale_range"]),
        "rot": rng.uniform(-cfg["rotate_deg"], cfg["rotate_deg"]),
        "factor": rng.choice(cfg["resample_factors"]),
        "sigma": rng.uniform(0.0, cfg["blur_sigma_max"]),
    }
    d = _affine(d, p["tx"], p["ty"], p["scale"], p["rot"])
    if p["factor"] > 1:
        small = F.interpolate(d, size=(H // p["factor"], W // p["factor"]), mode="bilinear", align_corners=False,
                              antialias=True)
        d = F.interpolate(small, size=(H, W), mode="bilinear", align_corners=False)
    d = _blur(d, p["sigma"]).clamp(0, 1)
    return (d[0] if squeeze else d), {k: round(v, 4) if isinstance(v, float) else v for k, v in p.items()}
