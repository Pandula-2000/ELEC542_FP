"""Training data: processed subject images -> cached latents, depth maps, latent-res masks."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from boothnet.utils import ROOT, load_json


def load_split(subject: str, processed_dir="data/processed") -> dict:
    return load_json(ROOT / processed_dir / subject / "split.json")


def _img_tensor(path: Path) -> torch.Tensor:
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)  # [3,H,W] in [0,1]


def load_train_set(subject: str, processed_dir="data/processed") -> dict:
    """Pixel-space tensors for the subject's TRAIN images (held-out excluded). No flips."""
    sp = load_split(subject, processed_dir)
    base = ROOT / processed_dir / subject
    ids = sp["train"]
    images = torch.stack([_img_tensor(base / i / "image.png") for i in ids])         # [N,3,H,W] [0,1]
    depths = torch.stack([_img_tensor(base / i / "depth.png") for i in ids])         # [N,3,H,W] [0,1]
    masks = torch.stack([_img_tensor(base / i / "mask.png")[:1] for i in ids])       # [N,1,H,W] {0,1}
    return {"ids": ids, "images": images, "depths": depths, "masks": masks, "split": sp}


@torch.no_grad()
def encode_latents(vae, images: torch.Tensor, device="cuda") -> torch.Tensor:
    """images [N,3,H,W] in [0,1] -> scaled latents [N,4,H/8,W/8] (posterior mean, fp32)."""
    x = (images * 2 - 1).to(device, vae.dtype)
    z = torch.cat([vae.encode(x[i:i + 1]).latent_dist.mean for i in range(len(x))])
    return (z * vae.config.scaling_factor).float()


def masks_to_latent(masks: torch.Tensor, latent_hw: int) -> torch.Tensor:
    """[N,1,H,W] -> [N,1,h,w] area-interpolated (soft) masks in [0,1]."""
    return F.interpolate(masks.float(), size=(latent_hw, latent_hw), mode="area")
