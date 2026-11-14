"""Model loading (SD1.5, depth ControlNet, LoRA) and LoRA saving in diffusers format."""
from __future__ import annotations

from pathlib import Path

import torch

SD_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CN_ID = "lllyasviel/control_v11f1p_sd15_depth"


def pick_dtype(name: str) -> torch.dtype:
    if name == "bf16" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def load_components(sd_id: str = SD_ID, cn_id: str | None = CN_ID, dtype=torch.bfloat16):
    """Frozen SD1.5 components (+ optional ControlNet), all requires_grad=False."""
    from diffusers import AutoencoderKL, ControlNetModel, DDPMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(sd_id, subfolder="tokenizer")
    te = CLIPTextModel.from_pretrained(sd_id, subfolder="text_encoder", dtype=dtype)
    vae = AutoencoderKL.from_pretrained(sd_id, subfolder="vae", dtype=dtype)
    unet = UNet2DConditionModel.from_pretrained(sd_id, subfolder="unet", dtype=dtype)
    sched = DDPMScheduler.from_pretrained(sd_id, subfolder="scheduler")
    cn = ControlNetModel.from_pretrained(cn_id, dtype=dtype) if cn_id else None
    for m in [te, vae, unet, cn]:
        if m is not None:
            m.requires_grad_(False)
            m.eval()
    return {"tokenizer": tok, "text_encoder": te, "vae": vae, "unet": unet, "scheduler": sched, "controlnet": cn}


def add_lora(unet, rank: int, alpha: int, target_modules: list[str], init: str = "gaussian") -> list[torch.nn.Parameter]:
    """Add a LoRA adapter (peft) to the UNet; LoRA params upcast to fp32. Returns trainable params."""
    from diffusers.training_utils import cast_training_params
    from peft import LoraConfig

    cfg = LoraConfig(r=rank, lora_alpha=alpha, init_lora_weights=init, target_modules=target_modules)
    unet.add_adapter(cfg)
    cast_training_params(unet, dtype=torch.float32)
    params = [p for p in unet.parameters() if p.requires_grad]
    assert params and all("lora" in n for n, p in unet.named_parameters() if p.requires_grad)
    return params


def save_lora(unet, out_dir: Path) -> Path:
    """Save UNet LoRA weights so that `pipe.load_lora_weights(out_dir)` works."""
    from diffusers import StableDiffusionPipeline
    from diffusers.utils import convert_state_dict_to_diffusers
    from peft.utils import get_peft_model_state_dict

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sd = convert_state_dict_to_diffusers(get_peft_model_state_dict(unet))
    StableDiffusionPipeline.save_lora_weights(str(out_dir), unet_lora_layers=sd, safe_serialization=True)
    return out_dir / "pytorch_lora_weights.safetensors"


@torch.no_grad()
def encode_prompt(tokenizer, text_encoder, prompt: str, device="cuda") -> torch.Tensor:
    ids = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True,
                    return_tensors="pt").input_ids.to(device)
    return text_encoder(ids)[0]  # [1, 77, 768]
