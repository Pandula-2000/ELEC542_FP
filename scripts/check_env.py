"""Phase 0 acceptance check.

Loads SD1.5 + depth ControlNet in fp16, estimates the depth of one test image,
generates one 512x512 image from that depth map, saves everything and prints
peak VRAM. Also prints library versions (copy them into NOTES.md).

    python scripts/check_env.py [--image dataset/Subjects/dog/00.jpg] [--steps 30]
"""
from __future__ import annotations

import argparse
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from boothnet.preprocess import DepthEstimator, center_crop_resize, depth_to_pil  # noqa: E402

SD_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CN_ID = "lllyasviel/control_v11f1p_sd15_depth"


def print_versions() -> None:
    import importlib

    print(f"python        {platform.python_version()}  ({platform.platform()})")
    for name in ["torch", "torchvision", "diffusers", "transformers", "peft", "accelerate",
                 "safetensors", "cv2", "PIL", "numpy", "pandas", "matplotlib", "yaml", "rembg",
                 "onnxruntime", "bitsandbytes"]:
        try:
            mod = importlib.import_module(name)
            print(f"{name:13s} {getattr(mod, '__version__', '?')}")
        except Exception as e:  # optional packages may be missing
            print(f"{name:13s} NOT AVAILABLE ({type(e).__name__}: {e})")
    print(f"cuda avail    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"device        {torch.cuda.get_device_name(0)}  cc={torch.cuda.get_device_capability(0)}")
        print(f"torch cuda    {torch.version.cuda}  bf16={torch.cuda.is_bf16_supported()}")
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        print(f"VRAM total    {total:.2f} GiB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", type=Path, default=None,
                    help="any photo; default: first image of dataset/Subjects/dog or data/processed/dog")
    ap.add_argument("--prompt", default="a photo of a dog")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "check_env")
    args = ap.parse_args()
    if args.image is None:
        cands = sorted((ROOT / "dataset" / "Subjects" / "dog").glob("*.jpg")) + \
            sorted((ROOT / "data" / "processed" / "dog").glob("*/image.png"))
        assert cands, "no default image found; pass --image <path>"
        args.image = cands[0]

    print_versions()
    assert torch.cuda.is_available(), "CUDA not available - wrong torch build?"
    args.out.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats()

    from PIL import Image

    img = center_crop_resize(Image.open(args.image), 512)
    img.save(args.out / "input.png")

    t0 = time.time()
    de = DepthEstimator()
    depth = de(img)
    depth_pil = depth_to_pil(depth)
    depth_pil.save(args.out / "depth.png")
    del de
    torch.cuda.empty_cache()
    print(f"depth done in {time.time() - t0:.1f}s; "
          f"mean brightness center={depth[192:320, 192:320].mean():.2f} border={depth[:32].mean():.2f}")

    from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler

    t0 = time.time()
    controlnet = ControlNetModel.from_pretrained(CN_ID, dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        SD_ID, controlnet=controlnet, dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.set_progress_bar_config(disable=True)
    print(f"pipeline loaded in {time.time() - t0:.1f}s; "
          f"scheduler prediction_type={pipe.scheduler.config.prediction_type}")

    t0 = time.time()
    out = pipe(args.prompt, image=depth_pil, num_inference_steps=args.steps, guidance_scale=7.5,
               controlnet_conditioning_scale=0.8, control_guidance_end=1.0,
               generator=torch.Generator("cuda").manual_seed(args.seed)).images[0]
    dt = time.time() - t0
    out.save(args.out / "generated.png")

    # side-by-side sheet: input | depth | generated
    sheet = Image.new("RGB", (512 * 3, 512))
    for k, im in enumerate([img, depth_pil, out]):
        sheet.paste(im, (512 * k, 0))
    sheet.save(args.out / "sheet.png")

    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"generation    {dt:.1f}s for {args.steps} steps ({out.size[0]}x{out.size[1]})")
    print(f"peak VRAM     {peak:.2f} GiB (torch.cuda.max_memory_allocated)")
    print(f"saved to      {args.out}")


if __name__ == "__main__":
    main()
