"""BoothNet training: DreamBooth-LoRA on SD1.5 with a frozen depth ControlNet in the loop (brief §6).

    python train.py --subject dog --variant M1                 # full run -> runs/dog/M1/
    python train.py --subject dog --variant M1 --max_steps 20  # smoke test -> runs/_smoke/dog/M1/

Outputs: checkpoint-<step>/ and final/ (pytorch_lora_weights.safetensors, diffusers format),
config.yaml (fully resolved), train_log.csv, val/step_XXXX.png, val/summary.png, loss.png, summary.json.
"""
from __future__ import annotations

import argparse
import copy
import csv
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from boothnet.data import encode_latents, load_train_set, masks_to_latent  # noqa: E402
from boothnet.degrade import degrade  # noqa: E402
from boothnet.models import add_lora, encode_prompt, load_components, pick_dtype, save_lora  # noqa: E402
from boothnet.utils import grid, load_yaml, save_json, save_yaml  # noqa: E402


# ----------------------------------------------------------------------------- config

def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def resolve_config(args) -> dict:
    cfg = deep_merge(load_yaml("configs/train_base.yaml"), load_yaml(f"configs/variants/{args.variant}.yaml"))
    subj = {s["name"]: s for s in load_yaml("configs/subjects.yaml")["subjects"]}[args.subject]
    cfg["subject"] = {"name": subj["name"], "class": subj["class"], "category": subj["category"]}
    cfg["prompt"] = f"a photo of {cfg['identifier']} {subj['class']}"
    smoke = args.max_steps is not None
    if smoke:
        cfg["max_steps"] = args.max_steps
        cfg["checkpoint_steps"] = []
        cfg["validation"]["every"] = max(1, args.max_steps // 2)
    if args.val_every is not None:
        cfg["validation"]["every"] = args.val_every
    if args.seed is not None:
        cfg["seed"] = args.seed
    cfg["checkpoint_steps"] = [s for s in cfg["checkpoint_steps"] if s <= cfg["max_steps"]]
    run_root = ROOT / cfg["runs_dir"] / ("_smoke" if smoke else "")
    cfg["output_dir"] = str((Path(args.output_dir) if args.output_dir else run_root / args.subject / args.variant))
    if cfg.get("prior_preservation") or args.prior_preservation:
        raise NotImplementedError("prior preservation is reserved for later (brief §6.5); not implemented yet")
    return cfg


# ----------------------------------------------------------------------------- helpers

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_optimizer(params, cfg):
    if cfg["optimizer"] == "adamw8bit":
        try:
            import bitsandbytes as bnb

            return bnb.optim.AdamW8bit(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"]), "AdamW8bit"
        except Exception as e:  # noqa: BLE001
            print(f"bitsandbytes unavailable ({e}); falling back to torch AdamW")
    return torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=cfg["weight_decay"]), "AdamW"


def tier_a_depth(cfg) -> tuple[Image.Image, str]:
    d = sorted((ROOT / cfg["test_bank_dir"] / cfg["subject"]["name"] / "A").iterdir())[0]
    return Image.open(d / "depth.png").convert("RGB"), d.name


@torch.no_grad()
def validate(step, c, cfg, out_dir: Path, depth_a: Image.Image, autocast) -> Image.Image:
    """§6.8: (a) text-only '... on the beach'; (b) ControlNet + Tier A depth. Fixed seed."""
    from diffusers import StableDiffusionControlNetPipeline, StableDiffusionPipeline, UniPCMultistepScheduler

    v = cfg["validation"]
    c["vae"].to("cuda")
    c["text_encoder"].to("cuda")
    common = dict(vae=c["vae"], text_encoder=c["text_encoder"], tokenizer=c["tokenizer"], unet=c["unet"],
                  scheduler=UniPCMultistepScheduler.from_config(c["scheduler"].config), safety_checker=None,
                  feature_extractor=None, requires_safety_checker=False)
    pipe = StableDiffusionPipeline(**common)
    cpipe = StableDiffusionControlNetPipeline(controlnet=c["controlnet"], **common)
    for p in (pipe, cpipe):
        p.set_progress_bar_config(disable=True)
    gen = lambda: torch.Generator("cuda").manual_seed(v["seed"])  # noqa: E731
    with autocast():
        img_a = pipe(f"{cfg['prompt']} {v['text_context']}", num_inference_steps=v["steps"],
                     guidance_scale=v["guidance_scale"], generator=gen()).images[0]
        img_b = cpipe(cfg["prompt"], image=depth_a, num_inference_steps=v["steps"],
                      guidance_scale=v["guidance_scale"], controlnet_conditioning_scale=v["controlnet_scale"],
                      generator=gen()).images[0]
    c["vae"].to("cpu")
    c["text_encoder"].to("cpu")
    torch.cuda.empty_cache()
    row = [img_a, img_b, depth_a]
    (out_dir / "val").mkdir(parents=True, exist_ok=True)
    grid([row], [f"step {step}"], cell=256,
         col_titles=[f"text-only: ...{v['text_context']}", f"ControlNet {v['controlnet_scale']} + Tier A", "Tier A depth"]
         ).save(out_dir / "val" / f"step_{step:04d}.png")
    return row


def plot_loss(log_path: Path, out_path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(log_path)
    per_step = df.groupby("step")["loss"].mean()
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(per_step.index, per_step.values, lw=0.5, alpha=0.35, color="tab:gray", label="per step")
    ax.plot(per_step.index, per_step.rolling(50, min_periods=1).mean(), lw=1.8, color="tab:blue", label="mean (50 steps)")
    if df["control_on"].nunique() > 1:
        for on, col in [(1, "tab:orange"), (0, "tab:green")]:
            s = df[df.control_on == on].groupby("step")["loss"].mean()
            ax.plot(s.index, s.rolling(50, min_periods=1).mean(), lw=1.2, color=col,
                    label=f"control {'on' if on else 'off'} (50-step mean)")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("loss (eps MSE)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", required=True)
    ap.add_argument("--variant", required=True, choices=["M0", "M1", "M2", "M3"])
    ap.add_argument("--max_steps", type=int, default=None, help="smoke-test mode (writes to runs/_smoke/)")
    ap.add_argument("--val_every", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--prior_preservation", action="store_true")
    args = ap.parse_args()

    cfg = resolve_config(args)
    out_dir = Path(cfg["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(cfg, out_dir / "config.yaml")
    seed_everything(cfg["seed"])
    rng = random.Random(cfg["seed"])                                  # coin / scale / degrade / index
    tgen = torch.Generator("cuda").manual_seed(cfg["seed"])           # noise / timesteps
    dtype = pick_dtype(cfg["precision"])
    autocast = lambda: torch.autocast("cuda", dtype=dtype)  # noqa: E731
    torch.cuda.reset_peak_memory_stats()
    t_start = time.time()

    # --- models + cached inputs (§6.6)
    c = load_components(cfg["pretrained_model"], cfg["controlnet_model"], dtype)
    sched = c["scheduler"]
    assert sched.config.prediction_type == "epsilon", sched.config.prediction_type
    for k in ("vae", "text_encoder", "unet", "controlnet"):
        c[k].to("cuda")
    data = load_train_set(cfg["subject"]["name"], cfg["processed_dir"])
    z0 = encode_latents(c["vae"], data["images"])                               # [N,4,64,64] fp32
    emb = encode_prompt(c["tokenizer"], c["text_encoder"], cfg["prompt"])        # [1,77,768]
    depths = data["depths"].to("cuda")                                           # [N,3,512,512] [0,1]
    mask_lat = masks_to_latent(data["masks"], z0.shape[-1]).to("cuda")          # [N,1,64,64]
    c["vae"].to("cpu")
    c["text_encoder"].to("cpu")
    torch.cuda.empty_cache()
    n_train = len(data["ids"])
    print(f"subject={cfg['subject']['name']} variant={cfg['variant']} prompt='{cfg['prompt']}' "
          f"train={data['ids']} dtype={dtype}")

    unet, cn = c["unet"], c["controlnet"]
    params = add_lora(unet, cfg["lora"]["rank"], cfg["lora"]["alpha"], cfg["lora"]["target_modules"],
                      cfg["lora"]["init"])
    if cfg["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
    unet.train()
    optimizer, opt_name = make_optimizer(params, cfg)
    from diffusers.optimization import get_scheduler

    lr_sched = get_scheduler(cfg["lr_scheduler"], optimizer=optimizer, num_warmup_steps=cfg["lr_warmup_steps"],
                             num_training_steps=cfg["max_steps"])
    n_params = sum(p.numel() for p in params)
    print(f"LoRA params: {n_params / 1e6:.2f}M  optimizer={opt_name}")

    depth_a, case_a = tier_a_depth(cfg)
    val_rows, val_labels = [], []
    if cfg["validation"]["every"]:
        val_rows.append(validate(0, c, cfg, out_dir, depth_a, autocast))
        val_labels.append("step 0")

    # --- training loop (§6.3)
    B, accum = cfg["batch_size"], cfg["grad_accum"]
    log_f = open(out_dir / "train_log.csv", "w", newline="", encoding="utf-8")
    log = csv.writer(log_f)
    log.writerow(["step", "micro", "sample", "img", "t", "loss", "control_on", "scale", "lr",
                  "deg_sigma", "deg_factor", "deg_tx", "deg_ty", "deg_scale", "deg_rot"])
    from tqdm import tqdm

    pbar = tqdm(range(1, cfg["max_steps"] + 1), desc=f"{cfg['subject']['name']}/{cfg['variant']}", dynamic_ncols=True)
    for step in pbar:
        step_loss = 0.0
        for micro in range(accum):
            idx = [rng.randrange(n_train) for _ in range(B)]
            z = z0[idx]
            noise = torch.randn(z.shape, generator=tgen, device="cuda")
            t = torch.randint(0, sched.config.num_train_timesteps, (B,), generator=tgen, device="cuda")
            zt = sched.add_noise(z, noise, t)
            # per-sample control coin, strength and degradation (tricks 1-3)
            on = [rng.random() < cfg["p_control"] for _ in range(B)]
            scales = [rng.uniform(cfg["scale_min"], cfg["scale_max"]) if o else 0.0 for o in on]
            dparams = [{} for _ in range(B)]
            down_res = mid_res = None
            if any(on):
                conds = []
                for b in range(B):
                    d, dparams[b] = degrade(depths[idx[b]], cfg["degrade"], rng) if on[b] else (depths[idx[b]], {})
                    conds.append(d)
                with torch.no_grad(), autocast():
                    # residuals are linear in conditioning_scale -> run at 1.0, scale per sample
                    # (identical to conditioning_scale=s for B=1; s=0 for samples with control off)
                    down_res, mid_res = cn(zt.to(dtype), t, encoder_hidden_states=emb.expand(B, -1, -1),
                                           controlnet_cond=torch.stack(conds).to(dtype), conditioning_scale=1.0,
                                           return_dict=False)
                s = torch.tensor(scales, device="cuda", dtype=dtype).view(B, 1, 1, 1)
                down_res = [(r * s).to(dtype) for r in down_res]
                mid_res = (mid_res * s).to(dtype)
            with autocast():
                pred = unet(zt.to(dtype), t, encoder_hidden_states=emb.expand(B, -1, -1),
                            down_block_additional_residuals=down_res, mid_block_additional_residual=mid_res).sample
            err = (pred.float() - noise.float()) ** 2                                    # [B,4,64,64]
            if cfg["masked_loss"]:                                                       # trick 4
                m = mask_lat[idx]
                w = m + (1 - m) * cfg["bg_weight"]
            else:
                w = torch.ones_like(err[:, :1])
            per_sample = (w * err).mean(dim=(1, 2, 3))
            loss = per_sample.mean()
            (loss / accum).backward()
            step_loss += loss.item() / accum
            lr_now = lr_sched.get_last_lr()[0]
            for b in range(B):
                dp = dparams[b]
                log.writerow([step, micro, b, data["ids"][idx[b]], int(t[b]), f"{per_sample[b].item():.6f}", int(on[b]),
                              f"{scales[b]:.4f}", f"{lr_now:.3e}", dp.get("sigma", ""), dp.get("factor", ""),
                              dp.get("tx", ""), dp.get("ty", ""), dp.get("scale", ""), dp.get("rot", "")])
        grad_norm = torch.nn.utils.clip_grad_norm_(params, cfg["max_grad_norm"])
        optimizer.step()
        lr_sched.step()
        optimizer.zero_grad(set_to_none=True)
        pbar.set_postfix(loss=f"{step_loss:.4f}", gnorm=f"{float(grad_norm):.3f}",
                         vram=f"{torch.cuda.max_memory_allocated() / 2**30:.1f}G")
        if not math.isfinite(step_loss):
            raise RuntimeError(f"non-finite loss at step {step}")
        if step in cfg["checkpoint_steps"]:
            save_lora(unet, out_dir / f"checkpoint-{step}")
            log_f.flush()
        if cfg["validation"]["every"] and step % cfg["validation"]["every"] == 0:
            val_rows.append(validate(step, c, cfg, out_dir, depth_a, autocast))
            val_labels.append(f"step {step}")
            unet.train()
    log_f.close()

    save_lora(unet, out_dir / "final")
    peak = torch.cuda.max_memory_allocated() / 2**30
    elapsed = time.time() - t_start
    if val_rows:
        grid(val_rows, val_labels, cell=224, col_titles=["text-only (beach)", "ControlNet + Tier A", "Tier A depth"]
             ).save(out_dir / "val" / "summary.png")
    plot_loss(out_dir / "train_log.csv", out_dir / "loss.png", f"{cfg['subject']['name']} / {cfg['variant']}")
    summary = {"subject": cfg["subject"]["name"], "variant": cfg["variant"], "steps": cfg["max_steps"],
               "peak_vram_gib": round(peak, 2), "minutes": round(elapsed / 60, 2), "optimizer": opt_name,
               "lora_params_M": round(n_params / 1e6, 3), "dtype": str(dtype), "tier_a_case": case_a,
               "train_ids": data["ids"]}
    save_json(summary, out_dir / "summary.json")
    print(f"done: {summary}")


if __name__ == "__main__":
    main()
