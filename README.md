# BoothNet — ELEC 542 Final Project

Subject personalization (DreamBooth-LoRA on Stable Diffusion 1.5) combined with layout control (a frozen, pretrained
depth ControlNet). The new part: the frozen ControlNet is active **inside the LoRA training loop**, fed each training
photo's own (degraded) depth map with random control dropout, so the LoRA learns the subject's *appearance* rather
than memorizing its training poses and backgrounds.

- Full specification: [PROJECT_BRIEF.md](PROJECT_BRIEF.md)
- Running log (environment, decisions, problems, results): [NOTES.md](NOTES.md)

**Status:** Phases 0–2 implemented (environment, data + test bank, training). Inference (`generate.py`) and
evaluation (`evaluate.py`, `scripts/aggregate.py`) are not written yet.

---

## 1. Setup (Windows, NVIDIA RTX 50-series)

Tested on native Windows 11 (not WSL) with an RTX 5070 Ti Laptop GPU (12 GB). Blackwell GPUs need PyTorch built for
CUDA 12.8 (`cu128`); `requirements.txt` already points pip at that wheel index.

```powershell
conda create -y -p X:\Environments\BoothNet python=3.11   # any path/name works; Python 3.11 recommended
conda activate X:\Environments\BoothNet
python -m pip install -r requirements.txt
python scripts\check_env.py                               # Phase 0 check, see below
```

Notes:
- Use Python 3.11. Python 3.14 is too new for `onnxruntime`/`rembg`.
- Keep `onnxruntime-gpu==1.23.2`. Newer builds target CUDA 13 and silently fall back to the CPU (very slow masking).
- `bitsandbytes` is optional. Training falls back to standard AdamW if it is missing.
- Models download from the Hugging Face Hub on first use (several GB, cached in `%USERPROFILE%\.cache\huggingface`).
  The "symlinks not supported" warning on Windows is harmless.
- In VS Code, select the interpreter `X:\Environments\BoothNet\python.exe` (already set in `.vscode/settings.json`).

`scripts/check_env.py` prints library versions, estimates depth for one dataset image, generates one 512×512 image with
SD1.5 + the depth ControlNet, and reports peak VRAM. Output: `outputs/check_env/sheet.png`.

## 2. Data

The dataset is **not in git** (the images have their own licenses) and is transferred separately. Place it as follows
(treated as read-only; nothing under `dataset/` is ever modified):

```
dataset/
├── Subjects/       # DreamBench (DreamBooth dataset): one folder per subject + prompts_and_classes.txt
└── Backgrounds/    # scene images used as layout sources for Tier C
```

All derived data goes to `data/` (git-ignored).

## 3. Pipeline

Run every command from the repository root with the BoothNet environment active.

### Phase 1: preprocess subjects and build the test bank

```powershell
python scripts\prepare_data.py                 # crops, depth maps, masks for the 6 subjects (~2 min)
python scripts\build_test_bank.py              # Tier A (held-out photo) + Tier B (other instances)
python scripts\build_test_bank.py --tiers C    # Tier C (background scenes); needs confirmed: true in configs/backgrounds.yaml
```

| Output | Contents |
|---|---|
| `data/processed/<subject>/<img_id>/{image,depth,mask}.png`, `meta.json` | 512×512 image, depth (near = bright), binary mask |
| `data/processed/<subject>/split.json`, `preview.png` | train / held-out split; contact sheet to check by eye |
| `data/test_bank/<subject>/<tier>/<case_id>/{depth,source,source_mask}.png`, `case.json` | test cases for tiers `A`, `B`, `C`, `C_bg` |
| `data/test_bank/<subject>/C_preview.png` | Tier C composites (background, pasted mask, depth) |

Useful flags: `--subjects dog cat2` (subset), `--overwrite` (prepare_data), `--preview_c` (render Tier C placement to
`data/tier_c_preview/` without writing the bank).

Configs:
- `configs/subjects.yaml`: subjects, class nouns, categories, masker settings.
- `configs/test_bank.yaml`: Tier B sources.
- `configs/backgrounds.yaml`: per-background prompt context, composite or scene type, and placement.

### Phase 2: training

```powershell
python train.py --subject dog --variant M1 --max_steps 20   # smoke test (~2 min) -> runs\_smoke\dog\M1\
python train.py --subject dog --variant M1                  # full run, 1000 steps (~26 min, ~3.6 GB VRAM)
python scripts\train_all.py                                 # all subjects x M0-M3; skips runs whose final\ exists
python scripts\train_all.py --dry_run                       # list what would run
python scripts\plot_losses.py --subject dog --variants M0 M1
```

Variants (`configs/variants/*.yaml`, on top of `configs/train_base.yaml`):

| Variant | ControlNet in training | Degraded depth | Masked loss | Role |
|---|---|---|---|---|
| M0 | never | – | no | naive baseline (standard DreamBooth-LoRA) |
| M1 | 50% of steps | yes | no | proposed |
| M2 | 50% of steps | yes | yes (background weight 0.2) | proposed + background disentanglement |
| M3 | always | no | no | ablation |

Each run writes `runs/<subject>/<variant>/`:
- `checkpoint-400/600/800/1000/` and `final/` (LoRA in diffusers format)
- `config.yaml` (fully resolved)
- `train_log.csv` (per micro-step: loss, timestep, control on/off, scale, degradation parameters)
- `val/step_XXXX.png` and `val/summary.png`
- `loss.png`
- `summary.json` (time, peak VRAM)

`train_all.py` writes logs to `runs/logs/`.

Loading a trained LoRA for inference:

```python
import torch
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, UniPCMultistepScheduler
from PIL import Image

cn = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=cn, dtype=torch.float16, safety_checker=None).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("runs/dog/M1/checkpoint-400")
depth = Image.open("data/test_bank/dog/C/city_roads/depth.png")
img = pipe("a photo of sks dog on the street of a mountain town", image=depth, num_inference_steps=30,
           guidance_scale=7.5, controlnet_conditioning_scale=0.8,
           generator=torch.Generator("cuda").manual_seed(0)).images[0]
pipe.unload_lora_weights()   # before loading another LoRA
```

### Phases 3–4: inference and evaluation (not implemented yet)

Planned: `python generate.py --config configs/eval_full.yaml` → `outputs/`, then `python evaluate.py` and
`python scripts/aggregate.py` → `results/`. See PROJECT_BRIEF.md §7–8.

## 4. Repository layout

```
boothnet/            library code
  preprocess.py      cropping, Depth-Anything-V2 depth, SubjectMasker (Grounding-DINO + SAM / BiRefNet)
  degrade.py         depth-map degradation (blur, resample, small affine)
  data.py            training data loading, latent caching
  models.py          SD1.5 / ControlNet / LoRA loading and saving
  utils.py           config I/O, contact sheets
scripts/             check_env, prepare_data, build_test_bank, train_all, plot_losses
configs/             subjects, backgrounds, test_bank, train_base, variants/M0-M3
train.py             training entry point
dataset/             provided data (read-only, not in git)
data/ runs/ outputs/ derived files (git-ignored)
```
