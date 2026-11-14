# BoothNet — Project Brief for the Coding Agent

> **Read this whole file before writing any code.** It describes the research task, the method, the exact pipeline, the experiments, and the deliverables. The attached papers (see §2) are the background; this file is the specification. When this file and a paper disagree, follow this file.

---

## 0. TL;DR

We are building **BoothNet**: subject personalization (DreamBooth-LoRA on Stable Diffusion 1.5) combined with layout control (a **frozen, pretrained depth ControlNet**). The **novel part** is that the frozen ControlNet is active **inside the LoRA training loop**, fed each training photo's own (degraded) depth map, with random **control dropout**. The hypothesis: because pose/shape/layout are "explained away" by the depth hints, the LoRA learns the subject's **appearance** instead of memorizing its training poses and backgrounds. At inference, a depth map from **any new scene** places the personalized subject in a new pose/layout.

We compare against the **naive combination** (standard DreamBooth-LoRA trained without ControlNet, with ControlNet attached only at inference) and other baselines, on a subset of the DreamBench dataset.

This is a short course project (Generative AI for Image Data). Favor **simple, correct, reproducible code** over generality. Get one subject working end-to-end before scaling up.

---

## 1. Working rules for the agent

1. **Work in phases (§10).** At the end of each phase: run its smoke test, summarize what was built and what was observed, and **stop for the user's review** before starting the next phase.
2. **Smoke-test before long runs.** Every training/eval script must support a tiny mode (e.g. `--max_steps 20`, 1 subject, 1 prompt) that finishes in a couple of minutes. Run it before any full run.
3. **Hardware limit: single laptop GPU, NVIDIA RTX 5070 Ti, 12 GB VRAM.** Everything must fit. Use the memory savers in §6.6. If something OOMs, reduce memory — do not silently change the method.
4. **Do not change the method or hyperparameter defaults** in this brief without asking. If you think something should change, propose it with a reason.
5. **Keep a log.** Maintain `NOTES.md` in the repo root: environment versions, decisions made, problems hit and fixes, and a short result summary per run.
6. **Everything config-driven and seeded.** Every run writes its full resolved config (YAML/JSON) next to its outputs. Fixed seeds everywhere.
7. **Check library APIs against the installed versions** (diffusers, transformers, peft APIs change between versions). Print versions into `NOTES.md` at setup.
8. **The datasets are already on disk and are read-only.** Subjects: `./dataset/Subjects`, inference backgrounds: `./dataset/Backgrounds` (§5). Never modify, move or delete files there; write all derived data under `data/`. Use `pathlib` for all paths (the user is on Windows).
9. **Use plain PyTorch + diffusers + peft.** A single-GPU custom training loop is preferred over `accelerate launch` complexity. You may adapt code from diffusers' `examples/dreambooth/train_dreambooth_lora.py`, but our training step is custom (§6.3).

---

## 2. Background papers (attached as PDFs)

| Paper | What to take from it |
|---|---|
| **DreamBooth** (Ruiz et al., 2023) | Rare-token identifier + class noun prompt ("a photo of sks dog"); few-shot fine-tuning objective; optional prior-preservation loss; DreamBench dataset and prompts; DINO / CLIP-I / CLIP-T metrics (§4.1). Failure modes we target: pose/background overfitting (Fig. 6, Fig. 9c). |
| **ControlNet** (Zhang et al., 2023) | Frozen SD + trainable copy connected by zero convs; outputs are **residuals added to the UNet's 12 skip connections + middle block** (Fig. 3). We use a *pretrained* ControlNet and keep it frozen. Note their 50% prompt dropout (§3.3) — the inspiration for our control dropout. Fig. 12 shows ControlNets transfer to fine-tuned SD models — this is why our naive baseline will be competitive. |
| **LoRA** (Hu et al., 2021) | Low-rank adapters on attention projections; the only trainable parameters in our setup. |
| **BLIP-Diffusion** (Li et al., 2023) | Closest related work. Composes a subject-driven model with ControlNet **at inference only** (Fig. 4). Uses random-background compositing to stop background leaking into the subject representation, and 15% subject-embedding dropout. Freezing the text encoder during per-subject fine-tuning reduced overfitting. Our difference: control is **inside per-subject training**. |
| **Project proposal** (BoothNet, ELEC 542) | The original plan. This brief refines it (depth-only, explicit dropout/degradation, masked loss, test-tier protocol). |

---

## 3. Research question and hypotheses

**Question:** Does training the DreamBooth-LoRA *with* a frozen depth ControlNet in the loop produce a LoRA that (a) follows new layouts better and (b) keeps subject identity at least as well, compared to training without it (naive combination)?

**Hypotheses:**
- H1: Proposed (M1/M2) gives better layout fidelity than the naive combo (M0+ControlNet) at equal identity, especially when the test layout is far from the training poses.
- H2: Control dropout keeps text-only generation (no ControlNet) close to M0 quality; training with control 100% of the time (M3) hurts text-only generation.
- H3: The benefit is larger for **live subjects** (pose = layout) than **rigid objects** (silhouette = identity). Report the two categories separately.

---

## 4. Environment setup (Phase 0)

- GPU is Blackwell (RTX 50-series): **install a PyTorch build with CUDA 12.8+ wheels (cu128)**. Older builds will not run on this GPU. Verify with `torch.cuda.is_available()` and `torch.cuda.get_device_name()`.
- The user works on **Windows** in VS Code (confirm, and note whether you are in native Windows or WSL in `NOTES.md`). Use `pathlib`, avoid shell-specific scripts (provide Python entry points, optionally `.ps1` helpers), and treat bitsandbytes as optional — fall back to standard AdamW if it fails to import or run.
- `requirements.txt` (pin the versions you actually install):
  `torch torchvision (cu128)`, `diffusers`, `transformers`, `peft`, `accelerate`, `safetensors`, `opencv-python`, `pillow`, `numpy`, `pandas`, `matplotlib`, `pyyaml`, `tqdm`, `rembg` (or BiRefNet via transformers), optional `bitsandbytes`.
- Use PyTorch SDPA attention (default in recent diffusers); do not require xformers.

**Models (Hugging Face IDs — verify they resolve):**
| Role | ID |
|---|---|
| Base model | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| Depth ControlNet | `lllyasviel/control_v11f1p_sd15_depth` |
| Depth estimator | `depth-anything/Depth-Anything-V2-Small-hf` |
| DINO (metric) | `facebook/dino-vits16` |
| CLIP (metrics) | `openai/clip-vit-base-patch32` |
| Optional baseline | `Salesforce/blipdiffusion-controlnet` (diffusers `BlipDiffusionControlNetPipeline`) |

**Phase 0 acceptance:** a script `scripts/check_env.py` that loads SD1.5 + depth ControlNet in fp16, generates one 512×512 image from a depth map of any test image, saves it, and prints peak VRAM.

---

## 5. Dataset (Phase 1)

### 5.0 Data already on disk (do not download)

| Path | Contents |
|---|---|
| `./dataset/Subjects/` | The DreamBooth dataset (DreamBench), one folder per subject (e.g. `dog`, `dog6`, `cat2`, `backpack`, …) |
| `./dataset/Backgrounds/` | Scene images to use as **layout sources at inference** (§5.3, Tier C) |

**First step of Phase 1:** list both folders and report in `NOTES.md`: subject folder names and image counts; number, resolution and a brief description of the background images, and **whether the backgrounds already contain a subject (e.g. an animal or object) or are empty scenes** — this decides how Tier C is built (§5.3). Show the user a contact sheet of the backgrounds and **stop for confirmation** before building Tier C.

If a DreamBench prompt/class file (e.g. `prompts_and_classes.txt`) exists anywhere under `./dataset`, use it for class nouns; otherwise use the class nouns in the table below.

### 5.1 Subjects
Read subject images from `./dataset/Subjects/<subject>/`.

Use **6 subjects** (configurable list in `configs/subjects.yaml`):

| Subject folder | Class noun | Category |
|---|---|---|
| `dog` | dog | live |
| `dog6` | dog | live |
| `cat2` | cat | live |
| `backpack` | backpack | object |
| `teapot` | teapot | object |
| `rc_car` | toy | object |

(Confirm these folders exist under `./dataset/Subjects`; check class nouns against the dataset's class file if present.)

**Split:** per subject, hold out **1 image** (`heldout`), train on the rest (3–5 images). Store the split in `data/processed/<subject>/split.json`. Choose the held-out image deterministically (e.g. the last file in sorted order).

### 5.2 Preprocessing — `scripts/prepare_data.py`
For every image (train and held-out):
1. Center-crop to square, resize to **512×512** RGB. Save `image.png`.
2. **Depth map** with Depth-Anything-V2-Small: convert to grayscale, normalize to [0, 1] per image, **near = bright** (the convention the SD1.5 depth ControlNet expects — verify visually against a ControlNet example), resize to 512×512, save as 3-channel `depth.png`.
3. **Subject mask** with rembg (or BiRefNet): binary {0,1}, 512×512, save `mask.png`. Visually spot-check masks; fix thresholds if needed.
4. Write `data/processed/<subject>/<img_id>/{image,depth,mask}.png` and a `meta.json` (subject, class noun, category, split).

Also save a **contact sheet** per subject (image | depth | mask) to `data/processed/<subject>/preview.png` for quick visual checking.

**No horizontal flips** anywhere: subjects have asymmetric features (e.g. a one-sided face streak, a number on a clock).

### 5.3 Test control bank — `scripts/build_test_bank.py`
Each test case = `(prompt_context, depth_map, source_image, source_mask, tier)`.

- **Tier A — same subject, unseen photo:** depth of the subject's held-out image. Prompt context: none (prompt = `"a photo of sks <class>"`).
- **Tier B — same class, different instance:** depth maps from photos of *other* instances of the same class in `./dataset/Subjects` (e.g. other dog folders for dogs; `cat` for `cat2`; `backpack_dog` for `backpack`). If no other instance exists (e.g. teapot, rc_car), generate 3 images with **base SD1.5 (no LoRA, no ControlNet)** from `"a photo of a <class>"` with fixed seeds and use their depth. 3 maps per subject. Prompt context: none.
- **Tier C — new backgrounds, from `./dataset/Backgrounds`:** one test case per background image (use all of them, or a fixed subset of ≤10 if there are many; record which). How the depth map is built depends on what the user confirmed in §5.0:
  - **C-scene (backgrounds already contain a subject of a matching class):** depth of the background image directly; source mask = segmented subject in it. Use only backgrounds whose subject class matches the test subject.
  - **C-composite (backgrounds are empty scenes — expected case):** build a **composite depth map** = background depth + a subject-shaped depth region pasted in. Take the subject geometry from a pose source (the subject's Tier A map, or a Tier B map), cut it out with its mask, scale it to a configured size (default: subject height ≈ 45% of image height), place it at a configured position (default: bottom-center, resting on the ground), and set its depth values so the subject is **nearer than the background behind it** (re-scale the cutout's depth range into [bg_depth_at_feet + δ, 1]). The pasted mask becomes the case's `source_mask` (used for Mask IoU). Save the composite and a visual overlay for checking.
  - Also save a **C-bg-only** variant (background depth with no pasted subject) for a small exploratory run on 2 subjects — it tests whether the model inserts the subject sensibly when only scene geometry is given. Not part of the main tables.
  - **Prompt context per background:** create `configs/backgrounds.yaml` mapping each background file → a short context phrase (e.g. `"on the beach"`, `"in a living room"`), written by looking at the images; **ask the user to confirm** this file. Prompt = `"a photo of sks <class> " + context`.
  - Placement parameters (size, position, δ) live in `configs/backgrounds.yaml` with per-background overrides, so obviously bad placements (subject floating in the sky) can be fixed by hand.
- Do **not** use DreamBench property-modification prompts ("a red sks backpack", "cube shaped") or accessorization prompts ("wearing a santa hat") — they conflict with fixed geometry/identity.

Save the bank as `data/test_bank/<subject>/<tier>/<case_id>/{depth.png, source.png, source_mask.png, case.json}`.

**Phase 1 acceptance:** previews for all 6 subjects look correct (depth near = bright, masks tight); `configs/backgrounds.yaml` confirmed by the user; test bank has Tier A (1), Tier B (≈3) and Tier C (one per selected background) cases per subject, with composite overlays checked visually; summary table in `NOTES.md`.

---

## 6. Model and training (Phase 2)

### 6.1 Components

| Component | Status |
|---|---|
| VAE (SD1.5) | frozen |
| CLIP text encoder (SD1.5) | frozen (BLIP-Diffusion: freezing reduces overfitting) |
| UNet (SD1.5) | frozen base weights |
| **LoRA on UNet attention** | **trainable — the only trainable parameters** |
| Depth ControlNet (`control_v11f1p_sd15_depth`) | frozen, used during training (variants M1–M3) and inference |

LoRA (peft `LoraConfig`): `r=16`, `lora_alpha=16`, `init_lora_weights="gaussian"`, `target_modules=["to_q","to_k","to_v","to_out.0"]`. Add with `unet.add_adapter(...)`.

Identifier prompt for training: **`"a photo of sks <class>"`** (e.g. `"a photo of sks dog"`). The identifier token is configurable.

### 6.2 Intuition (why this should work)
A plain DreamBooth-LoRA, trained on 3–5 photos, must explain *everything* in those photos with the "sks" token — appearance **and** pose, viewpoint, background. So it memorizes poses/backgrounds (DreamBooth Fig. 6, 9c). If, during training, a frozen ControlNet supplies the photo's **geometry** (depth), the UNet already "knows" where things are, so the LoRA only has to supply what depth cannot: **color, texture, markings** — the subject's appearance. At test time, any depth map supplies new geometry and the LoRA paints the subject into it.

The four tricks (§6.4) exist to stop this from going wrong.

### 6.3 One training step (the core — implement exactly)

```python
# Precomputed once (frozen, no grad): for each train image
#   z0[i]   = vae.encode(image).latent_dist.mean * vae.config.scaling_factor   # [4,64,64]
#   emb     = text_encoder(tokenize("a photo of sks dog"))                     # [77,768], fixed prompt
#   depth[i]: [3,512,512] in [0,1];   mask_lat[i]: mask downsampled to 64x64 (area interp), in [0,1]

i     = random.randrange(n_train)
z0_i  = z0[i]
noise = torch.randn_like(z0_i)
t     = torch.randint(0, scheduler.config.num_train_timesteps, (1,))
zt    = scheduler.add_noise(z0_i, noise, t)                 # DDPMScheduler from the SD1.5 repo

down_res, mid_res = None, None
if random.random() < cfg.p_control:                          # Trick 1: control dropout
    d = degrade(depth[i], cfg.degrade)                       # Trick 2: degrade the map
    s = random.uniform(cfg.scale_min, cfg.scale_max)         # Trick 3: random strength
    with torch.no_grad():                                    # ControlNet frozen; its inputs don't
        down_res, mid_res = controlnet(                      # depend on trainable params
            zt, t, encoder_hidden_states=emb,
            controlnet_cond=d, conditioning_scale=s, return_dict=False)

pred = unet(zt, t, encoder_hidden_states=emb,
            down_block_additional_residuals=None if down_res is None else [r.to(zt.dtype) for r in down_res],
            mid_block_additional_residual=None if mid_res is None else mid_res.to(zt.dtype)).sample

if cfg.masked_loss:                                          # Trick 4 (optional)
    w = mask_lat[i] + (1 - mask_lat[i]) * cfg.bg_weight      # broadcast over 4 channels
else:
    w = 1.0
loss = (w * (pred.float() - noise.float()) ** 2).mean()      # SD1.5 is epsilon-prediction
loss.backward(); optimizer.step(); lr_scheduler.step(); optimizer.zero_grad()
```

Notes:
- Batch size 1 with gradient accumulation 2 (configurable). With batching, sample `t`, noise, the dropout coin and scale **per sample**.
- The ControlNet receives the same `encoder_hidden_states` as the UNet (this matches how the inference pipeline calls it).
- `controlnet_cond` is the **pixel-space** depth image, `[B,3,512,512]`, values in [0,1] (no [-1,1] normalization).
- Check the scheduler's `prediction_type` is `epsilon`; if not, compute the target accordingly.

### 6.4 The four tricks

| Trick | Setting (default) | Problem it solves |
|---|---|---|
| 1. Control dropout | `p_control = 0.5` | Without it the LoRA becomes dependent on the control signal and text-only generation degrades. |
| 2. Degrade the depth map | see below | A perfectly aligned same-photo map explains too much (including fine surface detail) and never happens at test time (train–test mismatch). A coarser, slightly misaligned map explains pose/shape only, forcing appearance into the LoRA. |
| 3. Random conditioning scale | `scale ~ U(0.5, 1.0)` | Makes the LoRA robust to the range of ControlNet strengths used at inference. |
| 4. Masked loss (optional, variant M2) | `bg_weight = 0.2` | Reduces reward for memorizing the background. |

`degrade(depth)` — apply to the **depth map only**, never to the image; each op randomly per step:
- Gaussian blur, sigma ~ U(0, 4) px (at 512 res).
- Down-then-up resample by a random factor in {1, 2, 4, 8} (bilinear).
- Small random affine: translate ±4% of width/height, scale U(0.95, 1.05), rotate ±3°; fill with edge values.
- All magnitudes in config; `degrade.enabled=false` returns the map unchanged.

### 6.5 Hyperparameter defaults

| Param | Default |
|---|---|
| Resolution | 512 |
| Steps | 1000 (save LoRA checkpoints at 400, 600, 800, 1000) |
| Optimizer | AdamW (8-bit if bitsandbytes works on this OS, else standard) |
| LR | 1e-4, constant with 50-step warmup |
| Weight decay | 1e-2 |
| Batch / grad accumulation | 1 / 2 |
| LoRA rank / alpha | 16 / 16 |
| Precision | frozen weights in bf16 (fp16 if bf16 unsupported); LoRA params and loss in fp32 |
| Prior preservation | **off** by default (flag `--prior_preservation` for later; generates ~200 class images with base SD1.5) |
| Seed | 0 |

### 6.6 Memory savers (must fit in 12 GB)
- Cache latents and the (fixed) text embedding once; then move the VAE and text encoder to CPU or delete them during training.
- `unet.enable_gradient_checkpointing()`.
- ControlNet forward under `torch.no_grad()`.
- Frozen modules in bf16/fp16; only LoRA params require grad.
- Log peak VRAM (`torch.cuda.max_memory_allocated`) per run.

### 6.7 Variants (all share one script, differ only by config)

| Variant | `p_control` | degrade | masked_loss | Role |
|---|---|---|---|---|
| **M0** | 0.0 | — | no | Naive baseline: standard DreamBooth-LoRA |
| **M1** | 0.5 | on | no | **Proposed** |
| **M2** | 0.5 | on | yes | Proposed + background disentanglement |
| **M3** | 1.0 | off | no | Ablation: always-on, perfect maps (expected to show control dependence / mismatch) |

Configs live in `configs/variants/M0.yaml` … `M3.yaml`. Command shape:
`python train.py --subject dog --variant M1 [--max_steps 20]`
Outputs: `runs/<subject>/<variant>/{checkpoint-400,...,final}/pytorch_lora_weights.safetensors`, `config.yaml`, `train_log.csv` (step, loss, control_on, scale), `val/` images.

Save LoRA weights in **diffusers format** so `pipe.load_lora_weights(dir)` works (e.g. `get_peft_model_state_dict` → `convert_state_dict_to_diffusers` → `StableDiffusionPipeline.save_lora_weights(dir, unet_lora_layers=...)`; check the installed diffusers API).

### 6.8 Validation during training
Every 200 steps, with a fixed seed, generate and save:
- (a) **text-only** (no ControlNet): `"a photo of sks <class> on the beach"`
- (b) **with ControlNet** using the subject's **Tier A** (held-out) depth map: `"a photo of sks <class>"`

Expectation: (a) looks like the subject; (b) looks like the subject in the held-out pose. Save as a grid per checkpoint.

**Phase 2 acceptance:**
1. Smoke test (`--max_steps 20`) runs for M0 and M1 on one subject; peak VRAM < 12 GB.
2. Full M0 and M1 run on **`dog` only**; validation grids saved; loss curves plotted. Report observations and **stop for review** before training all 24 runs.
3. After approval: train all 6 subjects × 4 variants (consider a single `scripts/train_all.py` that skips runs whose `final/` exists, so it can resume).

---

## 7. Inference (Phase 3)

### 7.1 Pipeline
```python
controlnet = ControlNetModel.from_pretrained("lllyasviel/control_v11f1p_sd15_depth", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", controlnet=controlnet,
    torch_dtype=torch.float16, safety_checker=None).to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
pipe.load_lora_weights("runs/dog/M1/final")
img = pipe(prompt, image=depth_pil, num_inference_steps=30, guidance_scale=7.5,
           controlnet_conditioning_scale=0.8, control_guidance_end=1.0,
           generator=torch.Generator("cuda").manual_seed(seed)).images[0]
pipe.unload_lora_weights()   # before loading the next LoRA
```
For text-only methods use `StableDiffusionPipeline` with the same base model, scheduler, steps, CFG and seeds.

### 7.2 Prompt construction
- Personalized methods: `"a photo of sks <class>"` + (`" " + context` for Tier C, context from `configs/backgrounds.yaml`).
- ControlNet-alone baseline (no LoRA): `"a photo of a <class>"` + context.
- Text-only methods (B1 and the text-only sanity runs) use the same prompt as the personalized methods, without a depth map.
- Use the **same depth map, same prompt context, same seeds** for every method (paired comparison).

### 7.3 Methods compared

| Method | LoRA | ControlNet at inference |
|---|---|---|
| B1: DreamBooth alone | M0 | no |
| B2: ControlNet alone | none | yes |
| B3: Naive combo | M0 | yes |
| **Ours** | **M1, M2** | **yes** |
| Ablation | M3 | yes |
| Optional B4: BLIP-Diffusion + ControlNet | — | yes (only if time permits) |
| Text-only sanity | M1, M2, M3 | no (compare with B1) |

### 7.4 Settings and budget
- Seeds: `[0, 1, 2, 3]`.
- **Scale sweep (2 subjects only: `dog`, `backpack`):** `controlnet_conditioning_scale ∈ {0.4, 0.6, 0.8, 1.0}` × `control_guidance_end ∈ {1.0, 0.7}`, all tiers, methods B2, B3, M1, M2. Use it to pick one setting.
- **Full evaluation:** all 6 subjects, all tiers, all methods, at the chosen setting.
- Before the full run, time a batch of 20 images and estimate total runtime in `NOTES.md`; if too long, reduce the number of Tier C backgrounds to 6 or seeds to 3 and note it.
- Script: `python generate.py --config configs/eval_full.yaml`; outputs `outputs/<subject>/<method>/<tier>/<case_id>_s<scale>_e<end>_seed<k>.png` plus a manifest CSV. Skip files that already exist (resumable).

**Phase 3 acceptance:** sweep done, chosen setting recorded with justification, full generation complete, qualitative grids saved (rows = methods, columns = test cases) per subject.

---

## 8. Evaluation (Phase 4) — `evaluate.py`

For each generated image, compute and store in `results/metrics.csv` (one row per image, with subject, category, method, tier, case, seed, scale, end):

| Metric | How | Measures |
|---|---|---|
| **DINO-masked** ↑ | Segment the subject in the generated image (same masker as §5.2), crop to mask bbox, fill background with neutral gray, resize to 224; same for real **training + held-out** images; mean pairwise cosine of `facebook/dino-vits16` CLS embeddings | Identity (preferred, per DreamBooth) |
| **CLIP-I-masked** ↑ | Same crops, CLIP ViT-B/32 image embeddings | Identity |
| **DINO / CLIP-I full-image** ↑ | As in DreamBooth (no masking) | Reported for comparability; note it rewards copying training backgrounds |
| **CLIP-T** ↑ | Cosine between the image and the prompt with `sks ` removed (i.e. class-level prompt) | Prompt alignment |
| **Depth error** ↓ | Re-estimate depth of the output (same estimator); align to the input map with least-squares scale + shift; report RMSE on [0,1] maps | Layout fidelity |
| **Mask IoU** ↑ (Tier B/C) | IoU between output subject mask and the case's `source_mask` (for C-composite: the pasted subject mask) | Placement / silhouette fidelity |
| **Background fidelity** ↓ (Tier C) | Depth RMSE computed **outside** the source mask only (output depth vs background depth) | Did the scene follow the given background? |

If segmentation fails on a generated image (empty mask), record NaN and count failures per method — that count is itself informative.

Aggregation (`scripts/aggregate.py`):
1. Table: method × {DINO-m, CLIP-I-m, CLIP-T, depth RMSE, IoU}, mean ± std, **split by category (live / object)** and by tier.
2. **Headline plot:** identity (DINO-masked) vs layout (depth RMSE) for each method across the conditioning-scale sweep — one curve per method. Better = up and left.
3. Text-only sanity table: B1 vs M1/M2/M3 without ControlNet (DINO, CLIP-T).
4. Qualitative figure: for 2–3 subjects, one row per method on the same Tier B/C cases.

**Phase 4 acceptance:** `results/` contains the CSV, tables (markdown + CSV), the tradeoff plot, and qualitative figures, plus a short written summary in `NOTES.md` checking H1–H3.

---

## 9. Repository layout

```
boothnet/
├── PROJECT_BRIEF.md          # this file
├── NOTES.md                  # agent's running log
├── requirements.txt
├── dataset/                  # PROVIDED BY USER — READ-ONLY
│   ├── Subjects/             # DreamBench subject folders
│   └── Backgrounds/          # scene images for Tier C
├── configs/
│   ├── subjects.yaml
│   ├── backgrounds.yaml      # background file -> context phrase + placement params
│   ├── train_base.yaml
│   ├── variants/{M0,M1,M2,M3}.yaml
│   ├── eval_sweep.yaml
│   └── eval_full.yaml
├── boothnet/                 # library code
│   ├── data.py               # dataset loading, caching latents/embeddings
│   ├── degrade.py            # depth-map degradation
│   ├── models.py             # loading SD/ControlNet/LoRA, saving LoRA
│   ├── preprocess.py         # depth, masks
│   └── metrics.py            # DINO, CLIP, depth error, IoU
├── scripts/
│   ├── check_env.py
│   ├── prepare_data.py
│   ├── build_test_bank.py
│   ├── train_all.py
│   └── aggregate.py
├── train.py
├── generate.py
├── evaluate.py
├── data/{processed,test_bank}/   # all derived data (generated by scripts)
├── runs/                     # LoRA checkpoints + training logs
├── outputs/                  # generated images
└── results/                  # metrics, tables, plots, figures
```

---

## 10. Phase plan (stop for review after each)

| Phase | Deliverable | Acceptance |
|---|---|---|
| 0. Setup | env, `check_env.py`, `NOTES.md` | §4 |
| 1. Data | processed subjects + test bank + previews | §5 |
| 2. Training | `train.py`, variants, M0+M1 on `dog` → review → all runs | §6.8 |
| 3. Inference | `generate.py`, sweep, full generation, grids | §7.4 |
| 4. Evaluation | `evaluate.py`, `aggregate.py`, tables, plots, summary | §8 |

If time runs short, cut in this order: optional BLIP-Diffusion baseline → C-bg-only exploration → M3 on all but 2 subjects → Tier B → number of Tier C backgrounds. Never cut M0 vs M1 — that comparison is the project.

---

## 11. Known pitfalls checklist

- [ ] PyTorch build supports the RTX 50-series GPU (CUDA 12.8+ wheels).
- [ ] Depth convention: near = bright, 3-channel, [0,1], 512×512 — same estimator for training maps, test maps and the depth-error metric.
- [ ] Latents scaled by `vae.config.scaling_factor` (0.18215 for SD1.5).
- [ ] ControlNet residuals cast to the UNet's dtype; ControlNet under `no_grad`.
- [ ] Dropout coin / scale / degradation sampled **per step (per sample)**, and logged.
- [ ] Degradation applied to the depth map only, never the RGB image or the mask.
- [ ] No horizontal flips.
- [ ] Same seeds, depth maps and prompt contexts across all methods.
- [ ] CLIP-T computed on the prompt **without** the `sks` token.
- [ ] LoRA unloaded between methods; no LoRA leaking into the ControlNet-alone baseline.
- [ ] Safety checker disabled only to avoid false black images on benign DreamBench subjects; note it in `NOTES.md`.
- [ ] Every run's resolved config saved next to its outputs.
- [ ] Nothing under `./dataset` is modified; derived files go to `data/`.
- [ ] C-composite depth maps: pasted subject is nearer (brighter) than the background behind it, sits on the ground, and has a sensible size — check the overlays.
- [ ] Background images are center-cropped/resized to 512×512 exactly like subject images before depth estimation.
