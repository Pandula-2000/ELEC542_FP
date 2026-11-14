# BoothNet — Running Notes

Spec: [PROJECT_BRIEF.md](PROJECT_BRIEF.md). This file logs environment, decisions, problems and results.

---

## Phase 0 — Environment (2026-11-07)

### Platform
- **Native Windows 11** (10.0.26300), VS Code. **Not WSL.**
- GPU: NVIDIA GeForce RTX 5070 Ti Laptop GPU, compute capability 12.0 (Blackwell, sm_120), 11.91 GiB usable VRAM, driver 621.01.
- Conda env: `X:\Environments\BoothNet` (Python 3.11.17). Run scripts with
  `X:\Environments\BoothNet\python.exe <script>` (or `conda activate X:\Environments\BoothNet`).
  System Python is 3.14, which is too new for parts of the stack (onnxruntime/rembg), hence the separate env.
- HF cache: default (`C:\Users\pandu\.cache\huggingface`). Windows symlink warning from huggingface_hub is harmless (files are copied instead of symlinked).

### Versions (from `scripts/check_env.py`)
| Package | Version |
|---|---|
| torch / torchvision | 2.11.0+cu128 / 0.26.0+cu128 (CUDA 12.8, bf16 supported) |
| diffusers | 0.42.1 |
| transformers | 5.22.2 |
| peft | 0.21.2 |
| accelerate | 1.15.0 |
| safetensors | 0.8.0 |
| huggingface-hub | 2.4.0 |
| opencv-python | 5.0.0 |
| pillow | 13.0.0 |
| numpy | 2.4.6 |
| pandas | 3.1.1 |
| matplotlib | 3.11.2 |
| pyyaml | 6.0.3 |
| rembg / onnxruntime-gpu | 2.0.87 / 1.31.1 (CUDA + TensorRT providers listed) |
| bitsandbytes | 0.50.2 — **AdamW8bit works** on native Windows (tested one step on GPU) |

### Model IDs — all resolve on the Hub
SD1.5 `stable-diffusion-v1-5/stable-diffusion-v1-5`, depth ControlNet `lllyasviel/control_v11f1p_sd15_depth`,
`depth-anything/Depth-Anything-V2-Small-hf`, `facebook/dino-vits16`, `openai/clip-vit-base-patch32`,
`Salesforce/blipdiffusion-controlnet`.

### API notes for installed versions
- diffusers 0.42 and transformers 5.x deprecate `torch_dtype=` → use **`dtype=`** in `from_pretrained` (brief §7.1 snippet uses the old name).
- SD1.5 scheduler `prediction_type = epsilon` (confirmed) → loss target is the noise, as in brief §6.3.
- Depth-Anything-V2 outputs relative inverse depth (larger = nearer); per-image min-max → **near = bright**. Verified visually
  (`outputs/check_env/sheet.png`: tongue/nose brightest, background black) and the depth ControlNet follows it correctly.
- Safety checker disabled (`safety_checker=None`) only to avoid false black images on benign DreamBench subjects.

### Phase 0 acceptance — `scripts/check_env.py`
- Input `dataset/Subjects/dog/00.jpg` → Depth-Anything depth → SD1.5 + depth ControlNet fp16, UniPC 30 steps, CFG 7.5, cond scale 0.8, seed 0.
- Generation 7–8 s / image (30 steps, 512²); **peak VRAM 3.33 GiB**.
- Output: `outputs/check_env/{input,depth,generated,sheet}.png`.
- Observation: with base SD1.5 (no LoRA) and the subject's *own* depth map, the output already resembles the corgi strongly
  (pose, ear shape, tongue). Depth carries a lot of silhouette information — relevant for interpreting Tier A results later
  (Tier A alone will flatter every method; Tier B/C matter more for H1).

### Repo
- Added `PROJECT_BRIEF.md` (spec, copied in), `.gitignore` (ignores `data/ runs/ outputs/` and weights), `requirements.txt`,
  `boothnet/preprocess.py` (center-crop + depth estimator, reused in Phase 1), `scripts/check_env.py`.
- `dataset/` is untracked in git; left as-is (read-only, user's data).

---

## Phase 1 — Data (2026-11-07)

### Dataset inventory (`dataset/`, read-only)
- `Subjects/`: 30 DreamBench subjects, 4–6 images each, + `prompts_and_classes.txt` (used for class nouns — all 6
  configured classes match it; rc_car's class is `toy`).
  Used: dog (5), dog6 (5), cat2 (5), backpack (6), teapot (5), rc_car (5). Held-out = last sorted file
  (`04`, backpack `05`); train = rest (4 imgs; backpack 5).
- `Backgrounds/`: 9 images, all 1280×1280 except beach2 (2000×1500).
  Empty scenes (→ C-composite): beach1, beach2, city_roads, city_sunset_new_york, road-mountains, times_square1jpg.
  Scenes containing a cat (→ C-scene, cat2 only): cat_pose1 (tabby sitting on grass), cat_pose2 (watercolor painting,
  grey cat leaning over a wall), cat_pose3 (tortoiseshell lying on a rug, top view). User added the cat scenes on purpose:
  "generate the subject cat in that background pose". Decision (user): **depth only** for these too (no Canny), to
  keep training/inference consistent.

### Problems hit & fixes
1. **onnxruntime-gpu 1.31 is a CUDA-13 build** → silently fell back to CPU next to torch cu128 (~2 min/img).
   Pinned `onnxruntime-gpu==1.23.2` (CUDA 12). ~0.6 s/img on GPU.
2. **Salient masks (rembg/BiRefNet) are wrong for objects**: backpack masks included the person wearing it; teapot masks
   included saucer/cups/stand. **Grounding-DINO+SAM** (class-prompted) isolates objects but on close-up corgis
   DINO's box covers only the head, so SAM misses the white body. → **Per-category hybrid `SubjectMasker`**
   (`boothnet/preprocess.py`):
   - object: Grounding-DINO-base box for the class noun → SAM-ViT-L mask;
   - live: BiRefNet salient mask, keep connected components with ≥10% of their area inside the DINO box
     (rejects unrelated salient things in busy generated scenes; no detection → empty mask = eval failure).
   Same masker will be used for the eval metrics (DINO/CLIP-masked, IoU). rc_car uses `seg_prompt: toy car` (DINO only).
3. **BiRefNet ONNX needs ~9 GiB** of GPU → can't coexist with torch models (OOM). Masker is two-stage
   (all torch work, release, one BiRefNet session, destroy). ISNet (lighter) was tested and rejected: it lost the
   dark-grey cat's body/tail against dark sofas. Loading BiRefNet via HF `trust_remote_code` was blocked (remote code) —
   not used.
4. **Letterbox bars baked into source files**: cat2 00/01/04 (320 px black bars on 1920² files), teapot 03 (12/45 px).
   Center-crop kept them (LoRA would learn black bars). → `strip_borders` (rows/cols with max < 16, ≥ 8 px), then square
   crop; images that are non-square after stripping get a **subject-aware** crop (window keeping most of the
   DINO box). cat2 keeps 89–100 % of the cat. Affects only those 4 images; all others are plain center crops.
5. Known small mask flaws (left as is): rc_car antenna missing in 02/04 (DINO box excludes it), a few pinholes in
   rc_car/04; backpack 00/01 include a strap. No generic hole filling (would close the teapot handle loop).

Depth: near = bright verified on every preview. Previews: `data/processed/<subject>/preview.png`.

### Test bank (`scripts/build_test_bank.py`, `configs/test_bank.yaml`)
- **Tier A** (1/subject): held-out image, its depth and mask.
- **Tier B** (3/subject), hand-picked for pose diversity:
  dog & dog6: dog2/00 (walking, frontal), dog3/03 (standing, side), dog7/01 (lying);
  cat2: cat/00, cat/01, cat/04; backpack: backpack_dog/01, 02, 04;
  teapot / rc_car: base SD1.5 text-only (UniPC 30 steps, CFG 7.5). Seed rule: try seeds 0..15, keep the first 3 whose
  DINO box does not touch the border → teapot seeds [0, 2, 3]; rc_car seeds [5, 6, 8] (0–4, 7 were cut-off close-ups).
  **Deviation (flag):** rc_car Tier B uses `"a photo of a toy car"` instead of `"a photo of a toy"` (class noun `toy`
  yields arbitrary toys whose depth says nothing about a car).
- **Tier C** — built after user confirmed `configs/backgrounds.yaml` (contexts + default placement). Previews:
  `data/test_bank/<subject>/C_preview.png`, per-case `overlay.png`. Pose source rule (`auto`): Tier A unless its mask touches the border →
  first un-truncated Tier B case. Result: dog/dog6 → dog2_00, cat2 → cat_01 (held-out photos are cut off);
  backpack/teapot/rc_car → own held-out image. Planned counts: 6 composite cases per subject + 3 C-scene for cat2;
  C_bg (background only) for dog and backpack.
  Observation: with feet at y=0.92 the ground is already near (depth ≈ 0.85–0.9), so the pasted subject's relief is
  compressed into ~[0.9, 1] — nearly flat white silhouette. Physically consistent; raise `y_bottom` for more relief.

| Subject | Cat. | Train | Held-out | Tier A | Tier B | Tier C (planned) |
|---|---|---|---|---|---|---|
| dog | live | 4 | 04 | 1 | 3 | 6 + C_bg 6 |
| dog6 | live | 4 | 04 | 1 | 3 | 6 |
| cat2 | live | 4 | 04 | 1 | 3 | 6 + 3 scene |
| backpack | object | 5 | 05 | 1 | 3 | 6 + C_bg 6 |
| teapot | object | 4 | 04 | 1 | 3 (SD) | 6 |
| rc_car | object | 4 | 04 | 1 | 3 (SD) | 6 |

Extra HF models used for masks: `IDEA-Research/grounding-dino-base`, `facebook/sam-vit-large`; rembg `birefnet-general`.

### Phase 1 — final mask fixes (after Tier C review)
- rc_car/04 (held-out, = rc_car's Tier C pose source): SAM's argmax-IoU candidate excluded the driver figure → dark
  holes inside the pasted silhouette. Hole size (~630 px) overlaps real holes (teapot handle loop 690–1200 px), so
  size-based filling can't fix it. **SAM candidate rule changed** (objects): pick the *largest* of the 3 candidates with
  <2% of its area outside the DINO box and predicted IoU > 0.7 (fallback argmax). Effect on all 22 object masks:
  rc_car/04 +8% (driver included); others change < 2% of area (edge pixels); no saucers/people added.
- Also added `fill_small_holes` (enclosed holes < 0.1% of image = pinholes), all masks.
- All masks + test bank regenerated with the final masker.

### Phase 1 result
75 test cases, all 512², no empty masks, every composite subject nearer than the ground at its feet (automatic check).
| Subject | A | B | C | C_bg |
|---|---|---|---|---|
| dog | 1 | 3 | 6 | 6 |
| dog6 | 1 | 3 | 6 | – |
| cat2 | 1 | 3 | 9 (6 composite + 3 scene) | – |
| backpack | 1 | 3 | 6 | 6 |
| teapot | 1 | 3 | 6 | – |
| rc_car | 1 | 3 | 6 | – |

---

## Phase 2 — Training (2026-11-07)

### Code
- `train.py` (custom single-GPU loop, brief §6.3), `boothnet/{degrade,models,data}.py`, `configs/train_base.yaml`,
  `configs/variants/M0..M3.yaml`, `scripts/train_all.py` (resumable, skips runs with `final/`),
  `scripts/plot_losses.py`.
- Implementation notes:
  - bf16 frozen weights, LoRA params fp32 (`cast_training_params`), forward under `torch.autocast(bf16)`, loss in fp32.
  - ControlNet run at `conditioning_scale=1.0` and residuals multiplied by the per-sample scale s (0 when the coin says
    off). Identical to `conditioning_scale=s` for B=1 (diffusers multiplies every residual by it) and gives per-sample
    dropout/scale for B>1.
  - "steps" = optimizer steps (×2 micro-batches, grad-accum 2). Grad clipping 1.0 (diffusers example default; brief
    silent on it). LR constant with 50-step warmup from 0.
  - Separate RNGs: python `random.Random(seed)` for image index / coin / scale / degradation, torch generator for noise
    and timesteps → all variants see the **same t / noise sequence** (paired), which is why M0 and M1 loss curves overlap.
  - Validation at step 0 (untrained reference) and every 200 steps; VAE/text encoder live on CPU between validations.
  - M3 keeps random scale U(0.5, 1) (brief §6.7 only says "always-on, perfect maps").
  - `--prior_preservation` reserved → raises NotImplementedError.
- Smoke tests (20 steps, `runs/_smoke/`): M0 & M1 OK, peak 3.63 GiB, AdamW8bit. LoRA saved in diffusers format,
  `load_lora_weights` changes output; `unload_lora_weights` restores the base model bit-exactly.

### Full runs, dog (1000 steps)
| Run | Time | Peak VRAM | Final loss (step) |
|---|---|---|---|
| dog/M0 | 25.6 min | 3.63 GiB | 0.042 |
| dog/M1 | 26.3 min | 3.64 GiB | 0.049 |
Estimated cost of all 24 runs ≈ 24 × 26 min ≈ 10.5 h (22 remaining ≈ 9.5 h).

Observations (`runs/dog/{M0,M1}/val/summary.png`, `runs/dog/ckpt_diag_M0_M1.png` = ckpt 400/600/final × Tier B
dog2_00 / Tier C beach1 / text-only, seeds 0–1):
- Identity: both learn the corgi by step 200; Tier B (chow-chow pose) is rendered as our corgi in that pose by both.
- Background memorisation: with no context prompt (Tier A/B) **both** M0 and M1 paint the training background (red wall +
  cherry blossoms) already at step 400. Tier C (composite depth + "on the beach") gives a beach for both.
- Text-only "… on the beach": M0 drifts to the training background by step 800–1000; **M1 keeps a beach/sky background
  longer** (consistent with the hypothesis that ControlNet-in-the-loop reduces background memorisation).
- **M1 over-cooks after ~600 steps**: over-saturated colours, purple tongue/face patches, doubled facial features with
  ControlNet at steps 800–1000. M0 also over-saturates at 1000 but less. Checkpoint 400 looks best for both.
- Loss curves are dominated by timestep noise and not informative (expected for diffusion).

### Exploratory sets, dog, Tier C (user request) — `outputs/dog/_explore/{times_square1jpg→times_square,city_roads}_grid.png`
B2 / B3 (M0+CN) / M1+CN at ckpt 400 & 1000, plus text-only M0/M1; CN scale 0.8, 30 UniPC steps, seeds 0–3.
- All CN methods follow the composite layout (scene perspective, dog bottom-centre in the pasted pose). B2: realistic
  but generic dog. M0+CN vs M1+CN at 400: both give the corgi in place; differences small at 4 seeds.
- At 1000: M0+CN city_roads scene degrades (washed-out, road turns to texture); M1+CN keeps the scene but puts the dog
  on a snowy ledge (training-like) and over-saturates; Times Square: M1 1000 shows face artifacts.
- **Text-only is where M1 differs most**: M0 (400 and esp. 1000) reproduces training compositions (dog on a ledge,
  close-up, red wall background in 3/4 seeds at 1000). M1 at 400 gives new poses (corgi standing/walking/lying in a
  snowy street) and the requested scene → less pose/background memorisation. M1 at 1000 over-saturated.
- Checkpoint 400 is the better operating point for both variants.
