# Running BoothNet training on a remote GPU server

How to set up the code on a Linux GPU server, run the remaining LoRA trainings (6 subjects × 4 variants; `dog` M0 and
M1 are already trained locally), and bring the results back. For the project itself see [README.md](README.md) and
[PROJECT_BRIEF.md](PROJECT_BRIEF.md).

Training only needs the preprocessed data, which was made and checked by eye on the laptop. **Copy `data/` to the
server; do not regenerate it there.** Regenerating could give slightly different masks and test cases (different GPU,
different SD sampling), and the training inputs would no longer match what was reviewed.

---

## 1. File structure

```
ELEC542_FP/
├── README.md                     project overview + local instructions
├── README_SERVER_INSTRUCTIONS.md this file
├── PROJECT_BRIEF.md              specification (method, experiments)
├── NOTES.md                      running log: decisions, problems, results
├── requirements.txt              pinned Python packages (PyTorch from the CUDA 12.8 index)
├── train.py                      training entry point (one subject, one variant)
├── boothnet/                     library code
│   ├── data.py                   loads data/processed, caches VAE latents
│   ├── degrade.py                depth-map degradation used by M1/M2
│   ├── models.py                 SD1.5 / ControlNet / LoRA loading + LoRA saving
│   ├── preprocess.py             depth estimation + subject masks (only needed for preprocessing)
│   └── utils.py                  config I/O, image grids
├── scripts/
│   ├── train_all.py              runs every (subject, variant); skips runs whose final/ exists
│   ├── plot_losses.py            overlays loss curves of several variants
│   ├── check_env.py              environment check (one ControlNet generation)
│   ├── prepare_data.py           Phase 1 preprocessing (NOT needed on the server)
│   └── build_test_bank.py        Phase 1 test bank (NOT needed on the server)
├── configs/
│   ├── train_base.yaml           training hyperparameters (shared by all variants)
│   ├── variants/M0..M3.yaml      variant differences (control dropout, degradation, masked loss)
│   ├── subjects.yaml             the 6 subjects, class nouns, categories
│   ├── test_bank.yaml, backgrounds.yaml   Phase 1 configs
├── dataset/                      original DreamBench images + backgrounds   (NOT in git, copy manually)
├── data/                         preprocessed data                          (NOT in git, copy manually)
│   ├── processed/<subject>/      training images, depth maps, masks, split.json   <- training input
│   └── test_bank/<subject>/      test cases; Tier A depth is used for validation  <- training input
├── runs/                         training outputs (NOT in git)
│   └── <subject>/<variant>/      checkpoint-400/600/800/1000/, final/, train_log.csv, val/, loss.png, ...
└── outputs/                      generated images (NOT in git, not needed on the server)
```

What the server needs:

| Item | How it gets there | Needed for |
|---|---|---|
| code (`*.py`, `boothnet/`, `scripts/`, `configs/`, `requirements.txt`) | GitHub | everything |
| `data/` (~75 MB) | manual copy (§2.2) | training (required) |
| `dataset/` (~112 MB) | manual copy (§2.2) | only `check_env.py`'s default image and re-running Phase 1 |
| `runs/dog/M0`, `runs/dog/M1` (~160 MB) | manual copy (§2.2) | lets `train_all.py` skip the 2 runs already done |

---

## 2. Getting code and data onto the server

### 2.1 Code via GitHub (on the laptop)

`dataset/`, `data/`, `runs/` and `outputs/` are git-ignored, so only code and docs are committed. The images are
transferred separately (§2.2). The DreamBench images have their own licenses
(`dataset/Subjects/references_and_licenses.txt`), and the backgrounds were collected separately.

```powershell
git add .
git status            # check: no dataset/, data/, runs/ or outputs/ files listed
git commit -m "BoothNet code: preprocessing, test bank, training"
git push
```

On the server:

```bash
git clone <your-repo-url> ELEC542_FP
cd ELEC542_FP
```

### 2.2 Data via scp (Windows 11 has `tar` and `scp` built in)

On the laptop, from the repository root (PowerShell):

```powershell
tar -czf boothnet_data.tar.gz data dataset runs/dog/M0 runs/dog/M1
scp boothnet_data.tar.gz <user>@<server>:~/ELEC542_FP/
```

On the server:

```bash
cd ~/ELEC542_FP
tar -xzf boothnet_data.tar.gz && rm boothnet_data.tar.gz
ls data/processed data/test_bank      # expect: backpack cat2 dog dog6 rc_car teapot
```

---

## 3. Installing the environment

### 3.1 Check the GPU driver

```bash
nvidia-smi
```

The pinned PyTorch build (`torch==2.11.0+cu128`) needs an NVIDIA driver that supports **CUDA 12.8**. On Linux that is
driver **≥ 570**. If the server's driver is older, see §3.4.

### 3.2 Create the conda environment

If conda isn't available, install Miniconda in your home directory first
(https://docs.conda.io/en/latest/miniconda.html).

```bash
conda create -y -n boothnet python=3.11
conda activate boothnet
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` already contains `--extra-index-url https://download.pytorch.org/whl/cu128`, so pip fetches the
CUDA build of PyTorch.

Then:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### 3.3 Hugging Face model cache

The first run downloads SD1.5 and the depth ControlNet (several GB). Put the cache on a disk with space, and
optionally log in to avoid anonymous rate limits:

```bash
export HF_HOME=/path/with/space/hf_cache          # add to ~/.bashrc to make it permanent
export HF_TOKEN=<your Hugging Face read token>      # optional
```

### 3.4 Troubleshooting the install

- **Old driver (< 570):** install PyTorch for an older CUDA instead. Remove the `torch`/`torchvision` lines and the
  `--extra-index-url` line from a copy of `requirements.txt`, then:
  ```bash
  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
  pip install -r requirements_no_torch.txt
  ```
  Any recent PyTorch 2.x works for training. Record the version you used in NOTES.md.
- **`rembg` / `onnxruntime-gpu` fails to install:** these are only needed for Phase 1 preprocessing, not training.
  Delete those two lines and continue.
- **`bitsandbytes` fails:** optional. Training falls back to standard AdamW (logged in `summary.json` as
  `"optimizer": "AdamW"`). Prefer to keep 8-bit AdamW so the server runs match the `dog` runs.

---

## 4. Running the code

All commands from the repository root with `conda activate boothnet`.

### 4.1 Quick checks (a few minutes)

```bash
python scripts/check_env.py --image data/processed/dog/00/image.png   # versions, one ControlNet image, peak VRAM
python train.py --subject dog --variant M1 --max_steps 20              # training smoke test -> runs/_smoke/dog/M1/
```

The smoke test should finish without errors and print a `done: {...}` line. On the laptop it used ~3.6 GB VRAM.

### 4.2 List what will be trained

```bash
python scripts/train_all.py --dry_run
```

With `runs/dog/M0` and `runs/dog/M1` copied over, this lists **22 runs to go**.

### 4.3 Run all trainings (long job: keep it alive after logout)

Each run is 1000 optimizer steps. On the RTX 5070 Ti laptop a run took ~26 min, so 22 runs ≈ 9.5 h there. A
datacenter GPU should be noticeably faster (not measured).

**Option A: tmux (interactive server)**

```bash
tmux new -s boothnet
conda activate boothnet
cd ~/ELEC542_FP
python scripts/train_all.py 2>&1 | tee runs/logs/train_all.out
# detach: Ctrl-b then d      re-attach later: tmux attach -t boothnet
```

**Option B: nohup**

```bash
mkdir -p runs/logs
nohup python scripts/train_all.py > runs/logs/train_all.out 2>&1 &
```

**Option C: SLURM cluster.** Save as `train_all.sbatch` and submit with `sbatch train_all.sbatch`. Adjust partition,
time and conda path to your cluster.

```bash
#!/bin/bash
#SBATCH --job-name=boothnet
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=runs/logs/slurm_%j.out
source ~/miniconda3/etc/profile.d/conda.sh
conda activate boothnet
cd "$SLURM_SUBMIT_DIR"
python scripts/train_all.py
```

**Faster: several runs in parallel.** One run uses only ~3.6 GB of VRAM, so a large GPU can train several at once.
Start one `train_all.py` per group with **disjoint** subject lists, so no two processes ever train the same run:

```bash
# one 40-80 GB GPU: 3 processes on the same GPU
nohup python scripts/train_all.py --subjects dog dog6       > runs/logs/group1.out 2>&1 &
nohup python scripts/train_all.py --subjects cat2 backpack  > runs/logs/group2.out 2>&1 &
nohup python scripts/train_all.py --subjects teapot rc_car  > runs/logs/group3.out 2>&1 &

# several GPUs: one process per GPU
CUDA_VISIBLE_DEVICES=0 nohup python scripts/train_all.py --subjects dog dog6      > runs/logs/gpu0.out 2>&1 &
CUDA_VISIBLE_DEVICES=1 nohup python scripts/train_all.py --subjects cat2 backpack > runs/logs/gpu1.out 2>&1 &
CUDA_VISIBLE_DEVICES=2 nohup python scripts/train_all.py --subjects teapot rc_car > runs/logs/gpu2.out 2>&1 &
```

### 4.4 Monitoring

```bash
tail -f runs/logs/train_all.out              # which run is active, timing, failures
tail -c 300 runs/logs/<subject>_<variant>.log  # progress bar of the active run (loss, VRAM)
python scripts/train_all.py --dry_run        # how many runs remain
nvidia-smi                                   # GPU use
```

### 4.5 Interruptions and failures

`train_all.py` is resumable at the level of whole runs. Re-run the same command and every run with
`runs/<subject>/<variant>/final/pytorch_lora_weights.safetensors` is skipped. A run that was interrupted halfway
restarts from step 0, overwriting its partial folder. A failed run is reported at the end (`failed: [...]`). Its log
is `runs/logs/<subject>_<variant>.log`.

Training a single run by hand:

```bash
python train.py --subject cat2 --variant M2
```

### 4.6 Do not change the configs

Keep `configs/train_base.yaml` and `configs/variants/*.yaml` unchanged, so the server runs use exactly the settings of
the `dog` runs: 1000 steps, LR 1e-4, LoRA rank 16, checkpoints at 400/600/800/1000, seed 0. Every run saves its fully
resolved config to `runs/<subject>/<variant>/config.yaml`, so this can be verified afterwards.

---

## 5. Bringing the results back

Each run is ~80 MB (five LoRA checkpoints of 12.8 MB + validation images + logs), ≈ 2 GB for all 24.

On the server:

```bash
cd ~/ELEC542_FP
tar -czf boothnet_runs.tar.gz --exclude=runs/_smoke runs
```

On the laptop (PowerShell, repository root):

```powershell
scp <user>@<server>:~/ELEC542_FP/boothnet_runs.tar.gz .
tar -xzf boothnet_runs.tar.gz
python scripts\train_all.py --dry_run      # should report: 24 runs already finished, 0 to go
```

Extracting overwrites `runs/dog/M0` and `runs/dog/M1` with the identical copies that were uploaded. The other runs
are new.

After copying back, check:

- `runs/<subject>/<variant>/summary.json`: steps = 1000, optimizer, peak VRAM, minutes.
- `runs/<subject>/<variant>/val/summary.png`: validation images at steps 0, 200, …, 1000.
- `python scripts/plot_losses.py --subject <subject>`: loss curves of M0–M3.

Note the server GPU, driver, and package versions (`python scripts/check_env.py` prints them) in NOTES.md, since
those 22 runs come from different hardware than the local `dog` runs.
