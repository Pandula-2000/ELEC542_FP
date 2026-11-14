"""Train every (subject, variant) pair; skips runs whose final/ LoRA already exists (resumable).

    python scripts/train_all.py                          # all subjects in configs/subjects.yaml x M0..M3
    python scripts/train_all.py --subjects dog cat2 --variants M0 M1
    python scripts/train_all.py --dry_run
Each run is a separate process (clean GPU memory); stdout goes to runs/logs/<subject>_<variant>.log.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from boothnet.utils import load_json, load_yaml  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--variants", nargs="*", default=["M0", "M1", "M2", "M3"])
    ap.add_argument("--dry_run", action="store_true")
    args = ap.parse_args()

    subjects = args.subjects or [s["name"] for s in load_yaml("configs/subjects.yaml")["subjects"]]
    runs_dir = ROOT / load_yaml("configs/train_base.yaml")["runs_dir"]
    (runs_dir / "logs").mkdir(parents=True, exist_ok=True)
    todo = [(s, v) for s in subjects for v in args.variants
            if not (runs_dir / s / v / "final" / "pytorch_lora_weights.safetensors").exists()]
    done = len(subjects) * len(args.variants) - len(todo)
    print(f"{done} runs already finished, {len(todo)} to go: {todo}")
    if args.dry_run:
        return
    failed = []
    for k, (s, v) in enumerate(todo, 1):
        t0 = time.time()
        log = runs_dir / "logs" / f"{s}_{v}.log"
        print(f"[{k}/{len(todo)}] {s} {v} -> {log}", flush=True)
        with open(log, "w", encoding="utf-8") as f:
            rc = subprocess.call([sys.executable, str(ROOT / "train.py"), "--subject", s, "--variant", v],
                                 stdout=f, stderr=subprocess.STDOUT, cwd=ROOT)
        if rc != 0:
            failed.append((s, v))
            print(f"   FAILED (rc={rc}), see log", flush=True)
            continue
        summ = load_json(runs_dir / s / v / "summary.json")
        print(f"   ok in {(time.time() - t0) / 60:.1f} min, peak VRAM {summ['peak_vram_gib']} GiB", flush=True)
    print(f"finished; failed: {failed}")


if __name__ == "__main__":
    main()
