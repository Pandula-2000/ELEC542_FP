"""Phase 1 preprocessing (brief §5.2).

For each subject in configs/subjects.yaml and each of its images:
  strip black bars, square crop (subject-aware if non-square), resize 512 -> image.png
  Depth-Anything-V2-Small (near = bright, [0,1], 3-channel)               -> depth.png
  per-category SubjectMasker, binary {0,1}                                -> mask.png
  meta.json (subject, class, category, split, crop info, mask info)
Writes data/processed/<subject>/<img_id>/..., split.json and preview.png.

Two stages (GPU memory): all torch work first, then one BiRefNet pass for live subjects.

    python scripts/prepare_data.py [--subjects dog cat2] [--overwrite]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from boothnet.preprocess import (DepthEstimator, SubjectMasker, crop_for_subject,  # noqa: E402
                                 depth_to_pil, mask_to_pil)
from boothnet.utils import grid, image_files, load_yaml, read_class_file, save_json  # noqa: E402


def overlay(img: Image.Image, mask_pil: Image.Image) -> Image.Image:
    """Image with the background dimmed + red tint, to judge mask tightness."""
    red = Image.new("RGB", img.size, (255, 0, 0))
    return Image.composite(img, Image.blend(img, red, 0.45), mask_pil)


def build_masker(cfg) -> SubjectMasker:
    m = cfg["masker"]
    return SubjectMasker(m["rembg_model"], m["threshold"], m["box_threshold"], m["min_overlap"], m["min_area"],
                         m.get("max_hole_frac", 0.001))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/subjects.yaml")
    ap.add_argument("--subjects", nargs="*", default=None, help="subset of subject names")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    res = cfg["resolution"]
    classes = read_class_file(ROOT / cfg["class_file"])
    subjects = [s for s in cfg["subjects"] if args.subjects is None or s["name"] in args.subjects]
    depth_est = DepthEstimator()
    masker = build_masker(cfg)

    # ---- stage 1 (torch): crop, depth, detection (+ SAM for objects)
    todo, splits = [], {}
    for s in subjects:
        name, cls, cat = s["name"], s["class"], s["category"]
        seg_prompt = s.get("seg_prompt", cls)
        if classes and classes.get(name) != cls:
            print(f"WARNING: class for {name} is '{cls}' in config but '{classes.get(name)}' in class file")
        files = image_files(ROOT / cfg["subjects_dir"] / name)
        assert len(files) >= 2, f"{name}: need >= 2 images"
        heldout = files[-1]  # deterministic: last in sorted order
        splits[name] = {"subject": name, "class": cls, "category": cat,
                        "train": [f.stem for f in files if f != heldout], "heldout": [heldout.stem]}
        for f in files:
            d = ROOT / cfg["processed_dir"] / name / f.stem
            if (d / "mask.png").exists() and not args.overwrite:
                continue
            img, crop = crop_for_subject(Image.open(f), res, masker, seg_prompt,
                                         cfg.get("strip_borders", True), cfg.get("subject_aware_crop", True))
            info = masker.stage_torch(img, seg_prompt, cat)
            todo.append({"dir": d, "img": img, "depth": depth_to_pil(depth_est(img, res)), "crop": crop,
                         "info": info, "meta": {"subject": name, "class": cls, "category": cat,
                                                "split": "heldout" if f == heldout else "train", "img_id": f.stem,
                                                "source_file": str(f.relative_to(ROOT)).replace("\\", "/")},
                         "seg_prompt": seg_prompt})
    print(f"stage 1 done: {len(todo)} images")

    # ---- stage 2 (BiRefNet): salient masks for live subjects, then save everything
    masker.stage_salient([t["img"] for t in todo], [t["info"] for t in todo])
    for t in todo:
        d = t["dir"]
        d.mkdir(parents=True, exist_ok=True)
        m = t["info"].pop("_mask")
        t["img"].save(d / "image.png")
        t["depth"].save(d / "depth.png")
        mask_to_pil(m).save(d / "mask.png")
        save_json({**t["meta"], "crop": t["crop"], "mask": {**t["info"], "seg_prompt": t["seg_prompt"]}},
                  d / "meta.json")

    # ---- previews + splits
    for name, sp in splits.items():
        out_dir = ROOT / cfg["processed_dir"] / name
        save_json(sp, out_dir / "split.json")
        rows, labels = [], []
        for img_id in sp["train"] + sp["heldout"]:
            d = out_dir / img_id
            img = Image.open(d / "image.png").convert("RGB")
            m_pil = Image.open(d / "mask.png").convert("L")
            frac = (np.asarray(m_pil) > 127).mean()
            if frac < 0.01:
                print(f"WARNING: {name}/{img_id} mask is (nearly) empty")
            meta = __import__("json").loads((d / "meta.json").read_text())
            note = " bars-stripped" if meta["crop"].get("bars_stripped") else ""
            note += " subj-crop" if meta["crop"]["crop_mode"] == "subject_aware" else ""
            rows.append([img, Image.open(d / "depth.png"), m_pil, overlay(img, m_pil)])
            labels.append(f"{img_id} ({'heldout' if img_id in sp['heldout'] else 'train'}) "
                          f"mask={frac:.0%} {meta['mask']['method']}{note}")
        grid(rows, labels, cell=192, col_titles=["image", "depth (near=bright)", "mask", "overlay"]).save(
            out_dir / "preview.png")
        print(f"{name:10s} class={sp['class']:9s} {sp['category']:6s} train={sp['train']} heldout={sp['heldout']}")


if __name__ == "__main__":
    main()
