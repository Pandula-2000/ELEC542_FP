"""Phase 1: test control bank (brief §5.3).

    python scripts/build_test_bank.py                 # Tier A + B
    python scripts/build_test_bank.py --tiers C       # Tier C (needs confirmed: true in backgrounds.yaml)
    python scripts/build_test_bank.py --preview_c     # Tier C placement preview only (not written to the bank)

Layout: data/test_bank/<subject>/<tier>/<case_id>/{depth.png, source.png, source_mask.png, case.json}
Tiers: A (held-out photo), B (other instance), C (background composite / scene), C_bg (background only).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

from boothnet.preprocess import (DepthEstimator, SubjectMasker, crop_for_subject, depth_to_pil,  # noqa: E402
                                 load_depth, load_mask, mask_to_pil)
from boothnet.utils import grid, image_files, load_json, load_yaml, save_json  # noqa: E402


# ----------------------------------------------------------------------------- helpers

def write_case(d: Path, depth: Image.Image, source: Image.Image, mask: Image.Image, case: dict) -> None:
    d.mkdir(parents=True, exist_ok=True)
    depth.convert("RGB").save(d / "depth.png")
    source.convert("RGB").save(d / "source.png")
    mask.convert("L").save(d / "source_mask.png")
    save_json(case, d / "case.json")


def base_case(s: dict, tier: str, case_id: str, context: str = "") -> dict:
    ctx = f" {context}" if context else ""
    return {"subject": s["name"], "class": s["class"], "category": s["category"], "tier": tier,
            "case_id": case_id, "context": context,
            "prompt_personalized": f"a photo of sks {s['class']}{ctx}",
            "prompt_class": f"a photo of a {s['class']}{ctx}"}


def tint(img: Image.Image, mask: np.ndarray, color=(0, 255, 0), alpha=0.45) -> Image.Image:
    col = Image.new("RGB", img.size, color)
    return Image.composite(Image.blend(img.convert("RGB"), col, alpha), img.convert("RGB"), mask_to_pil(mask))


def find_source(subjects_dir: Path, ref: str) -> Path:
    folder, stem = ref.split("/")
    hits = [p for p in image_files(subjects_dir / folder) if p.stem == stem]
    assert hits, f"Tier B source not found: {ref}"
    return hits[0]


def is_truncated(mask: np.ndarray, frac: float = 0.02) -> bool:
    """Subject cut off by the image border (top/bottom/left/right)?"""
    m = mask > 0.5
    edges = [m[:2].any(0).mean(), m[-2:].any(0).mean(), m[:, :2].any(1).mean(), m[:, -2:].any(1).mean()]
    return max(edges) > frac


# ----------------------------------------------------------------------------- Tier A

def build_tier_a(subjects, cfg_s, out_root: Path) -> None:
    for s in subjects:
        proc = ROOT / cfg_s["processed_dir"] / s["name"]
        hid = load_json(proc / "split.json")["heldout"][0]
        src = proc / hid
        case = base_case(s, "A", f"heldout_{hid}")
        case["source_ref"] = f"data/processed/{s['name']}/{hid}"
        write_case(out_root / s["name"] / "A" / case["case_id"], Image.open(src / "depth.png"),
                   Image.open(src / "image.png"), Image.open(src / "mask.png"), case)
        print(f"A  {s['name']:9s} {case['case_id']}")


# ----------------------------------------------------------------------------- Tier B

def generate_sd_images(requests: list[tuple[str, int]], gen_cfg: dict, cache_dir: Path) -> dict:
    """Base SD1.5 text-only images for (noun, seed). Cached on disk."""
    paths = {(n, k): cache_dir / f"{n.replace(' ', '_')}_seed{k}.png" for n, k in requests}
    missing = [r for r, p in paths.items() if not p.exists()]
    if missing:
        from diffusers import StableDiffusionPipeline, UniPCMultistepScheduler

        pipe = StableDiffusionPipeline.from_pretrained(gen_cfg["model"], dtype=torch.float16, safety_checker=None,
                                                       requires_safety_checker=False).to("cuda")
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        cache_dir.mkdir(parents=True, exist_ok=True)
        for n, k in missing:
            img = pipe(f"a photo of a {n}", num_inference_steps=gen_cfg["steps"],
                       guidance_scale=gen_cfg["guidance_scale"],
                       generator=torch.Generator("cuda").manual_seed(k)).images[0]
            img.save(paths[(n, k)])
            print(f"SD generated '{n}' seed {k}")
        del pipe
        torch.cuda.empty_cache()
    return paths


def build_tier_b(subjects, cfg_s, cfg_t, out_root: Path, depth_est, masker: SubjectMasker) -> None:
    subjects_dir = ROOT / cfg_s["subjects_dir"]
    res = cfg_s["resolution"]
    # SD-generated sources first (needs the GPU on its own): all candidate seeds
    reqs = []
    for s in subjects:
        tb = cfg_t["tier_b"].get(s["name"], {})
        if tb.get("sd_count"):
            reqs += [(tb.get("tierb_noun", s["class"]), k) for k in range(cfg_t["sd_max_seed"] + 1)]
    sd_paths = generate_sd_images(reqs, cfg_t["sd_generation"], ROOT / cfg_t["out_dir"] / "_sd_generated")
    # keep the first sd_count seeds whose detected object is fully inside the frame
    margin = cfg_t.get("border_margin_px", 3)
    for s in subjects:
        tb = cfg_t["tier_b"].get(s["name"], {})
        if not tb.get("sd_count"):
            continue
        noun, seg_prompt = tb.get("tierb_noun", s["class"]), s.get("seg_prompt", s["class"])
        chosen, rejected = [], []
        for k in range(cfg_t["sd_max_seed"] + 1):
            box, _ = masker.detect(Image.open(sd_paths[(noun, k)]).convert("RGB").resize((res, res)), seg_prompt)
            ok = box is not None and box[0] > margin and box[1] > margin and box[2] < res - margin and box[3] < res - margin
            (chosen if ok else rejected).append(k)
            if len(chosen) == tb["sd_count"]:
                break
        tb["sd_seeds"] = chosen
        print(f"B  {s['name']}: SD seeds kept {chosen}, rejected (cut off / not detected) {rejected}")

    # unique sources -> image, depth, detection (torch); then one BiRefNet pass
    jobs, cache = [], {}
    for s in subjects:
        tb = cfg_t["tier_b"].get(s["name"], {})
        seg_prompt = s.get("seg_prompt", s["class"])
        noun = tb.get("tierb_noun", s["class"])
        srcs = [("real", r) for r in tb.get("real", [])] + [("sd", k) for k in tb.get("sd_seeds", [])]
        for kind, ref in srcs:
            key = (kind, ref, noun, seg_prompt, s["category"])
            if key not in cache:
                if kind == "real":
                    path = find_source(subjects_dir, ref)
                    img, crop = crop_for_subject(Image.open(path), res, masker, seg_prompt,
                                                 cfg_s.get("strip_borders", True), cfg_s.get("subject_aware_crop", True))
                    src_ref = str(path.relative_to(ROOT)).replace("\\", "/")
                else:
                    img, crop = Image.open(sd_paths[(noun, ref)]).convert("RGB").resize((res, res)), None
                    src_ref = f"SD1.5 'a photo of a {noun}' seed {ref}"
                cache[key] = {"img": img, "crop": crop, "src_ref": src_ref,
                              "depth": depth_to_pil(depth_est(img, res)),
                              "info": masker.stage_torch(img, seg_prompt, s["category"])}
            case_id = ref.replace("/", "_") if kind == "real" else f"sd_{noun.replace(' ', '_')}_seed{ref}"
            jobs.append((s, case_id, key))
    entries = list(cache.values())
    masker.stage_salient([e["img"] for e in entries], [e["info"] for e in entries])
    for e in entries:
        e["mask"] = e["info"].pop("_mask")
    for s, case_id, key in jobs:
        e = cache[key]
        case = base_case(s, "B", case_id)
        case.update({"source_ref": e["src_ref"], "crop": e["crop"],
                     "mask": {k: v for k, v in e["info"].items() if not k.startswith("_")}})
        write_case(out_root / s["name"] / "B" / case_id, e["depth"], e["img"], mask_to_pil(e["mask"]), case)
        flag = "  (EMPTY MASK!)" if e["mask"].mean() < 0.01 else ""
        print(f"B  {s['name']:9s} {case_id:24s} mask={e['mask'].mean():.0%}{flag}")


# ----------------------------------------------------------------------------- Tier C

def composite_depth(bg: np.ndarray, src_depth: np.ndarray, src_mask: np.ndarray, p: dict):
    """Paste a subject-shaped depth region into a background depth map (near = bright)."""
    H, W = bg.shape
    ys, xs = np.nonzero(src_mask > 0.5)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    cd, cm = src_depth[y0:y1, x0:x1], src_mask[y0:y1, x0:x1]
    th = max(8, round(p["size_frac"] * H))
    tw = max(8, round((x1 - x0) * th / (y1 - y0)))
    if tw > 0.9 * W:  # very wide subjects: limit width instead
        th, tw = max(8, round(th * 0.9 * W / tw)), round(0.9 * W)
    cd_r = np.asarray(Image.fromarray(cd).resize((tw, th), Image.BILINEAR), dtype=np.float32)
    cm_r = np.asarray(Image.fromarray((cm * 255).astype(np.uint8)).resize((tw, th), Image.NEAREST)) > 127
    bottom = round(p["y_bottom"] * H)
    top, left = bottom - th, round(p["x_center"] * W - tw / 2)
    # bg depth at the feet: median over a thin strip at the subject's base
    fx0, fx1 = max(left, 0), min(left + tw, W)
    feet = bg[max(bottom - 3, 0):min(bottom + 3, H), fx0:fx1]
    b = float(np.median(feet)) if feet.size else float(np.median(bg))
    lo = min(b + p["delta"], 0.95)
    vals = cd_r[cm_r]
    vmin, vmax = float(vals.min()), float(vals.max())
    scaled = lo + (cd_r - vmin) / max(vmax - vmin, 1e-6) * (1.0 - lo)
    out, mask = bg.copy(), np.zeros_like(bg)
    # clip paste region to the image
    sy0, sx0 = max(top, 0), max(left, 0)
    sy1, sx1 = min(top + th, H), min(left + tw, W)
    ry0, rx0 = sy0 - top, sx0 - left
    region = cm_r[ry0:ry0 + (sy1 - sy0), rx0:rx0 + (sx1 - sx0)]
    sub = out[sy0:sy1, sx0:sx1]
    sub[region] = scaled[ry0:ry0 + (sy1 - sy0), rx0:rx0 + (sx1 - sx0)][region]
    mask[sy0:sy1, sx0:sx1][region] = 1.0
    info = {"bbox_xyxy": [int(left), int(top), int(left + tw), int(top + th)], "bg_depth_at_feet": round(b, 3),
            "pasted_range": [round(lo, 3), 1.0]}
    return out, mask, info


def pick_pose_source(s, out_root: Path):
    """Tier A unless truncated by the border -> first un-truncated Tier B case."""
    cands = sorted((out_root / s["name"] / "A").iterdir()) + sorted((out_root / s["name"] / "B").iterdir())
    for d in cands:
        if not is_truncated(load_mask(d / "source_mask.png")):
            return d
    return cands[0]


def build_tier_c(subjects, cfg_s, cfg_bg, out_root: Path, depth_est, masker: SubjectMasker,
                 preview_dir: Path | None) -> None:
    res = cfg_s["resolution"]
    bg_dir = ROOT / cfg_bg["backgrounds_dir"]
    defaults = cfg_bg["placement_defaults"]
    bgs = {}
    for fname, b in cfg_bg["backgrounds"].items():
        img, crop = crop_for_subject(Image.open(bg_dir / fname), res, None)  # same bar-strip + center crop
        bgs[fname] = {"img": img, "crop": crop, "depth": depth_est(img, res), "cfg": b}
    # scene backgrounds: segment their subject (class-prompted, category of matching subjects)
    scene = [(f, b) for f, b in bgs.items() if b["cfg"]["type"] == "scene"]
    cat_of = {s["class"]: s["category"] for s in subjects}
    infos = [masker.stage_torch(b["img"], b["cfg"]["scene_class"], cat_of.get(b["cfg"]["scene_class"], "live"))
             for _, b in scene]
    masker.stage_salient([b["img"] for _, b in scene], infos)
    for (f, b), inf in zip(scene, infos):
        b["scene_mask"] = inf.pop("_mask")
        b["scene_mask_info"] = inf

    for s in subjects:
        pose_dir = pick_pose_source(s, out_root)
        pose_depth, pose_mask = load_depth(pose_dir / "depth.png"), load_mask(pose_dir / "source_mask.png")
        rows, labels = [], []
        for fname, b in bgs.items():
            stem = Path(fname).stem
            bc = b["cfg"]
            if bc["type"] == "composite":
                p = {**defaults, **{k: bc[k] for k in defaults if k in bc}}
                dep, msk, pinfo = composite_depth(b["depth"], pose_depth, pose_mask, p)
                case = base_case(s, "C", stem, bc["context"])
                case.update({"c_type": "composite", "background": fname, "placement": p, **pinfo,
                             "pose_source": str(pose_dir.relative_to(ROOT)).replace("\\", "/")})
            elif bc["scene_class"] == s["class"]:
                dep, msk = b["depth"], b["scene_mask"]
                case = base_case(s, "C", stem, bc["context"])
                case.update({"c_type": "scene", "background": fname, "scene_mask": b["scene_mask_info"]})
            else:
                continue
            ov = tint(b["img"], msk)
            rows.append([b["img"], ov, depth_to_pil(dep)])
            labels.append(f"{stem} [{case['c_type']}] '{bc['context']}'")
            if preview_dir is None:
                d = out_root / s["name"] / "C" / stem
                write_case(d, depth_to_pil(dep), b["img"], mask_to_pil(msk), case)
                grid([[ov, depth_to_pil(dep)]], cell=256).save(d / "overlay.png")
                if bc["type"] == "composite" and s["name"] in cfg_bg.get("bg_only_subjects", []):
                    cb = base_case(s, "C_bg", stem, bc["context"])
                    cb.update({"c_type": "bg_only", "background": fname})
                    write_case(out_root / s["name"] / "C_bg" / stem, depth_to_pil(b["depth"]), b["img"],
                               mask_to_pil(np.zeros_like(b["depth"])), cb)
        sheet = grid(rows, labels, cell=200, col_titles=["background", "pasted mask", "composite depth"])
        target = (preview_dir / f"{s['name']}.png") if preview_dir else (out_root / s["name"] / "C_preview.png")
        target.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(target)
        print(f"C  {s['name']:9s} {len(rows)} cases, pose source = {pose_dir.relative_to(out_root)} -> {target}")


# ----------------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiers", nargs="*", default=["A", "B"], choices=["A", "B", "C"])
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--preview_c", action="store_true", help="Tier C placement preview to data/tier_c_preview/")
    args = ap.parse_args()

    cfg_s = load_yaml("configs/subjects.yaml")
    cfg_t = load_yaml("configs/test_bank.yaml")
    cfg_bg = load_yaml("configs/backgrounds.yaml")
    subjects = [s for s in cfg_s["subjects"] if args.subjects is None or s["name"] in args.subjects]
    out_root = ROOT / cfg_t["out_dir"]
    m = cfg_s["masker"]
    masker = SubjectMasker(m["rembg_model"], m["threshold"], m["box_threshold"], m["min_overlap"], m["min_area"],
                         m.get("max_hole_frac", 0.001))

    if "A" in args.tiers:
        build_tier_a(subjects, cfg_s, out_root)
    depth_est = None
    if "B" in args.tiers:
        build_tier_b(subjects, cfg_s, cfg_t, out_root, DepthEstimator(), masker)
    if "C" in args.tiers or args.preview_c:
        if not args.preview_c and not cfg_bg.get("confirmed", False):
            sys.exit("configs/backgrounds.yaml is not confirmed (confirmed: false). Use --preview_c to preview.")
        depth_est = depth_est or DepthEstimator()
        build_tier_c(subjects, cfg_s, cfg_bg, out_root, depth_est, masker,
                     ROOT / "data" / "tier_c_preview" if args.preview_c else None)


if __name__ == "__main__":
    main()
