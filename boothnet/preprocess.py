"""Image preprocessing: cropping, depth estimation (Depth-Anything-V2-Small), subject masks."""
from __future__ import annotations

import gc

import numpy as np
import torch
from PIL import Image

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


# ----------------------------------------------------------------------------- cropping

def strip_borders(img: Image.Image, dark: int = 16, min_bar: int = 8) -> tuple[Image.Image, tuple]:
    """Remove baked-in black letterbox/pillarbox bars (rows/cols whose max value < `dark`).

    Only bars of at least `min_bar` px are removed. Returns (image, crop_box) where
    crop_box is (left, top, right, bottom) in the original image.
    """
    img = img.convert("RGB")
    a = np.asarray(img.convert("L"))
    rows = np.where(a.max(1) >= dark)[0]
    cols = np.where(a.max(0) >= dark)[0]
    if len(rows) == 0 or len(cols) == 0:
        return img, (0, 0, img.width, img.height)
    t, b = rows[0], a.shape[0] - 1 - rows[-1]
    l, r = cols[0], a.shape[1] - 1 - cols[-1]
    t, b, l, r = [v if v >= min_bar else 0 for v in (t, b, l, r)]
    box = (int(l), int(t), int(img.width - r), int(img.height - b))
    return img.crop(box), box


def square_offset(long_len: int, side: int, coverage: np.ndarray | None) -> int:
    """Offset of a `side`-long window along an axis of length `long_len`.

    With no coverage profile -> centered. Otherwise choose the window keeping the most
    subject pixels (coverage = per-position mask sum), ties broken towards the center.
    """
    center = (long_len - side) // 2
    if coverage is None or coverage.sum() == 0:
        return center
    cs = np.concatenate([[0], np.cumsum(coverage)])
    kept = cs[side:] - cs[:-side]  # kept[o] = mask kept with offset o
    best = kept.max()
    cands = np.where(kept >= best - 1e-6)[0]
    return int(cands[np.argmin(np.abs(cands - center))])


def square_crop_resize(img: Image.Image, size: int = 512, mask: np.ndarray | None = None) -> tuple[Image.Image, tuple]:
    """Square crop (centered, or subject-aware if `mask` at img resolution is given) + resize."""
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    left = top = 0
    if w > h:
        left = square_offset(w, s, None if mask is None else mask.sum(0))
    elif h > w:
        top = square_offset(h, s, None if mask is None else mask.sum(1))
    box = (left, top, left + s, top + s)
    return img.crop(box).resize((size, size), Image.BICUBIC), box


def center_crop_resize(img: Image.Image, size: int = 512) -> Image.Image:
    """Center-crop to a square, then resize to size x size RGB."""
    return square_crop_resize(img, size)[0]


def crop_for_subject(raw: Image.Image, size: int, masker=None, seg_prompt: str | None = None,
                     strip: bool = True, subject_aware: bool = True) -> tuple[Image.Image, dict]:
    """Strip black bars, then square crop + resize.

    Non-square images (after stripping) get a subject-aware crop: the window that keeps most of
    the Grounding-DINO box for `seg_prompt` (needs `masker` with a .detect method); otherwise centered.
    """
    rec = {"orig_size": list(raw.size)}
    img = raw.convert("RGB")
    if strip:
        img, box = strip_borders(img)
        rec["border_box"] = list(box)
        rec["bars_stripped"] = list(box) != [0, 0, *raw.size]
    cover, rec["crop_mode"] = None, "center"
    if subject_aware and masker is not None and img.width != img.height:
        scale = 1024 / max(img.size)
        small = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))))
        det, _ = masker.detect(small, seg_prompt)
        if det is not None:
            cover = box_mask([v / scale for v in det], (img.height, img.width))
            rec["crop_mode"] = "subject_aware"
    out, sq = square_crop_resize(img, size, cover)
    rec["square_box"] = list(sq)
    return out, rec


# ----------------------------------------------------------------------------- depth

class DepthEstimator:
    """Depth-Anything-V2 wrapper.

    The model predicts relative *inverse* depth (disparity): larger = nearer.
    After per-image min-max normalisation this gives near = bright, which is the
    convention the SD1.5 depth ControlNet was trained with (MiDaS-style).
    """

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float16):
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.device = device
        self.dtype = dtype
        self.processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID)
        self.model = AutoModelForDepthEstimation.from_pretrained(DEPTH_MODEL_ID, dtype=dtype).to(device).eval()

    @torch.no_grad()
    def __call__(self, img: Image.Image, size: int = 512) -> np.ndarray:
        """Return a float32 [size, size] depth map in [0, 1], near = bright."""
        inputs = self.processor(images=img.convert("RGB"), return_tensors="pt")
        pv = inputs["pixel_values"].to(self.device, self.dtype)
        pred = self.model(pixel_values=pv).predicted_depth  # [1, h, w]
        pred = torch.nn.functional.interpolate(
            pred[:, None].float(), size=(size, size), mode="bicubic", align_corners=False
        )[0, 0]
        lo, hi = pred.min(), pred.max()
        d = (pred - lo) / (hi - lo + 1e-8)
        return d.clamp(0, 1).cpu().numpy().astype(np.float32)


def depth_to_pil(depth: np.ndarray) -> Image.Image:
    """[H,W] float in [0,1] -> 3-channel uint8 PIL image."""
    d8 = (np.clip(depth, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    return Image.fromarray(np.stack([d8] * 3, axis=-1), mode="RGB")


def load_depth(path) -> np.ndarray:
    """Read a saved depth.png -> float32 [H,W] in [0,1]."""
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


# ----------------------------------------------------------------------------- masks

def mask_to_pil(mask: np.ndarray) -> Image.Image:
    """Binary/float [H,W] mask -> 'L' image with values {0,255}."""
    return Image.fromarray((np.asarray(mask) > 0.5).astype(np.uint8) * 255, mode="L")


def load_mask(path) -> np.ndarray:
    """Read a saved mask.png -> float32 [H,W] in {0,1}."""
    return (np.asarray(Image.open(path).convert("L")) > 127).astype(np.float32)


class Masker:
    """Salient-subject segmentation with rembg (BiRefNet by default), as a short-lived session.

    BiRefNet's ONNX graph needs ~9 GiB of GPU memory, so it cannot share the 12 GB GPU with
    the torch models. Use as a context manager: the onnxruntime session (and its CUDA arena)
    is created on enter and destroyed on exit.
    Returns binary {0,1} float32 masks at the input image's resolution.
    """

    def __init__(self, model_name: str = "birefnet-general", threshold: float = 0.5):
        self.model_name, self.threshold = model_name, threshold
        self.session = None

    def __enter__(self):
        import onnxruntime as ort
        from rembg import new_session

        if hasattr(ort, "preload_dlls"):
            try:
                ort.preload_dlls()
            except Exception:
                pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.session = new_session(self.model_name, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        if "CUDAExecutionProvider" not in self.session.inner_session.get_providers():
            print("WARNING: rembg is running on CPU (slow)")
        return self

    def __exit__(self, *exc):
        self.session = None
        gc.collect()

    def __call__(self, img: Image.Image) -> np.ndarray:
        from rembg import remove

        assert self.session is not None, "use `with Masker(...) as m:`"
        m = remove(img.convert("RGB"), session=self.session, only_mask=True, post_process_mask=False)
        m = np.asarray(m.convert("L"), dtype=np.float32) / 255.0
        return (m > self.threshold).astype(np.float32)


class GroundedSam:
    """Grounding-DINO (class-noun box) + SAM (mask from box). torch, ~2.5 GiB fp32."""

    DINO_ID = "IDEA-Research/grounding-dino-base"
    SAM_ID = "facebook/sam-vit-large"

    def __init__(self, device: str = "cuda", box_threshold: float = 0.30, text_threshold: float = 0.25):
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor, SamModel,
                                  SamProcessor)

        self.device = device
        self.box_threshold, self.text_threshold = box_threshold, text_threshold
        self.dino_proc = AutoProcessor.from_pretrained(self.DINO_ID)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(self.DINO_ID).to(device).eval()
        self.sam_proc = SamProcessor.from_pretrained(self.SAM_ID)
        self.sam = SamModel.from_pretrained(self.SAM_ID).to(device).eval()

    @torch.no_grad()
    def detect(self, img: Image.Image, text: str):
        """Return (best box [x0,y0,x1,y1] or None, score)."""
        text = text.strip().lower()
        if not text.endswith("."):
            text += "."
        img = img.convert("RGB")
        inputs = self.dino_proc(images=img, text=text, return_tensors="pt").to(self.device)
        out = self.dino(**inputs)
        res = self.dino_proc.post_process_grounded_object_detection(
            out, inputs["input_ids"], threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[img.size[::-1]])[0]
        if len(res["scores"]) == 0:
            return None, 0.0
        k = int(res["scores"].argmax())
        return [round(v, 1) for v in res["boxes"][k].tolist()], round(float(res["scores"][k]), 3)

    @torch.no_grad()
    def segment_box(self, img: Image.Image, box) -> np.ndarray:
        inputs = self.sam_proc(images=img.convert("RGB"), input_boxes=[[box]], return_tensors="pt").to(self.device)
        out = self.sam(**inputs, multimask_output=True)
        masks = self.sam_proc.image_processor.post_process_masks(
            out.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())[0][0].numpy()
        iou = out.iou_scores[0, 0].float().cpu().numpy()
        # Candidate choice: the LARGEST of SAM's 3 masks that stays inside the box (<2% outside) and has
        # predicted IoU > 0.7. Pure argmax-IoU sometimes drops a part (rc_car/04 lost the driver figure);
        # the box constraint keeps context (saucers, the person wearing a backpack) out.
        inb = box_mask(box, masks[0].shape)
        ok = [k for k in range(len(masks))
              if (masks[k] & ~inb).sum() < 0.02 * max(masks[k].sum(), 1) and iou[k] > 0.7]
        k = max(ok, key=lambda j: masks[j].sum()) if ok else int(iou.argmax())
        return masks[k].astype(np.float32)

    def release(self):
        self.dino = self.sam = None
        gc.collect()
        torch.cuda.empty_cache()


def box_mask(box, shape) -> np.ndarray:
    m = np.zeros(shape, bool)
    if box is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in box]
        m[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = True
    return m


def fill_small_holes(mask: np.ndarray, max_hole_frac: float) -> np.ndarray:
    """Fill enclosed background holes smaller than max_hole_frac of the image (segmentation pinholes).

    Large holes (e.g. a teapot handle loop) are kept.
    """
    import cv2

    if max_hole_frac <= 0:
        return mask
    m = (mask > 0.5).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(1 - m, connectivity=4)
    H, W = m.shape
    out = m.copy()
    for k in range(1, n):
        x, y, w, h, area = stats[k]
        touches = x == 0 or y == 0 or x + w == W or y + h == H
        if not touches and area < max_hole_frac * m.size:
            out[lab == k] = 1
    return out.astype(np.float32)


def filter_components(sal: np.ndarray, box, min_overlap: float, min_area: float) -> np.ndarray:
    """Keep connected components of `sal` with >= min_overlap of their area inside `box`."""
    import cv2

    out = np.zeros(sal.shape, np.float32)
    if box is None:
        return out
    inbox = box_mask(box, sal.shape)
    n, lab = cv2.connectedComponents(sal.astype(np.uint8), connectivity=8)
    for k in range(1, n):
        comp = lab == k
        area = comp.sum()
        if area >= min_area * sal.size and (comp & inbox).sum() >= min_overlap * area:
            out[comp] = 1.0
    return out


class SubjectMasker:
    """Per-category hybrid subject segmentation (used for training masks AND eval metrics).

    - object: Grounding-DINO box for the class noun -> SAM mask. Salient-object models grab
      whatever is salient (the person wearing the backpack, the teapot's saucer/cup/stand);
      class-prompted SAM isolates the object itself.
    - live:   BiRefNet salient mask, keeping only connected components that overlap the
      Grounding-DINO box for the class noun. On close-up pet photos DINO's box often covers
      only the head, so SAM-from-box misses the body; BiRefNet gets the whole animal, and the
      box filter rejects unrelated salient things (cars, people) in busy generated scenes.
    No detection -> empty mask (counted as a segmentation failure in eval).

    Two-stage, so the torch models and BiRefNet (~9 GiB) are never on the GPU together:
        sm.mask_batch(images, texts, categories) -> [(mask, info), ...]
    """

    def __init__(self, rembg_model: str = "birefnet-general", threshold: float = 0.5,
                 box_threshold: float = 0.30, min_overlap: float = 0.10, min_area: float = 0.002,
                 max_hole_frac: float = 0.001):
        self.rembg_model, self.threshold = rembg_model, threshold
        self.box_threshold = box_threshold
        self.min_overlap, self.min_area = min_overlap, min_area
        self.max_hole_frac = max_hole_frac
        self.gsam: GroundedSam | None = None

    def torch_models(self) -> GroundedSam:
        if self.gsam is None:
            self.gsam = GroundedSam(box_threshold=self.box_threshold)
        return self.gsam

    def release_torch(self):
        if self.gsam is not None:
            self.gsam.release()
            self.gsam = None

    def detect(self, img: Image.Image, text: str):
        return self.torch_models().detect(img, text)

    def stage_torch(self, img: Image.Image, text: str, category: str) -> dict:
        g = self.torch_models()
        box, score = g.detect(img, text)
        info = {"method": "gdino+sam" if category == "object" else "birefnet&gdino",
                "box": box, "score": score, "category": category}
        if category == "object":
            info["_mask"] = (fill_small_holes(g.segment_box(img, box), self.max_hole_frac) if box is not None
                             else np.zeros(img.size[::-1], np.float32))
        return info

    def stage_salient(self, imgs: list[Image.Image], infos: list[dict]) -> None:
        todo = []
        for i, inf in enumerate(infos):
            if inf["category"] == "object":
                continue
            if inf["box"] is None:
                inf["_mask"] = np.zeros(imgs[i].size[::-1], np.float32)
            else:
                todo.append(i)
        if not todo:
            return
        self.release_torch()  # free the GPU for BiRefNet
        with Masker(self.rembg_model, self.threshold) as sal:
            for i in todo:
                infos[i]["_mask"] = fill_small_holes(
                    filter_components(sal(imgs[i]), infos[i]["box"], self.min_overlap, self.min_area),
                    self.max_hole_frac)

    def mask_batch(self, imgs: list[Image.Image], texts: list[str], categories: list[str]):
        """Masks for a batch of images. Returns [(mask {0,1} float32, info dict), ...]."""
        infos = [self.stage_torch(im, t, c) for im, t, c in zip(imgs, texts, categories)]
        self.stage_salient(imgs, infos)
        return [(inf.pop("_mask"), inf) for inf in infos]
