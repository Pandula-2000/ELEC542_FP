"""Small shared helpers: repo root, config loading, contact sheets."""
from __future__ import annotations

import json
from pathlib import Path

import yaml
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path) -> dict:
    with open(ROOT / path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_class_file(path) -> dict[str, str]:
    """Parse DreamBench prompts_and_classes.txt -> {subject_name: class}."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if "," in line and not line.startswith(("subject_name", "'", "prompt")) and "{" not in line:
            name, cls = line.split(",", 1)
            out[name.strip()] = cls.strip()
    return out


def image_files(folder) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in exts)


def grid(rows: list[list[Image.Image]], labels: list[str] | None = None, cell: int = 256,
         col_titles: list[str] | None = None) -> Image.Image:
    """Make a contact sheet; rows of PIL images, optional row labels / column titles."""
    ncol = max(len(r) for r in rows)
    lab_h = 18 if labels else 0
    top = 18 if col_titles else 0
    sheet = Image.new("RGB", (ncol * cell, top + len(rows) * (cell + lab_h)), "white")
    d = ImageDraw.Draw(sheet)
    if col_titles:
        for j, t in enumerate(col_titles):
            d.text((j * cell + 4, 3), t, fill="black")
    for i, r in enumerate(rows):
        y = top + i * (cell + lab_h)
        if labels:
            d.text((4, y + 3), labels[i], fill="black")
        for j, im in enumerate(r):
            sheet.paste(im.convert("RGB").resize((cell, cell)), (j * cell, y + lab_h))
    return sheet
