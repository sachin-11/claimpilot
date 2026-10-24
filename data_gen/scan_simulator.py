"""Simulate a phone scan of a PDF page: slight rotation, blur, noise, low contrast."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pymupdf
from PIL import Image, ImageEnhance, ImageFilter

DPI = 150


def render_page(pdf_path: Path, dpi: int = DPI) -> Image.Image:
    """Rasterise page 1 of a PDF."""
    with pymupdf.open(pdf_path) as doc:
        pix = doc[0].get_pixmap(dpi=dpi, alpha=False)
        return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)


def simulate_scan(image: Image.Image, rng: random.Random) -> Image.Image:
    angle = rng.choice([-1, 1]) * rng.uniform(0.8, 2.2)
    img = image.rotate(angle, resample=Image.BICUBIC, expand=True, fillcolor=(232, 230, 226))
    img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.7, 1.3)))
    img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.72, 0.85))
    img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.92, 1.0))

    arr = np.asarray(img).astype(np.float32)
    np_rng = np.random.default_rng(rng.randint(0, 2**32 - 1))
    arr += np_rng.normal(0, rng.uniform(6, 12), arr.shape)  # sensor noise
    # soft shadow gradient across the page, like uneven lighting
    gradient = np.linspace(rng.uniform(0.9, 1.0), 1.0, arr.shape[1], dtype=np.float32)
    arr *= gradient[None, :, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def scan_pdf(pdf_path: Path, out_path: Path, rng: random.Random) -> None:
    simulate_scan(render_page(pdf_path), rng).save(out_path, format="PNG")
