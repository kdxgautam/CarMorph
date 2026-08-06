"""Deterministic rough bumper placement for generative guidance."""

import numpy as np
from PIL import Image

from app.errors import PipelineError


def _box(mask: np.ndarray) -> tuple[int, int, int, int]:
    points = np.argwhere(mask >= 128)
    if not len(points):
        raise PipelineError("bumper_quality_check_failed", "Bumper placement mask is empty", 502)
    y1, x1 = points.min(axis=0)
    y2, x2 = points.max(axis=0) + 1
    return int(x1), int(y1), int(x2), int(y2)


def create_rough_composite(*, original: Image.Image, reference_rgba: Image.Image, reference_mask: np.ndarray, target_core_mask: np.ndarray, allowed_mask: np.ndarray) -> Image.Image:
    """Centre an aspect-preserving bumper crop and clamp it to allowed pixels."""

    shape = (original.height, original.width)
    if any(mask.shape != shape for mask in (reference_mask, target_core_mask, allowed_mask)):
        # Reference masks are intentionally reference-sized, not target-sized.
        if reference_mask.shape != (reference_rgba.height, reference_rgba.width) or any(mask.shape != shape for mask in (target_core_mask, allowed_mask)):
            raise PipelineError("bumper_quality_check_failed", "Bumper mask dimensions are invalid", 502)
    rx1, ry1, rx2, ry2 = _box(reference_mask)
    tx1, ty1, tx2, ty2 = _box(target_core_mask)
    crop = reference_rgba.convert("RGBA").crop((rx1, ry1, rx2, ry2))
    scale = min((tx2 - tx1) / crop.width, (ty2 - ty1) / crop.height)
    size = (max(1, round(crop.width * scale)), max(1, round(crop.height * scale)))
    crop = crop.resize(size, Image.Resampling.LANCZOS)
    x = tx1 + (tx2 - tx1 - crop.width) // 2
    y = ty1 + (ty2 - ty1 - crop.height) // 2
    overlay = Image.new("RGBA", original.size)
    overlay.alpha_composite(crop, (x, y))
    alpha = np.asarray(overlay.getchannel("A")).copy()
    alpha[allowed_mask < 128] = 0
    overlay.putalpha(Image.fromarray(alpha))
    result = Image.alpha_composite(original.convert("RGBA"), overlay).convert("RGB")
    pixels = np.asarray(result).copy()
    pixels[allowed_mask < 128] = np.asarray(original.convert("RGB"))[allowed_mask < 128]
    return Image.fromarray(pixels)
