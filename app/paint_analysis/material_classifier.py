"""Conservative semantic and appearance-based material classification."""

import cv2
import numpy as np
from PIL import Image

from app.paint_analysis.schemas import MaterialType


SEMANTIC_MATERIALS = {
    "windows": MaterialType.GLASS,
    "lights": MaterialType.LIGHT_LENS,
    "wheels": MaterialType.METAL,
    "plate": MaterialType.UNKNOWN,
    "grille": MaterialType.MATTE_PLASTIC,
    "pillars": MaterialType.GLOSSY_PLASTIC,
}


def classify_material(
    image: Image.Image, mask: np.ndarray, part_type: str
) -> tuple[MaterialType, float, list[str]]:
    """Classify one region using semantic precedence and appearance statistics.

    The return tuple contains material, confidence, and stable reason codes for
    persisted diagnostics. Empty regions remain unknown rather than paintable.
    """

    # Reliable part identity outranks coincidental colour similarity (for
    # example, body-colour reflections in glass).
    if part_type in SEMANTIC_MATERIALS:
        return SEMANTIC_MATERIALS[part_type], 0.98, ["semantic_part_protection"]
    pixels = np.asarray(image.convert("RGB"))[mask >= 128]
    if not len(pixels):
        return MaterialType.UNKNOWN, 0, ["empty_region"]
    hsv = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2HSV).reshape(-1, 3)
    grey = cv2.cvtColor(pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY).ravel()
    saturation = float(np.median(hsv[:, 1]))
    brightness = float(np.median(grey))
    highlight_ratio = float(np.mean(grey > 235))
    spread = float(np.percentile(grey, 90) - np.percentile(grey, 10))
    if saturation < 28 and highlight_ratio > 0.08 and spread > 70:
        return MaterialType.CHROME, 0.82, ["low_saturation_sharp_highlights"]
    if brightness < 55 and saturation < 75:
        material = MaterialType.GLOSSY_PLASTIC if spread > 45 else MaterialType.MATTE_PLASTIC
        return material, 0.68, ["dark_neutral_material"]
    return MaterialType.PAINTED_SURFACE, 0.72, ["paint_like_colour_distribution"]
