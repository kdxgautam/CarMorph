import cv2
import numpy as np
from PIL import Image

from app.paint_analysis.schemas import PaintGroup


COLOURS = {
    PaintGroup.MAIN_BODY_PAINT: (40, 190, 80),
    PaintGroup.SECONDARY_BODY_PAINT: (240, 160, 30),
    PaintGroup.CONTRAST_ROOF_PAINT: (130, 60, 220),
    PaintGroup.UNKNOWN: (240, 210, 20),
}
PROTECTED_COLOUR = (220, 50, 50)


def paint_group_overlay(
    image: Image.Image, group_masks: dict[PaintGroup, np.ndarray]
) -> Image.Image:
    source = np.asarray(image.convert("RGB")).copy()
    tint = source.copy()
    for group, mask in group_masks.items():
        colour = COLOURS.get(group, PROTECTED_COLOUR)
        tint[mask >= 128] = colour
    return Image.fromarray(cv2.addWeighted(source, 0.55, tint, 0.45, 0))


def anchor_overlay(image: Image.Image, anchors: np.ndarray) -> Image.Image:
    source = np.asarray(image.convert("RGB")).copy()
    tint = source.copy()
    tint[anchors >= 128] = (30, 220, 220)
    return Image.fromarray(cv2.addWeighted(source, 0.55, tint, 0.45, 0))


def surface_completion_overlay(
    image: Image.Image,
    safe_candidate: np.ndarray,
    hard_protected: np.ndarray,
    seeds: np.ndarray,
    main_body: np.ndarray,
) -> Image.Image:
    source = np.asarray(image.convert("RGB")).copy()
    tint = source.copy()
    tint[safe_candidate >= 128] = (230, 190, 30)
    tint[main_body >= 128] = (40, 190, 80)
    tint[seeds >= 128] = (20, 220, 240)
    tint[hard_protected >= 128] = (220, 50, 50)
    return Image.fromarray(cv2.addWeighted(source, 0.55, tint, 0.45, 0))
