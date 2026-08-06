"""Build strict bumper-only edit masks from an existing processed asset."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.errors import PipelineError
from app.schemas import AssetBundle


@dataclass(frozen=True)
class BumperMasks:
    core: np.ndarray
    blend: np.ndarray
    allowed: np.ndarray
    protected: np.ndarray


def bumper_replacement_available(directory: Path, metadata: AssetBundle) -> bool:
    """Return a capability flag without changing immutable stored metadata."""

    if metadata.view not in {"front", "rear"}:
        return False
    path = metadata.masks.get("bumper")
    if not path:
        return False
    mask = cv2.imread(str(directory / path), cv2.IMREAD_GRAYSCALE)
    return bool(mask is not None and mask.shape == (metadata.height, metadata.width) and np.any(mask >= 128))


def _load(directory: Path, metadata: AssetBundle, key: str, required: bool = False) -> np.ndarray:
    path = metadata.masks.get(key)
    if not path:
        if required:
            raise PipelineError("missing_bumper_mask", f"{key} mask is missing", 400)
        return np.zeros((metadata.height, metadata.width), np.uint8)
    mask = cv2.imread(str(directory / path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise PipelineError("missing_bumper_mask", f"{key} mask is missing", 400)
    if mask.shape != (metadata.height, metadata.width):
        raise PipelineError("missing_bumper_mask", f"{key} mask dimensions do not match the image", 500)
    return np.where(mask >= 128, 255, 0).astype(np.uint8)


def build_bumper_masks(directory: Path, metadata: AssetBundle) -> BumperMasks:
    """Keep the edit strictly on the visible bumper and blending boundary."""

    if metadata.view not in {"front", "rear"}:
        raise PipelineError("unsupported_bumper_view", "Bumper replacement supports front and rear views only", 400)
    bumper = _load(directory, metadata, "bumper", True) > 0
    full_car = _load(directory, metadata, "full_car", True) > 0
    if not np.any(bumper):
        raise PipelineError("missing_bumper_mask", "Bumper mask is empty", 400)
    explicit = np.zeros_like(bumper)
    for key in ("lights", "plate", "grille", "wheels", "windows", "badge"):
        explicit |= _load(directory, metadata, key) > 0
    raw_core = bumper & full_car
    core = raw_core & ~explicit
    if not np.any(core):
        raise PipelineError("bumper_quality_check_failed", "Protected regions cover the bumper", 502)
    radius = max(2, min(12, round(min(metadata.width, metadata.height) * 0.005)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    blend = (cv2.dilate(np.where(core, 255, 0).astype(np.uint8), kernel) > 0) & full_car & ~explicit
    allowed = core | blend
    old_protected = _load(directory, metadata, "protected_mask") > 0
    protected = explicit | (old_protected & ~bumper)
    if not np.any(allowed) or np.any(core & protected) or np.any(allowed & explicit):
        raise PipelineError("bumper_quality_check_failed", "Bumper masks violate protection constraints", 502)
    return BumperMasks(*(
        np.where(mask, 255, 0).astype(np.uint8)
        for mask in (core, blend, allowed, protected)
    ))
