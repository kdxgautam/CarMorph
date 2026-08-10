"""Small rim-reference storage and wheel-mask helpers."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.bumper_analysis.reference_preprocessor import (
    MAX_REFERENCE_BYTES,
    _largest_component,
    _plain_background_mask,
    _png_bytes,
)
from app.bumper_analysis.schemas import BumperReferenceResponse
from app.errors import PipelineError
from app.image_ops import load_image_rgba, save_mask
from app.schemas import AssetBundle

RIM_REFERENCE_VERSION = "rim-reference-1"


def rim_replacement_available(directory: Path, metadata: AssetBundle) -> bool:
    path = metadata.masks.get("wheels")
    mask = cv2.imread(str(directory / path), cv2.IMREAD_GRAYSCALE) if path else None
    return bool(mask is not None and mask.shape == (metadata.height, metadata.width) and np.any(mask >= 128))


def store_rim_reference(*, directory: Path, metadata: AssetBundle, source: bytes) -> BumperReferenceResponse:
    if not rim_replacement_available(directory, metadata):
        raise PipelineError("unsupported_rim_view", "Rim replacement needs detected wheels", 400)
    if not source or len(source) > MAX_REFERENCE_BYTES:
        raise PipelineError("invalid_rim_reference", "Reference image is empty or exceeds the 20 MB limit", 400)
    try:
        rgba, suffix = load_image_rgba(source)
    except (OSError, ValueError) as exc:
        raise PipelineError("invalid_rim_reference", str(exc), 400) from exc
    canonical = _png_bytes(rgba)
    reference_id = hashlib.sha256(RIM_REFERENCE_VERSION.encode() + b"\0" + canonical).hexdigest()
    final = directory / "references" / "rims" / reference_id
    if (final / "metadata.json").is_file() and (final / "normalized.png").is_file() and (final / "reference-mask.png").is_file():
        return BumperReferenceResponse(reference_asset_id=reference_id, width=rgba.width, height=rgba.height, has_alpha=True)
    alpha = np.asarray(rgba.getchannel("A"))
    mask = _largest_component(np.where(alpha > 0, 255, 0).astype(np.uint8)) if np.any(alpha < 255) and np.any(alpha > 0) else _plain_background_mask(rgba)
    if mask is None or not np.any(mask >= 128):
        raise PipelineError("rim_reference_segmentation_failed", "Could not isolate a usable rim in the reference image", 400)
    rgba.putalpha(Image.fromarray(mask))
    final.parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".rim-", dir=final.parent))
    try:
        (work / f"source.{suffix}").write_bytes(source)
        (work / "normalized.png").write_bytes(_png_bytes(rgba))
        save_mask(mask, work / "reference-mask.png")
        (work / "metadata.json").write_text(json.dumps({
            "reference_asset_id": reference_id,
            "content_sha256": hashlib.sha256(canonical).hexdigest(),
            "normalized_image": "normalized.png",
            "reference_mask": "reference-mask.png",
            "source_image": f"source.{suffix}",
            "processing_version": RIM_REFERENCE_VERSION,
        }, indent=2), encoding="utf-8")
        if final.exists():
            shutil.rmtree(work, ignore_errors=True)
        else:
            work.rename(final)
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return BumperReferenceResponse(reference_asset_id=reference_id, width=rgba.width, height=rgba.height, has_alpha=True)


def wheel_mask(directory: Path, metadata: AssetBundle) -> np.ndarray:
    if not rim_replacement_available(directory, metadata):
        raise PipelineError("missing_rim_mask", "Rim replacement needs a usable wheel mask", 400)
    mask = cv2.imread(str(directory / metadata.masks["wheels"]), cv2.IMREAD_GRAYSCALE)
    return np.where(mask >= 128, 255, 0).astype(np.uint8)


def wheel_reference(directory: Path, reference_id: str) -> tuple[Path, dict]:
    if not reference_id or not all(char in "0123456789abcdef" for char in reference_id) or len(reference_id) != 64:
        raise PipelineError("rim_reference_not_found", "Rim reference was not found", 404)
    path = directory / "references" / "rims" / reference_id
    try:
        report = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PipelineError("rim_reference_not_found", "Rim reference was not found", 404) from exc
    if not all((path / name).is_file() for name in ("normalized.png", "reference-mask.png")):
        raise PipelineError("rim_reference_not_found", "Rim reference is incomplete", 404)
    return path, report
