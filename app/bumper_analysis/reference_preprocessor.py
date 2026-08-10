"""Validate, isolate, and atomically store one bumper reference image."""

import hashlib
import shutil
import tempfile
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.bumper_analysis.schemas import BumperReferenceReport, BumperReferenceResponse
from app.config import Settings
from app.errors import PipelineError
from app.image_ops import clean_mask, encode_jpeg, load_image_rgba, polygons_to_mask, save_mask
from app.roboflow import segment_concepts
from app.schemas import AssetBundle

REFERENCE_PROCESSING_VERSION = "bumper-reference-1"
MAX_REFERENCE_BYTES = 20 * 1024 * 1024


def _png_bytes(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, "PNG", exif=b"", icc_profile=None)
    return output.getvalue()


def _largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return np.zeros_like(mask)
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == index, 255, 0).astype(np.uint8)


def _plain_background_mask(image: Image.Image) -> np.ndarray | None:
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]))
    background = np.median(border, axis=0)
    border_noise = float(np.percentile(np.linalg.norm(border - background, axis=1), 90))
    if border_noise > 30:
        return None
    distance = np.linalg.norm(rgb - background, axis=2)
    mask = np.where(distance > max(22, border_noise + 12), 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = _largest_component(mask)
    coverage = float(np.mean(mask >= 128))
    return mask if 0.01 <= coverage <= 0.9 else None


def _complete_reference(path: Path) -> BumperReferenceReport | None:
    try:
        report = BumperReferenceReport.model_validate_json(
            (path / "metadata.json").read_text(encoding="utf-8")
        )
        if not all((path / item).is_file() for item in (
            report.normalized_image, report.reference_mask, report.source_image
        )):
            return None
        with Image.open(path / report.normalized_image) as image:
            if image.size != (report.normalized_width, report.normalized_height):
                return None
        mask = cv2.imread(str(path / report.reference_mask), cv2.IMREAD_GRAYSCALE)
        if mask is None or mask.shape != (report.normalized_height, report.normalized_width):
            return None
        return report if np.any(mask >= 128) else None
    except (OSError, ValueError):
        return None


def store_bumper_reference(
    *, directory: Path, metadata: AssetBundle, source: bytes, settings: Settings | None = None
) -> BumperReferenceResponse:
    """Persist a reusable isolated bumper, using source alpha before SAM3."""

    if metadata.view not in {"front", "rear"}:
        raise PipelineError("unsupported_bumper_view", "Bumper replacement supports front and rear views only", 400)
    if not source or len(source) > MAX_REFERENCE_BYTES:
        raise PipelineError("invalid_bumper_reference", "Reference image is empty or exceeds the 20 MB limit", 400)
    try:
        rgba, suffix = load_image_rgba(source)
    except (OSError, ValueError) as exc:
        raise PipelineError("invalid_bumper_reference", str(exc), 400) from exc

    canonical = _png_bytes(rgba)
    content_hash = hashlib.sha256(canonical).hexdigest()
    reference_id = hashlib.sha256(
        REFERENCE_PROCESSING_VERSION.encode() + b"\0" + canonical
    ).hexdigest()
    root = directory / "references" / "bumpers"
    final = root / reference_id
    cached = _complete_reference(final)
    if cached:
        return BumperReferenceResponse(
            reference_asset_id=reference_id,
            width=cached.normalized_width,
            height=cached.normalized_height,
            has_alpha=cached.has_alpha,
        )

    alpha = np.asarray(rgba.getchannel("A"))
    useful_alpha = bool(np.any(alpha < 255) and np.any(alpha > 0))
    try:
        if useful_alpha:
            mask = np.where(alpha > 0, 255, 0).astype(np.uint8)
            mask = _largest_component(mask)
            method = "source_alpha"
        else:
            mask = _plain_background_mask(rgba)
            if mask is not None:
                method = "plain_background"
            else:
                try:
                    settings = settings or Settings.from_env()
                    polygons = segment_concepts(
                        encode_jpeg(rgba.convert("RGB")),
                        [f"{metadata.view} car bumper"],
                        settings,
                    )[0]
                    mask = polygons_to_mask(polygons, rgba.size)
                    mask = clean_mask(mask, rgba.size, settings.mask_kernel_size, 0)
                    mask = _largest_component(np.where(mask >= 128, 255, 0).astype(np.uint8))
                except PipelineError as exc:
                    if exc.code == "configuration_error":
                        raise
                    raise PipelineError(
                        "bumper_reference_segmentation_failed",
                        "Could not isolate a usable bumper in the reference image",
                        400,
                    ) from exc
                method = "sam3"
            rgba.putalpha(Image.fromarray(mask))
        if not np.any(mask >= 128):
            raise PipelineError(
                "bumper_reference_segmentation_failed",
                "Could not isolate a usable bumper in the reference image",
                400,
            )

        root.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=".bumper-", dir=root))
        try:
            (work / f"source.{suffix}").write_bytes(source)
            normalized = _png_bytes(rgba)
            (work / "normalized.png").write_bytes(normalized)
            save_mask(mask, work / "reference-mask.png")
            report = BumperReferenceReport(
                reference_asset_id=reference_id,
                content_sha256=content_hash,
                source_width=rgba.width,
                source_height=rgba.height,
                normalized_width=rgba.width,
                normalized_height=rgba.height,
                has_alpha=useful_alpha,
                segmentation_method=method,
                mask_coverage_ratio=round(float(np.mean(mask >= 128)), 4),
                processing_version=REFERENCE_PROCESSING_VERSION,
                source_image=f"source.{suffix}",
            )
            (work / "metadata.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")
            if _complete_reference(final):
                shutil.rmtree(work, ignore_errors=True)
            else:
                try:
                    work.rename(final)
                except FileExistsError:
                    shutil.rmtree(work, ignore_errors=True)
            final_report = _complete_reference(final)
            if not final_report:
                raise PipelineError("invalid_bumper_reference", "Reference storage could not be validated", 500)
        except Exception:
            shutil.rmtree(work, ignore_errors=True)
            raise
    except PipelineError:
        raise
    except OSError as exc:
        raise PipelineError("invalid_bumper_reference", "Reference storage failed", 500) from exc
    return BumperReferenceResponse(
        reference_asset_id=reference_id,
        width=rgba.width,
        height=rgba.height,
        has_alpha=useful_alpha,
    )
