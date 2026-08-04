"""Image validation, mask operations, and deterministic LAB recolouring."""

import math
import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.errors import PipelineError

FORMATS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}
MIN_DIMENSION = 256
MAX_DIMENSION = 8192
PAINTABILITY_RULES_VERSION = "paintability-1"


@dataclass(frozen=True)
class PaintabilityMasks:
    """Legacy editable/protected/uncertain mask bundle."""

    editable: np.ndarray
    protected: np.ndarray
    uncertain: np.ndarray
    report: dict


def load_image(data: bytes) -> tuple[Image.Image, str]:
    """Decode one supported still image and enforce safe dimension limits."""

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data), formats=list(FORMATS)) as opened:
                if getattr(opened, "n_frames", 1) != 1:
                    raise ValueError("Animated or multi-frame images are not supported")
                opened.load()
                image_format = opened.format
                image = ImageOps.exif_transpose(opened).convert("RGB")
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise ValueError("Image has too many pixels") from exc

    if image_format not in FORMATS:
        raise ValueError("Only JPEG, PNG, and WebP images are supported")
    if not (MIN_DIMENSION <= min(image.size) and max(image.size) <= MAX_DIMENSION):
        raise ValueError(
            f"Image dimensions must be between {MIN_DIMENSION} and "
            f"{MAX_DIMENSION} pixels"
        )
    return image, FORMATS[image_format]


def encode_jpeg(image: Image.Image) -> bytes:
    """Encode a normalized provider input without metadata."""

    output = BytesIO()
    image.save(output, "JPEG", quality=95, exif=b"", icc_profile=None)
    return output.getvalue()


def save_base_assets(image: Image.Image, output: Path) -> None:
    """Persist a lossless original and grayscale luminance reference."""

    image.save(
        output / "original.webp",
        "WEBP",
        lossless=True,
        exif=b"",
        icc_profile=None,
    )
    image.convert("L").save(output / "luminance-map.png")


def polygons_to_mask(polygons: list, size: tuple[int, int]) -> np.ndarray:
    """Rasterize validated provider polygons into one binary image-sized mask."""

    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    if not polygons:
        raise PipelineError(
            "invalid_sam_response", "SAM 2 returned an empty mask", 502
        )

    valid_polygons = 0
    for polygon in polygons:
        try:
            points = np.asarray(polygon, dtype=np.float32)
        except (TypeError, ValueError):
            continue
        if (
            points.ndim != 2
            or points.shape[0] < 3
            or points.shape[1] != 2
            or not np.isfinite(points).all()
        ):
            continue
        points[:, 0] = np.clip(points[:, 0], 0, size[0] - 1)
        points[:, 1] = np.clip(points[:, 1], 0, size[1] - 1)
        cv2.fillPoly(mask, [np.rint(points).astype(np.int32)], 255)
        valid_polygons += 1

    if not valid_polygons or not np.any(mask):
        raise PipelineError(
            "invalid_sam_response", "SAM 2 returned no usable mask polygons", 502
        )
    return mask


def clean_mask(
    mask: np.ndarray,
    size: tuple[int, int],
    kernel_size: int,
    feather_radius: int,
) -> np.ndarray:
    """Normalize, denoise, close gaps, and optionally soften a semantic mask."""

    if mask.ndim != 2:
        raise PipelineError("invalid_sam_response", "Mask must be two-dimensional", 502)

    expected_width, expected_height = size
    if mask.shape != (expected_height, expected_width):
        source_ratio = mask.shape[1] / mask.shape[0]
        expected_ratio = expected_width / expected_height
        if not math.isclose(source_ratio, expected_ratio, rel_tol=0.01):
            raise PipelineError(
                "mask_dimension_mismatch",
                "Mask and image aspect ratios do not match",
                502,
            )
        mask = cv2.resize(
            mask, (expected_width, expected_height), interpolation=cv2.INTER_NEAREST
        )

    binary = np.where(mask >= 128, 255, 0).astype(np.uint8)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    if feather_radius:
        blur_size = feather_radius * 2 + 1
        binary = cv2.GaussianBlur(binary, (blur_size, blur_size), 0)
    return binary


def combine_masks(masks: list[np.ndarray]) -> np.ndarray:
    """Union same-sized masks while rejecting incomplete provider output."""

    if not masks:
        raise PipelineError("missing_masks", "No masks were supplied")
    shape = masks[0].shape
    if any(mask.shape != shape for mask in masks):
        raise PipelineError(
            "mask_dimension_mismatch", "Stored masks have different dimensions", 500
        )
    return np.maximum.reduce(masks)


def build_body_mask(
    full_car: np.ndarray,
    non_paintable: list[np.ndarray],
    kernel_size: int,
    feather_radius: int,
) -> np.ndarray:
    """Build the legacy body mask by subtracting non-paintable parts."""

    if any(mask.shape != full_car.shape for mask in non_paintable):
        raise PipelineError(
            "mask_dimension_mismatch",
            "Full-car and non-paintable masks have different dimensions",
            502,
        )
    excluded = (
        np.maximum.reduce([mask >= 128 for mask in non_paintable])
        if non_paintable
        else np.zeros_like(full_car, dtype=bool)
    )
    body = np.where((full_car >= 128) & ~excluded, 255, 0).astype(np.uint8)
    body = clean_mask(
        body,
        (body.shape[1], body.shape[0]),
        kernel_size,
        feather_radius,
    )
    if not np.any(body >= 128):
        raise PipelineError(
            "missing_masks", "The paintable-body mask is empty after subtraction"
        )
    return body


def build_paintability_masks(
    full_car: np.ndarray,
    protected_parts: list[np.ndarray],
    uncertain_parts: list[np.ndarray],
    kernel_size: int,
    feather_radius: int,
) -> PaintabilityMasks:
    """Build disjoint legacy masks with protected and uncertain precedence."""

    if any(mask.shape != full_car.shape for mask in [*protected_parts, *uncertain_parts]):
        raise PipelineError(
            "mask_dimension_mismatch",
            "Full-car and paintability masks have different dimensions",
            502,
        )
    protected = (
        np.maximum.reduce([mask >= 128 for mask in protected_parts])
        if protected_parts
        else np.zeros_like(full_car, dtype=bool)
    )
    uncertain = (
        np.maximum.reduce([mask >= 128 for mask in uncertain_parts])
        if uncertain_parts
        else np.zeros_like(full_car, dtype=bool)
    )
    uncertain &= ~protected
    editable = (full_car >= 128) & ~protected & ~uncertain

    editable_mask = clean_mask(
        np.where(editable, 255, 0).astype(np.uint8),
        (full_car.shape[1], full_car.shape[0]),
        kernel_size,
        feather_radius,
    )
    protected_mask = np.where(protected, 255, 0).astype(np.uint8)
    uncertain_mask = np.where(uncertain, 255, 0).astype(np.uint8)
    editable_mask[(protected_mask >= 128) | (uncertain_mask >= 128)] = 0
    if not np.any(editable_mask >= 128):
        raise PipelineError(
            "missing_masks", "The editable mask is empty after paintability rules"
        )

    total = full_car.size
    report = {
        "editable_ratio": round(float(np.count_nonzero(editable_mask >= 128) / total), 4),
        "protected_ratio": round(float(np.count_nonzero(protected_mask >= 128) / total), 4),
        "uncertain_ratio": round(float(np.count_nonzero(uncertain_mask >= 128) / total), 4),
        "warnings": [],
        "rules_version": PAINTABILITY_RULES_VERSION,
    }
    return PaintabilityMasks(editable_mask, protected_mask, uncertain_mask, report)


def save_mask(mask: np.ndarray, path: Path) -> None:
    """Write one grayscale mask and fail if OpenCV cannot persist it."""

    if not cv2.imwrite(str(path), mask):
        raise OSError(f"Could not write mask: {path.name}")


def dark_trim_mask(
    image: Image.Image,
    full_car: np.ndarray,
    car_box: tuple[int, int, int, int],
    kernel_size: int,
    feather_radius: int,
) -> np.ndarray:
    """Find dark neutral pixels only in the lower portion of the detected car."""

    source = np.asarray(image.convert("RGB"))
    if full_car.shape != source.shape[:2]:
        raise PipelineError(
            "mask_dimension_mismatch",
            "Full-car mask dimensions do not match the original image",
            502,
        )

    luminance = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)[:, :, 0]
    car_pixels = luminance[full_car >= 128]
    threshold = min(80, float(np.median(car_pixels)) * 0.6)
    dark = np.where(
        (full_car >= 128) & (luminance < threshold), 255, 0
    ).astype(np.uint8)
    _, y1, _, y2 = car_box
    dark[: round(y1 + (y2 - y1) * 0.45)] = 0
    return clean_mask(
        dark,
        (source.shape[1], source.shape[0]),
        kernel_size,
        feather_radius,
    )


def uncertain_dark_region_mask(
    image: Image.Image,
    full_car: np.ndarray,
    car_box: tuple[int, int, int, int],
    kernel_size: int,
    feather_radius: int,
) -> np.ndarray:
    """Backward-compatible alias for lower-car dark-region detection."""

    return dark_trim_mask(image, full_car, car_box, kernel_size, feather_radius)


def parse_colour(colour: str) -> tuple[str, tuple[int, int, int]]:
    """Validate a six-digit hex colour and return normalized text plus RGB."""

    value = colour.removeprefix("#")
    if len(value) != 6 or any(
        character not in "0123456789abcdefABCDEF" for character in value
    ):
        raise PipelineError(
            "invalid_colour", "Colour must be a six-digit hex value", 400
        )
    try:
        rgb = tuple(bytes.fromhex(value))
    except ValueError as exc:
        raise PipelineError(
            "invalid_colour", "Colour must be a six-digit hex value", 400
        ) from exc
    return value.lower(), rgb


def recolour(
    image: Image.Image,
    body_mask: np.ndarray,
    colour: str,
    finish: str = "glossy",
) -> bytes:
    """Transfer target paint chroma while preserving photographed luminance.

    Percentile shading maps retain shadows and neutral glare. Source chroma
    residuals carry reflections into the new paint, and the final alpha blend
    is strictly inside the supplied editable mask.
    """

    _, rgb = parse_colour(colour)

    source = np.asarray(image.convert("RGB"))
    if body_mask.shape != source.shape[:2]:
        raise PipelineError(
            "mask_dimension_mismatch",
            "Body mask dimensions do not match the original image",
            500,
        )

    core = body_mask >= 128
    if not np.any(core):
        raise PipelineError("missing_masks", "Paintable-body mask is empty", 500)

    # LAB separates photographed luminance from paint chroma, so panel seams and
    # lighting survive instead of being flattened by an RGB multiplier.
    source_lab = cv2.cvtColor(
        source.astype(np.float32) / 255, cv2.COLOR_RGB2LAB
    )
    target_lab = cv2.cvtColor(
        np.asarray(rgb, np.float32).reshape(1, 1, 3) / 255,
        cv2.COLOR_RGB2LAB,
    )[0, 0]
    anchor = cv2.erode(
        np.where(core, 255, 0).astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    ) >= 128
    if not np.any(anchor):
        anchor = core
    lightness = source_lab[:, :, 0]
    low, lower_mid, _, upper_mid, high = np.percentile(
        lightness[anchor], (5, 20, 50, 80, 98)
    )
    diffuse = anchor & (lightness >= lower_mid) & (lightness <= upper_mid)
    if not np.any(diffuse):
        diffuse = anchor
    source_base = np.median(source_lab[:, :, 1:][diffuse], axis=0)
    source_chroma = float(np.linalg.norm(source_base))
    shadow = (
        np.clip((lightness - low) / (lower_mid - low), 0, 1)
        if lower_mid - low > 1
        else np.ones_like(lightness)
    )
    highlight = (
        np.clip((lightness - upper_mid) / (high - upper_mid), 0, 1)
        if high - upper_mid > 1
        else np.zeros_like(lightness)
    )
    reflection_gain, shadow_floor, highlight_rolloff = {
        "glossy": (0.55, 0.75, 0.8),
        "matte": (0.25, 0.82, 0.35),
        "metallic": (0.65, 0.72, 0.65),
    }.get(finish, (0.55, 0.75, 0.8))
    source_ab = source_lab[:, :, 1:]
    if source_chroma > 8:
        source_direction = source_base / source_chroma
        projection = np.sum(source_ab * source_direction, axis=2)
        # A floor prevents desaturated shadow reflections from retaining the old
        # paint colour while highlights still roll off below.
        paint_strength = np.clip(projection / source_chroma, 0.65, 1.25)
        reflection = source_ab - projection[:, :, None] * source_direction
    else:
        paint_strength = np.ones_like(lightness)
        reflection = source_ab - source_base
    paint_strength *= (shadow_floor + (1 - shadow_floor) * shadow) * (
        1 - highlight_rolloff * highlight
    )
    result_lab = source_lab.copy()
    result_lab[:, :, 1:] = (
        target_lab[1:] * paint_strength[:, :, None]
        + reflection * reflection_gain
    )
    recoloured = np.rint(
        np.clip(
            cv2.cvtColor(result_lab, cv2.COLOR_LAB2RGB),
            0,
            1,
        )
        * 255
    ).astype(np.uint8)

    # Feather only pixels already inside the editable mask; protected/background
    # pixels are never used as blend destinations.
    coverage = body_mask.copy()
    binary_coverage = np.where(core, 255, 0).astype(np.uint8)
    softened = cv2.GaussianBlur(binary_coverage, (3, 3), 0)
    coverage[core] = np.minimum(
        coverage[core], np.maximum(softened[core], 128)
    )
    alpha = coverage.astype(np.float32)[:, :, None] / 255
    preview = np.rint(source * (1 - alpha) + recoloured * alpha).astype(np.uint8)
    output = BytesIO()
    Image.fromarray(preview).save(output, "PNG")
    return output.getvalue()
