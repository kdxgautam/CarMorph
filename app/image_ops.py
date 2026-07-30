import math
import warnings
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from app.errors import PipelineError

FORMATS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp"}
MIN_DIMENSION = 256
MAX_DIMENSION = 8192


def load_image(data: bytes) -> tuple[Image.Image, str]:
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
    output = BytesIO()
    image.save(output, "JPEG", quality=95, exif=b"", icc_profile=None)
    return output.getvalue()


def save_base_assets(image: Image.Image, output: Path) -> None:
    image.save(
        output / "original.webp",
        "WEBP",
        lossless=True,
        exif=b"",
        icc_profile=None,
    )
    image.convert("L").save(output / "luminance-map.png")


def polygons_to_mask(polygons: list, size: tuple[int, int]) -> np.ndarray:
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


def save_mask(mask: np.ndarray, path: Path) -> None:
    if not cv2.imwrite(str(path), mask):
        raise OSError(f"Could not write mask: {path.name}")


def dark_trim_mask(
    image: Image.Image,
    full_car: np.ndarray,
    car_box: tuple[int, int, int, int],
    kernel_size: int,
    feather_radius: int,
) -> np.ndarray:
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


def parse_colour(colour: str) -> tuple[str, tuple[int, int, int]]:
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


def recolour(image: Image.Image, body_mask: np.ndarray, colour: str) -> bytes:
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

    luminance = cv2.cvtColor(source, cv2.COLOR_RGB2GRAY).astype(np.float32)
    median_luminance = float(np.median(luminance[core]))
    detail = (
        np.clip(
            1 + (luminance / median_luminance - 1) * 0.4,
            0.55,
            1.25,
        )
        if median_luminance
        else np.ones_like(luminance)
    )
    recoloured = np.rint(
        np.clip(
            np.asarray(rgb, dtype=np.float32) * detail[:, :, None],
            0,
            255,
        )
    ).astype(np.uint8)

    alpha = body_mask.astype(np.float32)[:, :, None] / 255
    preview = np.rint(source * (1 - alpha) + recoloured * alpha).astype(np.uint8)
    output = BytesIO()
    Image.fromarray(preview).save(output, "PNG")
    return output.getvalue()
