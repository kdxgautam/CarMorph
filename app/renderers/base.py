"""Render result model and deterministic cache keys."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image

from app.errors import PipelineError
from app.modifications.schemas import ModificationRequest, normalised_request_json
from app.schemas import AssetBundle

RENDERER_CACHE_VERSION = "surface-render-2"


@dataclass(frozen=True)
class RenderResult:
    """Persisted render path plus cache and quality metadata for HTTP headers."""

    path: Path
    cached: bool
    renderer: str
    quality_status: str
    warnings: list[str] = field(default_factory=list)


def request_hash(
    modification: ModificationRequest,
    *,
    renderer: str,
    provider: str = "",
    settings: str = "",
    pipeline_version: str = "legacy",
    reference_content_hash: str = "",
    provider_model: str = "",
    renderer_version: str = "",
    base_image_hash: str = "",
) -> str:
    """Create a stable render cache key from request, provider, and versions."""

    parts = [
        RENDERER_CACHE_VERSION,
        renderer,
        provider,
        settings,
        pipeline_version,
        normalised_request_json(modification),
    ]
    # Keep existing deterministic paint cache keys unchanged.
    if reference_content_hash or provider_model or renderer_version:
        parts.extend([reference_content_hash, provider_model, renderer_version])
    if base_image_hash:
        parts.append(base_image_hash)
    payload = "|".join(parts)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def canonical_image_hash(image: Image.Image) -> str:
    """Hash exact RGB pixels plus dimensions for chained render cache identity."""

    rgb = image.convert("RGB")
    return hashlib.sha256(
        f"{rgb.width}x{rgb.height}\0".encode() + rgb.tobytes()
    ).hexdigest()


def render_base_image(
    *,
    directory: Path,
    metadata: AssetBundle,
    base_image: Image.Image | None = None,
) -> tuple[Image.Image, str]:
    """Load the immutable original, or validate a session-local working image."""

    if base_image is None:
        with Image.open(directory / metadata.original_image) as opened:
            return opened.convert("RGB"), ""
    image = base_image.convert("RGB")
    if image.size != (metadata.width, metadata.height):
        raise PipelineError(
            "invalid_modification",
            "Chained base image dimensions do not match the processed asset",
            400,
        )
    return image, canonical_image_hash(image)
