"""Render result model and deterministic cache keys."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

from app.modifications.schemas import ModificationRequest, normalised_request_json

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
    payload = "|".join(parts)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
