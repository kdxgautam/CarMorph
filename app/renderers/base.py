"""Renderer protocol, result model, and deterministic cache keys."""

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from app.modifications.schemas import SurfaceEditRequest, normalised_request_json
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


class ModificationRenderer(Protocol):
    """Common keyword-only interface implemented by surface renderers."""

    def render(
        self,
        *,
        directory: Path,
        metadata: AssetBundle,
        modification: SurfaceEditRequest,
    ) -> RenderResult:
        """Render one validated modification against a prepared asset."""

        ...


class GenerativePartReplacementRenderer:
    """Explicit placeholder for out-of-scope physical part replacement."""

    def render(self, *args, **kwargs) -> RenderResult:
        """Reject use until a safe part-replacement milestone exists."""

        raise NotImplementedError("Part replacement rendering is a future milestone")


def request_hash(
    modification: SurfaceEditRequest,
    *,
    renderer: str,
    provider: str = "",
    settings: str = "",
    pipeline_version: str = "legacy",
) -> str:
    """Create a stable render cache key from request, provider, and versions."""

    payload = "|".join(
        [
            RENDERER_CACHE_VERSION,
            renderer,
            provider,
            settings,
            pipeline_version,
            normalised_request_json(modification),
        ]
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
