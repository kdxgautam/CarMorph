import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from app.modifications.schemas import SurfaceEditRequest, normalised_request_json
from app.schemas import AssetBundle

RENDERER_CACHE_VERSION = "surface-render-1"


@dataclass(frozen=True)
class RenderResult:
    path: Path
    cached: bool
    renderer: str
    quality_status: str
    warnings: list[str] = field(default_factory=list)


class ModificationRenderer(Protocol):
    def render(
        self,
        *,
        directory: Path,
        metadata: AssetBundle,
        modification: SurfaceEditRequest,
    ) -> RenderResult:
        ...


class GenerativePartReplacementRenderer:
    def render(self, *args, **kwargs) -> RenderResult:
        raise NotImplementedError("Part replacement rendering is a future milestone")


def request_hash(
    modification: SurfaceEditRequest,
    *,
    renderer: str,
    provider: str = "",
    settings: str = "",
    pipeline_version: str = "legacy",
) -> str:
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
