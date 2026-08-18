"""Validated storage for user-supplied studio references."""

import hashlib
import shutil
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Literal

from PIL import Image, ImageOps
from pydantic import BaseModel, ConfigDict, Field

from app.errors import PipelineError
from app.modifications.schemas import VehicleIdentity

MAX_REFERENCE_BYTES = 20 * 1024 * 1024
MAX_REFERENCE_PIXELS = 40_000_000
REFERENCE_ID_LENGTH = 64


class StudioReferenceMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore")

    reference_asset_id: str
    kind: Literal["user"] = "user"
    title: str = ""
    normalized_image: str = "normalized.png"
    content_sha256: str
    width: int
    height: int


class StudioIdentityResponse(BaseModel):
    identity: VehicleIdentity
    reference_asset_ids: list[str] = Field(default_factory=list)


def _normalized_png(source: bytes, *, byte_limit: int) -> tuple[bytes, tuple[int, int]]:
    if not source or len(source) > byte_limit:
        raise PipelineError("invalid_studio_reference", "Reference image is empty or too large", 400)
    try:
        with Image.open(BytesIO(source)) as opened:
            if opened.width * opened.height > MAX_REFERENCE_PIXELS:
                raise ValueError("Reference image has too many pixels")
            opened.load()
            image = ImageOps.exif_transpose(opened).convert("RGB")
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        raise PipelineError("invalid_studio_reference", "Reference image is invalid", 400) from exc
    image.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    output = BytesIO()
    image.save(output, "PNG", optimize=True)
    return output.getvalue(), image.size


def store_studio_reference(
    *,
    directory: Path,
    source: bytes,
    kind: Literal["user"] = "user",
    title: str = "",
    byte_limit: int = MAX_REFERENCE_BYTES,
) -> StudioReferenceMetadata:
    canonical, (width, height) = _normalized_png(source, byte_limit=byte_limit)
    content_hash = hashlib.sha256(canonical).hexdigest()
    reference_id = hashlib.sha256(kind.encode() + b"\0" + canonical).hexdigest()
    parent = directory / "references" / "studio"
    final = parent / reference_id
    metadata = StudioReferenceMetadata(
        reference_asset_id=reference_id,
        kind=kind,
        title=title[:240],
        content_sha256=content_hash,
        width=width,
        height=height,
    )
    if (final / "normalized.png").is_file() and (final / "metadata.json").is_file():
        return StudioReferenceMetadata.model_validate_json(
            (final / "metadata.json").read_text(encoding="utf-8")
        )
    parent.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix=".studio-reference-", dir=parent))
    try:
        (work / "normalized.png").write_bytes(canonical)
        (work / "metadata.json").write_text(metadata.model_dump_json(indent=2), encoding="utf-8")
        if final.exists():
            shutil.rmtree(work)
        else:
            work.rename(final)
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise
    return metadata


def load_studio_reference(directory: Path, reference_id: str) -> tuple[Path, StudioReferenceMetadata]:
    if len(reference_id) != REFERENCE_ID_LENGTH or any(char not in "0123456789abcdef" for char in reference_id):
        raise PipelineError("studio_reference_not_found", "Studio reference was not found", 404)
    path = directory / "references" / "studio" / reference_id
    try:
        metadata = StudioReferenceMetadata.model_validate_json(
            (path / "metadata.json").read_text(encoding="utf-8")
        )
    except (OSError, ValueError) as exc:
        raise PipelineError("studio_reference_not_found", "Studio reference was not found", 404) from exc
    image = path / metadata.normalized_image
    if metadata.reference_asset_id != reference_id or not image.is_file():
        raise PipelineError("studio_reference_not_found", "Studio reference is incomplete", 404)
    return image, metadata
