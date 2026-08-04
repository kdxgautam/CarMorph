"""FastAPI routes for asset preparation and constrained customisation."""

import os
import re
from io import BytesIO
from pathlib import Path
from typing import Annotated

import cv2
from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from PIL import Image

from app.config import Settings
from app.errors import PipelineError
from app.flux import FluxSettings, render_flux
from app.image_ops import recolour
from app.modifications.instructions import merge_instruction
from app.modifications.planner import choose_renderer
from app.modifications.schemas import parse_modification
from app.pipeline import process_view
from app.schemas import AssetBundle, ViewSelection

app = FastAPI(title="Car Customisation API", version="0.1.0")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ASSET_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@app.exception_handler(PipelineError)
def pipeline_error_handler(_: Request, exc: PipelineError) -> JSONResponse:
    """Expose expected pipeline failures through the stable error envelope."""

    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "detail": exc.detail}},
    )


@app.exception_handler(RequestValidationError)
def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Normalize FastAPI/Pydantic request errors for API clients."""

    return JSONResponse(
        status_code=422,
        content={"error": {"code": "invalid_request", "detail": str(exc)}},
    )


def _asset(asset_id: str) -> tuple[Path, AssetBundle]:
    """Resolve and validate one stored asset without allowing path traversal."""

    # Validate before joining the user-controlled identifier to storage paths.
    if not ASSET_ID_PATTERN.fullmatch(asset_id):
        raise PipelineError("asset_not_found", "Asset was not found", 404)
    directory = Path(os.getenv("STORAGE_ROOT", "data/processed")).resolve() / asset_id
    try:
        metadata = AssetBundle.model_validate_json(
            (directory / "metadata.json").read_text(encoding="utf-8")
        )
    except FileNotFoundError as exc:
        raise PipelineError("asset_not_found", "Asset was not found", 404) from exc
    except (OSError, ValueError) as exc:
        raise PipelineError(
            "invalid_stored_assets", "Stored asset metadata is invalid", 500
        ) from exc
    return directory, metadata


@app.post("/cars", response_model=AssetBundle, status_code=201)
def upload_car(
    image: Annotated[UploadFile, File()],
    view: Annotated[ViewSelection, Form()] = "front",
) -> AssetBundle:
    """Validate an upload and prepare reusable masks for the requested view."""

    data = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise PipelineError(
            "upload_too_large", "Image exceeds the 20 MB limit", 413
        )
    if not data:
        raise PipelineError("invalid_image", "Uploaded image is empty", 400)
    return process_view(data, Settings.from_env(), view)


@app.get("/cars/{asset_id}", response_model=AssetBundle)
def get_car(asset_id: str) -> AssetBundle:
    """Return persisted metadata for one prepared asset."""

    return _asset(asset_id)[1]


@app.get("/cars/{asset_id}/assets/{asset_path:path}")
def get_asset(asset_id: str, asset_path: str) -> FileResponse:
    """Serve one explicitly listed file contained by an asset directory."""

    directory, metadata = _asset(asset_id)
    allowed = {
        metadata.source_image,
        metadata.original_image,
        metadata.luminance_map,
        *metadata.masks.values(),
    }
    if asset_path not in allowed:
        raise PipelineError("asset_not_found", "Asset was not found", 404)
    path = directory / asset_path
    if not path.is_file():
        raise PipelineError("missing_masks", "A stored asset is missing", 500)
    return FileResponse(path)


@app.get("/cars/{asset_id}/preview")
def preview(
    asset_id: str,
    colour: Annotated[str, Query(description="Six-digit RGB hex colour")] = "e63946",
) -> StreamingResponse:
    """Return the backward-compatible deterministic body-colour preview."""

    directory, metadata = _asset(asset_id)
    body_path = metadata.masks.get("paintable_body")
    if not body_path:
        raise PipelineError("missing_masks", "Paintable-body mask is missing", 500)
    try:
        with Image.open(directory / metadata.original_image) as opened:
            image = opened.convert("RGB")
        body_mask = cv2.imread(str(directory / body_path), cv2.IMREAD_GRAYSCALE)
    except OSError as exc:
        raise PipelineError("missing_masks", "A preview asset is missing", 500) from exc
    if body_mask is None:
        raise PipelineError("missing_masks", "Paintable-body mask is missing", 500)
    return StreamingResponse(
        BytesIO(recolour(image, body_mask, colour)),
        media_type="image/png",
        headers={"Cache-Control": "no-store"},
    )


@app.post("/cars/{asset_id}/render")
def render(
    asset_id: str,
    colour: Annotated[str, Query(description="Six-digit RGB hex colour")] = "e63946",
) -> FileResponse:
    """Return the backward-compatible cached FLUX colour render."""

    directory, metadata = _asset(asset_id)
    path, cached = render_flux(directory, metadata, colour, FluxSettings.from_env())
    return FileResponse(
        path,
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
            "X-Render-Cached": str(cached).lower(),
        },
    )


@app.post("/cars/{asset_id}/customise")
async def customise(asset_id: str, request: Request) -> FileResponse:
    """Validate a structured edit, select a renderer, and return its PNG."""

    directory, metadata = _asset(asset_id)
    try:
        body = await request.json()
    except ValueError as exc:
        raise PipelineError("invalid_request", "Request body must be JSON") from exc
    modification = merge_instruction(parse_modification(body))
    result = choose_renderer(modification).render(
        directory=directory,
        metadata=metadata,
        modification=modification,
    )
    return FileResponse(
        result.path,
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
            "X-Render-Cached": str(result.cached).lower(),
            "X-Renderer-Used": result.renderer,
            "X-Quality-Status": result.quality_status,
        },
    )
