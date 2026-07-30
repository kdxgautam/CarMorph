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
from app.pipeline import process_view
from app.schemas import AssetBundle, ViewName

app = FastAPI(title="Car Customisation API", version="0.1.0")
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ASSET_ID_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@app.exception_handler(PipelineError)
def pipeline_error_handler(_: Request, exc: PipelineError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": {"code": exc.code, "detail": exc.detail}},
    )


@app.exception_handler(RequestValidationError)
def validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "invalid_request", "detail": str(exc)}},
    )


def _asset(asset_id: str) -> tuple[Path, AssetBundle]:
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
    view: Annotated[ViewName, Form()] = "front",
) -> AssetBundle:
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
    return _asset(asset_id)[1]


@app.get("/cars/{asset_id}/assets/{asset_path:path}")
def get_asset(asset_id: str, asset_path: str) -> FileResponse:
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
