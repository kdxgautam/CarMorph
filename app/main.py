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

from app.bumper_analysis.reference_preprocessor import store_bumper_reference
from app.bumper_analysis.schemas import BumperReferenceResponse
from app.config import GenerativeSettings, Settings
from app.errors import PipelineError
from app.image_ops import recolour
from app.generative.vertex_ai import vertex_provider
from app.modifications.schemas import BumperReplacementRequest, StudioRenderRequest, SurfaceEditRequest, parse_modification
from app.pipeline import process_view
from app.renderers.deterministic import DeterministicSurfaceRenderer
from app.renderers.generative_bumper import GenerativeBumperRenderer
from app.renderers.generative_rim import GenerativeRimRenderer
from app.renderers.generative_studio import GenerativeStudioRenderer
from app.rim_analysis import store_rim_reference
from app.schemas import AssetBundle, ViewSelection
from app.studio_references import (
    StudioIdentityResponse,
    load_studio_reference,
    store_studio_reference,
)

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


@app.post("/cars/{asset_id}/bumper-references", response_model=BumperReferenceResponse, status_code=201)
def upload_bumper_reference(
    asset_id: str,
    image: Annotated[UploadFile, File()],
) -> BumperReferenceResponse:
    """Store one validated bumper reference alongside a prepared car asset."""

    directory, metadata = _asset(asset_id)
    if metadata.view not in {"front", "rear"}:
        raise PipelineError("unsupported_bumper_view", "Bumper replacement supports front and rear views only", 400)
    source = image.file.read(MAX_UPLOAD_BYTES + 1)
    if len(source) > MAX_UPLOAD_BYTES or not source:
        raise PipelineError("invalid_bumper_reference", "Reference image is empty or exceeds the 20 MB limit", 400)
    return store_bumper_reference(
        directory=directory,
        metadata=metadata,
        source=source,
        settings=None,
    )


@app.post("/cars/{asset_id}/rim-references", response_model=BumperReferenceResponse, status_code=201)
def upload_rim_reference(
    asset_id: str,
    image: Annotated[UploadFile, File()],
) -> BumperReferenceResponse:
    directory, metadata = _asset(asset_id)
    source = image.file.read(MAX_UPLOAD_BYTES + 1)
    return store_rim_reference(directory=directory, metadata=metadata, source=source)


@app.post("/cars/{asset_id}/studio-identity", response_model=StudioIdentityResponse)
def analyse_studio_identity(
    asset_id: str,
    images: list[UploadFile] | None = File(default=None),
) -> StudioIdentityResponse:
    """Identify the target car from its original and up to four supporting views."""

    directory, metadata = _asset(asset_id)
    uploads = images or []
    if len(uploads) > 4:
        raise PipelineError("invalid_studio_reference", "At most four supporting images are allowed", 400)
    stored = []
    for upload in uploads:
        stored.append(
            store_studio_reference(
                directory=directory,
                source=upload.file.read(MAX_UPLOAD_BYTES + 1),
                kind="user",
                title=upload.filename or "Supporting vehicle view",
            )
        )
    try:
        with Image.open(directory / metadata.original_image) as opened:
            identity_images = [opened.convert("RGB")]
        for reference in stored:
            path, _ = load_studio_reference(directory, reference.reference_asset_id)
            with Image.open(path) as opened:
                identity_images.append(opened.convert("RGB"))
    except OSError as exc:
        raise PipelineError("invalid_stored_assets", "A vehicle identity image is unreadable", 500) from exc
    identity = vertex_provider(GenerativeSettings.from_env()).identify_vehicle(identity_images)
    return StudioIdentityResponse(
        identity=identity,
        reference_asset_ids=[reference.reference_asset_id for reference in stored],
    )


@app.get("/cars/{asset_id}/studio-references/{reference_id}")
def get_studio_reference(asset_id: str, reference_id: str) -> FileResponse:
    directory, _ = _asset(asset_id)
    path, _ = load_studio_reference(directory, reference_id)
    return FileResponse(path, media_type="image/png", headers={"Cache-Control": "private, max-age=3600"})


def _render_response(result) -> FileResponse:
    return FileResponse(
        result.path,
        media_type="image/png",
        headers={
            "Cache-Control": "no-store",
            "X-Render-Cached": str(result.cached).lower(),
            "X-Renderer-Used": result.renderer,
            "X-Quality-Status": result.quality_status,
            "X-Quality-Warnings": ",".join(result.warnings),
        },
    )


@app.post("/cars/{asset_id}/studio-render")
def studio_render(asset_id: str, modification: StudioRenderRequest) -> FileResponse:
    """Render one prepared target using optional user-supplied vehicle views."""

    directory, metadata = _asset(asset_id)
    result = GenerativeStudioRenderer(vertex_provider(GenerativeSettings.from_env())).render(
        directory=directory, metadata=metadata, modification=modification
    )
    return _render_response(result)


@app.post("/cars/{asset_id}/customise")
async def customise(asset_id: str, request: Request) -> FileResponse:
    """Validate a deterministic paint edit and return its PNG."""

    directory, metadata = _asset(asset_id)
    try:
        body = await request.json()
    except ValueError as exc:
        raise PipelineError("invalid_request", "Request body must be JSON") from exc
    modification = parse_modification(body)
    if isinstance(modification, SurfaceEditRequest):
        result = DeterministicSurfaceRenderer().render(
            directory=directory, metadata=metadata, modification=modification
        )
    elif isinstance(modification, BumperReplacementRequest):
        result = GenerativeBumperRenderer(vertex_provider(GenerativeSettings.from_env())).render(
            directory=directory, metadata=metadata, modification=modification
        )
    elif isinstance(modification, StudioRenderRequest):
        raise PipelineError("invalid_modification", "Use the dedicated studio-render endpoint", 400)
    else:
        result = GenerativeRimRenderer(vertex_provider(GenerativeSettings.from_env())).render(
            directory=directory, metadata=metadata, modification=modification
        )
    return _render_response(result)
