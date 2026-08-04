"""Render constrained generative edits and restore all protected pixels."""

import json
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from app.errors import PipelineError
from app.flux import (
    FluxSettings,
    HuggingFaceFluxKontextProvider,
    composite_design,
    composite_plain_colour,
)
from app.image_ops import parse_colour
from app.modifications.prompts import build_surface_prompt
from app.modifications.schemas import SurfaceEditRequest, normalised_request_json
from app.paint_analysis.mask_builder import load_request_masks
from app.quality.checks import QualityStatus, check_render
from app.renderers.base import RenderResult, request_hash
from app.schemas import AssetBundle


class ImageEditProvider(Protocol):
    """Minimal interface required from an external image-edit provider."""

    name: str

    def edit(self, *, image_path: Path, prompt: str, settings: FluxSettings) -> Image.Image:
        """Return one generated RGB edit for the supplied source and prompt."""

        ...


class GenerativeSurfaceRenderer:
    """Constrained provider renderer with mask restoration and quality checks."""

    # Provider output is never trusted as the final composite; masks below restore
    # protected and background pixels from the original image.
    name = "generative"

    def __init__(
        self,
        settings: FluxSettings | None = None,
        provider: ImageEditProvider | None = None,
    ) -> None:
        """Accept injectable settings/provider for deterministic tests."""

        self.settings = settings or FluxSettings.from_env()
        self.provider = provider or HuggingFaceFluxKontextProvider()

    def render(
        self,
        *,
        directory: Path,
        metadata: AssetBundle,
        modification: SurfaceEditRequest,
    ) -> RenderResult:
        """Generate, mask-composite, verify, and cache a structured surface edit."""

        key = request_hash(
            modification,
            renderer=self.name,
            provider=self.provider.name,
            settings=(
                f"{self.settings.space}|{self.settings.guidance_scale}|"
                f"{self.settings.steps}|{self.settings.seed}"
            ),
            pipeline_version=metadata.pipeline_version,
        )
        output_dir = directory / "customisations" / key
        output = output_dir / "result.png"
        if output.is_file():
            return RenderResult(output, True, self.name, QualityStatus.PASSED.value, [])

        protected_path = metadata.masks.get("protected_mask")
        try:
            with Image.open(directory / metadata.original_image) as opened:
                original = opened.convert("RGB")
            body_mask, roof_mask = load_request_masks(
                directory, metadata, include_roof=bool(modification.roof_colour)
            )
            protected = (
                cv2.imread(str(directory / protected_path), cv2.IMREAD_GRAYSCALE)
                if protected_path
                else np.zeros((original.height, original.width), np.uint8)
            )
        except (OSError, UnidentifiedImageError) as exc:
            raise PipelineError("missing_masks", "A render asset is missing", 500) from exc
        if protected is None:
            raise PipelineError("missing_masks", "Protected mask is missing", 500)
        if modification.roof_colour and not np.any(roof_mask >= 128):
            raise PipelineError(
                "missing_masks", "A contrast-roof paint group is not available", 400
            )
        editable = np.maximum(
            body_mask
            if (
                modification.body_colour
                or modification.design_elements
                or modification.custom_instruction
            )
            else np.zeros_like(body_mask),
            roof_mask,
        )
        mask = Image.fromarray(editable)

        prompt = build_surface_prompt(modification)
        generated = self._edit_with_one_retry(directory / metadata.original_image, prompt)
        if modification.design_elements or modification.custom_instruction:
            result = composite_design(
                original, generated, Image.fromarray(body_mask)
            )
            if modification.roof_colour:
                _, roof_rgb = parse_colour(modification.roof_colour)
                result = composite_plain_colour(
                    result, generated, Image.fromarray(roof_mask), roof_rgb
                )
        else:
            result = original
            if modification.body_colour:
                _, rgb = parse_colour(modification.body_colour)
                result = composite_plain_colour(
                    result, generated, Image.fromarray(body_mask), rgb
                )
            if modification.roof_colour:
                _, roof_rgb = parse_colour(modification.roof_colour)
                result = composite_plain_colour(
                    result, generated, Image.fromarray(roof_mask), roof_rgb
                )

        quality = check_render(
            original=original,
            result=result,
            editable_mask=editable,
            protected_mask=protected,
        )
        if quality.status == QualityStatus.FAILED:
            raise PipelineError(
                "quality_check_failed",
                ", ".join(quality.warnings) or "Render failed quality checks",
                502,
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "request.json").write_text(
            normalised_request_json(modification), encoding="utf-8"
        )
        (output_dir / "quality.json").write_text(
            json.dumps(quality.model_dump(), indent=2), encoding="utf-8"
        )
        temporary = output.with_suffix(".tmp")
        result.save(temporary, "PNG")
        temporary.replace(output)
        return RenderResult(output, False, self.name, quality.status.value, quality.warnings)

    def _edit_with_one_retry(self, image_path: Path, prompt: str) -> Image.Image:
        """Retry one transient provider-unavailable failure, but no other errors."""

        last_error: PipelineError | None = None
        for _ in range(2):
            try:
                return self.provider.edit(
                    image_path=image_path,
                    prompt=prompt,
                    settings=self.settings,
                )
            except PipelineError as exc:
                if exc.code != "flux_unavailable":
                    raise
                last_error = exc
        raise last_error or PipelineError("flux_unavailable", "FLUX is unavailable", 503)
