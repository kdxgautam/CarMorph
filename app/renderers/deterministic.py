import json
from io import BytesIO
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.errors import PipelineError
from app.image_ops import recolour
from app.modifications.schemas import SurfaceEditRequest, normalised_request_json
from app.paint_analysis.mask_builder import load_request_masks
from app.quality.checks import QualityStatus, check_render
from app.renderers.base import RenderResult, request_hash
from app.schemas import AssetBundle


class DeterministicSurfaceRenderer:
    name = "deterministic"

    def render(
        self,
        *,
        directory: Path,
        metadata: AssetBundle,
        modification: SurfaceEditRequest,
    ) -> RenderResult:
        if not modification.body_colour and not modification.roof_colour:
            raise PipelineError("invalid_modification", "A paint colour is required")
        if modification.design_elements:
            raise PipelineError(
                "renderer_not_supported",
                "Deterministic rendering does not support racing stripes yet",
            )

        key = request_hash(
            modification,
            renderer=self.name,
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
        except OSError as exc:
            raise PipelineError("missing_masks", "A render asset is missing", 500) from exc
        if protected is None:
            raise PipelineError("missing_masks", "A render mask is missing", 500)
        if modification.roof_colour and not np.any(roof_mask >= 128):
            raise PipelineError(
                "missing_masks", "A contrast-roof paint group is not available", 400
            )
        result = original
        if modification.body_colour:
            result = Image.open(
                BytesIO(
                    recolour(
                        result,
                        body_mask,
                        modification.body_colour,
                        modification.finish.value,
                    )
                )
            ).convert("RGB")
        if modification.roof_colour:
            result = Image.open(
                BytesIO(
                    recolour(
                        result,
                        roof_mask,
                        modification.roof_colour,
                        modification.finish.value,
                    )
                )
            ).convert("RGB")
        editable = np.maximum(
            body_mask if modification.body_colour else np.zeros_like(body_mask),
            roof_mask,
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
