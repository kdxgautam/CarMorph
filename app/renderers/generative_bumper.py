"""Mask-bound generative bumper preview renderer."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.bumper_analysis.geometry import create_rough_composite
from app.bumper_analysis.mask_builder import build_bumper_masks
from app.bumper_analysis.reference_preprocessor import REFERENCE_PROCESSING_VERSION
from app.bumper_analysis.schemas import BumperReferenceReport
from app.errors import PipelineError
from app.generative.base import GenerativeImageEditProvider
from app.modifications.schemas import BumperReplacementRequest, normalised_request_json
from app.quality.bumper_checks import check_bumper_render
from app.quality.checks import QualityStatus
from app.renderers.base import RenderResult, render_base_image, request_hash
from app.schemas import AssetBundle


def build_bumper_prompt(request: BumperReplacementRequest) -> str:
    paint = (
        "Paint mode is match_body: recolour visible painted surfaces of the replacement or add-on to the target car's body colour under the target lighting; do not preserve the reference colour except for non-painted materials such as chrome, black plastic, rubber, glass, or bare metal."
        if request.paint_mode == "match_body"
        else "Paint mode is preserve_reference: keep the reference bumper or add-on's visible colour and material treatment as closely as possible; do not recolour it to match the target car body."
    )
    return (
        f"Replace only the masked {request.bumper_position} bumper. Image 1 is authoritative for the car, camera, lighting and protected content. "
        "Image 2 is authoritative for the bumper design. Image 3 is placement guidance. Image 4 is the only editable region. "
        "If Image 2 is a slim bumper guard, bumper bar, protector, diffuser strip, or chrome cross bar, install it as an add-on across the lower bumper instead of replacing or reshaping the full bumper skin. "
        "For slim add-on bars, reproduce only the supplied bar shape and materials; do not invent centre pads, red reflectors, extra end caps, chrome extensions, diffuser details, accessory trim, or styling not visible in Image 2. "
        "Preserve the bumper's major silhouette, openings, splitter and visible design details while adapting perspective and reflections. "
        f"{paint} Keep lights, grille, number plate, wheels, tyres, bonnet or boot, windows, badges, unrelated body panels, background, proportions and camera viewpoint unchanged. "
        "Do not add text, logos, accessories, body-kit parts, or make physical compatibility claims. Output one edited image."
    )


class GenerativeBumperRenderer:
    name = "generative-bumper"
    version = "bumper-render-3"

    def __init__(self, provider: GenerativeImageEditProvider, *, minimum_core_change_ratio: float = 0.01) -> None:
        if not 0 <= minimum_core_change_ratio <= 1:
            raise ValueError("minimum_core_change_ratio must be between zero and one")
        self.provider = provider
        self.minimum_core_change_ratio = minimum_core_change_ratio

    def _reference(self, directory: Path, reference_id: str) -> tuple[Path, BumperReferenceReport]:
        path = directory / "references" / "bumpers" / reference_id
        try:
            report = BumperReferenceReport.model_validate_json((path / "metadata.json").read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise PipelineError("bumper_reference_not_found", "Bumper reference was not found", 404) from exc
        except (OSError, ValueError) as exc:
            raise PipelineError("bumper_reference_not_found", "Bumper reference is invalid", 404) from exc
        if not all((path / item).is_file() for item in (report.normalized_image, report.reference_mask, report.source_image)):
            raise PipelineError("bumper_reference_not_found", "Bumper reference is incomplete", 404)
        content_hash = hashlib.sha256((path / report.normalized_image).read_bytes()).hexdigest()
        # The report stores the canonical pre-mask image hash. Stored transparent
        # references are canonical already; opaque references retain a derived alpha.
        if report.segmentation_method == "source_alpha" and content_hash != report.content_sha256:
            raise PipelineError("bumper_reference_not_found", "Bumper reference content changed", 404)
        return path, report

    def _cached(self, output_dir: Path, size: tuple[int, int], request: str) -> RenderResult | None:
        required = ("request.json", "quality.json", "provider.json", "prompt.txt", "rough-composite.png", "bumper-core-mask.png", "bumper-allowed-edit-mask.png", "bumper-protected-mask.png", "result.png")
        if not all((output_dir / name).is_file() for name in required):
            return None
        try:
            if (output_dir / "request.json").read_text(encoding="utf-8") != request:
                return None
            quality = json.loads((output_dir / "quality.json").read_text(encoding="utf-8"))
            provider = json.loads((output_dir / "provider.json").read_text(encoding="utf-8"))
            if provider.get("provider") != self.provider.name or provider.get("model_id") != self.provider.model_id:
                return None
            if quality.get("status") not in {item.value for item in QualityStatus}:
                return None
            with Image.open(output_dir / "result.png") as image:
                if image.convert("RGB").size != size:
                    return None
            return RenderResult(output_dir / "result.png", True, self.name, quality["status"], quality.get("warnings", []))
        except (OSError, ValueError, KeyError):
            return None

    def render(
        self,
        *,
        directory: Path,
        metadata: AssetBundle,
        modification: BumperReplacementRequest,
        base_image: Image.Image | None = None,
    ) -> RenderResult:
        if metadata.view not in {"front", "rear"} or metadata.view != modification.bumper_position:
            raise PipelineError("unsupported_bumper_view", "Bumper position must match the processed front or rear view", 400)
        reference_dir, reference_report = self._reference(directory, modification.reference_asset_id)
        original, base_hash = render_base_image(
            directory=directory, metadata=metadata, base_image=base_image
        )
        key = request_hash(
            modification,
            renderer=self.name,
            provider=self.provider.name,
            pipeline_version=metadata.pipeline_version,
            reference_content_hash=reference_report.content_sha256,
            provider_model=self.provider.model_id,
            renderer_version=f"{self.version}:{REFERENCE_PROCESSING_VERSION}",
            base_image_hash=base_hash,
        )
        output_dir = directory / "customisations" / key
        normalized_request = normalised_request_json(modification)
        cached = self._cached(output_dir, (metadata.width, metadata.height), normalized_request)
        if cached:
            return cached
        try:
            with Image.open(reference_dir / reference_report.normalized_image) as image:
                reference = image.convert("RGBA")
            reference_mask = cv2.imread(str(reference_dir / reference_report.reference_mask), cv2.IMREAD_GRAYSCALE)
            if reference_mask is None:
                raise OSError("Missing reference mask")
        except OSError as exc:
            raise PipelineError("bumper_reference_not_found", "Bumper reference is unreadable", 404) from exc
        masks = build_bumper_masks(directory, metadata)
        rough = create_rough_composite(original=original, reference_rgba=reference, reference_mask=reference_mask, target_core_mask=masks.core, allowed_mask=masks.allowed)
        edit_mask = masks.allowed
        reference_points = np.argwhere(reference_mask >= 128)
        if len(reference_points):
            ry1, rx1 = reference_points.min(axis=0)
            ry2, rx2 = reference_points.max(axis=0) + 1
            if (rx2 - rx1) / max(ry2 - ry1, 1) >= 4:
                guidance = np.any(np.asarray(rough) != np.asarray(original), axis=2)
                radius = max(3, min(12, round(min(metadata.width, metadata.height) * 0.01)))
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
                edit_mask = np.where((cv2.dilate(guidance.astype(np.uint8), kernel) > 0) & (masks.allowed >= 128), 255, 0).astype(np.uint8)
                if not np.any(edit_mask):
                    edit_mask = masks.allowed
        prompt = build_bumper_prompt(modification)
        generated = self.provider.edit(original=original, reference=reference, rough_composite=rough, edit_mask=edit_mask, instruction=prompt)
        if generated.size != original.size:
            raise PipelineError("invalid_generated_image", "Generated image dimensions do not match the original", 502)
        source_pixels = np.asarray(original).copy()
        final_pixels = np.asarray(generated.convert("RGB")).copy()
        final_pixels[edit_mask < 128] = source_pixels[edit_mask < 128]
        final_pixels[masks.protected >= 128] = source_pixels[masks.protected >= 128]
        result = Image.fromarray(final_pixels)
        quality = check_bumper_render(original=original, generated=generated, result=result, core_mask=masks.core, allowed_mask=edit_mask, protected_mask=masks.protected, minimum_core_change_ratio=self.minimum_core_change_ratio)
        if quality.status == QualityStatus.FAILED:
            raise PipelineError("bumper_quality_check_failed", ", ".join(quality.warnings), 502)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=".bumper-", dir=output_dir.parent))
        try:
            (work / "request.json").write_text(normalized_request, encoding="utf-8")
            (work / "quality.json").write_text(json.dumps(quality.model_dump(), indent=2), encoding="utf-8")
            (work / "provider.json").write_text(json.dumps({"provider": self.provider.name, "model_id": self.provider.model_id, "renderer_version": self.version}, indent=2), encoding="utf-8")
            (work / "prompt.txt").write_text(prompt, encoding="utf-8")
            rough.save(work / "rough-composite.png")
            for name, mask in (("bumper-core-mask.png", masks.core), ("bumper-blend-mask.png", masks.blend), ("bumper-allowed-edit-mask.png", edit_mask), ("bumper-protected-mask.png", masks.protected)):
                if not cv2.imwrite(str(work / name), mask):
                    raise OSError(f"Could not persist {name}")
            result.save(work / "result.png")
            if output_dir.exists():
                shutil.rmtree(work, ignore_errors=True)
            else:
                work.rename(output_dir)
        except Exception:
            shutil.rmtree(work, ignore_errors=True)
            raise
        return RenderResult(output_dir / "result.png", False, self.name, quality.status.value, quality.warnings)
