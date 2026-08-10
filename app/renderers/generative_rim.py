"""Mask-bound generative rim preview renderer."""

import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from app.errors import PipelineError
from app.generative.base import GenerativeImageEditProvider
from app.modifications.schemas import RimReplacementRequest, normalised_request_json
from app.renderers.base import RenderResult, request_hash
from app.rim_analysis import RIM_REFERENCE_VERSION, wheel_mask, wheel_reference
from app.schemas import AssetBundle


def build_rim_prompt() -> str:
    return (
        "Replace only the visible wheel rims inside Image 4's white mask. "
        "Image 1 is authoritative for the car, tyres, lighting, viewpoint, and background. "
        "Image 2 is authoritative for rim design. Image 3 is placement guidance. "
        "Preserve tyres, body panels, brakes hidden by the rim, shadows, windows, lights, plate, badges, and background. "
        "Do not change tyre size, wheel position, suspension, body shape, or add text/logos. Output one edited image."
    )


def _rough_rims(original: Image.Image, reference: Image.Image, reference_mask: np.ndarray, target: np.ndarray) -> Image.Image:
    ref_points = np.argwhere(reference_mask >= 128)
    if not len(ref_points):
        raise PipelineError("rim_reference_not_found", "Rim reference mask is empty", 404)
    ry1, rx1 = ref_points.min(axis=0)
    ry2, rx2 = ref_points.max(axis=0) + 1
    crop = reference.crop((int(rx1), int(ry1), int(rx2), int(ry2))).convert("RGBA")
    count, labels, stats, _ = cv2.connectedComponentsWithStats(target)
    overlay = Image.new("RGBA", original.size)
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        if area < 20:
            continue
        scale = min(width / crop.width, height / crop.height)
        rim = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
        overlay.alpha_composite(rim, (int(x + (width - rim.width) / 2), int(y + (height - rim.height) / 2)))
    alpha = np.asarray(overlay.getchannel("A")).copy()
    alpha[target < 128] = 0
    overlay.putalpha(Image.fromarray(alpha))
    result = Image.alpha_composite(original.convert("RGBA"), overlay).convert("RGB")
    pixels = np.asarray(result).copy()
    pixels[target < 128] = np.asarray(original.convert("RGB"))[target < 128]
    return Image.fromarray(pixels)


class GenerativeRimRenderer:
    name = "generative-rim"
    version = "rim-render-1"

    def __init__(self, provider: GenerativeImageEditProvider) -> None:
        self.provider = provider

    def render(self, *, directory: Path, metadata: AssetBundle, modification: RimReplacementRequest) -> RenderResult:
        reference_dir, reference_report = wheel_reference(directory, modification.reference_asset_id)
        key = request_hash(
            modification,
            renderer=self.name,
            provider=self.provider.name,
            pipeline_version=metadata.pipeline_version,
            reference_content_hash=reference_report.get("content_sha256", ""),
            provider_model=self.provider.model_id,
            renderer_version=f"{self.version}:{RIM_REFERENCE_VERSION}",
        )
        output_dir = directory / "customisations" / key
        normalized_request = normalised_request_json(modification)
        if (output_dir / "result.png").is_file() and (output_dir / "request.json").is_file() and (output_dir / "request.json").read_text(encoding="utf-8") == normalized_request:
            return RenderResult(output_dir / "result.png", True, self.name, "passed", [])
        try:
            original = Image.open(directory / metadata.original_image).convert("RGB")
            reference = Image.open(reference_dir / "normalized.png").convert("RGBA")
            reference_mask = cv2.imread(str(reference_dir / "reference-mask.png"), cv2.IMREAD_GRAYSCALE)
            if reference_mask is None:
                raise OSError("missing rim mask")
        except OSError as exc:
            raise PipelineError("rim_reference_not_found", "Rim reference is unreadable", 404) from exc
        mask = wheel_mask(directory, metadata)
        rough = _rough_rims(original, reference, reference_mask, mask)
        generated = self.provider.edit(original=original, reference=reference, rough_composite=rough, edit_mask=mask, instruction=build_rim_prompt())
        if generated.size != original.size:
            raise PipelineError("invalid_generated_image", "Generated image dimensions do not match the original", 502)
        source = np.asarray(original)
        final = np.asarray(generated.convert("RGB")).copy()
        final[mask < 128] = source[mask < 128]
        if not np.any(final[mask >= 128] != source[mask >= 128]):
            raise PipelineError("rim_quality_check_failed", "Generated rim preview did not change the wheel region", 502)
        result = Image.fromarray(final)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=".rim-render-", dir=output_dir.parent))
        try:
            (work / "request.json").write_text(normalized_request, encoding="utf-8")
            (work / "quality.json").write_text(json.dumps({"status": "passed", "warnings": []}, indent=2), encoding="utf-8")
            (work / "provider.json").write_text(json.dumps({"provider": self.provider.name, "model_id": self.provider.model_id, "renderer_version": self.version}, indent=2), encoding="utf-8")
            rough.save(work / "rough-composite.png")
            result.save(work / "result.png")
            if output_dir.exists():
                shutil.rmtree(work, ignore_errors=True)
            else:
                work.rename(output_dir)
        except Exception:
            shutil.rmtree(work, ignore_errors=True)
            raise
        return RenderResult(output_dir / "result.png", False, self.name, "passed", [])
