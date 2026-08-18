"""Identity-preserving automotive studio render."""

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter

from app.errors import PipelineError
from app.generative.base import GenerativeImageEditProvider
from app.modifications.schemas import StudioRenderRequest, normalised_request_json
from app.quality.checks import QualityStatus
from app.renderers.base import RenderResult, render_base_image, request_hash
from app.schemas import AssetBundle
from app.studio_references import load_studio_reference


STYLE_PROMPTS = {
    "light_studio": "Use a seamless light gray backdrop, clean white studio floor, soft grounded contact shadow, subtle floor reflection, high-key catalog lighting and balanced three-quarter automotive framing.",
    "dark_studio": "Use a charcoal studio with subtle dramatic lighting.",
    "premium_gradient": "Use a clean luxury neutral gradient with a soft halo behind the car.",
}


def build_studio_prompt(request: StudioRenderRequest, reference_count: int = 0) -> str:
    identity = request.vehicle_identity
    identity_prompt = ""
    if identity:
        identity_prompt = (
            f"The user-confirmed visual identity is {identity.make} {identity.model}, {identity.generation}, "
            f"{identity.body_style}{f', trim {identity.trim}' if identity.trim else ''}. "
            f"Visible cues: {', '.join(identity.visual_cues) or 'none supplied'}. "
        )
    reference_prompt = ""
    if reference_count:
        reference_prompt = (
            f"There are {reference_count} user-supplied supporting views of this same vehicle. "
            "Use them to preserve identity details, but the target image remains authoritative for camera angle, colour, accessories and trim. "
        )
    return (
        "This is an image editing task, not a new vehicle generation task. "
        "Preserve the exact vehicle from the supplied image. Do not redesign, replace, modernise, or reinterpret the vehicle. "
        "Keep exactly the same body proportions, silhouette, camera angle, perspective, grille, headlights, bumper, wheels, mirrors, windows, roof rails, doors, trim and vehicle geometry. "
        "Transform only the presentation. Make the existing vehicle appear like a premium photorealistic automotive CGI/product render. "
        "Use professional studio lighting, realistic paint reflections, physically plausible highlights, realistic tyre contact shadows, a subtle floor reflection and a clean minimal automotive studio. "
        "Do not change the wheels. Do not change the grille. Do not change the headlights. Do not change the body panels. Do not add body kits. Do not change the vehicle generation or model. "
        "The final image must clearly remain the exact source vehicle. "
        f"{identity_prompt}{reference_prompt}"
        f"{STYLE_PROMPTS[request.style.value]}"
    )


def _mask(directory: Path, metadata: AssetBundle, key: str) -> np.ndarray | None:
    path = metadata.masks.get(key)
    if not path:
        return None
    mask = cv2.imread(str(directory / path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise PipelineError("missing_masks", f"{key} mask is missing", 500)
    if mask.shape != (metadata.height, metadata.width):
        raise PipelineError("mask_dimension_mismatch", f"{key} mask dimensions are invalid", 500)
    return mask


def _canvas(size: tuple[int, int], style: str) -> Image.Image:
    width, height = size
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    x = np.linspace(-1, 1, width, dtype=np.float32)[None, :]
    if style == "dark_studio":
        base = 28 + 28 * (1 - y)
        halo = 28 * np.exp(-(x**2 / 0.55 + (y - 0.35) ** 2 / 0.12))
        rgb = np.dstack([base + halo, base + halo, base + halo + 4])
    elif style == "premium_gradient":
        base = 218 - 48 * y
        halo = 42 * np.exp(-(x**2 / 0.45 + (y - 0.42) ** 2 / 0.08))
        rgb = np.dstack([base + halo + 6, base + halo + 2, base + halo - 4])
    else:
        base = np.broadcast_to(242 - 42 * y, (height, width))
        rgb = np.dstack([base, base, base])
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


def _prepared_studio_image(image: Image.Image, full_car: np.ndarray, style: str) -> Image.Image:
    alpha = Image.fromarray(np.where(full_car >= 128, 255, 0).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(2)
    )
    background = _canvas(image.size, style)
    background.paste(image, (0, 0), alpha)
    return background


def _restore_masked_region(result: Image.Image, source: Image.Image, mask: np.ndarray) -> Image.Image:
    alpha_pixels = np.asarray(
        Image.fromarray(np.where(mask >= 128, 255, 0).astype(np.uint8)).filter(
            ImageFilter.GaussianBlur(1)
        )
    ).copy()
    alpha_pixels[mask >= 128] = 255
    alpha = Image.fromarray(alpha_pixels)
    output = result.convert("RGB")
    output.paste(source.convert("RGB"), (0, 0), alpha)
    return output


class GenerativeStudioRenderer:
    name = "generative-studio"
    version = "studio-render-2"

    def __init__(self, provider: GenerativeImageEditProvider) -> None:
        self.provider = provider

    def render(
        self,
        *,
        directory: Path,
        metadata: AssetBundle,
        modification: StudioRenderRequest,
        base_image: Image.Image | None = None,
    ) -> RenderResult:
        original, base_hash = render_base_image(
            directory=directory, metadata=metadata, base_image=base_image
        )
        full_car = _mask(directory, metadata, "full_car")
        if full_car is None or not np.any(full_car >= 128):
            raise PipelineError("missing_masks", "Full-car mask is missing", 500)
        loaded_references = []
        for reference_id in modification.reference_asset_ids:
            path, report = load_studio_reference(directory, reference_id)
            if hashlib.sha256(path.read_bytes()).hexdigest() != report.content_sha256:
                raise PipelineError("studio_reference_not_found", "Studio reference content changed", 404)
            with Image.open(path) as image:
                loaded_references.append((report, image.convert("RGB")))
        reference_hash = "|".join(item[0].content_sha256 for item in loaded_references)
        key = request_hash(
            modification,
            renderer=self.name,
            provider=self.provider.name,
            pipeline_version=metadata.pipeline_version,
            provider_model=self.provider.model_id,
            renderer_version=self.version,
            base_image_hash=base_hash,
            reference_content_hash=reference_hash,
        )
        output_dir = directory / "customisations" / key
        normalized_request = normalised_request_json(modification)
        if (
            (output_dir / "result.png").is_file()
            and (output_dir / "request.json").is_file()
            and (output_dir / "request.json").read_text(encoding="utf-8") == normalized_request
        ):
            try:
                quality = json.loads((output_dir / "quality.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                quality = {"status": "passed", "warnings": []}
            return RenderResult(output_dir / "result.png", True, self.name, quality["status"], quality.get("warnings", []))

        protected = np.zeros((metadata.height, metadata.width), np.uint8)
        for mask_name in ("wheels", "lights", "grille", "trim", "windows"):
            mask = _mask(directory, metadata, mask_name)
            if mask is not None:
                protected = np.maximum(protected, mask)
        warnings = []
        if modification.preserve_plate:
            plate = _mask(directory, metadata, "plate")
            if plate is not None and np.any(plate >= 128):
                protected = np.maximum(protected, plate)
            else:
                warnings.append("plate_mask_missing")

        prepared = _prepared_studio_image(original, full_car, modification.style.value)
        edit_mask = np.full((metadata.height, metadata.width), 255, np.uint8)
        edit_mask[protected >= 128] = 0
        prompt = build_studio_prompt(modification, len(loaded_references))
        generated = self.provider.edit(
            original=prepared,
            reference=original,
            rough_composite=prepared,
            edit_mask=edit_mask,
            instruction=prompt,
            additional_references=[item[1] for item in loaded_references],
        )
        if generated.size != original.size:
            raise PipelineError("invalid_generated_image", "Generated image dimensions do not match the original", 502)

        result = generated.convert("RGB")
        if np.any(protected >= 128):
            result = _restore_masked_region(result, original, protected)

        output_dir.parent.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(prefix=".studio-render-", dir=output_dir.parent))
        try:
            (work / "request.json").write_text(normalized_request, encoding="utf-8")
            (work / "quality.json").write_text(
                json.dumps({"status": QualityStatus.PASSED.value, "warnings": warnings}, indent=2),
                encoding="utf-8",
            )
            (work / "provider.json").write_text(
                json.dumps({"provider": self.provider.name, "model_id": self.provider.model_id, "renderer_version": self.version, "reference_asset_ids": modification.reference_asset_ids}, indent=2),
                encoding="utf-8",
            )
            (work / "prompt.txt").write_text(prompt, encoding="utf-8")
            prepared.save(work / "prepared-studio.png")
            result.save(work / "result.png")
            if output_dir.exists():
                shutil.rmtree(work, ignore_errors=True)
            else:
                work.rename(output_dir)
        except Exception:
            shutil.rmtree(work, ignore_errors=True)
            raise
        return RenderResult(output_dir / "result.png", False, self.name, QualityStatus.PASSED.value, warnings)
