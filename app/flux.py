"""Optional FLUX image editing and mask-safe result compositing."""

import colorsys
import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import cv2
import numpy as np
from gradio_client import Client, handle_file
from PIL import Image, UnidentifiedImageError

from app.errors import PipelineError
from app.image_ops import parse_colour
from app.modifications.prompts import build_surface_prompt
from app.modifications.schemas import SurfaceEditRequest
from app.schemas import AssetBundle

FLUX_RENDER_VERSION = "3"
# ponytail: process-local lock; use a shared job/lock store when running workers.
_FLUX_LOCK = Lock()


def _colour_name(rgb: tuple[int, int, int]) -> str:
    """Convert an RGB target to a coarse natural-language paint name."""

    hue, saturation, value = colorsys.rgb_to_hsv(*(channel / 255 for channel in rgb))
    if saturation < 0.12:
        return "white" if value > 0.85 else "black" if value < 0.18 else "grey"
    names = (
        "red",
        "orange",
        "yellow",
        "lime green",
        "green",
        "teal",
        "cyan",
        "sky blue",
        "blue",
        "violet",
        "magenta",
        "rose red",
    )
    return names[round(hue * 12) % 12]


@dataclass(frozen=True)
class FluxSettings:
    """Validated connection and deterministic inference settings for FLUX."""

    token: str
    space: str
    guidance_scale: float
    steps: int
    seed: int
    timeout: float

    @classmethod
    def from_env(cls) -> "FluxSettings":
        """Load FLUX credentials/settings and reject unsupported ranges."""

        token = os.getenv("HF_TOKEN")
        if not token:
            raise PipelineError(
                "configuration_error", "Missing environment variable: HF_TOKEN", 503
            )
        try:
            settings = cls(
                token=token,
                space=os.getenv(
                    "HF_FLUX_SPACE", "black-forest-labs/FLUX.1-Kontext-Dev"
                ),
                guidance_scale=float(os.getenv("HF_FLUX_GUIDANCE_SCALE", "2.5")),
                steps=int(os.getenv("HF_FLUX_STEPS", "28")),
                seed=int(os.getenv("HF_FLUX_SEED", "0")),
                timeout=float(os.getenv("HF_FLUX_TIMEOUT_SECONDS", "300")),
            )
        except ValueError as exc:
            raise PipelineError(
                "configuration_error",
                "Numeric FLUX environment variables contain an invalid value",
                503,
            ) from exc
        if not settings.space:
            raise PipelineError(
                "configuration_error", "HF_FLUX_SPACE cannot be empty", 503
            )
        if not 1 <= settings.guidance_scale <= 10:
            raise PipelineError(
                "configuration_error",
                "HF_FLUX_GUIDANCE_SCALE must be between 1 and 10",
                503,
            )
        if not 1 <= settings.steps <= 30:
            raise PipelineError(
                "configuration_error", "HF_FLUX_STEPS must be between 1 and 30", 503
            )
        if not 0 <= settings.seed <= 2_147_483_647:
            raise PipelineError(
                "configuration_error",
                "HF_FLUX_SEED must be between 0 and 2147483647",
                503,
            )
        if settings.timeout <= 0:
            raise PipelineError(
                "configuration_error",
                "HF_FLUX_TIMEOUT_SECONDS must be positive",
                503,
            )
        return settings


def composite_plain_colour(
    original: Image.Image,
    generated: Image.Image,
    mask: Image.Image,
    target_rgb: tuple[int, int, int],
) -> Image.Image:
    """Colour-correct generated paint, preserve luminance, and mask-composite it."""

    original = original.convert("RGB")
    mask = mask.convert("L")
    if mask.size != original.size:
        raise PipelineError(
            "mask_dimension_mismatch",
            "Body mask dimensions do not match the original image",
            500,
        )
    if not mask.getbbox():
        raise PipelineError("missing_masks", "Paintable-body mask is empty", 500)
    generated = generated.convert("RGB").resize(original.size, Image.Resampling.LANCZOS)
    original_pixels = np.asarray(original)
    generated_pixels = np.asarray(generated)
    original_lab = cv2.cvtColor(original_pixels, cv2.COLOR_RGB2LAB)
    generated_lab = cv2.cvtColor(generated_pixels, cv2.COLOR_RGB2LAB)
    generated_lab[:, :, 0] = original_lab[:, :, 0]
    structure_preserved = cv2.cvtColor(generated_lab, cv2.COLOR_LAB2RGB)
    core = np.asarray(mask) >= 128
    median_rgb = np.median(structure_preserved[core], axis=0)
    gains = np.asarray(target_rgb) / np.maximum(median_rgb, 1)
    colour_corrected = Image.fromarray(
        np.rint(np.clip(structure_preserved * gains, 0, 255)).astype(np.uint8)
    )
    return Image.composite(colour_corrected, original, mask)


def composite_design(
    original: Image.Image,
    generated: Image.Image,
    mask: Image.Image,
) -> Image.Image:
    """Composite a generated surface design while restoring source luminance."""

    original = original.convert("RGB")
    mask = mask.convert("L")
    if mask.size != original.size:
        raise PipelineError(
            "mask_dimension_mismatch",
            "Body mask dimensions do not match the original image",
            500,
        )
    if not mask.getbbox():
        raise PipelineError("missing_masks", "Paintable-body mask is empty", 500)
    generated = generated.convert("RGB").resize(original.size, Image.Resampling.LANCZOS)
    original_pixels = np.asarray(original)
    generated_pixels = np.asarray(generated)
    original_lab = cv2.cvtColor(original_pixels, cv2.COLOR_RGB2LAB)
    generated_lab = cv2.cvtColor(generated_pixels, cv2.COLOR_RGB2LAB)
    generated_lab[:, :, 0] = original_lab[:, :, 0]
    return Image.composite(
        Image.fromarray(cv2.cvtColor(generated_lab, cv2.COLOR_LAB2RGB)),
        original,
        mask,
    )


class HuggingFaceFluxKontextProvider:
    """Small adapter around the public Hugging Face FLUX Kontext Space."""

    name = "huggingface_flux_kontext"

    def edit(
        self,
        *,
        image_path: Path,
        prompt: str,
        settings: FluxSettings,
    ) -> Image.Image:
        """Submit a deterministic edit and validate the downloaded image."""

        try:
            with tempfile.TemporaryDirectory() as temporary:
                client = Client(
                    settings.space,
                    token=settings.token,
                    verbose=False,
                    download_files=temporary,
                    httpx_kwargs={"timeout": settings.timeout},
                )
                response = client.predict(
                    input_image=handle_file(image_path),
                    prompt=prompt,
                    seed=settings.seed,
                    randomize_seed=False,
                    guidance_scale=settings.guidance_scale,
                    steps=settings.steps,
                    api_name="/infer",
                )
                if (
                    not isinstance(response, (tuple, list))
                    or not response
                    or not isinstance(response[0], (str, Path))
                ):
                    raise PipelineError(
                        "invalid_flux_response",
                        "FLUX returned an invalid response",
                        502,
                    )
                with Image.open(response[0]) as opened:
                    return opened.convert("RGB")
        except PipelineError:
            raise
        except (OSError, UnidentifiedImageError) as exc:
            raise PipelineError(
                "invalid_flux_response", "FLUX returned an invalid image", 502
            ) from exc
        except Exception as exc:
            raise PipelineError(
                "flux_unavailable",
                "FLUX is unavailable, queued, or its Hugging Face quota is exhausted",
                503,
            ) from exc


def render_flux(
    directory: Path,
    metadata: AssetBundle,
    colour: str,
    settings: FluxSettings,
) -> tuple[Path, bool]:
    """Render and cache the legacy single-colour FLUX endpoint.

    The generated image is never returned directly: original luminance and all
    pixels outside the stored body mask are restored before an atomic write.
    """

    colour, rgb = parse_colour(colour)
    cache_key = hashlib.sha256(
        (
            f"{FLUX_RENDER_VERSION}|{settings.space}|{settings.guidance_scale}|"
            f"{settings.steps}|{settings.seed}|{colour}"
        ).encode()
    ).hexdigest()[:12]
    output = directory / "renders" / f"flux-{colour}-{cache_key}.png"

    with _FLUX_LOCK:
        if output.is_file():
            return output, True

        body_path = metadata.masks.get("paintable_body")
        if not body_path:
            raise PipelineError("missing_masks", "Paintable-body mask is missing", 500)
        try:
            with Image.open(directory / metadata.original_image) as opened:
                original = opened.convert("RGB")
            with Image.open(directory / body_path) as opened:
                mask = opened.convert("L")
        except (OSError, UnidentifiedImageError) as exc:
            raise PipelineError(
                "missing_masks", "A FLUX input asset is missing or invalid", 500
            ) from exc
        if mask.size != original.size:
            raise PipelineError(
                "mask_dimension_mismatch",
                "Body mask dimensions do not match the original image",
                500,
            )
        if not mask.getbbox():
            raise PipelineError("missing_masks", "Paintable-body mask is empty", 500)

        prompt = build_surface_prompt(
            SurfaceEditRequest(body_colour=f"#{colour}", renderer="generative")
        )

        try:
            generated = HuggingFaceFluxKontextProvider().edit(
                image_path=directory / metadata.original_image,
                prompt=prompt,
                settings=settings,
            )
            result = composite_plain_colour(original, generated, mask, rgb)
        except PipelineError:
            raise
        except Exception as exc:
            raise PipelineError(
                "flux_unavailable",
                "FLUX is unavailable, queued, or its Hugging Face quota is exhausted",
                503,
            ) from exc

        output.parent.mkdir(exist_ok=True)
        temporary_output = output.with_suffix(".tmp")
        result.save(temporary_output, "PNG")
        temporary_output.replace(output)
        return output, False
