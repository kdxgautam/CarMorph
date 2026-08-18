"""Vertex AI Gemini image provider with no public prompt or credential surface."""

from collections.abc import Sequence
from functools import lru_cache
from io import BytesIO

import httpx
import numpy as np
from PIL import Image

from app.config import GenerativeSettings
from app.errors import PipelineError
from app.modifications.schemas import VehicleIdentity


def _png(image: Image.Image) -> bytes:
    output = BytesIO()
    image.save(output, "PNG", exif=b"", icc_profile=None)
    return output.getvalue()


def _timeout_milliseconds(seconds: float) -> int:
    """Translate the app's seconds setting to google-genai's milliseconds."""

    return round(seconds * 1000)


def _normalise_generated_image(image: Image.Image, original_size: tuple[int, int]) -> Image.Image:
    """Return Gemini's same-framing RGB output at the target asset dimensions."""

    if "A" in image.getbands() and image.getchannel("A").getextrema()[0] < 255:
        raise ValueError("Generated image has transparency")
    source_ratio = image.width / image.height
    target_ratio = original_size[0] / original_size[1]
    if abs(source_ratio - target_ratio) / target_ratio > 0.01:
        raise ValueError("Generated image aspect ratio does not match the original")
    output = image.convert("RGB")
    return output if output.size == original_size else output.resize(
        original_size, Image.Resampling.LANCZOS
    )


@lru_cache(maxsize=4)
def vertex_provider(settings: GenerativeSettings) -> "VertexAIBumperProvider":
    return VertexAIBumperProvider(settings)


class VertexAIBumperProvider:
    name = "vertex-ai"

    def __init__(self, settings: GenerativeSettings) -> None:
        self.model_id = settings.model_id
        self._settings = settings
        try:
            from google import genai
            from google.genai import types
        except ImportError as exc:
            raise PipelineError(
                "generative_provider_not_configured",
                "google-genai is not installed",
                500,
            ) from exc
        self._types = types
        try:
            from google.auth import default
            from google.auth.credentials import with_scopes_if_required

            credentials, _ = default()
            credentials = with_scopes_if_required(
                credentials,
                ["https://www.googleapis.com/auth/cloud-platform"],
            )
            self._client = genai.Client(
                vertexai=True,
                credentials=credentials,
                project=settings.project_id,
                location=settings.location,
                # google-genai expects milliseconds; application settings use seconds.
                http_options=types.HttpOptions(
                    api_version="v1",
                    timeout=_timeout_milliseconds(settings.timeout_seconds),
                ),
            )
        except Exception as exc:
            raise PipelineError(
                "generative_provider_not_configured",
                "Could not configure Vertex AI credentials",
                500,
            ) from exc

    def identify_vehicle(self, images: Sequence[Image.Image]) -> VehicleIdentity:
        parts = []
        for index, image in enumerate(images, 1):
            parts.extend([
                f"Vehicle image {index}; image 1 is authoritative and later images are optional supporting views:",
                self._types.Part.from_bytes(data=_png(image.convert("RGB")), mime_type="image/png"),
            ])
        parts.append(
            "Identify only visual automotive identity: make, model, generation or year range, body style, optional trim, "
            "up to eight visible exterior cues, and confidence from 0 to 1. If the supporting images conflict, trust image 1 and lower confidence."
        )
        try:
            response = self._client.models.generate_content(
                model=self.model_id,
                contents=parts,
                config=self._types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=VehicleIdentity,
                ),
            )
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise PipelineError("generative_provider_timeout", "Vehicle identity analysis timed out", 504) from exc
        except Exception as exc:
            raise PipelineError("generative_provider_error", "Vehicle identity analysis failed", 502) from exc
        try:
            parsed = getattr(response, "parsed", None)
            if isinstance(parsed, VehicleIdentity):
                return parsed
            if isinstance(parsed, dict):
                return VehicleIdentity.model_validate(parsed)
            return VehicleIdentity.model_validate_json(response.text)
        except (AttributeError, TypeError, ValueError) as exc:
            raise PipelineError("invalid_vehicle_identity", "Vertex AI returned an invalid vehicle identity", 502) from exc

    def edit(self, *, original: Image.Image, reference: Image.Image, rough_composite: Image.Image, edit_mask: np.ndarray, instruction: str, additional_references: Sequence[Image.Image] = ()) -> Image.Image:
        if edit_mask.shape != (original.height, original.width):
            raise PipelineError("invalid_generated_image", "Edit mask dimensions are invalid", 502)
        mask = Image.fromarray(np.where(edit_mask >= 128, 255, 0).astype(np.uint8)).convert("RGB")
        parts = [
            "Image 1 — original target car:", self._types.Part.from_bytes(data=_png(original.convert("RGB")), mime_type="image/png"),
            "Image 2 — isolated reference part:", self._types.Part.from_bytes(data=_png(reference.convert("RGBA")), mime_type="image/png"),
            "Image 3 — deterministic rough placement:", self._types.Part.from_bytes(data=_png(rough_composite.convert("RGB")), mime_type="image/png"),
            "Image 4 — strict white edit mask; black pixels must remain unchanged:", self._types.Part.from_bytes(data=_png(mask), mime_type="image/png"),
        ]
        for index, image in enumerate(additional_references, 5):
            parts.extend([
                f"Image {index} — optional vehicle identity reference:",
                self._types.Part.from_bytes(data=_png(image.convert("RGB")), mime_type="image/png"),
            ])
        parts.append(instruction)
        try:
            response = self._client.models.generate_content(
                model=self.model_id,
                contents=parts,
                config=self._types.GenerateContentConfig(
                    response_modalities=[self._types.Modality.TEXT, self._types.Modality.IMAGE],
                    candidate_count=1,
                ),
            )
        except (TimeoutError, httpx.TimeoutException) as exc:
            raise PipelineError("generative_provider_timeout", "Vertex AI image generation timed out", 504) from exc
        except Exception as exc:
            raise PipelineError("generative_provider_error", "Vertex AI image generation failed", 502) from exc
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            raise PipelineError("invalid_generated_image", "Vertex AI returned no image", 502)
        images = [
            part.inline_data.data
            for part in (getattr(getattr(candidates[0], "content", None), "parts", None) or [])
            if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None)
        ]
        if not images:
            raise PipelineError("invalid_generated_image", "Vertex AI returned no image", 502)
        try:
            # Gemini can return explanatory text or multiple image parts in one
            # candidate despite candidate_count=1. The first image is the primary
            # result; the renderer still clamps it to the local edit mask.
            with Image.open(BytesIO(images[0])) as opened:
                opened.load()
                return _normalise_generated_image(opened, original.size)
        except (OSError, ValueError) as exc:
            raise PipelineError("invalid_generated_image", "Vertex AI returned an invalid image", 502) from exc
