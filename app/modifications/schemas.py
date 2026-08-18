"""Validated public models for deterministic paint modifications."""

import json
import re
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from app.errors import PipelineError

HEX_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")


class PaintFinish(StrEnum):
    """Supported deterministic surface finish descriptions."""

    GLOSSY = "glossy"
    MATTE = "matte"
    METALLIC = "metallic"


def normalise_hex(value: str) -> str:
    """Validate and canonicalize a six-digit RGB hex colour."""

    if not HEX_RE.fullmatch(value):
        raise ValueError("Colour must be a six-digit RGB hex colour")
    return "#" + value.removeprefix("#").lower()


class SurfaceEditRequest(BaseModel):
    """Validated paint-only edit accepted by the deterministic renderer."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["surface_edit"] = "surface_edit"
    body_colour: str | None = None
    roof_colour: str | None = None
    finish: PaintFinish = PaintFinish.GLOSSY

    @field_validator("body_colour")
    @classmethod
    def validate_body_colour(cls, value: str | None) -> str | None:
        """Canonicalize an optional body colour."""

        return normalise_hex(value) if value is not None else None

    @field_validator("roof_colour")
    @classmethod
    def validate_roof_colour(cls, value: str | None) -> str | None:
        """Canonicalize an optional independently targeted roof colour."""

        return normalise_hex(value) if value is not None else None

    @model_validator(mode="after")
    def validate_request(self) -> "SurfaceEditRequest":
        """Reject empty edits."""

        if not self.body_colour and not self.roof_colour:
            raise ValueError("Surface edit request is empty")
        return self


class BumperPosition(StrEnum):
    """The only vehicle ends supported by the bumper preview MVP."""

    FRONT = "front"
    REAR = "rear"


class BumperPaintMode(StrEnum):
    """How the generated bumper's visible finish should be guided."""

    MATCH_BODY = "match_body"
    PRESERVE_REFERENCE = "preserve_reference"


class BumperReplacementRequest(BaseModel):
    """Validated, prompt-free request for a constrained bumper preview."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["bumper_replacement"] = "bumper_replacement"
    bumper_position: BumperPosition
    reference_asset_id: str
    paint_mode: BumperPaintMode = BumperPaintMode.MATCH_BODY

    @field_validator("reference_asset_id")
    @classmethod
    def validate_reference_asset_id(cls, value: str) -> str:
        """Keep content-addressed references safely inside their asset directory."""

        normalized = value.lower()
        if not re.fullmatch(r"[0-9a-f]{64}", normalized):
            raise ValueError("Reference asset ID must be a SHA-256 identifier")
        return normalized


class RimReplacementRequest(BaseModel):
    """Validated, prompt-free request for a constrained rim preview."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["rim_replacement"] = "rim_replacement"
    reference_asset_id: str

    @field_validator("reference_asset_id")
    @classmethod
    def validate_reference_asset_id(cls, value: str) -> str:
        normalized = value.lower()
        if not re.fullmatch(r"[0-9a-f]{64}", normalized):
            raise ValueError("Reference asset ID must be a SHA-256 identifier")
        return normalized


class StudioRenderStyle(StrEnum):
    """Supported studio presentation presets."""

    LIGHT_STUDIO = "light_studio"
    DARK_STUDIO = "dark_studio"
    PREMIUM_GRADIENT = "premium_gradient"


class StudioRenderFidelity(StrEnum):
    """Identity preservation strength for studio renders."""

    HIGH = "high"


class VehicleIdentity(BaseModel):
    """User-confirmable visual identity cues used to guide studio rendering."""

    model_config = ConfigDict(extra="forbid")

    make: str = Field(min_length=1, max_length=80)
    model: str = Field(min_length=1, max_length=80)
    generation: str = Field(min_length=1, max_length=120)
    body_style: str = Field(min_length=1, max_length=80)
    trim: str | None = Field(default=None, max_length=120)
    visual_cues: list[str] = Field(default_factory=list, max_length=8)
    confidence: float = Field(ge=0, le=1)

    @field_validator("make", "model", "generation", "body_style")
    @classmethod
    def strip_identity_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Vehicle identity fields cannot be blank")
        return cleaned

    @field_validator("trim")
    @classmethod
    def strip_optional_trim(cls, value: str | None) -> str | None:
        return value.strip() or None if value is not None else None

    @field_validator("visual_cues")
    @classmethod
    def clean_visual_cues(cls, values: list[str]) -> list[str]:
        cleaned = [value.strip() for value in values if value.strip()]
        if any(len(value) > 160 for value in cleaned):
            raise ValueError("Visual cues must be at most 160 characters")
        return cleaned


class StudioRenderRequest(BaseModel):
    """Validated request for an identity-preserving studio presentation render."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["studio_render"] = "studio_render"
    style: StudioRenderStyle = StudioRenderStyle.LIGHT_STUDIO
    fidelity: StudioRenderFidelity = StudioRenderFidelity.HIGH
    preserve_plate: bool = True
    vehicle_identity: VehicleIdentity | None = None
    reference_asset_ids: list[str] = Field(default_factory=list, max_length=4)

    @field_validator("reference_asset_ids")
    @classmethod
    def validate_reference_asset_ids(cls, values: list[str]) -> list[str]:
        normalized = [value.lower() for value in values]
        if any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in normalized):
            raise ValueError("Reference asset IDs must be SHA-256 identifiers")
        if len(set(normalized)) != len(normalized):
            raise ValueError("Reference asset IDs must be unique")
        return normalized


ModificationRequest = Annotated[
    SurfaceEditRequest
    | BumperReplacementRequest
    | RimReplacementRequest
    | StudioRenderRequest,
    Field(discriminator="type"),
]
_MODIFICATION_ADAPTER = TypeAdapter(ModificationRequest)


def parse_modification(data: object) -> ModificationRequest:
    """Validate an untrusted JSON value and expose stable domain errors."""

    if not isinstance(data, dict):
        raise PipelineError("invalid_modification", "Request body must be an object")
    try:
        return _MODIFICATION_ADAPTER.validate_python(data)
    except ValueError as exc:
        raise PipelineError("invalid_modification", str(exc)) from exc


def normalised_request_json(modification: ModificationRequest) -> str:
    """Serialize a request canonically for cache identity and audit files."""

    return json.dumps(
        modification.model_dump(mode="json", exclude_none=True),
        sort_keys=True,
        separators=(",", ":"),
    )
