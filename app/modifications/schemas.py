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


ModificationRequest = Annotated[
    SurfaceEditRequest | BumperReplacementRequest,
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
