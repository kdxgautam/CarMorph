"""Validated public models for deterministic paint modifications."""

import json
import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

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


def parse_modification(data: object) -> SurfaceEditRequest:
    """Validate an untrusted JSON value and expose stable domain errors."""

    if not isinstance(data, dict):
        raise PipelineError("invalid_modification", "Request body must be an object")
    try:
        return SurfaceEditRequest.model_validate(data)
    except ValueError as exc:
        raise PipelineError("invalid_modification", str(exc)) from exc


def normalised_request_json(modification: SurfaceEditRequest) -> str:
    """Serialize a request canonically for cache identity and audit files."""

    return json.dumps(
        modification.model_dump(mode="json", exclude_none=True),
        sort_keys=True,
        separators=(",", ":"),
    )
