"""Validated public models for paint and design modifications."""

import json
import re
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.errors import PipelineError

HEX_RE = re.compile(r"^#?[0-9a-fA-F]{6}$")
MAX_CUSTOM_INSTRUCTION_LENGTH = 240


class PaintFinish(StrEnum):
    """Supported deterministic/generative surface finish descriptions."""

    GLOSSY = "glossy"
    MATTE = "matte"
    METALLIC = "metallic"


class RendererMode(StrEnum):
    """Caller preference for deterministic or generative execution."""

    AUTO = "auto"
    DETERMINISTIC = "deterministic"
    GENERATIVE = "generative"


class StripePlacement(StrEnum):
    """Supported visible surfaces for racing stripes."""

    BONNET = "bonnet"
    VISIBLE_ROOF = "visible_roof"
    BONNET_AND_VISIBLE_ROOF = "bonnet_and_visible_roof"
    VISIBLE_SIDE_PANELS = "visible_side_panels"


class StripeAlignment(StrEnum):
    """Supported stripe alignment relative to visible body geometry."""

    CENTRE = "centre"
    LOWER_SIDE = "lower_side"


class StripeWidth(StrEnum):
    """Provider-independent qualitative stripe widths."""

    THIN = "thin"
    MEDIUM = "medium"
    THICK = "thick"


def normalise_hex(value: str) -> str:
    """Validate and canonicalize a six-digit RGB hex colour."""

    if not HEX_RE.fullmatch(value):
        raise ValueError("Colour must be a six-digit RGB hex colour")
    return "#" + value.removeprefix("#").lower()


class RacingStripeElement(BaseModel):
    """Strict structured request for one or two racing stripes."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["racing_stripes"] = "racing_stripes"
    count: Literal[1, 2]
    colour: str
    width: StripeWidth
    placement: StripePlacement
    alignment: StripeAlignment

    @field_validator("colour")
    @classmethod
    def validate_colour(cls, value: str) -> str:
        """Canonicalize stripe colour during model validation."""

        return normalise_hex(value)


DesignElement = Annotated[RacingStripeElement, Field(discriminator="type")]


class SurfaceEditRequest(BaseModel):
    """Validated paint-only edit accepted by renderer planning."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["surface_edit"] = "surface_edit"
    body_colour: str | None = None
    roof_colour: str | None = None
    finish: PaintFinish = PaintFinish.GLOSSY
    design_elements: list[DesignElement] = Field(default_factory=list)
    custom_instruction: str | None = None
    renderer: RendererMode = RendererMode.AUTO

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

    @field_validator("custom_instruction")
    @classmethod
    def validate_custom_instruction(cls, value: str | None) -> str | None:
        """Normalize bounded instruction text and collapse blank input."""

        if value is None:
            return None
        value = value.strip()
        if not value:
            return None
        if len(value) > MAX_CUSTOM_INSTRUCTION_LENGTH:
            raise ValueError("Custom instruction is too long")
        return value

    @model_validator(mode="after")
    def validate_request(self) -> "SurfaceEditRequest":
        """Reject empty edits and conflicting duplicate stripe definitions."""

        if (
            not self.body_colour
            and not self.roof_colour
            and not self.design_elements
            and not self.custom_instruction
        ):
            raise ValueError("Surface edit request is empty")

        seen: dict[tuple[StripePlacement, StripeAlignment], dict] = {}
        for element in self.design_elements:
            key = (element.placement, element.alignment)
            value = element.model_dump(mode="json")
            if key in seen and seen[key] != value:
                raise ValueError("Conflicting duplicate stripe definitions")
            seen[key] = value
        return self


class PartReplacementRequest(BaseModel):
    """Reserved request shape rejected until physical replacement is supported."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["part_replacement"] = "part_replacement"
    target: str
    asset_id: str


def parse_modification(data: object) -> SurfaceEditRequest:
    """Validate an untrusted JSON value and expose stable domain errors."""

    if not isinstance(data, dict):
        raise PipelineError("invalid_modification", "Request body must be an object")
    if data.get("type") == "part_replacement":
        raise PipelineError(
            "future_not_supported",
            "Physical part replacement is not supported in this milestone",
        )
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
