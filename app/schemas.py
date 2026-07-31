from typing import Literal

from pydantic import BaseModel, Field

ViewName = Literal["front", "left", "right", "rear"]


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


class PaintabilityReport(BaseModel):
    editable_ratio: float = 0
    protected_ratio: float = 0
    uncertain_ratio: float = 0
    warnings: list[str] = Field(default_factory=list)
    rules_version: str = "legacy"


class AvailableModifications(BaseModel):
    body_colour: bool = True
    finish: bool = True
    racing_stripes: bool = True
    custom_instruction: bool = True
    roof_colour: bool = False
    rim_replacement: bool = False
    bumper_replacement: bool = False


class AssetBundle(BaseModel):
    asset_id: str
    view: ViewName
    status: Literal["ready"] = "ready"
    width: int
    height: int
    car_bbox: BoundingBox
    source_image: str
    original_image: str
    luminance_map: str
    masks: dict[str, str]
    models: dict[str, str]
    warnings: list[str] = Field(default_factory=list)
    paintability_report: PaintabilityReport = Field(default_factory=PaintabilityReport)
    available_modifications: AvailableModifications = Field(
        default_factory=AvailableModifications
    )
    pipeline_version: str = "legacy"
