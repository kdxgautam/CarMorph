"""Persisted metadata for content-addressed bumper references."""

from typing import Literal

from pydantic import BaseModel, Field


class BumperReferenceResponse(BaseModel):
    reference_asset_id: str
    width: int
    height: int
    has_alpha: bool


class BumperReferenceReport(BaseModel):
    reference_asset_id: str
    content_sha256: str
    source_width: int
    source_height: int
    normalized_width: int
    normalized_height: int
    has_alpha: bool
    segmentation_method: Literal["source_alpha", "plain_background", "sam3"]
    mask_coverage_ratio: float
    warnings: list[str] = Field(default_factory=list)
    processing_version: str
    normalized_image: str = "normalized.png"
    reference_mask: str = "reference-mask.png"
    source_image: str
