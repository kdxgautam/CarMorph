from typing import Literal

from pydantic import BaseModel

ViewName = Literal["front", "left", "right", "rear"]


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float


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
    warnings: list[str] = []
