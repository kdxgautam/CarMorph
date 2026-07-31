from enum import StrEnum

from pydantic import BaseModel, Field


class PaintGroup(StrEnum):
    MAIN_BODY_PAINT = "main_body_paint"
    SECONDARY_BODY_PAINT = "secondary_body_paint"
    CONTRAST_ROOF_PAINT = "contrast_roof_paint"
    BODY_COLOURED_HANDLE = "body_coloured_handle"
    CONTRASTING_HANDLE = "contrasting_handle"
    BODY_COLOURED_MIRROR_CAP = "body_coloured_mirror_cap"
    CONTRASTING_MIRROR_CAP = "contrasting_mirror_cap"
    PAINTED_BUMPER_SECTION = "painted_bumper_section"
    PAINTED_SPOILER = "painted_spoiler"
    PAINTED_TRIM = "painted_trim"
    BLACK_PLASTIC_TRIM = "black_plastic_trim"
    GLOSSY_BLACK_TRIM = "glossy_black_trim"
    CHROME_TRIM = "chrome_trim"
    SILVER_GARNISH = "silver_garnish"
    GLASS = "glass"
    RUBBER = "rubber"
    LIGHT_LENS = "light_lens"
    WHEEL = "wheel"
    TYRE = "tyre"
    GRILLE = "grille"
    NUMBER_PLATE = "number_plate"
    BADGE = "badge"
    UNKNOWN = "unknown"


class MaterialType(StrEnum):
    PAINTED_SURFACE = "painted_surface"
    MATTE_PLASTIC = "matte_plastic"
    GLOSSY_PLASTIC = "glossy_plastic"
    CHROME = "chrome"
    METAL = "metal"
    GLASS = "glass"
    RUBBER = "rubber"
    LIGHT_LENS = "light_lens"
    UNKNOWN = "unknown"


class Paintability(StrEnum):
    EDITABLE = "editable"
    SEPARATELY_EDITABLE = "separately_editable"
    PROTECTED = "protected"
    UNCERTAIN = "uncertain"


class BodyPaintProfile(BaseModel):
    dominant_lab: list[float] = Field(default_factory=list)
    median_lab: list[float] = Field(default_factory=list)
    lab_variance: list[float] = Field(default_factory=list)
    dominant_hsv: list[float] = Field(default_factory=list)
    highlight_lab_range: dict[str, float] = Field(default_factory=dict)
    midtone_lab_range: dict[str, float] = Field(default_factory=dict)
    shadow_lab_range: dict[str, float] = Field(default_factory=dict)
    sample_count: int = 0
    anchor_regions: list[str] = Field(default_factory=list)
    confidence: float = 0
    warnings: list[str] = Field(default_factory=list)


class RegionClassification(BaseModel):
    region_id: str
    part_type: str
    paint_group: PaintGroup
    material_type: MaterialType
    body_colour_similarity: float = 0
    texture_similarity: float = 0
    lightness_difference: float = 0
    confidence: float = 0
    paintability: Paintability
    reason_codes: list[str] = Field(default_factory=list)


class PaintGroupSummary(BaseModel):
    paint_group: PaintGroup
    pixel_count: int
    ratio_of_car: float
    confidence: float


class SurfaceRegionDecision(BaseModel):
    region_id: str
    pixel_count: int
    seed_coverage: float
    accepted_boundary_ratio: float
    protected_overlap_ratio: float
    global_chroma_similarity: float
    local_colour_consistency: float
    mean_gradient_boundary: float
    semantic_body_likelihood: float
    decision: PaintGroup
    confidence: float
    reason_codes: list[str] = Field(default_factory=list)


class FragmentationMetrics(BaseModel):
    seed_pixel_count: int = 0
    final_pixel_count: int = 0
    recovered_pixel_count: int = 0
    components_before: int = 0
    components_after: int = 0
    small_components_before: int = 0
    small_components_after: int = 0
    internal_gap_pixels_before: int = 0
    internal_gap_pixels_after: int = 0
    growth_iterations: int = 0


class SurfaceCompletionReport(BaseModel):
    regions: list[SurfaceRegionDecision] = Field(default_factory=list)
    fragmentation: FragmentationMetrics = Field(default_factory=FragmentationMetrics)


class PaintGroupReport(BaseModel):
    body_paint_profile: BodyPaintProfile = Field(default_factory=BodyPaintProfile)
    groups: list[PaintGroupSummary] = Field(default_factory=list)
    region_classifications: list[RegionClassification] = Field(default_factory=list)
    surface_completion: SurfaceCompletionReport | None = None
    warnings: list[str] = Field(default_factory=list)
    rules_version: str = "paint-groups-v2"
