"""Convert classified paint groups into disjoint request-specific masks."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.errors import PipelineError
from app.schemas import AssetBundle

from app.paint_analysis.schemas import PaintGroup


DEFAULT_BODY_GROUPS = {
    PaintGroup.MAIN_BODY_PAINT,
    PaintGroup.BODY_COLOURED_HANDLE,
    PaintGroup.BODY_COLOURED_MIRROR_CAP,
    PaintGroup.PAINTED_BUMPER_SECTION,
    PaintGroup.PAINTED_SPOILER,
}
PROTECTED_GROUPS = {
    PaintGroup.BLACK_PLASTIC_TRIM,
    PaintGroup.GLOSSY_BLACK_TRIM,
    PaintGroup.CHROME_TRIM,
    PaintGroup.SILVER_GARNISH,
    PaintGroup.GLASS,
    PaintGroup.RUBBER,
    PaintGroup.LIGHT_LENS,
    PaintGroup.WHEEL,
    PaintGroup.TYRE,
    PaintGroup.GRILLE,
    PaintGroup.NUMBER_PLATE,
    PaintGroup.BADGE,
    PaintGroup.CONTRASTING_HANDLE,
    PaintGroup.CONTRASTING_MIRROR_CAP,
    PaintGroup.PAINTED_TRIM,
    PaintGroup.SECONDARY_BODY_PAINT,
}


@dataclass(frozen=True)
class PaintAnalysisMasks:
    """Final disjoint masks consumed by renderers and quality checks."""

    editable: np.ndarray
    protected: np.ndarray
    uncertain: np.ndarray


def union_group_masks(
    group_masks: dict[PaintGroup, np.ndarray], groups: set[PaintGroup], shape: tuple[int, int]
) -> np.ndarray:
    """Union selected paint groups into a boolean mask of the requested shape."""

    selected = [mask >= 128 for group, mask in group_masks.items() if group in groups]
    return np.maximum.reduce(selected) if selected else np.zeros(shape, dtype=bool)


def build_default_masks(
    full_car: np.ndarray, group_masks: dict[PaintGroup, np.ndarray]
) -> PaintAnalysisMasks:
    """Apply fixed precedence to produce default body-edit masks."""

    editable = union_group_masks(group_masks, DEFAULT_BODY_GROUPS, full_car.shape)
    protected = union_group_masks(group_masks, PROTECTED_GROUPS, full_car.shape)
    uncertain = union_group_masks(group_masks, {PaintGroup.UNKNOWN}, full_car.shape)
    # Precedence is deliberate: protected > uncertain > editable.
    protected &= full_car >= 128
    uncertain &= (full_car >= 128) & ~protected
    editable &= ~protected & ~uncertain
    return PaintAnalysisMasks(
        np.where(editable, 255, 0).astype(np.uint8),
        np.where(protected, 255, 0).astype(np.uint8),
        np.where(uncertain, 255, 0).astype(np.uint8),
    )


def build_request_masks(
    group_masks: dict[PaintGroup, np.ndarray],
    fallback: np.ndarray,
    *,
    include_roof: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Select body and optional contrast-roof masks for one render request."""

    if not group_masks:
        return fallback, np.zeros_like(fallback)
    body = union_group_masks(group_masks, DEFAULT_BODY_GROUPS, fallback.shape)
    roof = union_group_masks(
        group_masks,
        {PaintGroup.CONTRAST_ROOF_PAINT},
        fallback.shape,
    ) if include_roof else np.zeros(fallback.shape, dtype=bool)
    return (
        np.where(body, 255, 0).astype(np.uint8),
        np.where(roof, 255, 0).astype(np.uint8),
    )


def load_request_masks(
    directory: Path, metadata: AssetBundle, *, include_roof: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Load persisted paint groups with a legacy editable-mask fallback."""

    fallback_path = metadata.masks.get("editable_mask") or metadata.masks.get(
        "paintable_body"
    )
    if not fallback_path:
        raise PipelineError("missing_masks", "Editable mask is missing", 500)
    fallback = cv2.imread(str(directory / fallback_path), cv2.IMREAD_GRAYSCALE)
    if fallback is None:
        raise PipelineError("missing_masks", "Editable mask is missing", 500)
    group_masks = {}
    for group in PaintGroup:
        path = metadata.masks.get(group.value)
        if path:
            mask = cv2.imread(str(directory / path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise PipelineError("missing_masks", "A paint-group mask is missing", 500)
            group_masks[group] = mask
    return build_request_masks(
        group_masks, fallback, include_roof=include_roof
    )
