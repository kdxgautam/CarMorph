from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from PIL import Image

from app.paint_analysis.mask_builder import PaintAnalysisMasks
from app.paint_analysis.schemas import PaintGroup


class QualityStatus(StrEnum):
    PASSED = "passed"
    PASSED_WITH_WARNINGS = "passed_with_warnings"
    FAILED = "failed"


@dataclass
class QualityResult:
    status: QualityStatus
    warnings: list[str] = field(default_factory=list)

    def model_dump(self) -> dict:
        return {"status": self.status.value, "warnings": self.warnings}


def check_paint_analysis(
    group_masks: dict[PaintGroup, np.ndarray],
    masks: PaintAnalysisMasks,
    *,
    profile_confidence: float | None = None,
    confidence_threshold: float = 0.45,
    seed_mask: np.ndarray | None = None,
    hard_protected_mask: np.ndarray | None = None,
    paint_like_residual_mask: np.ndarray | None = None,
    minimum_residual_pixels: int = 64,
) -> list[str]:
    warnings = []
    occupied = np.zeros(masks.editable.shape, np.uint8)
    for mask in group_masks.values():
        if np.any((occupied > 0) & (mask >= 128)):
            warnings.append("paint_groups_do_not_overlap_failed")
            break
        occupied[mask >= 128] = 1
    if not np.any(masks.editable >= 128):
        warnings.append("main_body_mask_non_empty_failed")
    if (
        profile_confidence is not None
        and profile_confidence < confidence_threshold
    ):
        warnings.append("body_paint_profile_confident_failed")
    if np.any((masks.protected >= 128) & (masks.editable >= 128)):
        warnings.append("protected_groups_override_editable_groups_failed")
    if np.any((masks.uncertain >= 128) & (masks.editable >= 128)):
        warnings.append("uncertain_groups_not_editable_failed")
    if seed_mask is not None and np.any(
        (seed_mask >= 128) & (masks.editable < 128)
    ):
        warnings.append("main_body_seeds_not_preserved_failed")
    if hard_protected_mask is not None and np.any(
        (hard_protected_mask >= 128) & (masks.editable >= 128)
    ):
        warnings.append("surface_growth_crossed_hard_protection_failed")
    if (
        paint_like_residual_mask is not None
        and np.count_nonzero(paint_like_residual_mask >= 128)
        >= minimum_residual_pixels
    ):
        warnings.append("paint_like_residual_region_detected")
    for group, warning in (
        (PaintGroup.BODY_COLOURED_HANDLE, "body_coloured_handles_follow_request_failed"),
        (
            PaintGroup.CONTRASTING_HANDLE,
            "contrasting_handles_remain_unchanged_failed",
        ),
        (
            PaintGroup.CONTRAST_ROOF_PAINT,
            "contrast_roof_remains_unchanged_when_not_requested_failed",
        ),
    ):
        mask = group_masks.get(group)
        if mask is None:
            continue
        should_edit = group == PaintGroup.BODY_COLOURED_HANDLE
        if np.any((mask >= 128) & ((masks.editable >= 128) != should_edit)):
            warnings.append(warning)
    return warnings


def check_render(
    *,
    original: Image.Image,
    result: Image.Image,
    editable_mask: np.ndarray,
    protected_mask: np.ndarray | None = None,
) -> QualityResult:
    warnings = []
    if result.size != original.size:
        return QualityResult(QualityStatus.FAILED, ["result_dimensions_mismatch"])
    if editable_mask.shape != (original.height, original.width):
        return QualityResult(QualityStatus.FAILED, ["editable_mask_dimensions_mismatch"])
    editable = editable_mask > 0
    if not np.any(editable):
        return QualityResult(QualityStatus.FAILED, ["empty_editable_region"])

    original_pixels = np.asarray(original.convert("RGB"))
    result_pixels = np.asarray(result.convert("RGB"))
    if not np.array_equal(result_pixels[~editable], original_pixels[~editable]):
        return QualityResult(QualityStatus.FAILED, ["outside_mask_changed"])

    if protected_mask is not None:
        protected = protected_mask > 0
        if protected.shape != editable.shape:
            return QualityResult(QualityStatus.FAILED, ["protected_mask_dimensions_mismatch"])
        if np.any(protected) and not np.array_equal(
            result_pixels[protected], original_pixels[protected]
        ):
            return QualityResult(QualityStatus.FAILED, ["protected_region_changed"])

    return QualityResult(
        QualityStatus.PASSED_WITH_WARNINGS if warnings else QualityStatus.PASSED,
        warnings,
    )
