from dataclasses import dataclass, field
from enum import StrEnum

import numpy as np
from PIL import Image


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
