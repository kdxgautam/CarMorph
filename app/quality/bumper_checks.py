"""Quality gates specific to locally clamped bumper previews."""

import numpy as np
from PIL import Image

from app.quality.checks import QualityResult, QualityStatus


def check_bumper_render(*, original: Image.Image, generated: Image.Image, result: Image.Image, core_mask: np.ndarray, allowed_mask: np.ndarray, protected_mask: np.ndarray, minimum_core_change_ratio: float) -> QualityResult:
    if result.size != original.size:
        return QualityResult(QualityStatus.FAILED, ["result_dimensions_mismatch"])
    shape = (original.height, original.width)
    if any(mask.shape != shape for mask in (core_mask, allowed_mask, protected_mask)):
        return QualityResult(QualityStatus.FAILED, ["bumper_region_invalid"])
    core, allowed, protected = core_mask >= 128, allowed_mask >= 128, protected_mask >= 128
    if not np.any(core) or not np.any(allowed):
        return QualityResult(QualityStatus.FAILED, ["empty_bumper_core"])
    source = np.asarray(original.convert("RGB"))
    raw = np.asarray(generated.convert("RGB"))
    final = np.asarray(result.convert("RGB"))
    changed = np.any(final != source, axis=2)
    warnings = []
    if np.any(changed & ~allowed):
        return QualityResult(QualityStatus.FAILED, ["outside_bumper_mask_changed"])
    if np.any(changed & protected):
        return QualityResult(QualityStatus.FAILED, ["bumper_protected_region_changed"])
    ratio = float(np.count_nonzero(changed & core) / np.count_nonzero(core))
    if ratio == 0:
        return QualityResult(QualityStatus.FAILED, ["bumper_result_unchanged"])
    if ratio < minimum_core_change_ratio:
        return QualityResult(QualityStatus.FAILED, ["bumper_core_change_too_small"])
    if np.all(final[core] <= 2):
        return QualityResult(QualityStatus.FAILED, ["bumper_region_invalid"])
    if np.any(np.any(raw != source, axis=2) & ~allowed):
        warnings.append("provider_changes_outside_mask_restored")
    if np.any(np.any(raw != source, axis=2) & protected):
        warnings.append("provider_protected_changes_restored")
    return QualityResult(QualityStatus.PASSED_WITH_WARNINGS if warnings else QualityStatus.PASSED, warnings)
