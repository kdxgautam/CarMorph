from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from app.paint_analysis.body_colour_estimator import estimate_body_paint
from app.paint_analysis.colour_profile import chroma_distance, rgb_to_lab
from app.paint_analysis.mask_builder import (
    PROTECTED_GROUPS,
    PaintAnalysisMasks,
    build_default_masks,
    union_group_masks,
)
from app.paint_analysis.material_classifier import classify_material
from app.paint_analysis.schemas import (
    MaterialType,
    PaintGroup,
    PaintGroupReport,
    PaintGroupSummary,
    Paintability,
    RegionClassification,
)
from app.paint_analysis.surface_completion import (
    SurfaceCompletionResult,
    complete_body_surface,
)


PROTECTED_SEMANTICS = {
    "windows": PaintGroup.GLASS,
    "wheels": PaintGroup.WHEEL,
    "plate": PaintGroup.NUMBER_PLATE,
    "lights": PaintGroup.LIGHT_LENS,
    "grille": PaintGroup.GRILLE,
    "pillars": PaintGroup.GLOSSY_BLACK_TRIM,
}


@dataclass(frozen=True)
class PaintAnalysisResult:
    report: PaintGroupReport
    group_masks: dict[PaintGroup, np.ndarray]
    masks: PaintAnalysisMasks
    anchors: np.ndarray
    surface: SurfaceCompletionResult


def _relation(
    lab: np.ndarray, mask: np.ndarray, profile_lab: np.ndarray, threshold: float
) -> tuple[float, float, float]:
    pixels = lab[mask >= 128]
    if not len(pixels) or not len(profile_lab):
        return 0, 0, 0
    median = np.median(pixels, axis=0)
    chroma = float(chroma_distance(median, profile_lab))
    lightness = abs(float(median[0] - profile_lab[0]))
    return max(0, 1 - chroma / max(threshold * 2, 1)), chroma, lightness


def analyse_paint_groups(
    image: Image.Image,
    full_car: np.ndarray,
    part_masks: dict[str, np.ndarray],
    settings: object,
) -> PaintAnalysisResult:
    shape = full_car.shape
    car_pixels = max(1, int(np.count_nonzero(full_car >= 128)))
    known = [
        mask >= 128
        for name, mask in part_masks.items()
        if name != "dark_trim"
    ]
    protected_seed = np.maximum.reduce(known) if known else np.zeros(shape, bool)
    body_candidates = np.where(
        (full_car >= 128) & ~protected_seed, 255, 0
    ).astype(np.uint8)
    profile, anchors = estimate_body_paint(
        image,
        body_candidates,
        erosion_pixels=settings.anchor_erosion_pixels,
        min_samples=settings.anchor_min_sample_count,
        chroma_threshold=settings.body_paint_chroma_threshold,
    )
    lab = rgb_to_lab(np.asarray(image.convert("RGB")))
    profile_lab = np.asarray(profile.median_lab)
    groups: dict[PaintGroup, np.ndarray] = {}
    regions = []
    claimed = np.zeros(shape, bool)

    def add(group: PaintGroup, mask: np.ndarray) -> None:
        binary = (mask >= 128) & (full_car >= 128) & ~claimed
        if np.any(binary):
            groups[group] = np.maximum(
                groups.get(group, np.zeros(shape, np.uint8)),
                np.where(binary, 255, 0).astype(np.uint8),
            )
            claimed[binary] = True

    for part_type, group in PROTECTED_SEMANTICS.items():
        mask = part_masks.get(part_type)
        if mask is None:
            continue
        material, confidence, reasons = classify_material(image, mask, part_type)
        add(group, mask)
        regions.append(
            RegionClassification(
                region_id=part_type,
                part_type=part_type,
                paint_group=group,
                material_type=material,
                confidence=confidence,
                paintability=Paintability.PROTECTED,
                reason_codes=reasons,
            )
        )

    for part_type in ("trim",):
        mask = part_masks.get(part_type)
        if mask is None:
            continue
        material, confidence, reasons = classify_material(image, mask, part_type)
        group = (
            PaintGroup.CHROME_TRIM
            if material == MaterialType.CHROME
            else PaintGroup.GLOSSY_BLACK_TRIM
            if material == MaterialType.GLOSSY_PLASTIC
            else PaintGroup.BLACK_PLASTIC_TRIM
        )
        add(group, mask)
        regions.append(
            RegionClassification(
                region_id=part_type,
                part_type=part_type,
                paint_group=group,
                material_type=material,
                confidence=confidence,
                paintability=Paintability.PROTECTED,
                reason_codes=reasons,
            )
        )

    configurable = {
        "handles": (PaintGroup.BODY_COLOURED_HANDLE, PaintGroup.CONTRASTING_HANDLE),
        "mirrors": (
            PaintGroup.BODY_COLOURED_MIRROR_CAP,
            PaintGroup.CONTRASTING_MIRROR_CAP,
        ),
        "roof": (PaintGroup.MAIN_BODY_PAINT, PaintGroup.CONTRAST_ROOF_PAINT),
        "spoiler": (PaintGroup.PAINTED_SPOILER, PaintGroup.SECONDARY_BODY_PAINT),
    }
    for part_type, (matching_group, other_group) in configurable.items():
        mask = part_masks.get(part_type)
        if mask is None:
            continue
        material, material_confidence, reasons = classify_material(image, mask, part_type)
        if part_type == "roof" and material in {
            MaterialType.MATTE_PLASTIC,
            MaterialType.GLOSSY_PLASTIC,
        }:
            material = MaterialType.PAINTED_SURFACE
            material_confidence = 0.65
            reasons = ["roof_semantics_prevent_colour_only_plastic_label"]
        similarity, chroma, lightness = _relation(
            lab, mask, profile_lab, settings.body_paint_chroma_threshold
        )
        if material == MaterialType.CHROME:
            group, paintability = PaintGroup.CHROME_TRIM, Paintability.PROTECTED
            confidence = material_confidence
            add(group, mask)
            regions.append(
                RegionClassification(
                    region_id=part_type,
                    part_type=part_type,
                    paint_group=group,
                    material_type=material,
                    confidence=confidence,
                    paintability=paintability,
                    reason_codes=reasons + ["material_overrides_colour"],
                )
            )
            continue
        if material in {MaterialType.MATTE_PLASTIC, MaterialType.GLOSSY_PLASTIC}:
            group = (
                PaintGroup.GLOSSY_BLACK_TRIM
                if material == MaterialType.GLOSSY_PLASTIC
                else PaintGroup.BLACK_PLASTIC_TRIM
            )
            add(group, mask)
            regions.append(
                RegionClassification(
                    region_id=part_type,
                    part_type=part_type,
                    paint_group=group,
                    material_type=material,
                    confidence=material_confidence,
                    paintability=Paintability.PROTECTED,
                    reason_codes=reasons + ["material_overrides_colour"],
                )
            )
            continue
        paint_like = material == MaterialType.PAINTED_SURFACE
        matching = paint_like and (
            chroma <= settings.body_paint_strict_chroma_threshold
            or (
                similarity >= settings.paint_group_min_confidence
                and lightness <= settings.body_paint_lightness_threshold
            )
        )
        group = matching_group if matching else other_group
        paintability = (
            Paintability.EDITABLE
            if matching and part_type != "roof"
            else Paintability.SEPARATELY_EDITABLE
        )
        confidence = min(material_confidence, similarity if matching else 1 - similarity)
        if confidence < settings.paint_group_uncertain_threshold:
            group, paintability = PaintGroup.UNKNOWN, Paintability.UNCERTAIN
        add(group, mask)
        regions.append(
            RegionClassification(
                region_id=part_type,
                part_type=part_type,
                paint_group=group,
                material_type=material,
                body_colour_similarity=round(similarity, 4),
                lightness_difference=round(lightness, 3),
                confidence=round(confidence, 4),
                paintability=paintability,
                reason_codes=reasons + ["part_semantics_applied"],
            )
        )

    bumper = part_masks.get("bumper")
    if bumper is not None:
        similarity = chroma_distance(lab, profile_lab)
        painted = (bumper >= 128) & (
            similarity <= settings.body_paint_chroma_threshold
        )
        add(PaintGroup.PAINTED_BUMPER_SECTION, np.where(painted, 255, 0).astype(np.uint8))
        add(PaintGroup.BLACK_PLASTIC_TRIM, bumper)

    residual = (full_car >= 128) & ~claimed
    hard_protected = union_group_masks(
        groups,
        PROTECTED_GROUPS | {PaintGroup.CONTRAST_ROOF_PAINT},
        shape,
    )
    surface = complete_body_surface(
        lab,
        np.where(residual, 255, 0).astype(np.uint8),
        np.where(hard_protected, 255, 0).astype(np.uint8),
        profile,
        settings,
    )
    if len(profile_lab) and profile.confidence >= settings.paint_group_uncertain_threshold:
        main = surface.main_body >= 128
        unclassified = residual & ~main
        add(PaintGroup.MAIN_BODY_PAINT, surface.main_body)
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            np.where(unclassified, 255, 0).astype(np.uint8)
        )
        minimum_secondary_area = max(100, round(car_pixels * 0.01))
        for index in range(1, component_count):
            component = labels == index
            component_mask = np.where(component, 255, 0).astype(np.uint8)
            material, confidence, reasons = classify_material(
                image, component_mask, "body_region"
            )
            group = (
                PaintGroup.SECONDARY_BODY_PAINT
                if stats[index, cv2.CC_STAT_AREA] >= minimum_secondary_area
                and material == MaterialType.PAINTED_SURFACE
                else PaintGroup.UNKNOWN
            )
            add(group, component_mask)
            regions.append(
                RegionClassification(
                    region_id=f"body_region_{index}",
                    part_type="body_region",
                    paint_group=group,
                    material_type=material,
                    confidence=confidence,
                    paintability=(
                        Paintability.SEPARATELY_EDITABLE
                        if group == PaintGroup.SECONDARY_BODY_PAINT
                        else Paintability.UNCERTAIN
                    ),
                    reason_codes=reasons + ["connected_body_region_context"],
                )
            )
        regions.append(
            RegionClassification(
                region_id="main_body_pixels",
                part_type="body_panel",
                paint_group=PaintGroup.MAIN_BODY_PAINT,
                material_type=MaterialType.PAINTED_SURFACE,
                body_colour_similarity=profile.confidence,
                confidence=profile.confidence,
                paintability=Paintability.EDITABLE,
                reason_codes=["dominant_anchor_chroma_match", "panel_interior_context"],
            )
        )
    else:
        add(PaintGroup.UNKNOWN, np.where(residual, 255, 0).astype(np.uint8))

    masks = build_default_masks(full_car, groups)
    confidences = defaultdict(list)
    for region in regions:
        confidences[region.paint_group].append(region.confidence)
    summaries = [
        PaintGroupSummary(
            paint_group=group,
            pixel_count=int(np.count_nonzero(mask >= 128)),
            ratio_of_car=round(float(np.count_nonzero(mask >= 128) / car_pixels), 4),
            confidence=round(
                float(np.mean(confidences[group])) if confidences[group] else profile.confidence,
                4,
            ),
        )
        for group, mask in groups.items()
    ]
    warnings = [*profile.warnings]
    if not np.any(masks.editable):
        warnings.append("main_body_mask_empty")
    return PaintAnalysisResult(
        PaintGroupReport(
            body_paint_profile=profile,
            groups=summaries,
            region_classifications=regions,
            surface_completion=surface.report,
            warnings=warnings,
        ),
        groups,
        masks,
        anchors,
        surface,
    )
