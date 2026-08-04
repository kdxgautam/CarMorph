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
    paint_like_residual: np.ndarray


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
    rgb = np.asarray(image.convert("RGB"))
    lab = rgb_to_lab(rgb)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    grey = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    profile_lab = np.asarray(profile.median_lab)
    groups: dict[PaintGroup, np.ndarray] = {}
    regions = []
    claimed = np.zeros(shape, bool)
    paint_like_residual = np.zeros(shape, np.uint8)

    def add(group: PaintGroup, mask: np.ndarray) -> None:
        binary = (mask >= 128) & (full_car >= 128) & ~claimed
        if np.any(binary):
            groups[group] = np.maximum(
                groups.get(group, np.zeros(shape, np.uint8)),
                np.where(binary, 255, 0).astype(np.uint8),
            )
            claimed[binary] = True

    body_is_chromatic = len(profile_lab) and np.linalg.norm(profile_lab[1:]) > 20

    def painted_part_edge(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        binary = mask >= 128
        interior = cv2.erode(
            np.where(binary, 255, 0).astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            ),
        ) >= 128
        chroma_limit = 60 if body_is_chromatic else settings.body_growth_chroma_threshold
        lightness_limit = 30 if body_is_chromatic else min(
            settings.body_paint_lightness_threshold, 20
        )
        compatible = (
            (chroma_distance(lab, profile_lab) <= chroma_limit)
            & (np.abs(lab[:, :, 0] - profile_lab[0]) <= lightness_limit)
        )
        if body_is_chromatic:
            compatible &= hsv[:, :, 1] >= 75
        outside = (~binary) & (full_car >= 128) & compatible
        count = cv2.boxFilter(
            outside.astype(np.float32), -1, (3, 3), normalize=False
        )
        sums = np.stack(
            [
                cv2.boxFilter(
                    lab[:, :, channel] * outside,
                    -1,
                    (3, 3),
                    normalize=False,
                )
                for channel in range(3)
            ],
            axis=2,
        )
        local_mean = sums / np.maximum(count[:, :, None], 1)
        return (
            binary
            & ~interior
            & compatible
            & (count > 0)
            & (np.linalg.norm(lab - local_mean, axis=2) <= 20)
        )

    for part_type, group in PROTECTED_SEMANTICS.items():
        mask = part_masks.get(part_type)
        if mask is None:
            continue
        if part_type == "windows" and len(profile_lab):
            binary = mask >= 128
            interior = cv2.erode(
                np.where(binary, 255, 0).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            ) >= 128
            painted_edge = (
                binary
                & ~interior
                & (
                    chroma_distance(lab, profile_lab)
                    <= settings.body_growth_chroma_threshold
                )
                & (
                    np.abs(lab[:, :, 0] - profile_lab[0])
                    <= min(settings.body_paint_lightness_threshold, 20)
                )
            )
            if body_is_chromatic:
                painted_edge |= painted_part_edge(mask)
            mask = np.where(binary & ~painted_edge, 255, 0).astype(np.uint8)
        elif part_type == "lights" and body_is_chromatic:
            binary = mask >= 128
            interior = cv2.erode(
                np.where(binary, 255, 0).astype(np.uint8),
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            ) >= 128
            candidate = (
                binary
                & ~interior
                & (chroma_distance(lab, profile_lab) <= 60)
                & (np.abs(lab[:, :, 0] - profile_lab[0]) <= 30)
                & (hsv[:, :, 1] >= 75)
            )
            painted_edge = np.zeros(shape, bool)
            count, labels = cv2.connectedComponents(
                np.where(binary, 255, 0).astype(np.uint8)
            )
            for index in range(1, count):
                component = labels == index
                if np.median(hsv[:, :, 1][component]) < 75:
                    painted_edge |= candidate & component
            mask = np.where(binary & ~painted_edge, 255, 0).astype(np.uint8)
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
        painted_edge = (
            (mask >= 128)
            & (
                chroma_distance(lab, profile_lab)
                <= settings.body_growth_chroma_threshold
            )
            & (
                np.abs(lab[:, :, 0] - profile_lab[0])
                <= settings.body_paint_lightness_threshold
            )
            & (hsv[:, :, 1] >= 50)
            if body_is_chromatic
            else np.zeros(shape, bool)
        )
        trim_pixels = (mask >= 128) & (
            ((grey < 100) & (hsv[:, :, 1] < 100))
            | (hsv[:, :, 1] < 35)
        ) & ~painted_edge
        mask = np.where(trim_pixels, 255, 0).astype(np.uint8)
        if not np.any(mask):
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

    for part_type, matching_group, other_group in (
        (
            "handles",
            PaintGroup.BODY_COLOURED_HANDLE,
            PaintGroup.CONTRASTING_HANDLE,
        ),
        (
            "mirrors",
            PaintGroup.BODY_COLOURED_MIRROR_CAP,
            PaintGroup.CONTRASTING_MIRROR_CAP,
        ),
    ):
        mask = part_masks.get(part_type)
        if mask is None:
            continue
        material, material_confidence, reasons = classify_material(
            image, mask, part_type
        )
        binary = mask >= 128
        painted = binary & ~((grey < 30) & (hsv[:, :, 1] < 75))
        compatible = (
            painted
            & (
                chroma_distance(lab, profile_lab)
                <= settings.body_paint_chroma_threshold
            )
            if len(profile_lab)
            else np.zeros(shape, bool)
        )
        minimum = max(3, round(np.count_nonzero(binary) * 0.08))
        has_painted_pixels = np.count_nonzero(compatible) >= minimum
        protected_group = (
            PaintGroup.CHROME_TRIM
            if material == MaterialType.CHROME
            else PaintGroup.GLOSSY_BLACK_TRIM
            if material == MaterialType.GLOSSY_PLASTIC
            else PaintGroup.BLACK_PLASTIC_TRIM
            if material == MaterialType.MATTE_PLASTIC
            else None
        )
        if protected_group == PaintGroup.CHROME_TRIM or (
            protected_group is not None and not has_painted_pixels
        ):
            add(protected_group, mask)
            regions.append(
                RegionClassification(
                    region_id=part_type,
                    part_type=part_type,
                    paint_group=protected_group,
                    material_type=material,
                    confidence=material_confidence,
                    paintability=Paintability.PROTECTED,
                    reason_codes=reasons + ["material_overrides_colour"],
                )
            )
            continue
        if has_painted_pixels:
            compatible_mask = np.where(compatible, 255, 0).astype(np.uint8)
            similarity, _, lightness = _relation(
                lab,
                compatible_mask,
                profile_lab,
                settings.body_paint_chroma_threshold,
            )
            add(
                matching_group,
                np.where(painted, 255, 0).astype(np.uint8),
            )
            regions.append(
                RegionClassification(
                    region_id=f"{part_type}_painted",
                    part_type=part_type,
                    paint_group=matching_group,
                    material_type=MaterialType.PAINTED_SURFACE,
                    body_colour_similarity=round(similarity, 4),
                    lightness_difference=round(lightness, 3),
                    confidence=round(min(0.9, similarity), 4),
                    paintability=Paintability.EDITABLE,
                    reason_codes=["body_compatible_part_pixels"],
                )
            )
            remaining = binary & ~painted
        else:
            remaining = binary
        if not np.any(remaining):
            continue
        remaining_mask = np.where(remaining, 255, 0).astype(np.uint8)
        material, confidence, reasons = classify_material(
            image, remaining_mask, part_type
        )
        group = (
            PaintGroup.CHROME_TRIM
            if material == MaterialType.CHROME
            else PaintGroup.GLOSSY_BLACK_TRIM
            if material == MaterialType.GLOSSY_PLASTIC
            else PaintGroup.BLACK_PLASTIC_TRIM
            if material == MaterialType.MATTE_PLASTIC
            else other_group
        )
        add(group, remaining_mask)
        regions.append(
            RegionClassification(
                region_id=f"{part_type}_contrasting",
                part_type=part_type,
                paint_group=group,
                material_type=material,
                confidence=confidence,
                paintability=Paintability.PROTECTED,
                reason_codes=reasons + ["contrasting_part_pixels"],
            )
        )

    configurable = {
        "roof": (PaintGroup.MAIN_BODY_PAINT, PaintGroup.CONTRAST_ROOF_PAINT),
        "spoiler": (PaintGroup.PAINTED_SPOILER, PaintGroup.SECONDARY_BODY_PAINT),
    }
    for part_type, (matching_group, other_group) in configurable.items():
        mask = part_masks.get(part_type)
        if mask is None:
            continue
        mask = np.where((mask >= 128) & ~claimed, 255, 0).astype(np.uint8)
        if not np.any(mask):
            continue
        compatible = (
            (mask >= 128)
            & (
                chroma_distance(lab, profile_lab)
                <= settings.body_growth_chroma_threshold
            )
            & (
                np.abs(lab[:, :, 0] - profile_lab[0])
                <= settings.body_paint_lightness_threshold
            )
        )
        if body_is_chromatic:
            compatible &= hsv[:, :, 1] >= 50
        if np.any(compatible):
            matching_mask = np.where(compatible, 255, 0).astype(np.uint8)
            similarity, _, lightness = _relation(
                lab,
                matching_mask,
                profile_lab,
                settings.body_paint_chroma_threshold,
            )
            add(matching_group, matching_mask)
            regions.append(
                RegionClassification(
                    region_id=f"{part_type}_body_compatible",
                    part_type=part_type,
                    paint_group=matching_group,
                    material_type=MaterialType.PAINTED_SURFACE,
                    body_colour_similarity=round(similarity, 4),
                    lightness_difference=round(lightness, 3),
                    confidence=round(min(0.9, similarity), 4),
                    paintability=Paintability.EDITABLE,
                    reason_codes=["body_compatible_part_pixels"],
                )
            )
            mask[compatible] = 0
        if not np.any(mask):
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

    dominant_hue = profile.dominant_hsv[0] if profile.dominant_hsv else 0
    if not (dominant_hue <= 35 or dominant_hue >= 325):
        contrast_lens = (
            (full_car >= 128)
            & ~claimed
            & (hsv[:, :, 1] >= 140)
            & ((hsv[:, :, 0] <= 25) | (hsv[:, :, 0] >= 170))
        )
        lens_regions = np.zeros(shape, bool)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(
            np.where(contrast_lens, 255, 0).astype(np.uint8)
        )
        for index in range(1, count):
            area = stats[index, cv2.CC_STAT_AREA]
            if 20 <= area <= car_pixels * 0.02:
                lens_regions |= labels == index
        add(
            PaintGroup.LIGHT_LENS,
            np.where(lens_regions, 255, 0).astype(np.uint8),
        )

    bumper = part_masks.get("bumper")
    if bumper is not None:
        bumper_candidate = bumper.copy()
        if body_is_chromatic:
            bumper_candidate[
                (grey < 80) & (hsv[:, :, 1] < 85)
                & (
                    chroma_distance(lab, profile_lab)
                    > settings.body_growth_chroma_threshold
                )
            ] = 0
        bumper_surface = complete_body_surface(
            lab,
            bumper_candidate,
            np.zeros_like(bumper),
            profile,
            settings,
        )
        painted_bumper = bumper_surface.main_body >= 128
        if body_is_chromatic:
            near_painted_bumper = cv2.dilate(
                bumper_surface.main_body,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
            ) >= 128
            painted_bumper |= (
                (bumper >= 128)
                & near_painted_bumper
                & (
                    chroma_distance(lab, profile_lab)
                    <= settings.body_growth_chroma_threshold
                )
                & (
                    np.abs(lab[:, :, 0] - profile_lab[0])
                    <= settings.body_paint_lightness_threshold
                )
                & (hsv[:, :, 1] >= 50)
            )
        add(
            PaintGroup.PAINTED_BUMPER_SECTION,
            np.where(painted_bumper, 255, 0).astype(np.uint8),
        )
        bumper_plastic = (
            (bumper >= 128)
            & (grey < 80)
            & (hsv[:, :, 1] < 85)
            & (
                chroma_distance(lab, profile_lab)
                > settings.body_growth_chroma_threshold
            )
        )
        add(
            PaintGroup.BLACK_PLASTIC_TRIM,
            np.where(bumper_plastic, 255, 0).astype(np.uint8),
        )

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
        adjacency_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fragment_chroma_threshold = settings.body_growth_chroma_threshold
        strict_compatible = unclassified & (
            chroma_distance(lab, profile_lab)
            <= settings.body_paint_chroma_threshold
        )
        compatible = unclassified & (
            chroma_distance(lab, profile_lab)
            <= fragment_chroma_threshold
        )
        compatible_count, compatible_labels = cv2.connectedComponents(
            np.where(compatible, 255, 0).astype(np.uint8)
        )
        for index in range(1, compatible_count):
            component = compatible_labels == index
            near_main = cv2.dilate(
                np.where(main, 255, 0).astype(np.uint8), adjacency_kernel
            ) > 0
            adjacency = float(
                np.count_nonzero(component & near_main)
                / max(1, np.count_nonzero(component))
            )
            if adjacency < min(settings.body_region_min_boundary_ratio, 0.04):
                paint_like_residual[component & strict_compatible] = 255
                continue
            component_mask = np.where(component, 255, 0).astype(np.uint8)
            similarity, _, lightness = _relation(
                lab, component_mask, profile_lab, settings.body_paint_chroma_threshold
            )
            add(PaintGroup.MAIN_BODY_PAINT, component_mask)
            main |= component
            regions.append(
                RegionClassification(
                    region_id=f"body_compatible_region_{index}",
                    part_type="body_region",
                    paint_group=PaintGroup.MAIN_BODY_PAINT,
                    material_type=MaterialType.PAINTED_SURFACE,
                    body_colour_similarity=round(similarity, 4),
                    lightness_difference=round(lightness, 3),
                    confidence=round(min(similarity, adjacency), 4),
                    paintability=Paintability.EDITABLE,
                    reason_codes=[
                        "pixelwise_main_body_chroma_match",
                        "adjacent_main_body_region",
                    ],
                )
            )

        remaining = unclassified & ~claimed
        component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
            np.where(remaining, 255, 0).astype(np.uint8)
        )
        minimum_secondary_area = max(100, round(car_pixels * 0.01))
        for index in range(1, component_count):
            component = labels == index
            component_mask = np.where(component, 255, 0).astype(np.uint8)
            material, confidence, reasons = classify_material(
                image, component_mask, "body_region"
            )
            similarity, chroma, lightness = _relation(
                lab, component_mask, profile_lab, settings.body_paint_chroma_threshold
            )
            paint_like = material == MaterialType.PAINTED_SURFACE
            group = (
                PaintGroup.SECONDARY_BODY_PAINT
                if paint_like
                and chroma > settings.body_paint_chroma_threshold
                and stats[index, cv2.CC_STAT_AREA] >= minimum_secondary_area
                and not np.any(component & strict_compatible)
                else PaintGroup.UNKNOWN
            )
            if np.any(component & strict_compatible):
                paint_like_residual[component & strict_compatible] = 255
            add(group, component_mask)
            if group == PaintGroup.SECONDARY_BODY_PAINT:
                reasons += ["contrasting_body_chroma"]
            regions.append(
                RegionClassification(
                    region_id=f"body_region_{index}",
                    part_type="body_region",
                    paint_group=group,
                    material_type=material,
                    body_colour_similarity=round(similarity, 4),
                    lightness_difference=round(lightness, 3),
                    confidence=round(
                        min(
                            confidence,
                            1 - similarity
                            if group == PaintGroup.SECONDARY_BODY_PAINT
                            else similarity,
                        ),
                        4,
                    ),
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
        paint_like_residual,
    )
