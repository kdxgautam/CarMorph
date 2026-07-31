from dataclasses import dataclass

import cv2
import numpy as np

from app.paint_analysis.colour_profile import chroma_distance
from app.paint_analysis.schemas import (
    BodyPaintProfile,
    FragmentationMetrics,
    PaintGroup,
    SurfaceCompletionReport,
    SurfaceRegionDecision,
)


@dataclass(frozen=True)
class SurfaceCompletionResult:
    safe_candidate: np.ndarray
    hard_protected: np.ndarray
    growth_candidate: np.ndarray
    seeds: np.ndarray
    main_body: np.ndarray
    report: SurfaceCompletionReport


def _binary(mask: np.ndarray) -> np.ndarray:
    return mask >= 128


def _mask(value: np.ndarray) -> np.ndarray:
    return np.where(value, 255, 0).astype(np.uint8)


def _component_metrics(mask: np.ndarray, minimum_area: int) -> tuple[int, int]:
    count, _, stats, _ = cv2.connectedComponentsWithStats(_mask(mask))
    areas = stats[1:, cv2.CC_STAT_AREA] if count > 1 else np.array([])
    return len(areas), int(np.count_nonzero(areas < minimum_area))


def _small_gap_pixels(
    accepted: np.ndarray, safe: np.ndarray, maximum_area: int
) -> int:
    count, _, stats, _ = cv2.connectedComponentsWithStats(_mask(safe & ~accepted))
    return int(
        sum(
            int(stats[index, cv2.CC_STAT_AREA])
            for index in range(1, count)
            if stats[index, cv2.CC_STAT_AREA] <= maximum_area
        )
    )


def _neighbour_mean(
    lab: np.ndarray, accepted: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    count = cv2.boxFilter(
        accepted.astype(np.float32),
        -1,
        (3, 3),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    sums = np.stack(
        [
            cv2.boxFilter(
                lab[:, :, channel] * accepted,
                -1,
                (3, 3),
                normalize=False,
                borderType=cv2.BORDER_CONSTANT,
            )
            for channel in range(3)
        ],
        axis=2,
    )
    return sums / np.maximum(count[:, :, None], 1), count


def complete_body_surface(
    lab: np.ndarray,
    safe_candidate: np.ndarray,
    hard_protected: np.ndarray,
    profile: BodyPaintProfile,
    settings: object,
) -> SurfaceCompletionResult:
    safe = _binary(safe_candidate) & ~_binary(hard_protected)
    protected = _binary(hard_protected)
    if not profile.median_lab or not np.any(safe):
        empty = np.zeros(safe.shape, np.uint8)
        return SurfaceCompletionResult(
            _mask(safe),
            _mask(protected),
            empty,
            empty,
            empty,
            SurfaceCompletionReport(),
        )

    profile_lab = np.asarray(profile.median_lab, np.float32)
    global_chroma = chroma_distance(lab, profile_lab)
    lightness = lab[:, :, 0]
    shadow_min = profile.shadow_lab_range.get("min", float(np.min(lightness[safe])))
    highlight_max = profile.highlight_lab_range.get(
        "max", float(np.max(lightness[safe]))
    )
    protected_margin = cv2.dilate(
        _mask(protected),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ) > 0
    strict = (
        safe
        & ~protected_margin
        & (global_chroma <= settings.body_paint_strict_chroma_threshold)
        & (lightness >= shadow_min - 5)
        & (lightness <= highlight_max + 5)
    )
    agreement = cv2.boxFilter(
        strict.astype(np.float32),
        -1,
        (3, 3),
        normalize=False,
        borderType=cv2.BORDER_CONSTANT,
    )
    seeds = strict & (agreement >= settings.body_seed_min_neighbours)

    growth_candidate = safe & (
        global_chroma <= settings.body_growth_chroma_threshold
    )
    blurred_l = cv2.GaussianBlur(lightness, (3, 3), 0)
    gradient_x = cv2.Sobel(blurred_l, cv2.CV_32F, 1, 0, ksize=3)
    gradient_y = cv2.Sobel(blurred_l, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(gradient_x, gradient_y)

    accepted = seeds.copy()
    iterations = 0
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    for iterations in range(1, settings.body_growth_max_iterations + 1):
        frontier = (
            (cv2.dilate(_mask(accepted), kernel) > 0)
            & growth_candidate
            & ~accepted
        )
        if not np.any(frontier):
            iterations -= 1
            break
        local_mean, neighbours = _neighbour_mean(lab, accepted)
        local_distance = np.linalg.norm(lab - local_mean, axis=2)
        additions = (
            frontier
            & (neighbours >= settings.body_growth_min_neighbours)
            & (local_distance <= settings.body_growth_local_lab_threshold)
            & (
                (gradient <= settings.body_growth_max_gradient)
                | (neighbours >= 5)
            )
        )
        if not np.any(additions):
            break
        accepted |= additions

    completion_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (settings.body_completion_kernel_size,) * 2,
    )
    closed = cv2.morphologyEx(_mask(accepted), cv2.MORPH_CLOSE, completion_kernel) > 0
    accepted |= (
        closed
        & safe
        & (global_chroma <= settings.body_growth_chroma_threshold * 1.25)
    )

    regions = []
    remaining = safe & ~accepted & (
        global_chroma <= settings.body_growth_chroma_threshold * 1.25
    )
    count, labels, stats, _ = cv2.connectedComponentsWithStats(_mask(remaining))
    seed_dilation = cv2.dilate(_mask(seeds), completion_kernel) > 0
    for index in range(1, count):
        component = labels == index
        area = int(stats[index, cv2.CC_STAT_AREA])
        boundary = (cv2.dilate(_mask(component), kernel) > 0) & ~component
        boundary_pixels = max(1, int(np.count_nonzero(boundary)))
        accepted_boundary = float(np.count_nonzero(boundary & accepted) / boundary_pixels)
        seed_coverage = float(np.count_nonzero(component & seed_dilation) / area)
        values = global_chroma[component]
        median_distance = float(np.median(values))
        global_similarity = max(
            0.0,
            1
            - median_distance
            / max(settings.body_growth_chroma_threshold * 1.25, 1),
        )
        consistency = max(
            0.0,
            1
            - float(np.median(np.abs(values - median_distance)))
            / max(settings.body_growth_chroma_threshold, 1),
        )
        mean_boundary_gradient = (
            float(np.mean(gradient[boundary])) if np.any(boundary) else 0.0
        )
        enclosed = (
            area <= settings.body_completion_max_hole_area
            and accepted_boundary >= 0.45
        )
        coherent = (
            accepted_boundary >= settings.body_region_min_boundary_ratio
            and global_similarity >= 0.25
            and consistency >= 0.5
        )
        accepted_region = enclosed or coherent
        if accepted_region:
            accepted |= component
        confidence = min(
            1.0,
            0.4 * global_similarity
            + 0.4 * accepted_boundary
            + 0.2 * consistency,
        )
        reasons = ["no_protected_overlap"]
        if accepted_boundary:
            reasons.append("connected_to_main_body")
        if enclosed:
            reasons.append("enclosed_surface_gap")
        if coherent:
            reasons.append("coherent_region_vote")
        regions.append(
            SurfaceRegionDecision(
                region_id=f"surface_region_{index}",
                pixel_count=area,
                seed_coverage=round(seed_coverage, 4),
                accepted_boundary_ratio=round(accepted_boundary, 4),
                protected_overlap_ratio=0,
                global_chroma_similarity=round(global_similarity, 4),
                local_colour_consistency=round(consistency, 4),
                mean_gradient_boundary=round(mean_boundary_gradient, 3),
                semantic_body_likelihood=1,
                decision=(
                    PaintGroup.MAIN_BODY_PAINT
                    if accepted_region
                    else PaintGroup.UNKNOWN
                ),
                confidence=round(confidence, 4),
                reason_codes=reasons,
            )
        )

    gap_count, gap_labels, gap_stats, _ = cv2.connectedComponentsWithStats(
        _mask(safe & ~accepted)
    )
    for index in range(1, gap_count):
        area = int(gap_stats[index, cv2.CC_STAT_AREA])
        if area > settings.body_completion_max_hole_area:
            continue
        gap = gap_labels == index
        boundary = (cv2.dilate(_mask(gap), kernel) > 0) & ~gap
        if np.mean(accepted[boundary]) >= 0.6:
            accepted |= gap

    accepted &= safe & ~protected
    before_components, before_small = _component_metrics(
        seeds, settings.body_fragment_min_area
    )
    after_components, after_small = _component_metrics(
        accepted, settings.body_fragment_min_area
    )
    metrics = FragmentationMetrics(
        seed_pixel_count=int(np.count_nonzero(seeds)),
        final_pixel_count=int(np.count_nonzero(accepted)),
        recovered_pixel_count=int(np.count_nonzero(accepted & ~seeds)),
        components_before=before_components,
        components_after=after_components,
        small_components_before=before_small,
        small_components_after=after_small,
        internal_gap_pixels_before=_small_gap_pixels(
            seeds, safe, settings.body_completion_max_hole_area
        ),
        internal_gap_pixels_after=_small_gap_pixels(
            accepted, safe, settings.body_completion_max_hole_area
        ),
        growth_iterations=iterations,
    )
    return SurfaceCompletionResult(
        _mask(safe),
        _mask(protected),
        _mask(growth_candidate),
        _mask(seeds),
        _mask(accepted),
        SurfaceCompletionReport(regions=regions, fragmentation=metrics),
    )
