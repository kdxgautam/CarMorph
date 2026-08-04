"""Orchestrate detection, segmentation, paint analysis, and asset persistence."""

import hashlib
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from threading import Lock

import cv2
import numpy as np

from app.config import (
    EXPECTED_PROTECTIVE_PART_GROUPS_BY_VIEW,
    OUTPUT_PART_GROUPS,
    REQUIRED_PART_GROUPS_BY_VIEW,
    Settings,
)
from app.detection import PartDetection, detect_car_and_parts
from app.errors import PipelineError
from app.image_ops import (
    clean_mask,
    combine_masks,
    dark_trim_mask,
    encode_jpeg,
    load_image,
    polygons_to_mask,
    save_base_assets,
    save_mask,
)
from app.paint_analysis.diagnostics import (
    anchor_overlay,
    paint_group_overlay,
    surface_completion_overlay,
)
from app.paint_analysis.paint_group_classifier import analyse_paint_groups
from app.paint_analysis.schemas import PaintGroup
from app.quality.checks import check_paint_analysis
from app.roboflow import segment_boxes, segment_concepts
from app.schemas import (
    AssetBundle,
    AvailableModifications,
    BoundingBox,
    PaintabilityReport,
    ViewName,
    ViewSelection,
)

# Asset identity includes the requested view and processing version, so automatic
# and manual rollback modes never reuse incompatible masks.
# ponytail: process-local lock; use a shared job/lock store when running workers.
_PROCESS_LOCK = Lock()
PIPELINE_VERSION = b"33"
PAINT_ANALYSIS_VERSION = "paint-groups-v12"

PAINT_GROUP_FILENAMES = {
    PaintGroup.MAIN_BODY_PAINT: "main-body-paint-mask.png",
    PaintGroup.SECONDARY_BODY_PAINT: "secondary-body-paint-mask.png",
    PaintGroup.CONTRAST_ROOF_PAINT: "contrast-roof-mask.png",
    PaintGroup.BODY_COLOURED_HANDLE: "body-coloured-handles-mask.png",
    PaintGroup.CONTRASTING_HANDLE: "contrasting-handles-mask.png",
    PaintGroup.BODY_COLOURED_MIRROR_CAP: "body-coloured-mirror-caps-mask.png",
    PaintGroup.CONTRASTING_MIRROR_CAP: "contrasting-mirror-caps-mask.png",
    PaintGroup.PAINTED_BUMPER_SECTION: "painted-bumper-sections-mask.png",
    PaintGroup.BLACK_PLASTIC_TRIM: "black-plastic-trim-mask.png",
    PaintGroup.GLOSSY_BLACK_TRIM: "glossy-black-trim-mask.png",
    PaintGroup.CHROME_TRIM: "chrome-trim-mask.png",
    PaintGroup.SILVER_GARNISH: "silver-garnish-mask.png",
    PaintGroup.UNKNOWN: "paint-group-uncertain-mask.png",
}


def _asset_id(source: bytes, view: ViewSelection) -> str:
    """Hash source bytes, requested view, and pipeline version into cache ID."""

    return hashlib.sha256(
        PIPELINE_VERSION + b"\0" + view.encode() + b"\0" + source
    ).hexdigest()


def _clip_fallback_mask(
    mask: np.ndarray, part: PartDetection | None
) -> np.ndarray:
    """Confine a generated mask to its part clip and recover a missing seed box."""

    if part is None or part.clip_box is None:
        return mask
    x1, y1, x2, y2 = map(round, part.clip_box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(mask.shape[1], x2), min(mask.shape[0], y2)
    clipped = np.zeros_like(mask)
    clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    prompt_x1, prompt_y1, prompt_x2, prompt_y2 = map(round, part.box)
    prompt = clipped[prompt_y1:prompt_y2, prompt_x1:prompt_x2]
    # ponytail: geometric seed fallback; replace if transparent glass gets a detector.
    if prompt.size and np.mean(prompt >= 128) < 0.5:
        clipped[prompt_y1:prompt_y2, prompt_x1:prompt_x2] = 255
    return clipped


def _refine_side_windows(
    windows: np.ndarray,
    mirrors: np.ndarray | None,
) -> np.ndarray:
    """Binarize combined glass and remove mirror overlap from window masks."""

    windows = np.where(windows >= 128, 255, 0).astype(np.uint8)
    if mirrors is not None:
        windows[mirrors >= 128] = 0
    return windows


def _hybrid_semantic_groups(view: ViewName) -> dict[str, str]:
    """Return SAM3 concepts that supplement SAM2 for the resolved view."""

    groups = {
        "handles": "car door handle",
        "lights": "car light",
        "mirrors": "car side mirror",
        "pillars": "car window pillar",
        "trim": "black car trim",
    }
    if view in {"front", "rear"}:
        groups["plate"] = "license plate"
    return groups


def _read_result(metadata: Path) -> AssetBundle:
    """Validate cached metadata and every file required for rendering."""

    try:
        result = AssetBundle.model_validate_json(metadata.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PipelineError(
            "invalid_stored_assets", "Stored asset metadata is invalid", 500
        ) from exc
    expected = {
        result.source_image,
        result.original_image,
        result.luminance_map,
        *result.masks.values(),
    }
    if any(not (metadata.parent / path).is_file() for path in expected):
        raise PipelineError("missing_masks", "A stored image or mask is missing", 500)
    return result


def process_view(
    source: bytes, settings: Settings, view: ViewSelection
) -> AssetBundle:
    """Prepare and persist one immutable, reusable car asset.

    Processing detects the car and parts, refines masks through the configured
    segmenters, classifies paint/material groups, checks mask invariants, and
    atomically promotes a temporary directory into the content-addressed cache.
    Existing complete assets return without rerunning models.
    """

    asset_id = _asset_id(source, view)
    final = settings.storage_root / asset_id

    with _PROCESS_LOCK:
        # A complete metadata file is the commit marker for an immutable asset.
        if (final / "metadata.json").is_file():
            return _read_result(final / "metadata.json")

        try:
            image, suffix = load_image(source)
        except (OSError, ValueError) as exc:
            raise PipelineError("invalid_image", str(exc), 400) from exc

        settings.storage_root.mkdir(parents=True, exist_ok=True)
        work = Path(tempfile.mkdtemp(dir=settings.storage_root))
        try:
            image_jpeg = encode_jpeg(image)
            car, parts, resolved_view, view_confidence = detect_car_and_parts(
                image, settings, view
            )
            grouped_parts = defaultdict(list)
            for part in parts:
                grouped_parts[part.group].append(part)

            required_parts = REQUIRED_PART_GROUPS_BY_VIEW[resolved_view]
            missing = sorted(required_parts - grouped_parts.keys())
            if missing:
                raise PipelineError(
                    "missing_masks",
                    "Part detection did not find required regions: "
                    + ", ".join(missing),
                )

            prompts = [tuple(float(value) for value in car.box)]
            concepts = ["car"]
            prompt_parts: list[PartDetection | None] = [None]
            for part in parts:
                if part.polygon is None:
                    prompts.append(part.box)
                    concepts.append(
                        {
                            "wheels": "car wheel",
                            "windows": "car window",
                            "plate": "license plate",
                            "lights": "car light",
                            "grille": "car grille",
                            "trim": "car trim",
                            "bumper": "car bumper",
                            "mirrors": "car side mirror",
                            "handles": "car door handle",
                            "roof": "car roof",
                            "spoiler": "car spoiler",
                            "pillars": "car window pillar",
                        }.get(part.group, f"car {part.group}")
                    )
                    prompt_parts.append(part)

            # SAM2 supplies stable geometry from local detector boxes.
            polygons = segment_boxes(image_jpeg, prompts, settings, concepts)
            masks_by_group: dict[str, list[np.ndarray]] = defaultdict(list)
            for part, mask_polygons in zip(prompt_parts, polygons):
                group = part.group if part is not None else "full_car"
                mask = _clip_fallback_mask(
                    polygons_to_mask(mask_polygons, image.size),
                    part,
                )
                masks_by_group[group].append(
                    clean_mask(
                        mask,
                        image.size,
                        settings.mask_kernel_size,
                        settings.mask_feather_radius,
                    )
                )
            for part in parts:
                if part.polygon is not None:
                    masks_by_group[part.group].append(
                        clean_mask(
                            polygons_to_mask([part.polygon], image.size),
                            image.size,
                            settings.mask_kernel_size,
                            settings.mask_feather_radius,
                        )
                    )

            # SAM3 supplements optional semantic details without replacing SAM2's
            # full-car and window boundaries.
            if settings.roboflow_segmenter == "hybrid":
                semantic_groups = _hybrid_semantic_groups(resolved_view)
                semantic_masks = segment_concepts(
                    image_jpeg, list(semantic_groups.values()), settings
                )
                for group, mask_polygons in zip(
                    semantic_groups, semantic_masks
                ):
                    if mask_polygons:
                        masks_by_group[group].append(
                            clean_mask(
                                polygons_to_mask(mask_polygons, image.size),
                                image.size,
                                settings.mask_kernel_size,
                                settings.mask_feather_radius,
                            )
                        )

            full_car = combine_masks(masks_by_group.pop("full_car"))
            if settings.roboflow_segmenter != "hybrid":
                full_car = np.maximum(
                    full_car,
                    combine_masks(
                        [mask for masks in masks_by_group.values() for mask in masks]
                    ),
                )
            part_masks = {}
            for group, masks in masks_by_group.items():
                mask = np.minimum(combine_masks(masks), full_car)
                if np.any(mask >= 128):
                    part_masks[group] = mask
            missing_masks = sorted(required_parts - part_masks.keys())
            if missing_masks:
                raise PipelineError(
                    "missing_masks",
                    "Segmentation did not produce required masks: "
                    + ", ".join(missing_masks),
                    502,
                )

            if "windows" in part_masks:
                part_masks["windows"] = _refine_side_windows(
                    part_masks["windows"],
                    part_masks.get("mirrors"),
                )

            part_masks["dark_trim"] = dark_trim_mask(
                image,
                full_car,
                car.box,
                settings.mask_kernel_size,
                settings.mask_feather_radius,
            )
            absent_output_masks = sorted(OUTPUT_PART_GROUPS - part_masks.keys())
            missing_expected_masks = sorted(
                EXPECTED_PROTECTIVE_PART_GROUPS_BY_VIEW[resolved_view]
                - part_masks.keys()
            )
            for group in absent_output_masks:
                part_masks[group] = np.zeros_like(full_car)

            # Material-aware analysis turns raw parts into the disjoint masks used
            # by every renderer.
            analysis = analyse_paint_groups(
                image,
                full_car,
                part_masks,
                settings,
            )
            analysis_warnings = check_paint_analysis(
                analysis.group_masks,
                analysis.masks,
                profile_confidence=analysis.report.body_paint_profile.confidence,
                confidence_threshold=settings.paint_group_uncertain_threshold,
                seed_mask=analysis.surface.seeds,
                hard_protected_mask=analysis.surface.hard_protected,
                paint_like_residual_mask=analysis.paint_like_residual,
                minimum_residual_pixels=max(
                    settings.body_fragment_min_area,
                    round(np.count_nonzero(full_car >= 128) * 0.005),
                ),
            )
            if analysis_warnings:
                analysis.report.warnings.extend(analysis_warnings)
            paintability = analysis.masks

            masks_dir = work / "masks"
            masks_dir.mkdir()
            save_mask(full_car, masks_dir / "full-car.png")
            save_mask(paintability.editable, masks_dir / "editable-mask.png")
            save_mask(paintability.protected, masks_dir / "protected-mask.png")
            save_mask(paintability.uncertain, masks_dir / "uncertain-mask.png")
            save_mask(paintability.editable, masks_dir / "paintable-body.png")
            mask_paths = {
                "full_car": "masks/full-car.png",
                "paintable_body": "masks/paintable-body.png",
                "editable_mask": "masks/editable-mask.png",
                "protected_mask": "masks/protected-mask.png",
                "uncertain_mask": "masks/uncertain-mask.png",
            }
            for group, mask in sorted(part_masks.items()):
                filename = f"{group}.png"
                save_mask(mask, masks_dir / filename)
                mask_paths[group] = f"masks/{filename}"
            for group, mask in analysis.group_masks.items():
                filename = PAINT_GROUP_FILENAMES.get(
                    group, f"{group.value.replace('_', '-')}-mask.png"
                )
                save_mask(mask, masks_dir / filename)
                mask_paths[group.value] = f"masks/{filename}"
            for key, filename, mask in (
                (
                    "safe_body_candidate",
                    "safe-body-candidate-mask.png",
                    analysis.surface.safe_candidate,
                ),
                (
                    "hard_protected",
                    "hard-protected-mask.png",
                    analysis.surface.hard_protected,
                ),
                (
                    "growth_candidate",
                    "growth-candidate-mask.png",
                    analysis.surface.growth_candidate,
                ),
                (
                    "main_body_seed",
                    "main-body-seed-mask.png",
                    analysis.surface.seeds,
                ),
            ):
                save_mask(mask, masks_dir / filename)
                mask_paths[key] = f"masks/{filename}"
            if settings.paint_analysis_diagnostics:
                paint_group_overlay(image, analysis.group_masks).save(
                    masks_dir / "paint-groups-overlay.png"
                )
                anchor_overlay(image, analysis.anchors).save(
                    masks_dir / "body-paint-anchor-overlay.png"
                )
                mask_paths["paint_groups_overlay"] = "masks/paint-groups-overlay.png"
                mask_paths["body_paint_anchor_overlay"] = (
                    "masks/body-paint-anchor-overlay.png"
                )
                surface_completion_overlay(
                    image,
                    analysis.surface.safe_candidate,
                    analysis.surface.hard_protected,
                    analysis.surface.seeds,
                    analysis.surface.main_body,
                ).save(masks_dir / "surface-completion-overlay.png")
                mask_paths["surface_completion_overlay"] = (
                    "masks/surface-completion-overlay.png"
                )

            (work / f"source.{suffix}").write_bytes(source)
            save_base_assets(image, work)
            car_pixels = max(1, int(np.count_nonzero(full_car >= 128)))
            paintability_report = PaintabilityReport(
                editable_ratio=round(
                    float(np.count_nonzero(paintability.editable >= 128) / car_pixels), 4
                ),
                protected_ratio=round(
                    float(np.count_nonzero(paintability.protected >= 128) / car_pixels), 4
                ),
                uncertain_ratio=round(
                    float(np.count_nonzero(paintability.uncertain >= 128) / car_pixels), 4
                ),
                warnings=analysis.report.warnings,
                rules_version=PAINT_ANALYSIS_VERSION,
            )
            (work / "paintability-report.json").write_text(
                paintability_report.model_dump_json(indent=2),
                encoding="utf-8",
            )
            (work / "paint-groups.json").write_text(
                analysis.report.model_dump_json(indent=2), encoding="utf-8"
            )
            (work / "body-paint-profile.json").write_text(
                analysis.report.body_paint_profile.model_dump_json(indent=2),
                encoding="utf-8",
            )
            (work / "surface-completion.json").write_text(
                analysis.surface.report.model_dump_json(indent=2),
                encoding="utf-8",
            )
            result = AssetBundle(
                asset_id=asset_id,
                view=resolved_view,
                requested_view=view,
                view_confidence=view_confidence,
                width=image.width,
                height=image.height,
                car_bbox=BoundingBox(
                    x1=car.box[0],
                    y1=car.box[1],
                    x2=car.box[2],
                    y2=car.box[3],
                    confidence=car.confidence,
                ),
                source_image=f"source.{suffix}",
                original_image="original.webp",
                luminance_map="luminance-map.png",
                masks=mask_paths,
                models={
                    "yolo_world": settings.yolo_model_id,
                    "car_parts": settings.car_parts_model_id,
                    "segmenter": (
                        f"sam2/{settings.roboflow_sam2_version_id}"
                        f"+{settings.roboflow_sam3_model_id}"
                        if settings.roboflow_segmenter == "hybrid"
                        else settings.roboflow_sam3_model_id
                        if settings.roboflow_segmenter == "sam3"
                        else f"sam2/{settings.roboflow_sam2_version_id}"
                    ),
                },
                warnings=(
                    [
                        "Expected protective masks were not detected: "
                        + ", ".join(missing_expected_masks)
                    ]
                    if missing_expected_masks
                    else []
                ),
                paintability_report=paintability_report,
                body_paint_profile=analysis.report.body_paint_profile,
                paint_group_report=analysis.report,
                paint_analysis_version=PAINT_ANALYSIS_VERSION,
                available_modifications=AvailableModifications(roof_colour=True),
                pipeline_version=PIPELINE_VERSION.decode(),
            )
            (work / "metadata.json").write_text(
                result.model_dump_json(indent=2), encoding="utf-8"
            )
            work.rename(final)
            return result
        except Exception:
            shutil.rmtree(work, ignore_errors=True)
            raise
