"""Local vehicle/part detection and automatic cardinal-view inference."""

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image

from app.config import PART_GROUPS, YOLO_PART_PROMPTS_BY_VIEW, Settings
from app.errors import PipelineError
from app.schemas import ViewName, ViewSelection

_YOLO_LOCK = Lock()
MAX_PART_AREA_RATIO = 0.45
MIN_VIEW_SCORE = 0.8
MIN_VIEW_CONFIDENCE = 0.65
FACE_TO_SIDE_RATIO = 0.8

# Each family contributes at most its strongest label, preventing duplicate boxes
# for one physical part from inflating an orientation score.
VIEW_FAMILIES = {
    "front": (
        {"grille", "hood"},
        {"windshield"},
        {"front_bumper"},
        {"left_headlight", "right_headlight"},
    ),
    "rear": (
        {"trunk"},
        {"back_windshield"},
        {"back_bumper"},
        {"left_tail_light", "right_tail_light"},
    ),
    "left": (
        {"left_front_door", "left_back_door"},
        {"left_front_window", "left_back_window"},
        {"left_front_wheel", "left_back_wheel"},
        {"left_mirror", "left_fender", "left_quarter_panel", "left_rocker_panel"},
    ),
    "right": (
        {"right_front_door", "right_back_door"},
        {"right_front_window", "right_back_window"},
        {"right_front_wheel", "right_back_wheel"},
        {
            "right_mirror",
            "right_fender",
            "right_quarter_panel",
            "right_rocker_panel",
        },
    ),
}


@dataclass(frozen=True)
class CarDetection:
    """A candidate vehicle box in full-image coordinates."""

    box: tuple[int, int, int, int]
    confidence: float

    @property
    def area(self) -> int:
        """Return box area for primary-car ranking."""

        x1, y1, x2, y2 = self.box
        return (x2 - x1) * (y2 - y1)


@dataclass(frozen=True)
class PartDetection:
    """A normalized part prediction, optionally carrying polygon and clip data."""

    group: str
    box: tuple[float, float, float, float]
    confidence: float
    polygon: tuple[tuple[float, float], ...] | None = None
    clip_box: tuple[float, float, float, float] | None = None


def _normalise(value: object) -> str:
    """Convert provider labels to lowercase underscore-separated identifiers."""

    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _part_group(class_name: object, view: ViewName | None = None) -> str | None:
    """Map a provider-specific label to the pipeline's semantic part group."""

    name = _normalise(class_name)
    return next(
        (
            group
            for group, aliases in PART_GROUPS.items()
            if name in aliases
            or re.sub(r"^(?:left|right)_", "", name) in aliases
        ),
        None,
    )


def infer_view(evidence: list[tuple[str, float]]) -> tuple[ViewName, float]:
    """Resolve a cardinal view from raw part evidence or raise ambiguity.

    Scores use the maximum confidence within each physical-part family so
    duplicate detections cannot dominate. Front/rear evidence wins a
    three-quarter image when it remains close to the strongest side score.
    """

    # Preserve the strongest observation for each raw model label.
    confidences: dict[str, float] = {}
    for label, confidence in evidence:
        label = _normalise(label)
        confidences[label] = max(confidences.get(label, 0), confidence)
    scores = {
        view: sum(
            max((confidences.get(label, 0) for label in family), default=0)
            for family in families
        )
        for view, families in VIEW_FAMILIES.items()
    }

    def winner(first: ViewName, second: ViewName) -> tuple[ViewName, float, float]:
        """Return the stronger opposite direction and its normalized margin."""

        selected, other = (
            (first, second) if scores[first] >= scores[second] else (second, first)
        )
        total = scores[selected] + scores[other]
        confidence = scores[selected] / total if total else 0
        return selected, scores[selected], confidence

    face, face_score, face_confidence = winner("front", "rear")
    side, side_score, side_confidence = winner("left", "right")
    face_valid = (
        face_score >= MIN_VIEW_SCORE
        and face_confidence >= MIN_VIEW_CONFIDENCE
    )
    side_valid = (
        side_score >= MIN_VIEW_SCORE
        and side_confidence >= MIN_VIEW_CONFIDENCE
    )
    # Prefer the visible face in three-quarter photographs when it is nearly as
    # strong as the side evidence; downstream schemas remain cardinal.
    if face_valid and (
        not side_valid or face_score >= side_score * FACE_TO_SIDE_RATIO
    ):
        return face, round(face_confidence, 4)
    if side_valid:
        return side, round(side_confidence, 4)
    raise PipelineError(
        "ambiguous_view",
        "Could not determine the car view confidently; choose front, rear, left, or right",
    )


def _model_classes(car_class: str) -> tuple[str, ...]:
    """Build one stable, deduplicated YOLO-World vocabulary for cached reuse."""

    return tuple(
        dict.fromkeys(
            [
                car_class,
                *(
                    prompt
                    for prompts in YOLO_PART_PROMPTS_BY_VIEW.values()
                    for prompt in prompts
                ),
            ]
        )
    )


@lru_cache(maxsize=1)
def _load_model(model_id: str, classes: tuple[str, ...]) -> Any:
    """Load and configure the shared YOLO-World model once per process."""

    from ultralytics import YOLOWorld

    model = YOLOWorld(model_id)
    model.set_classes(list(classes))
    return model


@lru_cache(maxsize=1)
def _load_parts_model(model_id: str) -> Any:
    """Load pinned local car-parts weights after checking that they exist."""

    if not Path(model_id).is_file():
        raise PipelineError(
            "configuration_error",
            f"Car-parts model weights are missing: {model_id}",
            503,
        )
    from ultralytics import YOLO

    return YOLO(model_id)


def select_primary_car(
    cars: list[CarDetection], competing_ratio: float
) -> CarDetection:
    """Select the largest car unless another candidate is similarly prominent."""

    if not cars:
        raise PipelineError("no_car_detected", "No car was detected in the image")

    cars.sort(key=lambda item: item.area, reverse=True)
    if len(cars) > 1 and cars[1].area >= cars[0].area * competing_ratio:
        raise PipelineError(
            "multiple_competing_cars",
            "Multiple similarly sized cars were detected; upload one primary car",
        )
    return cars[0]


def _iou(first: PartDetection, second: PartDetection) -> float:
    """Return intersection-over-union for duplicate part suppression."""

    ax1, ay1, ax2, ay2 = first.box
    bx1, by1, bx2, by2 = second.box
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = (ax2 - ax1) * (ay2 - ay1)
    second_area = (bx2 - bx1) * (by2 - by1)
    return intersection / (first_area + second_area - intersection)


def _deduplicate(parts: list[PartDetection]) -> list[PartDetection]:
    """Keep the strongest overlapping prediction for each semantic group."""

    kept: list[PartDetection] = []
    for part in sorted(parts, key=lambda item: item.confidence, reverse=True):
        if all(
            part.group != existing.group or _iou(part, existing) < 0.5
            for existing in kept
        ):
            kept.append(part)
    return kept


def _side_window_prompts(doors: list[PartDetection]) -> list[PartDetection]:
    """Derive upper-door SAM prompts that recover transparent side windows."""

    doors = _deduplicate(doors)
    if not doors:
        raise PipelineError(
            "missing_masks",
            "Part detection did not find doors needed to segment side windows",
        )
    # Door geometry is more reliable than a generic window box for recovering
    # transparent side glass. The clip keeps SAM out of the lower door panel.
    return [
        PartDetection(
            "windows",
            (
                x1 + (x2 - x1) * 0.2,
                y1 + (y2 - y1) * 0.08,
                x2 - (x2 - x1) * 0.05,
                y1 + (y2 - y1) * 0.32,
            ),
            0,
            clip_box=(x1, y1, x2, y1 + (y2 - y1) * 0.45),
        )
        for door in doors
        for x1, y1, x2, y2 in (door.box,)
    ]


def _with_side_window_prompts(
    parts: list[PartDetection], doors: list[PartDetection]
) -> list[PartDetection]:
    """Add a side-window prompt for every detected door, including 3/4 views."""

    if doors:
        return [*parts, *_side_window_prompts(doors)]
    if not any(part.group == "windows" for part in parts):
        _side_window_prompts(doors)
    return parts


def validate_view(view: ViewName, parts: list[PartDetection]) -> None:
    """Reject a clearly side-on image submitted with a manual face view."""

    if (
        view in {"front", "rear"}
        and not any(part.group == "plate" for part in parts)
        and sum(part.group == "wheels" for part in parts) >= 2
    ):
        raise PipelineError(
            "view_mismatch",
            "The image appears to be a side view; use view=left or view=right",
        )


def detect_car_and_parts(
    image: Image.Image, settings: Settings, view: ViewSelection
) -> tuple[CarDetection, list[PartDetection], ViewName, float | None]:
    """Detect one car and reusable part prompts in full-image coordinates.

    The specialized local model supplies detailed labels and view evidence;
    YOLO-World supplements generic semantic parts. Automatic view resolution
    occurs between those passes and manual views retain legacy validation.
    """

    try:
        with _YOLO_LOCK:
            model = _load_model(
                settings.yolo_model_id,
                _model_classes(settings.yolo_car_class),
            )
            car_result = model.predict(
                image,
                conf=settings.yolo_confidence,
                verbose=False,
            )[0]
    except Exception as exc:
        raise PipelineError("yolo_error", "YOLO-World car detection failed", 500) from exc

    cars = []
    if car_result.boxes is not None:
        for xyxy, confidence, class_id in zip(
            car_result.boxes.xyxy.cpu().tolist(),
            car_result.boxes.conf.cpu().tolist(),
            car_result.boxes.cls.cpu().tolist(),
        ):
            if _normalise(car_result.names[int(class_id)]) != _normalise(
                settings.yolo_car_class
            ):
                continue
            x1, y1, x2, y2 = (
                max(0, min(image.width, round(xyxy[0]))),
                max(0, min(image.height, round(xyxy[1]))),
                max(0, min(image.width, round(xyxy[2]))),
                max(0, min(image.height, round(xyxy[3]))),
            )
            if x2 > x1 and y2 > y1:
                cars.append(CarDetection((x1, y1, x2, y2), float(confidence)))
    car = select_primary_car(cars, settings.competing_car_ratio)

    car_x1, car_y1, car_x2, car_y2 = car.box
    crop = image.crop(car.box)
    parts = []
    doors = []
    view_evidence = []

    try:
        with _YOLO_LOCK:
            specialised_result = _load_parts_model(
                settings.car_parts_model_id
            ).predict(
                crop,
                conf=settings.car_parts_confidence,
                imgsz=settings.car_parts_image_size,
                verbose=False,
            )[0]
    except PipelineError:
        raise
    except Exception as exc:
        raise PipelineError(
            "yolo_error", "Car-parts segmentation failed", 500
        ) from exc

    if specialised_result.boxes is not None:
        polygons = (
            specialised_result.masks.xy
            if specialised_result.masks is not None
            else [None] * len(specialised_result.boxes)
        )
        for xyxy, confidence, class_id, raw_polygon in zip(
            specialised_result.boxes.xyxy.cpu().tolist(),
            specialised_result.boxes.conf.cpu().tolist(),
            specialised_result.boxes.cls.cpu().tolist(),
            polygons,
        ):
            class_name = _normalise(specialised_result.names[int(class_id)])
            group = _part_group(class_name)
            x1, y1, x2, y2 = (
                max(car_x1, min(car_x2, car_x1 + float(xyxy[0]))),
                max(car_y1, min(car_y2, car_y1 + float(xyxy[1]))),
                max(car_x1, min(car_x2, car_x1 + float(xyxy[2]))),
                max(car_y1, min(car_y2, car_y1 + float(xyxy[3]))),
            )
            area = (x2 - x1) * (y2 - y1)
            area_limit = 0.7 if group == "bumper" else MAX_PART_AREA_RATIO
            if x2 > x1 and y2 > y1 and area <= car.area * area_limit:
                view_evidence.append((class_name, float(confidence)))
            if (
                class_name.endswith("_door")
                and x2 > x1
                and y2 > y1
                and area <= car.area * MAX_PART_AREA_RATIO
            ):
                doors.append(
                    PartDetection(
                        "doors",
                        (x1, y1, x2, y2),
                        float(confidence),
                    )
                )
            polygon = (
                tuple(
                    (
                        max(car_x1, min(car_x2, car_x1 + float(point[0]))),
                        max(car_y1, min(car_y2, car_y1 + float(point[1]))),
                    )
                    for point in raw_polygon
                )
                if raw_polygon is not None and len(raw_polygon) >= 3
                else None
            )
            if (
                group is not None
                and x2 > x1
                and y2 > y1
                and area <= car.area * area_limit
            ):
                parts.append(
                    PartDetection(
                        group,
                        (x1, y1, x2, y2),
                        float(confidence),
                        polygon,
                    )
                )

    # Automatic selection consumes the same local detection pass; no extra model
    # or external request is needed.
    resolved_view, view_confidence = (
        infer_view(view_evidence) if view == "auto" else (view, None)
    )

    try:
        with _YOLO_LOCK:
            model = _load_model(
                settings.yolo_model_id,
                _model_classes(settings.yolo_car_class),
            )
            part_result = model.predict(
                crop,
                conf=settings.parts_confidence,
                verbose=False,
            )[0]
    except Exception as exc:
        raise PipelineError(
            "yolo_error", "YOLO-World part detection failed", 500
        ) from exc

    if part_result.boxes is not None:
        for xyxy, confidence, class_id in zip(
            part_result.boxes.xyxy.cpu().tolist(),
            part_result.boxes.conf.cpu().tolist(),
            part_result.boxes.cls.cpu().tolist(),
        ):
            group = _part_group(part_result.names[int(class_id)], resolved_view)
            x1, y1, x2, y2 = (
                max(car_x1, min(car_x2, car_x1 + float(xyxy[0]))),
                max(car_y1, min(car_y2, car_y1 + float(xyxy[1]))),
                max(car_x1, min(car_x2, car_x1 + float(xyxy[2]))),
                max(car_y1, min(car_y2, car_y1 + float(xyxy[3]))),
            )
            area = (x2 - x1) * (y2 - y1)
            if (
                group is not None
                and x2 > x1
                and y2 > y1
                and area <= car.area * MAX_PART_AREA_RATIO
            ):
                parts.append(
                    PartDetection(group, (x1, y1, x2, y2), float(confidence))
                )
    parts = _deduplicate(parts)
    if view != "auto":
        validate_view(resolved_view, parts)
    if resolved_view in {"front", "rear"} and not any(
        part.group == "windows" for part in parts
    ):
        width, height = car_x2 - car_x1, car_y2 - car_y1
        # ponytail: geometric fallback; replace if a trained detector is needed.
        parts.append(
            PartDetection(
                "windows",
                (
                    car_x1 + width * 0.18,
                    car_y1 + height * 0.05,
                    car_x2 - width * 0.18,
                    car_y1 + height * 0.45,
                ),
                0,
            )
        )
    # Three-quarter cars may resolve to front/rear while still exposing doors.
    parts = _with_side_window_prompts(parts, doors)
    return car, parts, resolved_view, view_confidence
