import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

from PIL import Image

from app.config import PART_GROUPS, YOLO_PART_PROMPTS_BY_VIEW, Settings
from app.errors import PipelineError
from app.schemas import ViewName

_YOLO_LOCK = Lock()
MAX_PART_AREA_RATIO = 0.45


@dataclass(frozen=True)
class CarDetection:
    box: tuple[int, int, int, int]
    confidence: float

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return (x2 - x1) * (y2 - y1)


@dataclass(frozen=True)
class PartDetection:
    group: str
    box: tuple[float, float, float, float]
    confidence: float
    polygon: tuple[tuple[float, float], ...] | None = None
    clip_box: tuple[float, float, float, float] | None = None


def _normalise(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _part_group(class_name: object) -> str | None:
    name = _normalise(class_name)
    return next(
        (group for group, aliases in PART_GROUPS.items() if name in aliases),
        None,
    )


@lru_cache(maxsize=4)
def _load_model(model_id: str, classes: tuple[str, ...]) -> Any:
    from ultralytics import YOLOWorld

    model = YOLOWorld(model_id)
    model.set_classes(list(classes))
    return model


@lru_cache(maxsize=1)
def _load_parts_model(model_id: str) -> Any:
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
    ax1, ay1, ax2, ay2 = first.box
    bx1, by1, bx2, by2 = second.box
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = (ax2 - ax1) * (ay2 - ay1)
    second_area = (bx2 - bx1) * (by2 - by1)
    return intersection / (first_area + second_area - intersection)


def _deduplicate(parts: list[PartDetection]) -> list[PartDetection]:
    kept: list[PartDetection] = []
    for part in sorted(parts, key=lambda item: item.confidence, reverse=True):
        if all(
            part.group != existing.group or _iou(part, existing) < 0.5
            for existing in kept
        ):
            kept.append(part)
    return kept


def _side_window_prompts(doors: list[PartDetection]) -> list[PartDetection]:
    doors = _deduplicate(doors)
    if not doors:
        raise PipelineError(
            "missing_masks",
            "Part detection did not find doors needed to segment side windows",
        )
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


def validate_view(view: ViewName, parts: list[PartDetection]) -> None:
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
    image: Image.Image, settings: Settings, view: ViewName
) -> tuple[CarDetection, list[PartDetection]]:
    try:
        with _YOLO_LOCK:
            car_result = _load_model(
                settings.yolo_model_id, (settings.yolo_car_class,)
            ).predict(
                image,
                conf=settings.yolo_confidence,
                verbose=False,
            )[0]
    except Exception as exc:
        raise PipelineError("yolo_error", "YOLO-World car detection failed", 500) from exc

    cars = []
    if car_result.boxes is not None:
        for xyxy, confidence in zip(
            car_result.boxes.xyxy.cpu().tolist(),
            car_result.boxes.conf.cpu().tolist(),
        ):
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

    try:
        with _YOLO_LOCK:
            specialised_result = _load_parts_model(
                settings.car_parts_model_id
            ).predict(
                crop,
                conf=settings.car_parts_confidence,
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

    try:
        with _YOLO_LOCK:
            part_result = _load_model(
                settings.yolo_model_id, tuple(YOLO_PART_PROMPTS_BY_VIEW[view])
            ).predict(
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
            group = _part_group(part_result.names[int(class_id)])
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
    validate_view(view, parts)
    if view in {"front", "rear"} and not any(
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
    if view in {"left", "right"} and not any(
        part.group == "windows" for part in parts
    ):
        parts.extend(_side_window_prompts(doors))
    return car, parts
