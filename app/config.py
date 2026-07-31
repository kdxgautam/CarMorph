import os
from dataclasses import dataclass
from pathlib import Path

from app.errors import PipelineError

PART_GROUPS = {
    "wheels": {
        "wheel",
        "wheels",
        "front_wheel",
        "back_wheel",
        "car_wheel",
        "alloy",
        "alloy_wheel",
        "rim",
        "tire",
        "tyre",
    },
    "windows": {
        "window",
        "windows",
        "glass",
        "front_glass",
        "back_glass",
        "front_window",
        "back_window",
        "back_windshield",
        "rear_glass",
        "windscreen",
        "windshield",
        "car_windshield",
        "car_window",
        "car_side_window",
    },
    "plate": {
        "plate",
        "number_plate",
        "license_plate",
        "registration_plate",
    },
    "lights": {
        "light",
        "lights",
        "back_left_light",
        "back_right_light",
        "front_left_light",
        "front_light",
        "front_right_light",
        "back_light",
        "headlight",
        "car_headlight",
        "headlights",
        "taillight",
        "tail_light",
        "car_taillight",
        "taillights",
        "fog_lamp",
    },
    "grille": {"grille", "grill", "car_grille"},
    "trim": {
        "trim",
        "car_trim",
        "black_plastic_trim",
        "chrome_trim",
        "black_trim",
        "plastic_trim",
        "badge",
    },
    "bumper": {"front_bumper", "back_bumper", "rear_bumper"},
    "mirrors": {"left_mirror", "right_mirror", "mirror"},
    "handles": {"door_handle", "car_door_handle", "handle"},
    "roof": {"roof", "car_roof"},
    "spoiler": {"spoiler", "car_spoiler"},
    "pillars": {"window_pillar", "b_pillar", "c_pillar"},
}
OUTPUT_PART_GROUPS = {
    "wheels",
    "windows",
    "plate",
    "lights",
    "grille",
    "trim",
    "bumper",
    "mirrors",
    "dark_trim",
    "handles",
    "roof",
    "spoiler",
    "pillars",
}
NON_PAINTABLE_PART_GROUPS = {
    "wheels",
    "windows",
    "plate",
    "lights",
    "grille",
    "trim",
}
REQUIRED_PART_GROUPS_BY_VIEW = {
    "front": {"windows", "plate"},
    "rear": {"windows", "plate"},
    "left": {"wheels", "windows"},
    "right": {"wheels", "windows"},
}
YOLO_PART_PROMPTS_BY_VIEW = {
    "front": [
        "license plate",
        "car grille",
        "black plastic trim",
        "car roof",
        "car spoiler",
    ],
    "rear": [
        "license plate",
        "black plastic trim",
        "car roof",
        "car spoiler",
    ],
    "left": [
        "license plate",
        "black plastic trim",
        "car door handle",
        "car roof",
        "window pillar",
    ],
    "right": [
        "license plate",
        "black plastic trim",
        "car door handle",
        "car roof",
        "window pillar",
    ],
}


@dataclass(frozen=True)
class Settings:
    roboflow_api_url: str
    roboflow_api_key: str
    roboflow_segmenter: str
    roboflow_sam2_version_id: str
    roboflow_sam3_model_id: str
    yolo_model_id: str
    car_parts_model_id: str
    car_parts_image_size: int
    yolo_car_class: str
    yolo_confidence: float
    competing_car_ratio: float
    parts_confidence: float
    car_parts_confidence: float
    roboflow_timeout: float
    mask_kernel_size: int
    mask_feather_radius: int
    storage_root: Path
    body_paint_chroma_threshold: float
    body_paint_lightness_threshold: float
    body_paint_strict_chroma_threshold: float
    paint_group_min_confidence: float
    paint_group_uncertain_threshold: float
    anchor_erosion_pixels: int
    anchor_min_sample_count: int
    paint_analysis_diagnostics: bool
    body_seed_min_neighbours: int
    body_growth_chroma_threshold: float
    body_growth_local_lab_threshold: float
    body_growth_max_gradient: float
    body_growth_min_neighbours: int
    body_growth_max_iterations: int
    body_completion_kernel_size: int
    body_completion_max_hole_area: int
    body_region_min_boundary_ratio: float
    body_fragment_min_area: int

    @classmethod
    def from_env(cls) -> "Settings":
        required = ("ROBOFLOW_API_KEY",)
        missing = [name for name in required if not os.getenv(name)]
        if missing:
            raise PipelineError(
                "configuration_error",
                f"Missing environment variables: {', '.join(missing)}",
                503,
            )

        try:
            settings = cls(
                roboflow_api_url=os.getenv(
                    "ROBOFLOW_API_URL", "https://serverless.roboflow.com"
                ).rstrip("/"),
                roboflow_api_key=os.environ["ROBOFLOW_API_KEY"],
                roboflow_segmenter=os.getenv("ROBOFLOW_SEGMENTER", "sam3").lower(),
                roboflow_sam2_version_id=os.getenv(
                    "ROBOFLOW_SAM2_VERSION_ID", "hiera_small"
                ),
                roboflow_sam3_model_id=os.getenv(
                    "ROBOFLOW_SAM3_MODEL_ID", "sam3/sam3_final"
                ),
                yolo_model_id=os.getenv("YOLO_MODEL_ID", "yolov8s-world.pt"),
                car_parts_model_id=os.getenv(
                    "CAR_PARTS_MODEL_ID", "models/carparts-v2.pt"
                ),
                car_parts_image_size=int(os.getenv("CAR_PARTS_IMAGE_SIZE", "896")),
                yolo_car_class=os.getenv("YOLO_CAR_CLASS", "car"),
                yolo_confidence=float(os.getenv("YOLO_CONFIDENCE", "0.35")),
                competing_car_ratio=float(
                    os.getenv("COMPETING_CAR_AREA_RATIO", "0.5")
                ),
                parts_confidence=float(os.getenv("PARTS_CONFIDENCE", "0.008")),
                car_parts_confidence=float(
                    os.getenv("CAR_PARTS_CONFIDENCE", "0.25")
                ),
                roboflow_timeout=float(os.getenv("ROBOFLOW_TIMEOUT_SECONDS", "180")),
                mask_kernel_size=int(os.getenv("MASK_KERNEL_SIZE", "5")),
                mask_feather_radius=int(os.getenv("MASK_FEATHER_RADIUS", "3")),
                storage_root=Path(
                    os.getenv("STORAGE_ROOT", "data/processed")
                ).resolve(),
                body_paint_chroma_threshold=float(
                    os.getenv("BODY_PAINT_CHROMA_THRESHOLD", "18")
                ),
                body_paint_lightness_threshold=float(
                    os.getenv("BODY_PAINT_LIGHTNESS_THRESHOLD", "35")
                ),
                body_paint_strict_chroma_threshold=float(
                    os.getenv("BODY_PAINT_STRICT_CHROMA_THRESHOLD", "10")
                ),
                paint_group_min_confidence=float(
                    os.getenv("PAINT_GROUP_MIN_CONFIDENCE", "0.7")
                ),
                paint_group_uncertain_threshold=float(
                    os.getenv("PAINT_GROUP_UNCERTAIN_THRESHOLD", "0.45")
                ),
                anchor_erosion_pixels=int(os.getenv("ANCHOR_EROSION_PIXELS", "5")),
                anchor_min_sample_count=int(
                    os.getenv("ANCHOR_MIN_SAMPLE_COUNT", "500")
                ),
                paint_analysis_diagnostics=os.getenv(
                    "PAINT_ANALYSIS_DIAGNOSTICS", "true"
                ).lower()
                not in {"0", "false", "no"},
                body_seed_min_neighbours=int(
                    os.getenv("BODY_SEED_MIN_NEIGHBOURS", "4")
                ),
                body_growth_chroma_threshold=float(
                    os.getenv("BODY_GROWTH_CHROMA_THRESHOLD", "48")
                ),
                body_growth_local_lab_threshold=float(
                    os.getenv("BODY_GROWTH_LOCAL_LAB_THRESHOLD", "20")
                ),
                body_growth_max_gradient=float(
                    os.getenv("BODY_GROWTH_MAX_GRADIENT", "55")
                ),
                body_growth_min_neighbours=int(
                    os.getenv("BODY_GROWTH_MIN_NEIGHBOURS", "2")
                ),
                body_growth_max_iterations=int(
                    os.getenv("BODY_GROWTH_MAX_ITERATIONS", "16")
                ),
                body_completion_kernel_size=int(
                    os.getenv("BODY_COMPLETION_KERNEL_SIZE", "7")
                ),
                body_completion_max_hole_area=int(
                    os.getenv("BODY_COMPLETION_MAX_HOLE_AREA", "2000")
                ),
                body_region_min_boundary_ratio=float(
                    os.getenv("BODY_REGION_MIN_BOUNDARY_RATIO", "0.18")
                ),
                body_fragment_min_area=int(
                    os.getenv("BODY_FRAGMENT_MIN_AREA", "64")
                ),
            )
        except ValueError as exc:
            raise PipelineError(
                "configuration_error",
                "Numeric environment variables contain an invalid value",
                503,
            ) from exc

        if settings.car_parts_image_size < 1:
            raise PipelineError(
                "configuration_error",
                "CAR_PARTS_IMAGE_SIZE must be positive",
                503,
            )
        if settings.roboflow_segmenter not in {"sam2", "sam3"}:
            raise PipelineError(
                "configuration_error", "ROBOFLOW_SEGMENTER must be sam2 or sam3", 503
            )
        if settings.mask_kernel_size < 1 or settings.mask_kernel_size % 2 == 0:
            raise PipelineError(
                "configuration_error",
                "MASK_KERNEL_SIZE must be a positive odd number",
                503,
            )
        if settings.mask_feather_radius < 0:
            raise PipelineError(
                "configuration_error", "MASK_FEATHER_RADIUS cannot be negative", 503
            )
        if settings.anchor_erosion_pixels < 0 or settings.anchor_min_sample_count < 1:
            raise PipelineError(
                "configuration_error",
                "Anchor erosion must be non-negative and sample count positive",
                503,
            )
        if (
            settings.body_seed_min_neighbours < 1
            or settings.body_growth_min_neighbours < 1
            or settings.body_growth_max_iterations < 1
            or settings.body_completion_max_hole_area < 1
            or settings.body_fragment_min_area < 1
        ):
            raise PipelineError(
                "configuration_error",
                "Body-surface growth counts must be positive",
                503,
            )
        if (
            settings.body_completion_kernel_size < 1
            or settings.body_completion_kernel_size % 2 == 0
        ):
            raise PipelineError(
                "configuration_error",
                "BODY_COMPLETION_KERNEL_SIZE must be a positive odd number",
                503,
            )
        if not all(
            0 <= value <= 1
            for value in (
                settings.yolo_confidence,
                settings.competing_car_ratio,
                settings.parts_confidence,
                settings.car_parts_confidence,
                settings.paint_group_min_confidence,
                settings.paint_group_uncertain_threshold,
                settings.body_region_min_boundary_ratio,
            )
        ):
            raise PipelineError(
                "configuration_error",
                "Confidence and competing-car ratio values must be between 0 and 1",
                503,
            )
        if settings.roboflow_timeout <= 0:
            raise PipelineError(
                "configuration_error",
                "ROBOFLOW_TIMEOUT_SECONDS must be positive",
                503,
            )
        if min(
            settings.body_paint_chroma_threshold,
            settings.body_paint_lightness_threshold,
            settings.body_paint_strict_chroma_threshold,
            settings.body_growth_chroma_threshold,
            settings.body_growth_local_lab_threshold,
            settings.body_growth_max_gradient,
        ) <= 0:
            raise PipelineError(
                "configuration_error",
                "Body-paint colour thresholds must be positive",
                503,
            )
        return settings
