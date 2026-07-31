import os
from dataclasses import dataclass
from pathlib import Path

from app.errors import PipelineError

PART_GROUPS = {
    "wheels": {
        "wheel",
        "wheels",
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
    ],
    "rear": [
        "license plate",
        "black plastic trim",
    ],
    "left": [
        "license plate",
        "black plastic trim",
    ],
    "right": [
        "license plate",
        "black plastic trim",
    ],
}


@dataclass(frozen=True)
class Settings:
    roboflow_api_url: str
    roboflow_api_key: str
    roboflow_sam2_version_id: str
    yolo_model_id: str
    car_parts_model_id: str
    yolo_car_class: str
    yolo_confidence: float
    competing_car_ratio: float
    parts_confidence: float
    car_parts_confidence: float
    roboflow_timeout: float
    mask_kernel_size: int
    mask_feather_radius: int
    storage_root: Path

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
                roboflow_sam2_version_id=os.getenv(
                    "ROBOFLOW_SAM2_VERSION_ID", "hiera_small"
                ),
                yolo_model_id=os.getenv("YOLO_MODEL_ID", "yolov8s-world.pt"),
                car_parts_model_id=os.getenv(
                    "CAR_PARTS_MODEL_ID", "models/carparts-seg.pt"
                ),
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
            )
        except ValueError as exc:
            raise PipelineError(
                "configuration_error",
                "Numeric environment variables contain an invalid value",
                503,
            ) from exc

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
        if not all(
            0 <= value <= 1
            for value in (
                settings.yolo_confidence,
                settings.competing_car_ratio,
                settings.parts_confidence,
                settings.car_parts_confidence,
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
        return settings
