import base64

import requests

from app.config import Settings
from app.errors import PipelineError


def _json_response(response: requests.Response, error_code: str) -> dict:
    if not response.ok:
        raise PipelineError(
            error_code,
            f"Roboflow returned HTTP {response.status_code}",
            502,
        )
    try:
        data = response.json()
    except requests.exceptions.JSONDecodeError as exc:
        raise PipelineError(error_code, "Roboflow returned invalid JSON", 502) from exc
    if not isinstance(data, dict):
        raise PipelineError(error_code, "Roboflow response must be an object", 502)
    return data


def segment_boxes(
    image_jpeg: bytes,
    boxes: list[tuple[float, float, float, float]],
    settings: Settings,
) -> list[list]:
    prompts = []
    for x1, y1, x2, y2 in boxes:
        prompts.append(
            {
                "box": {
                    "x": (x1 + x2) / 2,
                    "y": (y1 + y2) / 2,
                    "width": x2 - x1,
                    "height": y2 - y1,
                }
            }
        )

    payload = {
        "image": {
            "type": "base64",
            "value": base64.b64encode(image_jpeg).decode(),
        },
        "prompts": {"prompts": prompts},
        "sam2_version_id": settings.roboflow_sam2_version_id,
        "multimask_output": False,
        "format": "json",
    }
    try:
        response = requests.post(
            f"{settings.roboflow_api_url}/sam2/segment_image",
            params={"api_key": settings.roboflow_api_key},
            json=payload,
            timeout=settings.roboflow_timeout,
        )
    except requests.Timeout as exc:
        raise PipelineError(
            "sam_api_timeout",
            f"Roboflow SAM 2 timed out after {settings.roboflow_timeout:g} seconds",
            504,
        ) from exc
    except requests.ConnectionError as exc:
        raise PipelineError(
            "sam_api_error", "Could not connect to Roboflow SAM 2", 502
        ) from exc
    except requests.RequestException as exc:
        raise PipelineError("sam_api_error", "Roboflow SAM 2 request failed", 502) from exc

    predictions = _json_response(response, "sam_api_error").get("predictions")
    if not isinstance(predictions, list) or len(predictions) != len(prompts):
        raise PipelineError(
            "invalid_sam_response",
            "SAM 2 returned an unexpected number of predictions",
            502,
        )

    masks = []
    for prediction in predictions:
        if (
            not isinstance(prediction, dict)
            or prediction.get("format", "polygon") != "polygon"
            or not isinstance(prediction.get("masks"), list)
            or not prediction["masks"]
        ):
            raise PipelineError(
                "invalid_sam_response",
                "SAM 2 returned an invalid polygon mask",
                502,
            )
        masks.append(prediction["masks"])
    return masks
