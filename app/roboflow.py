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


def _post(
    payload: dict, endpoint: str, label: str, settings: Settings
) -> dict:
    try:
        response = requests.post(
            f"{settings.roboflow_api_url}{endpoint}",
            params={"api_key": settings.roboflow_api_key},
            json=payload,
            timeout=settings.roboflow_timeout,
        )
    except requests.Timeout as exc:
        raise PipelineError(
            "sam_api_timeout",
            f"Roboflow {label} timed out after {settings.roboflow_timeout:g} seconds",
            504,
        ) from exc
    except requests.ConnectionError as exc:
        raise PipelineError(
            "sam_api_error", f"Could not connect to Roboflow {label}", 502
        ) from exc
    except requests.RequestException as exc:
        raise PipelineError(
            "sam_api_error", f"Roboflow {label} request failed", 502
        ) from exc
    return _json_response(response, "sam_api_error")


def _sam3_masks(data: dict, count: int, *, require_first: bool) -> list[list]:
    results = data.get("prompt_results")
    if not isinstance(results, list) or len(results) != count:
        raise PipelineError(
            "invalid_sam_response",
            "SAM3 returned an unexpected number of predictions",
            502,
        )
    masks = []
    for index, result in enumerate(results):
        predictions = result.get("predictions") if isinstance(result, dict) else None
        polygons = [
            polygon
            for prediction in predictions or []
            if isinstance(prediction, dict)
            and prediction.get("format", "polygon") == "polygon"
            and isinstance(prediction.get("masks"), list)
            for polygon in prediction["masks"]
        ]
        if require_first and index == 0 and not polygons:
            raise PipelineError(
                "invalid_sam_response", "SAM3 did not find the primary car", 502
            )
        masks.append(polygons)
    return masks


def segment_boxes(
    image_jpeg: bytes,
    boxes: list[tuple[float, float, float, float]],
    settings: Settings,
    concepts: list[str] | None = None,
) -> list[list]:
    segmenter = getattr(settings, "roboflow_segmenter", "sam2")
    if segmenter == "sam3":
        if concepts is None or len(concepts) != len(boxes):
            raise PipelineError(
                "configuration_error", "SAM 3 requires one concept per box", 503
            )
        payload = {
            "image": {
                "type": "base64",
                "value": base64.b64encode(image_jpeg).decode(),
            },
            "prompts": [{"type": "text", "text": concept} for concept in concepts],
            "model_id": settings.roboflow_sam3_model_id,
            "output_prob_thresh": 0.5,
            "format": "polygon",
        }
        return _sam3_masks(
            _post(payload, "/sam3/concept_segment", "SAM3", settings),
            len(boxes),
            require_first=True,
        )
    else:
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
    predictions = _post(payload, "/sam2/segment_image", "SAM2", settings).get(
        "predictions"
    )
    if not isinstance(predictions, list) or len(predictions) != len(boxes):
        raise PipelineError(
            "invalid_sam_response",
            "SAM2 returned an unexpected number of predictions",
            502,
        )

    masks = []
    for prediction in predictions:
        polygons = prediction.get("masks") if isinstance(prediction, dict) else None
        if (
            not isinstance(prediction, dict)
            or prediction.get("format", "polygon") != "polygon"
            or not isinstance(polygons, list)
            or not polygons
        ):
            raise PipelineError(
                "invalid_sam_response", "SAM2 returned an invalid polygon mask", 502
            )
        masks.append(polygons)
    return masks


def segment_concepts(
    image_jpeg: bytes, concepts: list[str], settings: Settings
) -> list[list]:
    payload = {
        "image": {
            "type": "base64",
            "value": base64.b64encode(image_jpeg).decode(),
        },
        "prompts": [{"type": "text", "text": concept} for concept in concepts],
        "model_id": settings.roboflow_sam3_model_id,
        "output_prob_thresh": 0.5,
        "format": "polygon",
    }
    return _sam3_masks(
        _post(payload, "/sam3/concept_segment", "SAM3", settings),
        len(concepts),
        require_first=False,
    )
