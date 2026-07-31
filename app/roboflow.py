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
        endpoint = "/sam3/concept_segment"
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
        endpoint = "/sam2/segment_image"
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
            f"Roboflow {segmenter.upper()} timed out after {settings.roboflow_timeout:g} seconds",
            504,
        ) from exc
    except requests.ConnectionError as exc:
        raise PipelineError(
            "sam_api_error", f"Could not connect to Roboflow {segmenter.upper()}", 502
        ) from exc
    except requests.RequestException as exc:
        raise PipelineError(
            "sam_api_error", f"Roboflow {segmenter.upper()} request failed", 502
        ) from exc

    data = _json_response(response, "sam_api_error")
    predictions = (
        data.get("prompt_results") if segmenter == "sam3" else data.get("predictions")
    )
    if not isinstance(predictions, list) or len(predictions) != len(boxes):
        raise PipelineError(
            "invalid_sam_response",
            f"{segmenter.upper()} returned an unexpected number of predictions",
            502,
        )

    masks = []
    for index, prediction in enumerate(predictions):
        if segmenter == "sam3":
            items = prediction.get("predictions") if isinstance(prediction, dict) else None
            polygons = [
                polygon
                for item in items or []
                if isinstance(item, dict)
                and item.get("format", "polygon") == "polygon"
                and isinstance(item.get("masks"), list)
                for polygon in item["masks"]
            ]
            if polygons or index:
                masks.append(polygons)
                continue
        else:
            polygons = prediction.get("masks") if isinstance(prediction, dict) else None
            if (
                isinstance(prediction, dict)
                and prediction.get("format", "polygon") == "polygon"
                and isinstance(polygons, list)
                and polygons
            ):
                masks.append(polygons)
                continue
        if (
            segmenter == "sam3" and index == 0
        ):
            raise PipelineError(
                "invalid_sam_response",
                "SAM3 did not find the primary car",
                502,
            )
        raise PipelineError(
            "invalid_sam_response",
            f"{segmenter.upper()} returned an invalid polygon mask",
            502,
        )
    return masks
