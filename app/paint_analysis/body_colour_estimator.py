import cv2
import numpy as np
from PIL import Image

from app.paint_analysis.colour_profile import (
    chroma_distance,
    cluster_confidence,
    rgb_to_lab,
    robust_median_colour,
    robust_variance,
)
from app.paint_analysis.schemas import BodyPaintProfile


def estimate_body_paint(
    image: Image.Image,
    body_candidates: np.ndarray,
    *,
    erosion_pixels: int,
    min_samples: int,
    chroma_threshold: float,
) -> tuple[BodyPaintProfile, np.ndarray]:
    source = np.asarray(image.convert("RGB"))
    candidate = np.where(body_candidates >= 128, 255, 0).astype(np.uint8)
    if erosion_pixels:
        size = erosion_pixels * 2 + 1
        candidate = cv2.erode(
            candidate, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        )
    lab = rgb_to_lab(source)
    lightness = lab[:, :, 0]
    values = lightness[candidate > 0]
    if not len(values):
        warning = "no_reliable_body_paint_anchor"
        return BodyPaintProfile(warnings=[warning]), np.zeros_like(candidate)

    low, high = np.percentile(values, (10, 90))
    anchors = (candidate > 0) & (lightness >= low) & (lightness <= high)
    samples = lab[anchors]
    if not len(samples):
        return (
            BodyPaintProfile(warnings=["no_reliable_body_paint_anchor"]),
            np.zeros_like(candidate),
        )

    # Chroma clusters keep the same paint together across illumination changes.
    chroma = np.float32(samples[:, 1:])
    chroma_centre = np.median(chroma, axis=0)
    cluster_count = (
        min(3, len(chroma))
        if np.percentile(chroma_distance(chroma, chroma_centre), 90)
        > chroma_threshold
        else 1
    )
    if cluster_count > 1:
        cv2.setRNGSeed(0)
        _, labels, _ = cv2.kmeans(
            chroma,
            cluster_count,
            None,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
            3,
            cv2.KMEANS_PP_CENTERS,
        )
        counts = np.bincount(labels.ravel(), minlength=cluster_count)
        selected = int(np.argmax(counts))
        samples = samples[labels.ravel() == selected]
        selected_pixels = np.flatnonzero(anchors)[labels.ravel() == selected]
        anchors[:] = False
        anchors.flat[selected_pixels] = True

    median = robust_median_colour(samples)
    confidence = cluster_confidence(samples, median, chroma_threshold)
    warnings = []
    if len(samples) < min_samples:
        confidence *= len(samples) / max(min_samples, 1)
        warnings.append("insufficient_body_paint_anchor_samples")
    hsv = cv2.cvtColor(source, cv2.COLOR_RGB2HSV).astype(np.float32)
    median_hsv = np.median(hsv[anchors], axis=0)
    profile = BodyPaintProfile(
        dominant_lab=[round(float(value), 3) for value in median],
        median_lab=[round(float(value), 3) for value in median],
        lab_variance=[
            round(float(value), 3) for value in robust_variance(samples)
        ],
        dominant_hsv=[
            round(float(median_hsv[0] * 2), 3),
            round(float(median_hsv[1] / 255), 4),
            round(float(median_hsv[2] / 255), 4),
        ],
        highlight_lab_range={
            "min": round(float(np.percentile(samples[:, 0], 80)), 3),
            "max": round(float(np.max(samples[:, 0])), 3),
        },
        midtone_lab_range={
            "min": round(float(np.percentile(samples[:, 0], 20)), 3),
            "max": round(float(np.percentile(samples[:, 0], 80)), 3),
        },
        shadow_lab_range={
            "min": round(float(np.min(samples[:, 0])), 3),
            "max": round(float(np.percentile(samples[:, 0], 20)), 3),
        },
        sample_count=len(samples),
        anchor_regions=["reliable_body_panel_interior"],
        confidence=round(float(confidence), 4),
        warnings=warnings,
    )
    return profile, np.where(anchors, 255, 0).astype(np.uint8)
