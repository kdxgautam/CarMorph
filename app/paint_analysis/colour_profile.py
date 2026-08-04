"""Small LAB colour utilities used by paint classification and rendering."""

import cv2
import numpy as np


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert uint8-style RGB data to OpenCV's floating-point CIE LAB."""

    source = np.asarray(rgb, dtype=np.float32) / 255
    return cv2.cvtColor(source, cv2.COLOR_RGB2LAB)


def lab_colour_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return Euclidean distance across lightness and both chroma channels."""

    return np.linalg.norm(np.asarray(first) - np.asarray(second), axis=-1)


def chroma_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Compare paint chroma while deliberately ignoring illumination."""

    return np.linalg.norm(np.asarray(first)[..., 1:] - np.asarray(second)[..., 1:], axis=-1)


def lightness_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return absolute LAB lightness difference."""

    return np.abs(np.asarray(first)[..., 0] - np.asarray(second)[..., 0])


def robust_median_colour(lab: np.ndarray) -> np.ndarray:
    """Estimate a colour centre resistant to highlights and compression noise."""

    return np.median(np.asarray(lab), axis=0)


def robust_variance(lab: np.ndarray) -> np.ndarray:
    """Return median squared deviation per LAB channel."""

    values = np.asarray(lab)
    median = np.median(values, axis=0)
    return np.median((values - median) ** 2, axis=0)


def cluster_confidence(samples: np.ndarray, centre: np.ndarray, chroma_threshold: float) -> float:
    """Score cluster membership and compactness on a normalized 0..1 scale."""

    if not len(samples):
        return 0
    related = chroma_distance(samples, centre) <= chroma_threshold
    spread = float(np.median(chroma_distance(samples, centre)))
    return round(float(np.mean(related) * max(0, 1 - spread / max(chroma_threshold, 1))), 4)
