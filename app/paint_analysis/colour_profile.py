import cv2
import numpy as np


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    source = np.asarray(rgb, dtype=np.float32) / 255
    return cv2.cvtColor(source, cv2.COLOR_RGB2LAB)


def lab_colour_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(first) - np.asarray(second), axis=-1)


def chroma_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.linalg.norm(np.asarray(first)[..., 1:] - np.asarray(second)[..., 1:], axis=-1)


def lightness_difference(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    return np.abs(np.asarray(first)[..., 0] - np.asarray(second)[..., 0])


def robust_median_colour(lab: np.ndarray) -> np.ndarray:
    return np.median(np.asarray(lab), axis=0)


def robust_variance(lab: np.ndarray) -> np.ndarray:
    values = np.asarray(lab)
    median = np.median(values, axis=0)
    return np.median((values - median) ** 2, axis=0)


def cluster_confidence(samples: np.ndarray, centre: np.ndarray, chroma_threshold: float) -> float:
    if not len(samples):
        return 0
    related = chroma_distance(samples, centre) <= chroma_threshold
    spread = float(np.median(chroma_distance(samples, centre)))
    return round(float(np.mean(related) * max(0, 1 - spread / max(chroma_threshold, 1))), 4)
