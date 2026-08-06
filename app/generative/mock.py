"""A deterministic in-process provider for renderer tests."""

import numpy as np
from PIL import Image


class MockGenerativeImageEditProvider:
    name = "mock"
    model_id = "mock-bumper-1"

    def edit(self, *, original: Image.Image, reference: Image.Image, rough_composite: Image.Image, edit_mask: np.ndarray, instruction: str) -> Image.Image:
        pixels = np.asarray(rough_composite.convert("RGB")).copy()
        original_pixels = np.asarray(original.convert("RGB"))
        # Make a deterministic visible difference even with an all-black reference.
        pixels[edit_mask >= 128, 0] = np.maximum(pixels[edit_mask >= 128, 0], 48)
        pixels[edit_mask < 128] = original_pixels[edit_mask < 128]
        return Image.fromarray(pixels)
