"""Provider contract shared by production and test image editors."""

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from PIL import Image


class GenerativeImageEditProvider(Protocol):
    name: str
    model_id: str

    def edit(
        self,
        *,
        original: Image.Image,
        reference: Image.Image,
        rough_composite: Image.Image,
        edit_mask: np.ndarray,
        instruction: str,
        additional_references: Sequence[Image.Image] = (),
    ) -> Image.Image: ...
