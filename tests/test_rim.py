import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

from app.generative.mock import MockGenerativeImageEditProvider
from app.modifications.schemas import RimReplacementRequest, parse_modification
from app.renderers.generative_rim import GenerativeRimRenderer, _rim_edit_mask
from app.rim_analysis import rim_replacement_available, store_rim_reference, wheel_mask
from app.schemas import AssetBundle, BoundingBox


def mask(shape, boxes):
    output = np.zeros(shape, np.uint8)
    for x1, y1, x2, y2 in boxes:
        output[y1:y2, x1:x2] = 255
    return output


class RimTest(unittest.TestCase):
    def asset(self, root: Path, view="left") -> AssetBundle:
        shape = (256, 256)
        masks = {
            "full_car": mask(shape, [(10, 30, 246, 220)]),
            "wheels": mask(shape, [(35, 155, 95, 215), (160, 155, 220, 215)]),
        }
        paths = {}
        for key, value in masks.items():
            name = f"{key}.png"
            cv2.imwrite(str(root / name), value)
            paths[key] = name
        Image.new("RGB", (256, 256), (90, 90, 90)).save(root / "original.webp")
        return AssetBundle(
            asset_id="a" * 64,
            view=view,
            width=256,
            height=256,
            car_bbox=BoundingBox(x1=10, y1=30, x2=246, y2=220, confidence=1),
            source_image="source.jpg",
            original_image="original.webp",
            luminance_map="l.png",
            masks=paths,
            models={},
            pipeline_version="test",
        )

    def rim_source(self) -> bytes:
        image = Image.new("RGB", (256, 256), (255, 255, 255))
        pixels = np.asarray(image).copy()
        cv2.circle(pixels, (128, 128), 80, (20, 20, 20), -1)
        output = BytesIO()
        Image.fromarray(pixels).save(output, "PNG")
        return output.getvalue()

    def test_schema_reference_and_renderer(self):
        request = parse_modification({"type": "rim_replacement", "reference_asset_id": "A" * 64})
        self.assertIsInstance(request, RimReplacementRequest)
        self.assertEqual(request.reference_asset_id, "a" * 64)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = self.asset(root)
            self.assertTrue(rim_replacement_available(root, metadata))
            reference = store_rim_reference(directory=root, metadata=metadata, source=self.rim_source())
            renderer = GenerativeRimRenderer(MockGenerativeImageEditProvider())
            result = renderer.render(directory=root, metadata=metadata, modification=RimReplacementRequest(reference_asset_id=reference.reference_asset_id))
            cached = renderer.render(directory=root, metadata=metadata, modification=RimReplacementRequest(reference_asset_id=reference.reference_asset_id))
            self.assertTrue(cached.cached)
            original = np.asarray(Image.open(root / metadata.original_image).convert("RGB"))
            final = np.asarray(Image.open(result.path).convert("RGB"))
            wheels = wheel_mask(root, metadata)
            self.assertTrue(np.array_equal(final[wheels < 128], original[wheels < 128]))
            self.assertTrue(np.any(final[wheels >= 128] != original[wheels >= 128]))
            rim_mask = _rim_edit_mask(wheels)
            self.assertLess(np.count_nonzero(rim_mask), np.count_nonzero(wheels))
            self.assertTrue(np.array_equal(final[(wheels >= 128) & (rim_mask < 128)], original[(wheels >= 128) & (rim_mask < 128)]))

    def test_front_view_with_wheels_is_supported(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = self.asset(root, "front")
            self.assertTrue(rim_replacement_available(root, metadata))
            reference = store_rim_reference(directory=root, metadata=metadata, source=self.rim_source())
            self.assertEqual(len(reference.reference_asset_id), 64)


if __name__ == "__main__":
    unittest.main()
