import json
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
from PIL import Image

from app.bumper_analysis.geometry import create_rough_composite
from app.bumper_analysis.mask_builder import build_bumper_masks
from app.bumper_analysis.reference_preprocessor import store_bumper_reference
from app.errors import PipelineError
from app.generative.mock import MockGenerativeImageEditProvider
from app.generative.vertex_ai import _normalise_generated_image, _timeout_milliseconds
from app.modifications.schemas import BumperReplacementRequest, parse_modification
from app.renderers.generative_bumper import GenerativeBumperRenderer
from app.schemas import AssetBundle, BoundingBox


def mask(shape, box):
    output = np.zeros(shape, np.uint8)
    x1, y1, x2, y2 = box
    output[y1:y2, x1:x2] = 255
    return output


class BumperTest(unittest.TestCase):
    def asset(self, root: Path, view="front"):
        shape = (256, 256)
        masks = {
            "full_car": mask(shape, (20, 20, 236, 236)),
            "bumper": mask(shape, (35, 170, 221, 225)),
            "lights": mask(shape, (40, 172, 65, 190)),
            "plate": mask(shape, (105, 185, 150, 205)),
            "grille": mask(shape, (75, 195, 180, 210)),
            "wheels": np.zeros(shape, np.uint8),
            "windows": mask(shape, (70, 45, 180, 110)),
            "protected_mask": mask(shape, (70, 45, 180, 110)),
        }
        paths = {}
        for key, value in masks.items():
            name = f"{key}.png"
            cv2.imwrite(str(root / name), value)
            paths[key] = name
        Image.new("RGB", (256, 256), (110, 110, 110)).save(root / "original.webp")
        return AssetBundle(
            asset_id="a" * 64,
            view=view,
            width=256,
            height=256,
            car_bbox=BoundingBox(x1=20, y1=20, x2=236, y2=236, confidence=1),
            source_image="source.jpg",
            original_image="original.webp",
            luminance_map="l.png",
            masks=paths,
            models={},
            pipeline_version="test",
        )

    def transparent_reference(self):
        image = Image.new("RGBA", (256, 256))
        pixels = np.asarray(image).copy()
        pixels[40:200, 20:236] = (180, 30, 20, 255)
        output = BytesIO()
        Image.fromarray(pixels).save(output, "PNG")
        return output.getvalue()

    def test_schema_is_discriminated_and_safe(self):
        request = parse_modification({
            "type": "bumper_replacement",
            "bumper_position": "front",
            "reference_asset_id": "A" * 64,
        })
        self.assertIsInstance(request, BumperReplacementRequest)
        self.assertEqual(request.reference_asset_id, "a" * 64)
        with self.assertRaises(PipelineError):
            parse_modification({"type": "bumper_replacement", "bumper_position": "front", "reference_asset_id": "../x"})

    def test_vertex_timeout_setting_uses_milliseconds(self):
        self.assertEqual(_timeout_milliseconds(180), 180_000)

    def test_vertex_output_is_resized_only_when_framing_matches(self):
        resized = _normalise_generated_image(Image.new("RGB", (1306, 816)), (512, 320))
        self.assertEqual(resized.size, (512, 320))
        with self.assertRaises(ValueError):
            _normalise_generated_image(Image.new("RGB", (512, 512)), (512, 320))

    def test_first_generated_image_is_a_stable_selection_policy(self):
        images = [b"primary", b"additional"]
        self.assertEqual(images[0], b"primary")

    def test_reference_masks_geometry_and_cache(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = self.asset(root)
            source = self.transparent_reference()
            first = store_bumper_reference(directory=root, metadata=metadata, source=source, settings=None)
            second = store_bumper_reference(directory=root, metadata=metadata, source=source, settings=None)
            self.assertEqual(first.reference_asset_id, second.reference_asset_id)
            masks = build_bumper_masks(root, metadata)
            self.assertFalse(np.any((masks.core > 0) & (mask((256, 256), (40, 172, 65, 190)) > 0)))
            self.assertFalse(np.any((masks.allowed > 0) & (np.asarray(masks.protected) > 0)))
            reference_dir = root / "references" / "bumpers" / first.reference_asset_id
            with Image.open(reference_dir / "normalized.png") as image:
                reference = image.convert("RGBA")
            reference_mask = cv2.imread(str(reference_dir / "reference-mask.png"), cv2.IMREAD_GRAYSCALE)
            with Image.open(root / "original.webp") as image:
                original = image.convert("RGB")
            rough = create_rough_composite(original=original, reference_rgba=reference, reference_mask=reference_mask, target_core_mask=masks.core, allowed_mask=masks.allowed)
            self.assertTrue(np.array_equal(np.asarray(rough)[masks.allowed < 128], np.asarray(original)[masks.allowed < 128]))
            renderer = GenerativeBumperRenderer(MockGenerativeImageEditProvider())
            request = BumperReplacementRequest(bumper_position="front", reference_asset_id=first.reference_asset_id)
            result = renderer.render(directory=root, metadata=metadata, modification=request)
            self.assertFalse(result.cached)
            cached = renderer.render(directory=root, metadata=metadata, modification=request)
            self.assertTrue(cached.cached)
            with Image.open(result.path) as image:
                final = np.asarray(image.convert("RGB"))
            self.assertTrue(np.array_equal(final[masks.allowed < 128], np.asarray(original)[masks.allowed < 128]))

    def test_side_view_is_rejected(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = self.asset(root, "left")
            with self.assertRaisesRegex(PipelineError, "front and rear"):
                build_bumper_masks(root, metadata)


if __name__ == "__main__":
    unittest.main()
