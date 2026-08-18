import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from app.errors import PipelineError
from app.generative.mock import MockGenerativeImageEditProvider
from app.main import studio_render
from app.modifications.schemas import StudioRenderRequest, VehicleIdentity, parse_modification
from app.renderers.base import RenderResult
from app.renderers.generative_studio import GenerativeStudioRenderer, build_studio_prompt
from app.schemas import AssetBundle, BoundingBox
from app.studio_references import store_studio_reference


def mask(shape, box):
    output = np.zeros(shape, np.uint8)
    x1, y1, x2, y2 = box
    output[y1:y2, x1:x2] = 255
    return output


class StudioTest(unittest.TestCase):
    def asset(self, root: Path, *, plate: bool = True) -> AssetBundle:
        shape = (96, 128)
        masks = {"full_car": mask(shape, (18, 20, 110, 78))}
        if plate:
            masks["plate"] = mask(shape, (56, 50, 72, 58))
        paths = {}
        for key, value in masks.items():
            name = f"{key}.png"
            cv2.imwrite(str(root / name), value)
            paths[key] = name
        pixels = np.full((96, 128, 3), 90, np.uint8)
        pixels[20:78, 18:110] = (40, 80, 130)
        pixels[50:58, 56:72] = (230, 230, 210)
        Image.fromarray(pixels).save(root / "original.webp")
        return AssetBundle(
            asset_id="a" * 64,
            view="front",
            width=128,
            height=96,
            car_bbox=BoundingBox(x1=18, y1=20, x2=110, y2=78, confidence=1),
            source_image="source.jpg",
            original_image="original.webp",
            luminance_map="l.png",
            masks=paths,
            models={},
            pipeline_version="test",
        )

    def test_schema_prompt_and_styles(self):
        request = parse_modification(
            {"type": "studio_render", "style": "dark_studio", "fidelity": "high", "preserve_plate": False}
        )
        self.assertIsInstance(request, StudioRenderRequest)
        self.assertEqual(request.style.value, "dark_studio")
        self.assertFalse(request.preserve_plate)
        with self.assertRaises(PipelineError):
            parse_modification({"type": "studio_render", "style": "new_car"})
        prompt = build_studio_prompt(StudioRenderRequest(style="premium_gradient"))
        self.assertIn("Preserve the exact vehicle", prompt)
        self.assertIn("Do not change the wheels", prompt)
        self.assertIn("soft halo", prompt)

    def test_renderer_preserves_plate_and_uses_cache(self):
        class PlateChangingProvider(MockGenerativeImageEditProvider):
            def edit(self, **kwargs):
                generated = np.asarray(super().edit(**kwargs).convert("RGB")).copy()
                generated[50:58, 56:72] = (255, 0, 0)
                return Image.fromarray(generated)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = self.asset(root)
            renderer = GenerativeStudioRenderer(PlateChangingProvider())
            result = renderer.render(
                directory=root,
                metadata=metadata,
                modification=StudioRenderRequest(style="light_studio"),
            )
            cached = renderer.render(
                directory=root,
                metadata=metadata,
                modification=StudioRenderRequest(style="light_studio"),
            )
            final = np.asarray(Image.open(result.path).convert("RGB"))
            original = np.asarray(Image.open(root / metadata.original_image).convert("RGB"))
            self.assertFalse(result.cached)
            self.assertTrue(cached.cached)
            self.assertTrue(np.array_equal(final[52, 58], original[52, 58]))

    def test_missing_plate_mask_warns_but_renders(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = self.asset(root, plate=False)
            result = GenerativeStudioRenderer(MockGenerativeImageEditProvider()).render(
                directory=root,
                metadata=metadata,
                modification=StudioRenderRequest(preserve_plate=True),
            )
            self.assertEqual(result.quality_status, "passed")
            self.assertEqual(result.warnings, ["plate_mask_missing"])

    def test_provider_failure_bubbles(self):
        class FailingProvider(MockGenerativeImageEditProvider):
            def edit(self, **kwargs):
                raise PipelineError("generative_provider_error", "failed", 502)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = self.asset(root)
            with self.assertRaisesRegex(PipelineError, "failed"):
                GenerativeStudioRenderer(FailingProvider()).render(
                    directory=root,
                    metadata=metadata,
                    modification=StudioRenderRequest(),
                )

    def test_references_are_prioritized_and_protected_details_are_restored(self):
        class CapturingProvider(MockGenerativeImageEditProvider):
            def __init__(self):
                self.calls = []

            def edit(self, **kwargs):
                self.calls.append(kwargs)
                return Image.new("RGB", kwargs["original"].size, (255, 0, 0))

        def image_bytes(colour):
            from io import BytesIO

            output = BytesIO()
            Image.new("RGB", (24, 24), colour).save(output, "PNG")
            return output.getvalue()

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            metadata = self.asset(root)
            lights = mask((96, 128), (24, 28, 34, 38))
            cv2.imwrite(str(root / "lights.png"), lights)
            metadata.masks["lights"] = "lights.png"
            user = store_studio_reference(directory=root, source=image_bytes((10, 200, 10)), kind="user")
            side = store_studio_reference(directory=root, source=image_bytes((10, 10, 200)), kind="user")
            provider = CapturingProvider()
            renderer = GenerativeStudioRenderer(provider)
            request = StudioRenderRequest(
                vehicle_identity=VehicleIdentity(
                    make="Honda", model="Civic", generation="10th generation",
                    body_style="sedan", visual_cues=["C-shaped lamps"], confidence=0.9,
                ),
                reference_asset_ids=[user.reference_asset_id, side.reference_asset_id],
            )
            result = renderer.render(directory=root, metadata=metadata, modification=request)
            user_only = renderer.render(
                directory=root,
                metadata=metadata,
                modification=request.model_copy(update={"reference_asset_ids": [user.reference_asset_id]}),
            )

            references = provider.calls[0]["additional_references"]
            self.assertEqual(references[0].getpixel((0, 0)), (10, 200, 10))
            self.assertEqual(references[1].getpixel((0, 0)), (10, 10, 200))
            self.assertIn("2 user-supplied supporting views", provider.calls[0]["instruction"])
            self.assertEqual(provider.calls[0]["edit_mask"][30, 28], 0)
            final = Image.open(result.path).convert("RGB")
            original = Image.open(root / metadata.original_image).convert("RGB")
            self.assertEqual(final.getpixel((28, 30)), original.getpixel((28, 30)))
            self.assertNotEqual(result.path, user_only.path)

    def test_dedicated_endpoint_routes_studio_render(self):
        with TemporaryDirectory() as temporary:
            asset_id = "a" * 64
            directory = Path(temporary) / asset_id
            directory.mkdir()
            result = directory / "result.png"
            Image.new("RGB", (4, 4), (1, 2, 3)).save(result)
            metadata = AssetBundle(
                asset_id=asset_id,
                view="front",
                width=4,
                height=4,
                car_bbox=BoundingBox(x1=0, y1=0, x2=4, y2=4, confidence=1),
                source_image="source.jpg",
                original_image="original.webp",
                luminance_map="luminance-map.png",
                masks={"full_car": "full-car.png"},
                models={},
            )
            (directory / "metadata.json").write_text(metadata.model_dump_json(), encoding="utf-8")

            with patch.dict("os.environ", {"STORAGE_ROOT": temporary, "GOOGLE_CLOUD_PROJECT": "test"}), patch(
                "app.main.vertex_provider", return_value=MockGenerativeImageEditProvider()
            ), patch(
                "app.main.GenerativeStudioRenderer.render",
                return_value=RenderResult(result, False, "generative-studio", "passed", []),
            ) as render:
                response = studio_render(asset_id, StudioRenderRequest(style="light_studio"))

            render.assert_called_once()
            self.assertEqual(response.headers["x-renderer-used"], "generative-studio")


if __name__ == "__main__":
    unittest.main()
