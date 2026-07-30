import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from requests import Timeout

from app.detection import (
    CarDetection,
    PartDetection,
    _deduplicate,
    _side_window_prompts,
    select_primary_car,
    validate_view,
)
from app.config import NON_PAINTABLE_PART_GROUPS
from app.errors import PipelineError
from app.flux import FluxSettings, render_flux
from app.image_ops import (
    build_body_mask,
    clean_mask,
    dark_trim_mask,
    polygons_to_mask,
    recolour,
)
from app.roboflow import segment_boxes
from app.pipeline import _asset_id, _clip_fallback_mask, _refine_side_windows


class PipelineTest(unittest.TestCase):
    def test_car_selection_errors_and_selects_largest(self) -> None:
        with self.assertRaisesRegex(PipelineError, "No car"):
            select_primary_car([], 0.5)

        largest = CarDetection((0, 0, 100, 100), 0.9)
        self.assertIs(
            select_primary_car(
                [CarDetection((0, 0, 20, 20), 0.8), largest], 0.5
            ),
            largest,
        )
        with self.assertRaisesRegex(PipelineError, "Multiple similarly sized"):
            select_primary_car(
                [largest, CarDetection((0, 0, 80, 80), 0.8)], 0.5
            )

    def test_masks_and_preview(self) -> None:
        full = clean_mask(
            polygons_to_mask([[[10, 10], [90, 10], [90, 90], [10, 90]]], (100, 100)),
            (100, 100),
            3,
            2,
        )
        window = np.zeros((100, 100), dtype=np.uint8)
        window[20:40, 30:70] = 255
        body = build_body_mask(full, [window], 3, 2)

        self.assertGreater(body[60, 50], 127)
        self.assertLess(body[30, 50], 128)

        image = Image.new("RGB", (100, 100), (90, 90, 90))
        preview = Image.open(BytesIO(recolour(image, body, "ff0000"))).convert("RGB")
        self.assertEqual(preview.getpixel((0, 0)), (90, 90, 90))
        self.assertNotEqual(preview.getpixel((50, 60)), (90, 90, 90))

        solid = Image.new("RGB", (100, 100), (220, 220, 220))
        solid_blue = Image.open(
            BytesIO(recolour(solid, np.full((100, 100), 255, np.uint8), "2563eb"))
        ).convert("RGB")
        self.assertLessEqual(
            max(
                abs(actual - expected)
                for actual, expected in zip(
                    solid_blue.getpixel((50, 50)), (37, 99, 235)
                )
            ),
            3,
        )

        varied = np.full((100, 100, 3), 100, np.uint8)
        varied[:, 50:] = 240
        varied_blue = Image.open(
            BytesIO(
                recolour(
                    Image.fromarray(varied),
                    np.full((100, 100), 255, np.uint8),
                    "2563eb",
                )
            )
        ).convert("RGB")
        dark_pixel = varied_blue.getpixel((25, 50))
        bright_pixel = varied_blue.getpixel((75, 50))
        self.assertLess(bright_pixel[0], 60)
        self.assertGreater(bright_pixel[2] - bright_pixel[0], 190)
        self.assertNotEqual(dark_pixel, bright_pixel)

        with self.assertRaisesRegex(PipelineError, "dimensions"):
            recolour(image, np.zeros((50, 50), dtype=np.uint8), "ff0000")

    def test_dark_trim_is_limited_to_lower_car(self) -> None:
        pixels = np.full((100, 100, 3), 220, np.uint8)
        pixels[10:20, 20:80] = 10
        pixels[60:80, 20:80] = 10
        mask = dark_trim_mask(
            Image.fromarray(pixels),
            np.full((100, 100), 255, np.uint8),
            (0, 0, 100, 100),
            3,
            0,
        )
        self.assertEqual(mask[15, 50], 0)
        self.assertEqual(mask[70, 50], 255)

    def test_duplicate_part_boxes_are_removed(self) -> None:
        stronger = PartDetection("windows", (0, 0, 100, 100), 0.9)
        weaker = PartDetection("windows", (5, 5, 105, 105), 0.5)
        wheel = PartDetection("wheels", (5, 5, 105, 105), 0.8)
        self.assertEqual(_deduplicate([weaker, wheel, stronger]), [stronger, wheel])

    def test_side_windows_use_upper_deduplicated_doors(self) -> None:
        front = PartDetection("doors", (10, 20, 50, 100), 0.9)
        duplicate = PartDetection("doors", (12, 22, 52, 102), 0.5)
        rear = PartDetection("doors", (60, 20, 100, 100), 0.8)

        prompts = _side_window_prompts([duplicate, rear, front])

        self.assertEqual(
            [prompt.box for prompt in prompts],
            [(18, 26.4, 48, 45.6), (68, 26.4, 98, 45.6)],
        )
        self.assertTrue(
            all(
                prompt.group == "windows"
                and prompt.confidence == 0
                for prompt in prompts
            )
        )
        self.assertEqual(
            [prompt.clip_box for prompt in prompts],
            [(10, 20, 50, 56), (60, 20, 100, 56)],
        )

    def test_side_windows_require_a_door(self) -> None:
        with self.assertRaisesRegex(PipelineError, "doors"):
            _side_window_prompts([])

    def test_fallback_masks_are_clipped_to_their_prompt(self) -> None:
        mask = np.full((100, 100), 255, np.uint8)
        prompt = PartDetection(
            "windows",
            (25, 15, 55, 30),
            0,
            clip_box=(20, 10, 60, 40),
        )

        clipped = _clip_fallback_mask(mask, prompt)

        self.assertEqual(clipped[9, 20], 0)
        self.assertEqual(clipped[10, 20], 255)
        self.assertEqual(clipped[39, 59], 255)
        self.assertEqual(clipped[40, 60], 0)
        self.assertIs(
            _clip_fallback_mask(
                mask,
                PartDetection("windows", (20, 10, 60, 40), 0),
            ),
            mask,
        )

    def test_asset_ids_include_view(self) -> None:
        self.assertEqual(_asset_id(b"image", "left"), _asset_id(b"image", "left"))
        self.assertNotEqual(_asset_id(b"image", "left"), _asset_id(b"image", "right"))

    def test_side_window_trim_is_joined_without_swallowing_mirrors(self) -> None:
        windows = np.zeros((80, 100), np.uint8)
        windows[20:60, 10:45] = 255
        windows[20:60, 50:90] = 255
        mirrors = np.zeros_like(windows)
        mirrors[30:40, 10:20] = 255

        refined = _refine_side_windows(windows, mirrors, 100)

        self.assertEqual(refined[40, 47], 255)
        self.assertEqual(refined[35, 15], 0)
        self.assertEqual(refined[19, 30], 255)
        self.assertNotIn("dark_trim", NON_PAINTABLE_PART_GROUPS)

    def test_side_image_rejects_front_view(self) -> None:
        wheels = [
            PartDetection("wheels", (0, 0, 20, 20), 0.8),
            PartDetection("wheels", (80, 0, 100, 20), 0.8),
        ]
        with self.assertRaisesRegex(PipelineError, "side view"):
            validate_view("front", wheels)
        validate_view("right", wheels)

    def test_flux_changes_only_body_mask_and_caches(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "masks").mkdir()
            original = Image.new("RGB", (32, 32), (200, 200, 200))
            original.save(directory / "original.webp", "WEBP", lossless=True)
            mask = Image.new("L", original.size, 0)
            mask.paste(255, (8, 8, 24, 24))
            mask.save(directory / "masks/paintable-body.png")
            generated = Image.new("RGB", (16, 16), (10, 20, 200))
            generated_path = directory / "generated.png"
            generated.save(generated_path)

            client = Mock()
            client.predict.return_value = (str(generated_path), 0)
            metadata = SimpleNamespace(
                original_image="original.webp",
                masks={"paintable_body": "masks/paintable-body.png"},
            )
            settings = FluxSettings(
                token="test-token",
                space="black-forest-labs/FLUX.1-Kontext-Dev",
                guidance_scale=2.5,
                steps=28,
                seed=0,
                timeout=300,
            )

            with patch("app.flux.Client", return_value=client):
                output, cached = render_flux(directory, metadata, "2563eb", settings)
                cached_output, cached_again = render_flux(
                    directory, metadata, "#2563EB", settings
                )

            result = Image.open(output).convert("RGB")
            self.assertFalse(cached)
            self.assertTrue(cached_again)
            self.assertEqual(output, cached_output)
            self.assertEqual(result.getpixel((0, 0)), (200, 200, 200))
            self.assertLessEqual(
                max(
                    abs(actual - expected)
                    for actual, expected in zip(
                        result.getpixel((16, 16)), (37, 99, 235)
                    )
                ),
                2,
            )
            client.predict.assert_called_once()
            self.assertFalse(client.predict.call_args.kwargs["randomize_seed"])
            self.assertIn("input_image", client.predict.call_args.kwargs)
            self.assertNotIn("edit_images", client.predict.call_args.kwargs)

    def test_sam_timeout_is_reported_separately(self) -> None:
        settings = SimpleNamespace(
            roboflow_api_url="https://serverless.roboflow.com",
            roboflow_api_key="test-key",
            roboflow_sam2_version_id="hiera_small",
            roboflow_timeout=180,
        )
        with patch("app.roboflow.requests.post", side_effect=Timeout):
            with self.assertRaises(PipelineError) as raised:
                segment_boxes(b"image", [(0, 0, 10, 10)], settings)
        self.assertEqual(raised.exception.code, "sam_api_timeout")


if __name__ == "__main__":
    unittest.main()
