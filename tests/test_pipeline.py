import unittest
from dataclasses import dataclass
from inspect import signature
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock, patch

import cv2
import numpy as np
from PIL import Image
from requests import Timeout

from app.main import customise, upload_car
from app.modifications.instructions import RestrictedInstructionParser
from app.modifications.schemas import (
    RacingStripeElement,
    RendererMode,
    StripeAlignment,
    StripePlacement,
    StripeWidth,
    SurfaceEditRequest,
)
from app.renderers.base import RenderResult, request_hash
from app.renderers.generative import GenerativeSurfaceRenderer
from app.detection import (
    CarDetection,
    PartDetection,
    _deduplicate,
    _model_classes,
    _part_group,
    _side_window_prompts,
    _with_side_window_prompts,
    infer_view,
    select_primary_car,
    validate_view,
)
from app.config import (
    EXPECTED_PROTECTIVE_PART_GROUPS_BY_VIEW,
    NON_PAINTABLE_PART_GROUPS,
    REQUIRED_PART_GROUPS_BY_VIEW,
)
from app.errors import PipelineError
from app.flux import FluxSettings, render_flux
from app.flux import composite_design, composite_plain_colour
from app.image_ops import (
    build_paintability_masks,
    build_body_mask,
    clean_mask,
    dark_trim_mask,
    polygons_to_mask,
    recolour,
)
from app.roboflow import segment_boxes, segment_concepts
from app.pipeline import (
    _asset_id,
    _clip_fallback_mask,
    _hybrid_semantic_groups,
    _refine_side_windows,
)
from app.schemas import AssetBundle, BoundingBox, ViewSelection


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

        sharp = np.zeros((100, 100), np.uint8)
        sharp[20:80, 20:80] = 255
        antialiased = Image.open(
            BytesIO(recolour(image, sharp, "ff0000"))
        ).convert("RGB")
        self.assertEqual(antialiased.getpixel((19, 50)), (90, 90, 90))
        self.assertNotEqual(
            antialiased.getpixel((20, 50)), antialiased.getpixel((50, 50))
        )

        solid = Image.new("RGB", (100, 100), (220, 220, 220))
        solid_blue = Image.open(
            BytesIO(recolour(solid, np.full((100, 100), 255, np.uint8), "2563eb"))
        ).convert("RGB")
        solid_blue_pixel = solid_blue.getpixel((50, 50))
        self.assertGreater(solid_blue_pixel[2], solid_blue_pixel[0])
        self.assertGreater(solid_blue_pixel[2], solid_blue_pixel[1])

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
        self.assertGreater(bright_pixel[2], bright_pixel[0])
        self.assertNotEqual(dark_pixel, bright_pixel)

        with self.assertRaisesRegex(PipelineError, "dimensions"):
            recolour(image, np.zeros((50, 50), dtype=np.uint8), "ff0000")

    def test_recolour_preserves_luminance_edges_and_neutral_glare(self) -> None:
        lab = np.empty((80, 120, 3), np.float32)
        lab[:, :, 0] = np.linspace(25, 80, 120)
        lab[:, :, 1:] = (48, 30)
        lab[:, 58:62, 0] -= 14
        lab[20:35, 75:95] = (96, 0, 0)
        source = np.rint(
            np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1) * 255
        ).astype(np.uint8)
        result = np.asarray(
            Image.open(
                BytesIO(
                    recolour(
                        Image.fromarray(source),
                        np.full((80, 120), 255, np.uint8),
                        "183a63",
                    )
                )
            ).convert("RGB")
        )
        source_lab = cv2.cvtColor(source.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
        result_lab = cv2.cvtColor(result.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
        self.assertLess(float(np.mean(np.abs(source_lab[:, :, 0] - result_lab[:, :, 0]))), 1)
        source_gradient = cv2.Sobel(source_lab[:, :, 0], cv2.CV_32F, 1, 0).ravel()
        result_gradient = cv2.Sobel(result_lab[:, :, 0], cv2.CV_32F, 1, 0).ravel()
        self.assertGreater(float(np.corrcoef(source_gradient, result_gradient)[0, 1]), 0.98)
        highlight_chroma = np.linalg.norm(result_lab[20:35, 75:95, 1:], axis=2)
        midtone_chroma = np.linalg.norm(result_lab[40:55, 35:50, 1:], axis=2)
        target_lab = cv2.cvtColor(
            np.asarray([[[24, 58, 99]]], np.float32) / 255,
            cv2.COLOR_RGB2LAB,
        )[0, 0]
        self.assertLess(float(np.median(highlight_chroma)), float(np.median(midtone_chroma)))
        self.assertLess(
            float(
                np.linalg.norm(
                    np.median(result_lab[40:55, 35:50, 1:], axis=(0, 1))
                    - target_lab[1:]
                )
            ),
            3,
        )

    def test_recolour_preserves_reflection_residual_without_source_paint(self) -> None:
        lab = np.empty((50, 80, 3), np.float32)
        lab[:] = (50, 48, 30)
        lab[15:35, 45:65, 1:] = (30, 5)
        source = np.rint(
            np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1) * 255
        ).astype(np.uint8)
        result = np.asarray(
            Image.open(
                BytesIO(
                    recolour(
                        Image.fromarray(source),
                        np.full((50, 80), 255, np.uint8),
                        "183a63",
                    )
                )
            ).convert("RGB")
        )
        result_lab = cv2.cvtColor(result.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
        base = np.median(result_lab[5:15, 5:25, 1:], axis=(0, 1))
        reflected = np.median(result_lab[18:32, 48:62, 1:], axis=(0, 1))
        self.assertLess(float(base[1]), 0)
        self.assertGreater(float(np.linalg.norm(reflected - base)), 3)

    def test_recolour_keeps_target_chroma_in_desaturated_shadows(self) -> None:
        lab = np.empty((50, 80, 3), np.float32)
        lab[:] = (45, 3, -28)
        lab[:, 40:] = (24, 0, -3)
        source = np.rint(
            np.clip(cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), 0, 1) * 255
        ).astype(np.uint8)
        result = np.asarray(
            Image.open(
                BytesIO(
                    recolour(
                        Image.fromarray(source),
                        np.full((50, 80), 255, np.uint8),
                        "634718",
                    )
                )
            ).convert("RGB")
        )
        result_lab = cv2.cvtColor(result.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
        self.assertGreater(float(np.median(result_lab[:, 50:, 2])), 8)

    def test_metallic_recolour_does_not_invent_periodic_shimmer(self) -> None:
        source = Image.new("RGB", (70, 70), (150, 40, 30))
        result = np.asarray(
            Image.open(
                BytesIO(
                    recolour(
                        source,
                        np.full((70, 70), 255, np.uint8),
                        "183a63",
                        "metallic",
                    )
                )
            ).convert("RGB")
        )
        self.assertEqual(int(np.max(np.ptp(result[5:-5, 5:-5], axis=(0, 1)))), 0)

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

    def test_yolo_world_uses_one_deduplicated_class_set(self) -> None:
        classes = _model_classes("car")
        self.assertEqual(classes[0], "car")
        self.assertEqual(len(classes), len(set(classes)))
        self.assertIn("license plate", classes)

    def test_side_aware_replacement_model_labels_are_mapped(self) -> None:
        self.assertEqual(_part_group("left_front-wheel"), "wheels")
        self.assertEqual(_part_group("right_back-window"), "windows")
        self.assertEqual(_part_group("left_tail-light"), "lights")
        self.assertEqual(_part_group("right_mirror"), "mirrors")
        self.assertEqual(_part_group("windshield", "right"), "windows")
        self.assertEqual(_part_group("windshield", "front"), "windows")

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

    def test_partial_side_windows_still_prompt_every_door(self) -> None:
        detected = PartDetection(
            "windows", (18, 26, 48, 46), 0.9, ((18, 26), (48, 26), (48, 46))
        )
        doors = [
            PartDetection("doors", (10, 20, 50, 100), 0.9),
            PartDetection("doors", (60, 20, 100, 100), 0.8),
        ]

        completed = _with_side_window_prompts([detected], doors)

        self.assertIs(completed[0], detected)
        self.assertEqual(
            [part.clip_box for part in completed[1:]],
            [(10, 20, 50, 56), (60, 20, 100, 56)],
        )

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
        sparse = np.zeros_like(mask)
        sparse[15, 25] = 255
        recovered = _clip_fallback_mask(sparse, prompt)
        self.assertEqual(recovered[20, 30], 255)
        self.assertEqual(recovered[9, 20], 0)
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
        self.assertNotEqual(_asset_id(b"image", "auto"), _asset_id(b"image", "front"))

    def test_automatic_view_scoring(self) -> None:
        cases: list[tuple[list[tuple[str, float]], ViewSelection]] = [
            ([('grille', 0.7), ('windshield', 0.7)], "front"),
            ([('trunk', 0.7), ('back-windshield', 0.7)], "rear"),
            ([('left_front-door', 0.7), ('left_front-window', 0.7)], "left"),
            ([('right_front-door', 0.7), ('right_front-window', 0.7)], "right"),
            (
                [
                    ('grille', 0.8),
                    ('windshield', 0.8),
                    ('left_front-door', 0.8),
                    ('left_front-window', 0.8),
                ],
                "front",
            ),
            (
                [
                    ('trunk', 0.8),
                    ('back-windshield', 0.8),
                    ('right_back-door', 0.8),
                    ('right_back-window', 0.8),
                ],
                "rear",
            ),
        ]
        for evidence, expected in cases:
            with self.subTest(expected=expected):
                detected, confidence = infer_view(evidence)
                self.assertEqual(detected, expected)
                self.assertGreaterEqual(confidence, 0.65)

        for evidence in (
            [('grille', 0.7), ('trunk', 0.7)],
            [('grille', 0.5)],
        ):
            with self.assertRaisesRegex(PipelineError, "choose front") as raised:
                infer_view(evidence)
            self.assertEqual(raised.exception.code, "ambiguous_view")

    def test_side_window_pillars_stay_paintable(self) -> None:
        windows = np.zeros((80, 100), np.uint8)
        windows[20:60, 10:45] = 255
        windows[20:60, 50:90] = 255
        mirrors = np.zeros_like(windows)
        mirrors[30:40, 10:20] = 255

        refined = _refine_side_windows(windows, mirrors)

        self.assertEqual(refined[40, 47], 0)
        self.assertEqual(refined[35, 15], 0)
        self.assertEqual(refined[19, 30], 0)
        self.assertEqual(refined[20, 30], 255)
        self.assertNotIn("dark_trim", NON_PAINTABLE_PART_GROUPS)

    def test_side_image_rejects_front_view(self) -> None:
        wheels = [
            PartDetection("wheels", (0, 0, 20, 20), 0.8),
            PartDetection("wheels", (80, 0, 100, 20), 0.8),
        ]
        with self.assertRaisesRegex(PipelineError, "side view"):
            validate_view("front", wheels)
        validate_view("right", wheels)

    def test_plate_is_recovered_when_present_but_is_not_required(self) -> None:
        self.assertEqual(REQUIRED_PART_GROUPS_BY_VIEW["front"], {"windows"})
        self.assertEqual(REQUIRED_PART_GROUPS_BY_VIEW["rear"], {"windows"})
        self.assertEqual(
            _hybrid_semantic_groups("front")["plate"], "license plate"
        )
        self.assertEqual(
            _hybrid_semantic_groups("right")["lights"], "car light"
        )
        self.assertEqual(
            _hybrid_semantic_groups("right")["mirrors"], "car side mirror"
        )
        self.assertNotIn("plate", _hybrid_semantic_groups("right"))
        self.assertNotIn(
            "plate", EXPECTED_PROTECTIVE_PART_GROUPS_BY_VIEW["front"]
        )

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

    def test_sam3_uses_concepts_and_returns_polygon_masks(self) -> None:
        settings = SimpleNamespace(
            roboflow_api_url="https://serverless.roboflow.com",
            roboflow_api_key="test-key",
            roboflow_segmenter="sam3",
            roboflow_sam3_model_id="sam3/sam3_final",
            roboflow_timeout=180,
        )
        response = Mock(ok=True)
        response.json.return_value = {
            "prompt_results": [
                {
                    "predictions": [
                        {
                            "format": "polygon",
                            "masks": [[[0, 0], [10, 0], [10, 10]]],
                        }
                    ]
                }
            ]
        }
        with patch("app.roboflow.requests.post", return_value=response) as post:
            masks = segment_boxes(
                b"image", [(0, 0, 10, 10)], settings, ["car"]
            )

        self.assertEqual(masks, [[[[0, 0], [10, 0], [10, 10]]]])
        self.assertTrue(post.call_args.args[0].endswith("/sam3/concept_segment"))
        self.assertEqual(
            post.call_args.kwargs["json"]["prompts"],
            [{"type": "text", "text": "car"}],
        )

    def test_sam3_semantic_prompts_allow_missing_optional_parts(self) -> None:
        settings = SimpleNamespace(
            roboflow_api_url="https://serverless.roboflow.com",
            roboflow_api_key="test-key",
            roboflow_sam3_model_id="sam3/sam3_final",
            roboflow_timeout=180,
        )
        response = Mock(ok=True)
        response.json.return_value = {
            "prompt_results": [
                {"predictions": []},
                {
                    "predictions": [
                        {
                            "format": "polygon",
                            "masks": [[[1, 1], [3, 1], [3, 3]]],
                        }
                    ]
                },
            ]
        }
        with patch("app.roboflow.requests.post", return_value=response):
            masks = segment_concepts(
                b"image", ["car door handle", "car window pillar"], settings
            )

        self.assertEqual(masks[0], [])
        self.assertEqual(masks[1], [[[1, 1], [3, 1], [3, 3]]])

    def test_surface_edit_schema_validation(self) -> None:
        request = SurfaceEditRequest(
            body_colour="#183A63",
            finish="metallic",
            design_elements=[
                {
                    "type": "racing_stripes",
                    "count": 2,
                    "colour": "#D61F2C",
                    "width": "thin",
                    "placement": "bonnet_and_visible_roof",
                    "alignment": "centre",
                }
            ],
        )
        self.assertEqual(request.body_colour, "#183a63")
        self.assertEqual(request.design_elements[0].colour, "#d61f2c")
        with self.assertRaises(ValueError):
            SurfaceEditRequest(body_colour="red")
        with self.assertRaises(ValueError):
            RacingStripeElement(
                count=3,
                colour="#d61f2c",
                width=StripeWidth.THIN,
                placement=StripePlacement.BONNET,
                alignment=StripeAlignment.CENTRE,
            )
        with self.assertRaises(ValueError):
            SurfaceEditRequest()
        with self.assertRaises(ValueError):
            SurfaceEditRequest(body_colour="#123456", spoiler="large")

    def test_instruction_parser_rejects_physical_changes(self) -> None:
        parser = RestrictedInstructionParser()
        parsed = parser.parse("Use metallic blue with one white stripe.")
        self.assertEqual(parsed.body_colour, "#183a63")
        self.assertEqual(parsed.finish.value, "metallic")
        self.assertEqual(parsed.design_elements[0].count, 1)
        with self.assertRaises(PipelineError) as bumper:
            parser.parse("Replace the bumper.")
        self.assertEqual(bumper.exception.code, "future_physical_modification")
        with self.assertRaises(PipelineError) as rim:
            parser.parse("Change the rims.")
        self.assertEqual(rim.exception.code, "future_physical_modification")

    def test_paintability_masks_protected_precedence_and_uncertain(self) -> None:
        full = np.full((20, 20), 255, np.uint8)
        protected = np.zeros_like(full)
        protected[5:10, 5:10] = 255
        uncertain = np.zeros_like(full)
        uncertain[8:14, 8:14] = 255
        masks = build_paintability_masks(full, [protected], [uncertain], 1, 0)

        self.assertEqual(masks.protected[8, 8], 255)
        self.assertEqual(masks.uncertain[8, 8], 0)
        self.assertEqual(masks.editable[12, 12], 0)
        self.assertEqual(masks.editable[2, 2], 255)

    def test_customisation_cache_keys(self) -> None:
        plain = SurfaceEditRequest(body_colour="#2563EB")
        equivalent = SurfaceEditRequest(body_colour="2563eb", renderer=RendererMode.AUTO)
        stripe = SurfaceEditRequest(
            body_colour="#2563eb",
            design_elements=[
                RacingStripeElement(
                    count=2,
                    colour="#ffffff",
                    width=StripeWidth.THIN,
                    placement=StripePlacement.BONNET,
                    alignment=StripeAlignment.CENTRE,
                )
            ],
        )
        other_stripe = stripe.model_copy(
            update={
                "design_elements": [
                    RacingStripeElement(
                        count=1,
                        colour="#ffffff",
                        width=StripeWidth.THIN,
                        placement=StripePlacement.BONNET,
                        alignment=StripeAlignment.CENTRE,
                    )
                ]
            }
        )
        self.assertEqual(
            request_hash(plain, renderer="deterministic"),
            request_hash(equivalent, renderer="deterministic"),
        )
        self.assertNotEqual(
            request_hash(plain, renderer="deterministic"),
            request_hash(stripe, renderer="generative"),
        )
        self.assertNotEqual(
            request_hash(stripe, renderer="generative"),
            request_hash(other_stripe, renderer="generative"),
        )

    def test_composites_preserve_colour_and_outside_pixels(self) -> None:
        original = Image.new("RGB", (10, 10), (200, 200, 200))
        mask = Image.new("L", (10, 10), 0)
        mask.paste(255, (2, 2, 8, 8))
        generated = Image.new("RGB", (10, 10), (10, 20, 200))
        plain = composite_plain_colour(original, generated, mask, (37, 99, 235))
        self.assertEqual(plain.getpixel((0, 0)), (200, 200, 200))
        self.assertLessEqual(
            max(abs(a - b) for a, b in zip(plain.getpixel((5, 5)), (37, 99, 235))),
            2,
        )

        pixels = np.zeros((10, 10, 3), np.uint8)
        pixels[:, :5] = (210, 20, 20)
        pixels[:, 5:] = (20, 20, 210)
        design = composite_design(original, Image.fromarray(pixels), mask)
        self.assertEqual(design.getpixel((0, 0)), (200, 200, 200))
        self.assertNotEqual(design.getpixel((4, 5)), design.getpixel((6, 5)))

    def test_legacy_metadata_loads(self) -> None:
        bundle = AssetBundle(
            asset_id="a" * 64,
            view="front",
            width=10,
            height=10,
            car_bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10, confidence=1),
            source_image="source.jpg",
            original_image="original.webp",
            luminance_map="luminance-map.png",
            masks={"paintable_body": "masks/paintable-body.png"},
            models={},
        )
        self.assertEqual(bundle.pipeline_version, "legacy")
        self.assertIsNone(bundle.requested_view)
        self.assertIsNone(bundle.view_confidence)
        self.assertTrue(bundle.available_modifications.body_colour)

    def test_upload_api_keeps_front_default_and_accepts_auto(self) -> None:
        self.assertEqual(signature(upload_car).parameters["view"].default, "front")
        self.assertIn("auto", ViewSelection.__args__)

    def test_customise_endpoint_headers(self) -> None:
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
                masks={"paintable_body": "masks/paintable-body.png"},
                models={},
            )
            (directory / "metadata.json").write_text(
                metadata.model_dump_json(), encoding="utf-8"
            )

            class FakeRenderer:
                def render(self, **kwargs) -> RenderResult:
                    return RenderResult(result, False, "deterministic", "passed", [])

            class FakeRequest:
                async def json(self) -> dict:
                    return {"type": "surface_edit", "body_colour": "#123456"}

            with patch.dict("os.environ", {"STORAGE_ROOT": temporary}), patch(
                "app.main.choose_renderer", return_value=FakeRenderer()
            ):
                import asyncio

                response = asyncio.run(
                    customise(asset_id, FakeRequest())
                )
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["x-render-cached"], "false")
            self.assertEqual(response.headers["x-renderer-used"], "deterministic")
            self.assertEqual(response.headers["x-quality-status"], "passed")

    def test_provider_failures_have_stable_error_codes(self) -> None:
        @dataclass
        class FailingProvider:
            name: str = "failing"
            calls: int = 0

            def edit(self, **kwargs) -> Image.Image:
                self.calls += 1
                raise PipelineError("flux_unavailable", "No quota", 503)

        provider = FailingProvider()
        renderer = GenerativeSurfaceRenderer(
            settings=FluxSettings("token", "space", 2.5, 28, 0, 300),
            provider=provider,
        )
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "masks").mkdir()
            Image.new("RGB", (4, 4), (1, 2, 3)).save(
                directory / "original.webp", "WEBP", lossless=True
            )
            cv2.imwrite(str(directory / "masks/paintable-body.png"), np.full((4, 4), 255, np.uint8))
            metadata = AssetBundle(
                asset_id="a" * 64,
                view="front",
                width=4,
                height=4,
                car_bbox=BoundingBox(x1=0, y1=0, x2=4, y2=4, confidence=1),
                source_image="source.jpg",
                original_image="original.webp",
                luminance_map="luminance-map.png",
                masks={"paintable_body": "masks/paintable-body.png"},
                models={},
            )
            with self.assertRaises(PipelineError) as raised:
                renderer.render(
                    directory=directory,
                    metadata=metadata,
                    modification=SurfaceEditRequest(
                        body_colour="#123456",
                        custom_instruction="Make the car matte black.",
                    ),
                )
        self.assertEqual(raised.exception.code, "flux_unavailable")
        self.assertEqual(provider.calls, 2)


if __name__ == "__main__":
    unittest.main()
