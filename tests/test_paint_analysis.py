import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

from app.modifications.schemas import SurfaceEditRequest
from app.paint_analysis.body_colour_estimator import estimate_body_paint
from app.paint_analysis.colour_profile import (
    chroma_distance,
    lab_colour_distance,
    lightness_difference,
    rgb_to_lab,
)
from app.paint_analysis.mask_builder import build_request_masks, load_request_masks
from app.paint_analysis.material_classifier import classify_material
from app.paint_analysis.paint_group_classifier import analyse_paint_groups
from app.paint_analysis.schemas import (
    BodyPaintProfile,
    MaterialType,
    PaintGroup,
    PaintGroupReport,
)
from app.paint_analysis.surface_completion import complete_body_surface
from app.quality.checks import check_paint_analysis
from app.renderers.deterministic import DeterministicSurfaceRenderer
from app.renderers.base import request_hash
from app.pipeline import _asset_id
from app.schemas import AssetBundle, BoundingBox


SETTINGS = SimpleNamespace(
    anchor_erosion_pixels=2,
    anchor_min_sample_count=100,
    body_paint_chroma_threshold=18,
    body_paint_lightness_threshold=35,
    body_paint_strict_chroma_threshold=10,
    paint_group_min_confidence=0.7,
    paint_group_uncertain_threshold=0.45,
    body_seed_min_neighbours=4,
    body_growth_chroma_threshold=48,
    body_growth_local_lab_threshold=20,
    body_growth_max_gradient=55,
    body_growth_min_neighbours=2,
    body_growth_max_iterations=16,
    body_completion_kernel_size=7,
    body_completion_max_hole_area=2000,
    body_region_min_boundary_ratio=0.18,
    body_fragment_min_area=64,
)


def rectangle(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, np.uint8)
    x1, y1, x2, y2 = box
    mask[y1:y2, x1:x2] = 255
    return mask


class PaintAnalysisTest(unittest.TestCase):
    def test_surface_completion_recovers_panel_variation_without_crossing_protection(
        self,
    ) -> None:
        shape = (80, 120)
        lab = np.empty((*shape, 3), np.float32)
        lab[:] = (50, 50, 30)
        lab[25:29, 8:112] = (82, 65, 38)
        lab[52:58, 8:112] = (20, 55, 33)
        lab[:, :4] = (90, 60, 35)
        safe = np.full(shape, 255, np.uint8)
        protected = rectangle(shape, (45, 8, 75, 22))
        contrast = rectangle(shape, (35, 0, 85, 7))
        safe[(protected > 0) | (contrast > 0)] = 0
        profile = BodyPaintProfile(
            median_lab=[50, 50, 30],
            shadow_lab_range={"min": 30, "max": 40},
            highlight_lab_range={"min": 60, "max": 70},
            confidence=0.9,
        )

        result = complete_body_surface(lab, safe, protected, profile, SETTINGS)

        self.assertGreater(np.mean(result.main_body[25:29, 8:112] > 0), 0.9)
        self.assertGreater(np.mean(result.main_body[52:58, 8:112] > 0), 0.9)
        self.assertGreater(np.mean(result.main_body[:, :4] > 0), 0.9)
        self.assertFalse(np.any((result.main_body > 0) & (protected > 0)))
        self.assertFalse(np.any((result.main_body > 0) & (contrast > 0)))
        metrics = result.report.fragmentation
        self.assertGreater(metrics.recovered_pixel_count, 0)
        self.assertLessEqual(metrics.components_after, metrics.components_before)
        self.assertLess(
            metrics.internal_gap_pixels_after,
            metrics.internal_gap_pixels_before,
        )

    def test_surface_completion_preserves_large_contrast_region(self) -> None:
        shape = (80, 120)
        lab = np.empty((*shape, 3), np.float32)
        lab[:] = (50, 50, 30)
        contrast = rectangle(shape, (20, 25, 100, 60))
        lab[contrast > 0] = (50, -30, 35)
        profile = BodyPaintProfile(
            median_lab=[50, 50, 30],
            shadow_lab_range={"min": 30},
            highlight_lab_range={"max": 70},
            confidence=0.9,
        )

        result = complete_body_surface(
            lab,
            np.full(shape, 255, np.uint8),
            np.zeros(shape, np.uint8),
            profile,
            SETTINGS,
        )

        self.assertFalse(np.any((result.main_body > 0) & (contrast > 0)))
        self.assertGreater(result.report.fragmentation.final_pixel_count, 0)

    def test_uniform_panel_produces_reliable_json_safe_profile(self) -> None:
        image = Image.new("RGB", (60, 60), (180, 35, 30))
        profile, anchors = estimate_body_paint(
            image,
            np.full((60, 60), 255, np.uint8),
            erosion_pixels=2,
            min_samples=100,
            chroma_threshold=18,
        )
        self.assertGreater(profile.confidence, 0.9)
        self.assertGreater(profile.sample_count, 100)
        self.assertTrue(np.any(anchors))
        self.assertIsInstance(profile.model_dump(mode="json")["median_lab"], list)

    def test_highlights_and_deep_shadows_are_excluded(self) -> None:
        pixels = np.full((80, 80, 3), (180, 35, 30), np.uint8)
        pixels[:4] = 255
        pixels[-4:] = 0
        _, anchors = estimate_body_paint(
            Image.fromarray(pixels),
            np.full((80, 80), 255, np.uint8),
            erosion_pixels=0,
            min_samples=100,
            chroma_threshold=18,
        )
        self.assertFalse(np.any(anchors[:4]))
        self.assertFalse(np.any(anchors[-4:]))

    def test_same_paint_in_light_and_shadow_keeps_one_profile(self) -> None:
        lab = np.empty((80, 80, 3), np.float32)
        lab[:, :40] = (40, 50, 30)
        lab[:, 40:] = (70, 50, 30)
        pixels = np.rint(
            cv2.cvtColor(lab, cv2.COLOR_LAB2RGB) * 255
        ).clip(0, 255).astype(np.uint8)
        profile, _ = estimate_body_paint(
            Image.fromarray(pixels),
            np.full((80, 80), 255, np.uint8),
            erosion_pixels=0,
            min_samples=100,
            chroma_threshold=18,
        )
        self.assertGreater(profile.confidence, 0.45)
        self.assertGreater(
            profile.highlight_lab_range["max"],
            profile.shadow_lab_range["min"],
        )

    def test_insufficient_and_missing_anchors_warn(self) -> None:
        profile, _ = estimate_body_paint(
            Image.new("RGB", (10, 10), (100, 20, 20)),
            np.full((10, 10), 255, np.uint8),
            erosion_pixels=0,
            min_samples=500,
            chroma_threshold=18,
        )
        self.assertLess(profile.confidence, 0.45)
        self.assertIn("insufficient_body_paint_anchor_samples", profile.warnings)
        missing, _ = estimate_body_paint(
            Image.new("RGB", (10, 10)),
            np.zeros((10, 10), np.uint8),
            erosion_pixels=0,
            min_samples=1,
            chroma_threshold=18,
        )
        self.assertIn("no_reliable_body_paint_anchor", missing.warnings)

    def test_chroma_similarity_survives_lightness_but_rejects_other_colour(self) -> None:
        lab = np.array([[35, 45, 20], [70, 45, 20], [35, -35, 35]], np.float32)
        self.assertEqual(float(chroma_distance(lab[0], lab[1])), 0)
        self.assertEqual(float(lightness_difference(lab[0], lab[1])), 35)
        self.assertGreater(float(chroma_distance(lab[0], lab[2])), 70)
        self.assertGreater(float(lab_colour_distance(lab[0], lab[2])), 70)

    def test_material_semantics_override_colour_and_chrome_reflection(self) -> None:
        red = Image.new("RGB", (20, 20), (180, 35, 30))
        mask = np.full((20, 20), 255, np.uint8)
        material, _, _ = classify_material(red, mask, "windows")
        self.assertEqual(material, MaterialType.GLASS)

        chrome = np.zeros((20, 20, 3), np.uint8)
        chrome[:10] = 250
        chrome[10:] = 40
        material, _, _ = classify_material(Image.fromarray(chrome), mask, "handles")
        self.assertEqual(material, MaterialType.CHROME)

    def test_black_body_is_paint_but_semantic_black_trim_is_protected(self) -> None:
        image = Image.new("RGB", (80, 80), (20, 20, 20))
        full = np.full((80, 80), 255, np.uint8)
        trim = rectangle(full.shape, (5, 55, 75, 70))
        result = analyse_paint_groups(image, full, {"trim": trim}, SETTINGS)
        self.assertTrue(np.any(result.group_masks[PaintGroup.MAIN_BODY_PAINT]))
        protected = (
            PaintGroup.BLACK_PLASTIC_TRIM in result.group_masks
            or PaintGroup.GLOSSY_BLACK_TRIM in result.group_masks
        )
        self.assertTrue(protected)
        self.assertFalse(np.any((result.masks.editable > 0) & (trim > 0)))

    def test_black_trim_prompt_does_not_protect_saturated_body_paint(self) -> None:
        pixels = np.full((60, 100, 3), (180, 35, 30), np.uint8)
        trim = rectangle((60, 100), (10, 40, 90, 55))
        pixels[45:55, 10:90] = 20

        result = analyse_paint_groups(
            Image.fromarray(pixels),
            np.full((60, 100), 255, np.uint8),
            {"trim": trim},
            SETTINGS,
        )

        self.assertEqual(result.masks.editable[42, 50], 255)
        self.assertEqual(result.masks.protected[50, 50], 255)

    def test_part_aware_groups_and_bumper_subregions(self) -> None:
        pixels = np.full((100, 120, 3), (180, 35, 30), np.uint8)
        parts = {
            "windows": rectangle((100, 120), (30, 10, 90, 30)),
            "pillars": rectangle((100, 120), (58, 10, 63, 35)),
            "handles": rectangle((100, 120), (25, 50, 38, 56)),
            "mirrors": rectangle((100, 120), (12, 35, 27, 47)),
            "roof": rectangle((100, 120), (35, 2, 85, 9)),
            "bumper": rectangle((100, 120), (20, 78, 100, 95)),
        }
        pixels[parts["windows"] > 0] = (80, 100, 115)
        pixels[parts["pillars"] > 0] = (10, 10, 10)
        pixels[parts["roof"] > 0] = (30, 50, 180)
        bumper_pixels = np.argwhere(parts["bumper"] > 0)
        lower = bumper_pixels[len(bumper_pixels) // 2 :]
        pixels[lower[:, 0], lower[:, 1]] = (20, 20, 20)
        pixels[80:84, 45:75] = (245, 110, 75)
        result = analyse_paint_groups(
            Image.fromarray(pixels),
            np.full((100, 120), 255, np.uint8),
            parts,
            SETTINGS,
        )
        self.assertIn(PaintGroup.BODY_COLOURED_HANDLE, result.group_masks)
        self.assertIn(PaintGroup.BODY_COLOURED_MIRROR_CAP, result.group_masks)
        self.assertIn(PaintGroup.CONTRAST_ROOF_PAINT, result.group_masks)
        self.assertIn(PaintGroup.GLOSSY_BLACK_TRIM, result.group_masks)
        self.assertIn(PaintGroup.PAINTED_BUMPER_SECTION, result.group_masks)
        self.assertIn(PaintGroup.BLACK_PLASTIC_TRIM, result.group_masks)
        self.assertGreater(
            np.mean(
                result.group_masks[PaintGroup.PAINTED_BUMPER_SECTION][80:84, 45:75]
                > 0
            ),
            0.9,
        )
        self.assertFalse(np.any(result.masks.editable[88:94, 30:90]))

        pixels[parts["mirrors"] > 0] = (25, 45, 180)
        contrasting = analyse_paint_groups(
            Image.fromarray(pixels),
            np.full((100, 120), 255, np.uint8),
            parts,
            SETTINGS,
        )
        self.assertIn(PaintGroup.CONTRASTING_MIRROR_CAP, contrasting.group_masks)

    def test_body_coloured_roof_strip_is_separated_from_rear_glass(self) -> None:
        pixels = np.full((80, 100, 3), (180, 35, 30), np.uint8)
        roof = rectangle((80, 100), (20, 5, 80, 35))
        window = rectangle((80, 100), (25, 20, 75, 35))
        pixels[10:20, 20:80] = (15, 15, 15)
        pixels[window > 0] = (80, 100, 115)

        result = analyse_paint_groups(
            Image.fromarray(pixels),
            np.full((80, 100), 255, np.uint8),
            {"roof": roof, "windows": window},
            SETTINGS,
        )

        self.assertEqual(result.masks.editable[7, 50], 255)
        self.assertEqual(result.masks.protected[25, 50], 255)
        self.assertFalse(
            np.any((result.masks.editable > 0) & (result.masks.protected > 0))
        )

    def test_large_unrelated_painted_body_region_is_secondary(self) -> None:
        pixels = np.full((100, 100, 3), (180, 35, 30), np.uint8)
        pixels[55:90, 15:85] = (25, 60, 190)
        result = analyse_paint_groups(
            Image.fromarray(pixels),
            np.full((100, 100), 255, np.uint8),
            {},
            SETTINGS,
        )
        self.assertIn(PaintGroup.SECONDARY_BODY_PAINT, result.group_masks)
        secondary = result.group_masks[PaintGroup.SECONDARY_BODY_PAINT]
        self.assertFalse(np.any((secondary > 0) & (result.masks.editable > 0)))

    def test_adjacent_disconnected_same_paint_region_joins_main_body(self) -> None:
        pixels = np.full((50, 80, 3), (180, 35, 30), np.uint8)
        pixels[2] = (180, 35, 30)
        pixels[3] = (25, 60, 190)
        settings = SimpleNamespace(**vars(SETTINGS))
        settings.body_completion_kernel_size = 1

        result = analyse_paint_groups(
            Image.fromarray(pixels),
            np.full((50, 80), 255, np.uint8),
            {},
            settings,
        )

        self.assertGreater(
            np.mean(result.group_masks[PaintGroup.MAIN_BODY_PAINT][2] > 0),
            0.9,
        )
        self.assertFalse(np.any(result.paint_like_residual[2]))

    def test_chrome_and_contrasting_handles_are_not_editable(self) -> None:
        pixels = np.full((80, 100, 3), (180, 35, 30), np.uint8)
        handle = rectangle((80, 100), (20, 45, 40, 55))
        pixels[45:50, 20:40] = 250
        pixels[50:55, 20:40] = 40
        result = analyse_paint_groups(
            Image.fromarray(pixels),
            np.full((80, 100), 255, np.uint8),
            {"handles": handle},
            SETTINGS,
        )
        self.assertIn(PaintGroup.CHROME_TRIM, result.group_masks)
        self.assertFalse(np.any((result.masks.editable > 0) & (handle > 0)))

        pixels[handle > 0] = (15, 15, 15)
        result = analyse_paint_groups(
            Image.fromarray(pixels),
            np.full((80, 100), 255, np.uint8),
            {"handles": handle},
            SETTINGS,
        )
        self.assertFalse(np.any((result.masks.editable > 0) & (handle > 0)))

    def test_body_coloured_handles_and_mirror_caps_split_from_dark_bases(self) -> None:
        pixels = np.full((80, 120, 3), (180, 35, 30), np.uint8)
        handle = rectangle((80, 120), (20, 45, 44, 55))
        mirror = rectangle((80, 120), (70, 30, 102, 50))
        pixels[45:55, 40:44] = 15
        pixels[30:50, 96:102] = 15

        result = analyse_paint_groups(
            Image.fromarray(pixels),
            np.full((80, 120), 255, np.uint8),
            {"handles": handle, "mirrors": mirror},
            SETTINGS,
        )

        self.assertEqual(result.masks.editable[48, 25], 255)
        self.assertEqual(result.masks.editable[40, 75], 255)
        self.assertEqual(result.masks.protected[48, 42], 255)
        self.assertEqual(result.masks.protected[40, 99], 255)

        grey = np.full((80, 120, 3), (95, 100, 105), np.uint8)
        grey[30:50, 96:102] = 15
        result = analyse_paint_groups(
            Image.fromarray(grey),
            np.full((80, 120), 255, np.uint8),
            {"mirrors": mirror},
            SETTINGS,
        )
        self.assertEqual(result.masks.editable[40, 75], 255)
        self.assertEqual(result.masks.protected[40, 99], 255)

    def test_mixed_front_region_recovers_paint_without_crossing_light(self) -> None:
        pixels = np.full((50, 80, 3), (180, 35, 30), np.uint8)
        pixels[3] = (25, 60, 190)
        light = rectangle((50, 80), (30, 0, 42, 5))
        pixels[light > 0] = (245, 245, 220)
        settings = SimpleNamespace(**vars(SETTINGS))
        settings.body_completion_kernel_size = 1

        result = analyse_paint_groups(
            Image.fromarray(pixels),
            np.full((50, 80), 255, np.uint8),
            {"lights": light},
            settings,
        )

        self.assertEqual(result.masks.editable[2, 20], 255)
        self.assertEqual(result.masks.protected[2, 35], 255)
        self.assertFalse(
            np.any((result.masks.editable > 0) & (result.masks.protected > 0))
        )

    def test_body_paint_overspill_is_recovered_from_light_and_trim_edges(self) -> None:
        pixels = np.full((70, 100, 3), (180, 35, 30), np.uint8)
        light = rectangle((70, 100), (10, 10, 35, 30))
        trim = rectangle((70, 100), (10, 45, 90, 60))
        pixels[12:28, 12:33] = (240, 235, 210)
        pixels[48:58, 12:88] = 20

        result = analyse_paint_groups(
            Image.fromarray(pixels),
            np.full((70, 100), 255, np.uint8),
            {"lights": light, "trim": trim},
            SETTINGS,
        )

        self.assertEqual(result.masks.editable[10, 20], 255)
        self.assertEqual(result.masks.protected[20, 20], 255)
        self.assertEqual(result.masks.editable[45, 50], 255)
        self.assertEqual(result.masks.protected[52, 50], 255)

        red_lens = rectangle((70, 100), (55, 10, 85, 30))
        result = analyse_paint_groups(
            Image.fromarray(np.full((70, 100, 3), (180, 35, 30), np.uint8)),
            np.full((70, 100), 255, np.uint8),
            {"lights": red_lens},
            SETTINGS,
        )
        self.assertFalse(np.any((result.masks.editable > 0) & (red_lens > 0)))

    def test_dark_bumper_insert_does_not_follow_chromatic_body_colour(self) -> None:
        pixels = np.full((60, 100, 3), (180, 35, 30), np.uint8)
        bumper = rectangle((60, 100), (10, 25, 90, 55))
        pixels[35:50, 20:40] = 25

        result = analyse_paint_groups(
            Image.fromarray(pixels),
            np.full((60, 100), 255, np.uint8),
            {"bumper": bumper},
            SETTINGS,
        )

        self.assertEqual(result.masks.editable[30, 50], 255)
        self.assertEqual(result.masks.protected[40, 30], 255)

    def test_body_coloured_window_edge_is_released_but_glass_stays_protected(self) -> None:
        pixels = np.full((60, 100, 3), (180, 35, 30), np.uint8)
        window = rectangle((60, 100), (20, 10, 80, 35))
        pixels[11:34, 21:79] = (20, 25, 30)

        result = analyse_paint_groups(
            Image.fromarray(pixels),
            np.full((60, 100), 255, np.uint8),
            {"windows": window},
            SETTINGS,
        )

        self.assertEqual(result.masks.editable[10, 50], 255)
        self.assertEqual(result.masks.protected[20, 50], 255)

        pixels[:] = (95, 100, 105)
        pixels[11:34, 21:79] = (20, 25, 30)
        result = analyse_paint_groups(
            Image.fromarray(pixels),
            np.full((60, 100), 255, np.uint8),
            {"windows": window},
            SETTINGS,
        )
        self.assertEqual(result.masks.editable[10, 50], 255)
        self.assertEqual(result.masks.protected[20, 50], 255)

    def test_masks_are_disjoint_and_request_specific(self) -> None:
        shape = (20, 20)
        groups = {
            PaintGroup.MAIN_BODY_PAINT: rectangle(shape, (0, 0, 10, 20)),
            PaintGroup.BODY_COLOURED_HANDLE: rectangle(shape, (10, 0, 12, 5)),
            PaintGroup.CONTRASTING_HANDLE: rectangle(shape, (12, 0, 14, 5)),
            PaintGroup.CONTRAST_ROOF_PAINT: rectangle(shape, (10, 5, 20, 10)),
        }
        fallback = np.zeros(shape, np.uint8)
        body, roof = build_request_masks(groups, fallback)
        self.assertTrue(np.any(body))
        self.assertFalse(np.any(roof))
        self.assertEqual(body[2, 10], 255)
        self.assertEqual(body[2, 12], 0)
        _, roof = build_request_masks(groups, fallback, include_roof=True)
        self.assertEqual(roof[7, 15], 255)

    def test_quality_checks_group_overlap_and_precedence(self) -> None:
        image = Image.new("RGB", (30, 30), (180, 35, 30))
        result = analyse_paint_groups(
            image, np.full((30, 30), 255, np.uint8), {}, SETTINGS
        )
        self.assertEqual(
            check_paint_analysis(
                result.group_masks,
                result.masks,
                seed_mask=result.surface.seeds,
                hard_protected_mask=result.surface.hard_protected,
            ),
            [],
        )
        residual = np.zeros((30, 30), np.uint8)
        residual[5:15, 5:15] = 255
        self.assertIn(
            "paint_like_residual_region_detected",
            check_paint_analysis(
                result.group_masks,
                result.masks,
                paint_like_residual_mask=residual,
                minimum_residual_pixels=64,
            ),
        )

    def test_old_and_new_metadata_round_trip_and_legacy_mask_fallback(self) -> None:
        legacy = AssetBundle(
            asset_id="a" * 64,
            view="front",
            width=10,
            height=10,
            car_bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10, confidence=1),
            source_image="source.jpg",
            original_image="original.webp",
            luminance_map="luminance-map.png",
            masks={"editable_mask": "masks/editable.png"},
            models={},
        )
        self.assertIsNone(legacy.paint_group_report)
        report = PaintGroupReport()
        restored = AssetBundle.model_validate_json(
            legacy.model_copy(
                update={
                    "paint_group_report": report,
                    "paint_analysis_version": "paint-groups-v1",
                }
            ).model_dump_json()
        )
        self.assertEqual(restored.paint_analysis_version, "paint-groups-v1")
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "masks").mkdir()
            cv2.imwrite(
                str(directory / "masks/editable.png"),
                np.full((10, 10), 255, np.uint8),
            )
            body, roof = load_request_masks(directory, legacy)
            self.assertTrue(np.all(body == 255))
            self.assertFalse(np.any(roof))

    def test_pipeline_and_target_groups_participate_in_cache_keys(self) -> None:
        with patch("app.pipeline.PIPELINE_VERSION", b"old"):
            old = _asset_id(b"image", "front")
        with patch("app.pipeline.PIPELINE_VERSION", b"new"):
            new = _asset_id(b"image", "front")
        self.assertNotEqual(old, new)
        body = SurfaceEditRequest(body_colour="#123456")
        dual = SurfaceEditRequest(body_colour="#123456", roof_colour="#654321")
        self.assertNotEqual(
            request_hash(body, renderer="deterministic", pipeline_version="9"),
            request_hash(dual, renderer="deterministic", pipeline_version="9"),
        )
        self.assertNotEqual(
            request_hash(body, renderer="deterministic", pipeline_version="9"),
            request_hash(body, renderer="deterministic", pipeline_version="10"),
        )

    def test_deterministic_render_preserves_roof_until_targeted(self) -> None:
        with TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "masks").mkdir()
            original = Image.new("RGB", (20, 20), (100, 100, 100))
            original.save(directory / "original.webp", "WEBP", lossless=True)
            body = rectangle((20, 20), (0, 10, 20, 20))
            roof = rectangle((20, 20), (5, 2, 15, 7))
            body_handle = rectangle((20, 20), (1, 1, 4, 4))
            contrasting_handle = rectangle((20, 20), (16, 1, 19, 4))
            uncertain = rectangle((20, 20), (1, 5, 4, 8))
            protected = contrasting_handle.copy()
            for name, mask in {
                "body.png": body,
                "roof.png": roof,
                "body-handle.png": body_handle,
                "contrasting-handle.png": contrasting_handle,
                "uncertain.png": uncertain,
                "protected.png": protected,
                "editable.png": body,
            }.items():
                cv2.imwrite(str(directory / "masks" / name), mask)
            metadata = AssetBundle(
                asset_id="b" * 64,
                view="front",
                width=20,
                height=20,
                car_bbox=BoundingBox(x1=0, y1=0, x2=20, y2=20, confidence=1),
                source_image="source.jpg",
                original_image="original.webp",
                luminance_map="luminance-map.png",
                masks={
                    "editable_mask": "masks/editable.png",
                    "protected_mask": "masks/protected.png",
                    PaintGroup.MAIN_BODY_PAINT.value: "masks/body.png",
                    PaintGroup.CONTRAST_ROOF_PAINT.value: "masks/roof.png",
                    PaintGroup.BODY_COLOURED_HANDLE.value: "masks/body-handle.png",
                    PaintGroup.CONTRASTING_HANDLE.value: "masks/contrasting-handle.png",
                    PaintGroup.UNKNOWN.value: "masks/uncertain.png",
                },
                models={},
                pipeline_version="9",
            )
            renderer = DeterministicSurfaceRenderer()
            body_result = Image.open(
                renderer.render(
                    directory=directory,
                    metadata=metadata,
                    modification=SurfaceEditRequest(body_colour="#ff0000"),
                ).path
            ).convert("RGB")
            self.assertEqual(body_result.getpixel((10, 4)), (100, 100, 100))
            self.assertNotEqual(body_result.getpixel((2, 2)), (100, 100, 100))
            self.assertEqual(body_result.getpixel((17, 2)), (100, 100, 100))
            self.assertEqual(body_result.getpixel((2, 6)), (100, 100, 100))
            dual_result = Image.open(
                renderer.render(
                    directory=directory,
                    metadata=metadata,
                    modification=SurfaceEditRequest(
                        body_colour="#ff0000", roof_colour="#0000ff"
                    ),
                ).path
            ).convert("RGB")
            self.assertNotEqual(dual_result.getpixel((10, 4)), (100, 100, 100))
            self.assertEqual(dual_result.size, original.size)


if __name__ == "__main__":
    unittest.main()
