import hashlib
import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image
from streamlit.testing.v1 import AppTest

from app.schemas import AssetBundle, AvailableModifications, BoundingBox


def _mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    output = np.zeros(shape, np.uint8)
    x1, y1, x2, y2 = box
    output[y1:y2, x1:x2] = 255
    return output


def _rear_asset(root: Path) -> AssetBundle:
    shape = (64, 64)
    masks = {
        "full_car": _mask(shape, (5, 5, 59, 59)),
        "bumper": _mask(shape, (10, 42, 54, 58)),
        "lights": _mask(shape, (10, 42, 18, 48)),
        "plate": _mask(shape, (28, 45, 38, 51)),
        "grille": np.zeros(shape, np.uint8),
        "wheels": np.zeros(shape, np.uint8),
        "windows": _mask(shape, (18, 8, 46, 22)),
        "protected_mask": _mask(shape, (18, 8, 46, 22)),
    }
    paths = {}
    for key, value in masks.items():
        name = f"{key}.png"
        cv2.imwrite(str(root / name), value)
        paths[key] = name
    Image.new("RGB", (64, 64), (90, 90, 90)).save(root / "original.webp")
    return AssetBundle(
        asset_id="a" * 64,
        view="rear",
        requested_view="rear",
        view_confidence=1,
        width=64,
        height=64,
        car_bbox=BoundingBox(x1=5, y1=5, x2=59, y2=59, confidence=1),
        source_image="source.jpg",
        original_image="original.webp",
        luminance_map="l.png",
        masks=paths,
        models={},
        pipeline_version="test",
        available_modifications=AvailableModifications(bumper_replacement=True),
    )


def _side_asset(root: Path) -> AssetBundle:
    shape = (64, 64)
    masks = {
        "full_car": _mask(shape, (5, 5, 59, 55)),
        "wheels": np.zeros(shape, np.uint8),
    }
    masks["wheels"][42:58, 8:24] = 255
    masks["wheels"][42:58, 40:56] = 255
    paths = {}
    for key, value in masks.items():
        name = f"{key}.png"
        cv2.imwrite(str(root / name), value)
        paths[key] = name
    Image.new("RGB", (64, 64), (90, 90, 90)).save(root / "original.webp")
    return AssetBundle(
        asset_id="b" * 64,
        view="left",
        requested_view="left",
        view_confidence=1,
        width=64,
        height=64,
        car_bbox=BoundingBox(x1=5, y1=5, x2=59, y2=55, confidence=1),
        source_image="source.jpg",
        original_image="original.webp",
        luminance_map="l.png",
        masks=paths,
        models={},
        pipeline_version="test",
        available_modifications=AvailableModifications(rim_replacement=True),
    )


def _png(size=(64, 64)) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", size, (120, 120, 120)).save(buffer, "PNG")
    return buffer.getvalue()


class StreamlitTest(unittest.TestCase):
    def test_interface_starts_without_an_upload(self) -> None:
        app = AppTest.from_file(
            str(Path(__file__).parents[1] / "streamlit_app.py")
        ).run(timeout=10)

        self.assertFalse(app.exception)
        self.assertEqual(app.title[0].value, "Car Paint Studio")
        self.assertEqual(len(app.tabs), 2)
        self.assertTrue(app.button[0].disabled)
        self.assertEqual(app.selectbox[0].value, "auto")

    def test_rear_bumper_workflow_shows_reference_preview(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset_dir = root / ("a" * 64)
            asset_dir.mkdir()
            metadata = _rear_asset(asset_dir)
            car = _png()
            app = AppTest.from_file(str(Path(__file__).parents[1] / "streamlit_app.py"))
            app.session_state["processed_source_hash"] = hashlib.sha256(car).hexdigest()
            app.session_state["processed_view_selection"] = "rear"
            app.session_state["processed_metadata"] = metadata
            app.session_state["processed_storage_root"] = root

            app.run(timeout=10)
            app.file_uploader[0].upload("rear-car.png", car, "image/png")
            app.selectbox[0].set_value("rear")
            app.segmented_control[0].set_value("Bumper replacement")
            app.run(timeout=10)
            app.file_uploader[1].upload("rear-bumper.png", _png((32, 16)), "image/png")
            app.run(timeout=10)

            self.assertFalse(app.exception)
            self.assertEqual(app.subheader[0].value, "Rear bumper replacement")
            self.assertEqual(app.file_uploader[1].label, "Reference bumper")
            self.assertGreaterEqual(len(app.image), 2)
            generate = next(button for button in app.button if button.label == "Generate bumper preview")
            self.assertFalse(generate.disabled)

    def test_side_rim_workflow_shows_reference_preview(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset_dir = root / ("b" * 64)
            asset_dir.mkdir()
            metadata = _side_asset(asset_dir)
            car = _png()
            app = AppTest.from_file(str(Path(__file__).parents[1] / "streamlit_app.py"))
            app.session_state["processed_source_hash"] = hashlib.sha256(car).hexdigest()
            app.session_state["processed_view_selection"] = "left"
            app.session_state["processed_metadata"] = metadata
            app.session_state["processed_storage_root"] = root

            app.run(timeout=10)
            app.file_uploader[0].upload("side-car.png", car, "image/png")
            app.selectbox[0].set_value("left")
            app.segmented_control[0].set_value("Rim replacement")
            app.run(timeout=10)
            app.file_uploader[1].upload("rim.png", _png((32, 32)), "image/png")
            app.run(timeout=10)

            self.assertFalse(app.exception)
            self.assertEqual(app.subheader[0].value, "Rim replacement")
            self.assertEqual(app.file_uploader[1].label, "Reference rim")
            self.assertGreaterEqual(len(app.image), 2)
            generate = next(button for button in app.button if button.label == "Generate rim preview")
            self.assertFalse(generate.disabled)

    def test_pending_preview_must_be_kept_before_it_becomes_current(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            asset_dir = root / ("b" * 64)
            asset_dir.mkdir()
            metadata = _side_asset(asset_dir)
            car = _png()
            pending = _png()
            app = AppTest.from_file(str(Path(__file__).parents[1] / "streamlit_app.py"))
            app.session_state["processed_source_hash"] = hashlib.sha256(car).hexdigest()
            app.session_state["processed_view_selection"] = "left"
            app.session_state["processed_metadata"] = metadata
            app.session_state["processed_storage_root"] = root
            app.session_state["composition_history"] = [{
                "image_bytes": car,
                "kind": "Original",
                "quality_status": "passed",
                "warnings": [],
                "cached": False,
            }]
            app.session_state["pending_result_bytes"] = pending
            app.session_state["pending_result_metadata"] = SimpleNamespace(
                quality_status="passed", warnings=[], cached=False
            )
            app.session_state["pending_result_kind"] = "Paint"

            app.run(timeout=10)
            self.assertEqual(len(app.session_state["composition_history"]), 1)
            next(button for button in app.button if button.label == "Keep this change").click()
            app.run(timeout=10)

            self.assertEqual(len(app.session_state["composition_history"]), 2)
            self.assertIsNone(app.session_state["pending_result_bytes"])


if __name__ == "__main__":
    unittest.main()
