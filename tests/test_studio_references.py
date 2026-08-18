import unittest
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from app.errors import PipelineError
from app.generative.vertex_ai import VertexAIBumperProvider
from app.main import analyse_studio_identity, get_studio_reference
from app.modifications.schemas import StudioRenderRequest, VehicleIdentity
from app.schemas import AssetBundle, BoundingBox
from app.studio_references import StudioReferenceMetadata, load_studio_reference, store_studio_reference


def png(colour=(80, 100, 120)) -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 24), colour).save(output, "PNG")
    return output.getvalue()


def identity(confidence=0.9) -> VehicleIdentity:
    return VehicleIdentity(
        make="Honda",
        model="Civic",
        generation="10th generation, 2016-2021",
        body_style="sedan",
        trim="EX",
        visual_cues=["C-shaped tail lamps"],
        confidence=confidence,
    )


class StudioReferenceTest(unittest.TestCase):
    def test_request_limits_reference_ids(self):
        with self.assertRaises(ValueError):
            StudioRenderRequest(reference_asset_ids=["a" * 64, "a" * 64])
        with self.assertRaises(ValueError):
            StudioRenderRequest(reference_asset_ids=[character * 64 for character in "abcde"])

    def test_reference_storage_is_content_addressed(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = store_studio_reference(directory=root, source=png(), kind="user", title="Side")
            second = store_studio_reference(directory=root, source=png(), kind="user", title="Side")
            path, loaded = load_studio_reference(root, first.reference_asset_id)
            self.assertEqual(first.reference_asset_id, second.reference_asset_id)
            self.assertTrue(path.is_file())
            self.assertEqual(loaded.kind, "user")
            with self.assertRaises(PipelineError):
                store_studio_reference(directory=root, source=b"not an image", kind="user")

    def test_reference_metadata_ignores_retired_web_fields(self):
        loaded = StudioReferenceMetadata.model_validate(
            {
                "reference_asset_id": "a" * 64,
                "kind": "user",
                "title": "Front view",
                "source_page_url": "",
                "content_sha256": "b" * 64,
                "width": 32,
                "height": 24,
            }
        )
        self.assertEqual(loaded.title, "Front view")

class VertexIdentityTest(unittest.TestCase):
    @staticmethod
    def provider(response=None, error=None):
        class Part:
            @staticmethod
            def from_bytes(**kwargs):
                return kwargs

        class Models:
            def generate_content(self, **kwargs):
                if error:
                    raise error
                return response

        provider = VertexAIBumperProvider.__new__(VertexAIBumperProvider)
        provider.model_id = "test-model"
        provider._types = SimpleNamespace(Part=Part, GenerateContentConfig=lambda **kwargs: kwargs)
        provider._client = SimpleNamespace(models=Models())
        return provider

    def test_identity_accepts_confident_and_uncertain_structured_results(self):
        for confidence in (0.95, 0.3):
            provider = self.provider(response=SimpleNamespace(parsed=identity(confidence)))
            self.assertEqual(provider.identify_vehicle([Image.new("RGB", (4, 4))]).confidence, confidence)

    def test_identity_rejects_malformed_output_and_wraps_provider_failure(self):
        malformed = self.provider(response=SimpleNamespace(parsed=None, text="not-json"))
        with self.assertRaisesRegex(PipelineError, "invalid vehicle identity"):
            malformed.identify_vehicle([Image.new("RGB", (4, 4))])
        failed = self.provider(error=RuntimeError("offline"))
        with self.assertRaisesRegex(PipelineError, "analysis failed"):
            failed.identify_vehicle([Image.new("RGB", (4, 4))])


class StudioReferenceApiTest(unittest.TestCase):
    def test_identity_and_preview_routes(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.open(BytesIO(png())).save(root / "original.webp")
            metadata = AssetBundle(
                asset_id="a" * 64,
                view="front",
                width=32,
                height=24,
                car_bbox=BoundingBox(x1=0, y1=0, x2=32, y2=24, confidence=1),
                source_image="source.png",
                original_image="original.webp",
                luminance_map="l.png",
                masks={},
                models={},
            )
            stored = store_studio_reference(directory=root, source=png(), kind="user")
            fake_provider = SimpleNamespace(identify_vehicle=lambda images: identity())
            with patch("app.main._asset", return_value=(root, metadata)), patch(
                "app.main.GenerativeSettings.from_env", return_value=SimpleNamespace()
            ), patch("app.main.vertex_provider", return_value=fake_provider):
                analysis = analyse_studio_identity(metadata.asset_id, [])
                preview = get_studio_reference(metadata.asset_id, stored.reference_asset_id)
            self.assertEqual(analysis.identity.model, "Civic")
            self.assertEqual(Path(preview.path).name, "normalized.png")


if __name__ == "__main__":
    unittest.main()
