import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


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


if __name__ == "__main__":
    unittest.main()
