import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from click.testing import CliRunner

from tools.generate_web_data import generate_info_json, main


class WebDataTests(unittest.TestCase):
    def setUp(self):
        tmp = TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.book = Path(tmp.name) / "web" / "resources" / "sample"
        (self.book / "books").mkdir(parents=True)
        self.audio = self.book / "audios"
        self.audio.mkdir()
        self.manifest = self.book / "info.json"

    def test_manifest_uses_web_paths_and_skips_missing_or_empty_audio(self):
        for name in ("3", "2", "1", "0"):
            (self.book / "books" / f"{name}.txt").touch()
        (self.audio / "2.mp3").touch()
        for name in ("0", "3", "orphan"):
            (self.audio / f"{name}.mp3").write_bytes(b"audio fixture")
        generate_info_json(self.book, self.manifest)
        self.assertEqual(json.loads(self.manifest.read_text())["audios"], [
            {"title": n, "src": f"resources/sample/audios/{n}.mp3"} for n in ("0", "3")
        ])

    def test_special_characters_round_trip_in_browser_urls(self):
        from urllib.parse import unquote, urlsplit
        chapter = "测试#50%?"
        (self.book / "books" / f"{chapter}.txt").touch()
        (self.audio / f"{chapter}.mp3").write_bytes(b"audio")
        generate_info_json(self.book, self.manifest)
        track, = json.loads(self.manifest.read_text())["audios"]
        url = urlsplit(track["src"])
        self.assertEqual(track["title"], chapter)
        self.assertFalse(url.query or url.fragment)
        self.assertEqual(unquote(url.path), f"resources/sample/audios/{chapter}.mp3")

    def test_missing_references_fail_before_model_load(self):
        with patch.dict(sys.modules, {"tools.speech_inference": None}):
            result = CliRunner().invoke(main, ["--resources_dir", str(self.book)])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("Missing reference", result.output)
        self.assertFalse(self.manifest.exists())

    def test_partial_batch_preserves_previous_audio_and_returns_failure(self):
        for name in ("1", "2", "3"):
            (self.book / "books" / f"{name}.txt").write_text(name)
        (self.book / "reference.wav").touch()
        (self.book / "reference.txt").write_text("reference")
        (self.audio / "2.mp3").write_bytes(b"previous usable audio")
        inference = Mock()
        inference.SpeechInference.return_value.generate.side_effect = lambda text, **kw: (16000, text)

        def encode(audio, rate, path):
            Path(path).write_bytes(audio.encode())
            if audio != "1":
                raise OSError("interrupted encode")

        inference.bytes_to_audio.side_effect = encode
        with patch.dict(sys.modules, {"tools.speech_inference": inference}):
            result = CliRunner().invoke(main, ["--resources_dir", str(self.book)])
            self.assertEqual(result.exit_code, 1)
            self.assertIn("2 of 3 chapters failed", result.output)
            self.assertEqual((self.audio / "1.mp3").read_bytes(), b"1")
            self.assertEqual((self.audio / "2.mp3").read_bytes(), b"previous usable audio")
            self.assertFalse((self.audio / "3.mp3").exists())
            self.assertEqual([t["title"] for t in json.loads(self.manifest.read_text())["audios"]], ["1", "2"])

            inference.bytes_to_audio.side_effect = lambda audio, rate, path: Path(path).write_bytes(audio.encode())
            result = CliRunner().invoke(main, ["--resources_dir", str(self.book)])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual([t["title"] for t in json.loads(self.manifest.read_text())["audios"]], ["1", "2", "3"])


if __name__ == "__main__":
    unittest.main()
