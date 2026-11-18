import hashlib
import io
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory
import unittest

from scripts.install_weights import REQUIRED_FILES, install_archive


class WeightInstallTests(unittest.TestCase):
    def setUp(self):
        folder = TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        root = Path(folder.name)
        self.archive = root / "weights.tar.gz"
        self.destination = root / "weights"
        with tarfile.open(self.archive, "w:gz") as archive:
            for name in sorted(REQUIRED_FILES):
                data = b"test checkpoint data"
                member = tarfile.TarInfo(f"weights/{name}")
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))
        self.size = self.archive.stat().st_size
        self.digest = hashlib.sha256(self.archive.read_bytes()).hexdigest()

    def test_only_verified_archive_installs(self):
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            install_archive(self.archive, self.destination, expected_size=self.size, expected_sha256="0" * 64)
        self.assertFalse(self.destination.exists())
        install_archive(self.archive, self.destination, expected_size=self.size, expected_sha256=self.digest)
        self.assertEqual({file.name for file in self.destination.iterdir()}, REQUIRED_FILES)

    def test_existing_weights_are_preserved(self):
        self.destination.mkdir()
        model = self.destination / "model.pth"
        model.write_bytes(b"existing model")
        with self.assertRaises(FileExistsError):
            install_archive(self.archive, self.destination, expected_size=self.size, expected_sha256=self.digest)
        self.assertEqual(model.read_bytes(), b"existing model")
