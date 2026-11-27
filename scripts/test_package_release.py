import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from package_release import CORE_ASSET, LAUNCHER_ASSET, MANIFEST_ASSET, PROGRAM_ASSET, build, sha256_of


def make_dist(root: Path) -> Path:
    """A stand-in for the launcher's cmake --install tree."""
    dist = root / "dist"
    (dist / "platforms").mkdir(parents=True)
    (dist / "XSEmuLauncher.exe").write_bytes(b"launcher")
    (dist / "Qt6Core.dll").write_bytes(b"qt")
    (dist / "platforms" / "qwindows.dll").write_bytes(b"platform")
    (dist / "dxcompiler.dll").write_bytes(b"dxc")
    return dist


class PackageReleaseTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.dist = make_dist(self.root)
        self.core = self.root / "XSEmu.exe"
        self.core.write_bytes(b"the emulator")
        self.out = self.root / "out"

    def tearDown(self):
        self.tmp.cleanup()

    def run_build(self, version="0.1.0-pre.2", tag="v0.1.0-pre.2", out=None):
        notes = self.root / "notes.md"
        manifest = build(version, tag, self.dist, self.core, out or self.out, None, notes)
        return manifest, notes

    def names(self, asset):
        with zipfile.ZipFile(self.out / asset) as archive:
            return sorted(archive.namelist())

    def test_bundle_carries_the_core_inside_the_launcher(self):
        self.run_build()
        self.assertIn("XSEmuLauncher/bin/XSEmuLauncher.exe", self.names(LAUNCHER_ASSET))
        self.assertIn("XSEmuLauncher/bin/core/XSEmu.exe", self.names(LAUNCHER_ASSET))
        self.assertIn("XSEmuLauncher/bin/core/version.json", self.names(LAUNCHER_ASSET))
        self.assertIn("XSEmuLauncher/bin/Qt6Core.dll", self.names(LAUNCHER_ASSET))
        bundle = self.out / "XSEmuLauncher" / "bin" / "core"
        self.assertEqual(b"the emulator", (bundle / "XSEmu.exe").read_bytes())
        self.assertEqual("0.1.0-pre.2", json.loads((bundle / "version.json").read_text())["version"])

    def test_core_asset_is_the_core_alone(self):
        self.run_build()
        self.assertEqual(["XSEmu.exe", "version.json"], self.names(CORE_ASSET))

    def test_manifest_names_both_parts_with_their_hashes(self):
        manifest, _ = self.run_build()
        self.assertEqual(1, manifest["schema"])
        self.assertEqual("v0.1.0-pre.2", manifest["tag"])
        self.assertTrue(manifest["prerelease"])
        for part, asset in (("launcher", LAUNCHER_ASSET), ("core", CORE_ASSET)):
            entry = manifest[part]
            self.assertEqual("0.1.0-pre.2", entry["version"])
            self.assertEqual(asset, entry["asset"])
            self.assertEqual(sha256_of(self.out / asset), entry["sha256"])
            self.assertEqual((self.out / asset).stat().st_size, entry["size"])
        self.assertEqual(
            {"XSEmu.exe": sha256_of(self.core)},
            {name: digest for name, digest in manifest["core"]["files"].items() if name == "XSEmu.exe"},
        )
        self.assertIn("version.json", manifest["core"]["files"])
        self.assertEqual(manifest, json.loads((self.out / MANIFEST_ASSET).read_text()))

    def test_program_asset_is_published_too(self):
        self.run_build()
        self.assertEqual(b"the emulator", (self.out / PROGRAM_ASSET).read_bytes())

    def test_a_release_version_is_not_a_prerelease(self):
        manifest, _ = self.run_build(version="0.1.0", tag="v0.1.0")
        self.assertFalse(manifest["prerelease"])

    def test_archives_are_reproducible(self):
        first, _ = self.run_build()
        second, _ = self.run_build(out=self.root / "out2")
        self.assertEqual(first["launcher"]["sha256"], second["launcher"]["sha256"])
        self.assertEqual(first["core"]["sha256"], second["core"]["sha256"])

    def test_notes_describe_the_install_without_a_separate_download(self):
        manifest, notes = self.run_build()
        text = notes.read_text()
        self.assertIn(LAUNCHER_ASSET, text)
        self.assertIn("core/XSEmu.exe", text)
        self.assertIn(manifest["launcher"]["sha256"], text)
        self.assertIn(manifest["core"]["sha256"], text)

    def test_vc_redist_is_included_when_given(self):
        redist = self.root / "vc_redist.x64.exe"
        redist.write_bytes(b"redist")
        build("0.1.0-pre.2", "v0.1.0-pre.2", self.dist, self.core, self.out, redist, None)
        self.assertIn("XSEmuLauncher/bin/vc_redist.x64.exe", self.names(LAUNCHER_ASSET))
        self.assertNotIn("vc_redist", "\n".join(self.names(CORE_ASSET)))

    def test_a_dist_without_the_launcher_is_refused(self):
        (self.dist / "XSEmuLauncher.exe").unlink()
        with self.assertRaises(SystemExit):
            self.run_build()


if __name__ == "__main__":
    unittest.main()
