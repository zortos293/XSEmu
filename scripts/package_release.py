#!/usr/bin/env python3
"""Package one XSEmu release: the launcher with XSEmu Core inside it, the Core alone, and the manifest.

The launcher and the emulator update separately (launcher/src/core/UpdateService.cpp), so a release
publishes two archives and a manifest that names, per part, one asset with its SHA-256:

  XSEmuLauncher-windows.zip   XSEmuLauncher/bin/**  the launcher, its Qt runtime and core/XSEmu.exe
  XSEmuCore-windows.zip       XSEmu.exe, version.json   the Core alone, for a Core update
  XSEmu.exe                   the emulator alone (for someone who copies it by hand)
  xsemu-manifest.json         schema 1: the tag and one entry per part

The script only assembles files; .github/workflows/release.yml builds them, then hands the result to
`gh release create`. Run it by hand the same way:

  cargo build --release -p xcl
  cmake -S launcher -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DXSEMU_LAUNCHER_VERSION=0.1.0-pre.2
  cmake --build build && cmake --install build --prefix dist
  python3 scripts/package_release.py --dist dist/bin --core target/release/XSEmu.exe \
      --out out --version 0.1.0-pre.2 --tag v0.1.0-pre.2

Nothing here touches game data, keys or saves: the inputs are the launcher's install tree, the
emulator program and optionally Microsoft's VC++ runtime installer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

LAUNCHER_ASSET = "XSEmuLauncher-windows.zip"
CORE_ASSET = "XSEmuCore-windows.zip"
PROGRAM_ASSET = "XSEmu.exe"
MANIFEST_ASSET = "xsemu-manifest.json"
BUNDLE_DIR = "XSEmuLauncher"
CORE_DIR = "core"
PROGRAM = "XSEmu.exe"
VERSION_FILE = "version.json"
# A zip entry from before 1980 upsets some unpackers; a fixed stamp also keeps the archive
# byte-identical between runs, so the SHA-256 in the manifest is reproducible.
STAMP = (2026, 1, 1, 0, 0, 0)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_zip(zip_path: Path, files: list[tuple[str, Path]]) -> None:
    """Zip the given (archive name, file) pairs, sorted, with a fixed timestamp."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, source in sorted(files):
            info = zipfile.ZipInfo(name, STAMP)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = (0o755 if source.suffix.lower() == ".exe" else 0o644) << 16
            archive.writestr(info, source.read_bytes())


def tree_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*") if p.is_file())


def assemble(bundle: Path, dist: Path, core: Path, version: str, tag: str, vc_redist: Path | None) -> None:
    bin_dir = bundle / "bin"
    if bin_dir.exists():
        shutil.rmtree(bin_dir)
    shutil.copytree(dist, bin_dir)
    if not (bin_dir / "XSEmuLauncher.exe").is_file():
        raise SystemExit(f"{dist} has no XSEmuLauncher.exe; run cmake --install first")
    core_dir = bin_dir / CORE_DIR
    core_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(core, core_dir / PROGRAM)
    (core_dir / VERSION_FILE).write_text(
        json.dumps({"version": version, "tag": tag}, indent=2) + "\n", encoding="utf-8")
    if vc_redist is not None:
        shutil.copy2(vc_redist, bin_dir / vc_redist.name)


def build(version: str, tag: str, dist: Path, core: Path, out: Path, vc_redist: Path | None,
          notes: Path | None) -> dict:
    if not core.is_file():
        raise SystemExit(f"no emulator program at {core}")
    if out.exists():
        shutil.rmtree(out)
    bundle = out / BUNDLE_DIR
    assemble(bundle, dist, core, version, tag, vc_redist)

    launcher_zip = out / LAUNCHER_ASSET
    write_zip(launcher_zip, [(str(p.relative_to(out)).replace("\\", "/"), p) for p in tree_files(bundle)])

    core_dir = bundle / "bin" / CORE_DIR
    core_files = tree_files(core_dir)
    core_zip = out / CORE_ASSET
    write_zip(core_zip, [(p.relative_to(core_dir).as_posix(), p) for p in core_files])
    program = out / PROGRAM_ASSET
    shutil.copy2(core, program)

    manifest = {
        "schema": 1,
        "tag": tag,
        "version": version,
        "prerelease": "-" in version,
        "launcher": {
            "version": version,
            "asset": LAUNCHER_ASSET,
            "sha256": sha256_of(launcher_zip),
            "size": launcher_zip.stat().st_size,
        },
        "core": {
            "version": version,
            "asset": CORE_ASSET,
            "sha256": sha256_of(core_zip),
            "size": core_zip.stat().st_size,
            "files": {p.relative_to(core_dir).as_posix(): sha256_of(p) for p in core_files},
        },
    }
    (out / MANIFEST_ASSET).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if notes is not None:
        notes.write_text(release_notes(manifest, vc_redist is not None), encoding="utf-8")
    return manifest


def release_notes(manifest: dict, with_redist: bool) -> str:
    launcher = manifest["launcher"]
    core = manifest["core"]
    version = manifest["version"]
    install = [
        "1. Unzip **%s** anywhere; it holds `%s/bin/`." % (launcher["asset"], BUNDLE_DIR),
        "2. Start `%s/bin/XSEmuLauncher.exe`." % BUNDLE_DIR,
        "3. Add your game folder in the launcher and press Play.",
        "",
        "XSEmu Core is inside the launcher: `%s/bin/core/%s`. There is no separate download to place, "
        "and the launcher finds it there." % (BUNDLE_DIR, PROGRAM),
    ]
    if with_redist:
        install.append("")
        install.append("If Windows reports a missing runtime, run `%s/bin/vc_redist.x64.exe`." % BUNDLE_DIR)
    table = [
        "| File | What it is | SHA-256 |",
        "| --- | --- | --- |",
        "| %s | The launcher, Qt 6 / MSVC x64, with XSEmu Core in `core/` | `%s` |" % (launcher["asset"], launcher["sha256"]),
        "| %s | XSEmu Core alone, for a Core update | `%s` |" % (core["asset"], core["sha256"]),
    ]
    return "\n".join([
        "# XSEmu %s" % version,
        "",
        "The launcher and XSEmu Core, built from this tag by the release workflow.",
        "",
        "## Install",
        "",
        *install,
        "",
        "Updates: Settings → Updates checks this repository's releases and updates the launcher and "
        "XSEmu Core separately. Each asset is checked against the SHA-256 below before it is used, "
        "the previous copy is kept until the new one has started, and nothing updates while a game or "
        "a warm-up runs.",
        "",
        "## Assets",
        "",
        *table,
        "",
        "## Source and licence",
        "",
        "The source is this repository at tag `%s`: GPL-3.0-or-later. The launcher ships Qt 6 under "
        "the LGPL-3.0 and Microsoft's DXC (`dxcompiler.dll`, `dxil.dll`) under its own licence. No game "
        "files, keys or saves are included, and none are downloaded." % manifest["tag"],
        "",
    ])


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Package one XSEmu release.")
    parser.add_argument("--version", required=True, help="the release's version, e.g. 0.1.0-pre.2")
    parser.add_argument("--tag", required=True, help="the release's git tag, e.g. v0.1.0-pre.2")
    parser.add_argument("--dist", required=True, type=Path, help="the launcher's install tree (…/bin)")
    parser.add_argument("--core", required=True, type=Path, help="the emulator program (XSEmu.exe)")
    parser.add_argument("--out", required=True, type=Path, help="where the assets are written")
    parser.add_argument("--vc-redist", type=Path, default=None, help="Microsoft's VC++ runtime installer")
    parser.add_argument("--notes", type=Path, default=None, help="write the release body here")
    args = parser.parse_args(argv)
    manifest = build(args.version, args.tag, args.dist, args.core, args.out, args.vc_redist, args.notes)
    for part in ("launcher", "core"):
        entry = manifest[part]
        print("%-48s %8d bytes  %s" % (entry["asset"], entry["size"], entry["sha256"]))
    print("%-48s %8d bytes" % (MANIFEST_ASSET, (args.out / MANIFEST_ASSET).stat().st_size))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
