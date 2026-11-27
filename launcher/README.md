# XSEmu Launcher

`XSEmuLauncher` is the desktop front end for **XSEmu Core**, the emulator bundled next to it as
`XSEmu.exe`: a game library, the shader warm-up, settings, and the in-game window. It is a Qt 6
Widgets app in C++17. XSEmu Core stays a separate process: the launcher starts it, talks to it over
the pipe described in [docs/launcher-protocol.md](../docs/launcher-protocol.md), and runs only one
XSEmu Core process tree at a time: one game or one warm-up.

The design is the Paper file "XSEmu Launcher · Direction C, Cover grid". Its tokens (colours, type
scale, radii, spacing, control heights) are in `src/theme/Theme.h` and `Theme.cpp`.

## Build

You need CMake 3.21 or later, Ninja, and Qt 6.4 or later with the Core, Gui, Widgets, Network,
Svg and Test modules.

Windows (Qt 6.8, MSVC 2022), from a Developer PowerShell:

```powershell
cmake -S launcher -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=C:/Qt/6.8.3/msvc2022_64
cmake --build build
cmake --install build --prefix dist   # runs windeployqt into dist/bin
```

Put `XSEmu.exe` next to `XSEmuLauncher.exe`, or set its path in Settings.

Linux (Ubuntu 24.04, development only):

```sh
sudo apt install cmake ninja-build g++ qt6-base-dev qt6-svg-dev
cmake -S launcher -B build -G Ninja
cmake --build build
cd build && ctest --output-on-failure
```

### Options

| Option | Default | What it does |
| --- | --- | --- |
| `XSEMU_LAUNCHER_TESTS` | on (off on Windows) | Builds the unit and controller tests. |
| `XSEMU_FAKE_EMULATOR` | on (off on Windows) | Builds `fake-xsemu`, a stand-in for `XSEmu.exe`. |
| `XSEMU_DEV_TOOLS` | on (off on Windows) | Builds the screenshot lever (`--state`, `--grab`). |
| `XSEMU_DEPLOY_QT` | on | Runs `windeployqt` after the build and at install (Windows). |

## Run

```sh
./XSEmuLauncher
```

The settings file is `settings.ini` in the app's local data folder. Set `XSEMU_LAUNCHER_DATA` to
use another folder. The launcher reads emulator data from `%LOCALAPPDATA%\XCL` (`XCL_DATA_ROOT`
overrides it) and the shader cache from `%LOCALAPPDATA%\xcl\dxil` (`XCL_DXIL_CACHE`).

The launcher needs no wrapper script: it finds `XSEmu.exe` next to itself, or the one chosen in
Settings, About (or from the missing-emulator dialog); `XSEMU_EMULATOR` still wins over both. The
`XCL_*` switches XSEmu Core runs with are the run profile in Settings.

On Linux the launcher looks for `fake-xsemu` next to itself instead of `XSEmu.exe`. Choose its
behaviour with `FAKE_XSEMU_SCENARIO`: `ok`, `crash`, `device_removed`, `files_missing`,
`slow_boot`, `warmup_fail`, `warmup_partial`, `warmup_ram_floor` or `warmup_refuse`. `FAKE_XSEMU_TICK_MS` sets
its speed, and `FAKE_XSEMU_ENV_DUMP` names a folder where it writes the `XCL_*` environment of each
run and warm-up.

## Screenshot lever

Developer builds can put the window into a design state on fixture data and save a grab, so a state
can be compared with the Paper renders:

```sh
QT_QPA_PLATFORM=offscreen ./XSEmuLauncher --theme dark --size 1280x800 --state warmup-running --grab out.png
./XSEmuLauncher --list-states
```

The lever copies `tests/fixtures` to a temporary folder and points the launcher data, the emulator
data root and the shader cache there. It reads and writes nothing else on the machine. Without
`--grab` the window stays open in that state. Keep grabs out of the repository.

## Settings

Every control is backed by an `xcl.toml` key, an `XCL_*` variable or a launcher setting. Controls
XSEmu Core can't honour yet are shown disabled with the reason. The `xcl.toml` keys are launcher defaults
(`AppSettings::xclDefault`, under `xcl/`): `LaunchConfig::prepareTitle` writes them into the title's
`xcl.toml` at launch, and a game's own setting in Game details wins over the default. The `XCL_*`
variables come from `LaunchConfig::runEnvironment`.

| Section | Control | Backing |
| --- | --- | --- |
| Console | Device type | `device_type` (`lockhart` or `anaconda`), from `defaultDeviceType` |
| Network | Offline mode | `XCL_NETWORK=0` when on |
| Network | Store behaviour | `XCL_STORE=offline-error` (Offline error), unset (Not available) |
| Shaders | Warm up when the translator changes | Disabled: XSEmu Core doesn't report its translator version yet |
| Shaders | RAM floor | Read only, 1 GB (`EmulatorController::kRamFloorMb`), with the admission rule |
| Shaders | Cache folder | `XCL_DXIL_CACHE`; a value already in the environment wins |
| Shaders | Clear cache | Deletes the cache folder's contents when it looks like an XSEmu Core cache; never the driver's cache |
| Run profile | Variables, reset to defaults | `runProfile` (`KEY=VALUE` lines, `AppSettings::defaultRunProfile` until edited), in the environment of Play and of XSEmu Core's warm-up alike; `XCL_NETWORK`, `XCL_STORE`, `XCL_FULL_DUMP` and `XCL_DXIL_CACHE` lines are not used |
| Game details | Use a custom run profile for this game | The game's `runProfile` override, replacing the global profile for its Play and warm-up |
| Overlay | Stall and performance overlay | `overlayDefault`, sent over the control pipe at launch |
| Overlay | Position, size, metrics | Disabled: not in the protocol |
| Game folders | Folders, rescan | `gameFolders` |
| Game folders | Download game art (on by default), clear downloaded art, personal SteamGridDB key | `artLookups`, `CoverArtService::clearCache`, `steamGridDbKey` |
| Game details | Artwork: SteamGridDB match (Change...), Choose image... / Reset per kind, Refresh artwork | the cover cache's `pinned` and `custom` entries (`CoverArtService::setMatch`, `setCustomImage`) |
| Game folders | Runners folder (free space, warning under 20 GB; `XSEMU_RUNNERS_DIR` wins) | `runnersFolder` (`RunnerManager::setRunnersFolder`) |
| Runners page | Change location on a runner's card (Move or Rebuild there) | `runnerLocations` (`RunnerManager::relocate`) |
| Input | Keyboard and mouse as gamepad | `keyboard_mouse` |
| Input | Mouse sensitivity | `mouse_sensitivity` |
| Input | Controller | XInput state (Windows only) |
| Graphics | Theme, game bar in a window | `theme`, `ingame/windowBarAutoHide` |
| Graphics | Show the game image | `present` (`auto` or `off`) |
| Graphics | Display mode, D3D12 adapter, raytracing | `display_mode`, `d3d12_adapter`, `raytracing`; each is disabled while the run profile or the launcher's environment sets `XCL_DISPLAY_MODE`, `XCL_ADAPTER` or `XCL_RAYTRACING` |
| Graphics | Start games in, scaling, low-latency present | Disabled: not in this build |
| Logs | Log detail | `log_level` (`info` or `debug`) |
| Logs | Keep logs for | Disabled: XSEmu Core keeps `xcl.log` and `xcl.prev.log` |
| Logs | Crash reports | `XCL_FULL_DUMP=<data root>/<title>/logs/dumps` when on |
| About | XSEmu.exe version | `XSEmu.exe --version` |
| About | Emulator, Choose | `emulator/path`; `XSEMU_EMULATOR` wins |

## Layout

| Path | Contents |
| --- | --- |
| `src/core` | Services without UI, built as the `xsemu_core` library and unit tested: paths, the game library, `MicrosoftGame.config` and `xcl.toml` handling, the protocol, settings, the process gate and tree, the emulator and warm-up controllers, cover art, the shader cache. |
| `src/theme` | Design tokens, light and dark palettes, the global style sheet. |
| `src/widgets` | The component kit: Label, Button, Badge, Toggle, SegmentedControl, SearchBox, CoverTile, ProgressBlock, Card, Dialog, InfoNote, ChoiceCard, NavItem, PageHeader, Spinner, icons and the app mark. |
| `src/shell` | The frameless main window, title bar, sidebar, in-game view and error dialogs. |
| `src/pages` | Library, Warm-up, Settings, Game details and First launch. Each Settings section is a file in `src/pages/settings`. |
| `src/app` | Service wiring and the screenshot lever. |
| `resources` | Line icons as SVG, the Qt resource file and the Windows `.rc`. The app logo comes from `assets/logo` at the repository root. |
| `tools/fake-xsemu` | The protocol stand-in used by the tests and on Linux. |
| `tests` | Unit tests, controller tests against `fake-xsemu`, and text fixtures. |

The repository's PR check refuses binary files and game file names, so fixtures are text files
with a `.fixture` suffix. `DevFixtures::materialize` copies them to their real names at test time.

## What works today

- Library scan and watch, cover tiles, locked tiles while a game or warm-up runs, search, the paused
  banner and NOW PLAYING.
- Launch checks (emulator missing, game files missing, busy, another XSEmu Core process running, not enough RAM)
  with their dialogs, and the crash, GPU-removed and boot-error dialogs.
- Warm-up start, cancel and per-phase progress from `--progress-json`, plus the recent runs.
- Resumable pre-render: a warm-up that fails, is cancelled or stops at the RAM floor or its time
  budget keeps every pipeline it built (XSEmu Core keeps the driver blobs on disk), and the Warm-up
  page shows how many are left and offers Continue, which runs the same pass again and builds only
  those. A launcher that closes or dies mid-pass leaves a checkpoint (`warmup/inFlight` in
  `settings.ini`), so Continue is offered after a restart too.
- Settings for console model, network, giant-compile cap, overlay, game folders and theme
  (light, dark or system, switched live).
- First launch in three steps.
- Cover, banner, logo and icon art from SteamGridDB for every game, through XSEmu's proxy
  (`tools/sgdb-proxy`, no key needed) or with a personal key; cached under `<launcher data>\covers`,
  fetched again only on Refresh artwork, with a per-game match and per-kind local images.

Not done yet: the game window is not docked into the in-game view (the launcher starts XSEmu
without `--host-hwnd`, so the game opens its own window), and there is no in-game overlay control,
key handling or fullscreen. Those come with the in-game view.
