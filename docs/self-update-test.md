# Testing the self-update end to end

The Windows launcher updates itself and its bundled XSEmu Core from the releases of this repository.
Two parts move separately: `release.yml` publishes `XSEmuLauncher-windows.zip` (the launcher with the
Core in `core/`), `XSEmuCore-windows.zip` (the Core alone) and `xsemu-manifest.json`, which names each
part's asset and SHA-256.

Proving that on a real Windows machine needs two releases: the launcher from release N replaces itself
and its Core from release N+1.

## The test lever

Installing an update is a click in Settings → Updates, and the released launcher is built with
`XSEMU_DEV_TOOLS=OFF`, so it has no other way in. `XSEMU_UPDATE_TEST` adds one, for this test only:

| value | what the launcher does |
| --- | --- |
| `install` | takes whatever a check finds: the Core first, then the launcher, which stages itself, quits and lets its updater script replace the files and restart it. A failed step ends the sequence. It also checks whether or not the settings ask for checks. |
| `restore` | puts the previous Core (`core.old`) back on start, then quits. |

Set it to anything else and the launcher records that it armed nothing. Unset — every normal run — the
variable does nothing at all. Everything it does goes into `update-test.log` beside the settings file
(`%LOCALAPPDATA%\XSEmu\XSEmuLauncher\`), one line per step with the versions, and to stderr.

## The test

Use a folder of your own, never an installed launcher, and give it a data folder of its own so the
test's settings and logs cannot touch yours:

```powershell
# The release that carries the lever, and a newer one for it to find.
$from = 'v0.1.1-pre.2'
$to   = 'v0.1.1-pre.3'

$root = "$env:TEMP\xsemu-selfupdate-test"
$app  = "$root\app"; $env:XSEMU_LAUNCHER_DATA = "$root\data"
Remove-Item -Recurse -Force $root -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force $root | Out-Null

# 1. the from-release, unpacked into the scratch folder
$base = "https://github.com/zortos293/XSEmu/releases/download"
Invoke-WebRequest "$base/$from/xsemu-manifest.json" -OutFile "$root\from-manifest.json"
Invoke-WebRequest "$base/$from/XSEmuLauncher-windows.zip" -OutFile "$root\from.zip"
if ((Get-FileHash "$root\from.zip" -Algorithm SHA256).Hash -ne
    (Get-Content "$root\from-manifest.json" -Raw | ConvertFrom-Json).launcher.sha256) { throw 'the from-release archive is not its manifest hash' }
Expand-Archive "$root\from.zip" -DestinationPath "$root\from"
New-Item -ItemType Directory -Force $app | Out-Null
Copy-Item "$root\from\XSEmuLauncher\bin\*" $app -Recurse

# 2. arm the lever and let it run
$env:XSEMU_UPDATE_TEST = 'install'
Start-Process "$app\XSEmuLauncher.exe" -WorkingDirectory $app
```

The check compares versions, so `$to` only has to exist and be newer than `$from`; nothing points the
launcher at it.

The launcher checks six seconds in, installs the Core, then the launcher, quits, and the updater script
replaces its own files and starts it again. Watch it happen in `%TEMP%\xsemu-launcher-update.log` (the
updater script's own log) and in the data folder's `update-test.log`.

What has to be true afterwards:

- `core\XSEmu.exe` is the new release's program and `core.old\XSEmu.exe` is the old one, until the new
  Core has started once;
- the launcher's files are the new release's, and the updater log says `keeping core\...` for every
  staged Core file — a launcher update never replaces the Core;
- `core\version.json` names `$to`'s version, and `XSEmu.exe --version` agrees;
- with `XSEMU_UPDATE_TEST=restore` on a later start, `core.old` goes back and `core\XSEmu.exe` is the
  old program again, while the launcher stays new.

Delete the scratch folder afterwards. Nothing in this test writes outside it and `%TEMP%`, and the
installed launcher, its settings and its saves are never opened.
