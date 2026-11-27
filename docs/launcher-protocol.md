# XSEmu launcher ↔ emulator protocol (v1)

The launcher (`launcher/`, Qt 6) and the emulator (`XSEmu.exe`, crate `xcl`) are separate
processes. XSEmu.exe keeps every behaviour it has today when none of the options below are given.

## Command lines

```
XSEmu.exe run [--host-hwnd <hex>] [--control-pipe <name>] <game folder | file.exe> [-- game arguments]
XSEmu.exe warm-up [--full] [--progress-json]
```

- Launcher options sit between `run` and the game folder, so `XSEmu.exe run <folder> a b` still passes
  `a b` to the game exactly as before.
- `--host-hwnd <hex>`: the HWND (hex, with or without `0x`) of a native widget in the launcher. The game
  window is created as `WS_CHILD | WS_CLIPSIBLINGS | WS_CLIPCHILDREN` of it instead of a top-level window,
  fills the parent's client area and follows its resizes. The launcher sizes the host widget to 16:9 itself;
  the emulator letterboxes to 16:9 inside whatever client rect it gets, so both agree: the child window is the
  largest centred 16:9 rectangle of the host's client area, and the bars around it are the host widget's own
  background, so the launcher paints the host black. The emulator polls the host's client rect (about every
  33 ms) and moves the child; the host window closing ends the title (exit reason `stopped`). Fullscreen is
  the launcher's job (borderless top-level window, host fills the monitor); the child just follows the resize.
- `--control-pipe <name>`: the launcher listens on a local server called `<name>` (Qt `QLocalServer`; on
  Windows that is the named pipe `\\.\pipe\<name>`). The emulator connects as a client once its window exists
  (or immediately if it has none), and speaks the JSON lines below. The name is per process:
  `xsemu-<launcher pid>-<counter>`.
- `run` starts the title in a host child process (`XCL_HOST`); the window and the pipe live in that child.
  The outer process forwards the options to it (environment `XSEMU_HOST_HWND` and `XSEMU_CONTROL_PIPE`) and
  exits with the child's exit code, as today.
- `warm-up --progress-json`: stdout carries only the JSON lines described under Warm-up; human text goes to
  stderr. Commands arrive on stdin as JSON lines.

## Framing

UTF-8 JSON objects, one per line, `\n` terminated, no pretty printing. Unknown fields and unknown messages are
ignored by both sides. Numbers are JSON numbers. Every emulator message has `"ev"`; every launcher message has
`"cmd"`.

## Game control channel (`run --control-pipe`)

Launcher → emulator:

| Message | Meaning |
|---|---|
| `{"cmd":"pause"}` | Stop the title making progress: block its frame presentation and stop delivering keyboard, mouse and gamepad input. Answer with `state` `paused`. |
| `{"cmd":"resume"}` | Undo pause. Answer with the `state` it returns to: `running` (`booting` if no frame was shown yet). |
| `{"cmd":"stop"}` | End the title cleanly and exit with reason `stopped`. The launcher kills the process tree if it has not exited 5 s later. |
| `{"cmd":"overlay","on":true}` | Show or hide the in-frame performance overlay (drawn by the emulator into its own frame: a graph of the last 120 frame times in the top-left corner, green to 60 fps, amber to 30, red slower, with a 16.7 ms line; no text). Without `on` it toggles. |
| `{"cmd":"focus"}` | Take keyboard focus (`SetFocus` on the child window). Sent when the launcher shows the game view or the user clicks it. |

Emulator → launcher:

| Message | Meaning |
|---|---|
| `{"ev":"hello","proto":1,"pid":4242,"version":"0.1.0","title":"RockstarGamesInc.28872443D20A7","device_type":"lockhart","hwnd":"1a2b3c"}` | First line after connecting. `hwnd` is the child window (absent without `--host-hwnd`). |
| `{"ev":"state","state":"booting"}` | `booting` → `running` (first frame presented) → `paused`/`running` → `stopping`. |
| `{"ev":"stats","fps":58.2,"frame_ms":17.2,"ram_free_mb":9830,"vram_used_mb":5222,"vram_budget_mb":8192,"stalls":0,"worst_ms":34.0,"pipelines":41906,"compiling":0}` | Once a second while booting or running. Fields the emulator cannot measure yet are left out (never invented). |
| `{"ev":"key","key":"F1"}` | A launcher shortcut pressed while the game window has focus: `F1` (pause), `F3` (overlay), `F11` (fullscreen), `Escape`. The key is not passed to the title. |
| `{"ev":"exit","code":0,"reason":"stopped","message":""}` | Last line before a deliberate exit. `reason`: `stopped` (launcher asked), `title_exit` (the title ended), `crash` (written by the crash handler if it can; `message` has the exception and module), `device_removed` (D3D12 device removed; `message` has the removal reason), `error` (boot failed; `message` says why, e.g. game files missing). |

If the process ends without an `exit` line, the launcher treats it as a crash with the process exit code.

Details of the emulator side:

- `stats` covers the second since the last one: `fps` frames shown, `frame_ms` and `worst_ms` the mean and longest
  time between two shown frames, `stalls` the frames shown 100 ms or more after the one before, `ram_free_mb`
  available physical memory, `vram_used_mb` / `vram_budget_mb` the adapter's local segment
  (`QueryVideoMemoryInfo`), `pipelines` the pipelines built so far, `compiling` the ones being built. No `stats`
  while paused.
- `key` is sent only with `--host-hwnd` (the keys go to the child window); Escape then no longer reaches the
  title as B.
- `exit` `code` is the process exit code (an NTSTATUS for `crash` and `device_removed`). A boot failure in
  `XSEmu run` itself (game files missing, bad `xcl.toml`, CPU check) is sent by the outer process: its `hello`
  has that process's `pid` and no `title`.
- When the launcher closes the pipe (or ends), the emulator ends the title.

## Warm-up (`warm-up --progress-json`)

Emulator → launcher (stdout):

| Message | Meaning |
|---|---|
| `{"ev":"warmup_start","total":41906,"full":false,"threads":16}` | Once the target list is known. `total`: the kept pipelines this pass builds (after `--full` resume and the stale filter); the giants and compute blobs are found when their phases start. |
| `{"ev":"warmup_progress","done":27292,"total":41906,"keys_per_s":243.0,"eta_s":60,"ram_free_mb":12288,"giants_compiling":2,"in_flight":5,"failed":0,"phase":"bound","paused_for_ram":false}` | At most once a second, plus at the end. The same numbers as the existing progress window and console line. `done`/`total` count the current `phase` and start again at 0 of the next phase's total; `phase` is, in order, `starting`, `giants`, `bound`, `compute`, `compute_blobs`, `rest`. `keys_per_s` is the phase's average since it started and `eta_s` the rest of the phase at that rate, left out while the rate is 0. `failed` counts all phases so far. `giants_compiling` is the giants (1 MB of DXIL and up) among the `in_flight` driver compiles; there is no fixed number of giants at once, so no maximum is sent: every compile, giants included, starts when its expected peak fits in the RAM available less the peaks of those running and a 1 GB floor, and in the commit headroom (commit limit less commit charge, less the same peaks) with 1 GB kept. `paused_for_ram` is true when, since the last line, a compile was held back by either of those two memory tests; compiles held for other reasons (a bind waiting, the background limit, memory falling fast) do not set it. |
| `{"ev":"warmup_done","built":41890,"failed":16,"left":16,"too_big":0,"took_s":812.4,"complete":true,"cancelled":false,"code":0}` | Last line, printed even when the pass failed builds (exit code 1) or was stopped. `complete` false when a budget or cancel stopped it part way. `left` counts the pipelines this pass did not build: those that failed, those too big for the memory limits, the giants a budgeted pass leaves for a full one, and those not started when a budget or cancel stopped it (`built + left` is what the pass went through). A pass that stopped without this line built nothing the next run will not try again: the driver blobs and the `--full` record stay on disk, so running the same command again goes on from there and builds only what `left` named. Exit codes: 0 done, 1 some pipelines failed to build, 2 no D3D12 device, 3 another XSEmu process is running, 4 cancelled. |
| `{"ev":"error","message":"another XSEmu process is running (XSEmu.exe 1234)"}` | Refused or failed before starting (exit code 3 for the one-process refusal, 2 for no D3D12 device). |

Launcher → emulator (stdin): `{"cmd":"cancel"}`. The warm-up stops starting new compiles (one still waiting for
its memory slot gives up), lets the ones in flight finish, keeps everything already built (blobs and the `--full` resume record), prints
`warmup_done` with `cancelled` true and exits with code 4. Running it again goes on from there. EOF on stdin
counts as cancel, so a launcher crash never leaves a warm-up running unattended. `cancelled` is true only when the
cancel left builds unstarted; a cancel that arrives after the last build started ends as a normal finish.

`--progress-json` applies to `warm-up [--full]` only; `warm-up --mesh-only` and `warm-up --giants-only` keep their
console output.

## One process at a time

Only one XSEmu process tree runs at a time: one game or one warm-up. The emulator already refuses a warm-up
while another XSEmu.exe (or xcl.exe) runs; the launcher enforces the same rule in its UI and kills the whole tree
(Windows job object, `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`) when it has to.
