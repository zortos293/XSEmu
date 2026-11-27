#pragma once

#include <QString>

// Every file location the launcher reads or writes, mirroring the emulator (crates/xcl/src/launcher.rs,
// config.rs Boot::logs_dir/cache_dir, gpu/recomp/dxc.rs cache_dir). Functions of the environment and,
// for runners, of the folders the user chose (RunnerManager::setSettings applies them).
namespace Paths {

// %XCL_DATA_ROOT%, else %LOCALAPPDATA%\XCL (Linux: $XCL_DATA_ROOT, else ~/.local/share/XCL).
QString dataRoot();

// The emulator's per-title folder name: the Identity Name of MicrosoftGame.config (else the game
// folder's name) with every character outside [A-Za-z0-9._-] (Unicode letters and digits allowed) as '_'.
QString titleDirName(const QString& identityOrFolderName);

QString titleDir(const QString& titleDirName);
QString xclToml(const QString& titleDirName);
QString logsDir(const QString& titleDirName);    // <title>\logs (xcl.log, xcl.prev.log, report.json)
QString cacheDir(const QString& titleDirName);   // <title>\cache (boot.json)
QString reportJson(const QString& titleDirName); // <title>\logs\report.json
QString xclLog(const QString& titleDirName);     // <title>\logs\xcl.log

// Machine-wide shader cache: %XCL_DXIL_CACHE%, else %LOCALAPPDATA%\xcl\dxil.
QString dxilCacheDir();
QString pipelineManifestDir(); // <dxil>\pipelines (pipelines.bin, shaders\, runs, created.bin)
QString driverBlobDir();       // <dxil>\driver\<vendor-device-umd>\*.pso

// Launcher-owned data: %XSEMU_LAUNCHER_DATA%, else QStandardPaths::AppLocalDataLocation (%LOCALAPPDATA%\XSEmu).
QString launcherDataDir();
QString settingsFile(); // <launcher data>\settings.ini
QString coverCacheDir(); // <launcher data>\covers

// Third-party runners: %XSEMU_RUNNERS_DIR%, else the Runners folder the user chose, else
// %LOCALAPPDATA%\\XSEmu\\runners (Linux: ~/.local/share/XSEmu/runners). The Rust toolchain cache
// (_toolchains) lives here.
QString runnersDir();
QString defaultRunnersDir();
bool runnersDirFromEnvironment();
void setRunnersFolder(const QString& dir); // empty: the default
// Per runner: the folder the user moved it to, which holds <folder>\\<id>; it wins over the rest.
void setRunnerLocations(const QHash<QString, QString>& folderById);
QString runnerLocation(const QString& runnerId); // the folder that holds runnerDir(runnerId)
QString localAppDataDir(); // %LOCALAPPDATA% (Linux: ~/.local/share), where runners keep their caches
QString runnerDir(const QString& runnerId); // <runnerLocation(id)>\\<id>

// The folder the launcher runs from: its own folder, which holds the Qt runtime and the bundled
// XSEmu Core in core\.
QString appDir();
// The bundled XSEmu Core: <launcher dir>\core (Linux: core/fake-xsemu). A Core update replaces the
// files in it and keeps the previous copy in core.old until the new one has started.
QString coreDir();
// The emulator program's file name: XSEmu.exe (Linux: fake-xsemu).
QString emulatorExecutableName();

// The emulator executable's default location: the bundled Core at <launcher dir>\core\XSEmu.exe,
// else XSEmu.exe next to the launcher (a copy placed by hand, the layout of the 0.1.0 release), else
// where the bundled Core would be.
QString defaultEmulatorPath();

}
