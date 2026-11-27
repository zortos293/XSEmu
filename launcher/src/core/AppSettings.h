#pragma once

#include <QDateTime>
#include <QHash>
#include <QList>
#include <QMetaType>
#include <QObject>
#include <QSettings>
#include <QStringList>
#include <QVariantMap>

#include <memory>
#include <optional>

// One finished warm-up, as the Warm-up page's "Recent runs" table shows it.
struct WarmupRun {
    // NothingToBuild: a runner's pre-cache ended cleanly with no work, and message says why
    enum class Result { Finished, Cancelled, StoppedPartWay, Failed, NothingToBuild };

    QString titleId;    // the game the user started it for (the cache itself is machine-wide)
    QString titleName;
    QString runnerId;   // empty: XSEmu's warm-up; else the runner whose pre-cache this was
    qint64 done = 0;    // pipelines built (built + failed)
    qint64 total = 0;
    qint64 failed = 0;
    qint64 left = -1;   // pipelines this pass did not build (-1: the emulator did not say)
    double tookS = 0;
    Result result = Result::Finished;
    int exitCode = 0;
    bool full = false;
    bool stoppedForRam = false; // ended part way while waiting at the RAM floor
    QString message;            // error text for Failed; why for NothingToBuild; a note for Finished
    QString profile;            // XSEmu's warm-up: LaunchConfig::profileIdentity it ran with; empty: not recorded
    QDateTime finishedAt;

    QVariantMap toMap() const;
    static WarmupRun fromMap(const QVariantMap& m);
};

Q_DECLARE_METATYPE(WarmupRun)

// The warm-up the launcher is running now, kept in AppSettings (Key::WarmupInFlight) while it runs and
// dropped when its result lands. A launcher that closes or dies mid-pass leaves it behind, so the next
// start knows which pass it was and how far it got and offers Continue instead of a fresh pass.
struct WarmupCheckpoint {
    QString titleId;
    QString titleName;
    QString runnerId;  // empty: XSEmu Core's warm-up
    QString profile;   // the run profile the pass started with; empty: not recorded
    bool full = false; // the pass was started with --full
    qint64 built = 0;  // pipelines it had finished with when this was written, failed builds included
    qint64 failed = 0; // of those, the ones that failed to build
    QString phase;     // the phase it had reached; empty: none reported yet
    QDateTime startedAt;

    bool isValid() const { return startedAt.isValid(); }
    QVariantMap toMap() const;
    static WarmupCheckpoint fromMap(const QVariantMap& m);
};

// Launcher preferences (QSettings ini at Paths::settingsFile()). Every setter writes at once and
// emits changed(key) with one of the AppSettings::Key constants, so a page can follow one setting:
//
//   connect(&settings, &AppSettings::changed, this, [this](const QString& key) {
//       if (key == AppSettings::Key::Offline) refresh();
//   });
//
// Per-title emulator settings do not live here: they are the title's xcl.toml (TitleConfig).
class AppSettings : public QObject {
    Q_OBJECT
public:
    enum class ThemeMode { System, Light, Dark };

    struct Key {
        static constexpr const char* Theme = "ui/theme";
        static constexpr const char* GameFolders = "library/folders";
        static constexpr const char* FirstLaunchDone = "app/firstLaunchDone";
        static constexpr const char* DefaultDeviceType = "console/defaultDeviceType";
        static constexpr const char* Offline = "network/offline";
        static constexpr const char* OverlayDefault = "overlay/default";
        static constexpr const char* SteamGridDbKey = "covers/steamGridDbKey";
        static constexpr const char* ArtLookups = "covers/lookups";
        static constexpr const char* WindowGameBarAutoHide = "ingame/windowBarAutoHide";
        static constexpr const char* DefaultRunner = "runners/default";
        static constexpr const char* RunnersFolder = "runners/folder";
        static constexpr const char* RunnerLocations = "runners/locations";
        static constexpr const char* RunnerOptions = "runners/options";
        static constexpr const char* StoreOfflineError = "network/storeOfflineError";
        static constexpr const char* CrashDumps = "logs/crashDumps";
        static constexpr const char* ShaderCacheDir = "shaders/cacheDir";
        static constexpr const char* XclDefaultsPrefix = "xcl/"; // + an xcl.toml key
        static constexpr const char* EmulatorPath = "emulator/path";
        static constexpr const char* BusyMarker = "emulator/busyMarker";
        static constexpr const char* LaunchNeedMb = "memory/launchNeedMb";
        static constexpr const char* GameOverrides = "games/overrides";
        static constexpr const char* RecentWarmups = "warmup/recentRuns";
        static constexpr const char* WarmupInFlight = "warmup/inFlight";
        static constexpr const char* PlayStats = "games/play";
        static constexpr const char* LibrarySort = "library/sort";
        static constexpr const char* RunProfile = "emulator/runProfile";
        static constexpr const char* XsemuWarmup = "warmup/xsemu";
    };

    static constexpr int kRecentWarmupLimit = 20;
    // No fixed run floor: a launch only needs the 1 GB floor free (EmulatorController::kRamFloorMb)
    // plus whatever the user configures here.
    static constexpr int kDefaultLaunchNeedMb = 0;

    // Opens Paths::settingsFile(), or the given ini file.
    explicit AppSettings(QObject* parent = nullptr);
    explicit AppSettings(const QString& iniPath, QObject* parent = nullptr);
    ~AppSettings() override;

    QString fileName() const;

    ThemeMode theme() const;
    void setTheme(ThemeMode mode);
    static QString toString(ThemeMode mode);
    static ThemeMode themeFromString(const QString& s);

    QStringList gameFolders() const;
    void setGameFolders(const QStringList& folders);

    bool firstLaunchDone() const;
    void setFirstLaunchDone(bool done);

    // "lockhart" (Series S, the default) or "anaconda" (Series X): written into a title's xcl.toml
    // before its first launch when the file has no device_type yet.
    QString defaultDeviceType() const;
    void setDefaultDeviceType(const QString& deviceType);

    // On by default: games start with XCL_NETWORK=0.
    bool offline() const;
    void setOffline(bool offline);

    bool overlayDefault() const;
    void setOverlayDefault(bool on);

    QString steamGridDbKey() const;
    void setSteamGridDbKey(const QString& key);
    bool artLookups() const; // the user's switch for cover lookups (on; nothing happens without a key)
    void setArtLookups(bool on);

    // The in-game bar while the game plays in a window: always shown (false, the default) or
    // auto-hidden until the mouse reaches the top edge (true). Launcher-side only.
    bool windowGameBarAutoHide() const;
    void setWindowGameBarAutoHide(bool on);

    // The runner a game uses unless its "runner" override names another (Runners ids; "xsemu").
    QString defaultRunner() const;
    void setDefaultRunner(const QString& id);
    // The runner this game runs with: its override, else the default.
    QString runnerFor(const QString& titleId) const;
    // Where runners are kept (empty: %LOCALAPPDATA%\XSEmu\runners; XSEMU_RUNNERS_DIR wins over it).
    QString runnersFolder() const;
    void setRunnersFolder(const QString& dir);
    // Runners moved somewhere else: id -> the folder that holds <id>. An empty folder clears it.
    QHash<QString, QString> runnerLocations() const;
    void setRunnerLocation(const QString& id, const QString& folder);
    // A runner's switches (RunnerSpec::options) the user turned on or off; one not set keeps its default.
    QHash<QString, bool> runnerOptions(const QString& id) const;
    void setRunnerOption(const QString& id, const QString& key, bool on);

    // On by default: games start with XCL_STORE=offline-error (store context opens, every store
    // query fails with the store network error). Off: the store API is refused as unavailable.
    bool storeOfflineError() const;
    void setStoreOfflineError(bool on);

    // On by default: games start with XCL_FULL_DUMP=<title>\logs\dumps (LaunchConfig::crashDumpDir).
    bool crashDumps() const;
    void setCrashDumps(bool on);

    // The user's shader cache folder (empty: XSEmu's default); LaunchConfig applies it as XCL_DXIL_CACHE.
    QString shaderCacheDirOverride() const;
    void setShaderCacheDirOverride(const QString& dir);

    // Launcher-wide defaults for xcl.toml keys, written into each title's xcl.toml at launch
    // (LaunchConfig::titleValues). Invalid when the user never set one: the title keeps its own value.
    // changed() carries Key::XclDefaultsPrefix + the key.
    QVariant xclDefault(const QString& tomlKey) const;
    void setXclDefault(const QString& tomlKey, const QVariant& value);
    void clearXclDefault(const QString& tomlKey);

    // The user's override of the emulator executable (empty: none).
    QString emulatorPathOverride() const;
    void setEmulatorPathOverride(const QString& path);
    // The executable used: env XSEMU_EMULATOR, else the override, else Paths::defaultEmulatorPath().
    QString emulatorPath() const;
    static bool emulatorPathFromEnvironment(); // XSEMU_EMULATOR is set, so the override is not used
    // A file whose existence means another job holds the emulator (Play and Warm-up wait): env
    // XSEMU_BUSY_MARKER, else the setting; empty: none.
    QString busyMarker() const;
    void setBusyMarker(const QString& path);

    // Free RAM a game launch asks for on top of the 1 GB floor.
    int launchNeedMb() const;
    void setLaunchNeedMb(int mb);

    // The environment XSEmu runs with for Play and the warm-up, one KEY=VALUE per line (LaunchConfig::
    // runEnvironment). defaultRunProfile() until the user edits it; reset forgets the edit.
    static const QStringList& defaultRunProfile();
    QStringList runProfile() const;
    void setRunProfile(const QStringList& lines);
    void resetRunProfile();
    // A game's own run profile (the "runProfile" game override), replacing the global one; empty when it has none.
    std::optional<QStringList> gameRunProfile(const QString& titleId) const;

    // Launcher-side per-game choices (not xcl.toml): an open map per title id.
    QVariantMap gameOverrides(const QString& titleId) const;
    void setGameOverride(const QString& titleId, const QString& key, const QVariant& value);
    void clearGameOverride(const QString& titleId, const QString& key);

    // XSEmu's own warm-up (XSEmu.exe warm-up): off, with no control to turn it on, because the current
    // XSEmu build exits from it without building anything. Off hides it everywhere: Warm-up page, Game
    // details, the warm-up prompt before Play and the shader-state badges. Runner pre-caches are not affected.
    bool xsemuWarmup() const;
    void setXsemuWarmup(bool on);

    QList<WarmupRun> recentWarmups() const; // newest first
    void addRecentWarmup(const WarmupRun& run);
    void clearRecentWarmups();

    // The pass running now (WarmupCheckpoint); invalid when none is, so a pass that ended cleanly is
    // never offered as one to continue.
    std::optional<WarmupCheckpoint> warmupInFlight() const;
    void setWarmupInFlight(const std::optional<WarmupCheckpoint>& pass);

    qint64 playSeconds(const QString& titleId) const;
    QDateTime lastPlayed(const QString& titleId) const;
    void addPlayTime(const QString& titleId, qint64 seconds, const QDateTime& endedAt);
    // The title's last run as the Game details page shows it: an open map (result, exitCode, at,
    // seconds, fpsAverage, stalls, worstMs, heldPeakBytes, freeMinMb), empty when it never ran.
    QVariantMap lastRun(const QString& titleId) const;
    void setLastRun(const QString& titleId, const QVariantMap& run);

    // The library's sort: "recent" (default), "name" or "size".
    QString librarySort() const;
    void setLibrarySort(const QString& sort);

signals:
    void changed(const QString& key);

private:
    void write(const char* key, const QVariant& value);
    QVariant read(const char* key, const QVariant& fallback = {}) const;
    QVariantMap readMap(const char* key) const;

    std::unique_ptr<QSettings> m_settings;
};
