#pragma once

#include <QHash>
#include <QList>
#include <QProcessEnvironment>
#include <QString>
#include <QStringList>

// A runner is a program that can run a game for the launcher: XSEmu itself, or a third-party port that
// we start as a separate program (we never link or copy their code). The list is curated and compiled
// in: every third-party runner is pinned to one release asset (with its SHA-256) or one commit, and
// "update available" only ever means the pin in this list changed.
//
//   const RunnerSpec* r = Runners::find(QStringLiteral("victory-dexnom"));
//   QStringList args = r->expand(r->launchArgs, {{"gameDir", game.folder()}});
struct RunnerSpec {
    enum class Vendor { Any, Amd, Nvidia };
    enum class Install { BuiltIn, ReleaseAsset, BuildFromSource };
    // How a run is asked to stop before the tree is killed: our control pipe, WM_CLOSE to its
    // top-level window, WM_SYSCOMMAND SC_CLOSE to it (for programs that ignore WM_CLOSE), or nothing.
    // On X11 both window requests are WM_DELETE_WINDOW.
    enum class Stop { ControlPipe, CloseWindow, SysCommandClose, Kill };
    // How a pre-cache run is cancelled before it is killed: kill at once, or close its stdin (it
    // drains, saves and exits).
    enum class PrecacheCancel { Kill, CloseStdin };
    // How its game window ends up in the in-game view: it opens itself inside our host (--host-hwnd),
    // we reparent its top-level window into the host, or it stays a separate window with our bar docked.
    enum class Embed { HostHandle, Reparent, Docked };

    QString id;
    QString name;            // "VIctory (7900 XTX)"
    QString repo;            // "owner/name" on GitHub; empty for XSEmu
    QString badge;           // "AMD", "Vendor-neutral · tuned on RTX 5080"; empty: none
    Vendor recommendFor = Vendor::Any; // the GPU vendor the recommendation picks it for
    QString recommendWhy;    // why, as a clause: "it was tuned on an RTX 5080"
    QString summary;         // one line for the picker
    QString licence;         // SPDX id of the runner's licence
    QString note;            // behaviour the user must know about (licence handling); empty: none
    bool windowsOnly = false; // ships for Windows only

    Install install = Install::BuiltIn;
    QString releaseTag;      // ReleaseAsset: the pinned tag
    QString assetName;       //   its asset file name (zip or tar.gz)
    QString assetSha256;     //   lowercase hex of the asset
    QString assetUrl;        //   download URL; empty: the GitHub release URL (a feed may point elsewhere)
    QList<QPair<QString, QString>> fileSha256; // files inside the unpacked asset that must match
    QList<QPair<QString, QString>> copyFiles;  // {from, to} inside the unpacked asset
    QString commit;          // BuildFromSource: the pinned commit SHA
    QString tree;            //   its tree id, checked after the checkout; empty: the commit alone is checked
    QStringList remotes;     //   fetched from in order until one has the pin; empty: cloneUrl()
    QStringList toolchains;  //   bootstrapped first, in order: "git", "msvc" (VS C++ build tools), "rust"
    struct Fetch {
        QString url;
        QString sha256;
        QString to;          // relative to the checkout
    };
    QList<Fetch> fetches;    //   files the build needs that the repository does not carry
    QList<QStringList> buildSteps; // commands run in the checkout, each {program, args...}
    QList<QPair<QString, QString>> artifacts; // {built file relative to the target dir, file in bin}
    QString executable;      // the runner's program, relative to its install dir
    QString buildNote;       // shown before a source build starts

    // Tokens in arguments and environment: {gameDir} {gameExe} {gameExeDir} {titleId} {console}
    // ("series-s"/"series-x") {runnerDir} (runners\<id>) {installDir} {cacheDir} {jobs} {rtFlag} {hostHwnd}.
    QStringList launchArgs;
    QList<QPair<QString, QString>> environment; // set for runs and pre-cache runs
    QStringList clearEnvironment; // never passed on from the launcher's own environment
    // A switch on the runner's card that adds environment to its runs and pre-cache runs while on.
    struct Option {
        QString key;         // "ambient", as AppSettings::runnerOptions keeps it
        QString label;       // "Author's ambient light"
        QString description; // one line under it
        bool on = false;     // the default
        QList<QPair<QString, QString>> environment;
    };
    QList<Option> options;
    struct PrecacheStep {
        QStringList args;
        QString parser;      // ProgressParsers id
        bool check = false;  // prints how much is waiting; 0 skips the steps after it
        PrecacheCancel cancel = PrecacheCancel::Kill;
        QString label;       // what the step does, for the Warm-up page ("Building pipelines")
        QStringList needs;   // files (tokens allowed) that must exist and be non-empty; else this step and the rest are skipped
        QString nothingNote; // why there is nothing to do when the check finds nothing or needs are missing
    };
    QList<PrecacheStep> precache; // empty: the runner has no shader pre-cache
    QString precacheNote;    // what to know before a pre-cache (when it has work, what it needs)
    quint64 recommendMinRamBytes = 0; // the recommendation needs this much RAM to pick it
    QString cacheDir;        // its shader/pipeline cache (tokens allowed); empty: unknown
    // A config file written as TOML before every run and every pre-cache step, so the program finds
    // it wherever the runner is now: [paths] key = value for each entry (tokens allowed).
    struct ConfigPath {
        QString key;         // "cache_dir"
        QString value;       // "{runnerDir}/cache"
        bool create = false; // the folder is made before the run
    };
    QString configFile;      // tokens allowed; empty: none
    QList<ConfigPath> configPaths;
    QString windowReadyLine; // output line after which its game window may be touched; empty: when visible
    Stop stop = Stop::CloseWindow;
    int stopGraceMs = 5000;
    Embed embed = Embed::Docked;
    bool canPause = false;
    bool canOverlay = false;

    bool isBuiltIn() const { return install == Install::BuiltIn; }
    // The curated version string: the release tag, or the first 7 characters of the commit.
    QString pinnedVersion() const;
    // True when the list pins something installable on this OS (a tag with asset and hash, or a commit).
    bool isPinned() const;
    QString downloadUrl() const; // ReleaseAsset: the GitHub release download URL
    QString cloneUrl() const;    // BuildFromSource: https://github.com/<repo>.git
    QStringList fetchUrls() const; // remotes, else {cloneUrl()}
    QString dataDir() const;     // runners\<id>: install record, app\, and the runner's own data
    QString installDir() const;  // runners\<id>\app (release) or runners\<id>\bin (built)
    QString checkoutDir() const; // runners\<id>\s: the source checkout (short: deep paths break some builds)
    QString targetDir() const;   // runners\<id>\t: CARGO_TARGET_DIR
    // Values for every token, for one game (gameExe may be empty for runner-wide commands).
    QList<QPair<QString, QString>> tokens(const QString& gameDir, const QString& gameExe, const QString& titleId,
                                          const QString& deviceType, quintptr hostHwnd) const;
    // The cache dir with its tokens filled in; empty when unknown.
    QString resolvedCacheDir() const;
    // The system environment less clearEnvironment, plus the spec's and that of each option on (choices
    // from AppSettings::runnerOptions, else its default), tokens filled in; creates the *_DIR folders it names.
    QProcessEnvironment processEnvironment(const QList<QPair<QString, QString>>& tokens, const QHash<QString, bool>& choices) const;
    static bool optionOn(const Option& option, const QHash<QString, bool>& choices) { return choices.value(option.key, option.on); }
    // Writes configFile atomically and makes the folders it names; returns why it could not (empty: written or none).
    QString writeConfig(const QList<QPair<QString, QString>>& tokens) const;

    // Replace {token} in each argument with values[token]; unknown tokens stay as written.
    static QStringList expand(const QStringList& args, const QList<QPair<QString, QString>>& values);
    static QString expand(const QString& arg, const QList<QPair<QString, QString>>& values);
};

namespace Runners {

inline constexpr const char* XSEmu = "xsemu";

const QList<RunnerSpec>& curated();
const RunnerSpec* find(const QString& id);
const RunnerSpec& xsemu();

// PCI vendor ids as DXGI and Linux report them.
inline constexpr quint32 kVendorNvidia = 0x10DE;
inline constexpr quint32 kVendorAmd = 0x1002;
inline constexpr quint32 kVendorIntel = 0x8086;
QString vendorName(quint32 vendorId); // "NVIDIA", "AMD", "Intel", "" when unknown

struct Recommendation {
    QString runnerId;
    QString reason; // a human sentence: why this runner for this PC
};

// The first curated runner recommended for this GPU's vendor (RunnerSpec::recommendFor) that this
// PC has the RAM for and that is installed or installable; otherwise XSEmu. installable(id) says whether a runner is installed
// already or pinned so it can be installed. The reason always says why.
template <typename Installable>
Recommendation recommend(quint32 gpuVendorId, const QString& gpuName, quint64 ramBytes, Installable installable)
{
    const QString gpu = gpuName.isEmpty() ? QStringLiteral("this PC's GPU") : gpuName;
    const RunnerSpec::Vendor want = gpuVendorId == kVendorAmd      ? RunnerSpec::Vendor::Amd
                                    : gpuVendorId == kVendorNvidia ? RunnerSpec::Vendor::Nvidia
                                                                   : RunnerSpec::Vendor::Any;
    if (want != RunnerSpec::Vendor::Any) {
        for (const RunnerSpec& r : curated()) {
            if (r.recommendFor == want && ramBytes >= r.recommendMinRamBytes && installable(r.id))
                return {r.id, QStringLiteral("Recommended for %1: %2.").arg(gpu, r.recommendWhy)};
        }
        return {QLatin1String(XSEmu), QStringLiteral("No runner for %1 GPUs can be installed on this PC yet, so XSEmu Core runs games on %2.").arg(vendorName(gpuVendorId), gpu)};
    }
    if (gpuVendorId == 0)
        return {QLatin1String(XSEmu), QStringLiteral("The GPU could not be identified, so XSEmu Core is the safe choice.")};
    return {QLatin1String(XSEmu), QStringLiteral("XSEmu Core supports %1; the other runners are tuned for AMD or NVIDIA GPUs.").arg(gpu)};
}

}
