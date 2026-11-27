#pragma once

#include "Release.h"

#include <QDateTime>
#include <QObject>
#include <QString>
#include <QUrl>

#include <atomic>
#include <functional>

class AppSettings;
class ProcessGate;
class QThread;

// Updates of the launcher and of the bundled XSEmu Core, separately, from the releases of
// zortos293/XSEmu. A release publishes xsemu-manifest.json with one entry per part: a version, one
// asset and its SHA-256. The two parts update independently: a Core update replaces the files in
// <launcher dir>\core and needs no new launcher, and a launcher update replaces the launcher's own
// files and restarts it.
//
// Safety rules it enforces: nothing downloads or replaces anything while a game or a warm-up runs
// (ProcessGate), every asset is checked against its SHA-256 before it is used, the Core's new program
// is asked --version before it is put in place, and the previous Core is kept in core.old until the
// new one has started (noteEmulatorStarted) or the user restores it.
//
//   connect(&updates, &UpdateService::changed, this, [&]{ show(updates.status()); });
//   updates.check(true);           // asynchronously; changed() follows
//   updates.install(UpdateService::Part::Core);
class UpdateService : public QObject {
    Q_OBJECT
public:
    static constexpr const char* kRepository = "zortos293/XSEmu";
    static constexpr const char* kApiReleases = "https://api.github.com/repos/zortos293/XSEmu/releases?per_page=20";
    static constexpr const char* kManifestAsset = "xsemu-manifest.json";
    static constexpr const char* kCoreVersionFile = "version.json";
    static constexpr const char* kPreviousCore = "core.old";
    static constexpr const char* kUpdateDir = "update";

    enum class Part { Launcher, Core };
    enum class State {
        Idle,        // nothing asked yet
        Checking,
        Downloading, // an install
        Applying,    // unpacking, checking, putting it in place
        UpToDate,
        Available,
        Failed,
    };

    struct Status {
        State state = State::Idle;
        QString message;   // one line: what the section shows
        QString error;     // why it failed
        QString launcherVersion;
        QString installedCore; // the version file's answer; empty when unknown
        QString corePath;      // the program in use
        bool coreBundled = false;   // it is the launcher's own core\, so a Core update applies
        bool corePreviousKept = false; // core.old is there and can be restored
        bool launcherUpdate = false;
        bool coreUpdate = false;
        ReleaseInfo release;   // the newest release that offers an update
        qint64 done = -1;      // bytes, while downloading
        qint64 total = -1;
        QDateTime checkedAt;
        bool busy() const { return state == State::Checking || state == State::Downloading || state == State::Applying; }
        bool updateAvailable() const { return launcherUpdate || coreUpdate; }
    };

    // One HTTPS GET into memory, and one into a file. The defaults use Qt Network on the calling
    // (worker) thread; tests replace them so a check and an install run against local files.
    using Fetch = std::function<QByteArray(const QUrl&, QString* error, int timeoutMs)>;
    using Download = std::function<bool(const QUrl&, const QString& file, const std::function<bool(qint64, qint64)>& progress, QString* error)>;

    explicit UpdateService(AppSettings& settings, ProcessGate& gate, QObject* parent = nullptr);
    ~UpdateService() override;

    // The folders it works in: the launcher's own folder and its core\. Defaults to Paths.
    void setDirectories(const QString& appDir, const QString& coreDir);
    QString appDir() const { return m_appDir; }
    QString coreDir() const { return m_coreDir; }
    void setTransport(Fetch fetch, Download download);

    Status status() const { return m_status; }
    QString launcherVersion() const;      // XSEMU_VERSION
    QString installedCoreVersion() const; // core\version.json, else the emulator's kept --version answer

    // Why an install cannot start now (a game or a warm-up running, a part outside the launcher's
    // folder, a read-only install), or empty. A check needs none of this.
    bool canInstall(Part part, QString* why = nullptr) const;
    void check(bool manual = true);
    void install(Part part);
    void cancel();

    // The emulator answered the control pipe with the new Core: its previous copy is no longer needed.
    void noteEmulatorStarted();
    // Put core.old back (the user's way out of a Core that does not work).
    bool restorePreviousCore(QString* error);
    QString previousCoreDir() const;

    // Re-read the installed versions and the kept previous copy (no network).
    void refresh();

    // Screenshot fixtures only: show this status instead of the real one until refresh() is asked to
    // take it back (a screenshot state needs an update available without a network).
    void simulateForScreenshot(const Status& status);

signals:
    void changed();
    // A launcher update is staged and its updater script is running: quit so the script can replace
    // the launcher's files and start it again.
    void launcherUpdateReady();

private:
    // What the worker may read: QSettings belongs to the main thread.
    struct CheckInputs {
        QString launcher;
        QString core;
        bool prereleases = false;
    };

    void startWorker(std::function<void()> job);
    void releaseGate();
    void post(std::function<void()> f);
    void finishFailed(const QString& error);
    void setState(State state, const QString& message);

    void runCheck(const CheckInputs& in);
    void runInstall(Part part, const ReleasePart& releasePart);
    bool applyCore(const QString& staged, const ReleasePart& part, const QString& updateDir, QString* error);
    QString writeUpdaterScript(QString* error) const;

    AppSettings& m_settings;
    ProcessGate& m_gate;
    QString m_appDir;
    QString m_coreDir;
    Fetch m_fetch;
    Download m_download;
    Status m_status;
    bool m_manual = false;
    bool m_simulated = false;
    QThread* m_thread = nullptr;
    std::atomic<bool> m_cancel{false};
};
