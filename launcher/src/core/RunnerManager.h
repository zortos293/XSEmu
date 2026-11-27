#pragma once

#include "Runner.h"

#include <QDateTime>
#include <QHash>
#include <QObject>
#include <QPointer>
#include <QSet>

#include <functional>
#include <optional>

class AppSettings;
class RunnerInstall;

// What is installed under runners\<id>\ (Paths::runnerDir: %LOCALAPPDATA%\XSEmu\runners by default,
// or where the user put the Runners folder or moved one runner) and the actions on it. One install,
// update or move runs at a time; an install is a RunnerInstall task whose progress the UI shows and
// can cancel.
//
//   RunnerManager& m = services.runners;
//   if (m.status(id).state == RunnerManager::State::NotInstalled) m.install(id);
//   connect(&m, &RunnerManager::changed, this, [&](const QString& id) { refresh(m.status(id)); });
class RunnerManager : public QObject {
    Q_OBJECT
public:
    enum class State { BuiltIn, Unavailable, NotInstalled, Installing, Moving, Uninstalling, Installed, UpdateAvailable, Failed };

    struct Progress {
        QString step;      // "Downloading", "Checking the download", "Unpacking", "Building", "Moving"...
        int stepIndex = 0; // 1-based
        int stepCount = 0;
        qint64 done = -1;  // within the step; -1: unknown (indeterminate)
        qint64 total = -1;
        bool bytes = false; // done/total are bytes
        QString detail;    // the last line or item the step reported; Moving: the folder it moves to
        int phase = 0;     // 0 download (and toolchains), 1 unpack or build, 2 install
        QDateTime startedAt;
    };

    struct Status {
        State state = State::NotInstalled;
        QString installedVersion; // what runner.json says is installed
        QString pinnedVersion;    // what the curated list pins
        QString executable;       // absolute path of the installed program; empty when not installed
        Progress progress;        // Installing, Moving
        QString error;            // why the last install stopped (Failed, or a failed update of an installed copy)
    };

    explicit RunnerManager(QObject* parent = nullptr);
    ~RunnerManager() override;

    // The folders the user chose (AppSettings::runnersFolder, runnerLocations) decide where runners
    // live (Paths), now and whenever they change. Without settings, runners stay where Paths says.
    void setSettings(AppSettings* settings);
    // Why a runner is in use outside this manager (a game or a pre-cache running with it); empty: not.
    void setInUseCheck(std::function<QString(const QString& id)> check);

    Status status(const QString& id) const;
    bool isRunnable(const QString& id) const; // built in, or installed (an update may be pending)
    // Installed or installable now: what the recommendation may pick.
    bool isInstallable(const QString& id) const;
    QString busyId() const; // the runner being installed, or empty
    // Why the runner's files can't be moved or deleted now (installing or building, being moved or
    // uninstalled, running a game or a pre-cache), or empty.
    QString busyReason(const QString& id) const;

    // Install, or update to the curated pin. Returns false with *error when it cannot start now.
    bool install(const QString& id, QString* error = nullptr);
    void cancel();
    // Delete the runner's folder on a worker thread (state Uninstalling meanwhile; uninstalled()
    // follows). Refused with *error while busyReason() says why, a running game or pre-cache included.
    bool uninstall(const QString& id, QString* error = nullptr);

    // The folder that holds the runner's own folder (Paths::runnerLocation).
    QString location(const QString& id) const;
    enum class Relocation { Move, Rebuild };
    // Keep the runner in <folder>\<id> from now on: its checkout, build folder, program, data and cache.
    // Move copies its folder there (a rename on the same drive), checks file count, sizes and the
    // program, then deletes the old folder; the state is Moving meanwhile and relocated() follows.
    // Rebuild installs afresh there and deletes the old folder once that succeeds; if it fails the
    // runner stays where it was. Refuses with *error while busyReason() says why.
    bool relocate(const QString& id, const QString& folder, Relocation how, QString* error = nullptr);
    // Change the Runners folder (empty: the default). Runners already kept in the current one stay
    // there; Change location moves them. Refused while an install or move runs (they use it).
    bool setRunnersFolder(const QString& folder, QString* error = nullptr);
    // Move from into to as relocate does, on the calling thread: rename when allowed, else copy and
    // verify (file count, total bytes, and the program exeRel, relative to from), then delete from.
    // *note says what was left behind when the old folder could not be deleted.
    static bool moveFolder(const QString& from, const QString& to, const QString& exeRel, QString* error, QString* note,
                           const std::function<void(qint64 done, qint64 total)>& progress = {}, bool allowRename = true);

    QString cachePath(const QString& id) const;
    // Bytes the installed copy takes on disk (its install dir), measured on a worker thread: -1 until
    // the measurement lands and changed(id) follows; 0 for the built-in runner.
    qint64 installedBytes(const QString& id);
    QDateTime installedAt(const QString& id) const;

    // Screenshot fixtures only: report this status for the runner instead of the real one.
    void simulateForScreenshot(const QString& id, const Status& status);

    QString installLogPath(const QString& id) const; // <runner>\install.log of the last install run

signals:
    // Anything about the runner changed, install and move progress included.
    void changed(const QString& id);
    // Its install state may have changed (an install or move started or ended, an uninstall, a new
    // location): what pages that only show Installed or not follow, without a refresh per progress line.
    void stateChanged(const QString& id);
    void installFinished(const QString& id, bool ok, const QString& error);
    // A relocate() ended; note or error says what to tell the user (empty when there is nothing).
    void relocated(const QString& id, bool ok, const QString& message);
    // An uninstall() ended; error says what is left when files were in use.
    void uninstalled(const QString& id, bool ok, const QString& error);

private:
    void applyLocations();
    void setLocation(const QString& id, const QString& folder);
    void finishMove(bool ok, const QString& message);

    // A pointer that clears itself: the destructor rolls a move-and-rebuild back through the settings,
    // and that must not depend on `Services` destroying its settings after this manager.
    QPointer<AppSettings> m_settings;
    std::function<QString(const QString&)> m_inUse;
    QPointer<RunnerInstall> m_task;
    QString m_taskId;
    QHash<QString, QString> m_errors;
    QHash<QString, Status> m_simulated;
    QHash<QString, qint64> m_sizes; // -1 while measuring
    struct Move {
        QString id;
        QString folder; // the new location
        QString to;     // <folder>\<id>
        qint64 done = 0;
        qint64 total = -1;
        QDateTime startedAt;
    };
    std::optional<Move> m_move;
    struct Rebuild {
        QString id;
        QString previous; // the location setting before (empty: the Runners folder)
        QString oldDir;   // the runner's folder before
    };
    std::optional<Rebuild> m_rebuild;
    QSet<QString> m_removing; // being uninstalled
};
