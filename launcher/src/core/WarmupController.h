#pragma once

#include "AppSettings.h"
#include "Game.h"
#include "ProcessTree.h"
#include "Protocol.h"

#include <QDateTime>
#include <QElapsedTimer>
#include <QObject>
#include <QPointer>
#include <QStringList>

#include <optional>

class ProcessGate;
class RunnerManager;
struct RunnerSpec;

// Shader warm-ups: "XSEmu.exe warm-up [--full] --progress-json", its JSON progress and cancel.
// The cache is machine-wide (the emulator takes no game argument); the game picked in the UI is
// recorded with the run so the Warm-up page can say which game it was for, and its run environment
// (LaunchConfig::runEnvironment, the same as Play's) picks the translator switches and so the cache
// generation the warm-up fills.
// A game set to run with a third-party runner gets that runner's pre-cache instead: its steps
// (RunnerSpec::precache) run hidden one after another, their output read by the step's progress
// parser; a check step that finds nothing waiting, or a step whose needed files are missing, ends
// the run, and cancel uses the step's method. The runner's config file is written before each step.
//
//   warmup.start(/*full*/ false, game);
//   connect(&warmup, &WarmupController::progressChanged, ...);  // warmup.progress()
//   connect(&warmup, &WarmupController::stateChanged, ...);     // warmup.state()
//   connect(&warmup, &WarmupController::finished, ...);         // the WarmupRun just recorded
class WarmupController : public QObject {
    Q_OBJECT
public:
    enum class State { Idle, Running, PausedForRam, Finished, Cancelled, StoppedPartWay, Failed };
    Q_ENUM(State)

    struct Progress {
        qint64 targets = 0;      // warmup_start.total: kept pipelines this pass builds
        bool full = false;
        int threads = 0;
        QString phase;           // starting, giants, bound, compute, compute_blobs, rest
        qint64 done = 0;         // within the phase
        qint64 total = 0;        // within the phase
        qint64 passDone = 0;     // pipelines this pass has finished with, its finished phases included
        qint64 estimate = 0;     // runner pre-cache: a likely total while the step gives none
        double keysPerS = 0;     // the phase's average
        std::optional<qint64> etaS; // rest of the phase; absent while unknown
        std::optional<qint64> ramFreeMb;
        int giantsCompiling = 0;
        int inFlight = 0;
        qint64 failed = 0;       // all phases so far
        bool pausedForRam = false;
        double elapsedS = 0;
        QString runnerId;        // empty: XSEmu's warm-up
        int step = 0;            // runner pre-cache: 1-based step and how many
        int stepCount = 0;

        double phaseFraction() const { return total > 0 ? double(done) / double(total) : 0.0; }
    };

    static const QStringList& phases(); // in order
    static QString phaseLabel(const QString& phase);
    static int phaseIndex(const QString& phase); // 0-based, -1 unknown

    WarmupController(AppSettings& settings, ProcessGate& gate, QObject* parent = nullptr);
    ~WarmupController() override;

    void setRunners(RunnerManager* runners) { m_runners = runners; }
    // The runner whose warm-up runs for this game: its "Runs with" choice (XSEmu without a game).
    const RunnerSpec& runnerFor(const Game& game) const;
    // Why a warm-up for this game cannot start now (empty: it can); XSEmu's own warm-up never starts
    // while AppSettings::xsemuWarmup is off.
    QString checkStart(const Game& forGame = {}) const;
    bool start(bool full, const Game& forGame = {}, QString* error = nullptr);
    // Ask it to stop starting compiles; builds in flight finish (can take minutes for giants).
    void cancel();
    // Kill the warm-up tree now (what is built stays built).
    void forceStop();

    State state() const { return m_state; }
    bool isActive() const { return m_state == State::Running || m_state == State::PausedForRam; }
    bool cancelRequested() const { return m_cancelRequested; }
    Progress progress() const { return m_progress; }
    Game game() const { return m_game; }
    std::optional<Protocol::WarmupDone> doneLine() const { return m_done; }
    QString errorMessage() const { return m_error; }
    QStringList stderrTail() const { return m_stderr; }
    std::optional<WarmupRun> lastRun() const { return m_lastRun; }

    static QStringList warmupArguments(bool full);

    // What Continue does for a game: the pass a warm-up that stopped without finishing was running, so
    // that running the same command again builds only the keys it left (the emulator keeps every blob
    // it built, and a --full pass keeps its own record). Newest first: a pass the launcher did not see
    // the end of (it closed or died) beats the newest recorded run. nullopt: nothing to continue, either
    // because the newest run finished with nothing left, or because it stopped before building anything.
    struct Resume {
        bool full = false; // the pass to run again, as it was started
        qint64 built = 0;  // pipelines it had built
        qint64 left = -1;  // pipelines it left (-1: the pass did not say, as when it crashed)
        QString phase;     // the phase it stopped in; empty: unknown
    };
    static std::optional<Resume> resumable(const QList<WarmupRun>& runs, const std::optional<WarmupCheckpoint>& inFlight,
                                          const QString& titleId, const QString& runnerId, const QString& profile);
    // XSEmu's stderr of the newest warm-up, written as it runs: <launcher data>\logs\warm-up.log.
    static QString logPath();
    // Record of a finished process (tested): how the exit code, done line and last progress add up.
    static WarmupRun classify(int exitCode, bool crashed, const std::optional<Protocol::WarmupDone>& done,
                              const Progress& last, const QString& error, bool cancelRequested);

    // Screenshot fixtures only.
    void simulateForScreenshot(State state, const Game& game, const Progress& progress, const QString& error = {},
                               const std::optional<WarmupRun>& run = std::nullopt);

signals:
    void stateChanged(WarmupController::State state);
    void progressChanged();
    void finished(const WarmupRun& run);

private:
    void setState(State s);
    void onStdout();
    void onStderr();
    void onFinished(int exitCode, int exitStatus);
    void handle(const Protocol::Event& ev);
    void startRunnerStep();
    void onRunnerLine(const QString& line);
    void onRunnerStepFinished(int exitCode, int exitStatus);
    WarmupRun runnerRun(int exitCode) const;
    void finish(WarmupRun run);
    // AppSettings::WarmupInFlight: the pass running now, so a launcher that stops mid-pass still knows
    // what to continue.
    void writeCheckpoint();

    AppSettings& m_settings;
    ProcessGate& m_gate;
    QPointer<ProcessTree> m_process;
    State m_state = State::Idle;
    Game m_game;
    Progress m_progress;
    std::optional<Protocol::WarmupDone> m_done;
    std::optional<WarmupRun> m_lastRun;
    QString m_error;
    QStringList m_stderr;
    bool m_cancelRequested = false;
    QElapsedTimer m_clock;
    QElapsedTimer m_stepClock;
    RunnerManager* m_runners = nullptr;
    const RunnerSpec* m_spec = nullptr; // the runner of this pre-cache; nullptr: XSEmu's warm-up
    QString m_profile;                  // XSEmu's warm-up: the run profile identity it started with
    int m_step = 0;
    qint64 m_built = 0; // what the runner's finished steps built
    qint64 m_phaseSettled = 0; // pipelines the phases before the current one finished with
    QDateTime m_startedAt;
    QElapsedTimer m_checkpointClock; // throttles the checkpoint writes while progress ticks
};
