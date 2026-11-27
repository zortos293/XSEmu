#include "WarmupController.h"

#include "EmulatorBuild.h"
#include "HardwareInfo.h"
#include "Paths.h"
#include "ProcessGate.h"
#include "ProcessTree.h"
#include "ProgressParsers.h"
#include "Runner.h"
#include "RunnerManager.h"
#include "LaunchConfig.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QFileInfo>

namespace {
constexpr int kStderrTail = 40;
}

const QStringList& WarmupController::phases()
{
    static const QStringList p{QStringLiteral("starting"), QStringLiteral("giants"), QStringLiteral("bound"),
                               QStringLiteral("compute"), QStringLiteral("compute_blobs"), QStringLiteral("rest")};
    return p;
}

QString WarmupController::phaseLabel(const QString& phase)
{
    if (phase == QLatin1String("starting")) return QStringLiteral("Starting");
    if (phase == QLatin1String("giants")) return QStringLiteral("Giant shaders");
    if (phase == QLatin1String("bound")) return QStringLiteral("Bound pipelines");
    if (phase == QLatin1String("compute")) return QStringLiteral("Compute programs");
    if (phase == QLatin1String("compute_blobs")) return QStringLiteral("Compute blobs");
    if (phase == QLatin1String("rest")) return QStringLiteral("Other pipelines");
    return phase.isEmpty() ? QStringLiteral("Starting") : phase;
}

int WarmupController::phaseIndex(const QString& phase)
{
    return static_cast<int>(phases().indexOf(phase));
}

WarmupController::WarmupController(AppSettings& settings, ProcessGate& gate, QObject* parent)
    : QObject(parent)
    , m_settings(settings)
    , m_gate(gate)
{
    qRegisterMetaType<WarmupRun>();
}

WarmupController::~WarmupController()
{
    if (m_process && m_process->state() != QProcess::NotRunning) {
        m_process->disconnect(this);
        if (m_spec) {
            m_process->killTree();
            return;
        }
        m_process->write(Protocol::Command::cancel());
        m_process->closeWriteChannel();
        if (!m_process->waitForFinished(3000))
            m_process->killTree();
    }
}

QStringList WarmupController::warmupArguments(bool full)
{
    QStringList args{QStringLiteral("warm-up")};
    if (full)
        args << QStringLiteral("--full");
    args << QStringLiteral("--progress-json");
    return args;
}

QString WarmupController::logPath()
{
    return QDir(Paths::launcherDataDir()).filePath(QStringLiteral("logs/warm-up.log"));
}

const RunnerSpec& WarmupController::runnerFor(const Game& game) const
{
    const RunnerSpec* spec = game.isValid() ? Runners::find(m_settings.runnerFor(game.titleId())) : nullptr;
    return spec ? *spec : Runners::xsemu();
}

QString WarmupController::checkStart(const Game& forGame) const
{
    const RunnerSpec& spec = runnerFor(forGame);
    if (spec.isBuiltIn() && !m_settings.xsemuWarmup())
        return QStringLiteral("The XSEmu Core warm-up is turned off in this build.");
    if (!m_gate.isFree())
        return m_gate.blockedReason(ProcessGate::Action::StartWarmup);
    if (const ProcessGate::External other = m_gate.external(); other.busy())
        return other.reason();
    if (!spec.isBuiltIn()) {
        if (!m_runners || !m_runners->isRunnable(spec.id))
            return QStringLiteral("%1 is not installed. Install it on the Runners page, or set this game to run with XSEmu Core.").arg(spec.name);
        if (spec.precache.isEmpty())
            return QStringLiteral("%1 has no shader pre-cache; it builds what it needs while the game runs.").arg(spec.name);
        return {};
    }
    const QString emulator = m_settings.emulatorPath();
    if (!QFileInfo(emulator).isExecutable())
        return QStringLiteral("XSEmu Core was not found at %1.").arg(QDir::toNativeSeparators(emulator));
    return EmulatorBuild::probe(emulator).problem();
}

bool WarmupController::start(bool full, const Game& forGame, QString* error)
{
    const QString why = checkStart(forGame);
    if (!why.isEmpty() || !m_gate.acquire(ProcessGate::Owner::Warmup, forGame.title(), [this] { cancel(); })) {
        if (error)
            *error = why.isEmpty() ? m_gate.blockedReason(ProcessGate::Action::StartWarmup) : why;
        return false;
    }
    m_game = forGame;
    m_progress = {};
    m_progress.full = full;
    m_phaseSettled = 0;
    m_startedAt = QDateTime::currentDateTime();
    m_checkpointClock.start();
    m_done.reset();
    m_error.clear();
    m_stderr.clear();
    m_cancelRequested = false;
    m_clock.start();
    QDir().mkpath(QFileInfo(logPath()).absolutePath());
    QFile(logPath()).open(QIODevice::WriteOnly | QIODevice::Truncate);

    const RunnerSpec& spec = runnerFor(forGame);
    m_spec = spec.isBuiltIn() ? nullptr : &spec;
    m_profile = m_spec ? QString() : LaunchConfig::profileIdentity(m_settings, forGame.titleId());
    writeCheckpoint();
    if (m_spec) {
        m_progress.runnerId = m_spec->id;
        m_progress.stepCount = int(m_spec->precache.size());
        m_step = 0;
        m_built = 0;
        setState(State::Running);
        startRunnerStep();
        return true;
    }

    auto* p = new ProcessTree(this);
    m_process = p;
    p->setProcessEnvironment(LaunchConfig::runEnvironment(QProcessEnvironment::systemEnvironment(), m_settings, forGame.titleId()));
    p->setWorkingDirectory(QFileInfo(m_settings.emulatorPath()).absolutePath());
    connect(p, &QProcess::readyReadStandardOutput, this, &WarmupController::onStdout);
    connect(p, &QProcess::readyReadStandardError, this, &WarmupController::onStderr);
    connect(p, &QProcess::finished, this, [this](int code, QProcess::ExitStatus s) { onFinished(code, static_cast<int>(s)); });
    connect(p, &QProcess::errorOccurred, this, [this, p](QProcess::ProcessError e) {
        if (e == QProcess::FailedToStart) {
            m_error = QStringLiteral("XSEmu Core could not be started: %1").arg(p->errorString());
            onFinished(-1, static_cast<int>(QProcess::CrashExit));
        }
    });
    setState(State::Running);
    p->startTree(m_settings.emulatorPath(), warmupArguments(full));
    return true;
}

void WarmupController::cancel()
{
    if (!m_process || m_process->state() == QProcess::NotRunning || m_cancelRequested)
        return;
    m_cancelRequested = true;
    if (!m_spec)
        m_process->write(Protocol::Command::cancel());
    else if (m_spec->precache.at(m_step).cancel == RunnerSpec::PrecacheCancel::CloseStdin)
        m_process->closeWriteChannel();
    else
        m_process->killTree();
    emit progressChanged();
}

void WarmupController::forceStop()
{
    if (!m_process || m_process->state() == QProcess::NotRunning)
        return;
    m_cancelRequested = true;
    m_process->killTree();
}

void WarmupController::onStdout()
{
    while (m_process && m_process->canReadLine())
        handle(Protocol::parse(m_process->readLine()));
}

void WarmupController::onStderr()
{
    if (!m_process)
        return;
    const QByteArray bytes = m_process->readAllStandardError();
    QFile log(logPath());
    if (log.open(QIODevice::WriteOnly | QIODevice::Append))
        log.write(bytes);
    const QString text = QString::fromUtf8(bytes);
    for (const QString& line : text.split(QLatin1Char('\n'), Qt::SkipEmptyParts)) {
        if (m_spec)
            onRunnerLine(line.trimmed());
        m_stderr << line.trimmed();
        if (m_stderr.size() > kStderrTail)
            m_stderr.removeFirst();
    }
}

void WarmupController::handle(const Protocol::Event& ev)
{
    if (const auto* s = std::get_if<Protocol::WarmupStart>(&ev)) {
        m_progress.targets = s->total;
        m_progress.full = s->full;
        m_progress.threads = s->threads;
        m_phaseSettled = 0;
        m_progress.passDone = 0;
        emit progressChanged();
    } else if (const auto* p = std::get_if<Protocol::WarmupProgress>(&ev)) {
        // the phases run one after another, so the previous one is over when another starts: what it
        // finished with stays counted in passDone, which survives a crash the done line does not
        const bool nextPhase = !m_progress.phase.isEmpty() && p->phase != m_progress.phase;
        if (nextPhase)
            m_phaseSettled += m_progress.done;
        m_progress.passDone = m_phaseSettled + p->done;
        m_progress.phase = p->phase;
        m_progress.done = p->done;
        m_progress.total = p->total;
        m_progress.keysPerS = p->keysPerS;
        m_progress.etaS = p->etaS;
        m_progress.ramFreeMb = p->ramFreeMb;
        m_progress.giantsCompiling = p->giantsCompiling;
        m_progress.inFlight = p->inFlight;
        m_progress.failed = p->failed;
        m_progress.pausedForRam = p->pausedForRam;
        m_progress.elapsedS = m_clock.isValid() ? m_clock.elapsed() / 1000.0 : 0;
        setState(p->pausedForRam ? State::PausedForRam : State::Running);
        if (nextPhase || m_checkpointClock.elapsed() >= 10000)
            writeCheckpoint();
        emit progressChanged();
    } else if (const auto* d = std::get_if<Protocol::WarmupDone>(&ev)) {
        m_done = *d;
    } else if (const auto* e = std::get_if<Protocol::Error>(&ev)) {
        m_error = e->message;
    }
}

WarmupRun WarmupController::classify(int exitCode, bool crashed, const std::optional<Protocol::WarmupDone>& done,
                                     const Progress& last, const QString& error, bool cancelRequested)
{
    WarmupRun r;
    r.exitCode = exitCode;
    r.full = last.full;
    r.total = last.targets;
    r.finishedAt = QDateTime::currentDateTime();
    r.tookS = last.elapsedS;
    r.failed = last.failed;
    r.done = last.passDone; // a pass with no done line built no more than its last progress line said
    if (done) {
        r.done = done->built + done->failed;
        r.failed = done->failed;
        r.left = done->left;
        r.tookS = done->tookS;
        // The done line is the emulator's own account of the pass, so it decides what happened: exit
        // code 1 only says some builds failed, and those are exactly what a later pass goes on with
        if (done->cancelled)
            r.result = WarmupRun::Result::Cancelled;
        else if (!done->complete) {
            r.result = WarmupRun::Result::StoppedPartWay;
            r.stoppedForRam = last.pausedForRam;
        } else
            r.result = WarmupRun::Result::Finished;
        if (!crashed)
            return r;
    }
    if (cancelRequested && !done) {
        r.result = WarmupRun::Result::Cancelled;
        r.message = QStringLiteral("Stopped before it reported a result.");
        return r;
    }
    r.result = WarmupRun::Result::Failed;
    r.message = error;
    if (r.message.isEmpty())
        r.message = crashed ? QStringLiteral("XSEmu Core stopped unexpectedly.") : QStringLiteral("XSEmu Core ended with exit code %1.").arg(exitCode);
    return r;
}

void WarmupController::onFinished(int exitCode, int exitStatus)
{
    if (m_state != State::Running && m_state != State::PausedForRam)
        return;
    onStdout();
    onStderr();
    m_progress.elapsedS = m_clock.isValid() ? m_clock.elapsed() / 1000.0 : m_progress.elapsedS;
    if (m_error.isEmpty() && exitCode != 0 && exitCode != 4 && !m_stderr.isEmpty())
        m_error = m_stderr.last();
    finish(classify(exitCode, exitStatus == static_cast<int>(QProcess::CrashExit), m_done, m_progress, m_error, m_cancelRequested));
}

void WarmupController::finish(WarmupRun run)
{
    run.titleId = m_game.isValid() ? m_game.titleId() : QString();
    run.titleName = m_game.isValid() ? m_game.title() : QString();
    run.runnerId = m_spec ? m_spec->id : QString();
    run.profile = m_profile;
    m_lastRun = run;
    m_settings.addRecentWarmup(run);
    m_settings.setWarmupInFlight(std::nullopt);
    if (m_process) {
        m_process->disconnect(this);
        m_process->deleteLater();
        m_process = nullptr;
    }
    m_gate.release(ProcessGate::Owner::Warmup);
    switch (run.result) {
    case WarmupRun::Result::Finished:
    case WarmupRun::Result::NothingToBuild: setState(State::Finished); break;
    case WarmupRun::Result::Cancelled: setState(State::Cancelled); break;
    case WarmupRun::Result::StoppedPartWay: setState(State::StoppedPartWay); break;
    case WarmupRun::Result::Failed: setState(State::Failed); break;
    }
    emit finished(run);
}

void WarmupController::startRunnerStep()
{
    const RunnerSpec::PrecacheStep& step = m_spec->precache.at(m_step);
    const QString program = m_runners->status(m_spec->id).executable;
    const auto tokens = m_spec->tokens(m_game.folder, QDir(m_game.folder).filePath(m_game.exe), m_game.titleId(), LaunchConfig::deviceType(m_settings, m_game.titleId()), 0);
    for (const QString& need : step.needs) {
        const QFileInfo file(RunnerSpec::expand(need, tokens));
        if (file.isFile() && file.size() > 0)
            continue;
        QFile log(logPath());
        if (log.open(QIODevice::WriteOnly | QIODevice::Append))
            log.write(QStringLiteral("Skipped \"%1\": %2 is missing or empty.\n").arg(step.label, QDir::toNativeSeparators(file.filePath())).toUtf8());
        WarmupRun run = runnerRun(0);
        run.result = m_built > 0 ? WarmupRun::Result::Finished : WarmupRun::Result::NothingToBuild;
        run.message = step.nothingNote;
        finish(run);
        return;
    }
    m_progress.step = m_step + 1;
    m_progress.phase = step.label;
    m_progress.done = 0;
    m_progress.total = 0;
    m_progress.estimate = 0;
    m_progress.keysPerS = 0;
    m_progress.etaS.reset();
    m_progress.ramFreeMb = qint64(HardwareInfo::availableRam() / (1024 * 1024));
    m_stepClock.start();
    if (const QString why = m_spec->writeConfig(tokens); !why.isEmpty()) {
        m_error = why;
        onRunnerStepFinished(-1, static_cast<int>(QProcess::NormalExit));
        return;
    }

    auto* p = new ProcessTree(this);
    m_process = p;
    p->setProcessEnvironment(m_spec->processEnvironment(tokens, m_settings.runnerOptions(m_spec->id)));
    p->setWorkingDirectory(QFileInfo(program).absolutePath());
    connect(p, &QProcess::readyReadStandardOutput, this, [this, p] {
        while (p->canReadLine())
            onRunnerLine(QString::fromUtf8(p->readLine()).trimmed());
    });
    connect(p, &QProcess::readyReadStandardError, this, &WarmupController::onStderr);
    connect(p, &QProcess::finished, this, [this](int code, QProcess::ExitStatus s) { onRunnerStepFinished(code, static_cast<int>(s)); });
    connect(p, &QProcess::errorOccurred, this, [this, p](QProcess::ProcessError e) {
        if (e == QProcess::FailedToStart) {
            m_error = QStringLiteral("%1 could not be started: %2").arg(m_spec->name, p->errorString());
            onRunnerStepFinished(-1, static_cast<int>(QProcess::CrashExit));
        }
    });
    emit progressChanged();
    p->startTree(program, RunnerSpec::expand(step.args, tokens));
}

void WarmupController::onRunnerLine(const QString& line)
{
    if (line.isEmpty() || !m_spec)
        return;
    QFile log(logPath());
    if (log.open(QIODevice::WriteOnly | QIODevice::Append))
        log.write(line.toUtf8() + '\n');
    if (line.startsWith(QLatin1String("Error: ")))
        m_error = line.mid(7);
    const RunnerSpec::PrecacheStep& step = m_spec->precache.at(m_step);
    const auto u = ProgressParsers::parse(step.parser, line);
    if (!u)
        return;
    if (u->total >= 0)
        m_progress.total = u->total;
    if (u->done >= 0)
        m_progress.done = u->done;
    if (u->failed >= 0)
        m_progress.failed = u->failed;
    if (u->estimate > 0)
        m_progress.estimate = u->estimate;
    if (step.label.isEmpty() && !u->phase.isEmpty())
        m_progress.phase = u->phase;
    const double stepS = m_stepClock.elapsed() / 1000.0;
    m_progress.keysPerS = stepS > 1 ? m_progress.done / stepS : 0;
    if (m_progress.keysPerS > 0 && m_progress.total > m_progress.done)
        m_progress.etaS = qint64((m_progress.total - m_progress.done) / m_progress.keysPerS + 0.5);
    else
        m_progress.etaS.reset();
    m_progress.ramFreeMb = qint64(HardwareInfo::availableRam() / (1024 * 1024));
    m_progress.elapsedS = m_clock.elapsed() / 1000.0;
    emit progressChanged();
}

void WarmupController::onRunnerStepFinished(int exitCode, int exitStatus)
{
    if (!m_spec || (m_state != State::Running && m_state != State::PausedForRam))
        return;
    if (m_process) {
        while (m_process->canReadLine())
            onRunnerLine(QString::fromUtf8(m_process->readLine()).trimmed());
        onStderr();
    }
    m_progress.elapsedS = m_clock.elapsed() / 1000.0;
    const RunnerSpec::PrecacheStep& step = m_spec->precache.at(m_step);
    const bool crashed = exitStatus == static_cast<int>(QProcess::CrashExit);
    WarmupRun run = runnerRun(exitCode);
    if (!crashed && exitCode == 0)
        m_built += qMax<qint64>(0, m_progress.done);
    if (m_cancelRequested) {
        run.result = WarmupRun::Result::Cancelled;
    } else if (crashed || exitCode != 0) {
        run.result = WarmupRun::Result::Failed;
        if (m_error.isEmpty() && !m_stderr.isEmpty())
            m_error = m_stderr.last();
        run.message = !m_error.isEmpty() ? m_error
                    : crashed           ? QStringLiteral("%1 stopped unexpectedly.").arg(m_spec->name)
                                        : QStringLiteral("%1 ended with exit code %2.").arg(m_spec->name).arg(exitCode);
    } else if (step.check && m_progress.total <= 0) {
        run.result = WarmupRun::Result::NothingToBuild;
        run.message = step.nothingNote.isEmpty() ? QStringLiteral("Nothing is waiting to be built.") : step.nothingNote;
    } else if (m_step + 1 < m_spec->precache.size()) {
        if (m_process) {
            m_process->disconnect(this);
            m_process->deleteLater();
            m_process = nullptr;
        }
        ++m_step;
        startRunnerStep();
        return;
    } else if (m_built <= 0 && run.tookS < 2) {
        run.result = WarmupRun::Result::NothingToBuild;
        run.message = QStringLiteral("%1 finished without building anything; its log may say why.").arg(m_spec->name);
    }
    finish(run);
}

WarmupRun WarmupController::runnerRun(int exitCode) const
{
    WarmupRun run;
    run.exitCode = exitCode;
    run.done = m_progress.done;
    run.total = m_progress.total;
    run.failed = m_progress.failed;
    run.tookS = m_progress.elapsedS;
    run.finishedAt = QDateTime::currentDateTime();
    return run;
}

void WarmupController::writeCheckpoint()
{
    WarmupCheckpoint pass;
    pass.titleId = m_game.isValid() ? m_game.titleId() : QString();
    pass.titleName = m_game.isValid() ? m_game.title() : QString();
    pass.runnerId = m_spec ? m_spec->id : QString();
    pass.profile = m_profile;
    pass.full = m_progress.full;
    pass.built = m_progress.passDone;
    pass.phase = m_progress.phase;
    pass.startedAt = m_startedAt;
    m_settings.setWarmupInFlight(pass);
    m_checkpointClock.restart();
}

std::optional<WarmupController::Resume> WarmupController::resumable(const QList<WarmupRun>& runs,
                                                                    const std::optional<WarmupCheckpoint>& inFlight,
                                                                    const QString& titleId, const QString& runnerId,
                                                                    const QString& profile)
{
    // a pass the launcher did not see the end of: it closed or died, so no run recorded it
    if (inFlight && inFlight->isValid() && inFlight->titleId == titleId && inFlight->runnerId == runnerId
        && (profile.isEmpty() || inFlight->profile.isEmpty() || inFlight->profile == profile))
        return Resume{inFlight->full, inFlight->built, -1, inFlight->phase};
    for (const WarmupRun& r : runs) {
        if (r.titleId != titleId || r.runnerId != runnerId)
            continue;
        if (!profile.isEmpty() && !r.profile.isEmpty() && r.profile != profile)
            return std::nullopt; // the run profile changed, so its keys are not this profile's keys
        switch (r.result) {
        case WarmupRun::Result::Cancelled:
        case WarmupRun::Result::StoppedPartWay:
            return Resume{r.full, r.done, r.left, QString()};
        case WarmupRun::Result::Failed:
            // a refusal before the pass started (another process, no device) left nothing to continue
            if (r.done <= 0)
                return std::nullopt;
            return Resume{r.full, r.done, r.left, QString()};
        case WarmupRun::Result::Finished:
            // a pass that went through everything can still leave builds it could not make
            if (r.left > 0)
                return Resume{r.full, r.done, r.left, QString()};
            return std::nullopt;
        case WarmupRun::Result::NothingToBuild:
            return std::nullopt;
        }
    }
    return std::nullopt;
}

void WarmupController::setState(State s)
{
    if (m_state == s)
        return;
    m_state = s;
    emit stateChanged(s);
}

void WarmupController::simulateForScreenshot(State state, const Game& game, const Progress& progress, const QString& error,
                                             const std::optional<WarmupRun>& run)
{
    m_game = game;
    m_progress = progress;
    m_error = error;
    m_lastRun = run;
    const bool active = state == State::Running || state == State::PausedForRam;
    if (active && m_gate.isFree())
        m_gate.acquire(ProcessGate::Owner::Warmup, game.title(), [this] { simulateForScreenshot(State::Cancelled, m_game, m_progress); });
    if (!active)
        m_gate.release(ProcessGate::Owner::Warmup);
    setState(state);
    emit progressChanged();
}
