// EmulatorController and WarmupController against fake-xsemu: real processes, real pipe, real JSON.
#include "AppSettings.h"
#include "DevFixtures.h"
#include "EmulatorBuild.h"
#include "EmulatorController.h"
#include "LaunchConfig.h"
#include "Library.h"
#include "Paths.h"
#include "ProcessGate.h"
#include "Runner.h"
#include "RunnerManager.h"
#include "TitleConfig.h"
#include "WarmupController.h"

#include <QDebug>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QScopeGuard>
#include <QSignalSpy>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QTest>

#include <atomic>
#include <chrono>
#include <future>

class TestControllers : public QObject {
    Q_OBJECT
private slots:
    void initTestCase();
    void init();
    void cleanup();
    void warmupFinishesThroughEveryPhase();
    void warmupCancel();
    void warmupFailure();
    void warmupFailureKeepsWhatItBuilt();
    void warmupContinueBuildsOnlyWhatIsLeft();
    void warmupFinishedWithLeftoversIsNotAFailure();
    void warmupResumeSurvivesTheLauncherClosing();
    void warmupRefused();
    void runnerPrecacheRunsEachStep();
    void runnerPrecacheCancelClosesStdin();
    void victoryConfigWrittenForPrecacheAndLaunch();
    void xtxWarmWaitsForThePipelineBook();
    void kelvaroqNothingToBuildYet();
    void silentPrecacheIsNotFinished();
    void coolfoxRunsWithItsOwnEnvironment();
    void gameRunPauseResumeStop();
    void playAndWarmupShareTheRunEnvironment();
    void gameCrash();
    void gameFilesMissing();
    void oneProcessAtATime();
    void emulatorTooOldIsNeverStarted();
    void busyMarkerBlocksPlayAndWarmup();
    void runningEmulatorProcessBlocks();
    void probeChildIsNotAnotherEmulator();

private:
    Game fixtureGame(const QString& title) const;
    void installFakeKelvaroq() const { installFakeRunner(QStringLiteral("kelvaroq")); }
    void installFakeRunner(const QString& id) const;
    WarmupRun precache(const QString& id, const Game& game, QStringList* steps = nullptr);
    QTemporaryDir m_fixtures;
    std::unique_ptr<QTemporaryDir> m_dir;
    std::unique_ptr<AppSettings> m_settings;
    std::unique_ptr<ProcessGate> m_gate;
    GameList m_games;
};

void TestControllers::initTestCase()
{
    qputenv("XSEMU_EMULATOR", XSEMU_FAKE_EMULATOR_PATH);
    qputenv("FAKE_XSEMU_TICK_MS", "5");
    // Kelvaroq at its curated pin on any OS, played by a copy of fake-xsemu (read before the list is first used)
    qputenv("XSEMU_RUNNERS_DIR", m_fixtures.filePath(QStringLiteral("runners")).toUtf8());
    QFile feed(m_fixtures.filePath(QStringLiteral("feed.json")));
    QVERIFY(feed.open(QIODevice::WriteOnly));
    feed.write(R"({"kelvaroq": {"keep": true}, "victory-dexnom": {"keep": true}, "victory-7900xtx": {"keep": true}, "coolfox": {"keep": true}})");
    feed.close();
    qputenv("XSEMU_RUNNER_FEED", feed.fileName().toUtf8());
    QVERIFY(DevFixtures::materialize(QStringLiteral(XSEMU_FIXTURES_DIR "/games"), m_fixtures.filePath(QStringLiteral("games"))));
    m_games = Library::scan({m_fixtures.filePath(QStringLiteral("games"))}, 3, false);
    QCOMPARE(m_games.size(), 4);
}

void TestControllers::init()
{
    m_dir = std::make_unique<QTemporaryDir>();
    qputenv("XCL_DATA_ROOT", m_dir->filePath(QStringLiteral("xcl")).toUtf8());
    qputenv("FAKE_XSEMU_SCENARIO", "ok");
    m_settings = std::make_unique<AppSettings>(m_dir->filePath(QStringLiteral("settings.ini")));
    m_settings->setLaunchNeedMb(0);
    m_settings->setXsemuWarmup(true);
    m_gate = std::make_unique<ProcessGate>();
}

void TestControllers::cleanup()
{
    m_gate.reset();
    m_settings.reset();
    m_dir.reset();
}

Game TestControllers::fixtureGame(const QString& title) const
{
    for (const Game& g : m_games) {
        if (g.title() == title)
            return g;
    }
    return {};
}

void TestControllers::installFakeRunner(const QString& id) const
{
    const RunnerSpec* spec = Runners::find(id);
    QVERIFY(spec);
    const QString exe = QDir(spec->installDir()).filePath(spec->executable);
    QDir().mkpath(QFileInfo(exe).absolutePath());
    QFile::remove(exe);
    QVERIFY(QFile::copy(QStringLiteral(XSEMU_FAKE_EMULATOR_PATH), exe));
    QFile rec(QDir(spec->dataDir()).filePath(QStringLiteral("runner.json")));
    QVERIFY(rec.open(QIODevice::WriteOnly | QIODevice::Truncate));
    rec.write(QJsonDocument(QJsonObject{{QStringLiteral("id"), spec->id}, {QStringLiteral("version"), spec->pinnedVersion()}, {QStringLiteral("executable"), spec->executable}}).toJson());
}

void TestControllers::runnerPrecacheRunsEachStep()
{
    installFakeKelvaroq();
    const Game game = fixtureGame(QStringLiteral("Signal Garden"));
    m_settings->setGameOverride(game.titleId(), QStringLiteral("runner"), QStringLiteral("kelvaroq"));
    RunnerManager runners;
    WarmupController w(*m_settings, *m_gate);
    w.setRunners(&runners);
    QCOMPARE(w.runnerFor(game).id, QStringLiteral("kelvaroq"));
    QVERIFY2(w.checkStart(game).isEmpty(), qPrintable(w.checkStart(game)));
    QStringList steps;
    connect(&w, &WarmupController::progressChanged, this, [&] {
        if (steps.isEmpty() || steps.last() != w.progress().phase)
            steps << w.progress().phase;
    });
    QSignalSpy finished(&w, &WarmupController::finished);
    QString error;
    QVERIFY2(w.start(false, game, &error), qPrintable(error));
    QCOMPARE(m_gate->owner(), ProcessGate::Owner::Warmup);
    QVERIFY2(finished.wait(20000), qPrintable(w.stderrTail().join(QLatin1Char('|'))));
    QCOMPARE(w.state(), WarmupController::State::Finished);
    QCOMPARE(steps, (QStringList{QStringLiteral("Checking what to build"), QStringLiteral("Building pipelines")}));
    const WarmupRun run = *w.lastRun();
    QCOMPARE(run.runnerId, QStringLiteral("kelvaroq"));
    QCOMPARE(run.done, 812);
    QCOMPARE(run.total, 812);
    QCOMPARE(run.exitCode, 0);
    QVERIFY(m_gate->isFree());
    QCOMPARE(m_settings->recentWarmups().first().runnerId, QStringLiteral("kelvaroq"));
}

void TestControllers::runnerPrecacheCancelClosesStdin()
{
    qputenv("FAKE_XSEMU_TICK_MS", "40");
    installFakeKelvaroq();
    const Game game = fixtureGame(QStringLiteral("Signal Garden"));
    m_settings->setGameOverride(game.titleId(), QStringLiteral("runner"), QStringLiteral("kelvaroq"));
    RunnerManager runners;
    WarmupController w(*m_settings, *m_gate);
    w.setRunners(&runners);
    QSignalSpy finished(&w, &WarmupController::finished);
    QVERIFY(w.start(false, game));
    QTRY_VERIFY_WITH_TIMEOUT(w.progress().step == 2 && w.progress().done > 0, 10000);
    w.cancel();
    QVERIFY(finished.wait(10000));
    QCOMPARE(w.state(), WarmupController::State::Cancelled);
    QCOMPARE(w.lastRun()->exitCode, 0); // closing stdin let it stop on its own; nothing was killed
    QVERIFY(w.lastRun()->done < 812);
    qputenv("FAKE_XSEMU_TICK_MS", "5");
}

WarmupRun TestControllers::precache(const QString& id, const Game& game, QStringList* steps)
{
    m_settings->setGameOverride(game.titleId(), QStringLiteral("runner"), id);
    RunnerManager runners;
    WarmupController w(*m_settings, *m_gate);
    w.setRunners(&runners);
    connect(&w, &WarmupController::progressChanged, this, [&] {
        if (steps && (steps->isEmpty() || steps->last() != w.progress().phase))
            *steps << w.progress().phase;
    });
    QSignalSpy finished(&w, &WarmupController::finished);
    QString error;
    if (!w.start(false, game, &error)) {
        qWarning() << error;
        return {};
    }
    if (!finished.wait(20000))
        qWarning() << "no result" << w.stderrTail();
    if (w.state() != WarmupController::State::Finished && w.state() != WarmupController::State::Failed)
        qWarning() << "state" << int(w.state());
    return w.lastRun().value_or(WarmupRun{});
}

static QString tomlString(const QString& path)
{
    QString s = QDir::toNativeSeparators(QDir::cleanPath(path));
    s.replace(QLatin1Char('\\'), QLatin1String("\\\\"));
    return QLatin1Char('"') + s + QLatin1Char('"');
}

static QString readText(const QString& path)
{
    QFile f(path);
    return f.open(QIODevice::ReadOnly) ? QString::fromUtf8(f.readAll()) : QString();
}

void TestControllers::victoryConfigWrittenForPrecacheAndLaunch()
{
    const QString id = QStringLiteral("victory-dexnom");
    installFakeRunner(id);
    const RunnerSpec* spec = Runners::find(id);
    const QString exe = QDir(spec->installDir()).filePath(spec->executable);
    const QString toml = QDir(spec->dataDir()).filePath(QStringLiteral("data/victory.toml"));
    QFile::remove(toml);

    // the stand-in stops as the fork does when VICTORY_CONFIG names a missing file
    QProcess bare;
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    env.insert(QStringLiteral("VICTORY_CONFIG"), toml);
    bare.setProcessEnvironment(env);
    bare.start(exe, {QStringLiteral("warm"), QStringLiteral("--no-rt"), QStringLiteral("--jobs"), QStringLiteral("2"), m_fixtures.path()});
    QVERIFY(bare.waitForFinished(10000));
    QCOMPARE(bare.exitCode(), 1);
    QVERIFY2(bare.readAllStandardError().contains("which does not exist"), "no config error");

    const Game game = fixtureGame(QStringLiteral("Signal Garden"));
    QStringList steps;
    const WarmupRun run = precache(id, game, &steps);
    QVERIFY2(run.result == WarmupRun::Result::Finished, qPrintable(run.message));
    QCOMPARE(run.exitCode, 0);
    QCOMPARE(run.done, 4210);
    QCOMPARE(run.total, 4210);
    QCOMPARE(steps, QStringList{QStringLiteral("Translating shaders")});
    const auto expected = [&](const QString& runnerDir) {
        return QStringLiteral("[paths]\ngame_dir = %1\ncache_dir = %2\ndata_dir = %3\n")
            .arg(tomlString(game.folder), tomlString(runnerDir + QStringLiteral("/cache")), tomlString(runnerDir + QStringLiteral("/data")));
    };
    QVERIFY2(readText(toml).endsWith(expected(spec->dataDir())), qPrintable(readText(toml)));
    QVERIFY(QFileInfo(QDir(spec->dataDir()).filePath(QStringLiteral("cache"))).isDir());

    // a launch writes it too, and the stand-in keeps running with it
    QFile::remove(toml);
    {
        RunnerManager runners;
        EmulatorController emu(*m_settings, *m_gate);
        emu.setRunners(&runners);
        QSignalSpy exited(&emu, &EmulatorController::exited);
        QString error;
        QVERIFY2(emu.start(game, 0, &error), qPrintable(error));
        QVERIFY(QFileInfo::exists(toml));
        QVERIFY(!exited.wait(1500));
        QVERIFY(emu.isActive());
        emu.stop();
        QTRY_COMPARE_WITH_TIMEOUT(emu.state(), EmulatorController::State::Idle, 15000);
    }

    // moved elsewhere: the next run's file names the new folders
    const QString elsewhere = m_dir->filePath(QStringLiteral("D/runners"));
    {
        RunnerManager runners;
        runners.setSettings(m_settings.get());
        QSignalSpy moved(&runners, &RunnerManager::relocated);
        QString error;
        QVERIFY2(runners.relocate(id, elsewhere, RunnerManager::Relocation::Move, &error), qPrintable(error));
        QVERIFY(moved.count() == 1 || moved.wait(30000));
        QVERIFY2(moved.first().at(1).toBool(), qPrintable(moved.first().at(2).toString()));
    }
    const QString movedDir = QDir(elsewhere).filePath(id);
    QCOMPARE(QDir::cleanPath(spec->dataDir()), QDir::cleanPath(movedDir));
    QVERIFY(precache(id, game).result == WarmupRun::Result::Finished);
    const QString movedToml = QDir(movedDir).filePath(QStringLiteral("data/victory.toml"));
    QVERIFY2(readText(movedToml).endsWith(expected(movedDir)), qPrintable(readText(movedToml)));
    RunnerManager back;
    back.setSettings(m_settings.get());
    QSignalSpy home(&back, &RunnerManager::relocated);
    QVERIFY(back.relocate(id, Paths::runnersDir(), RunnerManager::Relocation::Move));
    QVERIFY(home.count() == 1 || home.wait(30000));
    QVERIFY(home.first().at(1).toBool());
}

void TestControllers::xtxWarmWaitsForThePipelineBook()
{
    const QString id = QStringLiteral("victory-7900xtx");
    installFakeRunner(id);
    const RunnerSpec* spec = Runners::find(id);
    QDir pipelines(QDir(spec->dataDir()).filePath(QStringLiteral("cache/pipelines")));
    pipelines.removeRecursively();
    const Game game = fixtureGame(QStringLiteral("Signal Garden"));
    const QString note = QStringLiteral("Play once with VIctory 7900 XTX to record pipelines; the next warm-up builds them.");

    // no book yet: shaders only, with the note
    QStringList steps;
    WarmupRun run = precache(id, game, &steps);
    QVERIFY2(run.result == WarmupRun::Result::Finished, qPrintable(run.message));
    QCOMPARE(steps, QStringList{QStringLiteral("Translating shaders")});
    QCOMPARE(run.done, 4210);
    QCOMPARE(run.message, note);
    QVERIFY(QFileInfo::exists(QDir(spec->dataDir()).filePath(QStringLiteral("data/victory.toml"))));

    // a book, but an empty early list: still skipped
    QVERIFY(QDir().mkpath(pipelines.path()));
    const auto write = [&](const QString& name, const QByteArray& bytes) {
        QFile f(pipelines.filePath(name));
        QVERIFY(f.open(QIODevice::WriteOnly | QIODevice::Truncate));
        f.write(bytes);
    };
    write(QStringLiteral("known-pipelines.bin"), QByteArray(2000, '\n'));
    write(QStringLiteral("early-pipelines.txt"), {});
    steps.clear();
    run = precache(id, game, &steps);
    QCOMPARE(steps, QStringList{QStringLiteral("Translating shaders")});
    QCOMPARE(run.message, note);

    // both: the pipelines are built
    write(QStringLiteral("early-pipelines.txt"), "0011223344556677\n");
    steps.clear();
    run = precache(id, game, &steps);
    QVERIFY2(run.result == WarmupRun::Result::Finished, qPrintable(run.message));
    QCOMPARE(steps, (QStringList{QStringLiteral("Translating shaders"), QStringLiteral("Building pipelines")}));
    QCOMPARE(run.total, 1996);
    QVERIFY(run.message.isEmpty());
    QVERIFY(spec->environment.contains(qMakePair(QStringLiteral("VICTORY_PIPELINE_BOOK"), QStringLiteral("1"))));
    QVERIFY(!Runners::find(QStringLiteral("victory-dexnom"))->environment.contains(qMakePair(QStringLiteral("VICTORY_PIPELINE_BOOK"), QStringLiteral("1"))));
}

void TestControllers::kelvaroqNothingToBuildYet()
{
    installFakeKelvaroq();
    qputenv("FAKE_XSEMU_WAITING", "0");
    QStringList steps;
    const WarmupRun run = precache(QStringLiteral("kelvaroq"), fixtureGame(QStringLiteral("Signal Garden")), &steps);
    qunsetenv("FAKE_XSEMU_WAITING");
    QCOMPARE(run.result, WarmupRun::Result::NothingToBuild);
    QCOMPARE(run.exitCode, 0);
    QCOMPARE(run.message, QStringLiteral("Play once with Kelvaroq, then warm up."));
    QCOMPARE(steps, QStringList{QStringLiteral("Checking what to build")});
    QCOMPARE(m_settings->recentWarmups().first().result, WarmupRun::Result::NothingToBuild);
    QCOMPARE(m_settings->recentWarmups().first().message, run.message);
}

void TestControllers::silentPrecacheIsNotFinished()
{
    installFakeRunner(QStringLiteral("victory-dexnom"));
    qputenv("FAKE_XSEMU_SCENARIO", "silent");
    const WarmupRun run = precache(QStringLiteral("victory-dexnom"), fixtureGame(QStringLiteral("Signal Garden")));
    QCOMPARE(run.result, WarmupRun::Result::NothingToBuild);
    QCOMPARE(run.exitCode, 0);
    QVERIFY2(run.message.contains(QStringLiteral("finished without building anything")), qPrintable(run.message));
}

// what fake-xsemu recorded about its last invocation (FAKE_XSEMU_RECORD)
static QJsonObject lastRecord(const QString& path)
{
    const QStringList lines = readText(path).split(QLatin1Char('\n'), Qt::SkipEmptyParts);
    return lines.isEmpty() ? QJsonObject() : QJsonDocument::fromJson(lines.last().toUtf8()).object();
}

void TestControllers::coolfoxRunsWithItsOwnEnvironment()
{
    const QString id = QStringLiteral("coolfox");
    installFakeRunner(id);
    const RunnerSpec* spec = Runners::find(id);
    const QString record = m_dir->filePath(QStringLiteral("record.jsonl"));
    qputenv("FAKE_XSEMU_RECORD", record.toUtf8());
    // never passed on, even when the launcher's own environment has them
    qputenv("VICTORY_KEYS", "2:F1");
    qputenv("VICTORY_CONFIG", m_dir->filePath(QStringLiteral("missing.toml")).toUtf8());
    const auto checkEnv = [&](bool ambient) {
        const QJsonObject e = lastRecord(record).value(QStringLiteral("env")).toObject();
        QVERIFY(!e.contains(QStringLiteral("VICTORY_KEYS")));
        QVERIFY(!e.contains(QStringLiteral("VICTORY_CONFIG")));
        QCOMPARE(e.value(QStringLiteral("VICTORY_PIPELINE_BOOK")).toString(), QStringLiteral("1"));
        QCOMPARE(e.value(QStringLiteral("VICTORY_TEXTURE_NO_WAIT")).toString(), QStringLiteral("1"));
        QCOMPARE(e.value(QStringLiteral("VICTORY_RELAY")).toString(), QStringLiteral("2"));
        QCOMPARE(e.value(QStringLiteral("VICTORY_CARD_BUFFERS_MIN_KB")).toString(), QStringLiteral("64"));
        QCOMPARE(QDir::cleanPath(QDir::fromNativeSeparators(e.value(QStringLiteral("VICTORY_CACHE_DIR")).toString())), QDir(spec->dataDir()).filePath(QStringLiteral("cache")));
        QCOMPARE(QDir::cleanPath(QDir::fromNativeSeparators(e.value(QStringLiteral("VICTORY_DATA_DIR")).toString())), QDir(spec->dataDir()).filePath(QStringLiteral("data")));
        QCOMPARE(e.value(QStringLiteral("VICTORY_AMBIENT")).toString(), ambient ? QStringLiteral("0.8") : QString());
        QCOMPARE(e.value(QStringLiteral("VICTORY_AMBIENT_AIR")).toString(), ambient ? QStringLiteral("1.5") : QString());
        // QFileInfo compares canonical paths, case-insensitively on Windows (drive letters, short names)
        QCOMPARE(QFileInfo(lastRecord(record).value(QStringLiteral("cwd")).toString()), QFileInfo(spec->installDir()));
    };
    QDir(QDir(spec->dataDir()).filePath(QStringLiteral("cache/pipelines"))).removeRecursively();
    const Game game = fixtureGame(QStringLiteral("Signal Garden"));

    // no pipeline book yet: shaders only
    QStringList steps;
    WarmupRun run = precache(id, game, &steps);
    QVERIFY2(run.result == WarmupRun::Result::Finished, qPrintable(run.message));
    QCOMPARE(steps, QStringList{QStringLiteral("Translating shaders")});
    QCOMPARE(run.done, 4210);
    QCOMPARE(run.message, QStringLiteral("Play once with cool_fox to record pipelines; the next warm-up builds them."));
    const QJsonArray shaders = lastRecord(record).value(QStringLiteral("args")).toArray();
    QCOMPARE(shaders.at(0).toString(), QStringLiteral("shaders"));
    QCOMPARE(shaders.at(1).toString(), QStringLiteral("--jobs"));
    QCOMPARE(QDir::cleanPath(QDir::fromNativeSeparators(shaders.at(3).toString())), QDir::cleanPath(game.folder));
    checkEnv(false);
    QVERIFY(!QFileInfo::exists(QDir(spec->dataDir()).filePath(QStringLiteral("data/victory.toml"))));

    // after a run recorded the book: both steps
    QVERIFY(QDir().mkpath(QDir(spec->dataDir()).filePath(QStringLiteral("cache/pipelines"))));
    QFile book(QDir(spec->dataDir()).filePath(QStringLiteral("cache/pipelines/known-pipelines.bin")));
    QVERIFY(book.open(QIODevice::WriteOnly));
    book.write(QByteArray(2000, '\n'));
    book.close();
    steps.clear();
    run = precache(id, game, &steps);
    QCOMPARE(steps, (QStringList{QStringLiteral("Translating shaders"), QStringLiteral("Building pipelines")}));
    QCOMPARE(run.total, 1996);
    QCOMPARE(lastRecord(record).value(QStringLiteral("args")).toArray().at(1).toString(), QStringLiteral("--used"));

    // the launch: its own exe, always a Series S; the author's ambient light only when switched on
    for (const bool ambient : {false, true}) {
        m_settings->setRunnerOption(id, QStringLiteral("ambient"), ambient);
        RunnerManager runners;
        EmulatorController emu(*m_settings, *m_gate);
        emu.setRunners(&runners);
        QSignalSpy exited(&emu, &EmulatorController::exited);
        QString error;
        QVERIFY2(emu.start(game, 0, &error), qPrintable(error));
        QVERIFY(!exited.wait(1500));
        QVERIFY(emu.isActive());
        const QJsonArray args = lastRecord(record).value(QStringLiteral("args")).toArray();
        QCOMPARE(args.size(), 5);
        QCOMPARE(args.at(0).toString(), QStringLiteral("run"));
        QCOMPARE(QDir::cleanPath(QDir::fromNativeSeparators(args.at(1).toString())), QDir::cleanPath(QDir(game.folder).filePath(game.exe)));
        QCOMPARE(args.at(2).toString(), QStringLiteral("--console"));
        QCOMPARE(args.at(3).toString(), QStringLiteral("series-s"));
        QCOMPARE(args.at(4).toString(), QStringLiteral("--execute"));
        checkEnv(ambient);
        emu.stop();
        QTRY_COMPARE_WITH_TIMEOUT(emu.state(), EmulatorController::State::Idle, 15000);
    }
    qunsetenv("FAKE_XSEMU_RECORD");
    qunsetenv("VICTORY_KEYS");
    qunsetenv("VICTORY_CONFIG");
}

void TestControllers::warmupFinishesThroughEveryPhase()
{
    WarmupController w(*m_settings, *m_gate);
    QStringList phases;
    connect(&w, &WarmupController::progressChanged, this, [&] {
        if (!w.progress().phase.isEmpty() && (phases.isEmpty() || phases.last() != w.progress().phase))
            phases << w.progress().phase;
    });
    QSignalSpy finished(&w, &WarmupController::finished);
    QString error;
    QVERIFY2(w.start(false, fixtureGame(QStringLiteral("Signal Garden")), &error), qPrintable(error));
    QCOMPARE(m_gate->owner(), ProcessGate::Owner::Warmup);
    QVERIFY2(finished.wait(20000), qPrintable(QStringLiteral("state %1 phase %2 done %3 stderr %4").arg(int(w.state())).arg(w.progress().phase).arg(w.progress().done).arg(w.stderrTail().join(QLatin1Char('|')))));
    QCOMPARE(w.state(), WarmupController::State::Finished);
    QCOMPARE(phases, WarmupController::phases());
    QCOMPARE(w.progress().targets, 3120 + 96);
    QVERIFY(m_gate->isFree());
    const QList<WarmupRun> runs = m_settings->recentWarmups();
    QCOMPARE(runs.size(), 1);
    QCOMPARE(runs.first().result, WarmupRun::Result::Finished);
    QCOMPARE(runs.first().titleId, QStringLiteral("NorthwindInteractive.SignalGarden"));
    QVERIFY(runs.first().done > 3000);
    // what a pass leaves is what it failed to build: the fake fails a random pipeline every so often
    QCOMPARE(runs.first().left, runs.first().failed);
    // the pass is over, so no checkpoint offers to continue it: only the builds it could not make are
    // left for a later pass
    QVERIFY(!m_settings->warmupInFlight().has_value());
    QCOMPARE(WarmupController::resumable(runs, m_settings->warmupInFlight(), runs.first().titleId, QString(), {}).has_value(),
             runs.first().left > 0);
}

void TestControllers::warmupFailureKeepsWhatItBuilt()
{
    qputenv("FAKE_XSEMU_SCENARIO", "warmup_fail");
    WarmupController w(*m_settings, *m_gate);
    QSignalSpy finished(&w, &WarmupController::finished);
    QVERIFY(w.start(false, fixtureGame(QStringLiteral("Signal Garden"))));
    QVERIFY2(finished.wait(20000), qPrintable(w.stderrTail().join(QLatin1Char('|'))));
    QCOMPARE(w.state(), WarmupController::State::Failed);
    const WarmupRun run = *w.lastRun();
    // a crash has no done line: what the pass built is what its last progress line said, and what it
    // left is unknown rather than zero, so the page cannot claim a count it does not have
    QVERIFY2(run.done > 0, qPrintable(QStringLiteral("done %1").arg(run.done)));
    QCOMPARE(run.left, qint64(-1));
    QVERIFY(run.message.contains(QStringLiteral("driver compile")));
    // a failed pass left a record the Warm-up page can offer Continue from
    const auto resume = WarmupController::resumable(m_settings->recentWarmups(), m_settings->warmupInFlight(),
                                                   run.titleId, QString(), {});
    QVERIFY(resume.has_value());
    QCOMPARE(resume->built, run.done - run.failed);
    QVERIFY(!resume->full);
    // and the launcher is usable again: the gate is free for Play or another pass
    QVERIFY(m_gate->isFree());
}

void TestControllers::warmupContinueBuildsOnlyWhatIsLeft()
{
    // the cache the fake keeps its built count in, so it can go on from an earlier pass as the real
    // emulator's driver blobs do
    QTemporaryDir cache;
    QVERIFY(cache.isValid());
    qputenv("XCL_DXIL_CACHE", cache.path().toUtf8());
    const auto restoreCache = qScopeGuard([] { qunsetenv("XCL_DXIL_CACHE"); });
    const Game game = fixtureGame(QStringLiteral("Signal Garden"));
    qint64 firstTargets = 0;
    qint64 builtFirst = 0;
    {
        qputenv("FAKE_XSEMU_SCENARIO", "warmup_fail");
        WarmupController w(*m_settings, *m_gate);
        QSignalSpy finished(&w, &WarmupController::finished);
        QVERIFY(w.start(false, game));
        QVERIFY2(finished.wait(20000), qPrintable(w.stderrTail().join(QLatin1Char('|'))));
        QCOMPARE(w.state(), WarmupController::State::Failed);
        firstTargets = w.progress().targets;
        builtFirst = w.lastRun() ? w.lastRun()->done : 0;
        QVERIFY(firstTargets > 0);
        QVERIFY(builtFirst > 0);
        QVERIFY(builtFirst < firstTargets);
    }
    const auto resume = WarmupController::resumable(m_settings->recentWarmups(), m_settings->warmupInFlight(), game.titleId(), QString(), {});
    QVERIFY(resume.has_value());
    qputenv("FAKE_XSEMU_SCENARIO", "ok");
    {
        WarmupController w(*m_settings, *m_gate);
        QSignalSpy finished(&w, &WarmupController::finished);
        QSignalSpy progress(&w, &WarmupController::progressChanged);
        QVERIFY(w.start(resume->full, game));
        QVERIFY2(finished.wait(20000), qPrintable(w.stderrTail().join(QLatin1Char('|'))));
        QCOMPARE(w.state(), WarmupController::State::Finished);
        // the second pass is offered only what the first one did not build
        const qint64 left = firstTargets - builtFirst;
        QVERIFY2(qAbs(w.progress().targets - left) <= 200,
                 qPrintable(QStringLiteral("first pass %1, built %2, so about %3 left; the second pass took %4").arg(firstTargets).arg(builtFirst).arg(left).arg(w.progress().targets)));
        QVERIFY(w.progress().targets < firstTargets / 2);
        QVERIFY(w.lastRun()->left >= 0);
    }
}

void TestControllers::warmupFinishedWithLeftoversIsNotAFailure()
{
    // exit code 1 says some builds failed, not that the pass did not run: the game builds those when
    // it needs them, and Continue goes on with them
    qputenv("FAKE_XSEMU_SCENARIO", "warmup_partial");
    WarmupController w(*m_settings, *m_gate);
    QSignalSpy finished(&w, &WarmupController::finished);
    QVERIFY(w.start(false, fixtureGame(QStringLiteral("Signal Garden"))));
    QVERIFY2(finished.wait(20000), qPrintable(w.stderrTail().join(QLatin1Char('|'))));
    QCOMPARE(w.state(), WarmupController::State::Finished);
    const WarmupRun run = *w.lastRun();
    QCOMPARE(run.exitCode, 1);
    QVERIFY(run.failed > 0);
    QVERIFY(run.left > 0);
    QCOMPARE(m_settings->recentWarmups().first().result, WarmupRun::Result::Finished);
    // the pass went through everything, but the builds it could not make are still there to try again
    const auto resume = WarmupController::resumable(m_settings->recentWarmups(), m_settings->warmupInFlight(), run.titleId, QString(), {});
    QVERIFY(resume.has_value());
    QCOMPARE(resume->left, run.left);
}

void TestControllers::warmupResumeSurvivesTheLauncherClosing()
{
    // the launcher closing mid-pass kills the tree and never sees the result: the checkpoint it wrote
    // while the pass ran is what offers Continue next time
    qputenv("FAKE_XSEMU_TICK_MS", "40");
    const Game game = fixtureGame(QStringLiteral("Signal Garden"));
    {
        WarmupController w(*m_settings, *m_gate);
        QVERIFY(w.start(false, game));
        QTRY_VERIFY_WITH_TIMEOUT(w.progress().phase == QLatin1String("bound") && w.progress().passDone > 100, 10000);
    }
    qputenv("FAKE_XSEMU_TICK_MS", "5");
    m_gate->release(ProcessGate::Owner::Warmup); // the launcher that held it is gone too
    QVERIFY(m_settings->recentWarmups().isEmpty());
    const auto pass = m_settings->warmupInFlight();
    QVERIFY2(pass.has_value(), "a pass the launcher did not see the end of leaves a checkpoint");
    QCOMPARE(pass->titleId, game.titleId());
    QVERIFY(pass->built > 0);
    QVERIFY(!pass->phase.isEmpty());
    QVERIFY(!pass->full);
    // and it is what Continue runs, before any recorded run
    const auto resume = WarmupController::resumable(m_settings->recentWarmups(), m_settings->warmupInFlight(), game.titleId(), QString(), {});
    QVERIFY(resume.has_value());
    QCOMPARE(resume->built, pass->built);
    QCOMPARE(resume->phase, pass->phase);
    // a pass that runs to its end drops it
    {
        WarmupController w(*m_settings, *m_gate);
        QSignalSpy finished(&w, &WarmupController::finished);
        QVERIFY(w.start(resume->full, game));
        QVERIFY2(finished.wait(20000), qPrintable(w.stderrTail().join(QLatin1Char('|'))));
    }
    QVERIFY(!m_settings->warmupInFlight().has_value());
}

void TestControllers::warmupCancel()
{
    qputenv("FAKE_XSEMU_TICK_MS", "40");
    WarmupController w(*m_settings, *m_gate);
    QSignalSpy progress(&w, &WarmupController::progressChanged);
    QSignalSpy finished(&w, &WarmupController::finished);
    QVERIFY(w.start(true, {}));
    QVERIFY(progress.wait(5000));
    w.cancel();
    QVERIFY(w.cancelRequested());
    QVERIFY(finished.wait(10000));
    QCOMPARE(w.state(), WarmupController::State::Cancelled);
    QVERIFY(w.doneLine().has_value());
    QVERIFY(w.doneLine()->cancelled);
    QVERIFY(w.progress().full);
    qputenv("FAKE_XSEMU_TICK_MS", "5");
}

void TestControllers::warmupFailure()
{
    qputenv("FAKE_XSEMU_SCENARIO", "warmup_fail");
    WarmupController w(*m_settings, *m_gate);
    QSignalSpy finished(&w, &WarmupController::finished);
    QVERIFY(w.start(false, {}));
    QVERIFY(finished.wait(20000));
    QCOMPARE(w.state(), WarmupController::State::Failed);
    QCOMPARE(w.lastRun()->exitCode, 3);
    QVERIFY2(w.lastRun()->message.contains(QStringLiteral("driver compile")), qPrintable(w.lastRun()->message));
}

void TestControllers::warmupRefused()
{
    qputenv("FAKE_XSEMU_SCENARIO", "warmup_refuse");
    WarmupController w(*m_settings, *m_gate);
    QSignalSpy finished(&w, &WarmupController::finished);
    QVERIFY(w.start(false, {}));
    QVERIFY(finished.wait(10000));
    QCOMPARE(w.state(), WarmupController::State::Failed);
    QCOMPARE(w.lastRun()->exitCode, 3);
    QCOMPARE(w.lastRun()->message, QStringLiteral("another XSEmu process is running (XSEmu.exe 1234)"));
}

void TestControllers::gameRunPauseResumeStop()
{
    EmulatorController emu(*m_settings, *m_gate);
    const Game game = fixtureGame(QStringLiteral("Signal Garden"));
    QVERIFY(emu.checkLaunch(game).ok());
    QSignalSpy states(&emu, &EmulatorController::stateChanged);
    QSignalSpy exited(&emu, &EmulatorController::exited);
    QString error;
    QVERIFY2(emu.start(game, 0, &error), qPrintable(error));
    QCOMPARE(emu.state(), EmulatorController::State::Booting);
    QCOMPARE(TitleConfig(Paths::xclToml(game.titleId())).value(QStringLiteral("device_type")).toString(), QStringLiteral("lockhart"));
    QTRY_COMPARE_WITH_TIMEOUT(emu.state(), EmulatorController::State::Running, 10000);
    QVERIFY(emu.hello().has_value());
    QCOMPARE(emu.hello()->proto, 1);
    QTRY_VERIFY_WITH_TIMEOUT(emu.stats().fps.has_value(), 5000);
    QVERIFY(emu.stats().ramFreeMb.has_value());

    emu.pause();
    QTRY_COMPARE_WITH_TIMEOUT(emu.state(), EmulatorController::State::Paused, 5000);
    QVERIFY(m_gate->ownerPaused());
    QTRY_VERIFY_WITH_TIMEOUT(emu.stats().heldBytes > 0, 5000);
    emu.resume();
    QTRY_COMPARE_WITH_TIMEOUT(emu.state(), EmulatorController::State::Running, 5000);
    // each protocol stats message is one sample; an overlay toggle is not a new one
    QVERIFY(emu.stats().sample > 0);
    const quint64 before = emu.stats().sample;
    emu.setOverlay(true);
    QCOMPARE(emu.stats().sample, before);
    QTRY_VERIFY_WITH_TIMEOUT(emu.stats().sample > before, 5000);
    emu.stop();
    QCOMPARE(emu.state(), EmulatorController::State::Stopping);
    QVERIFY(exited.wait(10000));
    const auto info = exited.first().first().value<EmulatorController::ExitInfo>();
    QCOMPARE(info.kind, EmulatorController::ExitInfo::Kind::Stopped);
    QVERIFY(info.sawExitLine);
    QVERIFY(!info.killed);
    QCOMPARE(emu.state(), EmulatorController::State::Idle);
    QVERIFY(m_gate->isFree());
    QVERIFY(m_settings->lastPlayed(game.titleId()).isValid());
}

// fake-xsemu writes the XCL_* environment it was started with (FAKE_XSEMU_ENV_DUMP); the warm-up must
// see exactly what Play sees, or it fills a shader cache generation the game never reads
void TestControllers::playAndWarmupShareTheRunEnvironment()
{
    const QString dumps = m_dir->filePath(QStringLiteral("env"));
    QVERIFY(QDir().mkpath(dumps));
    qputenv("FAKE_XSEMU_ENV_DUMP", dumps.toUtf8());
    const auto cleanup = qScopeGuard([] { qunsetenv("FAKE_XSEMU_ENV_DUMP"); });
    const auto read = [&dumps](const QString& mode) {
        QFile f(QDir(dumps).filePath(mode + QStringLiteral(".env")));
        return f.open(QIODevice::ReadOnly) ? QString::fromUtf8(f.readAll()).split(QLatin1Char('\n')) : QStringList();
    };
    const auto runBoth = [&](const Game& game) {
        QFile::remove(QDir(dumps).filePath(QStringLiteral("warm-up.env")));
        QFile::remove(QDir(dumps).filePath(QStringLiteral("run.env")));
        WarmupController warmup(*m_settings, *m_gate);
        QSignalSpy finished(&warmup, &WarmupController::finished);
        QString error;
        QVERIFY2(warmup.start(false, game, &error), qPrintable(error));
        QVERIFY(finished.wait(20000));
        EmulatorController emu(*m_settings, *m_gate);
        QSignalSpy exited(&emu, &EmulatorController::exited);
        QVERIFY2(emu.start(game, 0, &error), qPrintable(error));
        QTRY_COMPARE_WITH_TIMEOUT(emu.state(), EmulatorController::State::Running, 10000);
        emu.stop();
        QVERIFY(exited.wait(10000));
    };

    const Game garden = fixtureGame(QStringLiteral("Signal Garden"));
    runBoth(garden);
    const QString defaults = LaunchConfig::profileIdentity(*m_settings, garden.titleId());
    QCOMPARE(m_settings->recentWarmups().first().profile, defaults);
    const QStringList played = read(QStringLiteral("run"));
    QCOMPARE(read(QStringLiteral("warm-up")), played);
    for (const QString& line : AppSettings::defaultRunProfile())
        QVERIFY2(played.contains(line), qPrintable(line));
    QVERIFY(played.contains(QStringLiteral("XCL_NETWORK=0")));

    // the game's own profile replaces the global one in both
    m_settings->setGameOverride(garden.titleId(), QStringLiteral("runProfile"), QStringList{QStringLiteral("XCL_LOOP_GUARD=4096")});
    QVERIFY(LaunchConfig::profileIdentity(*m_settings, garden.titleId()) != defaults);
    runBoth(garden);
    QCOMPARE(m_settings->recentWarmups().first().profile, LaunchConfig::profileIdentity(*m_settings, garden.titleId()));
    const QStringList own = read(QStringLiteral("run"));
    QCOMPARE(read(QStringLiteral("warm-up")), own);
    QVERIFY(own.contains(QStringLiteral("XCL_LOOP_GUARD=4096")));
    QVERIFY(!own.contains(QStringLiteral("XCL_SHADER_BACKEND=dxil")));
}

void TestControllers::gameCrash()
{
    qputenv("FAKE_XSEMU_SCENARIO", "crash");
    EmulatorController emu(*m_settings, *m_gate);
    QSignalSpy exited(&emu, &EmulatorController::exited);
    QVERIFY(emu.start(fixtureGame(QStringLiteral("Coastline Drive"))));
    QVERIFY(exited.wait(15000));
    const auto info = exited.first().first().value<EmulatorController::ExitInfo>();
    QCOMPARE(info.kind, EmulatorController::ExitInfo::Kind::Crash);
    QVERIFY(info.message.contains(QStringLiteral("ACCESS_VIOLATION")));
}

void TestControllers::gameFilesMissing()
{
    EmulatorController emu(*m_settings, *m_gate);
    const Game noExe = fixtureGame(QStringLiteral("Tide Pools"));
    const auto check = emu.checkLaunch(noExe);
    QCOMPARE(check.problem, EmulatorController::LaunchCheck::Problem::GameFilesMissing);
    QVERIFY(check.reason.contains(QStringLiteral("TidePools.exe")));
    QVERIFY(!emu.start(noExe));

    qputenv("FAKE_XSEMU_SCENARIO", "files_missing");
    QSignalSpy exited(&emu, &EmulatorController::exited);
    QVERIFY(emu.start(fixtureGame(QStringLiteral("Signal Garden"))));
    QVERIFY(exited.wait(10000));
    const auto info = exited.first().first().value<EmulatorController::ExitInfo>();
    QCOMPARE(info.kind, EmulatorController::ExitInfo::Kind::Error);
    QCOMPARE(info.exitCode, 2);
    QVERIFY(info.message.contains(QStringLiteral("game files missing")));
}

void TestControllers::oneProcessAtATime()
{
    m_settings->setLaunchNeedMb(1 << 30);
    EmulatorController emu(*m_settings, *m_gate);
    const auto ram = emu.checkLaunch(fixtureGame(QStringLiteral("Signal Garden")));
    QVERIFY2(ram.problem == EmulatorController::LaunchCheck::Problem::NotEnoughRam,
             qPrintable(QStringLiteral("problem %1 need %2 free %3 setting %4: %5").arg(int(ram.problem)).arg(ram.needMb).arg(ram.freeMb).arg(m_settings->launchNeedMb()).arg(ram.reason)));
    QVERIFY(ram.needMb > ram.freeMb);
    m_settings->setLaunchNeedMb(0);

    WarmupController w(*m_settings, *m_gate);
    QVERIFY(emu.start(fixtureGame(QStringLiteral("Signal Garden"))));
    QString error;
    QVERIFY(!w.start(false, {}, &error));
    QVERIFY2(error.contains(QStringLiteral("Stop it to warm up")), qPrintable(error));
    QCOMPARE(emu.checkLaunch(fixtureGame(QStringLiteral("Coastline Drive"))).problem, EmulatorController::LaunchCheck::Problem::Busy);
    QSignalSpy exited(&emu, &EmulatorController::exited);
    m_gate->stopOwner();
    QVERIFY(exited.wait(10000));
    QVERIFY(m_gate->isFree());
    QVERIFY(w.checkStart().isEmpty());
}

void TestControllers::emulatorTooOldIsNeverStarted()
{
    QVERIFY(EmulatorBuild::speaksProtocol(QStringLiteral("XSEmu 0.9.0"), QStringLiteral("run [--control-pipe <name>] ... warm-up [--progress-json]")));
    QVERIFY(!EmulatorBuild::speaksProtocol(QStringLiteral("xcl 0.1.0"), QStringLiteral("--control-pipe --progress-json")));
    QVERIFY(!EmulatorBuild::speaksProtocol(QStringLiteral("XSEmu 0.8.0"), QStringLiteral("run <game folder>")));
    const EmulatorBuild fake = EmulatorBuild::probe(QStringLiteral(XSEMU_FAKE_EMULATOR_PATH));
    QVERIFY2(fake.capable, qPrintable(fake.version));
#ifdef Q_OS_WIN
    QSKIP("the old emulator stand-in is a shell script");
#else
    // an older xcl: answers --version and --help, and would start the game for anything else
    const QString started = m_dir->filePath(QStringLiteral("started"));
    const QString old = m_dir->filePath(QStringLiteral("xcl.exe"));
    QFile script(old);
    QVERIFY(script.open(QIODevice::WriteOnly));
    script.write(QStringLiteral("#!/bin/sh\ncase \"$1\" in\n--version) echo 'xcl 0.1.0' ;;\n--help) echo 'usage: xcl run <game folder>' ;;\n*) touch '%1' ;;\nesac\n").arg(started).toUtf8());
    script.close();
    script.setPermissions(script.permissions() | QFileDevice::ExeOwner);
    const QByteArray saved = qgetenv("XSEMU_EMULATOR");
    qputenv("XSEMU_EMULATOR", old.toUtf8());

    const EmulatorBuild build = EmulatorBuild::probe(old);
    QVERIFY(build.found);
    QCOMPARE(build.version, QStringLiteral("xcl 0.1.0"));
    QVERIFY(!build.capable);
    EmulatorController emu(*m_settings, *m_gate);
    const Game game = fixtureGame(QStringLiteral("Signal Garden"));
    const auto check = emu.checkLaunch(game);
    QCOMPARE(check.problem, EmulatorController::LaunchCheck::Problem::EmulatorTooOld);
    QVERIFY2(check.reason.contains(QStringLiteral("too old")), qPrintable(check.reason));
    QString error;
    QVERIFY(!emu.start(game, 0, &error));
    WarmupController w(*m_settings, *m_gate);
    QVERIFY2(w.checkStart(game).contains(QStringLiteral("too old")), qPrintable(w.checkStart(game)));
    QVERIFY(!w.start(false, game, &error));
    QTest::qWait(200);
    QVERIFY(!QFile::exists(started));
    QVERIFY(m_gate->isFree());
    qputenv("XSEMU_EMULATOR", saved);
#endif
}

void TestControllers::busyMarkerBlocksPlayAndWarmup()
{
    const QString marker = m_dir->filePath(QStringLiteral("game-pending"));
    m_settings->setBusyMarker(marker);
    QCOMPARE(m_settings->busyMarker(), marker);
    qputenv("XSEMU_BUSY_MARKER", "/elsewhere/marker");
    QCOMPARE(m_settings->busyMarker(), QStringLiteral("/elsewhere/marker"));
    qunsetenv("XSEMU_BUSY_MARKER");
    m_gate->setBusyMarker(m_settings->busyMarker());

    EmulatorController emu(*m_settings, *m_gate);
    WarmupController w(*m_settings, *m_gate);
    const Game game = fixtureGame(QStringLiteral("Signal Garden"));
    QVERIFY(emu.checkLaunch(game).ok());
    QSignalSpy changed(m_gate.get(), &ProcessGate::changed);
    QFile f(marker);
    QVERIFY(f.open(QIODevice::WriteOnly));
    f.close();
    QVERIFY(changed.wait(3000));
    const auto check = emu.checkLaunch(game);
    QCOMPARE(check.problem, EmulatorController::LaunchCheck::Problem::OtherEmulatorRunning);
    QCOMPARE(check.marker, marker);
    QVERIFY2(check.reason.startsWith(QStringLiteral("Emulator busy")), qPrintable(check.reason));
    QVERIFY(w.checkStart(game).startsWith(QStringLiteral("Emulator busy")));
    QString error;
    QVERIFY(!emu.start(game, 0, &error));
    QVERIFY(!w.start(false, game, &error));
    QVERIFY(m_gate->isFree());
    QVERIFY(QFile::remove(marker));
    QVERIFY(changed.wait(3000));
    QVERIFY(emu.checkLaunch(game).ok());
    QVERIFY(w.checkStart(game).isEmpty());
}

void TestControllers::runningEmulatorProcessBlocks()
{
#ifdef Q_OS_WIN
    QSKIP("needs a long-running stand-in named xcl.exe");
#else
    const QString sleep = QStandardPaths::findExecutable(QStringLiteral("sleep"));
    if (sleep.isEmpty())
        QSKIP("no sleep binary");
    const QString xcl = m_dir->filePath(QStringLiteral("xcl.exe"));
    QVERIFY(QFile::copy(sleep, xcl));
    QSignalSpy changed(m_gate.get(), &ProcessGate::changed);
    QProcess p;
    p.start(xcl, {QStringLiteral("30")});
    QVERIFY(p.waitForStarted());
    EmulatorController emu(*m_settings, *m_gate);
    const auto check = emu.checkLaunch(fixtureGame(QStringLiteral("Signal Garden")));
    QCOMPARE(check.problem, EmulatorController::LaunchCheck::Problem::OtherEmulatorRunning);
    QVERIFY2(check.processes.contains(QStringLiteral("xcl.exe %1").arg(p.processId())), qPrintable(check.processes.join(QLatin1Char(','))));
    QVERIFY(changed.wait(3000));
    p.kill();
    p.waitForFinished();
    QVERIFY(emu.checkLaunch(fixtureGame(QStringLiteral("Signal Garden"))).ok());
#endif
}

QTEST_GUILESS_MAIN(TestControllers)
void TestControllers::probeChildIsNotAnotherEmulator()
{
    // The launcher's own XSEmu.exe --version child matches the names of another emulator; while it
    // answers, the gate stays free. The copies sit beside the fake so they find the same libraries.
    const QDir fakeDir = QFileInfo(QStringLiteral(XSEMU_FAKE_EMULATOR_PATH)).dir();
    const QString exe = fakeDir.filePath(QStringLiteral("XSEmu.exe"));
    const QString abandoned = fakeDir.filePath(QStringLiteral("XSEmu-abandoned.exe"));
    for (const QString& copy : {exe, abandoned}) {
        QFile::remove(copy);
        QVERIFY(QFile::copy(QStringLiteral(XSEMU_FAKE_EMULATOR_PATH), copy));
    }
    const auto removeCopies = qScopeGuard([exe, abandoned] {
        QFile::remove(exe);
        QFile::remove(abandoned);
    });
    qputenv("FAKE_XSEMU_VERSION_DELAY_MS", "1500");
    const auto unset = qScopeGuard([] { qunsetenv("FAKE_XSEMU_VERSION_DELAY_MS"); });

    QElapsedTimer clock;
    clock.start();
    std::future<EmulatorBuild> probe = std::async(std::launch::async, [exe] { return EmulatorBuild::probe(exe); });
    const auto answering = [&probe] { return probe.wait_for(std::chrono::seconds(0)) != std::future_status::ready; };
    QSignalSpy changed(m_gate.get(), &ProcessGate::changed);
    QTest::qWait(500);
    QVERIFY2(answering(), "the --version child should still be answering");
    QStringList seen;
    while (answering()) {
        seen << m_gate->external().processes;
        QTest::qWait(100);
    }
    const EmulatorBuild answer = probe.get();
    QVERIFY2(seen.isEmpty(), qPrintable(seen.join(QLatin1Char(','))));
    QVERIFY(clock.elapsed() >= 1500);
    QVERIFY2(answer.capable, qPrintable(answer.version));
    QCOMPARE(changed.count(), 0);
    QVERIFY(m_gate->isFree());

    // a launch starting while it asks abandons the question; nothing is kept, so it is asked again later
    std::atomic<bool> launched = false;
    clock.restart();
    probe = std::async(std::launch::async, [abandoned, &launched] { return EmulatorBuild::probe(abandoned, [&launched] { return launched.load(); }); });
    QTest::qWait(300);
    launched = true;
    const EmulatorBuild cut = probe.get();
    QVERIFY2(clock.elapsed() < 1200, qPrintable(QString::number(clock.elapsed())));
    QVERIFY(!cut.found);
    QVERIFY(!EmulatorBuild::cached(abandoned).found);
}

#include "tst_controllers.moc"
