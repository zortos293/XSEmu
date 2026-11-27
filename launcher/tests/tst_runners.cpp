// Runners: the curated list, the recommendation, progress parsing, and real installs (a release
// asset over file:// with its SHA-256, and a source build from a local Git repository).
#include "AppSettings.h"
#include "DiskSpace.h"
#include "Paths.h"
#include "ProgressParsers.h"
#include "Runner.h"
#include "RunnerInstall.h"
#include "RunnerManager.h"

#include <QCryptographicHash>
#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QSignalSpy>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QTest>
#include <QUrl>

namespace {

constexpr quint64 kGb = 1024ULL * 1024 * 1024;

bool run(const QString& program, const QStringList& args, const QString& dir)
{
    QProcess p;
    p.setWorkingDirectory(dir);
    p.start(program, args);
    return p.waitForFinished(30000) && p.exitStatus() == QProcess::NormalExit && p.exitCode() == 0;
}

QString sha256(const QString& path)
{
    QFile f(path);
    QCryptographicHash h(QCryptographicHash::Sha256);
    f.open(QIODevice::ReadOnly);
    h.addData(&f);
    return QString::fromLatin1(h.result().toHex());
}

}

class TestRunners : public QObject {
    Q_OBJECT
private slots:
    void initTestCase();
    void curatedList();
    void recommendation();
    void parsers_data();
    void parsers();
    void installsReleaseAsset();
    void cancellingLeavesNoFailure();
    void aFailedDownloadIsRetried();
    void rejectsWrongHash();
    void buildsFromSource();
    void stepProgramsUseTheStepsPath();
    void locationPersistsAcrossRestarts();
    void moveKeepsTheInstall();
    void moveCopiesAcrossDrives();
    void rebuildThere();
    void refusesWhileBusy();
    void runnersFolderKeepsInstalledRunners();
    void freeSpaceWarning();
    void uninstallDeletesOffTheUiThread();
    void uninstallRefusedWhileInUse();
    void fetchFallsBackAndChecksIds();

private:
    void ensureBuilt(RunnerManager& m, const QString& id);
    void resetLocations();

    QTemporaryDir m_dir;
    QString m_zip;
    bool m_haveGit = false;
    QString m_commit;
    QString m_tree;
};

void TestRunners::initTestCase()
{
    qputenv("XSEMU_RUNNERS_DIR", m_dir.filePath(QStringLiteral("runners")).toUtf8());

    // a release asset: one top-level folder holding the program and a file the spec copies
    const QString pkg = m_dir.filePath(QStringLiteral("pkg/fake-1.0"));
    QVERIFY(QDir().mkpath(pkg + QStringLiteral("/native")));
    for (const QString& name : {QStringLiteral("native/engine"), QStringLiteral("runtime.dll")}) {
        QFile f(QDir(pkg).filePath(name));
        QVERIFY(f.open(QIODevice::WriteOnly));
        f.write(name.toUtf8());
    }
    m_zip = m_dir.filePath(QStringLiteral("fake-1.0.zip"));
#ifdef Q_OS_WIN
    QVERIFY(run(QStringLiteral("tar"), {QStringLiteral("-a"), QStringLiteral("-c"), QStringLiteral("-f"), m_zip, QStringLiteral("fake-1.0")}, m_dir.filePath(QStringLiteral("pkg"))));
#else
    if (!QStandardPaths::findExecutable(QStringLiteral("unzip")).isEmpty())
        QVERIFY(run(QStringLiteral("python3"), {QStringLiteral("-c"), QStringLiteral("import shutil,sys; shutil.make_archive(sys.argv[1][:-4], 'zip', sys.argv[2], 'fake-1.0')"), m_zip, m_dir.filePath(QStringLiteral("pkg"))}, m_dir.path()));
#endif

    // a source repository with one commit
    const QString repo = m_dir.filePath(QStringLiteral("repo"));
    m_haveGit = !QStandardPaths::findExecutable(QStringLiteral("git")).isEmpty();
    if (m_haveGit) {
        QDir().mkpath(repo);
        QFile f(QDir(repo).filePath(QStringLiteral("README")));
        QVERIFY(f.open(QIODevice::WriteOnly));
        f.write("runner source\n");
        f.close();
        const QStringList id = {QStringLiteral("-c"), QStringLiteral("user.name=t"), QStringLiteral("-c"), QStringLiteral("user.email=t@t")};
        QVERIFY(run(QStringLiteral("git"), {QStringLiteral("init"), QStringLiteral("-q")}, repo));
        QVERIFY(run(QStringLiteral("git"), {QStringLiteral("add"), QStringLiteral("README")}, repo));
        QVERIFY(run(QStringLiteral("git"), id + QStringList{QStringLiteral("commit"), QStringLiteral("-q"), QStringLiteral("-m"), QStringLiteral("one")}, repo));
        QProcess p;
        p.setWorkingDirectory(repo);
        p.start(QStringLiteral("git"), {QStringLiteral("rev-parse"), QStringLiteral("HEAD")});
        QVERIFY(p.waitForFinished());
        m_commit = QString::fromLatin1(p.readAllStandardOutput()).trimmed();
        // allow fetching an unadvertised commit by SHA from this repository
        QVERIFY(run(QStringLiteral("git"), {QStringLiteral("config"), QStringLiteral("uploadpack.allowReachableSHA1InWant"), QStringLiteral("true")}, repo));
        QProcess tree;
        tree.setWorkingDirectory(repo);
        tree.start(QStringLiteral("git"), {QStringLiteral("rev-parse"), QStringLiteral("HEAD^{tree}")});
        QVERIFY(tree.waitForFinished());
        m_tree = QString::fromLatin1(tree.readAllStandardOutput()).trimmed();

        // cool_fox's three sources: one gone, one a different commit (another fork's tip), one the pin
        const QString other = m_dir.filePath(QStringLiteral("other"));
        QDir().mkpath(other);
        QFile o(QDir(other).filePath(QStringLiteral("README")));
        QVERIFY(o.open(QIODevice::WriteOnly));
        o.write("a later fork\n");
        o.close();
        QVERIFY(run(QStringLiteral("git"), {QStringLiteral("-c"), QStringLiteral("init.defaultBranch=main"), QStringLiteral("init"), QStringLiteral("-q")}, other));
        QVERIFY(run(QStringLiteral("git"), {QStringLiteral("add"), QStringLiteral("README")}, other));
        QVERIFY(run(QStringLiteral("git"), id + QStringList{QStringLiteral("commit"), QStringLiteral("-q"), QStringLiteral("-m"), QStringLiteral("fork")}, other));
        QVERIFY(run(QStringLiteral("git"), {QStringLiteral("clone"), QStringLiteral("-q"), QStringLiteral("--bare"), other, m_dir.filePath(QStringLiteral("remote-fork.git"))}, m_dir.path()));
        QVERIFY(run(QStringLiteral("git"), {QStringLiteral("clone"), QStringLiteral("-q"), QStringLiteral("--bare"), repo, m_dir.filePath(QStringLiteral("remote-pin.git"))}, m_dir.path()));
    }

    // re-pin two runners at the local asset and repository before the list is first read
    // (append, not QJsonArray{QJsonArray{...}}: MSVC takes that as a copy of the inner array)
    QJsonArray buildSteps;
    buildSteps.append(QJsonArray{QStringLiteral("git"), QStringLiteral("archive"), QStringLiteral("-o"), QStringLiteral("{targetDir}/runner.tar"), QStringLiteral("HEAD")});
    QJsonObject feed;
    feed.insert(QStringLiteral("kelvaroq"), QJsonObject{{QStringLiteral("tag"), QStringLiteral("v1.0")},
                                                         {QStringLiteral("asset"), QStringLiteral("fake-1.0.zip")},
                                                         {QStringLiteral("sha256"), QFileInfo::exists(m_zip) ? sha256(m_zip) : QString(64, QLatin1Char('0'))},
                                                         {QStringLiteral("url"), QUrl::fromLocalFile(m_zip).toString()},
                                                         {QStringLiteral("executable"), QStringLiteral("native/engine")}});
    feed.insert(QStringLiteral("victory-7900xtx"), QJsonObject{{QStringLiteral("tag"), QStringLiteral("v1.0")},
                                                                {QStringLiteral("asset"), QStringLiteral("fake-1.0.zip")},
                                                                {QStringLiteral("sha256"), QString(64, QLatin1Char('a'))},
                                                                {QStringLiteral("url"), QUrl::fromLocalFile(m_zip).toString()},
                                                                {QStringLiteral("executable"), QStringLiteral("native/engine")}});
    feed.insert(QStringLiteral("victory-dexnom"), QJsonObject{{QStringLiteral("commit"), m_commit.isEmpty() ? QString(40, QLatin1Char('0')) : m_commit},
                                                               {QStringLiteral("repo"), QUrl::fromLocalFile(repo).toString()},
                                                               {QStringLiteral("buildSteps"), buildSteps},
                                                               {QStringLiteral("artifacts"), QJsonObject{{QStringLiteral("runner.tar"), QStringLiteral("runner.tar")}}},
                                                               {QStringLiteral("executable"), QStringLiteral("runner.tar")}});
    feed.insert(QStringLiteral("coolfox"), QJsonObject{{QStringLiteral("commit"), m_commit.isEmpty() ? QString(40, QLatin1Char('0')) : m_commit},
                                                        {QStringLiteral("tree"), m_tree},
                                                        {QStringLiteral("remotes"), QJsonArray{QUrl::fromLocalFile(m_dir.filePath(QStringLiteral("remote-gone.git"))).toString(),
                                                                                               QUrl::fromLocalFile(m_dir.filePath(QStringLiteral("remote-fork.git"))).toString(),
                                                                                               QUrl::fromLocalFile(m_dir.filePath(QStringLiteral("remote-pin.git"))).toString()}},
                                                        {QStringLiteral("buildSteps"), buildSteps},
                                                        {QStringLiteral("artifacts"), QJsonObject{{QStringLiteral("runner.tar"), QStringLiteral("runner.tar")}}},
                                                        {QStringLiteral("executable"), QStringLiteral("runner.tar")}});
    const QString feedPath = m_dir.filePath(QStringLiteral("feed.json"));
    QFile f(feedPath);
    QVERIFY(f.open(QIODevice::WriteOnly));
    f.write(QJsonDocument(feed).toJson());
    f.close();
    qputenv("XSEMU_RUNNER_FEED", feedPath.toUtf8());
}

void TestRunners::curatedList()
{
    const QList<RunnerSpec>& list = Runners::curated();
    QCOMPARE(list.size(), 5);
    QCOMPARE(list.first().id, QStringLiteral("xsemu"));
    QVERIFY(Runners::xsemu().isBuiltIn());
    for (const RunnerSpec& r : list) {
        QVERIFY2(Runners::find(r.id) == &r, qPrintable(r.id));
        QVERIFY2(r.isPinned(), qPrintable(r.id)); // the feed re-pins the Windows-only ones here
        if (!r.isBuiltIn()) {
            QVERIFY(!r.canPause && !r.canOverlay);
            QCOMPARE(r.embed, RunnerSpec::Embed::Docked);
        }
    }
    const RunnerSpec* k = Runners::find(QStringLiteral("kelvaroq"));
    QCOMPARE(k->precache.size(), 2);
    QVERIFY(k->precache.first().check);
    QCOMPARE(k->precache.last().cancel, RunnerSpec::PrecacheCancel::CloseStdin);
    QCOMPARE(k->stop, RunnerSpec::Stop::SysCommandClose);
    QVERIFY(!k->note.isEmpty());

    const QStringList args = RunnerSpec::expand(k->launchArgs, k->tokens(QStringLiteral("/g"), QStringLiteral("/g/Game.exe"), QStringLiteral("T"), QStringLiteral("lockhart"), 0));
    QCOMPARE(args, (QStringList{QStringLiteral("run"), QDir::toNativeSeparators(QStringLiteral("/g/Game.exe")), QStringLiteral("--console"), QStringLiteral("series-s"), QStringLiteral("--execute")}));
    QCOMPARE(QDir::fromNativeSeparators(k->resolvedCacheDir()), QDir(Paths::runnerDir(QStringLiteral("kelvaroq"))).filePath(QStringLiteral("cache")));
    QCOMPARE(RunnerSpec::expand(QStringLiteral("{rtFlag} {unknown}"), k->tokens({}, {}, {}, QStringLiteral("anaconda"), 0)), QStringLiteral("--no-rt {unknown}"));

    // cool_fox: always a Series S, whatever the game's console (its environment: tst_controllers)
    const RunnerSpec* f = Runners::find(QStringLiteral("coolfox"));
    QVERIFY(f);
    QCOMPARE(f->name, QStringLiteral("cool_fox (codec_d)"));
    QCOMPARE(f->badge, QStringLiteral("NVIDIA \u00B7 tuned on RTX 5090 (32 GB RAM)"));
    const auto tokens = f->tokens(QStringLiteral("/g"), QStringLiteral("/g/Game.exe"), QStringLiteral("T"), QStringLiteral("anaconda"), 0);
    QCOMPARE(RunnerSpec::expand(f->launchArgs, tokens),
             (QStringList{QStringLiteral("run"), QDir::toNativeSeparators(QStringLiteral("/g/Game.exe")), QStringLiteral("--console"), QStringLiteral("series-s"), QStringLiteral("--execute")}));
    QCOMPARE(f->precache.size(), 2);
    QCOMPARE(RunnerSpec::expand(f->precache.first().args, tokens).mid(0, 2), (QStringList{QStringLiteral("shaders"), QStringLiteral("--jobs")}));
    QCOMPARE(RunnerSpec::expand(f->precache.last().args, tokens).mid(0, 2), (QStringList{QStringLiteral("warm"), QStringLiteral("--used")}));
    QCOMPARE(f->precache.last().needs, QStringList{QStringLiteral("{cacheDir}/pipelines/known-pipelines.bin")});
}

void TestRunners::recommendation()
{
    auto all = [](const QString&) { return true; };
    auto none = [](const QString& id) { return id == QLatin1String(Runners::XSEmu); };
    QCOMPARE(Runners::recommend(Runners::kVendorAmd, QStringLiteral("Radeon RX 7900 XTX"), 64 * kGb, all).runnerId, QStringLiteral("victory-7900xtx"));
    QCOMPARE(Runners::recommend(Runners::kVendorAmd, QStringLiteral("Radeon RX 7800 XT"), 32 * kGb, all).runnerId, QStringLiteral("victory-dexnom"));
    const auto nv = Runners::recommend(Runners::kVendorNvidia, QStringLiteral("NVIDIA GPU (sample)"), 32 * kGb, all);
    QCOMPARE(nv.runnerId, QStringLiteral("kelvaroq"));
    QVERIFY(nv.reason.contains(QStringLiteral("RTX 5080")));
    QVERIFY(nv.reason.contains(QStringLiteral("NVIDIA GPU (sample)")));
    QCOMPARE(Runners::recommend(Runners::kVendorIntel, QStringLiteral("Arc A770"), 32 * kGb, all).runnerId, QStringLiteral("xsemu"));
    QCOMPARE(Runners::recommend(0, {}, 32 * kGb, all).runnerId, QStringLiteral("xsemu"));
    const auto blocked = Runners::recommend(Runners::kVendorAmd, QStringLiteral("Radeon"), 64 * kGb, none);
    QCOMPARE(blocked.runnerId, QStringLiteral("xsemu"));
    QVERIFY(blocked.reason.contains(QStringLiteral("AMD")));
}

void TestRunners::parsers_data()
{
    QTest::addColumn<QString>("parser");
    QTest::addColumn<QString>("line");
    QTest::addColumn<qint64>("done");
    QTest::addColumn<qint64>("total");
    QTest::newRow("git") << "git" << "Receiving objects:  45% (450/1000), 1.2 MiB" << qint64(450) << qint64(1000);
    QTest::newRow("cargo bar") << "cargo" << "    Building [=====>      ] 120/300: serde" << qint64(120) << qint64(300);
    QTest::newRow("cargo step") << "cargo" << "   Compiling serde v1.0.200" << qint64(-1) << qint64(-1);
    QTest::newRow("kelvaroq check") << "kelvaroq" << "precompile-check 812\r" << qint64(0) << qint64(812);
    QTest::newRow("kelvaroq step") << "kelvaroq" << "precompile 120 812 2" << qint64(120) << qint64(812);
    QTest::newRow("victory shaders") << "victory-shaders" << "4000 shaders in the cache, 12 s" << qint64(4000) << qint64(-1);
    QTest::newRow("victory shaders done") << "victory-shaders" << "41906 shaders in the cache, 3 do not translate, 95 s" << qint64(41906) << qint64(41906);
    QTest::newRow("victory warm header") << "victory-warm" << "9000 pipelines known, 7000 made before (last run), 4 bad; 1996 to make, 16 processes" << qint64(0) << qint64(1996);
    QTest::newRow("victory warm tick") << "victory-warm" << "500 of 1996, 40 s" << qint64(500) << qint64(1996);
    QTest::newRow("dexnom tick") << "victory-dexnom" << "   12 s  2000 shaders" << qint64(2000) << qint64(-1);
    QTest::newRow("dexnom summary") << "victory-dexnom" << "warm         4210 translated, 3 refused; 812 compute pipelines, 0 refused; 0 omitted by ray-tracing mode; 7 s"
                                    << qint64(4210) << qint64(4210);
    // cool_fox, lines as apps/victory/src/translate.rs and warm.rs print them at f4b48ad
    QTest::newRow("coolfox shaders") << "coolfox-shaders" << "4000 shaders in the cache, 12 s" << qint64(4000) << qint64(-1);
    QTest::newRow("coolfox shaders done") << "coolfox-shaders" << "25985 shaders in the cache, 12 do not translate, 340 s" << qint64(25985) << qint64(25985);
    QTest::newRow("coolfox cause") << "coolfox-shaders" << "     12  wave64 lane intrinsic  (e.g. 00ff13a2c4b5d6e7)" << qint64(-1) << qint64(-1);
    QTest::newRow("coolfox warm header") << "victory-warm" << "84012 pipelines known, 0 made before (3f9a0c11d2e4b5a6), 0 bad; 2104 to make, 21 processes" << qint64(0) << qint64(2104);
    QTest::newRow("coolfox warm tick") << "victory-warm" << "1000 of 2104, 61 s" << qint64(1000) << qint64(2104);
    QTest::newRow("coolfox warm failed") << "victory-warm" << "pipeline 17 00ff13a2c4b5d6e7aa: the compiler ended the process" << qint64(-1) << qint64(-1);
    QTest::newRow("coolfox warm made") << "victory-warm"
                                       << "2101 made in 133 s (2 have no host pipeline, 1 ended their process); ms each: median 9.1, p90 30.2, p99 120.0, most 900, sum 40 s"
                                       << qint64(2101) << qint64(-1);
    QTest::newRow("generic") << "other" << "step 3/9" << qint64(3) << qint64(9);
}

void TestRunners::parsers()
{
    QFETCH(QString, parser);
    QFETCH(QString, line);
    QFETCH(qint64, done);
    QFETCH(qint64, total);
    const auto u = ProgressParsers::parse(parser, line);
    QVERIFY(u.has_value());
    QCOMPARE(u->done, done);
    QCOMPARE(u->total, total);
    QVERIFY(!ProgressParsers::parse(QStringLiteral("kelvaroq"), QStringLiteral("presenting to a window")).has_value());
    // cool_fox gives no total while it translates: the bar stays indeterminate, with an estimate
    QCOMPARE(ProgressParsers::parse(QStringLiteral("coolfox-shaders"), QStringLiteral("4000 shaders in the cache, 12 s"))->estimate, 25985);
    QCOMPARE(ProgressParsers::parse(QStringLiteral("coolfox-shaders"), QStringLiteral("25985 shaders in the cache, 12 do not translate, 340 s"))->estimate, -1);
    QCOMPARE(ProgressParsers::parse(QStringLiteral("coolfox-shaders"), QStringLiteral("     12  wave64 lane intrinsic  (e.g. 00ff13a2c4b5d6e7)"))->detail,
             QStringLiteral("12 do not translate: wave64 lane intrinsic"));
    QCOMPARE(ProgressParsers::parse(QStringLiteral("victory-warm"), QStringLiteral("2101 made in 133 s (2 have no host pipeline, 1 ended their process); ms each: median 9.1, p90 30.2, p99 120.0, most 900, sum 40 s"))->failed, 3);
}

void TestRunners::installsReleaseAsset()
{
    if (!QFileInfo::exists(m_zip))
        QSKIP("no zip tool on this machine");
    RunnerManager m;
    const QString id = QStringLiteral("kelvaroq");
    QCOMPARE(m.status(id).state, RunnerManager::State::NotInstalled);
    QSignalSpy done(&m, &RunnerManager::installFinished);
    QString error;
    QVERIFY2(m.install(id, &error), qPrintable(error));
    QCOMPARE(m.status(id).state, RunnerManager::State::Installing);
    QVERIFY(!m.install(QStringLiteral("victory-7900xtx"), &error)); // one install at a time
    QVERIFY(done.wait(30000));
    QVERIFY2(done.first().at(1).toBool(), qPrintable(done.first().at(2).toString()));
    const RunnerManager::Status s = m.status(id);
    QCOMPARE(s.state, RunnerManager::State::Installed);
    QCOMPARE(s.installedVersion, QStringLiteral("v1.0"));
    QVERIFY(QFileInfo::exists(s.executable));
    QVERIFY(m.isRunnable(id));
    QVERIFY(!QFileInfo::exists(QDir(Paths::runnerDir(id)).filePath(QStringLiteral("staging"))));

    // an older install record against the curated pin offers the update
    QFile rec(QDir(Paths::runnerDir(id)).filePath(QStringLiteral("runner.json")));
    QVERIFY(rec.open(QIODevice::ReadOnly));
    QJsonObject o = QJsonDocument::fromJson(rec.readAll()).object();
    rec.close();
    o.insert(QStringLiteral("version"), QStringLiteral("v0.9"));
    QVERIFY(rec.open(QIODevice::WriteOnly | QIODevice::Truncate));
    rec.write(QJsonDocument(o).toJson());
    rec.close();
    QCOMPARE(m.status(id).state, RunnerManager::State::UpdateAvailable);

    QSignalSpy gone(&m, &RunnerManager::uninstalled);
    QVERIFY(m.uninstall(id, &error));
    QVERIFY(gone.wait(10000));
    QVERIFY(gone.first().at(1).toBool());
    QCOMPARE(m.status(id).state, RunnerManager::State::NotInstalled);
}

void TestRunners::cancellingLeavesNoFailure()
{
    if (!QFileInfo::exists(m_zip))
        QSKIP("no zip tool on this machine");
    RunnerManager m;
    const QString id = QStringLiteral("kelvaroq");
    const QDir dir(Paths::runnerDir(id));
    QCOMPARE(m.status(id).state, RunnerManager::State::NotInstalled);
    for (int round = 0; round < 2; ++round) {
        QSignalSpy done(&m, &RunnerManager::installFinished);
        QString error;
        QVERIFY2(m.install(id, &error), qPrintable(error));
        QCOMPARE(m.status(id).state, RunnerManager::State::Installing);
        m.cancel();
        QVERIFY(done.count() == 1 || done.wait(10000));
        QVERIFY(!done.first().at(1).toBool());
        QCOMPARE(done.first().at(2).toString(), QStringLiteral("Cancelled."));
        // stopped, not failed: not installed, no reason to show, and nothing remembered for a restart
        const RunnerManager::Status s = m.status(id);
        QCOMPARE(s.state, RunnerManager::State::NotInstalled);
        QVERIFY2(s.error.isEmpty(), qPrintable(s.error));
        QVERIFY(!QFileInfo::exists(RunnerInstall::failurePath(id)));
        QVERIFY(!QFileInfo::exists(dir.filePath(QStringLiteral("staging"))));
        QVERIFY(!QFileInfo::exists(dir.filePath(QStringLiteral("runner.json"))));
        // and another attempt starts at once: cancelling does not keep the manager busy
        QVERIFY(m.busyId().isEmpty());
    }

    // the runner still installs normally afterwards
    QSignalSpy done(&m, &RunnerManager::installFinished);
    QString error;
    QVERIFY2(m.install(id, &error), qPrintable(error));
    QVERIFY(done.wait(30000));
    QVERIFY2(done.first().at(1).toBool(), qPrintable(done.first().at(2).toString()));
    QCOMPARE(m.status(id).state, RunnerManager::State::Installed);
    QSignalSpy gone(&m, &RunnerManager::uninstalled);
    QVERIFY(m.uninstall(id, &error));
    QVERIFY(gone.wait(10000));
    QVERIFY(gone.first().at(1).toBool());
}

void TestRunners::aFailedDownloadIsRetried()
{
    // a source that is not there: every attempt fails, and the last one fails the install
    RunnerSpec spec = *Runners::find(QStringLiteral("kelvaroq"));
    spec.id = QStringLiteral("kelvaroq-retried");
    spec.assetUrl = QUrl::fromLocalFile(m_dir.filePath(QStringLiteral("missing.zip"))).toString();
    RunnerInstall install(spec);
    QSignalSpy finished(&install, &RunnerInstall::finished);
    install.start();
    QVERIFY(finished.wait(30000));
    QVERIFY(!finished.first().at(0).toBool());
    QVERIFY2(finished.first().at(1).toString().contains(QStringLiteral("failed")), qPrintable(finished.first().at(1).toString()));
    QFile log(RunnerInstall::logPath(spec.id));
    QVERIFY(log.open(QIODevice::ReadOnly));
    const QString text = QString::fromUtf8(log.readAll());
    QCOMPARE(text.count(QStringLiteral("GET ") + spec.downloadUrl()), RunnerInstall::kDownloadAttempts);
    QCOMPARE(text.count(QStringLiteral("trying again")), RunnerInstall::kDownloadAttempts - 1);
    QVERIFY(!QFileInfo::exists(spec.installDir()));
}

void TestRunners::rejectsWrongHash()
{
    if (!QFileInfo::exists(m_zip))
        QSKIP("no zip tool on this machine");
    RunnerManager m;
    const QString id = QStringLiteral("victory-7900xtx");
    QSignalSpy done(&m, &RunnerManager::installFinished);
    QVERIFY(m.install(id));
    QVERIFY(done.wait(30000));
    QVERIFY(!done.first().at(1).toBool());
    QVERIFY(done.first().at(2).toString().contains(QStringLiteral("SHA-256")));
    const RunnerManager::Status s = m.status(id);
    QCOMPARE(s.state, RunnerManager::State::Failed);
    QVERIFY(s.executable.isEmpty());
    QVERIFY(!QFileInfo::exists(Runners::find(id)->installDir()));
    QVERIFY(QFileInfo::exists(m.installLogPath(id)));
    // a download whose bytes are not the pinned ones is not retried: the SHA-256 is the answer
    QFile log(m.installLogPath(id));
    QVERIFY(log.open(QIODevice::ReadOnly));
    QCOMPARE(QString::fromUtf8(log.readAll()).count(QStringLiteral("GET ")), 1);

    // a restart still says Failed, with the reason, never Not installed
    RunnerManager restarted;
    QCOMPARE(restarted.status(id).state, RunnerManager::State::Failed);
    QVERIFY(restarted.status(id).error.contains(QStringLiteral("SHA-256")));
}

void TestRunners::buildsFromSource()
{
    if (!m_haveGit)
        QSKIP("git is not installed");
    RunnerManager m;
    const QString id = QStringLiteral("victory-dexnom");
    QSignalSpy done(&m, &RunnerManager::installFinished);
    QVERIFY(m.install(id));
    QVERIFY(done.wait(60000));
    QVERIFY2(done.first().at(1).toBool(), qPrintable(done.first().at(2).toString()));
    const RunnerManager::Status s = m.status(id);
    QCOMPARE(s.state, RunnerManager::State::Installed);
    QCOMPARE(s.installedVersion, m_commit.left(7));
    QCOMPARE(QDir::cleanPath(QFileInfo(s.executable).absolutePath()), QDir::cleanPath(Runners::find(id)->installDir()));
    QVERIFY(QFileInfo(s.executable).size() > 0);
}

void TestRunners::fetchFallsBackAndChecksIds()
{
    if (!m_haveGit)
        QSKIP("git is not installed");
    resetLocations();
    const QString id = QStringLiteral("coolfox");
    const RunnerSpec* spec = Runners::find(id);
    QCOMPARE(spec->fetchUrls().size(), 3);
    RunnerManager m;
    QSignalSpy done(&m, &RunnerManager::installFinished);
    QString error;
    QVERIFY2(m.install(id, &error), qPrintable(error));
    QVERIFY(done.wait(60000));
    QVERIFY2(done.first().at(1).toBool(), qPrintable(done.first().at(2).toString()));
    QCOMPARE(m.status(id).state, RunnerManager::State::Installed);
    QFile log(m.installLogPath(id));
    QVERIFY(log.open(QIODevice::ReadOnly));
    const QString text = QString::fromUtf8(log.readAll());
    // the first source is gone, the second has another commit and is refused, the third is the pin
    QVERIFY2(text.contains(QStringLiteral("Could not fetch from ") + spec->fetchUrls().at(0)), qPrintable(text));
    QVERIFY2(text.contains(QStringLiteral("Refused %1: it has commit ").arg(spec->fetchUrls().at(1))), qPrintable(text));
    QVERIFY2(!text.contains(QStringLiteral("Refused ") + spec->fetchUrls().at(2)), qPrintable(text));
    QProcess head;
    head.setWorkingDirectory(spec->checkoutDir());
    head.start(QStringLiteral("git"), {QStringLiteral("rev-parse"), QStringLiteral("HEAD"), QStringLiteral("HEAD^{tree}")});
    QVERIFY(head.waitForFinished());
    QCOMPARE(QString::fromLatin1(head.readAllStandardOutput()).simplified().split(QLatin1Char(' ')), (QStringList{m_commit, m_tree}));

    // the pinned commit with a tree other than the pinned one is refused at every source
    RunnerSpec wrongTree = *spec;
    wrongTree.id = QStringLiteral("coolfox-wrong-tree");
    wrongTree.tree = QString(40, QLatin1Char('e'));
    RunnerInstall install(wrongTree);
    QSignalSpy finished(&install, &RunnerInstall::finished);
    install.start();
    QVERIFY(finished.wait(60000));
    QVERIFY(!finished.first().at(0).toBool());
    const QString why = finished.first().at(1).toString();
    QVERIFY2(why.startsWith(QStringLiteral("No source had cool_fox (codec_d) %1").arg(m_commit.left(7))), qPrintable(why));
    QVERIFY2(why.contains(QStringLiteral("has tree %1, not %2").arg(m_tree, wrongTree.tree)), qPrintable(why));
    QVERIFY(!QFileInfo::exists(wrongTree.installDir()));
}

QTEST_GUILESS_MAIN(TestRunners)
void TestRunners::stepProgramsUseTheStepsPath()
{
    // a cargo the install bootstrapped is on the step's PATH only, never on the launcher's
    QTemporaryDir dir;
    const QString bin = QDir(dir.path()).filePath(QStringLiteral("cargo/bin"));
    QVERIFY(QDir().mkpath(bin));
#ifdef Q_OS_WIN
    const QString cargo = QDir(bin).filePath(QStringLiteral("xsemu-test-cargo.exe"));
#else
    const QString cargo = QDir(bin).filePath(QStringLiteral("xsemu-test-cargo"));
#endif
    QFile f(cargo);
    QVERIFY(f.open(QIODevice::WriteOnly));
    f.write("#!/bin/sh\n");
    f.close();
    f.setPermissions(f.permissions() | QFileDevice::ExeOwner);
    QProcessEnvironment env;
    env.insert(QStringLiteral("PATH"), QDir::toNativeSeparators(bin));
    QCOMPARE(QFileInfo(RunnerInstall::resolveProgram(QStringLiteral("xsemu-test-cargo"), env)).canonicalFilePath(), QFileInfo(cargo).canonicalFilePath());
    QCOMPARE(RunnerInstall::resolveProgram(QStringLiteral("xsemu-test-missing"), env), QStringLiteral("xsemu-test-missing"));
    QCOMPARE(RunnerInstall::resolveProgram(cargo, {}), cargo);
}

void TestRunners::resetLocations()
{
    Paths::setRunnersFolder({});
    Paths::setRunnerLocations({});
}

void TestRunners::ensureBuilt(RunnerManager& m, const QString& id)
{
    if (m.status(id).state == RunnerManager::State::Installed)
        return;
    QSignalSpy done(&m, &RunnerManager::installFinished);
    QString error;
    QVERIFY2(m.install(id, &error), qPrintable(error));
    QVERIFY(done.wait(60000));
    QVERIFY2(done.first().at(1).toBool(), qPrintable(done.first().at(2).toString()));
    QCOMPARE(m.status(id).state, RunnerManager::State::Installed);
}

void TestRunners::locationPersistsAcrossRestarts()
{
    QTemporaryDir dir;
    const QString ini = dir.filePath(QStringLiteral("settings.ini"));
    const QString elsewhere = dir.filePath(QStringLiteral("D/Games"));
    const QString id = QStringLiteral("victory-7900xtx");
    {
        AppSettings settings(ini);
        RunnerManager m;
        m.setSettings(&settings);
        QCOMPARE(m.location(id), Paths::runnersDir());
        QSignalSpy moved(&m, &RunnerManager::relocated); // its failed install's log and record come along
        QString error;
        QVERIFY2(m.relocate(id, elsewhere, RunnerManager::Relocation::Move, &error), qPrintable(error));
        QVERIFY(moved.count() == 1 || moved.wait(30000));
        QVERIFY2(moved.first().at(1).toBool(), qPrintable(moved.first().at(2).toString()));
        QCOMPARE(Paths::runnerDir(id), QDir(elsewhere).filePath(id));
        QCOMPARE(m.status(id).state, RunnerManager::State::Failed);
    }
    resetLocations();
    QCOMPARE(Paths::runnerDir(id), QDir(Paths::runnersDir()).filePath(id));

    AppSettings settings(ini);
    RunnerManager m;
    m.setSettings(&settings);
    QCOMPARE(m.location(id), QDir::cleanPath(elsewhere));
    const RunnerSpec* spec = Runners::find(id);
    QCOMPARE(spec->dataDir(), QDir(elsewhere).filePath(id));
    QCOMPARE(spec->installDir(), QDir(elsewhere).filePath(id + QStringLiteral("/app")));
    QCOMPARE(m.status(id).state, RunnerManager::State::Failed);
    QSignalSpy moved(&m, &RunnerManager::relocated);
    QVERIFY(m.relocate(id, Paths::runnersDir(), RunnerManager::Relocation::Move));
    QVERIFY(moved.count() == 1 || moved.wait(30000));
    QVERIFY(settings.runnerLocations().isEmpty()); // back in the Runners folder: nothing to remember
    resetLocations();
}

void TestRunners::moveKeepsTheInstall()
{
    if (!m_haveGit)
        QSKIP("git is not installed");
    QTemporaryDir dir;
    const QString ini = dir.filePath(QStringLiteral("settings.ini"));
    const QString elsewhere = dir.filePath(QStringLiteral("E/runners"));
    const QString id = QStringLiteral("victory-dexnom");
    const RunnerSpec* spec = Runners::find(id);
    {
        AppSettings settings(ini);
        RunnerManager m;
        m.setSettings(&settings);
        ensureBuilt(m, id);
        const QString oldDir = spec->dataDir();
        QSignalSpy moved(&m, &RunnerManager::relocated);
        QString error;
        QVERIFY2(m.relocate(id, elsewhere, RunnerManager::Relocation::Move, &error), qPrintable(error));
        QVERIFY(moved.count() == 1 || moved.wait(30000));
        QVERIFY2(moved.first().at(1).toBool(), qPrintable(moved.first().at(2).toString()));

        const RunnerManager::Status s = m.status(id);
        QCOMPARE(s.state, RunnerManager::State::Installed);
        QCOMPARE(s.installedVersion, m_commit.left(7));
        QCOMPARE(s.executable, QDir(elsewhere).filePath(id + QStringLiteral("/bin/runner.tar")));
        QVERIFY(QFileInfo(s.executable).size() > 0);
        QVERIFY(m.isRunnable(id));
        QVERIFY(!QFileInfo::exists(oldDir));
        QVERIFY(QFileInfo::exists(QDir(elsewhere).filePath(id + QStringLiteral("/s/README"))));

        // what a launch gets: every token and the cache follow the new folder
        const auto tokens = spec->tokens(QStringLiteral("/g"), QStringLiteral("/g/Game.exe"), QStringLiteral("T"), QStringLiteral("lockhart"), 0);
        const auto token = [&tokens](const char* name) {
            for (const auto& [k, v] : tokens)
                if (k == QLatin1String(name))
                    return QDir::fromNativeSeparators(v);
            return QString();
        };
        QCOMPARE(token("runnerDir"), QDir(elsewhere).filePath(id));
        QCOMPARE(token("installDir"), QDir(elsewhere).filePath(id + QStringLiteral("/bin")));
        QCOMPARE(RunnerSpec::expand(QStringLiteral("{runnerDir}/data/victory.toml"), tokens), QDir::toNativeSeparators(QDir(elsewhere).filePath(id)) + QStringLiteral("/data/victory.toml"));
        QCOMPARE(QDir::fromNativeSeparators(spec->resolvedCacheDir()), QDir(elsewhere).filePath(id + QStringLiteral("/cache")));
        QCOMPARE(QDir::fromNativeSeparators(m.cachePath(id)), QDir(elsewhere).filePath(id + QStringLiteral("/cache")));
    }
    resetLocations();

    // a restart: the install is found where it went
    AppSettings settings(ini);
    RunnerManager m;
    m.setSettings(&settings);
    QCOMPARE(m.status(id).state, RunnerManager::State::Installed);
    QCOMPARE(m.status(id).executable, QDir(elsewhere).filePath(id + QStringLiteral("/bin/runner.tar")));

    // and back, so the tests after this one find it in the Runners folder
    QSignalSpy moved(&m, &RunnerManager::relocated);
    QVERIFY(m.relocate(id, Paths::runnersDir(), RunnerManager::Relocation::Move));
    QVERIFY(moved.count() == 1 || moved.wait(30000));
    QVERIFY(moved.first().at(1).toBool());
    QCOMPARE(m.status(id).state, RunnerManager::State::Installed);
    resetLocations();
}

void TestRunners::moveCopiesAcrossDrives()
{
    QTemporaryDir dir;
    const QString from = dir.filePath(QStringLiteral("C/runners/victory-dexnom"));
    const QString to = dir.filePath(QStringLiteral("D/runners/victory-dexnom"));
    QVERIFY(QDir().mkpath(from + QStringLiteral("/bin")));
    QVERIFY(QDir().mkpath(from + QStringLiteral("/t/x86_64-pc-windows-msvc/release")));
    QVERIFY(QDir().mkpath(from + QStringLiteral("/t/release/incremental"))); // empty folders come along too
    for (const auto& [path, size] : {qMakePair(QStringLiteral("bin/victory.exe"), 4096), qMakePair(QStringLiteral("runner.json"), 120),
                                     qMakePair(QStringLiteral("t/x86_64-pc-windows-msvc/release/victory-warm.exe"), 70000)}) {
        QFile f(QDir(from).filePath(path));
        QVERIFY(f.open(QIODevice::WriteOnly));
        f.write(QByteArray(size, 'x'));
    }
    qint64 last = -1;
    qint64 total = -1;
    QString error;
    QString note;
    QVERIFY2(RunnerManager::moveFolder(from, to, QStringLiteral("bin/victory.exe"), &error, &note,
                                       [&](qint64 d, qint64 t) {
                                           last = d;
                                           total = t;
                                       },
                                       false),
             qPrintable(error));
    QVERIFY(note.isEmpty());
    QCOMPARE(total, qint64(4096 + 120 + 70000));
    QCOMPARE(last, total);
    QVERIFY(!QFileInfo::exists(from));
    QCOMPARE(QFileInfo(QDir(to).filePath(QStringLiteral("bin/victory.exe"))).size(), qint64(4096));
    QCOMPARE(QFileInfo(QDir(to).filePath(QStringLiteral("t/x86_64-pc-windows-msvc/release/victory-warm.exe"))).size(), qint64(70000));
    QVERIFY(QFileInfo(QDir(to).filePath(QStringLiteral("t/release/incremental"))).isDir());

    // a missing source fails and leaves nothing half-copied behind
    QVERIFY(!RunnerManager::moveFolder(dir.filePath(QStringLiteral("nothing")), dir.filePath(QStringLiteral("D/x")), QStringLiteral("bin/victory.exe"), &error, &note, {}, false));
    QVERIFY(!QFileInfo::exists(dir.filePath(QStringLiteral("D/x"))));
}

void TestRunners::rebuildThere()
{
    if (!m_haveGit)
        QSKIP("git is not installed");
    QTemporaryDir dir;
    AppSettings settings(dir.filePath(QStringLiteral("settings.ini")));
    RunnerManager m;
    m.setSettings(&settings);
    const QString id = QStringLiteral("victory-dexnom");
    ensureBuilt(m, id);
    const QString oldDir = Runners::find(id)->dataDir();
    const QString elsewhere = dir.filePath(QStringLiteral("F"));
    QSignalSpy done(&m, &RunnerManager::installFinished);
    QString error;
    QVERIFY2(m.relocate(id, elsewhere, RunnerManager::Relocation::Rebuild, &error), qPrintable(error));
    QCOMPARE(m.status(id).state, RunnerManager::State::Installing);
    QVERIFY(done.wait(60000));
    QVERIFY2(done.first().at(1).toBool(), qPrintable(done.first().at(2).toString()));
    QCOMPARE(m.status(id).state, RunnerManager::State::Installed);
    QCOMPARE(m.status(id).executable, QDir(elsewhere).filePath(id + QStringLiteral("/bin/runner.tar")));
    QTRY_VERIFY_WITH_TIMEOUT(!QFileInfo::exists(oldDir), 10000);
    QCOMPARE(settings.runnerLocations().value(id), QDir::cleanPath(elsewhere));
    QSignalSpy gone(&m, &RunnerManager::uninstalled);
    QVERIFY(m.uninstall(id));
    QVERIFY(gone.wait(10000));
    QVERIFY(m.relocate(id, Paths::runnersDir(), RunnerManager::Relocation::Move));
    resetLocations();
}

void TestRunners::refusesWhileBusy()
{
    QTemporaryDir dir;
    AppSettings settings(dir.filePath(QStringLiteral("settings.ini")));
    RunnerManager m;
    m.setSettings(&settings);
    const QString id = QStringLiteral("victory-dexnom");
    const QString elsewhere = dir.filePath(QStringLiteral("G"));

    // a game or a pre-cache running with it (EmulatorController, WarmupController through Services)
    m.setInUseCheck([id](const QString& which) { return which == id ? QStringLiteral("VIctory (DexNom) is running Signal Garden; stop the game first.") : QString(); });
    QString error;
    QVERIFY(!m.relocate(id, elsewhere, RunnerManager::Relocation::Move, &error));
    QCOMPARE(error, QStringLiteral("VIctory (DexNom) is running Signal Garden; stop the game first."));
    QVERIFY(m.relocate(QStringLiteral("kelvaroq"), elsewhere, RunnerManager::Relocation::Move, &error)); // another runner is free
    QVERIFY(m.relocate(QStringLiteral("kelvaroq"), Paths::runnersDir(), RunnerManager::Relocation::Move, &error));
    m.setInUseCheck({});

    // installing or building
    if (m_haveGit) {
        QSignalSpy done(&m, &RunnerManager::installFinished);
        QVERIFY(m.install(id));
        QVERIFY(!m.relocate(id, elsewhere, RunnerManager::Relocation::Move, &error));
        QVERIFY2(error.contains(QStringLiteral("is building")), qPrintable(error));
        QVERIFY(!m.relocate(id, elsewhere, RunnerManager::Relocation::Rebuild, &error));
        const QByteArray env = qgetenv("XSEMU_RUNNERS_DIR");
        qunsetenv("XSEMU_RUNNERS_DIR");
        QVERIFY(!m.setRunnersFolder(elsewhere, &error)); // it bootstraps into the Runners folder
        qputenv("XSEMU_RUNNERS_DIR", env);
        QVERIFY2(error.contains(QStringLiteral("installing")), qPrintable(error));
        QVERIFY(done.wait(60000));
        QVERIFY(m.relocate(id, elsewhere, RunnerManager::Relocation::Move, &error));
        QVERIFY(m.busyReason(id).isEmpty() || m.status(id).state == RunnerManager::State::Moving);
        QSignalSpy moved(&m, &RunnerManager::relocated);
        if (m.status(id).state == RunnerManager::State::Moving) {
            QVERIFY(!m.install(id, &error)); // and while it moves
            QVERIFY2(error.contains(QStringLiteral("being moved")), qPrintable(error));
            QVERIFY(!m.uninstall(id, &error));
            QVERIFY(moved.wait(30000));
        }
        QVERIFY(m.relocate(id, Paths::runnersDir(), RunnerManager::Relocation::Move, &error));
        QTRY_COMPARE_WITH_TIMEOUT(m.status(id).state, RunnerManager::State::Installed, 30000);
    }

    // a folder that already has one, or its own folder
    QVERIFY(QDir().mkpath(QDir(elsewhere).filePath(id + QStringLiteral("/stuff"))));
    QVERIFY(!m.relocate(id, elsewhere, RunnerManager::Relocation::Move, &error));
    QVERIFY2(error.contains(QStringLiteral("already has")), qPrintable(error));
    QVERIFY(!m.relocate(id, Runners::find(id)->dataDir(), RunnerManager::Relocation::Move, &error));
    QVERIFY(!m.relocate(QStringLiteral("xsemu"), elsewhere, RunnerManager::Relocation::Move, &error));
    resetLocations();
}

void TestRunners::runnersFolderKeepsInstalledRunners()
{
    QTemporaryDir dir;
    const QByteArray env = qgetenv("XSEMU_RUNNERS_DIR");
    qunsetenv("XSEMU_RUNNERS_DIR");
    const QString ini = dir.filePath(QStringLiteral("settings.ini"));
    const QString a = dir.filePath(QStringLiteral("A"));
    const QString b = dir.filePath(QStringLiteral("B"));
    {
        AppSettings settings(ini);
        settings.setRunnersFolder(a);
        RunnerManager m;
        m.setSettings(&settings);
        QCOMPARE(Paths::runnersDir(), QDir::cleanPath(a));
        // one installed runner in A
        const RunnerSpec* k = Runners::find(QStringLiteral("kelvaroq"));
        QVERIFY(QDir().mkpath(QDir(k->installDir()).filePath(QStringLiteral("native"))));
        QFile exe(QDir(k->installDir()).filePath(k->executable));
        QVERIFY(exe.open(QIODevice::WriteOnly));
        exe.write("engine");
        exe.close();
        QFile rec(QDir(k->dataDir()).filePath(QStringLiteral("runner.json")));
        QVERIFY(rec.open(QIODevice::WriteOnly));
        rec.write(QJsonDocument(QJsonObject{{QStringLiteral("id"), k->id}, {QStringLiteral("version"), k->pinnedVersion()}, {QStringLiteral("executable"), k->executable}}).toJson());
        rec.close();
        QCOMPARE(m.status(k->id).state, RunnerManager::State::Installed);

        QString error;
        QVERIFY2(m.setRunnersFolder(b, &error), qPrintable(error));
        QCOMPARE(Paths::runnersDir(), QDir::cleanPath(b));
        QCOMPARE(m.status(k->id).state, RunnerManager::State::Installed); // it stayed in A
        QCOMPARE(k->dataDir(), QDir(a).filePath(k->id));
        QCOMPARE(Runners::find(QStringLiteral("victory-dexnom"))->dataDir(), QDir(b).filePath(QStringLiteral("victory-dexnom")));
    }
    resetLocations();
    {
        AppSettings settings(ini);
        RunnerManager m;
        m.setSettings(&settings);
        QCOMPARE(Paths::runnersDir(), QDir::cleanPath(b));
        QCOMPARE(m.status(QStringLiteral("kelvaroq")).state, RunnerManager::State::Installed);
        QCOMPARE(Runners::find(QStringLiteral("kelvaroq"))->dataDir(), QDir(a).filePath(QStringLiteral("kelvaroq")));
        QVERIFY(m.setRunnersFolder({}));
        QCOMPARE(Paths::runnersDir(), Paths::defaultRunnersDir());
        QVERIFY(settings.runnersFolder().isEmpty());
    }
    resetLocations();
    qputenv("XSEMU_RUNNERS_DIR", env);
    {
        AppSettings settings(ini);
        RunnerManager m;
        m.setSettings(&settings);
        QString error;
        QVERIFY(!m.setRunnersFolder(b, &error)); // the environment wins
        QVERIFY(error.contains(QStringLiteral("XSEMU_RUNNERS_DIR")));
    }
    resetLocations();
}

void TestRunners::freeSpaceWarning()
{
    constexpr qint64 gb = 1024LL * 1024 * 1024;
    QString asked;
    DiskSpace::setFreeBytesFunction([&asked](const QString& path) {
        asked = path;
        return 5 * gb;
    });
    DiskSpace::Check c = DiskSpace::checkRunnerFolder(m_dir.filePath(QStringLiteral("not/made/yet")));
    QCOMPARE(asked, m_dir.filePath(QStringLiteral("not/made/yet")));
    QVERIFY(c.low);
    QCOMPARE(c.freeBytes, 5 * gb);
    QVERIFY2(c.text.startsWith(QStringLiteral("Only 5.0 GB free")) && c.text.contains(QStringLiteral("20.0 GB")), qPrintable(c.text));
    DiskSpace::setFreeBytesFunction([](const QString&) { return DiskSpace::kRunnerLowBytes; });
    c = DiskSpace::checkRunnerFolder(m_dir.path());
    QVERIFY(!c.low);
    QVERIFY2(c.text.endsWith(QStringLiteral(" free on ") + c.drive), qPrintable(c.text));
    DiskSpace::setFreeBytesFunction([](const QString&) { return DiskSpace::kRunnerLowBytes - 1; });
    QVERIFY(DiskSpace::checkRunnerFolder(m_dir.path()).low);
    DiskSpace::setFreeBytesFunction([](const QString&) { return qint64(-1); });
    c = DiskSpace::checkRunnerFolder(m_dir.path());
    QVERIFY(!c.low);
    QVERIFY(c.text.contains(QStringLiteral("unknown")));
    DiskSpace::setFreeBytesFunction(nullptr);
    QVERIFY(DiskSpace::freeBytes(m_dir.filePath(QStringLiteral("not/made/yet"))) > 0); // the real drive, from the nearest folder that exists
}

void TestRunners::uninstallDeletesOffTheUiThread()
{
    if (!m_haveGit)
        QSKIP("git is not installed");
    RunnerManager m;
    const QString id = QStringLiteral("victory-dexnom");
    ensureBuilt(m, id);
    const QString dir = Runners::find(id)->dataDir();
    QVERIFY(QFileInfo::exists(QDir(dir).filePath(QStringLiteral("t"))));
    QSignalSpy gone(&m, &RunnerManager::uninstalled);
    QString error;
    QVERIFY2(m.uninstall(id, &error), qPrintable(error));
    // the delete runs on a worker; until it lands the runner is Uninstalling and nothing else may start
    QCOMPARE(m.status(id).state, RunnerManager::State::Uninstalling);
    QVERIFY(!m.isRunnable(id));
    QVERIFY(!m.install(id, &error));
    QVERIFY2(error.contains(QStringLiteral("being uninstalled")), qPrintable(error));
    QVERIFY(!m.uninstall(id, &error));
    QVERIFY(!m.relocate(id, m_dir.filePath(QStringLiteral("H")), RunnerManager::Relocation::Move, &error));
    QVERIFY(gone.wait(10000));
    QCOMPARE(gone.first().at(0).toString(), id);
    QVERIFY2(gone.first().at(1).toBool(), qPrintable(gone.first().at(2).toString()));
    QCOMPARE(m.status(id).state, RunnerManager::State::NotInstalled);
    QVERIFY(!QFileInfo::exists(dir));
    QVERIFY(m.busyReason(id).isEmpty());
}

void TestRunners::uninstallRefusedWhileInUse()
{
    RunnerManager m;
    const QString id = QStringLiteral("kelvaroq");
    const RunnerSpec* k = Runners::find(id);
    QVERIFY(QDir().mkpath(QDir(k->installDir()).filePath(QStringLiteral("native"))));
    QFile exe(QDir(k->installDir()).filePath(k->executable));
    QVERIFY(exe.open(QIODevice::WriteOnly));
    exe.write("engine");
    exe.close();
    QFile rec(QDir(k->dataDir()).filePath(QStringLiteral("runner.json")));
    QVERIFY(rec.open(QIODevice::WriteOnly));
    rec.write(QJsonDocument(QJsonObject{{QStringLiteral("id"), id}, {QStringLiteral("version"), k->pinnedVersion()}, {QStringLiteral("executable"), k->executable}}).toJson());
    rec.close();
    QCOMPARE(m.status(id).state, RunnerManager::State::Installed);

    m.setInUseCheck([id](const QString& which) { return which == id ? QStringLiteral("Kelvaroq is running Signal Garden; stop the game first.") : QString(); });
    QString error;
    QVERIFY(!m.uninstall(id, &error));
    QCOMPARE(error, QStringLiteral("Kelvaroq is running Signal Garden; stop the game first."));
    QCOMPARE(m.status(id).state, RunnerManager::State::Installed);
    QVERIFY(QFileInfo::exists(rec.fileName()));

    m.setInUseCheck({});
    QSignalSpy gone(&m, &RunnerManager::uninstalled);
    QVERIFY(m.uninstall(id, &error));
    QVERIFY(gone.wait(10000));
    QCOMPARE(m.status(id).state, RunnerManager::State::NotInstalled);
}

#include "tst_runners.moc"
