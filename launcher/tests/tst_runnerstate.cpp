// A runner whose build output has VIctory (DexNom)'s real layout, installed the way the user did it:
// a first build that fails, then one that succeeds. Every surface that shows a runner's install
// state is read right after the install finishes and again after a restart (new Services).
#include "DevFixtures.h"
#include "Library.h"
#include "Paths.h"
#include "Runner.h"
#include "RunnerManager.h"
#include "app/Services.h"
#include "pages/GameDetailsPage.h"
#include "pages/RunnerUi.h"
#include "pages/RunsWithPicker.h"
#include "shell/MainWindow.h"
#include "theme/Theme.h"
#include "widgets/Badge.h"

#include <QAbstractButton>
#include <QApplication>
#include <QDir>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QProcess>
#include <QSignalSpy>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QTest>
#include <QUrl>

namespace {

const QString kId = QStringLiteral("victory-dexnom");

bool run(const QString& program, const QStringList& args, const QString& dir)
{
    QProcess p;
    p.setWorkingDirectory(dir);
    p.start(program, args);
    return p.waitForFinished(30000) && p.exitStatus() == QProcess::NormalExit && p.exitCode() == 0;
}

// Every label and badge text a person can see under root.
QStringList shown(const QWidget* root)
{
    QStringList out;
    for (const QLabel* l : root->findChildren<QLabel*>()) {
        if (l->isVisibleTo(root) && !l->text().isEmpty())
            out << l->text();
    }
    for (const Badge* b : root->findChildren<Badge*>()) {
        if (b->isVisibleTo(root) && !b->text().isEmpty())
            out << b->text();
    }
    return out;
}

bool anyContains(const QStringList& texts, const QString& needle)
{
    for (const QString& t : texts) {
        if (t.contains(needle, Qt::CaseInsensitive))
            return true;
    }
    return false;
}

}

class TestRunnerState : public QObject {
    Q_OBJECT
private slots:
    void initTestCase();
    void dexnomReadsInstalledEverywhere();
    void changeLocationWaitsForItsRuns();
    void leftOverFilesOfferReinstallAndUninstall();
    void coolfoxAmbientSwitchIsOffByDefault();
    void the7900xtxPinIsFetchedFromSourcesThatExist();

private:
    struct Surfaces {
        QStringList card, details, picker, warmup;
    };
    Surfaces surfaces(Services& s, MainWindow& w);
    Game game() const;

    QTemporaryDir m_dir;
    QString m_commit;
};

void TestRunnerState::initTestCase()
{
    QStandardPaths::setTestModeEnabled(true);
    qputenv("XSEMU_RUNNERS_DIR", m_dir.filePath(QStringLiteral("runners")).toUtf8());
    qputenv("XSEMU_LAUNCHER_DATA", m_dir.filePath(QStringLiteral("launcher")).toUtf8());
    qputenv("XCL_DATA_ROOT", m_dir.filePath(QStringLiteral("xcl")).toUtf8());
    qputenv("XCL_DXIL_CACHE", m_dir.filePath(QStringLiteral("dxil")).toUtf8());
    if (QStandardPaths::findExecutable(QStringLiteral("git")).isEmpty())
        QSKIP("needs git");

    const QString repo = m_dir.filePath(QStringLiteral("repo"));
    QVERIFY(QDir().mkpath(repo));
    QFile readme(QDir(repo).filePath(QStringLiteral("Cargo.toml")));
    QVERIFY(readme.open(QIODevice::WriteOnly));
    readme.write("[workspace]\n");
    readme.close();
    const QStringList id = {QStringLiteral("-c"), QStringLiteral("user.name=t"), QStringLiteral("-c"), QStringLiteral("user.email=t@t")};
    QVERIFY(run(QStringLiteral("git"), {QStringLiteral("init"), QStringLiteral("-q")}, repo));
    QVERIFY(run(QStringLiteral("git"), {QStringLiteral("add"), QStringLiteral("Cargo.toml")}, repo));
    QVERIFY(run(QStringLiteral("git"), id + QStringList{QStringLiteral("commit"), QStringLiteral("-q"), QStringLiteral("-m"), QStringLiteral("one")}, repo));
    QVERIFY(run(QStringLiteral("git"), {QStringLiteral("config"), QStringLiteral("uploadpack.allowReachableSHA1InWant"), QStringLiteral("true")}, repo));
    QProcess p;
    p.setWorkingDirectory(repo);
    p.start(QStringLiteral("git"), {QStringLiteral("rev-parse"), QStringLiteral("HEAD")});
    QVERIFY(p.waitForFinished());
    m_commit = QString::fromLatin1(p.readAllStandardOutput()).trimmed();

    // cargo as DexNom's build runs it (tests/fake-cargo.cpp): the first run fails like a broken build does
#ifdef Q_OS_WIN
    const QString cargo = m_dir.filePath(QStringLiteral("cargo.exe"));
#else
    const QString cargo = m_dir.filePath(QStringLiteral("cargo"));
#endif
    QVERIFY(QFile::copy(QStringLiteral(XSEMU_FAKE_CARGO), cargo));

    QJsonArray buildSteps;
    buildSteps.append(QJsonArray{cargo, QStringLiteral("build"), QStringLiteral("--locked"), QStringLiteral("--release"), QStringLiteral("-p"), QStringLiteral("victory"),
                                 QStringLiteral("--features"), QStringLiteral("diagnostics"), QStringLiteral("--bins"), QStringLiteral("--target"),
                                 QStringLiteral("x86_64-pc-windows-msvc")});
    QJsonObject feed;
    feed.insert(kId, QJsonObject{{QStringLiteral("commit"), m_commit},
                                 {QStringLiteral("repo"), QUrl::fromLocalFile(repo).toString()},
                                 {QStringLiteral("buildSteps"), buildSteps},
                                 {QStringLiteral("artifacts"), QJsonObject{{QStringLiteral("x86_64-pc-windows-msvc/release/victory.exe"), QStringLiteral("victory.exe")}}},
                                 {QStringLiteral("executable"), QStringLiteral("victory.exe")}});
    QFile f(m_dir.filePath(QStringLiteral("feed.json")));
    QVERIFY(f.open(QIODevice::WriteOnly));
    f.write(QJsonDocument(feed).toJson());
    f.close();
    qputenv("XSEMU_RUNNER_FEED", f.fileName().toUtf8());

    QVERIFY(DevFixtures::materialize(QStringLiteral(XSEMU_FIXTURES_DIR "/games"), m_dir.filePath(QStringLiteral("games"))));
    Theme::instance().install(*qApp);
}

Game TestRunnerState::game() const
{
    for (const Game& g : Library::scan({m_dir.filePath(QStringLiteral("games"))}, 3, false)) {
        if (g.title() == QLatin1String("Signal Garden"))
            return g;
    }
    return {};
}

TestRunnerState::Surfaces TestRunnerState::surfaces(Services& s, MainWindow& w)
{
    Surfaces out;
    w.showPage(Page::Runners);
    QApplication::processEvents();
    if (QWidget* card = w.findChild<QWidget*>(QStringLiteral("runner-") + kId))
        out.card = shown(card);

    const Game g = game();
    w.showGameDetails(g);
    QApplication::processEvents();
    out.details = shown(w.gameDetailsPage());
    w.gameDetailsPage()->runsWith()->openPicker();
    QApplication::processEvents();
    if (QWidget* pop = w.findChild<QWidget*>(QStringLiteral("runsWithPicker"))) {
        if (QWidget* row = pop->findChild<QWidget*>(QStringLiteral("runsWith-") + kId))
            out.picker = shown(row);
        pop->close();
        QApplication::processEvents();
    }

    w.showPage(Page::Warmup);
    QApplication::processEvents();
    for (QWidget* page = w.gameDetailsPage()->parentWidget(); page; page = nullptr) {
        for (QWidget* child : page->findChildren<QWidget*>(Qt::FindDirectChildrenOnly)) {
            if (QByteArray(child->metaObject()->className()) == "WarmupPage")
                out.warmup = shown(child);
        }
    }
    Q_UNUSED(s);
    return out;
}

void TestRunnerState::dexnomReadsInstalledEverywhere()
{
    const Game g = game();
    QVERIFY(g.isValid());
    const RunnerSpec* spec = Runners::find(kId);
    QVERIFY(spec && spec->isPinned());

    {
        Services s;
        s.settings.setFirstLaunchDone(true);
        s.settings.setGameFolders({m_dir.filePath(QStringLiteral("games"))});
        s.settings.setGameOverride(g.titleId(), QStringLiteral("runner"), kId);
        QTRY_VERIFY_WITH_TIMEOUT(s.library.game(g.titleId()).isValid(), 10000);
        MainWindow w(s);
        w.resize(1280, 800);
        w.show();
        QVERIFY(QTest::qWaitForWindowExposed(&w));
        w.showGameDetails(g);
        QVERIFY(anyContains(shown(w.gameDetailsPage()), QStringLiteral("not installed")));

        // the first build fails: Failed with the reason everywhere, never "Not installed"
        QSignalSpy done(&s.runners, &RunnerManager::installFinished);
        QString error;
        QVERIFY2(s.runners.install(kId, &error), qPrintable(error));
        QVERIFY(done.wait(60000));
        QVERIFY(!done.first().at(1).toBool());
        QCOMPARE(s.runners.status(kId).state, RunnerManager::State::Failed);
        QVERIFY2(s.runners.status(kId).error.contains(QStringLiteral("101")), qPrintable(s.runners.status(kId).error));
        Surfaces failed = surfaces(s, w);
        QVERIFY2(anyContains(failed.card, QStringLiteral("Install failed")), qPrintable(failed.card.join(QLatin1Char('|'))));
        QVERIFY2(anyContains(failed.card, QStringLiteral("could not compile")), qPrintable(failed.card.join(QLatin1Char('|'))));
        QVERIFY2(!anyContains(failed.card, QStringLiteral("Not installed")), qPrintable(failed.card.join(QLatin1Char('|'))));
        QVERIFY2(anyContains(failed.details, QStringLiteral("install failed")), qPrintable(failed.details.join(QLatin1Char('|'))));
        QVERIFY2(anyContains(failed.picker, QStringLiteral("failed")), qPrintable(failed.picker.join(QLatin1Char('|'))));
        const QString log = QDir(spec->dataDir()).filePath(QStringLiteral("install.log"));
        QVERIFY(QFileInfo::exists(log));
        QWidget* card = w.findChild<QWidget*>(QStringLiteral("runner-") + kId);
        QVERIFY(card->findChild<QWidget*>(QStringLiteral("installLog-") + kId)->isVisibleTo(card));
        QFile logFile(log);
        QVERIFY(logFile.open(QIODevice::ReadOnly));
        const QByteArray logText = logFile.readAll();
        QVERIFY2(logText.contains("could not compile") && logText.contains("Install failed"), logText.constData());

        // details is on screen while the second build runs and finishes, as on the user's PC
        w.showGameDetails(g);
        done.clear();
        QVERIFY2(s.runners.install(kId, &error), qPrintable(error));
        QVERIFY(done.wait(60000));
        QVERIFY2(done.first().at(1).toBool(), qPrintable(done.first().at(2).toString()));
        QStringList details = shown(w.gameDetailsPage());
        QVERIFY2(!anyContains(details, QStringLiteral("not installed")), qPrintable(details.join(QLatin1Char('|'))));

        const RunnerManager::Status st = s.runners.status(kId);
        QCOMPARE(st.state, RunnerManager::State::Installed);
        QCOMPARE(st.installedVersion, m_commit.left(7));
        QCOMPARE(QDir::cleanPath(st.executable), QDir(spec->installDir()).filePath(QStringLiteral("victory.exe")));
        QVERIFY(QFileInfo::exists(st.executable));
        QVERIFY(s.runners.isRunnable(kId));
        QVERIFY(QFileInfo::exists(QDir(spec->targetDir()).filePath(QStringLiteral("x86_64-pc-windows-msvc/release/victory-warm.exe"))));
        QVERIFY(QFileInfo(QDir(spec->targetDir()).filePath(QStringLiteral("release/deps"))).isDir());
        QVERIFY(!QFileInfo::exists(QDir(spec->dataDir()).filePath(QStringLiteral("staging"))));
        QVERIFY(s.emulator.checkLaunch(g).problem != EmulatorController::LaunchCheck::Problem::RunnerNotInstalled);
        QVERIFY2(s.warmup.checkStart(g).isEmpty(), qPrintable(s.warmup.checkStart(g)));

        const Surfaces now = surfaces(s, w);
        QVERIFY2(anyContains(now.card, QStringLiteral("Installed")) && !anyContains(now.card, QStringLiteral("failed")), qPrintable(now.card.join(QLatin1Char('|'))));
        QVERIFY2(!anyContains(now.details, QStringLiteral("not installed")) && !anyContains(now.details, QStringLiteral("failed")), qPrintable(now.details.join(QLatin1Char('|'))));
        QVERIFY2(anyContains(now.picker, QStringLiteral("Installed")), qPrintable(now.picker.join(QLatin1Char('|'))));
        QVERIFY2(!anyContains(now.warmup, QStringLiteral("not installed")), qPrintable(now.warmup.join(QLatin1Char('|'))));
    }

    // a restart: the install record alone decides
    Services s;
    QTRY_VERIFY_WITH_TIMEOUT(s.library.game(g.titleId()).isValid(), 10000);
    MainWindow w(s);
    w.resize(1280, 800);
    w.show();
    QVERIFY(QTest::qWaitForWindowExposed(&w));
    QCOMPARE(s.runners.status(kId).state, RunnerManager::State::Installed);
    QVERIFY(s.runners.isRunnable(kId));
    const Surfaces later = surfaces(s, w);
    QVERIFY2(anyContains(later.card, QStringLiteral("Installed")) && !anyContains(later.card, QStringLiteral("failed")), qPrintable(later.card.join(QLatin1Char('|'))));
    QVERIFY2(!anyContains(later.details, QStringLiteral("not installed")), qPrintable(later.details.join(QLatin1Char('|'))));
    QVERIFY2(anyContains(later.picker, QStringLiteral("Installed")), qPrintable(later.picker.join(QLatin1Char('|'))));
    QVERIFY2(!anyContains(later.warmup, QStringLiteral("not installed")), qPrintable(later.warmup.join(QLatin1Char('|'))));
}

void TestRunnerState::changeLocationWaitsForItsRuns()
{
    const Game g = game();
    Services s;
    s.settings.setGameOverride(g.titleId(), QStringLiteral("runner"), kId);
    MainWindow w(s);
    w.resize(1280, 800);
    w.show();
    QVERIFY(QTest::qWaitForWindowExposed(&w));
    const QString elsewhere = m_dir.filePath(QStringLiteral("D/runners"));

    // a game running with it
    s.emulator.simulateForScreenshot(EmulatorController::State::Running, g, {});
    QString error;
    QVERIFY(!s.runners.relocate(kId, elsewhere, RunnerManager::Relocation::Move, &error));
    QCOMPARE(error, QStringLiteral("VIctory (DexNom) is running Signal Garden; stop the game first."));
    // the card's dialog says why and keeps Move disabled
    w.showPage(Page::Runners);
    QWidget* card = w.findChild<QWidget*>(QStringLiteral("runner-") + kId);
    auto* change = card->findChild<QAbstractButton*>(QStringLiteral("location-") + kId);
    QVERIFY(change && change->isVisible());
    QTest::mouseClick(change, Qt::LeftButton);
    QWidget* dialog = w.findChild<QWidget*>(QStringLiteral("changeLocation"));
    QVERIFY(dialog);
    QVERIFY(anyContains(shown(dialog), QStringLiteral("stop the game first")));
    QVERIFY(!dialog->findChild<QAbstractButton*>(QStringLiteral("locationConfirm"))->isEnabled());
    QMetaObject::invokeMethod(dialog, "dismiss");
    // and Uninstall waits too: refused, and the card's button is off with the reason
    QVERIFY(!s.runners.uninstall(kId, &error));
    QCOMPARE(error, QStringLiteral("VIctory (DexNom) is running Signal Garden; stop the game first."));
    QCOMPARE(s.runners.status(kId).state, RunnerManager::State::Installed);
    auto* uninstall = card->findChild<QAbstractButton*>(QStringLiteral("uninstall-") + kId);
    QVERIFY(uninstall && uninstall->isVisible());
    QVERIFY(!uninstall->isEnabled());
    QCOMPARE(uninstall->toolTip(), error);
    s.emulator.simulateForScreenshot(EmulatorController::State::Idle, g, {});
    QVERIFY(uninstall->isEnabled());

    // its pre-cache running
    WarmupController::Progress p;
    p.runnerId = kId;
    s.warmup.simulateForScreenshot(WarmupController::State::Running, g, p);
    QVERIFY(!s.runners.relocate(kId, elsewhere, RunnerManager::Relocation::Move, &error));
    QCOMPARE(error, QStringLiteral("VIctory (DexNom) is pre-caching Signal Garden; cancel the pre-cache first."));
    QVERIFY(!s.runners.uninstall(kId, &error));
    QCOMPARE(error, QStringLiteral("VIctory (DexNom) is pre-caching Signal Garden; cancel the pre-cache first."));
    QVERIFY(!uninstall->isEnabled());
    s.warmup.simulateForScreenshot(WarmupController::State::Idle, g, {});

    // free again: Move from the dialog, and every surface reads Installed at the new place
    QVERIFY(s.runners.busyReason(kId).isEmpty());
    QSignalSpy moved(&s.runners, &RunnerManager::relocated);
    RunnerUi::changeLocation(&w, s, kId, elsewhere);
    dialog = w.findChildren<QWidget*>(QStringLiteral("changeLocation")).last();
    QVERIFY(dialog->findChild<QWidget*>(QStringLiteral("locationMove"))->isVisibleTo(dialog));
    auto* confirm = dialog->findChild<QAbstractButton*>(QStringLiteral("locationConfirm"));
    QVERIFY(confirm->isEnabled());
    QTest::mouseClick(confirm, Qt::LeftButton);
    QVERIFY(moved.count() == 1 || moved.wait(30000));
    QVERIFY2(moved.first().at(1).toBool(), qPrintable(moved.first().at(2).toString()));
    QCOMPARE(s.runners.status(kId).state, RunnerManager::State::Installed);
    QCOMPARE(s.runners.status(kId).executable, QDir(elsewhere).filePath(kId + QStringLiteral("/bin/victory.exe")));
    const Surfaces after = surfaces(s, w);
    QVERIFY2(anyContains(after.card, QStringLiteral("Installed")), qPrintable(after.card.join(QLatin1Char('|'))));
    QVERIFY2(!anyContains(after.details, QStringLiteral("not installed")), qPrintable(after.details.join(QLatin1Char('|'))));
    QVERIFY(s.emulator.checkLaunch(g).problem != EmulatorController::LaunchCheck::Problem::RunnerNotInstalled);
    QVERIFY(s.warmup.checkStart(g).isEmpty());

    // an open picker follows a location change (every runner's state at once) as one picker
    w.showGameDetails(g);
    w.gameDetailsPage()->runsWith()->openPicker();
    s.settings.setRunnerLocation(QStringLiteral("kelvaroq"), m_dir.filePath(QStringLiteral("K")));
    QTest::qWait(50);
    int open = 0;
    for (QWidget* pop : w.findChildren<QWidget*>(QStringLiteral("runsWithPicker")))
        open += pop->isVisible() ? 1 : 0;
    QCOMPARE(open, 1);
    s.settings.setRunnerLocation(QStringLiteral("kelvaroq"), {});
}

void TestRunnerState::leftOverFilesOfferReinstallAndUninstall()
{
    // an uninstall that could not delete everything keeps runner.json; its program is gone
    const RunnerSpec* spec = Runners::find(kId);
    {
        Services before;
        QCOMPARE(before.runners.status(kId).state, RunnerManager::State::Installed);
        QVERIFY(QFile::remove(before.runners.status(kId).executable));
    }
    QVERIFY(QFileInfo::exists(QDir(spec->dataDir()).filePath(QStringLiteral("runner.json"))));

    Services s;
    const RunnerManager::Status st = s.runners.status(kId);
    QCOMPARE(st.state, RunnerManager::State::Failed);
    QCOMPARE(st.error, QStringLiteral("Some files of VIctory (DexNom) were left in %1. Install it again over them, or uninstall to delete them.")
                           .arg(QDir::toNativeSeparators(spec->dataDir())));
    QVERIFY(!s.runners.isRunnable(kId));
    MainWindow w(s);
    w.resize(1280, 800);
    w.show();
    QVERIFY(QTest::qWaitForWindowExposed(&w));
    w.showPage(Page::Runners);
    QApplication::processEvents();
    QWidget* card = w.findChild<QWidget*>(QStringLiteral("runner-") + kId);
    QVERIFY(card);
    auto* install = card->findChild<QAbstractButton*>(QStringLiteral("install-") + kId);
    auto* uninstall = card->findChild<QAbstractButton*>(QStringLiteral("uninstall-") + kId);
    QVERIFY(install && install->isVisibleTo(card) && install->isEnabled());
    QCOMPARE(install->text(), QStringLiteral("Reinstall"));
    QVERIFY(uninstall && uninstall->isVisibleTo(card) && uninstall->isEnabled());
    QVERIFY2(anyContains(shown(card), QStringLiteral("were left in")), qPrintable(shown(card).join(QLatin1Char('|'))));

    // Reinstall builds over the left-over files
    QSignalSpy done(&s.runners, &RunnerManager::installFinished);
    QString error;
    QVERIFY2(s.runners.install(kId, &error), qPrintable(error));
    QVERIFY(done.wait(60000));
    QVERIFY2(done.first().at(1).toBool(), qPrintable(done.first().at(2).toString()));
    QCOMPARE(s.runners.status(kId).state, RunnerManager::State::Installed);
    QVERIFY(s.runners.isRunnable(kId));
    QApplication::processEvents();
    QVERIFY(!install->isVisibleTo(card));
}

void TestRunnerState::coolfoxAmbientSwitchIsOffByDefault()
{
    Services s;
    MainWindow w(s);
    w.resize(1280, 800);
    w.show();
    QVERIFY(QTest::qWaitForWindowExposed(&w));
    w.showPage(Page::Runners);
    QApplication::processEvents();
    QWidget* card = w.findChild<QWidget*>(QStringLiteral("runner-coolfox"));
    QVERIFY(card);
    QVERIFY(anyContains(shown(card), QStringLiteral("Author's ambient light")));
    QVERIFY(anyContains(shown(card), QStringLiteral("reports the game's licence as active")));
    QVERIFY(anyContains(shown(card), QStringLiteral("always runs the game as a Series S")));
    auto* toggle = card->findChild<QAbstractButton*>(QStringLiteral("option-coolfox-ambient"));
    QVERIFY(toggle && toggle->isVisibleTo(card));
    QVERIFY(!toggle->isChecked());
    QVERIFY(!s.settings.runnerOptions(QStringLiteral("coolfox")).value(QStringLiteral("ambient"), false));
    QTest::mouseClick(toggle, Qt::LeftButton);
    QVERIFY(toggle->isChecked());
    QVERIFY(s.settings.runnerOptions(QStringLiteral("coolfox")).value(QStringLiteral("ambient"), false));
    QTest::mouseClick(toggle, Qt::LeftButton);
    QVERIFY(!s.settings.runnerOptions(QStringLiteral("coolfox")).value(QStringLiteral("ambient"), true));
}

void TestRunnerState::the7900xtxPinIsFetchedFromSourcesThatExist()
{
    // The fork's own repository (and its author's account) is gone; the curated entry must therefore
    // name the audited commit, its tree, and more than one source to fetch it from. The feed above
    // re-pins VIctory (DexNom) only, so this reads the curated list as it ships.
    const RunnerSpec* r = Runners::find(QStringLiteral("victory-7900xtx"));
    QVERIFY(r);
    QCOMPARE(r->install, RunnerSpec::Install::BuildFromSource);
    QCOMPARE(r->commit.size(), 40);
    QCOMPARE(r->tree.size(), 40);
    QVERIFY(r->remotes.size() >= 2);
    for (const QString& url : r->fetchUrls()) {
        QVERIFY2(url.startsWith(QLatin1String("https://")) && url.endsWith(QLatin1String(".git")), qPrintable(url));
        QVERIFY2(url != r->cloneUrl(), qPrintable(url)); // the curated repo URL is the deleted one
    }
    // the build it pins is the fork's own: the diag profile and the diagnostics feature
    QCOMPARE(r->buildSteps.size(), 1);
    QVERIFY(r->buildSteps.first().contains(QStringLiteral("diag")));
    QVERIFY(r->buildSteps.first().contains(QStringLiteral("diagnostics")));
    QCOMPARE(r->artifacts, (QList<QPair<QString, QString>>{{QStringLiteral("diag/victory.exe"), QStringLiteral("victory.exe")}}));

    // and the other source builds name their audited tree too
    for (const QString& id : {QStringLiteral("victory-dexnom"), QStringLiteral("coolfox")}) {
        const RunnerSpec* s = Runners::find(id);
        QVERIFY(s);
        if (s->commit != m_commit) // the feed re-pins VIctory (DexNom) here
            QCOMPARE(s->tree.size(), 40);
    }
}

QTEST_MAIN(TestRunnerState)
#include "tst_runnerstate.moc"
