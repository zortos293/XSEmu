// The in-game view end to end: MainWindow launches fake-xsemu through EmulatorController with
// --host-hwnd and --control-pipe, and the test drives it with real clicks on the game bar and the
// exit screens, plus the launcher's own F1/F3/F11/Esc keys. A third-party runner (fake-xsemu playing
// Kelvaroq) is never embedded: it keeps its own window and the floating bar follows it.
#include "DevFixtures.h"
#include "EmulatorController.h"
#include "LaunchConfig.h"
#include "Library.h"
#include "Runner.h"
#include "RunnerManager.h"
#include "app/Services.h"
#include "shell/FloatingBar.h"
#include "shell/InGameView.h"
#include "pages/GameDetailsPage.h"
#include "shell/LaunchErrors.h"
#include "shell/MainWindow.h"
#include "theme/Theme.h"
#include "widgets/SegmentedControl.h"
#include "widgets/Text.h"

#include <QAbstractButton>
#include <QApplication>
#include <QComboBox>
#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QStandardPaths>
#include <QTemporaryDir>
#include <QTest>

using State = EmulatorController::State;

class TestInGame : public QObject {
    Q_OBJECT
private slots:
    void initTestCase();
    void init();
    void cleanup();
    void playPauseOverlayFullscreenBack();
    void stopReturnsToLibrary();
    void runnerKeepsItsOwnWindow();
    void crashScreen_data();
    void crashScreen();
    void xsemuWarmupIsOff();
    void warmupFailureOffersContinueAndPlay();
    void qualityPresetsShowAndApply();

private:
    QAbstractButton* button(const char* name) const;
    void click(const char* name);
    Game fixtureGame(const QString& title) const;

    QTemporaryDir m_fixtures;
    std::unique_ptr<QTemporaryDir> m_dir;
    std::unique_ptr<Services> m_s;
    std::unique_ptr<MainWindow> m_w;
    GameList m_games;
};

void TestInGame::initTestCase()
{
    QStandardPaths::setTestModeEnabled(true);
    qputenv("XSEMU_EMULATOR", XSEMU_FAKE_EMULATOR_PATH);
    qputenv("FAKE_XSEMU_TICK_MS", "5");
    qputenv("XSEMU_RUNNERS_DIR", m_fixtures.filePath(QStringLiteral("runners")).toUtf8());
    QFile feed(m_fixtures.filePath(QStringLiteral("feed.json")));
    QVERIFY(feed.open(QIODevice::WriteOnly));
    feed.write(R"({"kelvaroq": {"keep": true}})");
    feed.close();
    qputenv("XSEMU_RUNNER_FEED", feed.fileName().toUtf8());
    QVERIFY(DevFixtures::materialize(QStringLiteral(XSEMU_FIXTURES_DIR "/games"), m_fixtures.filePath(QStringLiteral("games"))));
    m_games = Library::scan({m_fixtures.filePath(QStringLiteral("games"))}, 3, false);
    QCOMPARE(m_games.size(), 4);
    Theme::instance().install(*qApp);
}

void TestInGame::init()
{
    m_dir = std::make_unique<QTemporaryDir>();
    qputenv("XCL_DATA_ROOT", m_dir->filePath(QStringLiteral("xcl")).toUtf8());
    qputenv("XSEMU_LAUNCHER_DATA", m_dir->filePath(QStringLiteral("launcher")).toUtf8()); // settings start fresh for each test
    qputenv("FAKE_XSEMU_SCENARIO", "ok");
    m_s = std::make_unique<Services>();
    m_s->settings.setLaunchNeedMb(0);
    m_s->settings.setFirstLaunchDone(true);
    m_s->settings.setGameFolders({m_fixtures.filePath(QStringLiteral("games"))});
    m_w = std::make_unique<MainWindow>(*m_s);
    m_w->resize(1280, 800);
    m_w->show();
    QVERIFY(QTest::qWaitForWindowExposed(m_w.get()));
}

void TestInGame::cleanup()
{
    if (m_s->emulator.isActive()) {
        m_s->emulator.stop();
        QTRY_COMPARE_WITH_TIMEOUT(m_s->emulator.state(), State::Idle, 10000);
    }
    m_w.reset();
    m_s.reset();
    m_dir.reset();
}

QAbstractButton* TestInGame::button(const char* name) const
{
    return m_w->inGameView()->findChild<QAbstractButton*>(QLatin1String(name));
}

void TestInGame::click(const char* name)
{
    QAbstractButton* b = button(name);
    QVERIFY2(b, name);
    QTRY_VERIFY2_WITH_TIMEOUT(b->isVisible() && b->isEnabled(), qPrintable(QStringLiteral("%1 ingame=%2 view=%3 win=%4 visibleToView=%5 st=%6").arg(QLatin1String(name)).arg(m_w->isInGame()).arg(m_w->inGameView()->isVisible()).arg(m_w->isVisible()).arg(b->isVisibleTo(m_w->inGameView())).arg(int(m_s->emulator.state()))), 5000);
    QTest::mouseClick(b, Qt::LeftButton);
}

Game TestInGame::fixtureGame(const QString& title) const
{
    for (const Game& g : m_games) {
        if (g.title() == title)
            return g;
    }
    return {};
}

void TestInGame::playPauseOverlayFullscreenBack()
{
    EmulatorController& emu = m_s->emulator;
    m_w->play(fixtureGame(QStringLiteral("Signal Garden")));
    QVERIFY(m_w->isInGame());
    QTRY_COMPARE_WITH_TIMEOUT(emu.state(), State::Running, 10000);
    QTRY_VERIFY_WITH_TIMEOUT(emu.stats().fps.has_value(), 5000);

    click("barPause");
    QTRY_COMPARE_WITH_TIMEOUT(emu.state(), State::Paused, 5000);
    click("pausedResume");
    QTRY_COMPARE_WITH_TIMEOUT(emu.state(), State::Running, 5000);

    QVERIFY(!emu.overlayOn());
    click("barOverlay");
    QTRY_VERIFY_WITH_TIMEOUT(emu.overlayOn(), 5000);
    QVERIFY(button("barOverlay")->isChecked());
    m_w->inGameView()->handleKey(QStringLiteral("F3"));
    QTRY_VERIFY_WITH_TIMEOUT(!emu.overlayOn(), 5000);

    m_w->inGameView()->handleKey(QStringLiteral("F1"));
    QTRY_COMPARE_WITH_TIMEOUT(emu.state(), State::Paused, 5000);
    m_w->inGameView()->handleKey(QStringLiteral("F1"));
    QTRY_COMPARE_WITH_TIMEOUT(emu.state(), State::Running, 5000);

    click("barFullscreen");
    QTRY_VERIFY_WITH_TIMEOUT(m_w->inGameView()->isFullScreen(), 5000);
    QTRY_VERIFY_WITH_TIMEOUT(m_w->inGameView()->findChild<QWidget*>(QStringLiteral("keyHint"))->isVisible(), 5000);
    m_w->inGameView()->handleKey(QStringLiteral("Escape"));
    QTRY_VERIFY_WITH_TIMEOUT(!m_w->inGameView()->isFullScreen(), 5000);

    // Back to Library keeps the game, paused, under Now playing.
    click("barBack");
    QVERIFY(!m_w->isInGame());
    QTRY_COMPARE_WITH_TIMEOUT(emu.state(), State::Paused, 5000);
    QVERIFY(emu.isActive());
}

void TestInGame::runnerKeepsItsOwnWindow()
{
    const RunnerSpec* spec = Runners::find(QStringLiteral("kelvaroq"));
    QVERIFY(spec);
    const QString exe = QDir(spec->installDir()).filePath(spec->executable);
    QDir().mkpath(QFileInfo(exe).absolutePath());
    QFile::remove(exe);
    QVERIFY(QFile::copy(QStringLiteral(XSEMU_FAKE_EMULATOR_PATH), exe));
    QFile rec(QDir(spec->dataDir()).filePath(QStringLiteral("runner.json")));
    QVERIFY(rec.open(QIODevice::WriteOnly | QIODevice::Truncate));
    rec.write(QJsonDocument(QJsonObject{{QStringLiteral("id"), spec->id}, {QStringLiteral("version"), spec->pinnedVersion()}, {QStringLiteral("executable"), spec->executable}}).toJson());
    rec.close();

    const Game game = fixtureGame(QStringLiteral("Signal Garden"));
    m_s->settings.setGameOverride(game.titleId(), QStringLiteral("runner"), QStringLiteral("kelvaroq"));
    EmulatorController& emu = m_s->emulator;
    m_w->play(game);
    QVERIFY(emu.isActive());
    QCOMPARE(emu.runner().id, QStringLiteral("kelvaroq"));
    QVERIFY(!m_w->isInGame()); // never embedded: the library stays, the game gets its own window
    QVERIFY(!emu.canPause() && !emu.canOverlay());
    if (QGuiApplication::platformName() != QLatin1String("xcb") && QGuiApplication::platformName() != QLatin1String("windows")) {
        emu.stop();
        QTRY_COMPARE_WITH_TIMEOUT(emu.state(), State::Idle, 15000);
        QSKIP("finding the runner's own window needs X11 or Windows");
    }
    QTRY_COMPARE_WITH_TIMEOUT(emu.state(), State::Running, 10000);
    QVERIFY(emu.gameWindow() != 0);
    FloatingBar* bar = m_w->floatingBar();
    QTRY_VERIFY_WITH_TIMEOUT(bar->isVisible(), 5000);
    auto* stop = bar->findChild<QAbstractButton*>(QStringLiteral("floatingStop"));
    QVERIFY(stop);
    QTest::mouseClick(stop, Qt::LeftButton);
    QTRY_COMPARE_WITH_TIMEOUT(emu.state(), State::Idle, 15000);
    QTRY_VERIFY_WITH_TIMEOUT(!bar->isVisible(), 5000);
}

void TestInGame::stopReturnsToLibrary()
{
    qputenv("FAKE_XSEMU_SCENARIO", "slow_boot");
    m_w->play(fixtureGame(QStringLiteral("Signal Garden")));
    QCOMPARE(m_s->emulator.state(), State::Booting);
    QTRY_VERIFY_WITH_TIMEOUT(m_s->emulator.hello().has_value(), 10000);
    QCOMPARE(m_s->emulator.state(), State::Booting);
    click("barStop");
    QTRY_COMPARE_WITH_TIMEOUT(m_s->emulator.state(), State::Idle, 10000);
    QTRY_VERIFY_WITH_TIMEOUT(!m_w->isInGame(), 5000);
    QVERIFY(!m_w->inGameView()->showingExit());
}

void TestInGame::crashScreen_data()
{
    QTest::addColumn<QByteArray>("scenario");
    QTest::addColumn<QString>("relaunchText");
    QTest::newRow("crash") << QByteArray("crash") << QStringLiteral("Relaunch");
    QTest::newRow("device_removed") << QByteArray("device_removed") << QStringLiteral("Relaunch");
    QTest::newRow("files_missing") << QByteArray("files_missing") << QStringLiteral("Try again");
}

void TestInGame::crashScreen()
{
    QFETCH(QByteArray, scenario);
    QFETCH(QString, relaunchText);
    qputenv("FAKE_XSEMU_SCENARIO", scenario);
    const Game game = fixtureGame(QStringLiteral("Coastline Drive"));
    m_w->play(game);
    QTRY_COMPARE_WITH_TIMEOUT(m_s->emulator.state(), State::Idle, 15000);
    QVERIFY(m_w->isInGame());
    QTRY_VERIFY_WITH_TIMEOUT(m_w->inGameView()->showingExit(), 5000);
    QCOMPARE(button("exitRelaunch")->text(), relaunchText);

    // Restart from the exit screen launches the same game again.
    qputenv("FAKE_XSEMU_SCENARIO", "ok");
    click("exitRelaunch");
    QTRY_COMPARE_WITH_TIMEOUT(m_s->emulator.state(), State::Running, 10000);
    QCOMPARE(m_s->emulator.game().titleId(), game.titleId());
    QVERIFY(!m_w->inGameView()->showingExit());
    m_s->emulator.stop();
    QTRY_VERIFY_WITH_TIMEOUT(!m_w->isInGame(), 10000);
}

// The XSEmu warm-up ships turned off: it can't start, Play never asks for it, and the Warm-up page and
// Game details offer none, even for a game whose last warm-up failed (which asks once it is on).
void TestInGame::xsemuWarmupIsOff()
{
    const Game garden = fixtureGame(QStringLiteral("Signal Garden"));
    QTRY_VERIFY_WITH_TIMEOUT(m_s->library.game(garden.titleId()).isValid(), 10000);
    QVERIFY(!m_s->settings.xsemuWarmup());
    WarmupRun failed;
    failed.titleId = garden.titleId();
    failed.titleName = garden.title();
    failed.result = WarmupRun::Result::Failed;
    failed.finishedAt = QDateTime::currentDateTime();
    m_s->settings.addRecentWarmup(failed);

    QString error;
    QVERIFY(!m_s->warmup.start(false, garden, &error));
    QCOMPARE(error, QStringLiteral("The XSEmu Core warm-up is turned off in this build."));
    QVERIFY(m_s->gate.isFree());
    QCOMPARE(m_s->warmup.state(), WarmupController::State::Idle);

    QVERIFY(!LaunchErrors::shadersCold(*m_s, garden));
    bool played = false;
    bool warmedUp = false;
    LaunchErrors::confirmPlay(m_w.get(), *m_s, garden, [&] { played = true; }, [&] { warmedUp = true; });
    QVERIFY(played);
    QVERIFY(!warmedUp);

    const auto visibleText = [](const QWidget* root) {
        QStringList out;
        for (const QAbstractButton* b : root->findChildren<QAbstractButton*>()) {
            if (b->isVisibleTo(root))
                out << b->text();
        }
        for (const QLabel* l : root->findChildren<QLabel*>()) {
            if (l->isVisibleTo(root))
                out << l->text();
        }
        return out.join(QLatin1Char('\n'));
    };
    m_w->showGameDetails(garden);
    QApplication::processEvents();
    const QString details = visibleText(m_w->gameDetailsPage());
    QVERIFY2(!details.contains(QLatin1String("Warm up")) && !details.contains(QLatin1String("Warm-up failed")), qPrintable(details));
    m_w->showPage(Page::Warmup);
    QApplication::processEvents();
    QWidget* warmupPage = nullptr;
    for (QWidget* child : m_w->gameDetailsPage()->parentWidget()->findChildren<QWidget*>(Qt::FindDirectChildrenOnly)) {
        if (QByteArray(child->metaObject()->className()) == "WarmupPage")
            warmupPage = child;
    }
    QVERIFY(warmupPage);
    const QString page = visibleText(warmupPage);
    QVERIFY2(page.contains(QLatin1String("The XSEmu Core warm-up is turned off in this build.")), qPrintable(page));
    QVERIFY2(!page.contains(QLatin1String("Start warm-up")) && !page.contains(QLatin1String("Warm up ")) && !page.contains(QLatin1String("Retry")), qPrintable(page));

    m_s->settings.setXsemuWarmup(true);
    QVERIFY(LaunchErrors::shadersCold(*m_s, garden));
}

// A warm-up that failed leaves the page usable: what is left is named, Continue continues that pass,
// a full pass is one click away, and Play stays available (the gate is free once the pass ended).
void TestInGame::warmupFailureOffersContinueAndPlay()
{
    const Game garden = fixtureGame(QStringLiteral("Signal Garden"));
    QTRY_VERIFY_WITH_TIMEOUT(m_s->library.game(garden.titleId()).isValid(), 10000);
    m_s->settings.setXsemuWarmup(true);

    WarmupRun run;
    run.titleId = garden.titleId();
    run.titleName = garden.title();
    run.done = 2180;
    run.total = 3120;
    run.tookS = 96;
    run.exitCode = 3;
    run.result = WarmupRun::Result::Failed;
    run.finishedAt = QDateTime::currentDateTime();
    WarmupController::Progress progress;
    progress.targets = 3120;
    progress.done = 2180;
    progress.total = 3120;
    progress.phase = QStringLiteral("bound");
    progress.passDone = 2180;
    m_s->settings.clearRecentWarmups();
    m_s->settings.addRecentWarmup(run);
    m_s->warmup.simulateForScreenshot(WarmupController::State::Failed, garden, progress,
                                      QStringLiteral("the driver compile crashed (0xC0000005)"), run);
    m_w->showPage(Page::Warmup);
    QApplication::processEvents();

    QWidget* warmupPage = nullptr;
    for (QWidget* child : m_w->gameDetailsPage()->parentWidget()->findChildren<QWidget*>(Qt::FindDirectChildrenOnly)) {
        if (QByteArray(child->metaObject()->className()) == "WarmupPage")
            warmupPage = child;
    }
    QVERIFY(warmupPage);
    QStringList shown;
    for (const QAbstractButton* b : warmupPage->findChildren<QAbstractButton*>()) {
        if (b->isVisibleTo(warmupPage) && b->isEnabled())
            shown << b->text();
    }
    QStringList labels;
    for (const QLabel* l : warmupPage->findChildren<QLabel*>()) {
        if (l->isVisibleTo(warmupPage))
            labels << l->text();
    }
    const QString page = labels.join(QLatin1Char('\n'));
    // a crash has no done line, so the card says where it stopped rather than claiming a total
    QVERIFY2(page.contains(QLatin1String("stopped in bound pipelines at 2,180 of 3,120")), qPrintable(page));
    QVERIFY2(page.contains(QLatin1String("XSEmu Core exited with code 3")), qPrintable(page));
    QVERIFY2(page.contains(QLatin1String("Continue goes on from where this pass stopped")), qPrintable(page));
    // Continue, Play and a full pass are all offered, and nothing is stuck: no Cancel, no running bar
    QVERIFY2(shown.contains(QStringLiteral("Continue")), qPrintable(shown.join(QLatin1Char('|'))));
    QVERIFY2(shown.contains(QStringLiteral("Play Signal Garden")), qPrintable(shown.join(QLatin1Char('|'))));
    QVERIFY2(!shown.contains(QStringLiteral("Cancel")), qPrintable(shown.join(QLatin1Char('|'))));
    QVERIFY(m_s->gate.isFree());

    // the same for a pass the 20-minute budget stopped: Continue carries on, and a full pass is offered
    // because that is the one result a full pass answers
    run.result = WarmupRun::Result::StoppedPartWay;
    run.left = 3120 - 2180;
    m_s->settings.clearRecentWarmups();
    m_s->settings.addRecentWarmup(run);
    m_s->warmup.simulateForScreenshot(WarmupController::State::StoppedPartWay, garden, progress, {}, run);
    QApplication::processEvents();
    shown.clear();
    labels.clear();
    for (const QAbstractButton* b : warmupPage->findChildren<QAbstractButton*>()) {
        if (b->isVisibleTo(warmupPage) && b->isEnabled())
            shown << b->text();
    }
    for (const QLabel* l : warmupPage->findChildren<QLabel*>()) {
        if (l->isVisibleTo(warmupPage))
            labels << l->text();
    }
    const QString stopped = labels.join(QLatin1Char('\n'));
    QVERIFY2(stopped.contains(QStringLiteral("2,180 built · 940 left")), qPrintable(stopped));
    QVERIFY2(shown.contains(QStringLiteral("Continue")), qPrintable(shown.join(QLatin1Char('|'))));
    QVERIFY2(shown.contains(QStringLiteral("Full pass")), qPrintable(shown.join(QLatin1Char('|'))));
    QVERIFY2(shown.contains(QStringLiteral("Play Signal Garden")), qPrintable(shown.join(QLatin1Char('|'))));

    // Continue runs the pass that stopped: this one was a full pass, so the pass it starts is full too
    // (the budgeted one it would otherwise start filters stale pipelines and has a time limit)
    run.full = true;
    m_s->settings.clearRecentWarmups();
    m_s->settings.addRecentWarmup(run);
    m_s->warmup.simulateForScreenshot(WarmupController::State::StoppedPartWay, garden, progress, {}, run);
    QApplication::processEvents();
    QAbstractButton* cont = nullptr;
    for (QAbstractButton* b : warmupPage->findChildren<QAbstractButton*>()) {
        if (b->isVisibleTo(warmupPage) && b->text() == QLatin1String("Continue"))
            cont = b;
    }
    QVERIFY(cont);
    cont->click();
    QTRY_VERIFY_WITH_TIMEOUT(m_s->warmup.isActive(), 10000);
    QVERIFY2(m_s->warmup.progress().full, "Continue kept the pass the stop belonged to");
    m_s->warmup.cancel();
    QTRY_VERIFY_WITH_TIMEOUT(!m_s->warmup.isActive(), 20000);

    // the idle card's Continue is the same promise: it is the warm-up button with another label, and
    // it too must run the pass that stopped rather than a budgeted one
    QComboBox* games = warmupPage->findChild<QComboBox*>();
    QVERIFY(games);
    const QString idleTitle = games->currentData().toString();
    QVERIFY(!idleTitle.isEmpty());
    WarmupRun idleRun = run;
    idleRun.titleId = idleTitle;
    m_s->settings.clearRecentWarmups();
    m_s->settings.addRecentWarmup(idleRun);
    m_s->warmup.simulateForScreenshot(WarmupController::State::Idle, Game{}, WarmupController::Progress{});
    QApplication::processEvents();
    QAbstractButton* idleContinue = nullptr;
    for (QAbstractButton* b : warmupPage->findChildren<QAbstractButton*>()) {
        if (b->isVisibleTo(warmupPage) && b->isEnabled() && b->text().startsWith(QLatin1String("Continue")))
            idleContinue = b;
    }
    QVERIFY2(idleContinue, "the idle card offers Continue for a pass that stopped");
    idleContinue->click();
    QTRY_VERIFY_WITH_TIMEOUT(m_s->warmup.isActive(), 10000);
    QVERIFY2(m_s->warmup.progress().full, "the idle Continue kept the pass the stop belonged to");
    m_s->warmup.cancel();
    QTRY_VERIFY_WITH_TIMEOUT(!m_s->warmup.isActive(), 20000);
}

// The quality preset row of Per-game overrides: choosing one stores it, shows what it sets, and
// hands LaunchConfig the two switches and the title budget; Default takes both back.
void TestInGame::qualityPresetsShowAndApply()
{
    const Game garden = fixtureGame(QStringLiteral("Signal Garden"));
    QTRY_VERIFY_WITH_TIMEOUT(m_s->library.game(garden.titleId()).isValid(), 10000);
    m_w->showGameDetails(garden);
    QApplication::processEvents();

    SegmentedControl* preset = nullptr;
    for (SegmentedControl* seg : m_w->gameDetailsPage()->findChildren<SegmentedControl*>()) {
        if (seg->items().contains(QStringLiteral("Quality")))
            preset = seg;
    }
    QVERIFY2(preset, "Game details has a quality preset row");
    QCOMPARE(preset->currentIndex(), 0);

    const QString id = garden.titleId();
    preset->setCurrentIndex(1); // Quality
    QTRY_COMPARE(m_s->settings.gameOverrides(id).value(QStringLiteral("qualityPreset")).toString(), QStringLiteral("quality"));
    QCOMPARE(LaunchConfig::titleValues(m_s->settings, id).value(QStringLiteral("title_memory_mb")).toLongLong(), 8192);
    const QProcessEnvironment env = LaunchConfig::runEnvironment(QProcessEnvironment(), m_s->settings, id);
    QCOMPARE(env.value(QStringLiteral("XCL_UPSCALER")), QStringLiteral("dlss"));
    QCOMPARE(env.value(QStringLiteral("XCL_DLSS_MODE")), QStringLiteral("quality"));
    QString note;
    for (const Label* label : m_w->gameDetailsPage()->findChildren<Label*>()) {
        if (label->text().contains(QStringLiteral("XCL_DLSS_MODE=quality")))
            note = label->text();
    }
    QVERIFY2(!note.isEmpty(), "the active preset names the switches it sets");

    preset->setCurrentIndex(0); // Default (blit)
    QTRY_VERIFY(!m_s->settings.gameOverrides(id).contains(QStringLiteral("qualityPreset")));
    // the budget the preset wrote goes back to the standard one
    QTRY_COMPARE(m_s->settings.gameOverrides(id).value(QStringLiteral("title_memory_mb")).toLongLong(), 0);
    QCOMPARE(LaunchConfig::titleValues(m_s->settings, id).value(QStringLiteral("title_memory_mb")).toLongLong(), 0);
    QVERIFY(!LaunchConfig::runEnvironment(QProcessEnvironment(), m_s->settings, id).contains(QStringLiteral("XCL_UPSCALER")));
    QVERIFY(!LaunchConfig::runEnvironment(QProcessEnvironment(), m_s->settings, id).contains(QStringLiteral("XCL_DLSS_MODE")));
    QString cleared;
    for (const Label* label : m_w->gameDetailsPage()->findChildren<Label*>()) {
        if (label->text().contains(QStringLiteral("XCL_UPSCALER and XCL_DLSS_MODE unset")))
            cleared = label->text();
    }
    QVERIFY2(!cleared.isEmpty(), "Default says the switches are unset");

    // Use defaults takes the preset, and the budget it wrote, with everything else
    preset->setCurrentIndex(2); // Balanced
    QTRY_COMPARE(m_s->settings.gameOverrides(id).value(QStringLiteral("qualityPreset")).toString(), QStringLiteral("balanced"));
    QAbstractButton* reset = nullptr;
    for (QAbstractButton* b : m_w->gameDetailsPage()->findChildren<QAbstractButton*>()) {
        if (b->isVisibleTo(m_w->gameDetailsPage()) && b->text() == QLatin1String("Use defaults"))
            reset = b;
    }
    QVERIFY2(reset, "Per-game overrides has a Use defaults button");
    reset->click();
    QTRY_VERIFY(!m_s->settings.gameOverrides(id).contains(QStringLiteral("qualityPreset")));
    QTRY_COMPARE(m_s->settings.gameOverrides(id).value(QStringLiteral("title_memory_mb")).toLongLong(), 0);
    QVERIFY(!LaunchConfig::runEnvironment(QProcessEnvironment(), m_s->settings, id).contains(QStringLiteral("XCL_DLSS_MODE")));
}

int main(int argc, char** argv)
{
    QApplication::setAttribute(Qt::AA_DontCreateNativeWidgetSiblings);
    QApplication app(argc, argv);
    QApplication::setApplicationName(QStringLiteral("XSEmuLauncherTest"));
    TestInGame t;
    return QTest::qExec(&t, argc, argv);
}

#include "tst_ingame.moc"
