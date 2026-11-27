#include "DevLever.h"

#ifdef XSEMU_DEV_TOOLS

#include "Services.h"
#include "core/DevFixtures.h"
#include "core/DiskSpace.h"
#include "core/EmulatorController.h"
#include "core/Paths.h"
#include "core/Runner.h"
#include "core/RunnerManager.h"
#include "pages/RunnerUi.h"
#include "core/TitleConfig.h"
#include "pages/ArtworkCard.h"
#include "pages/FirstLaunchPage.h"
#include "pages/GameDetailsPage.h"
#include "pages/ImportSettingsDialog.h"
#include "pages/ShareSettingsDialog.h"
#include "core/SettingsProfile.h"
#include "pages/RunsWithPicker.h"
#include "pages/SettingsPage.h"
#include "shell/FloatingBar.h"
#include "shell/LaunchErrors.h"
#include "shell/MainWindow.h"
#include "widgets/SearchBox.h"
#include "pages/LibraryPage.h"

#include <QAbstractButton>
#include <QApplication>
#include <QGuiApplication>
#include <QScreen>
#include <QScrollArea>
#include <QScrollBar>
#include <QDir>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLineEdit>
#include <QLinearGradient>
#include <QPainter>
#include <QPainterPath>
#include <QRadialGradient>
#include <cmath>
#include <QTemporaryDir>
#include <QTextStream>
#include <QEventLoop>
#include <QTimer>

#include <memory>

namespace {

std::unique_ptr<QTemporaryDir> g_root;

QString root(const QString& sub) { return QDir(g_root->path()).filePath(sub); }

Game findGame(const Services& s, const QString& title)
{
    for (const Game& g : s.library.games()) {
        if (g.title() == title)
            return g;
    }
    return s.library.games().value(0);
}

void seedWarmups(Services& s)
{
    const QDateTime now = QDateTime::currentDateTime();
    auto run = [&](const QString& name, qint64 done, qint64 total, double took, WarmupRun::Result result, qint64 agoS) {
        WarmupRun r;
        const Game g = findGame(s, name);
        r.titleId = g.titleId();
        r.titleName = g.title();
        r.done = done;
        r.total = total;
        r.tookS = took;
        r.result = result;
        r.finishedAt = now.addSecs(-agoS);
        return r;
    };
    WarmupRun stopped = run(QStringLiteral("Signal Garden"), 9874, 39552, 125, WarmupRun::Result::StoppedPartWay, 3 * 86400);
    stopped.stoppedForRam = true;
    WarmupRun failed = run(QStringLiteral("Signal Garden"), 31044, 41906, 160, WarmupRun::Result::Failed, 2 * 86400);
    failed.exitCode = 3;
    s.settings.addRecentWarmup(stopped);
    s.settings.addRecentWarmup(failed);
    s.settings.addRecentWarmup(run(QStringLiteral("Coastline Drive"), 1240, 3120, 22, WarmupRun::Result::Cancelled, 30 * 3600));
    s.settings.addRecentWarmup(run(QStringLiteral("Signal Garden"), 41906, 41906, 192, WarmupRun::Result::Finished, 2 * 3600));
}

WarmupController::Progress runningProgress()
{
    WarmupController::Progress p;
    p.targets = 41906;
    p.threads = 16;
    p.phase = QStringLiteral("bound");
    p.done = 27292;
    p.total = 41906;
    p.keysPerS = 243;
    p.etaS = 60;
    p.ramFreeMb = 12 * 1024;
    p.giantsCompiling = 2;
    p.inFlight = 14;
    p.elapsedS = 124;
    return p;
}

// A small shader cache and one crash dump in the temporary fixture folders, so Settings has numbers.
// Runner installs as RunnerInstall leaves them, with fake-xsemu copied in as each runner's program
// (it acts as the runner its file name names), so playing with a runner in a fixture state is real.
void seedRunner(const QString& id, const QString& version, qint64 agoS)
{
    const RunnerSpec* spec = Runners::find(id);
    const QString exe = QDir(spec->installDir()).filePath(spec->executable);
    QDir().mkpath(QFileInfo(exe).absolutePath());
    QFile::remove(exe);
    QFile::copy(QDir(QCoreApplication::applicationDirPath()).filePath(QStringLiteral("fake-xsemu")), exe);
    QFile f(QDir(spec->dataDir()).filePath(QStringLiteral("runner.json")));
    if (f.open(QIODevice::WriteOnly | QIODevice::Truncate))
        f.write(QJsonDocument(QJsonObject{{QStringLiteral("id"), id},
                                          {QStringLiteral("version"), version},
                                          {QStringLiteral("executable"), spec->executable},
                                          {QStringLiteral("installedAt"), QDateTime::currentDateTimeUtc().addSecs(-agoS).toString(Qt::ISODate)}})
                    .toJson());
}

void seedRunners()
{
    seedRunner(QStringLiteral("kelvaroq"), Runners::find(QStringLiteral("kelvaroq"))->pinnedVersion(), 26 * 3600);
    seedRunner(QStringLiteral("victory-dexnom"), QStringLiteral("2e01ab4"), 9 * 24 * 3600);
}

void seedShaderCache()
{
    QDir dxil(root(QStringLiteral("dxil")));
    dxil.mkpath(QStringLiteral("pipelines"));
    dxil.mkpath(QStringLiteral("driver/10de-2786-fixture"));
    QFile manifest(dxil.filePath(QStringLiteral("pipelines/pipelines.bin")));
    if (manifest.open(QIODevice::WriteOnly)) {
        for (int i = 0; i < 3120; ++i) {
            const QByteArray entry = QByteArray("fixture pipeline ") + QByteArray::number(i) + QByteArray(64, 'x');
            const quint32 n = quint32(entry.size());
            const char len[4] = {char(n & 0xff), char((n >> 8) & 0xff), char((n >> 16) & 0xff), char((n >> 24) & 0xff)};
            manifest.write(len, 4);
            manifest.write(entry);
        }
    }
    for (int i = 0; i < 40; ++i) {
        QFile blob(dxil.filePath(QStringLiteral("driver/10de-2786-fixture/%1.pso").arg(i)));
        if (blob.open(QIODevice::WriteOnly))
            blob.write(QByteArray(256 * 1024, '\0'));
    }
}

void seedCrashDump(const Game& game)
{
    const QString dir = QDir(root(QStringLiteral("xcl"))).filePath(game.titleId() + QStringLiteral("/logs/dumps"));
    QDir().mkpath(dir);
    QFile dump(QDir(dir).filePath(QStringLiteral("xcl-small-4242.dmp")));
    if (dump.open(QIODevice::WriteOnly))
        dump.write(QByteArray(3 * 1024 * 1024 + 123456, '\0'));
}

EmulatorController::Stats pausedStats()
{
    EmulatorController::Stats st;
    st.heldBytes = qint64(8.6 * 1024 * 1024 * 1024);
    return st;
}


// On X11 the window is grabbed from the screen, so the game's child window (another process) is in
// the picture; QWidget::grab() only sees the launcher's own widgets.
void grabAndQuit(QWidget& w, const QString& path)
{
    QPixmap shot;
    if (QGuiApplication::platformName() == QLatin1String("xcb") && w.screen()) {
        const QRect g = w.frameGeometry();
        shot = w.screen()->grabWindow(0, g.x(), g.y(), g.width(), g.height());
    } else {
        shot = w.grab();
    }
    const bool ok = shot.save(path);
    QTextStream(ok ? stdout : stderr) << (ok ? "saved " : "cannot save ") << path << Qt::endl;
    QApplication::exit(ok ? 0 : 1);
}

// The in-game states run the real thing: XSEmu (fake-xsemu on Linux) launched through
// EmulatorController with --host-hwnd and --control-pipe, driven to the state, then grabbed.
void playForReal(const DevLever::Options& o, Services& s, MainWindow& w, const Game& game)
{
    const QString& st = o.state;
    const QString scenario = st == QLatin1String("in-game-booting")          ? QStringLiteral("slow_boot")
                           : st == QLatin1String("in-game-crashed")          ? QStringLiteral("crash")
                           : st == QLatin1String("in-game-device-removed")   ? QStringLiteral("device_removed")
                           : st == QLatin1String("in-game-boot-error")       ? QStringLiteral("files_missing")
                                                                             : QStringLiteral("ok");
    qputenv("FAKE_XSEMU_SCENARIO", scenario.toUtf8());
    const QString path = o.grab;
    auto done = std::make_shared<bool>(false);
    auto grabIn = [&w, path, done](int ms) {
        if (*done)
            return;
        *done = true;
        if (!path.isEmpty())
            QTimer::singleShot(ms, &w, [&w, path] { grabAndQuit(w, path); });
    };
    auto* ctx = new QObject(&w);
    if (st == QLatin1String("in-game-booting")) {
        QObject::connect(&s.emulator, &EmulatorController::helloReceived, ctx, [grabIn] { grabIn(900); });
    } else if (st == QLatin1String("in-game-crashed") || st == QLatin1String("in-game-device-removed") || st == QLatin1String("in-game-boot-error")) {
        QObject::connect(&s.emulator, &EmulatorController::exited, ctx, [grabIn] { grabIn(600); });
    } else {
        auto once = std::make_shared<bool>(false);
        QObject::connect(&s.emulator, &EmulatorController::stateChanged, ctx, [&s, &w, st, grabIn, once](EmulatorController::State state) {
            if (state == EmulatorController::State::Paused && st == QLatin1String("in-game-paused"))
                grabIn(600);
            if (state == EmulatorController::State::Paused && st == QLatin1String("in-game-back-to-library"))
                grabIn(600);
            if (state != EmulatorController::State::Running || *once)
                return;
            *once = true;
            if (st == QLatin1String("in-game-overlay"))
                s.emulator.setOverlay(true);
            if (st == QLatin1String("in-game-paused"))
                QTimer::singleShot(1500, &w, [&s] { s.emulator.pause(); });
            else if (st == QLatin1String("in-game-back-to-library"))
                QTimer::singleShot(1500, &w, [&w] { w.leaveInGame(); });
            else if (st == QLatin1String("in-game-fullscreen-hint")) {
                QTimer::singleShot(1200, &w, [&w] { w.toggleFullScreen(); });
                grabIn(2000);
            } else if (st == QLatin1String("in-game-fullscreen")) {
                QTimer::singleShot(800, &w, [&w] { w.toggleFullScreen(); });
                grabIn(5200);
            } else {
                grabIn(2500);
            }
        });
    }
    if (!path.isEmpty()) {
        QTimer::singleShot(30000, &w, [path] {
            QTextStream(stderr) << "timed out waiting for the state; nothing saved to " << path << Qt::endl;
            QApplication::exit(1);
        });
    }
    w.play(game);
}

// ---- Library, First launch, Game details and launch errors (fixture art and records) ----

bool isLibraryState(const QString& st)
{
    return st == QLatin1String("library") || st.startsWith(QLatin1String("library-")) || st.startsWith(QLatin1String("details"))
        || st == QLatin1String("cold-cache") || st == QLatin1String("error-files-missing") || st == QLatin1String("error-not-enough-ram");
}

// Invented, abstract art for the fixture titles (no real game art): a dusk sky over the sea for
// Coastline Drive, a night beam for The Long Quiet. Painted at runtime into the temporary cover cache.
QImage paintArt(const QString& kind, QSize size)
{
    QImage img(size, QImage::Format_ARGB32_Premultiplied);
    QPainter p(&img);
    p.setRenderHint(QPainter::Antialiasing);
    const qreal w = size.width();
    const qreal h = size.height();
    if (kind == QLatin1String("coast")) {
        QLinearGradient sky(0, 0, 0, h);
        sky.setColorAt(0, QColor(0x2B, 0x2D, 0x6E));
        sky.setColorAt(0.45, QColor(0xE0, 0x6C, 0x7A));
        sky.setColorAt(0.62, QColor(0xF6, 0xB3, 0x6B));
        sky.setColorAt(0.63, QColor(0x1F, 0x4E, 0x79));
        sky.setColorAt(1, QColor(0x0E, 0x22, 0x3D));
        p.fillRect(img.rect(), sky);
        const qreal horizon = h * 0.625;
        const qreal r = qMin(w, h) * 0.16;
        QRadialGradient sun(w * 0.68, horizon - r * 0.35, r * 1.8);
        sun.setColorAt(0, QColor(0xFF, 0xE7, 0xA8));
        sun.setColorAt(0.5, QColor(0xFF, 0xC2, 0x7A, 160));
        sun.setColorAt(1, QColor(0xFF, 0xC2, 0x7A, 0));
        p.setPen(Qt::NoPen);
        p.setBrush(sun);
        p.drawEllipse(QPointF(w * 0.68, horizon - r * 0.35), r * 1.8, r * 1.8);
        p.setBrush(QColor(0xFF, 0xE9, 0xB5));
        p.drawEllipse(QPointF(w * 0.68, horizon - r * 0.35), r, r);
        p.fillRect(QRectF(0, horizon, w, h - horizon), QColor(0x1F, 0x4E, 0x79));
        for (int i = 0; i < 14; ++i) {
            const qreal y = horizon + 6 + i * (h - horizon) / 14.0;
            p.fillRect(QRectF(w * 0.68 - r * (1.2 - i * 0.05), y, r * (2.4 - i * 0.1), 3), QColor(0xFF, 0xD3, 0x92, 150 - i * 9));
        }
        QPainterPath hills;
        hills.moveTo(0, horizon);
        hills.cubicTo(w * 0.15, horizon - h * 0.09, w * 0.3, horizon - h * 0.05, w * 0.42, horizon);
        hills.closeSubpath();
        p.setBrush(QColor(0x2A, 0x1F, 0x45));
        p.drawPath(hills);
        QPainterPath road;
        road.moveTo(w * 0.05, h);
        road.cubicTo(w * 0.25, horizon + (h - horizon) * 0.4, w * 0.35, horizon + 8, w * 0.42, horizon + 2);
        road.lineTo(w * 0.44, horizon + 2);
        road.cubicTo(w * 0.4, horizon + 12, w * 0.4, horizon + (h - horizon) * 0.5, w * 0.32, h);
        road.closeSubpath();
        p.setBrush(QColor(0x14, 0x14, 0x24));
        p.drawPath(road);
    } else {
        QLinearGradient sky(0, 0, 0, h);
        sky.setColorAt(0, QColor(0x07, 0x1B, 0x26));
        sky.setColorAt(1, QColor(0x12, 0x4A, 0x55));
        p.fillRect(img.rect(), sky);
        p.setPen(Qt::NoPen);
        for (int i = 0; i < 60; ++i)
            p.fillRect(QRectF(std::fmod(i * 97.3, w), std::fmod(i * 53.7, h * 0.55), 2, 2), QColor(255, 255, 255, 60 + (i * 37) % 120));
        const QPointF lamp(w * 0.62, h * 0.48);
        QPainterPath beam;
        beam.moveTo(lamp);
        beam.lineTo(0, h * 0.3);
        beam.lineTo(0, h * 0.42);
        beam.closeSubpath();
        p.setBrush(QColor(0xFF, 0xF1, 0xC4, 70));
        p.drawPath(beam);
        p.setBrush(QColor(0xE9, 0xEE, 0xF0));
        p.drawRect(QRectF(lamp.x() - w * 0.035, lamp.y(), w * 0.07, h * 0.3));
        p.setBrush(QColor(0xC2, 0x3B, 0x3B));
        p.drawRect(QRectF(lamp.x() - w * 0.035, lamp.y() + h * 0.1, w * 0.07, h * 0.05));
        p.setBrush(QColor(0xFF, 0xF1, 0xC4));
        p.drawEllipse(lamp, w * 0.03, w * 0.03);
        p.fillRect(QRectF(0, h * 0.78, w, h * 0.22), QColor(0x05, 0x12, 0x1A));
    }
    return img;
}

// A stand-in logo: the title in heavy white capitals on a transparent background.
QImage paintLogo(const QString& title, QSize size)
{
    QImage img(size, QImage::Format_ARGB32_Premultiplied);
    img.fill(Qt::transparent);
    QPainter p(&img);
    p.setRenderHint(QPainter::Antialiasing);
    QFont f = p.font();
    f.setPixelSize(size.height() / 3);
    f.setWeight(QFont::Black);
    f.setLetterSpacing(QFont::PercentageSpacing, 104);
    p.setFont(f);
    p.setPen(QColor(0xFF, 0xF4, 0xE0));
    p.drawText(QRect(QPoint(0, 0), size), Qt::AlignCenter | Qt::TextWordWrap, title.toUpper());
    return img;
}

void seedArt(Services& s, const Game& g, const QString& kind, bool hero)
{
    if (!g.isValid())
        return;
    const QString dir = s.covers.cache().dir();
    QDir().mkpath(dir);
    CoverCache::Entry e;
    e.name = g.title();
    e.fetchedAt = QDateTime::currentDateTime();
    e.gameId = kind == QLatin1String("coast") ? 1001 : 1002;
    QString& grid = e.file(CoverCache::Kind::Grid);
    grid = CoverCache::fileName(g.titleId(), CoverCache::Kind::Grid, QStringLiteral("png"));
    paintArt(kind, QSize(600, 900)).save(QDir(dir).filePath(grid));
    if (hero) {
        QString& heroFile = e.file(CoverCache::Kind::Hero);
        heroFile = CoverCache::fileName(g.titleId(), CoverCache::Kind::Hero, QStringLiteral("png"));
        paintArt(kind, QSize(1920, 620)).save(QDir(dir).filePath(heroFile));
        QString& logo = e.file(CoverCache::Kind::Logo);
        logo = CoverCache::fileName(g.titleId(), CoverCache::Kind::Logo, QStringLiteral("png"));
        paintLogo(g.title(), QSize(800, 240)).save(QDir(dir).filePath(logo));
    }
    s.covers.cache().put(g.titleId(), e);
    emit s.covers.artChanged(g.titleId()); // drops the decoded art Services kept from before
}

// Three games as in the design: one played last with art and ready shaders, one with art and a
// warm-up that stopped part way, one without art that was never warmed up.
void seedLibrary(Services& s)
{
    const QDateTime now = QDateTime::currentDateTime();
    const Game coast = findGame(s, QStringLiteral("Coastline Drive"));
    const Game quiet = findGame(s, QStringLiteral("The Long Quiet Between Lighthouses"));
    seedArt(s, coast, QStringLiteral("coast"), true);
    seedArt(s, quiet, QStringLiteral("quiet"), false);
    CoverCache::Entry notFound;
    notFound.notFound = true;
    notFound.fetchedAt = QDateTime::currentDateTimeUtc();
    s.covers.cache().put(findGame(s, QStringLiteral("Signal Garden")).titleId(), notFound);
    s.settings.addPlayTime(coast.titleId(), 41 * 3600 + 20 * 60, now.addSecs(-3 * 3600));
    s.settings.addPlayTime(quiet.titleId(), 6 * 3600, now.addDays(-2));
    WarmupRun cancelled;
    cancelled.titleId = quiet.titleId();
    cancelled.titleName = quiet.title();
    cancelled.done = 1240;
    cancelled.total = 3120;
    cancelled.tookS = 10;
    cancelled.result = WarmupRun::Result::Cancelled;
    cancelled.finishedAt = now.addDays(-2);
    s.settings.addRecentWarmup(cancelled);
    WarmupRun ready;
    ready.titleId = coast.titleId();
    ready.titleName = coast.title();
    ready.done = 41906;
    ready.total = 41906;
    ready.tookS = 14 * 60;
    ready.result = WarmupRun::Result::Finished;
    ready.finishedAt = now.addSecs(-4 * 3600);
    s.settings.addRecentWarmup(ready);
    s.settings.setLastRun(coast.titleId(), {
        {QStringLiteral("result"), QStringLiteral("Stopped by you")},
        {QStringLiteral("at"), now.addSecs(-3 * 3600).toUTC().toString(Qt::ISODate)},
        {QStringLiteral("seconds"), 2 * 3600 + 14 * 60},
        {QStringLiteral("fpsAverage"), 58.2},
        {QStringLiteral("stalls"), 1},
        {QStringLiteral("heldPeakBytes"), qint64(8.6 * 1024 * 1024 * 1024)},
        {QStringLiteral("freeMinMb"), 9830},
    });
    TitleConfig(Paths::xclToml(coast.titleId())).set(QStringLiteral("device_type"), QStringLiteral("lockhart"));
    s.settings.setGameOverride(coast.titleId(), QStringLiteral("overlay"), true);
}

}

QStringList DevLever::states()
{
    return {
        QStringLiteral("library"), QStringLiteral("library-empty"), QStringLiteral("library-search"),
        QStringLiteral("library-search-none"), QStringLiteral("library-paused"), QStringLiteral("library-stop-first"),
        QStringLiteral("cold-cache"), QStringLiteral("details"), QStringLiteral("details-no-art"),
        QStringLiteral("warmup-idle"), QStringLiteral("warmup-running"), QStringLiteral("warmup-paused-ram"),
        QStringLiteral("warmup-failed"), QStringLiteral("warmup-finished"), QStringLiteral("warmup-blocked"),
        QStringLiteral("warmup-cancelled"), QStringLiteral("warmup-stopped"), QStringLiteral("warmup-off"), QStringLiteral("details-warmup-off"),
        QStringLiteral("settings"), QStringLiteral("settings-network"), QStringLiteral("settings-shaders"), QStringLiteral("settings-run-profile"),
        QStringLiteral("settings-shaders-clear"), QStringLiteral("settings-overlay"), QStringLiteral("settings-folders"),
        QStringLiteral("settings-input"), QStringLiteral("settings-graphics"), QStringLiteral("settings-logs"),
        QStringLiteral("settings-runners-folder"),
        QStringLiteral("settings-about"),
        QStringLiteral("first-launch"), QStringLiteral("first-launch-network"), QStringLiteral("first-launch-folders"),
        QStringLiteral("first-launch-folders-found"),
        QStringLiteral("error-files-missing"), QStringLiteral("error-not-enough-ram"), QStringLiteral("error-crash"),
        QStringLiteral("error-device-removed"), QStringLiteral("error-emulator-missing"),
        QStringLiteral("in-game-booting"), QStringLiteral("in-game-playing"), QStringLiteral("in-game-overlay"),
        QStringLiteral("in-game-paused"), QStringLiteral("in-game-fitted"), QStringLiteral("in-game-fullscreen-hint"),
        QStringLiteral("in-game-fullscreen"), QStringLiteral("in-game-crashed"), QStringLiteral("in-game-device-removed"),
        QStringLiteral("in-game-boot-error"), QStringLiteral("in-game-back-to-library"),
        QStringLiteral("runners"), QStringLiteral("runners-install-confirm"), QStringLiteral("runners-building"),
        QStringLiteral("runners-install-error"), QStringLiteral("runners-change-location"), QStringLiteral("details-runs-with-kelvaroq"),
        QStringLiteral("details-runs-with-picker"), QStringLiteral("details-run-profile"), QStringLiteral("details-artwork"), QStringLiteral("details-artwork-match"),
        QStringLiteral("warmup-runner"), QStringLiteral("library-runner-own-window"), QStringLiteral("library-runner-floating-bar"),
        QStringLiteral("share-settings"), QStringLiteral("import-preview"), QStringLiteral("import-error"),
    };
}

bool DevLever::parse(const QStringList& args, Options* out, QString* error)
{
    bool any = false;
    for (int i = 1; i < args.size(); ++i) {
        const QString& a = args[i];
        auto value = [&]() -> QString {
            if (i + 1 >= args.size()) {
                *error = QStringLiteral("%1 needs a value").arg(a);
                return {};
            }
            return args[++i];
        };
        if (a == QLatin1String("--list-states")) {
            out->listStates = true;
            any = true;
        } else if (a == QLatin1String("--theme")) {
            out->theme = value();
            if (out->theme != QLatin1String("light") && out->theme != QLatin1String("dark"))
                *error = QStringLiteral("--theme takes light or dark");
            any = true;
        } else if (a == QLatin1String("--size")) {
            const QStringList wh = value().split(QLatin1Char('x'));
            out->size = wh.size() == 2 ? QSize(wh[0].toInt(), wh[1].toInt()) : QSize();
            if (out->size.width() < 640 || out->size.height() < 400)
                *error = QStringLiteral("--size takes WIDTHxHEIGHT, at least 640x400");
            any = true;
        } else if (a == QLatin1String("--state")) {
            out->state = value();
            if (!states().contains(out->state))
                *error = QStringLiteral("unknown state %1 (see --list-states)").arg(out->state);
            any = true;
        } else if (a == QLatin1String("--grab")) {
            out->grab = value();
            any = true;
        }
        if (!error->isEmpty())
            return false;
    }
    if (any && out->state.isEmpty() && !out->listStates)
        out->state = QStringLiteral("library");
    return any;
}

bool DevLever::prepare(QString* error)
{
    g_root = std::make_unique<QTemporaryDir>();
    if (!g_root->isValid()) {
        *error = QStringLiteral("cannot create a temporary folder");
        return false;
    }
    const QString fixtures = QStringLiteral(XSEMU_FIXTURES_DIR);
    if (!DevFixtures::materialize(QDir(fixtures).filePath(QStringLiteral("games")), root(QStringLiteral("games")))
        || !DevFixtures::materialize(QDir(fixtures).filePath(QStringLiteral("xcl-data")), root(QStringLiteral("xcl")))) {
        *error = QStringLiteral("cannot copy the fixtures from %1").arg(fixtures);
        return false;
    }
    seedShaderCache(); // the driver blobs the fixtures' finished warm-ups left
    qputenv("XSEMU_LAUNCHER_DATA", root(QStringLiteral("launcher")).toUtf8());
    qputenv("XCL_DATA_ROOT", root(QStringLiteral("xcl")).toUtf8());
    qputenv("XCL_DXIL_CACHE", root(QStringLiteral("dxil")).toUtf8());
    qputenv("XSEMU_RUNNERS_DIR", root(QStringLiteral("runners")).toUtf8());
    if (qEnvironmentVariableIsEmpty("XSEMU_SGDB_PROXY"))
        qputenv("XSEMU_SGDB_PROXY", "http://127.0.0.1:9/api/v2"); // fixtures never reach SteamGridDB unless told to
    if (qEnvironmentVariableIsEmpty("XSEMU_GPU"))
        qputenv("XSEMU_GPU", "10de:NVIDIA GPU (sample)");
    QFile feed(root(QStringLiteral("runner-feed.json")));
    if (feed.open(QIODevice::WriteOnly))
        feed.write(R"({"victory-7900xtx": {"keep": true}, "victory-dexnom": {"keep": true}, "kelvaroq": {"keep": true}, "coolfox": {"keep": true}})");
    feed.close();
    qputenv("XSEMU_RUNNER_FEED", feed.fileName().toUtf8());
    seedRunners();
    return true;
}

void DevLever::apply(const Options& o, Services& s, MainWindow& w)
{
    s.settings.setTheme(o.theme == QLatin1String("dark") ? AppSettings::ThemeMode::Dark : AppSettings::ThemeMode::Light);
    s.settings.setDefaultDeviceType(QStringLiteral("lockhart"));
    s.settings.setOffline(true);
    // the design states show XSEmu's warm-up; the -warmup-off states show it turned off, as the build ships
    s.settings.setXsemuWarmup(!o.state.endsWith(QLatin1String("warmup-off")));
    const bool firstLaunch = o.state.startsWith(QLatin1String("first-launch"));
    s.settings.setArtLookups(false); // until the fixtures' art is seeded, so no lookup races it
    const bool libraryState = isLibraryState(o.state) && o.state != QLatin1String("library-empty");
    const QString games = root(QStringLiteral("games"));
    if (libraryState) {
        s.settings.setGameFolders({QDir(games).filePath(QStringLiteral("Coastline Drive")), QDir(games).filePath(QStringLiteral("Archive")),
                                   QDir(games).filePath(QStringLiteral("SignalGarden"))});
        s.library.rescanNow();
        seedLibrary(s);
    } else {
        s.settings.setGameFolders(o.state == QLatin1String("library-empty") || o.state == QLatin1String("first-launch")
                                      || o.state == QLatin1String("first-launch-network") || o.state == QLatin1String("first-launch-folders")
                                      ? QStringList()
                                      : QStringList{games});
        s.library.rescanNow();
    }
    if (o.state == QLatin1String("first-launch-folders-found")) {
        seedArt(s, findGame(s, QStringLiteral("Coastline Drive")), QStringLiteral("coast"), true);
        seedArt(s, findGame(s, QStringLiteral("The Long Quiet Between Lighthouses")), QStringLiteral("quiet"), false);
    }
    s.settings.setArtLookups(true);
    s.settings.setFirstLaunchDone(!firstLaunch);
    if (!firstLaunch && !isLibraryState(o.state))
        seedWarmups(s);

    const Game signal = findGame(s, QStringLiteral("Signal Garden"));
    const Game coast = findGame(s, QStringLiteral("Coastline Drive"));
    const QString& st = o.state;

    const Game quiet = findGame(s, QStringLiteral("The Long Quiet Between Lighthouses"));
    if (st == QLatin1String("library-search"))
        w.libraryPage()->searchBox()->setText(QStringLiteral("coast"));
    else if (st == QLatin1String("library-search-none"))
        w.libraryPage()->searchBox()->setText(QStringLiteral("racing"));
    else if (st == QLatin1String("library-runner-own-window") || st == QLatin1String("library-runner-floating-bar")) {
        // a third-party runner in its own window (it is never embedded): the library banner and the floating bar
        s.settings.setGameOverride(coast.titleId(), QStringLiteral("runner"), QStringLiteral("victory-dexnom"));
        s.emulator.simulateForScreenshot(EmulatorController::State::Running, coast, {});
    } else if (st == QLatin1String("library-paused") || st == QLatin1String("library-stop-first"))
        s.emulator.simulateForScreenshot(EmulatorController::State::Paused, coast, pausedStats());
    else if (st == QLatin1String("details") || st == QLatin1String("details-warmup-off"))
        w.showGameDetails(coast);
    else if (st == QLatin1String("details-no-art"))
        w.showGameDetails(signal);
    else if (st == QLatin1String("details-runs-with-kelvaroq")) {
        s.settings.setGameOverride(coast.titleId(), QStringLiteral("runner"), QStringLiteral("kelvaroq"));
        w.showGameDetails(coast);
    } else if (st == QLatin1String("details-runs-with-picker"))
        w.showGameDetails(coast);
    else if (st.startsWith(QLatin1String("details-artwork"))) {
        CoverCache::Entry e = s.covers.cache().entry(coast.titleId());
        e.pinned = true;
        e.authors = {QStringLiteral("Mira Okafor"), QStringLiteral("Mira Okafor"), QStringLiteral("tidewright"), {}};
        s.covers.cache().put(coast.titleId(), e);
        const QString icon = root(QStringLiteral("coast-icon.png"));
        paintArt(QStringLiteral("coast"), QSize(256, 256)).save(icon);
        s.covers.setCustomImage(coast.titleId(), CoverCache::Kind::Icon, icon);
        w.showGameDetails(coast);
    } else if (st == QLatin1String("details-run-profile")) {
        s.settings.setGameOverride(coast.titleId(), QStringLiteral("runProfile"),
                                   QStringList{QStringLiteral("XCL_SHADER_BACKEND=dxil"), QStringLiteral("XCL_LOOP_GUARD=4096"),
                                               QStringLiteral("XCL_DISPLAY_MODE=2560x1440"), QStringLiteral("XCL_NETWORK=1")});
        w.showGameDetails(coast);
    } else if (st == QLatin1String("share-settings") || st.startsWith(QLatin1String("import-"))) {
        if (st == QLatin1String("share-settings")) {
            s.settings.setGameOverride(coast.titleId(), QStringLiteral("device_type"), QStringLiteral("anaconda"));
            s.settings.setGameOverride(coast.titleId(), QStringLiteral("display_mode"), QStringLiteral("1920x1080"));
            s.settings.setGameOverride(coast.titleId(), QStringLiteral("raytracing"), QStringLiteral("tier1_1"));
            s.settings.setGameOverride(coast.titleId(), QStringLiteral("overlay"), true);
            s.settings.setGameOverride(coast.titleId(), QStringLiteral("runner"), QStringLiteral("kelvaroq"));
        }
        w.showGameDetails(coast);
    }
    else if (st.startsWith(QLatin1String("warmup"))) {
        w.showPage(Page::Warmup);
        if (st == QLatin1String("warmup-runner"))
            s.settings.setGameOverride(signal.titleId(), QStringLiteral("runner"), QStringLiteral("kelvaroq"));
        if (st == QLatin1String("warmup-running"))
            s.warmup.simulateForScreenshot(WarmupController::State::Running, signal, runningProgress());
        else if (st == QLatin1String("warmup-paused-ram")) {
            WarmupController::Progress p = runningProgress();
            p.pausedForRam = true;
            p.ramFreeMb = 980;
            p.etaS.reset();
            s.warmup.simulateForScreenshot(WarmupController::State::PausedForRam, signal, p);
        } else if (st == QLatin1String("warmup-failed")) {
            WarmupController::Progress p = runningProgress();
            p.done = 31044;
            WarmupRun run;
            run.titleId = signal.titleId();
            run.titleName = signal.title();
            run.done = 31044;
            run.total = 41906;
            run.tookS = 160;
            run.result = WarmupRun::Result::Failed;
            run.exitCode = 3;
            run.left = -1; // a crash reports no count of what it left
            run.finishedAt = QDateTime::currentDateTime();
            s.settings.clearRecentWarmups();
            s.settings.addRecentWarmup(run); // the record Continue is offered from
            QDir().mkpath(QFileInfo(WarmupController::logPath()).absolutePath());
            QFile log(WarmupController::logPath());
            if (log.open(QIODevice::WriteOnly | QIODevice::Truncate))
                log.write("warm-up: the driver compile of DXIL 3f2a91c07d1e4b55 crashed (0xC0000005)\n");
            s.warmup.simulateForScreenshot(WarmupController::State::Failed, signal, p,
                                           QStringLiteral("warm-up: the driver compile of DXIL 3f2a91c07d1e4b55 crashed (0xC0000005)"), run);
        } else if (st == QLatin1String("warmup-finished")) {
            WarmupController::Progress p = runningProgress();
            p.phase = QStringLiteral("rest");
            p.done = p.total;
            p.ramFreeMb = 11674;
            WarmupRun run;
            run.titleId = signal.titleId();
            run.titleName = signal.title();
            run.done = 41906;
            run.total = 41906;
            run.tookS = 192;
            run.finishedAt = QDateTime::currentDateTime();
            s.warmup.simulateForScreenshot(WarmupController::State::Finished, signal, p, {}, run);
        } else if (st == QLatin1String("warmup-cancelled") || st == QLatin1String("warmup-stopped")) {
            const bool stopped = st == QLatin1String("warmup-stopped");
            WarmupController::Progress p = runningProgress();
            WarmupRun run;
            run.titleId = signal.titleId();
            run.titleName = signal.title();
            run.done = 27292;
            run.total = 41906;
            run.left = 41906 - 27292;
            run.tookS = 124;
            run.result = stopped ? WarmupRun::Result::StoppedPartWay : WarmupRun::Result::Cancelled;
            run.exitCode = stopped ? 0 : 4;
            run.finishedAt = QDateTime::currentDateTime();
            s.settings.clearRecentWarmups();
            s.settings.addRecentWarmup(run); // the record Continue is offered from
            s.warmup.simulateForScreenshot(stopped ? WarmupController::State::StoppedPartWay : WarmupController::State::Cancelled, signal, p, {}, run);
        } else if (st == QLatin1String("warmup-blocked"))
            s.emulator.simulateForScreenshot(EmulatorController::State::Paused, signal, pausedStats());
    } else if (st.startsWith(QLatin1String("settings"))) {
        w.showPage(Page::Settings);
        if (st.startsWith(QLatin1String("settings-shaders"))) {
            w.settingsPage()->showSection(SettingsPage::Section::Shaders);
        } else if (st == QLatin1String("settings-network"))
            w.settingsPage()->showSection(SettingsPage::Section::Network);
        else if (st == QLatin1String("settings-run-profile"))
            w.settingsPage()->showSection(SettingsPage::Section::RunProfile);
        else if (st == QLatin1String("settings-overlay"))
            w.settingsPage()->showSection(SettingsPage::Section::Overlay);
        else if (st == QLatin1String("settings-folders"))
            w.settingsPage()->showSection(SettingsPage::Section::GameFolders);
        else if (st == QLatin1String("settings-runners-folder")) {
            // the Runners folder as a user sets it: from settings, not from the lever's XSEMU_RUNNERS_DIR
            qunsetenv("XSEMU_RUNNERS_DIR");
            s.settings.setRunnersFolder(root(QStringLiteral("runners")));
            w.settingsPage()->showSection(SettingsPage::Section::GameFolders);
        }
        else if (st == QLatin1String("settings-input"))
            w.settingsPage()->showSection(SettingsPage::Section::Input);
        else if (st == QLatin1String("settings-graphics"))
            w.settingsPage()->showSection(SettingsPage::Section::Graphics);
        else if (st == QLatin1String("settings-logs")) {
            seedCrashDump(signal);
            w.settingsPage()->showSection(SettingsPage::Section::Logs);
        } else if (st == QLatin1String("settings-about"))
            w.settingsPage()->showSection(SettingsPage::Section::About);
    } else if (st.startsWith(QLatin1String("runners"))) {
        w.showPage(Page::Runners);
        const QString xtx = QStringLiteral("victory-7900xtx");
        if (st == QLatin1String("runners-building")) {
            RunnerManager::Status b;
            b.state = RunnerManager::State::Installing;
            b.pinnedVersion = Runners::find(xtx)->pinnedVersion();
            b.progress.step = QStringLiteral("Building");
            b.progress.stepIndex = 3;
            b.progress.stepCount = 4;
            b.progress.phase = 1;
            b.progress.done = 212;
            b.progress.total = 389;
            b.progress.detail = QStringLiteral("Compiling naga v24.0.0");
            b.progress.startedAt = QDateTime::currentDateTime().addSecs(-252);
            s.runners.simulateForScreenshot(xtx, b);
        } else if (st == QLatin1String("runners-install-error")) {
            RunnerManager::Status f;
            f.state = RunnerManager::State::Failed;
            f.pinnedVersion = Runners::find(xtx)->pinnedVersion();
            f.error = QStringLiteral("The build stopped: cargo exited with code 101 (linker link.exe not found). Nothing was installed.");
            s.runners.simulateForScreenshot(xtx, f);
        }
    } else if (st == QLatin1String("first-launch-network"))
        w.firstLaunchPage()->setStep(1);
    else if (st == QLatin1String("first-launch-folders") || st == QLatin1String("first-launch-folders-found"))
        w.firstLaunchPage()->setStep(2);

    w.resize(st == QLatin1String("in-game-fitted") && o.size == QSize(1280, 800) ? QSize(1280, 760) : o.size);
    w.show();
    // a state is shown as a warm launcher shows it: after the first background scans
    QEventLoop loop;
    const auto settled = [&s, &loop] {
        if (s.shaderCache.hasSnapshot() && s.hardware.ready())
            loop.quit();
    };
    QObject::connect(&s.shaderCache, &ShaderCacheMonitor::changed, &loop, settled);
    QObject::connect(&s.hardware, &HardwareMonitor::changed, &loop, settled);
    if (!s.shaderCache.hasSnapshot() || !s.hardware.ready())
        loop.exec();
    if (st.startsWith(QLatin1String("in-game"))) {
        playForReal(o, s, w, signal);
        return;
    }

    if (st == QLatin1String("details-runs-with-picker"))
        QTimer::singleShot(0, &w, [&w] { w.gameDetailsPage()->runsWith()->openPicker(); });
    if (st == QLatin1String("details-run-profile"))
        QTimer::singleShot(0, &w, [&w] {
            QScrollBar* bar = w.gameDetailsPage()->findChild<QScrollArea*>()->verticalScrollBar();
            bar->setValue(bar->maximum());
        });
    if (st == QLatin1String("details-artwork"))
        QTimer::singleShot(0, &w, [&w] {
            w.gameDetailsPage()->findChild<QScrollArea*>()->ensureWidgetVisible(w.gameDetailsPage()->findChild<ArtworkCard*>(), 0, 24);
        });
    if (st == QLatin1String("details-artwork-match"))
        QTimer::singleShot(0, &w, [&w] { w.gameDetailsPage()->findChild<ArtworkCard*>()->chooseMatch(); });
    if (st == QLatin1String("runners-install-confirm"))
        QTimer::singleShot(0, &w, [&w, &s] { RunnerUi::confirmInstall(&w, s, QStringLiteral("victory-7900xtx")); });
    if (st == QLatin1String("runners-change-location")) {
        // an installed runner moving to another drive with little room left on it (the free space is invented)
        DiskSpace::setFreeBytesFunction([](const QString&) { return 14LL * 1024 * 1024 * 1024; });
        QTimer::singleShot(0, &w, [&w, &s] { RunnerUi::changeLocation(&w, s, QStringLiteral("victory-dexnom"), root(QStringLiteral("Games/Runners"))); });
    }
    if (st == QLatin1String("settings-runners-folder")) {
        QWidget* page = w.settingsPage()->sectionPage(SettingsPage::Section::GameFolders);
        if (auto* scroll = qobject_cast<QScrollArea*>(page->parentWidget() ? page->parentWidget()->parentWidget() : nullptr)) {
            if (QWidget* row = page->findChild<QWidget*>(QStringLiteral("runnersFolder")))
                QTimer::singleShot(0, scroll, [scroll, row] { scroll->ensureWidgetVisible(row, 0, 160); });
        }
    }
    if (st == QLatin1String("settings-shaders-clear")) {
        if (auto* clear = w.settingsPage()->sectionPage(SettingsPage::Section::Shaders)->findChild<QAbstractButton*>(QStringLiteral("clearShaderCache")))
            QTimer::singleShot(300, clear, &QAbstractButton::click);
    }
    if (st == QLatin1String("share-settings")) {
        QTimer::singleShot(0, &w, [&w, &s, id = coast.titleId()] {
            SettingsShare::openForGame(&w, s, id);
            if (auto* note = w.findChild<QLineEdit*>(QStringLiteral("shareNote"))) {
                note->setText(QStringLiteral("Locked 60 in the city on an RTX 5080. Series X needs 32 GB of RAM."));
                note->setCursorPosition(0);
            }
        });
    } else if (st.startsWith(QLatin1String("import-"))) {
        // a profile made on another PC: three changes, a runner this PC lacks, a feature this build
        // doesn't have, a value out of range and a key this launcher doesn't know
        const QByteArray shared = QJsonDocument(QJsonObject{
            {QStringLiteral("schema"), QStringLiteral("xsemu-settings")},
            {QStringLiteral("version"), 1},
            {QStringLiteral("name"), QStringLiteral("Coastline Drive · 1440p 120 Hz")},
            {QStringLiteral("note"), QStringLiteral("Smooth in the city at 120 Hz. Series X needs 32 GB of RAM.")},
            {QStringLiteral("madeWith"), QJsonObject{{QStringLiteral("launcher"), QStringLiteral("0.1.0")}, {QStringLiteral("gpuVendor"), QStringLiteral("AMD")},
                                                     {QStringLiteral("gpu"), QStringLiteral("AMD Radeon RX 7900 XTX")}}},
            {QStringLiteral("game"), QJsonObject{{QStringLiteral("titleId"), coast.titleId()}, {QStringLiteral("title"), coast.title()}}},
            {QStringLiteral("settings"), QJsonObject{{QStringLiteral("device_type"), QStringLiteral("anaconda")},
                                                     {QStringLiteral("display_mode"), QStringLiteral("2560x1440@120")},
                                                     {QStringLiteral("raytracing"), QStringLiteral("tier1_1")},
                                                     {QStringLiteral("mouse_sensitivity"), 9},
                                                     {QStringLiteral("startIn"), QStringLiteral("fullscreen")},
                                                     {QStringLiteral("fsr_sharpness"), 0.4}}},
            {QStringLiteral("runner"), QJsonObject{{QStringLiteral("id"), QStringLiteral("victory-7900xtx")},
                                                   {QStringLiteral("version"), Runners::find(QStringLiteral("victory-7900xtx"))->pinnedVersion()}}},
        }).toJson(QJsonDocument::Compact);
        const QString code = QLatin1String(SettingsProfile::kCodePrefix)
                             + QString::fromLatin1(qCompress(shared).toBase64(QByteArray::Base64UrlEncoding | QByteArray::OmitTrailingEquals));
        const QByteArray input = st == QLatin1String("import-error") ? code.left(code.size() - 24).toUtf8() : code.toUtf8();
        QTimer::singleShot(0, &w, [&w, &s, input, id = coast.titleId()] { SettingsImport::fromText(&w, s, input, id); });
    }
    if (st == QLatin1String("library-stop-first"))
        w.play(quiet); // the real launch check: Busy while Coastline Drive is paused
    else if (st == QLatin1String("cold-cache"))
        w.libraryPage()->requestPlay(quiet); // the real prompt: its last warm-up stopped part way

    if (st.startsWith(QLatin1String("error"))) {
        if (st == QLatin1String("error-files-missing")) {
            // the game as the library knew it, on a drive that is no longer there; the check is real
            Game gone = quiet;
            gone.folder = root(QStringLiteral("unplugged/Archive/Quiet"));
            gone.configPath = QDir(gone.folder).filePath(QStringLiteral("MicrosoftGame.config"));
            LaunchErrors::showLaunchProblem(&w, s, gone, s.emulator.checkLaunch(gone), [] {});
        } else if (st == QLatin1String("error-emulator-missing")) {
            // the emulator as a user sets it, at a path that no longer holds it; the check is real
            qunsetenv("XSEMU_EMULATOR");
            s.settings.setEmulatorPathOverride(root(QStringLiteral("XSEmu/XSEmu.exe")));
            LaunchErrors::showLaunchProblem(&w, s, coast, s.emulator.checkLaunch(coast), [] {});
        } else if (st == QLatin1String("error-not-enough-ram")) {
            // fixture numbers: the need is the real one (1 GB floor + the launch need setting), free RAM is invented
            EmulatorController::LaunchCheck check;
            check.problem = EmulatorController::LaunchCheck::Problem::NotEnoughRam;
            check.needMb = s.emulator.launchNeedMb();
            check.freeMb = 812;
            LaunchErrors::showLaunchProblem(&w, s, coast, check, [] {});
        } else {
            EmulatorController::ExitInfo info;
            info.game = signal;
            if (st == QLatin1String("error-crash")) {
                info.kind = EmulatorController::ExitInfo::Kind::Crash;
                info.exitCode = int(0xC0000005u);
                info.message = QStringLiteral("XSEmu Core stopped with an access violation while the game was loading.");
            } else {
                info.kind = EmulatorController::ExitInfo::Kind::DeviceRemoved;
                info.exitCode = int(0x887A0006u);
                info.message = QStringLiteral("DXGI_ERROR_DEVICE_HUNG (0x887A0006)");
            }
            LaunchErrors::showExit(&w, info, [] {});
        }
    }

    if (st == QLatin1String("library-runner-floating-bar")) {
        FloatingBar* bar = w.floatingBar();
        bar->showForScreenshot();
        if (!o.grab.isEmpty())
            QTimer::singleShot(500, bar, [bar, path = o.grab] { grabAndQuit(*bar, path); });
        return;
    }
    if (st == QLatin1String("warmup-runner")) {
        // the real pre-cache: fake-xsemu stands in for Kelvaroq's precompile (check, then build)
        QString error;
        if (!s.warmup.start(false, signal, &error))
            QTextStream(stderr) << "cannot start the pre-cache: " << error << Qt::endl;
        if (!o.grab.isEmpty())
            QTimer::singleShot(3000, &w, [&w, path = o.grab] { grabAndQuit(w, path); });
        return;
    }
    if (!o.grab.isEmpty())
        QTimer::singleShot(500, &w, [&w, path = o.grab] { grabAndQuit(w, path); });
}

#endif
