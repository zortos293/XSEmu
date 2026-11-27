#include "MainWindow.h"

#include "FloatingBar.h"
#include "InGameView.h"
#include "LaunchErrors.h"
#include "Sidebar.h"
#include "TitleBar.h"
#include "app/Services.h"
#include "core/NativeWindow.h"
#include "pages/FirstLaunchPage.h"
#include "pages/GameDetailsPage.h"
#include "pages/ImportSettingsDialog.h"
#include "pages/LibraryPage.h"
#include "pages/SettingsPage.h"
#include "pages/RunnersPage.h"
#include "pages/WarmupPage.h"
#include "theme/Theme.h"
#include "widgets/Icon.h"

#include <QApplication>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QPainter>
#include <QStackedWidget>
#include <QVBoxLayout>
#include <QWindow>

#ifdef Q_OS_WIN
#include <windows.h>
#include <windowsx.h>
#endif

namespace {
constexpr int kResizeBorder = 6;
enum Mode { SetupMode, ShellMode, InGameMode };
}

MainWindow::MainWindow(Services& services, QWidget* parent)
    : QWidget(parent, Qt::Window | Qt::FramelessWindowHint)
    , m_s(services)
    , m_titleBar(new TitleBar)
    , m_modes(new QStackedWidget)
    , m_shell(new QWidget)
    , m_sidebar(new Sidebar(services))
    , m_pages(new QStackedWidget)
    , m_library(new LibraryPage(services))
    , m_warmup(new WarmupPage(services))
    , m_runners(new RunnersPage(services))
    , m_floating(new FloatingBar(services))
    , m_settings(new SettingsPage(services))
    , m_details(new GameDetailsPage(services))
    , m_firstLaunch(new FirstLaunchPage(services))
    , m_inGame(new InGameView(services))
{
    setWindowTitle(QStringLiteral("XSEmu Launcher"));
    setWindowIcon(Icon::appIcon());
    setMinimumSize(960, 600);

    auto* v = new QVBoxLayout(this);
    v->setContentsMargins(0, 0, 0, 0);
    v->setSpacing(0);
    v->addWidget(m_titleBar);
    v->addWidget(m_modes, 1);

    auto* h = new QHBoxLayout(m_shell);
    h->setContentsMargins(0, 0, 0, 0);
    h->setSpacing(0);
    h->addWidget(m_sidebar);
    h->addWidget(m_pages, 1);
    m_pages->addWidget(m_library);
    m_pages->addWidget(m_warmup);
    m_pages->addWidget(m_runners);
    m_pages->addWidget(m_settings);
    m_pages->addWidget(m_details);

    m_modes->insertWidget(SetupMode, m_firstLaunch);
    m_modes->insertWidget(ShellMode, m_shell);
    m_modes->insertWidget(InGameMode, m_inGame);

    connect(m_sidebar, &Sidebar::navigate, this, &MainWindow::showPage);
    connect(m_sidebar, &Sidebar::nowPlayingClicked, this, &MainWindow::enterInGame);
    connect(m_library, &LibraryPage::playRequested, this, &MainWindow::play);
    connect(m_library, &LibraryPage::detailsRequested, this, &MainWindow::showGameDetails);
    connect(m_library, &LibraryPage::addGamesRequested, this, &MainWindow::addGameFolder);
    connect(m_library, &LibraryPage::nowPlayingRequested, this, &MainWindow::enterInGame);
    connect(m_settings, &SettingsPage::addGamesRequested, this, &MainWindow::addGameFolder);
    connect(m_details, &GameDetailsPage::back, this, [this] { showPage(Page::Library); });
    connect(m_details, &GameDetailsPage::playRequested, this, &MainWindow::play);
    const auto warmUp = [this](const Game& game) {
        m_s.warmup.start(false, game);
        showPage(Page::Warmup);
    };
    connect(m_library, &LibraryPage::warmupRequested, this, warmUp);
    connect(m_details, &GameDetailsPage::warmupRequested, this, warmUp);
    connect(m_details, &GameDetailsPage::manageRunners, this, [this] { showPage(Page::Runners); });
    // a third-party runner's own window: the floating bar over it brings the launcher back
    connect(this, &QObject::destroyed, m_floating, &QObject::deleteLater);
    connect(m_floating, &FloatingBar::showLauncher, this, [this] {
        if (isMinimized())
            showNormal();
        raise();
        activateWindow();
        showPage(Page::Library);
    });
    connect(m_firstLaunch, &FirstLaunchPage::addGamesRequested, this, &MainWindow::addGameFolder);
    connect(m_firstLaunch, &FirstLaunchPage::finished, this, [this] { showPage(Page::Library); });
    connect(m_inGame, &InGameView::backToLibrary, this, &MainWindow::leaveInGame);
    connect(m_inGame, &InGameView::relaunchRequested, this, &MainWindow::play);
    connect(m_inGame, &InGameView::fullScreenToggled, this, &MainWindow::toggleFullScreen);

    connect(&m_s.settings, &AppSettings::changed, this, [this](const QString& key) {
        if (key == QLatin1String(AppSettings::Key::FirstLaunchDone))
            syncMode();
    });
    connect(&m_s.emulator, &EmulatorController::exited, this, [this](const EmulatorController::ExitInfo& info) {
        using K = EmulatorController::ExitInfo::Kind;
        if (info.kind == K::Stopped || info.kind == K::TitleExit) {
            if (m_inGameMode)
                leaveInGame();
            return;
        }
        if (m_inGameMode) {
            if (isFullScreen())
                toggleFullScreen();
            m_inGame->showExit(info);
            return;
        }
        const Game game = info.game;
        LaunchErrors::showExit(this, info, [this, game] { play(game); });
    });
    connect(&Theme::instance(), &Theme::changed, this, [this] {
        setWindowIcon(Icon::appIcon());
        update();
    });

    SettingsImport::install(
        this, m_s, [this] { return m_pages->currentWidget() == m_details ? m_details->game().titleId() : QString(); },
        [this] { return !m_inGameMode && m_s.settings.firstLaunchDone(); });

#ifndef Q_OS_WIN
    qApp->installEventFilter(this);
#endif
    syncMode();
}

void MainWindow::syncMode()
{
    const bool setup = !m_s.settings.firstLaunchDone();
    const bool inGame = !setup && m_inGameMode;
    m_modes->setCurrentIndex(setup ? SetupMode : inGame ? InGameMode : ShellMode);
    m_titleBar->setVisible(!inGame);
    m_titleBar->setCaption(setup ? QStringLiteral("XSEmu Launcher \u00B7 Setup") : QString());
}

void MainWindow::showPage(Page page)
{
    switch (page) {
    case Page::Library: m_pages->setCurrentWidget(m_library); break;
    case Page::Warmup: m_pages->setCurrentWidget(m_warmup); break;
    case Page::Runners: m_pages->setCurrentWidget(m_runners); break;
    case Page::Settings: m_pages->setCurrentWidget(m_settings); break;
    case Page::GameDetails: m_pages->setCurrentWidget(m_details); break;
    }
    m_sidebar->setCurrent(page);
    if (m_inGameMode)
        leaveInGame();
    syncMode();
    m_s.shaderCache.refreshIfStale();
}

void MainWindow::showGameDetails(const Game& game)
{
    m_details->setGame(game);
    showPage(Page::GameDetails);
}

void MainWindow::play(const Game& game)
{
    if (m_s.emulator.isActive() && m_s.emulator.game().titleId() == game.titleId()) {
        enterInGame();
        return;
    }
    const EmulatorController::LaunchCheck check = m_s.emulator.checkLaunch(game);
    if (!check.ok()) {
        LaunchErrors::showLaunchProblem(this, m_s, game, check, [this, game] { play(game); });
        return;
    }
    QString error;
    m_inGame->clearExit();
    if (!m_s.emulator.start(game, m_inGame->hostWinId(), &error)) {
        EmulatorController::ExitInfo info;
        info.kind = EmulatorController::ExitInfo::Kind::Error;
        info.message = error;
        info.game = game;
        LaunchErrors::showExit(this, info, [this, game] { play(game); });
        return;
    }
    if (!m_s.emulator.runner().isBuiltIn()) {
        showPage(Page::Library); // it opens its own window; the library says so and the floating bar follows it
        return;
    }
    enterInGame();
}

void MainWindow::enterInGame()
{
    if (m_s.emulator.isActive() && !m_s.emulator.runner().isBuiltIn()) {
        NativeWindow::activate(m_s.emulator.gameWindow());
        return;
    }
    m_inGameMode = true;
    syncMode();
}

void MainWindow::leaveInGame()
{
    if (isFullScreen())
        toggleFullScreen();
    const EmulatorController::State st = m_s.emulator.state();
    if ((st == EmulatorController::State::Running || st == EmulatorController::State::Booting) && m_s.emulator.canPause())
        m_s.emulator.pause();
    m_inGame->clearExit();
    m_inGameMode = false;
    syncMode();
}

void MainWindow::toggleFullScreen()
{
    if (!isFullScreen()) {
        if (!m_inGameMode)
            return;
        m_wasMaximized = isMaximized();
        showFullScreen();
        m_inGame->setFullScreen(true);
        return;
    }
    m_inGame->setFullScreen(false);
    if (m_wasMaximized)
        showMaximized();
    else
        showNormal();
}

void MainWindow::addGameFolder()
{
    const QString dir = QFileDialog::getExistingDirectory(this, QStringLiteral("Add game folder"));
    if (dir.isEmpty())
        return;
    QStringList folders = m_s.settings.gameFolders();
    if (!folders.contains(dir))
        folders << dir;
    m_s.settings.setGameFolders(folders);
}

void MainWindow::paintEvent(QPaintEvent*)
{
    QPainter p(this);
    p.fillRect(rect(), Theme::color(Theme::Role::Bg));
}

#ifdef Q_OS_WIN
bool MainWindow::nativeEvent(const QByteArray& eventType, void* message, qintptr* result)
{
    MSG* msg = static_cast<MSG*>(message);
    if (msg->message == WM_NCHITTEST && !isMaximized() && !isFullScreen()) {
        RECT r;
        GetWindowRect(msg->hwnd, &r);
        const LONG x = GET_X_LPARAM(msg->lParam);
        const LONG y = GET_Y_LPARAM(msg->lParam);
        const LONG b = LONG(kResizeBorder * devicePixelRatioF());
        const bool left = x < r.left + b;
        const bool right = x >= r.right - b;
        const bool top = y < r.top + b;
        const bool bottom = y >= r.bottom - b;
        LRESULT hit = 0;
        if (top && left) hit = HTTOPLEFT;
        else if (top && right) hit = HTTOPRIGHT;
        else if (bottom && left) hit = HTBOTTOMLEFT;
        else if (bottom && right) hit = HTBOTTOMRIGHT;
        else if (left) hit = HTLEFT;
        else if (right) hit = HTRIGHT;
        else if (top) hit = HTTOP;
        else if (bottom) hit = HTBOTTOM;
        if (hit) {
            *result = hit;
            return true;
        }
    }
    return QWidget::nativeEvent(eventType, message, result);
}
#else
bool MainWindow::eventFilter(QObject* o, QEvent* e)
{
    if (e->type() == QEvent::MouseButtonPress && o->isWidgetType() && static_cast<QWidget*>(o)->window() == this
        && !isMaximized() && !isFullScreen() && windowHandle()) {
        auto* me = static_cast<QMouseEvent*>(e);
        const QPoint pos = mapFromGlobal(me->globalPosition().toPoint());
        Qt::Edges edges;
        if (pos.x() < kResizeBorder) edges |= Qt::LeftEdge;
        if (pos.x() >= width() - kResizeBorder) edges |= Qt::RightEdge;
        if (pos.y() < kResizeBorder) edges |= Qt::TopEdge;
        if (pos.y() >= height() - kResizeBorder) edges |= Qt::BottomEdge;
        if (me->button() == Qt::LeftButton && edges && windowHandle()->startSystemResize(edges))
            return true;
    }
    return QWidget::eventFilter(o, e);
}
#endif
