#include "FloatingBar.h"

#include "app/Services.h"
#include "core/NativeWindow.h"
#include "widgets/Button.h"
#include "widgets/CoverThumb.h"
#include "widgets/Icon.h"
#include "widgets/Text.h"

#include <QGuiApplication>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QScreen>
#include <QVBoxLayout>

#ifdef Q_OS_WIN
#include <windows.h>
#endif

namespace {

#ifdef Q_OS_WIN
constexpr int kHotkeyId = 0x5845; // "XE"
#endif

class Rule : public QWidget {
public:
    Rule() { setFixedSize(1, 16); }

protected:
    void paintEvent(QPaintEvent*) override { QPainter(this).fillRect(rect(), Theme::withAlpha(Theme::color(Theme::Role::HudText2), 0.3)); }
};

QString runningFor(qint64 s)
{
    const qint64 m = s / 60;
    if (m < 1)
        return QStringLiteral("Running");
    if (m < 60)
        return QStringLiteral("Running %1 min").arg(m);
    return QStringLiteral("Running %1 h %2 min").arg(m / 60).arg(m % 60);
}

}

FloatingBar::FloatingBar(Services& services)
    : QWidget(nullptr, Qt::Tool | Qt::FramelessWindowHint | Qt::WindowStaysOnTopHint | Qt::WindowDoesNotAcceptFocus)
    , m_s(services)
    , m_pill(new QWidget)
    , m_cover(new CoverThumb(QSize(14, 20)))
    , m_title(new Label({}, Label::Style::SmallStrong, Theme::Role::HudText))
    , m_runner(new Label({}, Label::Style::Small, Theme::Role::HudText))
    , m_time(new Label({}, Label::Style::Small, Theme::Role::HudText2))
    , m_show(new Button(QStringLiteral("Show XSEmu Launcher"), Button::Variant::OnArt, Button::Size::Tiny, QStringLiteral("back")))
    , m_stop(new Button(QStringLiteral("Stop"), Button::Variant::DangerGhost, Button::Size::Tiny, QStringLiteral("stop")))
{
    setObjectName(QStringLiteral("floatingBar"));
    setAttribute(Qt::WA_TranslucentBackground);
    setAttribute(Qt::WA_ShowWithoutActivating);
    setWindowTitle(QStringLiteral("XSEmu Launcher"));
    m_show->setObjectName(QStringLiteral("floatingShow"));
    m_stop->setObjectName(QStringLiteral("floatingStop"));

    auto* v = new QVBoxLayout(this);
    v->setContentsMargins(10, 8, 10, 8);
    v->setSpacing(6);
    auto* h = new QHBoxLayout(m_pill);
    h->setContentsMargins(12, 8, 8, 8);
    h->setSpacing(10);
    h->addWidget(new BrandMark(18));
    h->addWidget(m_cover);
    h->addSpacing(-4);
    h->addWidget(m_title);
    h->addWidget(new Rule);
    h->addWidget(new IconView(QStringLiteral("runner"), Theme::Role::HudText2, 12));
    h->addWidget(m_runner);
    h->addSpacing(-6);
    h->addWidget(new Label(QStringLiteral("\u00B7 own window"), Label::Style::Small, Theme::Role::HudText2));
    h->addWidget(new Rule);
    h->addWidget(m_time);
    h->addWidget(m_show);
    h->addWidget(m_stop);
    v->addWidget(m_pill);
#ifdef Q_OS_WIN
    const QString hint = QStringLiteral("Always on top while the runner's window has focus. Drag to move, Ctrl+Shift+H hides it.");
#else
    const QString hint = QStringLiteral("Drag to move.");
#endif
    auto* hintLabel = new Label(hint, Label::Style::Caption, Theme::Role::Text3);
    hintLabel->setContentsMargins(4, 0, 0, 0);
    v->addWidget(hintLabel);

    connect(m_show, &Button::clicked, this, &FloatingBar::showLauncher);
    connect(m_stop, &Button::clicked, &m_s.emulator, &EmulatorController::stop);
    connect(&m_s.emulator, &EmulatorController::stateChanged, this, [this](EmulatorController::State st) {
        if (st == EmulatorController::State::Idle)
            m_moved = false;
        sync();
    });
    connect(&m_s.emulator, &EmulatorController::gameWindowChanged, this, &FloatingBar::sync);
    connect(&m_s.covers, &CoverArtService::artChanged, this, [this] {
        m_coverTitle.clear();
        sync();
    });
    connect(&Theme::instance(), &Theme::changed, this, qOverload<>(&QWidget::update));
    m_poll.setInterval(250);
    connect(&m_poll, &QTimer::timeout, this, &FloatingBar::sync);
    m_poll.start();
#ifdef Q_OS_WIN
    m_hotkey = RegisterHotKey(reinterpret_cast<HWND>(winId()), kHotkeyId, MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT, 'H');
#endif
}

FloatingBar::~FloatingBar()
{
#ifdef Q_OS_WIN
    if (m_hotkey)
        UnregisterHotKey(reinterpret_cast<HWND>(winId()), kHotkeyId);
#endif
}

void FloatingBar::sync()
{
    using S = EmulatorController::State;
    const S st = m_s.emulator.state();
    const bool ownWindow = st != S::Idle && !m_s.emulator.runner().isBuiltIn();
    const quintptr game = m_s.emulator.gameWindow();
    if (!ownWindow) {
        m_userHidden = false;
        hide();
        return;
    }
    const Game& g = m_s.emulator.game();
    if (g.titleId() != m_coverTitle) {
        m_coverTitle = g.titleId();
        m_cover->setCover(m_s.art(g.titleId(), CoverCache::Kind::Grid), g.title());
    }
    m_title->setText(g.title());
    m_runner->setText(m_s.emulator.runner().name);
    m_time->setText(st == S::Booting    ? QStringLiteral("Starting")
                    : st == S::Stopping ? QStringLiteral("Stopping")
                                        : runningFor(m_s.emulator.sessionSeconds()));
    m_stop->setEnabled(st != S::Stopping);
    const bool wanted = m_screenshot || (game && !m_userHidden && NativeWindow::isForeground(game, isVisible() ? winId() : 0));
    if (!wanted) {
        hide();
        return;
    }
    adjustSize();
    if (!m_moved)
        place();
    if (!isVisible())
        show();
}

void FloatingBar::showForScreenshot()
{
    m_screenshot = true;
    sync();
}

void FloatingBar::place()
{
    const QRect game = NativeWindow::geometry(m_s.emulator.gameWindow());
    QScreen* screen = game.isEmpty() ? QGuiApplication::primaryScreen() : QGuiApplication::screenAt(game.center());
    if (!screen)
        screen = QGuiApplication::primaryScreen();
    const QRect avail = screen->availableGeometry();
    const QRect over = game.isEmpty() ? avail : game;
    int y = over.top() - height() - 4;
    if (y < avail.top())
        y = over.top() + 8;
    move(over.center().x() - width() / 2, y);
}

void FloatingBar::paintEvent(QPaintEvent*)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    const QRectF pill = QRectF(m_pill->geometry()).adjusted(0.5, 0.5, -0.5, -0.5);
    QPainterPath path;
    path.addRoundedRect(pill, 10, 10);
    p.fillPath(path.translated(0, 2), QColor(0, 0, 0, 60));
    p.fillPath(path, Theme::color(Theme::Role::Hud));
    p.setPen(QPen(QColor(255, 255, 255, 22), 1));
    p.drawPath(path);
}

void FloatingBar::mousePressEvent(QMouseEvent* e)
{
    if (e->button() != Qt::LeftButton)
        return;
    m_dragging = true;
    m_dragFrom = e->globalPosition().toPoint() - pos();
}

void FloatingBar::mouseMoveEvent(QMouseEvent* e)
{
    if (!m_dragging)
        return;
    move(e->globalPosition().toPoint() - m_dragFrom);
    m_moved = true;
}

void FloatingBar::mouseReleaseEvent(QMouseEvent*)
{
    m_dragging = false;
}

bool FloatingBar::nativeEvent(const QByteArray& type, void* message, qintptr* result)
{
#ifdef Q_OS_WIN
    const MSG* msg = static_cast<const MSG*>(message);
    if (msg->message == WM_HOTKEY && msg->wParam == kHotkeyId) {
        m_userHidden = !m_userHidden;
        sync();
        return true;
    }
#endif
    return QWidget::nativeEvent(type, message, result);
}
