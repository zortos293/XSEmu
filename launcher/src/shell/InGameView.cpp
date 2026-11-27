#include "InGameView.h"

#include "TitleBar.h"
#include "WindowButtons.h"
#include "app/Services.h"
#include "core/Format.h"
#include "core/Paths.h"
#include "core/ShaderCacheInfo.h"
#include "theme/Theme.h"
#include "widgets/Badge.h"
#include "widgets/Button.h"
#include "widgets/Card.h"
#include "widgets/Icon.h"
#include "widgets/Spinner.h"
#include "widgets/Text.h"

#include <QApplication>
#include <QDesktopServices>
#include <QDir>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QPropertyAnimation>
#include <QScreen>
#include <QShortcut>
#include <QUrl>
#include <QVBoxLayout>
#include <QWindow>

#ifdef Q_OS_WIN
#include <windows.h>
#endif

namespace {

QString placeText(Services& s)
{
    const auto hello = s.emulator.hello();
    const QString device = hello && !hello->deviceType.isEmpty() ? hello->deviceType : s.settings.defaultDeviceType();
    return QStringLiteral("%1 · %2").arg(Services::consoleName(device), s.settings.offline() ? QStringLiteral("Offline") : QStringLiteral("Online"));
}

class Separator : public QWidget {
public:
    Separator() { setFixedSize(1, 20); }

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.fillRect(rect(), Theme::color(Theme::Role::Border));
    }
};

class GameBar : public QWidget {
public:
    GameBar()
    {
        setFixedHeight(Metrics::GameBar);
        connect(&Theme::instance(), &Theme::changed, this, qOverload<>(&QWidget::update));
    }

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.fillRect(rect(), Theme::color(Theme::Role::Surface));
        p.fillRect(QRect(0, height() - 1, width(), 1), Theme::color(Theme::Role::Border));
    }
    void mousePressEvent(QMouseEvent* e) override { WindowDrag::press(this, e); }
    void mouseDoubleClickEvent(QMouseEvent* e) override { WindowDrag::doubleClick(this, e); }
};

// Everything around the 16:9 host is letterbox, so the area is black too.
class GameArea : public QWidget {
public:
    std::function<void()> onResize;
    std::function<void()> onPress;

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.fillRect(rect(), Qt::black);
    }
    void resizeEvent(QResizeEvent*) override
    {
        if (onResize)
            onResize();
    }
    void mousePressEvent(QMouseEvent*) override
    {
        if (onPress)
            onPress();
    }
};

// A launcher panel drawn above the game's child window: a native child of the host, rounded with a
// window mask (native windows cannot be translucent), raised whenever the game window may have been
// created or restacked.
class OverlayPanel : public QWidget {
public:
    OverlayPanel(QWidget* host, int radius)
        : QWidget(host)
        , m_radius(radius)
    {
        setAttribute(Qt::WA_NativeWindow);
        setAttribute(Qt::WA_DontCreateNativeAncestors);
        hide();
    }

protected:
    void resizeEvent(QResizeEvent*) override
    {
        if (m_radius <= 0) {
            clearMask();
            return;
        }
        QPainterPath path;
        path.addRoundedRect(QRectF(rect()), m_radius, m_radius);
        setMask(QRegion(path.toFillPolygon().toPolygon()));
    }
    int m_radius;
};

}

// The game's small cover in the bar and the boot screen: cached SteamGridDB art, or a neutral tile.
class GameCover : public QWidget {
public:
    GameCover(QSize size, int radius, bool onDark)
        : m_radius(radius)
        , m_onDark(onDark)
    {
        setFixedSize(size);
        connect(&Theme::instance(), &Theme::changed, this, qOverload<>(&QWidget::update));
    }
    void setArt(const QPixmap& art)
    {
        m_art = art;
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        p.setRenderHint(QPainter::SmoothPixmapTransform);
        QPainterPath clip;
        clip.addRoundedRect(QRectF(rect()), m_radius, m_radius);
        if (m_art.isNull()) {
            p.fillPath(clip, m_onDark ? QColor(0x2A, 0x2A, 0x2E) : Theme::color(Theme::Role::Sunken));
            Icon::paint(p, QRectF(rect()), QStringLiteral("gamepad"), m_onDark ? QColor(0x8A, 0x8A, 0x91) : Theme::color(Theme::Role::Text3), qMax(10, width() / 3));
            return;
        }
        p.setClipPath(clip);
        const QSizeF s = QSizeF(m_art.size()).scaled(QSizeF(size()), Qt::KeepAspectRatioByExpanding);
        p.drawPixmap(QRectF((width() - s.width()) / 2, (height() - s.height()) / 2, s.width(), s.height()), m_art, QRectF(m_art.rect()));
    }

private:
    QPixmap m_art;
    int m_radius;
    bool m_onDark;
};

class GameHost : public QWidget {
public:
    std::function<void()> onClick;
    GameHost()
    {
        setAttribute(Qt::WA_DontCreateNativeAncestors);
        setAttribute(Qt::WA_NativeWindow);
        setAttribute(Qt::WA_NoSystemBackground);
        setFocusPolicy(Qt::NoFocus);
    }

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.fillRect(rect(), Qt::black);
    }
    void mousePressEvent(QMouseEvent*) override
    {
        if (onClick)
            onClick();
    }
#ifdef Q_OS_WIN
    // A click on the game's child window (another process) reaches the host as WM_PARENTNOTIFY.
    bool nativeEvent(const QByteArray& type, void* message, qintptr* result) override
    {
        const MSG* msg = static_cast<MSG*>(message);
        if (msg->message == WM_PARENTNOTIFY) {
            const UINT what = LOWORD(msg->wParam);
            if ((what == WM_LBUTTONDOWN || what == WM_RBUTTONDOWN || what == WM_MBUTTONDOWN) && onClick)
                onClick();
        }
        return QWidget::nativeEvent(type, message, result);
    }
#endif
};

class BootPanel : public OverlayPanel {
public:
    explicit BootPanel(QWidget* host)
        : OverlayPanel(host, 0)
        , m_cover(new GameCover(QSize(72, 108), 6, true))
        , m_title(new Label({}, Label::Style::SectionTitle, Theme::Role::HudText))
        , m_subtitle(new Label({}, Label::Style::Small, Theme::Role::HudText2))
        , m_step1Icon(new IconView(QStringLiteral("check"), Theme::Role::Success, 14))
        , m_step1Spin(new Spinner(14))
        , m_step1(new Label({}, Label::Style::Small, Theme::Role::HudText))
        , m_step1Time(new Label({}, Label::Style::Mono, Theme::Role::HudText2))
        , m_step2Spin(new Spinner(14))
        , m_step2(new Label(QStringLiteral("Waiting for the first frame"), Label::Style::Small, Theme::Role::HudText))
    {
        m_title->setFont(Theme::font(18, 600, -0.01));
        auto* v = new QVBoxLayout(this);
        v->addStretch(1);
        v->addWidget(m_cover, 0, Qt::AlignHCenter);
        v->addSpacing(22);
        v->addWidget(m_title, 0, Qt::AlignHCenter);
        v->addSpacing(4);
        v->addWidget(m_subtitle, 0, Qt::AlignHCenter);
        v->addSpacing(28);
        auto* steps = new QWidget;
        steps->setFixedWidth(340);
        auto* g = new QVBoxLayout(steps);
        g->setContentsMargins(0, 0, 0, 0);
        g->setSpacing(10);
        auto row = [](QWidget* icon, QWidget* extra, Label* text, Label* time) {
            auto* h = new QHBoxLayout;
            h->setSpacing(10);
            h->addWidget(icon);
            if (extra)
                h->addWidget(extra);
            h->addWidget(text, 1);
            if (time)
                h->addWidget(time);
            return h;
        };
        g->addLayout(row(m_step1Icon, m_step1Spin, m_step1, m_step1Time));
        g->addLayout(row(m_step2Spin, nullptr, m_step2, nullptr));
        v->addWidget(steps, 0, Qt::AlignHCenter);
        v->addStretch(1);
    }
    void set(const QString& title, const QPixmap& art, const QString& subtitle, const QString& emulatorName, double helloSecs)
    {
        m_title->setText(QStringLiteral("Starting %1").arg(title));
        m_cover->setArt(art);
        m_subtitle->setText(subtitle);
        m_subtitle->setVisible(!subtitle.isEmpty());
        m_step1->setText(QStringLiteral("Starting %1").arg(emulatorName));
        const bool done = helloSecs >= 0;
        m_step1Icon->setVisible(done);
        m_step1Spin->setVisible(!done);
        m_step1->setRole(done ? Theme::Role::HudText2 : Theme::Role::HudText);
        m_step1Time->setText(done ? QStringLiteral("%1 s").arg(helloSecs, 0, 'f', 1) : QString());
        m_step2->setVisible(done);
        m_step2Spin->setVisible(done);
    }

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.fillRect(rect(), QColor(0x0E, 0x0E, 0x10));
    }

private:
    GameCover* m_cover;
    Label* m_title;
    Label* m_subtitle;
    IconView* m_step1Icon;
    Spinner* m_step1Spin;
    Label* m_step1;
    Label* m_step1Time;
    Spinner* m_step2Spin;
    Label* m_step2;
};

class PausedPanel : public OverlayPanel {
public:
    PausedPanel(QWidget* host, std::function<void()> resume, std::function<void()> stop)
        : OverlayPanel(host, 14)
        , m_text(new Label({}, Label::Style::Small, Theme::Role::HudText2))
    {
        setFixedSize(360, 228);
        auto* v = new QVBoxLayout(this);
        v->setContentsMargins(26, 22, 26, 20);
        v->setSpacing(0);
        auto* icon = new QWidget;
        icon->setFixedSize(40, 40);
        v->addWidget(icon, 0, Qt::AlignHCenter);
        m_icon = icon;
        v->addSpacing(12);
        auto* title = new Label(QStringLiteral("Paused"), Label::Style::SectionTitle, Theme::Role::HudText);
        title->setFont(Theme::font(18, 600, -0.01));
        v->addWidget(title, 0, Qt::AlignHCenter);
        v->addSpacing(8);
        m_text->setWordWrap(true);
        m_text->setAlignment(Qt::AlignHCenter);
        v->addWidget(m_text);
        v->addSpacing(16);
        auto* buttons = new QHBoxLayout;
        buttons->setSpacing(8);
        buttons->addStretch(1);
        auto* r = new Button(QStringLiteral("Resume"), Button::Variant::Primary, Button::Size::Small, QStringLiteral("play"));
        r->setObjectName(QStringLiteral("pausedResume"));
        auto* s = new Button(QStringLiteral("Stop game"), Button::Variant::OnArt, Button::Size::Small, QStringLiteral("stop"));
        s->setObjectName(QStringLiteral("pausedStop"));
        buttons->addWidget(r);
        buttons->addWidget(s);
        buttons->addStretch(1);
        v->addLayout(buttons);
        v->addSpacing(14);
        v->addWidget(new Label(QStringLiteral("F1 resumes · F3 overlay · F11 fullscreen"), Label::Style::Caption, Theme::Role::HudText2), 0, Qt::AlignHCenter);
        connect(r, &Button::clicked, this, resume);
        connect(s, &Button::clicked, this, stop);
    }
    void setHeld(qint64 bytes)
    {
        m_text->setText(bytes > 0 ? QStringLiteral("The game still holds %1. Stop it to free memory for a warm-up or another game.").arg(Format::gigabytes(bytes))
                                  : QStringLiteral("The game keeps its memory while paused. Stop it to free memory for a warm-up or another game."));
    }

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        p.fillRect(rect(), QColor(0x1C, 0x1C, 0x1F));
        p.setPen(QPen(QColor(255, 255, 255, 26), 1));
        p.setBrush(Qt::NoBrush);
        p.drawRoundedRect(QRectF(rect()).adjusted(0.5, 0.5, -0.5, -0.5), 14, 14);
        const QRectF c = QRectF(m_icon->geometry());
        p.setPen(Qt::NoPen);
        p.setBrush(QColor(255, 255, 255, 30));
        p.drawEllipse(c);
        Icon::paint(p, c, QStringLiteral("pause"), QColor(0xF2, 0xF2, 0xF3), 16);
    }

private:
    Label* m_text;
    QWidget* m_icon = nullptr;
};

class ExitPanel : public QWidget {
public:
    ExitPanel()
        : m_card(new Card(Card::Kind::Surface, 16))
        , m_iconTile(new QWidget)
        , m_title(new Label({}, Label::Style::SectionTitle))
        , m_body(new Label({}, Label::Style::Small, Theme::Role::Text2))
        , m_message(new Label({}, Label::Style::Mono, Theme::Role::Text2))
        , m_box(new Card(Card::Kind::Sunken, Metrics::RadiusLg))
        , m_path(new Label({}, Label::Style::Mono, Theme::Role::Text))
        , m_caption(new Label({}, Label::Style::Caption, Theme::Role::Text3))
        , relaunch(new Button(QStringLiteral("Relaunch"), Button::Variant::Primary, Button::Size::Large, QStringLiteral("play")))
        , report(new Button(QStringLiteral("Open crash report"), Button::Variant::Secondary, Button::Size::Large, QStringLiteral("file")))
        , back(new Button(QStringLiteral("Back to library"), Button::Variant::Ghost, Button::Size::Large))
    {
        relaunch->setObjectName(QStringLiteral("exitRelaunch"));
        report->setObjectName(QStringLiteral("exitReport"));
        back->setObjectName(QStringLiteral("exitBack"));
        m_card->setFixedWidth(500);
        auto* outer = new QVBoxLayout(this);
        outer->addStretch(1);
        outer->addWidget(m_card, 0, Qt::AlignHCenter);
        outer->addStretch(1);
        auto* v = new QVBoxLayout(m_card);
        v->setContentsMargins(24, 24, 24, 22);
        v->setSpacing(0);
        m_iconTile->setFixedSize(36, 36);
        m_iconTile->installEventFilter(this);
        v->addWidget(m_iconTile);
        v->addSpacing(16);
        v->addWidget(m_title);
        v->addSpacing(8);
        m_body->setWordWrap(true);
        v->addWidget(m_body);
        v->addSpacing(8);
        m_message->setWordWrap(true);
        m_message->setTextInteractionFlags(Qt::TextSelectableByMouse);
        v->addWidget(m_message);
        v->addSpacing(8);
        auto* bv = new QVBoxLayout(m_box);
        bv->setContentsMargins(12, 10, 12, 11);
        bv->setSpacing(4);
        m_path->setTextInteractionFlags(Qt::TextSelectableByMouse);
        m_caption->setWordWrap(true);
        bv->addWidget(m_path);
        bv->addWidget(m_caption);
        v->addWidget(m_box);
        v->addSpacing(20);
        auto* buttons = new QHBoxLayout;
        buttons->setSpacing(8);
        buttons->addWidget(relaunch);
        buttons->addWidget(report);
        buttons->addStretch(1);
        buttons->addWidget(back);
        v->addLayout(buttons);
        hide();
    }

    void set(const EmulatorController::ExitInfo& info)
    {
        using K = EmulatorController::ExitInfo::Kind;
        const QString title = info.game.title();
        const QString code = Format::exitCode(info.exitCode);
        const QString name = Format::exitCodeName(info.exitCode);
        const QString after = info.playedSeconds > 0 ? QStringLiteral(" after %1").arg(Format::playTime(info.playedSeconds)) : QString();
        const QString saved = info.playedSeconds > 0 ? QStringLiteral("play time was saved, and ") : QString();
        m_logs = Paths::logsDir(info.game.titleId());
        m_report = Paths::reportJson(info.game.titleId());
        QString body;
        QString caption = QStringLiteral("The log (xcl.log) and the crash report (report.json). Stays on this PC, nothing is uploaded.");
        switch (info.kind) {
        case K::DeviceRemoved:
            m_title->setText(QStringLiteral("The graphics driver reset"));
            body = QStringLiteral("Windows removed the GPU device%1. This is usually a driver timeout. %2the memory is free again.")
                       .arg(after, info.playedSeconds > 0 ? QStringLiteral("Play time was saved and ") : QStringLiteral("The launcher is fine and "));
            relaunch->setText(QStringLiteral("Relaunch"));
            report->setText(QStringLiteral("Open crash report"));
            break;
        case K::Error:
            m_title->setText(QStringLiteral("%1 couldn't start").arg(title));
            body = QStringLiteral("XSEmu Core stopped before the game showed a frame (exit code %1). Nothing was changed, and the memory is free again.").arg(code);
            caption = QStringLiteral("XSEmu Core's log for this game. Stays on this PC, nothing is uploaded.");
            relaunch->setText(QStringLiteral("Try again"));
            report->setText(QStringLiteral("Open log"));
            break;
        case K::Crash:
        case K::Stopped:
        case K::TitleExit:
            m_title->setText(QStringLiteral("%1 stopped").arg(title));
            if (!info.sawExitLine)
                body = QStringLiteral("XSEmu Core ended without saying why, with code %1%2%3. The launcher is fine, %4the memory is free again.")
                           .arg(code, name.isEmpty() ? QString() : QStringLiteral(" (%1)").arg(name), after, saved);
            else
                body = QStringLiteral("XSEmu Core exited with code %1%2%3. The launcher is fine, %4the memory is free again.")
                           .arg(code, name.isEmpty() ? QString() : QStringLiteral(" (%1)").arg(name), after, saved);
            relaunch->setText(QStringLiteral("Relaunch"));
            report->setText(QStringLiteral("Open crash report"));
            break;
        }
        m_body->setText(body);
        m_message->setText(info.message);
        m_message->setVisible(!info.message.isEmpty());
        const QString path = QDir::toNativeSeparators(m_logs);
        m_path->setText(m_path->fontMetrics().elidedText(path, Qt::ElideMiddle, m_card->width() - 72));
        m_path->setToolTip(path);
        m_caption->setText(caption);
        m_iconName = info.kind == K::Error ? QStringLiteral("file") : QStringLiteral("warning");
        m_iconTile->update();
        m_card->setFixedHeight(m_card->heightForWidth(m_card->width()));
    }

    void openReport() const
    {
        QDesktopServices::openUrl(QUrl::fromLocalFile(QFileInfo::exists(m_report) ? m_report : m_logs));
    }

protected:
    bool eventFilter(QObject* o, QEvent* e) override
    {
        if (o == m_iconTile && e->type() == QEvent::Paint) {
            QPainter p(m_iconTile);
            p.setRenderHint(QPainter::Antialiasing);
            p.setPen(Qt::NoPen);
            p.setBrush(Theme::toneSoft(Theme::Tone::Danger));
            p.drawRoundedRect(QRectF(m_iconTile->rect()), 8, 8);
            Icon::paint(p, QRectF(m_iconTile->rect()), m_iconName, Theme::toneColor(Theme::Tone::Danger), 16);
            return true;
        }
        return QWidget::eventFilter(o, e);
    }

private:
    Card* m_card;
    QWidget* m_iconTile;
    Label* m_title;
    Label* m_body;
    Label* m_message;
    Card* m_box;
    Label* m_path;
    Label* m_caption;
    QString m_logs;
    QString m_report;
    QString m_iconName = QStringLiteral("warning");

public:
    Button* relaunch;
    Button* report;
    Button* back;
};

// "Esc or F11 exits fullscreen · F1 pause · F3 overlay": shown on entering fullscreen, then fades.
// A frameless tool window owned by the main window, so it stacks above the game's child window.
class KeyHint : public QWidget {
public:
    explicit KeyHint(QWidget* owner)
        : QWidget(owner, Qt::Tool | Qt::FramelessWindowHint | Qt::WindowDoesNotAcceptFocus)
        , m_fade(this, "windowOpacity")
    {
        setAttribute(Qt::WA_ShowWithoutActivating);
        setObjectName(QStringLiteral("keyHint"));
        const QFont text = Theme::font(12, 500);
        const QFont key = Theme::mono(11, 500);
        int w = 12;
        for (const Token& t : tokens())
            w += tokenWidth(t, text, key) + 6;
        setFixedSize(w + 6, 30);
        QPainterPath path;
        path.addRoundedRect(QRectF(rect()), 8, 8);
        setMask(QRegion(path.toFillPolygon().toPolygon()));
        m_hold.setSingleShot(true);
        m_hold.setInterval(2600);
        connect(&m_hold, &QTimer::timeout, this, [this] {
            m_fade.stop();
            m_fade.setDuration(600);
            m_fade.setStartValue(1.0);
            m_fade.setEndValue(0.0);
            m_fade.start();
        });
        connect(&m_fade, &QPropertyAnimation::finished, this, &QWidget::hide);
    }

    void showOn(const QRect& area)
    {
        m_fade.stop();
        setWindowOpacity(1.0);
        move(area.center().x() - width() / 2, area.top() + 24);
        show();
        raise();
        m_hold.start();
    }

    void dismiss()
    {
        m_hold.stop();
        m_fade.stop();
        hide();
    }

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        p.fillRect(rect(), QColor(0x1C, 0x1C, 0x1F));
        p.setPen(QPen(QColor(255, 255, 255, 30), 1));
        p.drawRoundedRect(QRectF(rect()).adjusted(0.5, 0.5, -0.5, -0.5), 8, 8);
        const QFont text = Theme::font(12, 500);
        const QFont key = Theme::mono(11, 500);
        qreal x = 12;
        for (const Token& t : tokens()) {
            const int w = tokenWidth(t, text, key);
            if (t.kind == Token::Sep) {
                p.fillRect(QRectF(x + w / 2.0, 8, 1, height() - 16), QColor(255, 255, 255, 40));
            } else if (t.kind == Token::Key) {
                const QRectF cap(x, 6, w, 18);
                p.setPen(QPen(QColor(255, 255, 255, 70), 1));
                p.setBrush(QColor(255, 255, 255, 18));
                p.drawRoundedRect(cap.adjusted(0.5, 0.5, -0.5, -0.5), 4, 4);
                p.setFont(key);
                p.setPen(QColor(0xF2, 0xF2, 0xF3));
                p.drawText(cap, Qt::AlignCenter, t.text);
            } else {
                p.setFont(text);
                p.setPen(t.quiet ? QColor(0xA8, 0xA8, 0xAE) : QColor(0xF2, 0xF2, 0xF3));
                p.drawText(QRectF(x, 0, w, height()), Qt::AlignCenter, t.text);
            }
            x += w + 6;
        }
    }

private:
    struct Token {
        enum Kind { Key, Text, Sep } kind;
        QString text;
        bool quiet = false;
    };
    static QList<Token> tokens()
    {
        return {{Token::Key, QStringLiteral("Esc")}, {Token::Text, QStringLiteral("or"), true}, {Token::Key, QStringLiteral("F11")},
                {Token::Text, QStringLiteral("exits fullscreen")}, {Token::Sep, {}}, {Token::Key, QStringLiteral("F1")},
                {Token::Text, QStringLiteral("pause")}, {Token::Key, QStringLiteral("F3")}, {Token::Text, QStringLiteral("overlay")}};
    }
    static int tokenWidth(const Token& t, const QFont& text, const QFont& key)
    {
        switch (t.kind) {
        case Token::Key: return QFontMetrics(key).horizontalAdvance(t.text) + 12;
        case Token::Text: return QFontMetrics(text).horizontalAdvance(t.text);
        case Token::Sep: return 9;
        }
        return 0;
    }

    QTimer m_hold;
    QPropertyAnimation m_fade;
};

InGameView::InGameView(Services& services, QWidget* parent)
    : QWidget(parent)
    , m_s(services)
    , m_bar(new GameBar)
    , m_cover(new GameCover(QSize(16, 22), 3, false))
    , m_title(new Label({}, Label::Style::SmallStrong))
    , m_runner(new Label({}, Label::Style::Small, Theme::Role::Text2))
    , m_spinner(new Spinner(14))
    , m_paused(new Badge({}, Theme::Tone::Warning, Badge::Kind::Pill))
    , m_place(new Badge({}, Theme::Tone::Success, Badge::Kind::Plain))
    , m_stopped(new Badge({}, Theme::Tone::Danger, Badge::Kind::Plain))
    , m_perfSep(new Separator)
    , m_perf(new Label({}, Label::Style::Number, Theme::Role::Text2))
    , m_ramSep(new Separator)
    , m_ramLabel(new Label(QStringLiteral("RAM free"), Label::Style::Small, Theme::Role::Text3))
    , m_ramValue(new Label({}, Label::Style::Number, Theme::Role::Text2))
    , m_pause(new Button({}, Button::Variant::Ghost, Button::Size::Small, QStringLiteral("pause")))
    , m_overlay(new Button({}, Button::Variant::Ghost, Button::Size::Small, QStringLiteral("overlay")))
    , m_full(new Button({}, Button::Variant::Ghost, Button::Size::Small, QStringLiteral("fullscreen")))
    , m_stop(new Button(QStringLiteral("Stop"), Button::Variant::DangerGhost, Button::Size::Small, QStringLiteral("stop")))
    , m_relaunch(new Button(QStringLiteral("Relaunch"), Button::Variant::Secondary, Button::Size::Small, QStringLiteral("play")))
    , m_area(new GameArea)
    , m_host(new GameHost)
    , m_boot(nullptr)
    , m_pausedPanel(nullptr)
    , m_exitPanel(new ExitPanel)
    , m_hint(nullptr)
{
    m_pause->setObjectName(QStringLiteral("barPause"));
    m_overlay->setObjectName(QStringLiteral("barOverlay"));
    m_full->setObjectName(QStringLiteral("barFullscreen"));
    m_stop->setObjectName(QStringLiteral("barStop"));
    m_relaunch->setObjectName(QStringLiteral("barRelaunch"));
    m_overlay->setCheckable(true);
    m_overlay->setToolTip(QStringLiteral("Performance overlay (F3)"));
    m_full->setToolTip(QStringLiteral("Fullscreen (F11)"));
    for (Button* b : {m_pause, m_overlay, m_full, m_stop, m_relaunch})
        b->setFocusPolicy(Qt::TabFocus);

    auto* v = new QVBoxLayout(this);
    v->setContentsMargins(0, 0, 0, 0);
    v->setSpacing(0);
    auto* h = new QHBoxLayout(m_bar);
    h->setContentsMargins(12, 0, 0, 1);
    h->setSpacing(8);
    h->addWidget(new BrandMark(16));
    auto* back = new Button(QStringLiteral("Library"), Button::Variant::Ghost, Button::Size::Small, QStringLiteral("back"));
    back->setObjectName(QStringLiteral("barBack"));
    back->setFocusPolicy(Qt::TabFocus);
    h->addWidget(back);
    h->addWidget(new Separator);
    h->addSpacing(2);
    h->addWidget(m_cover);
    h->addWidget(m_title);
    h->addSpacing(6);
    auto* runnerIcon = new IconView(QStringLiteral("runner"), Theme::Role::Text3, 12);
    h->addWidget(runnerIcon);
    m_runner->setObjectName(QStringLiteral("barRunner"));
    m_runner->setToolTip(QStringLiteral("The runner this game runs with"));
    h->addWidget(m_runner);
    h->addSpacing(4);
    h->addWidget(m_spinner);
    h->addWidget(m_paused);
    h->addWidget(m_place);
    h->addWidget(m_stopped);
    h->addSpacing(4);
    h->addWidget(m_perfSep);
    h->addWidget(m_perf);
    h->addWidget(m_ramSep);
    h->addWidget(m_ramLabel);
    h->addWidget(m_ramValue);
    h->addStretch(1);
    h->addWidget(m_pause);
    h->addWidget(m_overlay);
    h->addWidget(m_full);
    h->addWidget(m_stop);
    h->addWidget(m_relaunch);
    h->addSpacing(4);
    h->addWidget(new Separator);
    h->addWidget(new WindowButtons(Metrics::GameBar));
    v->addWidget(m_bar);
    v->addWidget(m_area, 1);

    auto* area = static_cast<GameArea*>(m_area);
    m_host->setParent(m_area);
    m_exitPanel->setParent(m_area);
    area->onResize = [this] { layoutArea(); };
    area->onPress = [this] { m_s.emulator.focusGame(); };
    m_host->onClick = [this] { m_s.emulator.focusGame(); };

    m_boot = new BootPanel(m_host);
    m_pausedPanel = new PausedPanel(m_host, [this] { m_s.emulator.resume(); }, [this] { m_s.emulator.stop(); });
    m_hint = new KeyHint(this);

    connect(back, &Button::clicked, this, &InGameView::backToLibrary);
    connect(m_pause, &Button::clicked, this, [this] { handleKey(QStringLiteral("F1")); });
    connect(m_overlay, &Button::clicked, this, [this] { m_s.emulator.setOverlay(!m_s.emulator.overlayOn()); });
    connect(m_full, &Button::clicked, this, &InGameView::fullScreenToggled);
    connect(m_stop, &Button::clicked, &m_s.emulator, &EmulatorController::stop);
    connect(m_relaunch, &Button::clicked, this, [this] { emit relaunchRequested(currentGame()); });
    connect(m_exitPanel->relaunch, &Button::clicked, this, [this] { emit relaunchRequested(currentGame()); });
    connect(m_exitPanel->report, &Button::clicked, this, [this] { m_exitPanel->openReport(); });
    connect(m_exitPanel->back, &Button::clicked, this, &InGameView::backToLibrary);

    for (const auto& [key, name] : {std::pair{Qt::Key_F1, "F1"}, {Qt::Key_F3, "F3"}, {Qt::Key_F11, "F11"}, {Qt::Key_Escape, "Escape"}}) {
        auto* sc = new QShortcut(QKeySequence(key), this);
        const QString k = QString::fromLatin1(name);
        connect(sc, &QShortcut::activated, this, [this, k] { handleKey(k); });
    }

    connect(&m_s.emulator, &EmulatorController::stateChanged, this, [this](EmulatorController::State st) {
        if (st == EmulatorController::State::Running)
            focusGameSoon();
        refresh();
    });
    connect(&m_s.emulator, &EmulatorController::statsChanged, this, &InGameView::refresh);
    connect(&m_s.shaderCache, &ShaderCacheMonitor::changed, this, &InGameView::refresh);
    connect(&m_s.emulator, &EmulatorController::helloReceived, this, [this] {
        refresh();
        raisePanels();
        focusGameSoon();
    });
    connect(&m_s.emulator, &EmulatorController::keyPressed, this, &InGameView::handleKey);
    connect(&m_s.settings, &AppSettings::changed, this, &InGameView::refresh);
    connect(&m_s.covers, &CoverArtService::artChanged, this, &InGameView::refresh);

    m_restack.setInterval(250);
    connect(&m_restack, &QTimer::timeout, this, &InGameView::raisePanels);
    m_autoHide.setInterval(150);
    connect(&m_autoHide, &QTimer::timeout, this, &InGameView::updateAutoHide);
    refresh();
}

quintptr InGameView::hostWinId()
{
    return static_cast<quintptr>(m_host->winId());
}

QWidget* InGameView::host() const
{
    return m_host;
}

Game InGameView::currentGame() const
{
    return m_exit ? m_exit->game : m_s.emulator.game();
}

void InGameView::setFullScreen(bool on)
{
    m_fullScreen = on;
    if (on)
        m_hint->showOn(screen() ? screen()->geometry() : window()->geometry());
    else
        m_hint->dismiss();
    refresh();
}

void InGameView::showExit(const EmulatorController::ExitInfo& info)
{
    m_exit = info;
    m_exitPanel->set(info);
    refresh();
}

void InGameView::clearExit()
{
    m_exit.reset();
    refresh();
}

void InGameView::handleKey(const QString& key)
{
    using S = EmulatorController::State;
    const S st = m_s.emulator.state();
    if (key == QLatin1String("F1")) {
        if (st == S::Paused)
            m_s.emulator.resume();
        else if (st == S::Running || st == S::Booting)
            m_s.emulator.pause();
    } else if (key == QLatin1String("F3")) {
        if (st == S::Running || st == S::Paused)
            m_s.emulator.setOverlay(!m_s.emulator.overlayOn());
    } else if (key == QLatin1String("F11")) {
        if (st != S::Idle || m_fullScreen)
            emit fullScreenToggled();
    } else if (key == QLatin1String("Escape")) {
        if (m_fullScreen)
            emit fullScreenToggled();
    }
}

void InGameView::showEvent(QShowEvent* e)
{
    QWidget::showEvent(e);
    layoutArea();
    raisePanels();
    focusGameSoon();
}

void InGameView::resizeEvent(QResizeEvent* e)
{
    QWidget::resizeEvent(e);
    layoutArea();
}

void InGameView::layoutArea()
{
    const QSize a = m_area->size();
    QSize s(a.width(), a.width() * 9 / 16);
    if (s.height() > a.height())
        s = QSize(a.height() * 16 / 9, a.height());
    m_host->setGeometry(QRect(QPoint((a.width() - s.width()) / 2, (a.height() - s.height()) / 2), s));
    m_exitPanel->setGeometry(m_area->rect());
    m_boot->setGeometry(m_host->rect());
    m_pausedPanel->move((s.width() - m_pausedPanel->width()) / 2, (s.height() - m_pausedPanel->height()) / 2);
}

void InGameView::raisePanels()
{
    // QWidget::raise() skips the window system when the widget is already Qt's topmost child, which
    // it stays while the game's foreign window sits above it, so restack the native windows directly.
    for (QWidget* panel : {static_cast<QWidget*>(m_boot), static_cast<QWidget*>(m_pausedPanel)}) {
        if (panel->isVisible() && panel->windowHandle())
            panel->windowHandle()->raise();
    }
}

void InGameView::focusGameSoon()
{
    if (!isVisible())
        return;
    QTimer::singleShot(120, this, [this] {
        if (isVisible())
            m_s.emulator.focusGame();
    });
}

void InGameView::updateAutoHide()
{
    const bool active = !m_fullScreen && !m_exit && m_s.settings.windowGameBarAutoHide() && m_s.emulator.state() == EmulatorController::State::Running && isVisible();
    if (!active) {
        m_autoHide.stop();
        if (m_barHidden) {
            m_barHidden = false;
            m_bar->setVisible(!m_fullScreen);
        }
        return;
    }
    const QPoint c = QCursor::pos();
    const QRect win = window()->geometry();
    const int reach = m_barHidden ? 6 : Metrics::GameBar + 8;
    const bool nearTop = win.contains(c) && c.y() - win.top() < reach;
    if (nearTop) {
        m_away.restart();
        if (m_barHidden) {
            m_barHidden = false;
            m_bar->show();
        }
    } else if (!m_barHidden && (!m_away.isValid() || m_away.elapsed() > 1500)) {
        m_barHidden = true;
        m_bar->hide();
    }
}

void InGameView::refresh()
{
    using S = EmulatorController::State;
    const S st = m_s.emulator.state();
    const Game g = currentGame();
    const EmulatorController::Stats stats = m_s.emulator.stats();
    const bool exit = m_exit.has_value();
    const bool live = !exit && st != S::Idle;

    m_title->setText(g.title());
    m_runner->setText(m_s.emulator.runnerFor(g).name);
    m_cover->setArt(m_s.art(g.titleId(), CoverCache::Kind::Grid));

    m_spinner->setVisible(live && (st == S::Booting || st == S::Stopping));
    m_paused->setVisible(live && st == S::Paused);
    if (st == S::Paused)
        m_paused->setText(stats.heldBytes > 0 ? QStringLiteral("Paused · %1 held").arg(Format::gigabytes(stats.heldBytes)) : QStringLiteral("Paused"));
    m_place->setVisible(live);
    m_place->setDotVisible(st == S::Running || st == S::Paused);
    m_place->setTone(Theme::Tone::Success);
    m_place->setText(st == S::Booting ? QStringLiteral("Starting · %1").arg(placeText(m_s))
                     : st == S::Stopping ? QStringLiteral("Stopping")
                                         : placeText(m_s));
    m_stopped->setVisible(exit);
    if (exit) {
        using K = EmulatorController::ExitInfo::Kind;
        m_stopped->setText(m_exit->kind == K::DeviceRemoved ? QStringLiteral("Stopped · GPU device removed")
                           : m_exit->kind == K::Error        ? QStringLiteral("Couldn't start")
                                                             : QStringLiteral("Stopped · exit %1").arg(Format::exitCode(m_exit->exitCode)));
    }

    const bool showPerf = live && st == S::Running && stats.fps.has_value();
    if (showPerf) {
        QString perf = QStringLiteral("%1 fps").arg(qRound(*stats.fps));
        if (stats.frameMs)
            perf += QStringLiteral(" · %1 ms").arg(*stats.frameMs, 0, 'f', 1);
        m_perf->setText(perf);
    }
    m_perfSep->setVisible(showPerf);
    m_perf->setVisible(showPerf);
    const bool showRam = live && st != S::Stopping && stats.ramFreeMb.has_value();
    if (showRam)
        m_ramValue->setText(Format::gigabytesFromMb(*stats.ramFreeMb));
    m_ramSep->setVisible(showRam);
    m_ramLabel->setVisible(showRam);
    m_ramValue->setVisible(showRam);

    const bool playing = live && (st == S::Running || st == S::Paused);
    m_pause->setVisible(playing);
    m_pause->setEnabled(m_s.emulator.canPause());
    if (st == S::Paused) {
        m_pause->setVariant(Button::Variant::Soft);
        m_pause->setText(QStringLiteral("Resume"));
        m_pause->setIconName(QStringLiteral("play"));
        m_pause->setToolTip(QStringLiteral("Resume (F1)"));
    } else {
        m_pause->setVariant(Button::Variant::Ghost);
        m_pause->setText({});
        m_pause->setIconName(QStringLiteral("pause"));
        m_pause->setToolTip(QStringLiteral("Pause (F1)"));
    }
    m_overlay->setVisible(playing);
    m_overlay->setEnabled(m_s.emulator.canOverlay());
    m_overlay->setChecked(m_s.emulator.overlayOn());
    m_full->setVisible(playing);
    m_full->setIconName(QStringLiteral("fullscreen"));
    m_stop->setVisible(live);
    m_stop->setEnabled(st != S::Stopping);
    if (st == S::Booting) {
        m_stop->setVariant(Button::Variant::Ghost);
        m_stop->setText(QStringLiteral("Cancel"));
        m_stop->setIconName(QStringLiteral("close"));
    } else {
        m_stop->setVariant(Button::Variant::DangerGhost);
        m_stop->setText(QStringLiteral("Stop"));
        m_stop->setIconName(QStringLiteral("stop"));
    }
    m_relaunch->setVisible(exit);

    m_host->setVisible(!exit);
    m_exitPanel->setVisible(exit);
    const bool booting = live && st == S::Booting;
    if (booting) {
        const ShaderCacheInfo& cache = m_s.shaderCache.snapshot();
        QString shaders;
        switch (shownShaderState(m_s.settings, g.titleId(), m_s.settings.recentWarmups(), cache)) {
        case TitleShaderState::Ready: shaders = QStringLiteral("Shaders ready"); break;
        case TitleShaderState::NeverWarmed: shaders = QStringLiteral("Shaders not warmed up"); break;
        case TitleShaderState::LastRunIncomplete: shaders = QStringLiteral("Last warm-up didn't finish"); break;
        case TitleShaderState::LastRunFailed: shaders = QStringLiteral("Last warm-up failed"); break;
        case TitleShaderState::ProfileChanged: shaders = QStringLiteral("Run profile changed since the warm-up"); break;
        case TitleShaderState::Unknown: break;
        }
        if (cache.pipelines > 0)
            shaders += (shaders.isEmpty() ? QString() : QStringLiteral(" · ")) + QStringLiteral("%1 pipelines cached").arg(Format::count(cache.pipelines));
        m_boot->set(g.title(), m_s.art(g.titleId(), CoverCache::Kind::Grid), shaders, QFileInfo(m_s.settings.emulatorPath()).fileName(),
                    m_s.emulator.hello() ? double(m_s.emulator.helloAfterMs()) / 1000.0 : -1.0);
    }
    m_boot->setVisible(booting);
    m_pausedPanel->setVisible(live && st == S::Paused);
    if (st == S::Paused)
        m_pausedPanel->setHeld(stats.heldBytes);
    raisePanels();
    if (booting || st == S::Paused)
        m_restack.start();
    else
        m_restack.stop();

    if (m_fullScreen) {
        m_bar->hide();
    } else if (!m_barHidden) {
        m_bar->show();
    }
    if (!m_fullScreen && !exit && m_s.settings.windowGameBarAutoHide() && st == S::Running && !m_autoHide.isActive()) {
        m_away.restart();
        m_autoHide.start();
    } else if (!(m_s.settings.windowGameBarAutoHide() && st == S::Running)) {
        updateAutoHide();
    }
}
