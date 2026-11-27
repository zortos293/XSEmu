#include "Sidebar.h"

#include "app/Services.h"
#include "core/Format.h"
#include "theme/Theme.h"
#include "widgets/Badge.h"
#include "widgets/Icon.h"
#include "widgets/NavItem.h"
#include "widgets/Text.h"

#include <QButtonGroup>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QVBoxLayout>

class NowPlayingCard : public QWidget {
public:
    explicit NowPlayingCard(Sidebar* sidebar)
        : QWidget(sidebar)
        , m_title(new Label({}, Label::Style::SmallStrong))
        , m_status(new Badge({}, Theme::Tone::Warning, Badge::Kind::Plain))
        , m_sidebar(sidebar)
    {
        setCursor(Qt::PointingHandCursor);
        setFixedHeight(60);
        auto* v = new QVBoxLayout(this);
        v->setContentsMargins(48, 10, 10, 10);
        v->setSpacing(2);
        v->addWidget(m_title);
        v->addWidget(m_status, 0, Qt::AlignLeft);
        connect(&Theme::instance(), &Theme::changed, this, qOverload<>(&QWidget::update));
    }
    void set(const QString& title, const QString& status, Theme::Tone tone, const QPixmap& art)
    {
        m_title->setText(QFontMetrics(m_title->font()).elidedText(title, Qt::ElideRight, 140));
        m_status->setText(status);
        m_status->setTone(tone);
        m_art = art;
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override
    {
        const Palette& c = Theme::instance().palette();
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        p.setRenderHint(QPainter::SmoothPixmapTransform);
        const QRectF r = QRectF(rect()).adjusted(0.5, 0.5, -0.5, -0.5);
        p.setPen(QPen(c.border, 1));
        p.setBrush(c.surface);
        p.drawRoundedRect(r, Metrics::RadiusLg, Metrics::RadiusLg);
        const QRectF thumb(8, 8, 30, 44);
        QPainterPath clip;
        clip.addRoundedRect(thumb, 3, 3);
        if (m_art.isNull()) {
            p.fillPath(clip, c.sunken);
            Icon::paint(p, thumb, QStringLiteral("gamepad"), c.text3, 14);
        } else {
            p.save();
            p.setClipPath(clip);
            const QSizeF s = QSizeF(m_art.size()).scaled(thumb.size(), Qt::KeepAspectRatioByExpanding);
            p.drawPixmap(QRectF(thumb.center().x() - s.width() / 2, thumb.center().y() - s.height() / 2, s.width(), s.height()), m_art, QRectF(m_art.rect()));
            p.restore();
        }
    }
    void mouseReleaseEvent(QMouseEvent* e) override
    {
        if (e->button() == Qt::LeftButton && rect().contains(e->position().toPoint()))
            emit m_sidebar->nowPlayingClicked();
    }

private:
    Label* m_title;
    Badge* m_status;
    QPixmap m_art;
    Sidebar* m_sidebar;
};

Sidebar::Sidebar(Services& services, QWidget* parent)
    : QWidget(parent)
    , m_s(services)
    , m_library(new NavItem(QStringLiteral("Library"), QStringLiteral("library")))
    , m_warmup(new NavItem(QStringLiteral("Warm-up"), QStringLiteral("bolt")))
    , m_runners(new NavItem(QStringLiteral("Runners"), QStringLiteral("runner")))
    , m_settings(new NavItem(QStringLiteral("Settings"), QStringLiteral("gear")))
    , m_nowLabel(new Label(QStringLiteral("Now playing"), Label::Style::CapsMedium, Theme::Role::Text3))
    , m_now(new NowPlayingCard(this))
    , m_console(new Badge({}, Theme::Tone::Success, Badge::Kind::Chip))
    , m_network(new Badge({}, Theme::Tone::Neutral, Badge::Kind::Chip))
{
    setFixedWidth(Metrics::Sidebar);
    auto* v = new QVBoxLayout(this);
    v->setContentsMargins(12, 0, 13, 16);
    v->setSpacing(2);

    auto* brand = new QHBoxLayout;
    brand->setContentsMargins(8, 18, 0, 19);
    brand->setSpacing(10);
    brand->addWidget(new BrandMark(22));
    auto* name = new Label(QStringLiteral("XSEmu"), Label::Style::Title);
    name->setFont(Theme::font(Metrics::TextLg, 600, -0.01));
    connect(&Theme::instance(), &Theme::changed, name, [name] { name->setFont(Theme::font(Metrics::TextLg, 600, -0.01)); });
    brand->addWidget(name);
    brand->addStretch(1);
    v->addLayout(brand);

    auto* group = new QButtonGroup(this);
    group->setExclusive(true);
    for (NavItem* item : {m_library, m_warmup, m_runners, m_settings}) {
        group->addButton(item);
        v->addWidget(item);
    }
    connect(m_library, &NavItem::clicked, this, [this] { emit navigate(Page::Library); });
    connect(m_warmup, &NavItem::clicked, this, [this] { emit navigate(Page::Warmup); });
    connect(m_runners, &NavItem::clicked, this, [this] { emit navigate(Page::Runners); });
    connect(m_settings, &NavItem::clicked, this, [this] { emit navigate(Page::Settings); });
    connect(&m_s.runners, &RunnerManager::changed, this, &Sidebar::refresh);

    v->addSpacing(22);
    auto* nowRow = new QHBoxLayout;
    nowRow->setContentsMargins(10, 0, 0, 6);
    nowRow->addWidget(m_nowLabel);
    v->addLayout(nowRow);
    v->addWidget(m_now);
    v->addStretch(1);

    auto* chips = new QHBoxLayout;
    chips->setContentsMargins(4, 0, 0, 0);
    chips->setSpacing(6);
    chips->addWidget(m_console);
    chips->addWidget(m_network);
    chips->addStretch(1);
    m_network->setDotVisible(false);
    v->addLayout(chips);

    m_library->setChecked(true);
    connect(&Theme::instance(), &Theme::changed, this, qOverload<>(&QWidget::update));
    connect(&m_s.library, &Library::gamesChanged, this, &Sidebar::refresh);
    connect(&m_s.gate, &ProcessGate::changed, this, &Sidebar::refresh);
    connect(&m_s.emulator, &EmulatorController::stateChanged, this, &Sidebar::refresh);
    connect(&m_s.emulator, &EmulatorController::statsChanged, this, &Sidebar::refresh);
    connect(&m_s.warmup, &WarmupController::stateChanged, this, &Sidebar::refresh);
    connect(&m_s.warmup, &WarmupController::progressChanged, this, &Sidebar::refresh);
    connect(&m_s.settings, &AppSettings::changed, this, &Sidebar::refresh);
    connect(&m_s.covers, &CoverArtService::artChanged, this, &Sidebar::refresh);
    connect(&m_s.updates, &UpdateService::changed, this, &Sidebar::refresh);
    refresh();
}

void Sidebar::setCurrent(Page page)
{
    switch (page) {
    case Page::Library:
    case Page::GameDetails:
        m_library->setChecked(true);
        break;
    case Page::Runners:
        m_runners->setChecked(true);
        break;
    case Page::Warmup:
        m_warmup->setChecked(true);
        break;
    case Page::Settings:
        m_settings->setChecked(true);
        break;
    }
}

void Sidebar::refresh()
{
    m_library->setTrailingText(QString::number(m_s.library.games().size()));

    if (m_s.warmup.state() == WarmupController::State::PausedForRam) {
        m_warmup->setTrailingText(QStringLiteral("Paused"), Theme::Role::Warning);
    } else if (m_s.warmup.isActive()) {
        const WarmupController::Progress pr = m_s.warmup.progress();
        m_warmup->setTrailingText(QStringLiteral("%1%").arg(qRound(pr.phaseFraction() * 100)), Theme::Role::AccentText);
    } else if (m_s.warmup.state() == WarmupController::State::Failed) {
        m_warmup->setTrailingText(QStringLiteral("Failed"), Theme::Role::Danger);
    } else if (m_s.gate.owner() == ProcessGate::Owner::Game) {
        m_warmup->setTrailingIcon(QStringLiteral("lock"));
    } else {
        m_warmup->setTrailingText({});
    }

    // Runners: the install in progress, else how many curated pins are newer than what is installed
    if (const QString busy = m_s.runners.busyId(); !busy.isEmpty()) {
        const RunnerManager::Progress p = m_s.runners.status(busy).progress;
        m_runners->setTrailingText(p.total > 0 && p.done >= 0 ? QStringLiteral("%1%").arg(qRound(100.0 * double(p.done) / double(p.total))) : QStringLiteral("Installing"),
                                   Theme::Role::AccentText);
    } else {
        int updates = 0;
        for (const RunnerSpec& r : Runners::curated())
            updates += m_s.runners.status(r.id).state == RunnerManager::State::UpdateAvailable ? 1 : 0;
        m_runners->setTrailingText(updates == 0 ? QString() : updates == 1 ? QStringLiteral("1 update") : QStringLiteral("%1 updates").arg(updates), Theme::Role::AccentText);
    }

    // Settings: a launcher or Core update the update check found
    const UpdateService::Status updates = m_s.updates.status();
    m_settings->setTrailingText(updates.updateAvailable() ? QStringLiteral("Update") : QString(), Theme::Role::AccentText);

    const EmulatorController::State st = m_s.emulator.state();
    const bool playing = st != EmulatorController::State::Idle;
    m_nowLabel->setVisible(playing);
    m_now->setVisible(playing);
    if (playing) {
        const Game g = m_s.emulator.game();
        const qint64 held = m_s.emulator.stats().heldBytes;
        QString status;
        Theme::Tone tone = Theme::Tone::Success;
        switch (st) {
        case EmulatorController::State::Booting:
            status = QStringLiteral("Starting");
            tone = Theme::Tone::Accent;
            break;
        case EmulatorController::State::Running:
            status = m_s.emulator.runner().isBuiltIn() ? QStringLiteral("Playing") : QStringLiteral("Running · own window");
            if (!m_s.emulator.runner().isBuiltIn())
                tone = Theme::Tone::Accent;
            break;
        case EmulatorController::State::Paused:
            status = QStringLiteral("Paused");
            tone = Theme::Tone::Warning;
            break;
        case EmulatorController::State::Stopping:
            status = QStringLiteral("Stopping");
            tone = Theme::Tone::Neutral;
            break;
        case EmulatorController::State::Idle:
            break;
        }
        if (held > 0)
            status += QStringLiteral(" · %1 held").arg(Format::gigabytes(held));
        m_now->set(g.title(), status, tone, m_s.art(g.titleId(), CoverCache::Kind::Grid));
    }

    m_console->setText(Services::consoleName(m_s.settings.defaultDeviceType()));
    m_network->setText(m_s.settings.offline() ? QStringLiteral("Offline") : QStringLiteral("Online"));
}

void Sidebar::paintEvent(QPaintEvent*)
{
    QPainter p(this);
    p.fillRect(rect(), Theme::color(Theme::Role::Panel));
    p.fillRect(QRect(width() - 1, 0, 1, height()), Theme::color(Theme::Role::Border));
}
