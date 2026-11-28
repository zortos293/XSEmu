#include "GameDetailsPage.h"

#include "ArtworkCard.h"
#include "ImportSettingsDialog.h"
#include "RunProfileEditor.h"
#include "RunsWithPicker.h"
#include "ShareSettingsDialog.h"

#include "app/Services.h"
#include "core/EmulatorBuild.h"
#include "core/Format.h"
#include "core/GraphicsPreset.h"
#include "core/LaunchConfig.h"
#include "core/Paths.h"
#include "core/ShaderCacheInfo.h"
#include "shell/LaunchErrors.h"
#include "widgets/Badge.h"
#include "widgets/Button.h"
#include "widgets/Card.h"
#include "widgets/CoverThumb.h"
#include "widgets/Icon.h"
#include "widgets/SegmentedControl.h"
#include "widgets/Text.h"
#include "widgets/Toggle.h"

#include <QDesktopServices>
#include <QDir>
#include <QFileInfo>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLinearGradient>
#include <QMenu>
#include <QMouseEvent>
#include <QPainter>
#include <QPainterPath>
#include <QRegularExpression>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>

#include <cmath>

namespace {

QString lowerFirst(const QString& s)
{
    return s.isEmpty() ? s : s.at(0).toLower() + s.mid(1);
}

// "Today 09:02", "Yesterday 21:40", "24 Nov 21:40"
QString when(const QDateTime& at)
{
    if (!at.isValid())
        return QStringLiteral("Never");
    const QDateTime local = at.toLocalTime();
    const qint64 days = local.date().daysTo(QDate::currentDate());
    const QString time = local.toString(QStringLiteral("HH:mm"));
    if (days == 0)
        return QStringLiteral("Today %1").arg(time);
    if (days == 1)
        return QStringLiteral("Yesterday %1").arg(time);
    return QStringLiteral("%1 %2").arg(local.toString(QStringLiteral("d MMM")), time);
}

QString minutes(qint64 s)
{
    return Format::duration(s).remove(QRegularExpression(QStringLiteral(" \\d+ s$")));
}

}

// The action bar: surface with a hairline below.
class ActionBar : public QWidget {
protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.fillRect(rect(), Theme::color(Theme::Role::Surface));
        p.fillRect(QRect(0, height() - 1, width(), 1), Theme::color(Theme::Role::Border));
    }
};

// The 232 px band on top: hero art under a dark scrim, or the no-art fallback; the back link, the
// cover and the title with the play record.
class DetailsHero : public QWidget {
public:
    explicit DetailsHero(QWidget* parent = nullptr)
        : QWidget(parent)
        , m_cover(new CoverThumb(QSize(96, 144), this))
        , m_findArt(new Button(QStringLiteral("Find art"), Button::Variant::Link, Button::Size::Tiny, {}, this))
    {
        setFixedHeight(232);
        setMouseTracking(true);
        connect(&Theme::instance(), &Theme::changed, this, qOverload<>(&QWidget::update));
    }
    Button* findArt() const { return m_findArt; }
    std::function<void()> onBack;

    void set(const QString& title, const QString& meta, const QPixmap& cover, const QPixmap& hero, const QPixmap& logo, bool canFindArt)
    {
        m_title = title;
        m_meta = meta;
        m_hero = hero;
        m_logo = logo;
        m_cover->setCover(cover, title);
        m_findArt->setVisible(hero.isNull() && canFindArt);
        layoutChildren();
        update();
    }

protected:
    void resizeEvent(QResizeEvent*) override { layoutChildren(); }

    void mouseMoveEvent(QMouseEvent* e) override
    {
        const bool over = backRect().contains(e->position());
        if (over != m_backHover) {
            m_backHover = over;
            setCursor(over ? Qt::PointingHandCursor : Qt::ArrowCursor);
            update();
        }
    }
    void leaveEvent(QEvent*) override
    {
        m_backHover = false;
        update();
    }
    void mouseReleaseEvent(QMouseEvent* e) override
    {
        if (e->button() == Qt::LeftButton && backRect().contains(e->position()) && onBack)
            onBack();
    }

    void paintEvent(QPaintEvent*) override
    {
        const Palette& c = Theme::instance().palette();
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        p.setRenderHint(QPainter::SmoothPixmapTransform);
        const QRectF r = rect();
        const bool art = !m_hero.isNull();
        if (art) {
            p.fillRect(r, QColor(0x1A, 0x1A, 0x1A));
            const QSizeF scaled = QSizeF(m_hero.size()).scaled(QSizeF(r.width(), r.width() * 340.0 / 1056.0), Qt::KeepAspectRatioByExpanding);
            p.drawPixmap(QRectF(r.center().x() - scaled.width() / 2, 0, scaled.width(), scaled.height()), m_hero, QRectF(m_hero.rect()));
            QLinearGradient g(r.topLeft(), r.topRight());
            g.setColorAt(0, QColor(0x08, 0x08, 0x0C, 0xD1));
            g.setColorAt(0.55, QColor(0x08, 0x08, 0x0C, 0x59));
            g.setColorAt(1, QColor(0x08, 0x08, 0x0C, 0x00));
            p.fillRect(r, g);
        } else {
            p.fillRect(r, c.sunken);
            p.fillRect(QRectF(0, r.height() - 1, r.width(), 1), c.border);
            p.setFont(Theme::font(12));
            p.setPen(c.text3);
            const qreal right = m_findArt->isVisible() ? m_findArt->x() - 8 : r.width() - 32;
            p.drawText(QRectF(0, 24, right, 16), Qt::AlignRight | Qt::AlignVCenter, QStringLiteral("No hero art"));
        }
        const QColor back = art ? (m_backHover ? QColor(Qt::white) : QColor(255, 255, 255, 0xE6)) : (m_backHover ? c.text : c.text2);
        Icon::paint(p, QRectF(32, 24, 14, 16), QStringLiteral("back"), back, 14);
        p.setFont(Theme::font(13, 500));
        p.setPen(back);
        p.drawText(QRectF(52, 24, 80, 16), Qt::AlignLeft | Qt::AlignVCenter, QStringLiteral("Library"));

        const qreal x = 32 + 96 + 20;
        const qreal textW = r.width() - x - 32;
        const QRectF titleRect(x, r.height() - 24 - 4 - 16 - 6 - 34, textW, 34);
        if (art && !m_logo.isNull()) {
            const QRectF box(x, titleRect.bottom() - 72, qMin<qreal>(textW, 360), 72);
            const QSizeF fit = QSizeF(m_logo.size()).scaled(box.size(), Qt::KeepAspectRatio);
            p.drawPixmap(QRectF(box.left(), box.bottom() - fit.height(), fit.width(), fit.height()), m_logo, QRectF(m_logo.rect()));
        } else {
            const QFont tf = Theme::font(28, 600, -0.02);
            p.setFont(tf);
            p.setPen(art ? QColor(Qt::white) : c.text);
            p.drawText(titleRect, Qt::AlignLeft | Qt::AlignVCenter, QFontMetrics(tf).elidedText(m_title, Qt::ElideRight, int(textW)));
        }
        p.setFont(Theme::font(13));
        p.setPen(art ? QColor(255, 255, 255, 0xBF) : c.text2);
        p.drawText(QRectF(x, r.height() - 24 - 4 - 16, textW, 16), Qt::AlignLeft | Qt::AlignVCenter, m_meta);
    }

private:
    QRectF backRect() const { return QRectF(28, 20, 92, 24); }
    void layoutChildren()
    {
        m_cover->move(32, height() - 24 - 144);
        const QSize s = m_findArt->sizeHint();
        m_findArt->setGeometry(width() - 32 - s.width() + 8, 19, s.width(), s.height());
    }

    CoverThumb* m_cover;
    Button* m_findArt;
    QString m_title;
    QString m_meta;
    QPixmap m_hero;
    QPixmap m_logo;
    bool m_backHover = false;
};

// A details card: caps title with an optional widget on the right, then 36 px label/value rows with
// a hairline above each. The rows sit in one grid, so showing the page lays out one layout per card
// rather than several per row.
class InfoCard : public Card {
public:
    explicit InfoCard(const QString& title)
        : Card(Card::Kind::Surface, 12)
        , m_grid(new QGridLayout)
        , m_trailing(new QHBoxLayout)
    {
        auto* v = new QVBoxLayout(this);
        v->setContentsMargins(18, 16, 18, 6);
        v->setSpacing(0);
        auto* head = new QHBoxLayout;
        head->setContentsMargins(0, 0, 0, 10);
        head->addWidget(new Label(title, Label::Style::CapsMedium, Theme::Role::Text3));
        head->addStretch(1);
        m_trailing->setSpacing(10);
        head->addLayout(m_trailing);
        v->addLayout(head);
        m_grid->setHorizontalSpacing(12);
        m_grid->setVerticalSpacing(0);
        m_grid->setColumnStretch(1, 1);
        v->addLayout(m_grid);
    }

    void setTrailing(const QList<QWidget*>& widgets)
    {
        while (QLayoutItem* item = m_trailing->takeAt(0)) {
            if (item->widget())
                item->widget()->deleteLater();
            delete item;
        }
        for (QWidget* w : widgets)
            m_trailing->addWidget(w);
    }

    // Rows are kept between refreshes. A refresh calls begin(), adds its rows in order and calls end():
    // a row whose label (and key) matches the row already in that place is updated in place, and from
    // the first mismatch on the rows are rebuilt. Building rows is what made switching games slow.
    void begin() { m_next = 0; }
    void end() { truncate(m_next); }

    // True when the custom row in this place has this label and key, so the caller keeps it as it is.
    bool reuse(const QString& label, const QString& key)
    {
        if (m_next >= m_entries.size() || m_entries[m_next].text || m_entries[m_next].label != label || m_entries[m_next].key != key)
            return false;
        ++m_next;
        return true;
    }

    // Returns the row's label, for a tooltip.
    QWidget* addRow(const QString& label, QWidget* value, QWidget* below = nullptr, const QString& key = {}, Qt::Alignment align = {})
    {
        truncate(m_next);
        return append({label, key, nullptr, nullptr, value, below, nullptr, Theme::Role::Text}, align);
    }

    void addText(const QString& label, const QString& value, Theme::Role role = Theme::Role::Text)
    {
        if (m_next < m_entries.size() && m_entries[m_next].text && m_entries[m_next].label == label) {
            Entry& e = m_entries[m_next++];
            e.text->setText(value);
            if (e.role != role) {
                e.text->setRole(role);
                e.role = role;
            }
            return;
        }
        truncate(m_next);
        auto* v = new Label(value, Label::Style::SmallMedium, role);
        v->setWordWrap(true);
        v->setTextInteractionFlags(Qt::TextSelectableByMouse);
        append({label, {}, nullptr, nullptr, v, nullptr, v, role}, {});
    }

private:
    class Hairline : public QWidget {
    public:
        Hairline() { setFixedHeight(1); }

    protected:
        void paintEvent(QPaintEvent*) override
        {
            QPainter p(this);
            p.fillRect(rect(), Theme::color(Theme::Role::Border));
        }
    };

    struct Entry {
        QString label;
        QString key;
        QWidget* line;    // the hairline above the row
        QWidget* caption; // the label column
        QWidget* value;
        QWidget* below;   // a note under the value, in a grid row of its own; may be null
        Label* text;      // the value of a text row; null for a custom row
        Theme::Role role;
        int gridRow = 0;
    };

    QWidget* append(Entry e, Qt::Alignment align)
    {
        e.gridRow = m_entries.isEmpty() ? 0 : m_entries.last().gridRow + (m_entries.last().below ? 2 : 1);
        auto* caption = new Label(e.label, Label::Style::Small, Theme::Role::Text2);
        caption->setFixedWidth(132);
        e.caption = caption;
        e.line = new Hairline;
        m_grid->setRowMinimumHeight(e.gridRow, 36);
        m_grid->addWidget(e.line, e.gridRow, 0, 1, 2, Qt::AlignTop);
        m_grid->addWidget(caption, e.gridRow, 0, Qt::AlignVCenter);
        m_grid->addWidget(e.value, e.gridRow, 1, align | Qt::AlignVCenter);
        if (e.below) {
            e.below->setContentsMargins(0, 0, 0, 8);
            m_grid->setRowMinimumHeight(e.gridRow + 1, 0);
            m_grid->addWidget(e.below, e.gridRow + 1, 1);
        }
        m_entries.append(e);
        ++m_next;
        return caption;
    }

    void truncate(qsizetype n)
    {
        while (m_entries.size() > n) {
            const Entry e = m_entries.takeLast();
            for (QWidget* w : {e.line, e.caption, e.value, e.below}) {
                if (!w)
                    continue;
                m_grid->removeWidget(w);
                w->deleteLater();
            }
            m_grid->setRowMinimumHeight(e.gridRow, 0);
        }
    }

    QGridLayout* m_grid;
    QHBoxLayout* m_trailing;
    QList<Entry> m_entries;
    qsizetype m_next = 0;
};

namespace {

// " XCL_DISPLAY_MODE in the run profile wins over Display mode.": XSEmu reads these variables over xcl.toml.
QString profileWins(const AppSettings& settings, const QString& titleId)
{
    const QHash<QString, QString> rows{{QStringLiteral("XCL_DISPLAY_MODE"), QStringLiteral("Display mode")},
                                       {QStringLiteral("XCL_RAYTRACING"), QStringLiteral("Raytracing")},
                                       {QStringLiteral("XCL_UPSCALER"), QStringLiteral("the quality preset")},
                                       {QStringLiteral("XCL_DLSS_MODE"), QStringLiteral("the quality preset")}};
    QString out;
    for (const auto& e : LaunchConfig::parseProfile(LaunchConfig::runProfile(settings, titleId).join(QLatin1Char('\n'))).entries) {
        if (rows.contains(e.first))
            out += QStringLiteral(" %1 in the run profile wins over %2.").arg(e.first, rows.value(e.first));
    }
    return out;
}

QStringList qualityPresetLabels()
{
    QStringList out{QStringLiteral("Default (blit)")};
    for (const GraphicsPreset::Preset& preset : GraphicsPreset::all())
        out << preset.label;
    return out;
}

QVariantList qualityPresetValues()
{
    QVariantList out{QVariant()};
    for (const GraphicsPreset::Preset& preset : GraphicsPreset::all())
        out << preset.key;
    return out;
}

struct OverrideSpec {
    QString key;            // xcl.toml key, or the launcher's per-game key when launcherSide
    QString label;
    QStringList labels;     // [0] is "Default"
    QVariantList values;    // [0] unused
    QVariant defaultValue;  // what applies without an override
    bool launcherSide = false;
    bool backed = true;
};

bool sameValue(const QVariant& a, const QVariant& b)
{
    if (a.typeId() == QMetaType::QString || b.typeId() == QMetaType::QString)
        return a.toString().compare(b.toString(), Qt::CaseInsensitive) == 0;
    if (a.typeId() == QMetaType::Bool || b.typeId() == QMetaType::Bool)
        return a.toBool() == b.toBool();
    return std::abs(a.toDouble() - b.toDouble()) < 1e-6;
}

}

GameDetailsPage::GameDetailsPage(Services& services, QWidget* parent)
    : QWidget(parent)
    , m_s(services)
    , m_hero(new DetailsHero)
    , m_primary(new Button(QStringLiteral("Play"), Button::Variant::Primary, Button::Size::Large, QStringLiteral("play")))
    , m_secondary(new Button(QStringLiteral("Warm up again"), Button::Variant::Secondary, Button::Size::Large, QStringLiteral("bolt")))
    , m_state(new Badge)
    , m_runsWith(new RunsWithSelector(services))
    , m_openFolder(new Button(QStringLiteral("Open folder"), Button::Variant::Ghost, Button::Size::Small, QStringLiteral("folder")))
    , m_more(new Button({}, Button::Variant::Ghost, Button::Size::Small, QStringLiteral("more")))
    , m_gameCard(new InfoCard(QStringLiteral("Game")))
    , m_cacheCard(new InfoCard(QStringLiteral("Shader cache")))
    , m_runCard(new InfoCard(QStringLiteral("Last run")))
    , m_artCard(new ArtworkCard(services))
    , m_overrides(new InfoCard(QStringLiteral("Per-game overrides")))
    , m_overridesNote(new Label({}, Label::Style::Caption, Theme::Role::Text3))
    , m_profileCard(new InfoCard(QStringLiteral("Run profile")))
    , m_ownProfile(new Toggle)
    , m_profileEditor(new RunProfileEditor(12))
    , m_profileNote(new Label({}, Label::Style::Caption, Theme::Role::Text3))
{
    auto* v = new QVBoxLayout(this);
    v->setContentsMargins(0, 0, 0, 0);
    v->setSpacing(0);
    v->addWidget(m_hero);

    auto* bar = new ActionBar;
    bar->setFixedHeight(68);
    auto* bh = new QHBoxLayout(bar);
    bh->setContentsMargins(32, 0, 32, 1);
    bh->setSpacing(8);
    bh->addWidget(m_primary);
    bh->addWidget(m_secondary);
    bh->addSpacing(12);
    bh->addWidget(m_state);
    bh->addStretch(1);
    bh->addWidget(m_runsWith);
    bh->addSpacing(8);
    bh->addWidget(m_openFolder);
    connect(m_runsWith, &RunsWithSelector::manageRunners, this, &GameDetailsPage::manageRunners);
    m_more->setToolTip(QStringLiteral("More"));
    bh->addWidget(m_more);
    v->addWidget(bar);

    auto* scroll = new QScrollArea;
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    auto* inner = new QWidget;
    inner->setObjectName(QStringLiteral("scrollContents"));
    auto* cols = new QHBoxLayout(inner);
    cols->setContentsMargins(32, 20, 32, 24);
    cols->setSpacing(16);
    auto* left = new QVBoxLayout;
    left->setSpacing(16);
    left->addWidget(m_gameCard);
    left->addWidget(m_runCard);
    left->addWidget(m_artCard);
    left->addStretch(1);
    auto* right = new QVBoxLayout;
    right->setSpacing(16);
    right->addWidget(m_cacheCard);
    m_cacheCard->setTrailing({new Label(QStringLiteral("Shared by all games"), Label::Style::Caption, Theme::Role::Text3)});
    right->addWidget(m_overrides);
    m_overridesNote->setWordWrap(true);
    right->addWidget(m_overridesNote);
    auto* own = new QWidget;
    auto* oh = new QHBoxLayout(own);
    oh->setContentsMargins(0, 0, 0, 0);
    oh->setSpacing(10);
    m_ownProfile->setObjectName(QStringLiteral("ownRunProfile"));
    oh->addWidget(m_ownProfile);
    oh->addWidget(new Label(QStringLiteral("Use a custom run profile for this game"), Label::Style::Small));
    oh->addStretch(1);
    auto* profileBody = new QWidget;
    auto* pv = new QVBoxLayout(profileBody);
    pv->setContentsMargins(0, 0, 0, 0);
    pv->setSpacing(8);
    m_profileNote->setWordWrap(true);
    pv->addWidget(m_profileNote);
    pv->addWidget(m_profileEditor);
    m_profileCard->addRow(QStringLiteral("Custom profile"), own, profileBody);
    right->addWidget(m_profileCard);
    right->addStretch(1);
    cols->addLayout(left, 1);
    cols->addLayout(right, 1);
    scroll->setWidget(inner);
    v->addWidget(scroll, 1);

    m_hero->onBack = [this] { emit back(); };
    connect(m_hero->findArt(), &Button::clicked, this, [this] { m_s.covers.refresh(m_game); });
    connect(m_primary, &Button::clicked, this, [this] {
        if (m_primary->property("warmup").toBool())
            emit warmupRequested(m_game);
        else
            requestPlay();
    });
    connect(m_secondary, &Button::clicked, this, [this] {
        if (m_secondary->property("warmup").toBool())
            emit warmupRequested(m_game);
        else
            emit playRequested(m_game);
    });
    connect(m_openFolder, &Button::clicked, this, [this] { QDesktopServices::openUrl(QUrl::fromLocalFile(m_game.folder)); });
    connect(m_more, &Button::clicked, this, [this] {
        QMenu menu(this);
        const QString toml = Paths::xclToml(m_game.titleId());
        QAction* openToml = menu.addAction(QStringLiteral("Open xcl.toml"));
        openToml->setEnabled(QFileInfo::exists(toml));
        QAction* openLogs = menu.addAction(QStringLiteral("Open logs folder"));
        openLogs->setEnabled(QFileInfo(LaunchErrors::logsFolder(m_game)).isDir());
        QAction* findArt = nullptr;
        if (m_s.covers.canLookUp())
            findArt = menu.addAction(QStringLiteral("Refresh artwork"));
        menu.addSeparator();
        QAction* share = menu.addAction(QStringLiteral("Share settings..."));
        QAction* paste = menu.addAction(QStringLiteral("Paste settings code\tCtrl+Shift+V"));
        QAction* importFile = menu.addAction(QStringLiteral("Import settings file..."));
        QAction* chosen = menu.exec(m_more->mapToGlobal(QPoint(0, m_more->height() + 4)));
        if (chosen == share)
            SettingsShare::openForGame(window(), m_s, m_game.titleId());
        else if (chosen == paste)
            SettingsImport::fromClipboard(window(), m_s, m_game.titleId());
        else if (chosen == importFile)
            SettingsImport::chooseFile(window(), m_s, m_game.titleId());
        if (chosen == openToml)
            QDesktopServices::openUrl(QUrl::fromLocalFile(toml));
        else if (chosen == openLogs)
            QDesktopServices::openUrl(QUrl::fromLocalFile(LaunchErrors::logsFolder(m_game)));
        else if (chosen && chosen == findArt)
            m_s.covers.refresh(m_game);
    });
    connect(&m_s.gate, &ProcessGate::changed, this, &GameDetailsPage::refresh);
    connect(&m_s.runners, &RunnerManager::stateChanged, this, [this](const QString& id) {
        if (m_game.isValid() && id == m_s.emulator.runnerFor(m_game).id)
            refresh();
    });
    connect(&m_s.shaderCache, &ShaderCacheMonitor::changed, this, &GameDetailsPage::refresh);
    connect(&m_s.emulator, &EmulatorController::stateChanged, this, &GameDetailsPage::refresh);
    connect(&m_s.emulator, &EmulatorController::statsChanged, this, &GameDetailsPage::onStats);
    connect(&m_s.emulator, &EmulatorController::exited, this, &GameDetailsPage::onExit);
    connect(&m_s.covers, &CoverArtService::artChanged, this, [this](const QString& id) {
        if (id == m_game.titleId())
            refresh();
    });
    connect(&m_s.settings, &AppSettings::changed, this, [this](const QString& key) {
        if (!m_game.isValid())
            return;
        if (key == QLatin1String(AppSettings::Key::RecentWarmups) || key == QLatin1String(AppSettings::Key::PlayStats)
            || key == QLatin1String(AppSettings::Key::SteamGridDbKey) || key == QLatin1String(AppSettings::Key::RunProfile)
            || key == QLatin1String(AppSettings::Key::GameOverrides))
            refresh();
        else if (key == QLatin1String(AppSettings::Key::DefaultDeviceType) || key == QLatin1String(AppSettings::Key::OverlayDefault))
            refreshOverrides();
    });
    connect(m_ownProfile, &Toggle::toggled, this, [this](bool on) {
        if (on)
            m_s.settings.setGameOverride(m_game.titleId(), QStringLiteral("runProfile"), m_s.settings.runProfile());
        else
            m_s.settings.clearGameOverride(m_game.titleId(), QStringLiteral("runProfile"));
    });
    connect(m_profileEditor, &RunProfileEditor::edited, this, [this](const QStringList& lines) {
        if (m_s.settings.gameRunProfile(m_game.titleId()))
            m_s.settings.setGameOverride(m_game.titleId(), QStringLiteral("runProfile"), lines);
    });
}

void GameDetailsPage::setGame(const Game& game)
{
    m_game = game;
    m_runsWith->setGame(game);
    refresh();
}

void GameDetailsPage::requestPlay()
{
    const Game g = m_game;
    LaunchErrors::confirmPlay(window(), m_s, g, [this, g] { emit playRequested(g); }, [this, g] { emit warmupRequested(g); });
}

void GameDetailsPage::onStats()
{
    const EmulatorController::State st = m_s.emulator.state();
    if (st != EmulatorController::State::Running && st != EmulatorController::State::Paused && st != EmulatorController::State::Booting)
        return;
    const QString id = m_s.emulator.game().titleId();
    if (id != m_runId) {
        m_runId = id;
        m_fpsSum = 0;
        m_fpsSamples = 0;
        m_stalls = 0;
        m_worstMs = 0;
        m_heldPeak = -1;
        m_freeMinMb = -1;
    }
    const EmulatorController::Stats s = m_s.emulator.stats();
    // statsChanged also fires for memory polls and the overlay: sum each protocol sample once
    if (s.sample != 0 && s.sample != m_lastSample) {
        m_lastSample = s.sample;
        if (st == EmulatorController::State::Running && s.fps) {
            m_fpsSum += *s.fps;
            ++m_fpsSamples;
        }
        if (s.stalls)
            m_stalls += *s.stalls;
    }
    if (s.worstMs)
        m_worstMs = qMax(m_worstMs, *s.worstMs);
    if (s.heldBytes > m_heldPeak)
        m_heldPeak = s.heldBytes;
    if (s.ramFreeMb && (m_freeMinMb < 0 || *s.ramFreeMb < m_freeMinMb))
        m_freeMinMb = *s.ramFreeMb;
}

void GameDetailsPage::onExit(const EmulatorController::ExitInfo& info)
{
    using K = EmulatorController::ExitInfo::Kind;
    const QString id = info.game.titleId();
    QString result;
    switch (info.kind) {
    case K::Stopped: result = info.killed ? QStringLiteral("Stopped by you (forced after 5 s)") : QStringLiteral("Stopped by you"); break;
    case K::TitleExit: result = QStringLiteral("The game exited"); break;
    case K::Crash: result = QStringLiteral("Crashed (exit code %1)").arg(Format::exitCode(info.exitCode)); break;
    case K::DeviceRemoved: result = QStringLiteral("GPU device removed"); break;
    case K::Error: result = QStringLiteral("Couldn't start"); break;
    }
    QVariantMap run{
        {QStringLiteral("result"), result},
        {QStringLiteral("at"), info.at.toUTC().toString(Qt::ISODate)},
        {QStringLiteral("seconds"), info.playedSeconds},
    };
    if (id == m_runId) {
        if (m_fpsSamples > 0)
            run.insert(QStringLiteral("fpsAverage"), m_fpsSum / m_fpsSamples);
        if (m_fpsSamples > 0 || m_stalls > 0)
            run.insert(QStringLiteral("stalls"), m_stalls);
        if (m_heldPeak > 0)
            run.insert(QStringLiteral("heldPeakBytes"), m_heldPeak);
        if (m_freeMinMb >= 0)
            run.insert(QStringLiteral("freeMinMb"), m_freeMinMb);
    }
    m_runId.clear();
    m_s.settings.setLastRun(id, run);
}

void GameDetailsPage::refresh()
{
    if (!m_game.isValid())
        return;
    const Game& g = m_game;
    const QString id = g.titleId();
    const qint64 played = m_s.settings.playSeconds(id);
    const QDateTime last = m_s.settings.lastPlayed(id);
    QStringList meta;
    if (!g.publisher.isEmpty())
        meta << g.publisher;
    if (played > 0)
        meta << QStringLiteral("%1 played").arg(Format::playTime(played)) << QStringLiteral("last played %1").arg(lowerFirst(Format::ago(last)));
    else
        meta << QStringLiteral("Not played");
    m_hero->set(g.title(), meta.join(QStringLiteral(" · ")), m_s.art(id, CoverCache::Kind::Grid), m_s.art(id, CoverCache::Kind::Hero),
                m_s.art(id, CoverCache::Kind::Logo), m_s.covers.canLookUp());
    m_artCard->setGame(g);

    // actions and the shader state
    const QList<WarmupRun> recent = m_s.settings.recentWarmups();
    const ShaderCacheInfo& cache = m_s.shaderCache.snapshot();
    const TitleShaderState state = shownShaderState(m_s.settings, id, recent, cache);
    std::optional<WarmupRun> lastWarmup;
    for (const WarmupRun& r : recent) {
        if (r.titleId == id) {
            lastWarmup = r;
            break;
        }
    }
    const bool playingThis = m_s.emulator.isActive() && m_s.emulator.game().titleId() == id;
    const bool cold = LaunchErrors::shadersCold(m_s, g);
    m_primary->setProperty("warmup", cold && !playingThis);
    m_secondary->setProperty("warmup", !cold || playingThis);
    if (playingThis) {
        m_primary->setText(QStringLiteral("Return to game"));
        m_primary->setIconName(QStringLiteral("play"));
    } else if (cold) {
        m_primary->setText(QStringLiteral("Warm up"));
        m_primary->setIconName(QStringLiteral("bolt"));
        m_secondary->setText(QStringLiteral("Play anyway"));
        m_secondary->setIconName(QStringLiteral("play"));
    } else {
        m_primary->setText(QStringLiteral("Play"));
        m_primary->setIconName(QStringLiteral("play"));
        m_secondary->setText(state == TitleShaderState::Ready ? QStringLiteral("Warm up again") : QStringLiteral("Warm up"));
        m_secondary->setIconName(QStringLiteral("bolt"));
    }
    m_secondary->setVisible(!playingThis && m_s.settings.xsemuWarmup());
    const RunnerSpec& runner = m_s.emulator.runnerFor(g);
    if (!runner.isBuiltIn()) {
        m_secondary->setText(QStringLiteral("Pre-cache with %1").arg(runner.name));
        m_secondary->setIconName(QStringLiteral("bolt"));
        m_secondary->setVisible(!playingThis && !runner.precache.isEmpty());
    }
    QString stateText;
    QString cacheState;
    Theme::Tone tone = Theme::Tone::Neutral;
    Theme::Role cacheRole = Theme::Role::Text2;
    switch (state) {
    case TitleShaderState::Ready: stateText = QStringLiteral("Shaders ready"); cacheState = QStringLiteral("Ready"); tone = Theme::Tone::Success; cacheRole = Theme::Role::Success; break;
    case TitleShaderState::NeverWarmed: stateText = QStringLiteral("Never warmed up"); cacheState = QStringLiteral("Empty"); cacheRole = Theme::Role::Text3; break;
    case TitleShaderState::LastRunIncomplete: stateText = QStringLiteral("Warm-up needed"); cacheState = QStringLiteral("Last warm-up stopped part way"); tone = Theme::Tone::Warning; cacheRole = Theme::Role::Warning; break;
    case TitleShaderState::LastRunFailed: stateText = QStringLiteral("Warm-up failed"); cacheState = QStringLiteral("Last warm-up failed"); tone = Theme::Tone::Danger; cacheRole = Theme::Role::Danger; break;
    case TitleShaderState::ProfileChanged: stateText = QStringLiteral("Warm-up needed"); cacheState = QStringLiteral("Run profile changed since the last warm-up"); tone = Theme::Tone::Warning; cacheRole = Theme::Role::Warning; break;
    case TitleShaderState::Unknown:
        if (m_s.settings.xsemuWarmup()) {
            stateText = QStringLiteral("Not warmed up here");
            cacheState = QStringLiteral("Not warmed up from this launcher");
        } else {
            cacheState = QStringLiteral("Built while the game runs");
        }
        break;
    }
    if (!runner.isBuiltIn()) {
        std::optional<WarmupRun> pre;
        for (const WarmupRun& r : recent) {
            if (r.titleId == id && r.runnerId == runner.id) {
                pre = r;
                break;
            }
        }
        const RunnerManager::State installState = m_s.runners.status(runner.id).state;
        if (installState == RunnerManager::State::Installing) {
            stateText = QStringLiteral("Installing %1").arg(runner.name);
            tone = Theme::Tone::Accent;
        } else if (installState == RunnerManager::State::Failed) {
            stateText = QStringLiteral("%1 install failed").arg(runner.name);
            tone = Theme::Tone::Danger;
        } else if (!m_s.runners.isRunnable(runner.id)) {
            stateText = QStringLiteral("%1 not installed").arg(runner.name);
            tone = Theme::Tone::Warning;
        } else if (pre && pre->result == WarmupRun::Result::Finished && pre->total > 0) {
            stateText = QStringLiteral("%1 cache ready").arg(runner.name);
            tone = Theme::Tone::Success;
        } else if (pre && pre->result == WarmupRun::Result::Failed) {
            stateText = QStringLiteral("%1 pre-cache failed").arg(runner.name);
            tone = Theme::Tone::Danger;
        } else {
            stateText = QStringLiteral("%1 keeps its own cache").arg(runner.name);
            tone = Theme::Tone::Neutral;
        }
    }
    // an older emulator build, or another emulator (or job) holding the PC: Play and Warm-up wait, saying why
    // shown from the gate's last look and the emulator's kept answer, so a refresh starts no process;
    // Play and Warm-up check again before they start anything
    QString heldReason;
    bool tooOld = false;
    if (!playingThis && m_s.gate.isFree()) {
        if (const ProcessGate::External& other = m_s.gate.lastExternal(); other.busy()) {
            heldReason = other.reason();
        } else if (runner.isBuiltIn()) {
            const EmulatorBuild build = EmulatorBuild::cached(m_s.settings.emulatorPath());
            tooOld = build.found && !build.capable;
            if (tooOld)
                heldReason = build.problem();
        }
    }
    const bool held = !heldReason.isEmpty();
    for (Button* b : {m_primary, m_secondary}) {
        b->setEnabled(!held);
        b->setToolTip(heldReason);
    }
    if (held) {
        stateText = tooOld ? QStringLiteral("XSEmu Core build too old") : QStringLiteral("Emulator busy");
        tone = tooOld ? Theme::Tone::Danger : Theme::Tone::Warning;
    }
    m_state->setText(stateText);
    m_state->setVisible(!stateText.isEmpty());
    m_state->setToolTip(heldReason);
    m_state->setTone(tone);

    // GAME
    m_gameCard->begin();
    m_gameCard->addText(QStringLiteral("Version"), g.version.isEmpty() ? QStringLiteral("Unknown") : g.version);
    m_gameCard->addText(QStringLiteral("Publisher"), g.publisher.isEmpty() ? QStringLiteral("Unknown") : g.publisher);
    m_gameCard->addText(QStringLiteral("Install size"), g.sizeBytes < 0 ? QStringLiteral("Not measured") : Format::bytes(g.sizeBytes));
    if (!m_gameCard->reuse(QStringLiteral("Folder"), {})) {
        auto* box = new QWidget;
        auto* h = new QHBoxLayout(box);
        h->setContentsMargins(0, 0, 0, 0);
        h->setSpacing(8);
        m_folderPath = new Label({}, Label::Style::Mono, Theme::Role::Text);
        m_folderPath->setFont(Theme::mono(12));
        m_folderPath->setMinimumWidth(40);
        m_folderPath->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Preferred);
        h->addWidget(m_folderPath, 1);
        auto* show = new Button(QStringLiteral("Show"), Button::Variant::Ghost, Button::Size::Tiny);
        connect(show, &Button::clicked, this, [this] { QDesktopServices::openUrl(QUrl::fromLocalFile(m_game.folder)); });
        h->addWidget(show);
        m_gameCard->addRow(QStringLiteral("Folder"), box);
    }
    const QString native = QDir::toNativeSeparators(g.folder);
    m_folderPath->setToolTip(native);
    m_folderPath->setText(QFontMetrics(m_folderPath->font()).elidedText(native, Qt::ElideMiddle, 230));
    m_gameCard->end();

    // SHADER CACHE (machine-wide; per title only what this launcher's warm-ups recorded)
    m_cacheCard->begin();
    m_cacheCard->addText(QStringLiteral("State"), cacheState, cacheRole);
    if (lastWarmup && lastWarmup->total > 0)
        m_cacheCard->addText(QStringLiteral("Pipelines"), QStringLiteral("%1 of %2 built · %3 failed")
                                                              .arg(Format::count(lastWarmup->done - lastWarmup->failed), Format::count(lastWarmup->total), Format::count(lastWarmup->failed)));
    else
        m_cacheCard->addText(QStringLiteral("Pipelines"), cache.pipelines >= 0 ? QStringLiteral("%1 recorded on this PC").arg(Format::count(cache.pipelines)) : QStringLiteral("0 cached"));
    m_cacheCard->addText(QStringLiteral("Last warm-up"), lastWarmup ? QStringLiteral("%1 · %2").arg(when(lastWarmup->finishedAt), lastWarmup->tookS >= 60 ? minutes(qint64(lastWarmup->tookS)) : Format::shortDuration(lastWarmup->tookS))
                                                                   : QStringLiteral("Never"));
    m_cacheCard->end();

    // LAST RUN
    m_runCard->begin();
    const QVariantMap run = m_s.settings.lastRun(id);
    const QString log = QFileInfo::exists(Paths::xclLog(id)) ? Paths::xclLog(id) : QString();
    if (log != m_runLog) {
        m_runLog = log;
        QList<QWidget*> trailing;
        if (!log.isEmpty()) {
            auto* open = new Button(QStringLiteral("Open log"), Button::Variant::Link, Button::Size::Tiny);
            connect(open, &Button::clicked, this, [log] { QDesktopServices::openUrl(QUrl::fromLocalFile(log)); });
            trailing << open;
        }
        m_runCard->setTrailing(trailing);
    }
    if (run.isEmpty()) {
        m_runCard->addText(QStringLiteral("Result"), played > 0 ? QStringLiteral("Not recorded") : QStringLiteral("Not played yet"), Theme::Role::Text3);
    } else {
        m_runCard->addText(QStringLiteral("Result"), run.value(QStringLiteral("result")).toString());
        const QDateTime at = QDateTime::fromString(run.value(QStringLiteral("at")).toString(), Qt::ISODate);
        const qint64 secs = run.value(QStringLiteral("seconds")).toLongLong();
        m_runCard->addText(QStringLiteral("When"), secs > 0 ? QStringLiteral("%1 · ran %2").arg(when(at), minutes(secs)) : when(at));
        if (run.contains(QStringLiteral("fpsAverage"))) {
            const qint64 stalls = run.value(QStringLiteral("stalls")).toLongLong();
            m_runCard->addText(QStringLiteral("Frame rate"), QStringLiteral("%1 fps average · %2")
                                                                 .arg(qRound(run.value(QStringLiteral("fpsAverage")).toDouble()))
                                                                 .arg(stalls == 1 ? QStringLiteral("1 stall over 100 ms") : QStringLiteral("%1 stalls over 100 ms").arg(stalls)));
        }
        QStringList ram;
        if (run.contains(QStringLiteral("heldPeakBytes")))
            ram << QStringLiteral("%1 held").arg(Format::gigabytes(run.value(QStringLiteral("heldPeakBytes")).toLongLong()));
        if (run.contains(QStringLiteral("freeMinMb")))
            ram << QStringLiteral("%1 free at lowest").arg(Format::gigabytesFromMb(run.value(QStringLiteral("freeMinMb")).toLongLong()));
        if (!ram.isEmpty())
            m_runCard->addText(QStringLiteral("Peak RAM"), ram.join(QStringLiteral(" · ")));
    }
    m_runCard->end();

    refreshOverrides();
    refreshRunProfile();
}

void GameDetailsPage::refreshRunProfile()
{
    if (!m_game.isValid())
        return;
    const std::optional<QStringList> own = m_s.settings.gameRunProfile(m_game.titleId());
    {
        const QSignalBlocker quiet(m_ownProfile);
        m_ownProfile->setChecked(own.has_value());
    }
    m_profileEditor->setVisible(own.has_value());
    if (own)
        m_profileEditor->setLines(*own);
    const qsizetype global = m_s.settings.runProfile().size();
    const RunnerSpec& runner = m_s.emulator.runnerFor(m_game);
    const QString applies = own ? QStringLiteral("Play and Warm-up for %1 use this profile instead of the one in Settings.").arg(m_game.title())
                                : QStringLiteral("Play and Warm-up use the profile in Settings, Run profile (%1).")
                                      .arg(global == 1 ? QStringLiteral("1 variable") : QStringLiteral("%1 variables").arg(global));
    m_profileNote->setText(runner.isBuiltIn() ? applies : applies + QStringLiteral(" It applies when the game runs with XSEmu Core; %1 keeps its own environment.").arg(runner.name));
}

void GameDetailsPage::refreshOverrides()
{
    if (!m_game.isValid())
        return;
    const QString id = m_game.titleId();
    const QString deviceDefault = m_s.settings.defaultDeviceType();
    const bool defaultIsS = deviceDefault != QLatin1String("anaconda");
    const QList<OverrideSpec> specs = {
        {QStringLiteral("device_type"), QStringLiteral("Console"),
         {defaultIsS ? QStringLiteral("Default (S)") : QStringLiteral("Default (X)"), QStringLiteral("Series S"), QStringLiteral("Series X")},
         {QVariant(), QStringLiteral("lockhart"), QStringLiteral("anaconda")}, deviceDefault},
        {QStringLiteral("startIn"), QStringLiteral("Start in"), {QStringLiteral("Default"), QStringLiteral("Window"), QStringLiteral("Fullscreen")},
         {QVariant(), QStringLiteral("window"), QStringLiteral("fullscreen")}, {}, true, false},
        {QStringLiteral("overlay"), QStringLiteral("Overlay"), {QStringLiteral("Default"), QStringLiteral("On"), QStringLiteral("Off")},
         {QVariant(), true, false}, m_s.settings.overlayDefault(), true},
        {QStringLiteral("lowLatency"), QStringLiteral("Low latency"), {QStringLiteral("Default"), QStringLiteral("On"), QStringLiteral("Off")},
         {QVariant(), true, false}, {}, true, false},
        {QStringLiteral("display_mode"), QStringLiteral("Display mode"),
         {QStringLiteral("Default"), QStringLiteral("720p"), QStringLiteral("1080p"), QStringLiteral("1440p"), QStringLiteral("4K")},
         {QVariant(), QStringLiteral("1280x720"), QStringLiteral("1920x1080"), QStringLiteral("2560x1440"), QStringLiteral("3840x2160")}, QStringLiteral("auto")},
        {QStringLiteral("raytracing"), QStringLiteral("Raytracing"),
         {QStringLiteral("Default"), QStringLiteral("Tier 1.1"), QStringLiteral("Tier 1.0"), QStringLiteral("Off")},
         {QVariant(), QStringLiteral("tier1_1"), QStringLiteral("tier1_0"), QStringLiteral("off")}, QStringLiteral("auto")},
        {QStringLiteral("qualityPreset"), QStringLiteral("Quality preset"), qualityPresetLabels(), qualityPresetValues(), {}, true},
        {QStringLiteral("keyboard_mouse"), QStringLiteral("Keyboard & mouse"), {QStringLiteral("Default"), QStringLiteral("On"), QStringLiteral("Off")},
         {QVariant(), true, false}, true},
        {QStringLiteral("mouse_sensitivity"), QStringLiteral("Mouse speed"),
         {QStringLiteral("Default"), QStringLiteral("0.5×"), QStringLiteral("1.5×"), QStringLiteral("2×")},
         {QVariant(), 0.5, 1.5, 2.0}, 1.0},
        {QStringLiteral("log_level"), QStringLiteral("Log level"),
         {QStringLiteral("Default"), QStringLiteral("Warn"), QStringLiteral("Debug"), QStringLiteral("Trace")},
         {QVariant(), QStringLiteral("warn"), QStringLiteral("debug"), QStringLiteral("trace")}, QStringLiteral("info")},
    };
    // every per-game choice lives in AppSettings::gameOverrides: LaunchConfig::prepareTitle writes the
    // xcl.toml keys from there into the title's xcl.toml when the game starts
    const QString tomlPath = Paths::xclToml(id);
    const QVariantMap chosen = m_s.settings.gameOverrides(id);

    // the rows are built once; a refresh only sets each control's labels and choice
    const bool build = m_overrideSegs.isEmpty();
    int count = 0;
    for (int i = 0; i < specs.size(); ++i) {
        const OverrideSpec& spec = specs[i];
        QStringList labels = spec.labels;
        int current = 0;
        if (spec.backed) {
            const QVariant value = chosen.value(spec.key);
            if (value.isValid() && !sameValue(value, spec.defaultValue)) {
                current = -1;
                for (int v = 1; v < spec.values.size(); ++v) {
                    if (sameValue(value, spec.values[v]))
                        current = v;
                }
                if (current < 0) {
                    labels << (value.typeId() == QMetaType::Double ? QStringLiteral("%1×").arg(value.toDouble()) : value.toString());
                    current = int(labels.size()) - 1;
                }
                ++count;
            }
        }
        if (build) {
            auto* seg = new SegmentedControl(labels);
            seg->setSizePolicy(QSizePolicy::Fixed, QSizePolicy::Fixed);
            Label* note = nullptr;
            if (!spec.backed) {
                seg->setEnabled(false);
                note = new Label(QStringLiteral("Not in this XSEmu Core build yet"), Label::Style::Caption, Theme::Role::Text3);
            } else if (spec.key == QLatin1String("qualityPreset")) {
                note = new Label({}, Label::Style::Caption, Theme::Role::Text3);
                note->setWordWrap(true);
                m_presetNote = note;
            }
            const QString tip = spec.launcherSide ? QString() : QStringLiteral("%1 in xcl.toml").arg(spec.key);
            m_overrides->addRow(spec.label, seg, note, {}, Qt::AlignLeft)->setToolTip(tip);
            seg->setToolTip(tip);
            m_overrideSegs << seg;
            if (spec.backed) {
                const QVariantList values = spec.values;
                const QString key = spec.key;
                connect(seg, &SegmentedControl::currentChanged, this, [this, key, values](int index) {
                    if (index >= values.size())
                        return;
                    const QString titleId = m_game.titleId();
                    if (key == QLatin1String("qualityPreset")) {
                        // Clear puts the standard title budget back when a preset wrote one;
                        // choosing a preset drops that reset so the preset's number applies.
                        const bool hadPreset = m_s.settings.gameOverrides(titleId).contains(key);
                        if (index == 0) {
                            m_s.settings.clearGameOverride(titleId, key);
                            if (hadPreset)
                                m_s.settings.setGameOverride(titleId, QStringLiteral("title_memory_mb"), 0);
                        } else {
                            m_s.settings.clearGameOverride(titleId, QStringLiteral("title_memory_mb"));
                            m_s.settings.setGameOverride(titleId, key, values[index]);
                        }
                    } else if (index == 0) {
                        m_s.settings.clearGameOverride(titleId, key);
                    } else {
                        m_s.settings.setGameOverride(titleId, key, values[index]);
                    }
                    QTimer::singleShot(0, this, &GameDetailsPage::refreshOverrides);
                });
            }
        }
        SegmentedControl* seg = m_overrideSegs[i];
        const QSignalBlocker quiet(seg);
        if (seg->items() != labels)
            seg->setItems(labels);
        seg->setCurrentIndex(current);
    }

    if (m_presetNote) {
        if (const GraphicsPreset::Preset* preset = GraphicsPreset::find(chosen.value(QStringLiteral("qualityPreset")).toString()))
            m_presetNote->setText(QStringLiteral("XCL_UPSCALER=dlss · XCL_DLSS_MODE=%1 · title memory %2 MB").arg(preset->dlssMode).arg(preset->titleMemoryMb));
        else
            m_presetNote->setText(QStringLiteral("XCL_UPSCALER and XCL_DLSS_MODE unset"));
    }

    if (build) {
        m_overrideCount = new Label({}, Label::Style::Caption, Theme::Role::Text3);
        m_overrideReset = new Button(QStringLiteral("Use defaults"), Button::Variant::Link, Button::Size::Tiny);
        QStringList keys;
        for (const OverrideSpec& spec : specs) {
            if (spec.backed)
                keys << spec.key;
        }
        connect(m_overrideReset, &Button::clicked, this, [this, keys] {
            const QString titleId = m_game.titleId();
            const bool hadPreset = m_s.settings.gameOverrides(titleId).contains(QStringLiteral("qualityPreset"));
            for (const QString& k : keys)
                m_s.settings.clearGameOverride(titleId, k);
            if (hadPreset)
                m_s.settings.setGameOverride(titleId, QStringLiteral("title_memory_mb"), 0);
            QTimer::singleShot(0, this, &GameDetailsPage::refreshOverrides);
        });
        m_overrides->setTrailing({m_overrideCount, m_overrideReset});
    }
    m_overrideCount->setText(count == 0 ? QStringLiteral("None") : count == 1 ? QStringLiteral("1 override") : QStringLiteral("%1 overrides").arg(count));
    m_overrideReset->setVisible(count > 0);

    const bool playingThis = m_s.emulator.isActive() && m_s.emulator.game().titleId() == id;
    const RunnerSpec& runner = m_s.emulator.runnerFor(m_game);
    m_overridesNote->setText(playingThis ? QStringLiteral("Changes apply the next time %1 starts.").arg(m_game.title())
                                         : runner.isBuiltIn()
                                         ? QStringLiteral("These choices apply when the game starts. Console, display mode, raytracing, keyboard and mouse and log level go into %1; the quality preset also sets the launch environment.%2")
                                               .arg(QDir::toNativeSeparators(tomlPath), profileWins(m_s.settings, id))
                                         : QStringLiteral("%1 gets the console when the game starts; the other choices apply when it runs with XSEmu Core.").arg(runner.name));
}
