#include "LibraryPage.h"

#include "PageScaffold.h"
#include "app/Services.h"
#include "core/Format.h"
#include "core/ShaderCacheInfo.h"
#include "shell/LaunchErrors.h"
#include "widgets/Badge.h"
#include "widgets/Button.h"
#include "widgets/Card.h"
#include "widgets/CoverThumb.h"
#include "widgets/CoverTile.h"
#include "widgets/Icon.h"
#include "widgets/PageHeader.h"
#include "widgets/SearchBox.h"
#include "widgets/Text.h"

#include <QAbstractButton>
#include <QDir>
#include <QDragEnterEvent>
#include <QFileInfo>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLinearGradient>
#include <QMenu>
#include <QMimeData>
#include <QPainter>
#include <QPainterPath>
#include <QVBoxLayout>

#include <algorithm>

namespace {

constexpr int kTileGap = 24 - 2 * CoverTile::Ring; // 24 px between the art
constexpr int kInset = CoverTile::Ring;            // the grid's tiles carry their ring; everything else is inset by it

struct TileStatus {
    QString text;
    Theme::Tone tone = Theme::Tone::Neutral;
};

TileStatus shaderStatus(TitleShaderState state)
{
    switch (state) {
    case TitleShaderState::Ready: return {QStringLiteral("Shaders ready"), Theme::Tone::Success};
    case TitleShaderState::NeverWarmed: return {QStringLiteral("Never warmed up"), Theme::Tone::Neutral};
    case TitleShaderState::LastRunIncomplete:
    case TitleShaderState::ProfileChanged: return {QStringLiteral("Warm-up needed"), Theme::Tone::Warning};
    case TitleShaderState::LastRunFailed: return {QStringLiteral("Warm-up failed"), Theme::Tone::Danger};
    case TitleShaderState::Unknown: break;
    }
    return {};
}

QWidget* inset(QWidget* w)
{
    auto* box = new QWidget;
    auto* h = new QHBoxLayout(box);
    h->setContentsMargins(kInset, 0, kInset, 0);
    h->setSpacing(0);
    h->addWidget(w);
    return box;
}

// A word-wrapped label of a fixed width, centred (heightForWidth needs it outside an aligned item).
QHBoxLayout* centred(QWidget* w)
{
    auto* h = new QHBoxLayout;
    h->setContentsMargins(0, 0, 0, 0);
    h->addStretch(1);
    h->addWidget(w);
    h->addStretch(1);
    return h;
}

QString sentenceCase(const QString& s)
{
    return s.isEmpty() ? s : s.at(0).toLower() + s.mid(1);
}

QStringList droppedFolders(const QMimeData* mime)
{
    QStringList out;
    if (!mime || !mime->hasUrls())
        return out;
    for (const QUrl& url : mime->urls()) {
        if (url.isLocalFile() && QFileInfo(url.toLocalFile()).isDir())
            out << QDir::cleanPath(url.toLocalFile());
    }
    return out;
}

}

// "Jump back in" (Library: returning user): 208 px, hero art cropped from the top third under a dark
// left-to-right scrim, or the no-art fallback (sunken card, initials as a watermark).
class JumpBackHero : public QWidget {
public:
    JumpBackHero()
        : m_play(new Button(QStringLiteral("Play"), Button::Variant::Primary, Button::Size::Large, QStringLiteral("play"), this))
        , m_details(new Button(QStringLiteral("Details"), Button::Variant::OnArt, Button::Size::Large, {}, this))
    {
        setFixedHeight(208);
        connect(&Theme::instance(), &Theme::changed, this, qOverload<>(&QWidget::update));
    }
    Button* play() const { return m_play; }
    Button* details() const { return m_details; }

    void set(const QString& title, const QString& meta, const QPixmap& art)
    {
        m_title = title;
        m_meta = meta;
        m_art = art;
        m_details->setVariant(art.isNull() ? Button::Variant::Secondary : Button::Variant::OnArt);
        update();
    }

protected:
    void resizeEvent(QResizeEvent*) override
    {
        const int y = height() - 24 - 36;
        m_play->setGeometry(28, y, m_play->sizeHint().width(), 36);
        m_details->setGeometry(28 + m_play->sizeHint().width() + 8, y, m_details->sizeHint().width(), 36);
    }

    void paintEvent(QPaintEvent*) override
    {
        const Palette& c = Theme::instance().palette();
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        p.setRenderHint(QPainter::SmoothPixmapTransform);
        const QRectF r = rect();
        QPainterPath clip;
        clip.addRoundedRect(r, 14, 14);
        const bool art = !m_art.isNull();
        p.save();
        p.setClipPath(clip);
        if (art) {
            p.fillRect(r, QColor(0x1A, 0x1A, 0x1A));
            const QSizeF scaled = QSizeF(m_art.size()).scaled(QSizeF(r.width(), r.width() * 0.323), Qt::KeepAspectRatioByExpanding);
            const qreal y = -(scaled.height() - r.height()) * 0.3;
            p.drawPixmap(QRectF(r.center().x() - scaled.width() / 2, y, scaled.width(), scaled.height()), m_art, QRectF(m_art.rect()));
            QLinearGradient g(r.topLeft(), r.topRight());
            g.setColorAt(0, QColor(0x0C, 0x0A, 0x14, 0xDB));
            g.setColorAt(0.38, QColor(0x0C, 0x0A, 0x14, 0x8C));
            g.setColorAt(0.70, QColor(0x0C, 0x0A, 0x14, 0x00));
            p.fillRect(r, g);
        } else {
            p.fillRect(r, c.sunken);
            p.setFont(Theme::font(200, 800, -0.04));
            p.setPen(Theme::withAlpha(c.text3, 0.16));
            p.drawText(r.adjusted(0, 0, -40, 40), Qt::AlignRight | Qt::AlignVCenter, CoverThumb::initials(m_title));
        }
        p.restore();
        if (!art) {
            p.setPen(QPen(c.border, 1));
            p.setBrush(Qt::NoBrush);
            p.drawRoundedRect(r.adjusted(0.5, 0.5, -0.5, -0.5), 13.5, 13.5);
        }

        const QColor caps = art ? QColor(255, 255, 255, 0xB8) : c.text3;
        const QColor title = art ? QColor(Qt::white) : c.text;
        const QColor meta = art ? QColor(255, 255, 255, 0xC7) : c.text2;
        const qreal textW = qMin<qreal>(width() * 0.55, 560);
        p.setFont(Theme::font(12, 600, 0.06));
        p.setPen(caps);
        p.drawText(QRectF(28, 24, textW, 16), Qt::AlignLeft | Qt::AlignVCenter, QStringLiteral("JUMP BACK IN"));
        const QFont tf = Theme::font(30, 700, -0.025);
        p.setFont(tf);
        p.setPen(title);
        p.drawText(QRectF(28, 66, textW, 34), Qt::AlignLeft | Qt::AlignVCenter, QFontMetrics(tf).elidedText(m_title, Qt::ElideRight, int(textW)));
        p.setFont(Theme::font(13));
        p.setPen(meta);
        p.drawText(QRectF(28, 106, textW, 16), Qt::AlignLeft | Qt::AlignVCenter, m_meta);
    }

private:
    Button* m_play;
    Button* m_details;
    QString m_title;
    QString m_meta;
    QPixmap m_art;
};

// "Recently played ⌄": the sort menu on the right of ALL GAMES.
class SortButton : public QAbstractButton {
public:
    SortButton()
    {
        setCursor(Qt::PointingHandCursor);
        setFocusPolicy(Qt::TabFocus);
        connect(&Theme::instance(), &Theme::changed, this, qOverload<>(&QWidget::update));
    }
    QSize sizeHint() const override
    {
        return {QFontMetrics(Theme::font(13)).horizontalAdvance(text()) + 6 + 14 + 2, 22};
    }

protected:
    void paintEvent(QPaintEvent*) override
    {
        const Palette& c = Theme::instance().palette();
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        const QFont f = Theme::font(13);
        p.setFont(f);
        p.setPen(underMouse() ? c.text : c.text2);
        const int w = QFontMetrics(f).horizontalAdvance(text());
        p.drawText(QRectF(0, 0, w, height()), Qt::AlignLeft | Qt::AlignVCenter, text());
        Icon::paint(p, QRectF(w + 6, (height() - 14) / 2.0, 14, 14), QStringLiteral("chevron-down"), underMouse() ? c.text : c.text2, 14);
    }
    void enterEvent(QEnterEvent* e) override
    {
        update();
        QAbstractButton::enterEvent(e);
    }
    void leaveEvent(QEvent* e) override
    {
        update();
        QAbstractButton::leaveEvent(e);
    }
};

namespace {

// Library · empty: three cards, the middle one dashed with a plus.
class EmptyArt : public QWidget {
public:
    EmptyArt()
    {
        setFixedSize(240, 180);
        connect(&Theme::instance(), &Theme::changed, this, qOverload<>(&QWidget::update));
    }

protected:
    void paintEvent(QPaintEvent*) override
    {
        const Palette& c = Theme::instance().palette();
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        const QRectF card(-48, -72, 96, 144);
        auto side = [&](qreal dx, qreal angle) {
            p.save();
            p.translate(width() / 2.0 + dx, 92);
            p.rotate(angle);
            p.setPen(Qt::NoPen);
            p.setBrush(QColor(0, 0, 0, Theme::instance().isDark() ? 60 : 14));
            p.drawRoundedRect(card.translated(0, 4), 8, 8);
            p.setBrush(c.surface);
            p.setPen(QPen(c.border, 1));
            p.drawRoundedRect(card, 8, 8);
            p.restore();
        };
        side(-52, -8);
        side(52, 8);
        p.save();
        p.translate(width() / 2.0, 90);
        p.setBrush(c.surface);
        p.setPen(Qt::NoPen);
        p.drawRoundedRect(card, 8, 8);
        QPen dash(c.borderStrong, 1, Qt::DashLine);
        dash.setDashPattern({3, 3});
        p.setPen(dash);
        p.setBrush(c.bg);
        p.drawRoundedRect(card.adjusted(0.5, 0.5, -0.5, -0.5), 8, 8);
        Icon::paint(p, QRectF(-12, -12, 24, 24), QStringLiteral("plus"), c.text3, 20);
        p.restore();
    }
};

// While a folder is dragged over the page.
class DropOverlay : public QWidget {
public:
    explicit DropOverlay(QWidget* parent)
        : QWidget(parent)
    {
        setAttribute(Qt::WA_TransparentForMouseEvents);
        hide();
    }

protected:
    void paintEvent(QPaintEvent*) override
    {
        const Palette& c = Theme::instance().palette();
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        const QRectF r = QRectF(rect()).adjusted(12, 12, -12, -12);
        p.setBrush(Theme::withAlpha(c.accentSoft, 0.85));
        QPen pen(c.accent, 1.5, Qt::DashLine);
        pen.setDashPattern({4, 3});
        p.setPen(pen);
        p.drawRoundedRect(r, 14, 14);
        Icon::paint(p, QRectF(r.center().x() - 12, r.center().y() - 36, 24, 24), QStringLiteral("folder"), c.accentText, 22);
        p.setFont(Theme::font(14, 600));
        p.setPen(c.accentText);
        p.drawText(QRectF(r.left(), r.center().y() - 4, r.width(), 20), Qt::AlignCenter, QStringLiteral("Drop to add this folder to the library"));
    }
};

}

LibraryPage::LibraryPage(Services& services, QWidget* parent)
    : QWidget(parent)
    , m_s(services)
    , m_search(new SearchBox)
    , m_addGames(new Button(QStringLiteral("Add games"), Button::Variant::Secondary, Button::Size::Small, QStringLiteral("plus")))
    , m_hero(new JumpBackHero)
    , m_banner(new Card(Card::Kind::Surface))
    , m_bannerCover(new CoverThumb(QSize(36, 54)))
    , m_bannerTitle(new Label({}, Label::Style::Title))
    , m_bannerText(new Label({}, Label::Style::Small, Theme::Role::Text2))
    , m_bannerHeld(new Badge({}, Theme::Tone::Warning))
    , m_resume(new Button(QStringLiteral("Resume"), Button::Variant::Secondary, Button::Size::Small, QStringLiteral("play")))
    , m_stop(new Button(QStringLiteral("Stop game"), Button::Variant::DangerSecondary, Button::Size::Small, QStringLiteral("stop")))
    , m_sectionLabel(new Label({}, Label::Style::CapsMedium, Theme::Role::Text3))
    , m_sectionHint(new Label(QStringLiteral("Title, publisher and folder name"), Label::Style::Small, Theme::Role::Text3))
    , m_sort(new SortButton)
    , m_gridHost(new QWidget)
    , m_grid(new QGridLayout(m_gridHost))
    , m_addTile(new CoverTile(CoverTile::Kind::AddGames))
    , m_empty(new QWidget)
    , m_noResults(new QWidget)
    , m_noResultsTitle(new Label({}, Label::Style::SectionTitle))
{
    setAcceptDrops(true);
    const PageScaffold page = makePageScaffold(this, QStringLiteral("Library"));
    m_header = page.header;
    m_header->addAction(m_search);
    m_header->addAction(m_addGames);
    page.content->setContentsMargins(Metrics::PagePadding - kInset, Metrics::S6, Metrics::PagePadding - kInset, Metrics::PagePadding);
    page.content->setSpacing(0);

    // the paused / running banner
    auto* bh = new QHBoxLayout(m_banner);
    bh->setContentsMargins(18, 16, 18, 16);
    bh->setSpacing(16);
    bh->addWidget(m_bannerCover, 0, Qt::AlignVCenter);
    auto* bannerText = new QVBoxLayout;
    bannerText->setSpacing(4);
    auto* titleRow = new QHBoxLayout;
    titleRow->setSpacing(10);
    titleRow->addWidget(m_bannerTitle);
    titleRow->addWidget(m_bannerHeld);
    titleRow->addStretch(1);
    bannerText->addLayout(titleRow);
    m_bannerText->setWordWrap(true);
    bannerText->addWidget(m_bannerText);
    bh->addLayout(bannerText, 1);
    bh->addSpacing(24);
    bh->addWidget(m_resume, 0, Qt::AlignVCenter);
    bh->addWidget(m_stop, 0, Qt::AlignVCenter);

    page.content->addWidget(inset(m_hero));
    page.content->addWidget(inset(m_banner));

    // ALL GAMES / RESULTS FOR "..." and the sort menu
    m_sectionRow = new QWidget;
    auto* sr = new QHBoxLayout(m_sectionRow);
    sr->setContentsMargins(kInset, 24, kInset, 10);
    sr->setSpacing(10);
    sr->addWidget(m_sectionLabel);
    sr->addWidget(m_sectionHint);
    sr->addStretch(1);
    sr->addWidget(m_sort);
    page.content->addWidget(m_sectionRow);

    m_grid->setContentsMargins(0, 0, 0, 0);
    m_grid->setHorizontalSpacing(kTileGap);
    m_grid->setVerticalSpacing(24 - 2 * CoverTile::Ring);
    m_grid->setAlignment(Qt::AlignLeft | Qt::AlignTop);
    page.content->addWidget(m_gridHost);

    // Library · empty
    auto* ev = new QVBoxLayout(m_empty);
    ev->setContentsMargins(0, 104, 0, 0);
    ev->setSpacing(0);
    ev->addWidget(new EmptyArt, 0, Qt::AlignHCenter);
    ev->addSpacing(28);
    ev->addWidget(new Label(QStringLiteral("Add your first game"), Label::Style::Display), 0, Qt::AlignHCenter);
    ev->addSpacing(12);
    auto* emptyText = new Label(QStringLiteral("Pick a folder that holds extracted Xbox Series titles. XSEmu Launcher checks each subfolder for a MicrosoftGame.config and adds what it finds."),
                                Label::Style::Body, Theme::Role::Text2);
    emptyText->setAlignment(Qt::AlignHCenter);
    emptyText->setWordWrap(true);
    emptyText->setFixedWidth(440);
    ev->addLayout(centred(emptyText));
    ev->addSpacing(24);
    auto* emptyAdd = new Button(QStringLiteral("Add games folder"), Button::Variant::Primary, Button::Size::Large, QStringLiteral("folder"));
    ev->addWidget(emptyAdd, 0, Qt::AlignHCenter);
    ev->addSpacing(14);
    ev->addWidget(new Label(QStringLiteral("You can also drop a folder anywhere on this window"), Label::Style::Caption, Theme::Role::Text3), 0, Qt::AlignHCenter);
    page.content->addWidget(m_empty);

    // Library · search, no results
    auto* nv = new QVBoxLayout(m_noResults);
    nv->setContentsMargins(0, 190, 0, 0);
    nv->setSpacing(0);
    auto* searchTile = new Card(Card::Kind::Sunken, 12);
    searchTile->setFixedSize(48, 48);
    auto* st = new QVBoxLayout(searchTile);
    st->setContentsMargins(0, 0, 0, 0);
    st->addWidget(new IconView(QStringLiteral("search"), Theme::Role::Text2, 18), 0, Qt::AlignCenter);
    nv->addWidget(searchTile, 0, Qt::AlignHCenter);
    nv->addSpacing(20);
    nv->addWidget(m_noResultsTitle, 0, Qt::AlignHCenter);
    nv->addSpacing(10);
    auto* noText = new Label(QStringLiteral("Search looks at titles, publishers and folder names in your library. The game may be in a folder XSEmu Launcher doesn't scan yet."),
                             Label::Style::Body, Theme::Role::Text2);
    noText->setAlignment(Qt::AlignHCenter);
    noText->setWordWrap(true);
    noText->setFixedWidth(390);
    nv->addLayout(centred(noText));
    nv->addSpacing(20);
    auto* noRow = new QHBoxLayout;
    noRow->setSpacing(6);
    noRow->addStretch(1);
    auto* clear = new Button(QStringLiteral("Clear search"), Button::Variant::Secondary, Button::Size::Small);
    auto* noAdd = new Button(QStringLiteral("Add games folder"), Button::Variant::Secondary, Button::Size::Small, QStringLiteral("folder"));
    noRow->addWidget(clear);
    noRow->addWidget(noAdd);
    noRow->addStretch(1);
    nv->addLayout(noRow);
    page.content->addWidget(m_noResults);
    page.content->addStretch(1);

    m_dropOverlay = new DropOverlay(this);

    m_search->installShortcut(this);
    connect(m_search, &SearchBox::textChanged, this, &LibraryPage::rebuild);
    connect(m_addGames, &Button::clicked, this, &LibraryPage::addGamesRequested);
    connect(m_addTile, &CoverTile::clicked, this, &LibraryPage::addGamesRequested);
    connect(emptyAdd, &Button::clicked, this, &LibraryPage::addGamesRequested);
    connect(noAdd, &Button::clicked, this, &LibraryPage::addGamesRequested);
    connect(clear, &Button::clicked, this, [this] { m_search->setText({}); });
    connect(m_hero->play(), &Button::clicked, this, [this] { requestPlay(m_heroGame); });
    connect(m_hero->details(), &Button::clicked, this, [this] { emit detailsRequested(m_heroGame); });
    connect(m_resume, &Button::clicked, this, [this] {
        if (m_s.emulator.state() == EmulatorController::State::Paused)
            m_s.emulator.resume();
        emit nowPlayingRequested();
    });
    connect(m_stop, &Button::clicked, &m_s.gate, &ProcessGate::stopOwner);
    connect(m_sort, &QAbstractButton::clicked, this, [this] {
        QMenu menu(this);
        const QList<QPair<QString, QString>> options = {
            {QStringLiteral("recent"), QStringLiteral("Recently played")},
            {QStringLiteral("name"), QStringLiteral("Name")},
            {QStringLiteral("size"), QStringLiteral("Install size")},
        };
        for (const auto& o : options) {
            QAction* a = menu.addAction(o.second);
            a->setCheckable(true);
            a->setChecked(m_s.settings.librarySort() == o.first);
            a->setData(o.first);
        }
        if (QAction* chosen = menu.exec(m_sort->mapToGlobal(QPoint(0, m_sort->height() + 4))))
            m_s.settings.setLibrarySort(chosen->data().toString());
    });
    connect(&m_s.library, &Library::gamesChanged, this, &LibraryPage::rebuild);
    connect(&m_s.covers, &CoverArtService::artChanged, this, &LibraryPage::rebuild);
    connect(&m_s.gate, &ProcessGate::changed, this, &LibraryPage::refreshState);
    connect(&m_s.shaderCache, &ShaderCacheMonitor::changed, this, &LibraryPage::refreshState);
    connect(&m_s.emulator, &EmulatorController::stateChanged, this, &LibraryPage::refreshState);
    connect(&m_s.emulator, &EmulatorController::statsChanged, this, &LibraryPage::refreshState);
    connect(&m_s.settings, &AppSettings::changed, this, [this](const QString& key) {
        if (key == QLatin1String(AppSettings::Key::LibrarySort) || key == QLatin1String(AppSettings::Key::PlayStats))
            rebuild();
        else if (key == QLatin1String(AppSettings::Key::RecentWarmups) || key == QLatin1String(AppSettings::Key::RunProfile)
                 || key == QLatin1String(AppSettings::Key::GameOverrides))
            refreshState();
    });
    rebuild();
}

void LibraryPage::requestPlay(const Game& game)
{
    if (!game.isValid())
        return;
    LaunchErrors::confirmPlay(window(), m_s, game, [this, game] { emit playRequested(game); }, [this, game] { emit warmupRequested(game); });
}

GameList LibraryPage::sorted(const GameList& games) const
{
    GameList out = games;
    const QString sort = m_s.settings.librarySort();
    auto byTitle = [](const Game& a, const Game& b) { return QString::localeAwareCompare(a.title(), b.title()) < 0; };
    if (sort == QLatin1String("name")) {
        std::stable_sort(out.begin(), out.end(), byTitle);
    } else if (sort == QLatin1String("size")) {
        std::stable_sort(out.begin(), out.end(), [](const Game& a, const Game& b) { return a.sizeBytes > b.sizeBytes; });
    } else {
        QHash<QString, QDateTime> last;
        for (const Game& g : games)
            last.insert(g.titleId(), m_s.settings.lastPlayed(g.titleId()));
        std::stable_sort(out.begin(), out.end(), [&](const Game& a, const Game& b) {
            const QDateTime la = last.value(a.titleId());
            const QDateTime lb = last.value(b.titleId());
            if (la.isValid() != lb.isValid())
                return la.isValid();
            if (la.isValid() && la != lb)
                return la > lb;
            return QString::localeAwareCompare(a.title(), b.title()) < 0;
        });
    }
    return out;
}

void LibraryPage::rebuild()
{
    const GameList all = sorted(m_s.library.games());
    const QString needle = m_search->text().trimmed();
    m_shown.clear();
    for (const Game& g : all) {
        if (needle.isEmpty() || g.title().contains(needle, Qt::CaseInsensitive) || g.publisher.contains(needle, Qt::CaseInsensitive)
            || QFileInfo(g.folder).fileName().contains(needle, Qt::CaseInsensitive))
            m_shown << g;
    }
    qDeleteAll(m_tiles);
    m_tiles.clear();
    for (const Game& g : std::as_const(m_shown)) {
        auto* tile = new CoverTile(CoverTile::Kind::NoArt, m_gridHost);
        tile->setTitle(g.title());
        if (const QPixmap art = m_s.art(g.titleId(), CoverCache::Kind::Grid); !art.isNull())
            tile->setArt(art);
        connect(tile, &CoverTile::clicked, this, [this, g] { emit detailsRequested(g); });
        connect(tile, &CoverTile::playClicked, this, [this, g] { requestPlay(g); });
        connect(tile, &CoverTile::doubleClicked, this, [this, g] { requestPlay(g); });
        m_tiles << tile;
        m_s.covers.request(g);
    }

    const bool noGames = all.isEmpty();
    const bool searching = !needle.isEmpty();
    const bool noMatches = !noGames && m_shown.isEmpty();
    if (noGames)
        m_header->setSubtitle(QStringLiteral("No games yet"));
    else if (searching)
        m_header->setSubtitle(QStringLiteral("%1 of %2 games").arg(m_shown.size()).arg(all.size()));
    else
        m_header->setSubtitle(all.size() == 1 ? QStringLiteral("1 game") : QStringLiteral("%1 games").arg(all.size()));
    m_search->setVisible(!noGames);
    m_empty->setVisible(noGames);
    m_noResults->setVisible(noMatches);
    m_noResultsTitle->setText(QStringLiteral("No games match \u201C%1\u201D").arg(needle));
    m_sectionRow->setVisible(!noGames && !noMatches);
    m_gridHost->setVisible(!noGames && !noMatches);
    m_sectionLabel->setText(searching ? QStringLiteral("Results for \u201C%1\u201D").arg(needle) : QStringLiteral("All games"));
    m_sectionHint->setVisible(searching);
    m_sort->setVisible(!searching);
    const QString sort = m_s.settings.librarySort();
    m_sort->setText(sort == QLatin1String("name") ? QStringLiteral("Name") : sort == QLatin1String("size") ? QStringLiteral("Install size") : QStringLiteral("Recently played"));
    m_sort->updateGeometry();
    m_addTile->setVisible(!searching);

    // the game played last
    m_heroGame = {};
    QDateTime newest;
    for (const Game& g : all) {
        const QDateTime last = m_s.settings.lastPlayed(g.titleId());
        if (last.isValid() && (!newest.isValid() || last > newest)) {
            newest = last;
            m_heroGame = g;
        }
    }
    m_columns = 0;
    layoutGrid();
    refreshState();
}

void LibraryPage::refreshState()
{
    const ShaderCacheInfo& cache = m_s.shaderCache.snapshot();
    const QList<WarmupRun> recent = m_s.settings.recentWarmups();
    const EmulatorController::State st = m_s.emulator.state();
    const QString playingId = st == EmulatorController::State::Idle ? QString() : m_s.emulator.game().titleId();
    const QString blockedBy = m_s.gate.isFree() ? QString() : m_s.gate.ownerTitle();
    for (int i = 0; i < m_tiles.size(); ++i) {
        const Game& g = m_shown[i];
        CoverTile* tile = m_tiles[i];
        const bool isPlaying = !playingId.isEmpty() && g.titleId() == playingId;
        TileStatus status = shaderStatus(shownShaderState(m_s.settings, g.titleId(), recent, cache));
        if (!g.exeExists)
            status = {QStringLiteral("Game files missing"), Theme::Tone::Danger};
        if (isPlaying) {
            switch (st) {
            case EmulatorController::State::Booting: status = {QStringLiteral("Starting"), Theme::Tone::Accent}; break;
            case EmulatorController::State::Running:
                status = m_s.emulator.runner().isBuiltIn() ? TileStatus{QStringLiteral("Playing · in Now playing"), Theme::Tone::Success}
                                                           : TileStatus{QStringLiteral("Running · own window"), Theme::Tone::Accent};
                break;
            case EmulatorController::State::Paused: status = {QStringLiteral("Paused · in Now playing"), Theme::Tone::Warning}; break;
            case EmulatorController::State::Stopping: status = {QStringLiteral("Stopping"), Theme::Tone::Neutral}; break;
            case EmulatorController::State::Idle: break;
            }
        }
        tile->setStatus(status.text, status.tone);
        tile->setSelected(isPlaying);
        const bool locked = !blockedBy.isEmpty() && !isPlaying;
        const QString reason = m_s.gate.owner() == ProcessGate::Owner::Warmup ? QStringLiteral("Wait for the warm-up")
                                                                                : QStringLiteral("Stop %1 to play").arg(blockedBy);
        tile->setLocked(locked, locked ? reason : QString());
    }

    const bool active = st != EmulatorController::State::Idle;
    const bool searching = !m_search->text().trimmed().isEmpty();
    m_banner->parentWidget()->setVisible(active && !searching);
    m_hero->parentWidget()->setVisible(!active && !searching && m_heroGame.isValid());
    const bool topCard = !searching && (active || m_heroGame.isValid());
    m_sectionRow->layout()->setContentsMargins(kInset, topCard ? 24 : 0, kInset, 10);
    if (m_heroGame.isValid()) {
        QStringList meta;
        const qint64 played = m_s.settings.playSeconds(m_heroGame.titleId());
        if (played > 0)
            meta << QStringLiteral("%1 played").arg(Format::playTime(played));
        meta << QStringLiteral("last played %1").arg(sentenceCase(Format::ago(m_s.settings.lastPlayed(m_heroGame.titleId()))));
        const QString shaders = shaderStatus(shownShaderState(m_s.settings, m_heroGame.titleId(), recent, cache)).text;
        if (!shaders.isEmpty())
            meta << sentenceCase(shaders);
        m_hero->set(m_heroGame.title(), meta.join(QStringLiteral(" \u00B7 ")), m_s.art(m_heroGame.titleId(), CoverCache::Kind::Hero));
    }
    if (active) {
        const Game g = m_s.emulator.game();
        m_bannerCover->setCover(m_s.art(g.titleId(), CoverCache::Kind::Grid), g.title());
        const bool paused = st == EmulatorController::State::Paused;
        m_bannerTitle->setText(paused ? QStringLiteral("%1 is paused").arg(g.title())
                                      : st == EmulatorController::State::Booting ? QStringLiteral("%1 is starting").arg(g.title())
                                      : st == EmulatorController::State::Stopping ? QStringLiteral("%1 is stopping").arg(g.title())
                                                                                   : QStringLiteral("%1 is running").arg(g.title()));
        const qint64 held = m_s.emulator.stats().heldBytes;
        m_bannerHeld->setVisible(held > 0);
        m_bannerHeld->setText(QStringLiteral("%1 held").arg(Format::gigabytes(held)));
        m_bannerText->setText(paused ? QStringLiteral("A paused game keeps its memory. Warm-ups and other games wait until you stop it, because only one XSEmu Core process runs at a time.")
                                     : QStringLiteral("Warm-ups and other games wait until you stop it, because only one XSEmu Core process runs at a time."));
        m_resume->setText(paused ? QStringLiteral("Resume") : QStringLiteral("Back to game"));
        m_resume->setIconName(paused ? QStringLiteral("play") : QString());
        m_resume->setVisible(st != EmulatorController::State::Stopping);
        m_stop->setEnabled(st != EmulatorController::State::Stopping);
        const RunnerSpec& runner = m_s.emulator.runner();
        if (!runner.isBuiltIn()) {
            if (st == EmulatorController::State::Running)
                m_bannerTitle->setText(QStringLiteral("%1 is running in its own window").arg(g.title()));
            m_bannerHeld->setTone(Theme::Tone::Accent);
            m_bannerHeld->setText(runner.name);
            m_bannerHeld->setVisible(true);
            m_bannerText->setText(QStringLiteral("%1 can't embed in XSEmu Core, so the game has a separate window. Pause and the overlay aren't available here; "
                                                 "Stop game ends it. Other games and warm-ups wait until it stops.").arg(runner.name));
            m_resume->setText(QStringLiteral("Show window"));
            m_resume->setIconName(QStringLiteral("play"));
            m_resume->setVisible(st == EmulatorController::State::Running);
        } else {
            m_bannerHeld->setTone(Theme::Tone::Warning);
        }
    }
}

void LibraryPage::resizeEvent(QResizeEvent* e)
{
    QWidget::resizeEvent(e);
    m_dropOverlay->setGeometry(rect().adjusted(0, Metrics::PageHeader, 0, 0));
    layoutGrid();
}

void LibraryPage::layoutGrid()
{
    const int tileW = CoverTile::ArtWidth + 2 * CoverTile::Ring;
    const int avail = width() - 2 * (Metrics::PagePadding - kInset);
    const int columns = qMax(1, (avail + kTileGap) / (tileW + kTileGap));
    if (columns == m_columns)
        return;
    m_columns = columns;
    while (m_grid->count())
        delete m_grid->takeAt(0);
    QList<CoverTile*> all = m_tiles;
    if (m_search->text().trimmed().isEmpty())
        all << m_addTile;
    for (int i = 0; i < all.size(); ++i)
        m_grid->addWidget(all[i], i / columns, i % columns, Qt::AlignTop | Qt::AlignLeft);
}

void LibraryPage::dragEnterEvent(QDragEnterEvent* e)
{
    if (droppedFolders(e->mimeData()).isEmpty())
        return;
    e->acceptProposedAction();
    m_dropOverlay->raise();
    m_dropOverlay->show();
}

void LibraryPage::dragLeaveEvent(QDragLeaveEvent*)
{
    m_dropOverlay->hide();
}

void LibraryPage::dropEvent(QDropEvent* e)
{
    m_dropOverlay->hide();
    const QStringList dropped = droppedFolders(e->mimeData());
    if (dropped.isEmpty())
        return;
    QStringList folders = m_s.settings.gameFolders();
    for (const QString& f : dropped) {
        if (!folders.contains(f))
            folders << f;
    }
    m_s.settings.setGameFolders(folders);
    e->acceptProposedAction();
}
