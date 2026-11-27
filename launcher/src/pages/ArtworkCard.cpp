#include "ArtworkCard.h"

#include "app/Services.h"
#include "widgets/Button.h"
#include "widgets/Dialog.h"
#include "widgets/Text.h"

#include <QFileDialog>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QPainter>
#include <QScrollArea>
#include <QVBoxLayout>

namespace {

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

QString kindLabel(CoverCache::Kind kind)
{
    switch (kind) {
    case CoverCache::Kind::Grid: return QStringLiteral("Cover");
    case CoverCache::Kind::Hero: return QStringLiteral("Banner");
    case CoverCache::Kind::Logo: return QStringLiteral("Logo");
    case CoverCache::Kind::Icon: return QStringLiteral("Icon");
    }
    return {};
}

}

ArtworkCard::ArtworkCard(Services& services, QWidget* parent)
    : Card(Card::Kind::Surface, 12, parent)
    , m_s(services)
    , m_match(new Label({}, Label::Style::SmallMedium))
    , m_change(new Button(QStringLiteral("Change..."), Button::Variant::Ghost, Button::Size::Tiny))
    , m_refresh(new Button(QStringLiteral("Refresh artwork"), Button::Variant::Ghost, Button::Size::Tiny, QStringLiteral("refresh")))
{
    auto* v = new QVBoxLayout(this);
    v->setContentsMargins(18, 16, 18, 6);
    v->setSpacing(0);
    auto* head = new QHBoxLayout;
    head->setContentsMargins(0, 0, 0, 10);
    head->addWidget(new Label(QStringLiteral("Artwork"), Label::Style::CapsMedium, Theme::Role::Text3));
    head->addStretch(1);
    head->addWidget(m_refresh);
    v->addLayout(head);

    auto* grid = new QGridLayout;
    grid->setHorizontalSpacing(12);
    grid->setVerticalSpacing(0);
    grid->setColumnStretch(1, 1);
    v->addLayout(grid);
    int row = 0;
    const auto addRow = [&](const QString& label, QWidget* value, int height) {
        grid->setRowMinimumHeight(row, height);
        grid->addWidget(new Hairline, row, 0, 1, 2, Qt::AlignTop);
        auto* caption = new Label(label, Label::Style::Small, Theme::Role::Text2);
        caption->setFixedWidth(132);
        grid->addWidget(caption, row, 0, Qt::AlignVCenter);
        grid->addWidget(value, row, 1, Qt::AlignVCenter);
        ++row;
    };

    auto* matchBox = new QWidget;
    auto* mh = new QHBoxLayout(matchBox);
    mh->setContentsMargins(0, 0, 0, 0);
    mh->setSpacing(8);
    m_match->setWordWrap(true);
    mh->addWidget(m_match, 1);
    mh->addWidget(m_change);
    addRow(QStringLiteral("SteamGridDB game"), matchBox, 36);

    for (CoverCache::Kind kind : CoverCache::kKinds) {
        Slot& slot = m_slots[size_t(kind)];
        auto* box = new QWidget;
        auto* h = new QHBoxLayout(box);
        h->setContentsMargins(0, 6, 0, 6);
        h->setSpacing(10);
        slot.preview = new QLabel;
        slot.preview->setFixedSize(kind == CoverCache::Kind::Hero || kind == CoverCache::Kind::Logo ? QSize(96, 40) : QSize(40, 40));
        slot.preview->setAlignment(Qt::AlignCenter);
        if (kind == CoverCache::Kind::Logo)
            slot.preview->setStyleSheet(QStringLiteral("background: #1A1A1A; border-radius: 6px;"));
        h->addWidget(slot.preview);
        slot.source = new Label({}, Label::Style::Small, Theme::Role::Text2);
        slot.source->setWordWrap(true);
        h->addWidget(slot.source, 1);
        auto* choose = new Button(QStringLiteral("Choose..."), Button::Variant::Ghost, Button::Size::Tiny);
        slot.reset = new Button(QStringLiteral("Reset"), Button::Variant::Ghost, Button::Size::Tiny);
        h->addWidget(choose);
        h->addWidget(slot.reset);
        addRow(kindLabel(kind), box, 52);
        connect(choose, &Button::clicked, this, [this, kind] {
            const QString file = QFileDialog::getOpenFileName(window(), QStringLiteral("Choose the %1 image").arg(kindLabel(kind).toLower()), {},
                                                              QStringLiteral("Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif)"));
            if (!file.isEmpty() && !m_s.covers.setCustomImage(m_game.titleId(), kind, file))
                m_slots[size_t(kind)].source->setText(QStringLiteral("That file can't be used: pick a PNG, JPEG, WebP, BMP or GIF under 20 MB."));
        });
        connect(slot.reset, &Button::clicked, this, [this, kind] { m_s.covers.clearCustomImage(m_game.titleId(), kind); });
    }

    connect(m_refresh, &Button::clicked, this, [this] { m_s.covers.refresh(m_game); refresh(); });
    connect(m_change, &Button::clicked, this, &ArtworkCard::chooseMatch);
    connect(&m_s.covers, &CoverArtService::artChanged, this, [this](const QString& id) {
        if (id.isEmpty() || id == m_game.titleId())
            refresh();
    });
    connect(&m_s.covers, &CoverArtService::lookupFailed, this, [this](const QString& id) {
        if (id == m_game.titleId())
            refresh();
    });
    connect(&m_s.settings, &AppSettings::changed, this, [this](const QString& key) {
        if (key == QLatin1String(AppSettings::Key::ArtLookups) || key == QLatin1String(AppSettings::Key::SteamGridDbKey))
            refresh();
    });
}

void ArtworkCard::setGame(const Game& game)
{
    m_game = game;
    refresh();
}

void ArtworkCard::refresh()
{
    if (!m_game.isValid())
        return;
    const QString id = m_game.titleId();
    const CoverCache::Entry e = m_s.covers.cache().entry(id);
    const bool can = m_s.covers.canLookUp();
    const QString error = m_s.covers.lastError(id);
    Theme::Role role = Theme::Role::Text;
    QString match;
    if (!can) {
        match = QStringLiteral("Art downloads are off (Settings, Game folders)");
        role = Theme::Role::Text3;
    } else if (m_s.covers.isPending(id)) {
        match = QStringLiteral("Looking it up...");
        role = Theme::Role::Text3;
    } else if (!error.isEmpty()) {
        match = QStringLiteral("Couldn't get art: %1").arg(error);
        role = Theme::Role::Danger;
    } else if (e.gameId > 0) {
        match = e.pinned ? QStringLiteral("%1 (chosen by you)").arg(e.name) : e.name;
    } else if (e.notFound) {
        match = QStringLiteral("Not found on SteamGridDB");
        role = Theme::Role::Text3;
    } else {
        match = QStringLiteral("Not looked up yet");
        role = Theme::Role::Text3;
    }
    m_match->setText(match);
    m_match->setRole(role);
    m_change->setEnabled(can);
    m_refresh->setEnabled(can && !m_s.covers.isPending(id));

    const qreal dpr = devicePixelRatioF();
    for (CoverCache::Kind kind : CoverCache::kKinds) {
        Slot& slot = m_slots[size_t(kind)];
        const QPixmap art = m_s.art(id, kind);
        if (art.isNull()) {
            slot.preview->setPixmap({});
        } else {
            QPixmap scaled = art.scaled((QSizeF(slot.preview->size()) * dpr).toSize(), Qt::KeepAspectRatio, Qt::SmoothTransformation);
            scaled.setDevicePixelRatio(dpr);
            slot.preview->setPixmap(scaled);
        }
        const bool custom = m_s.covers.cache().isCustom(id, kind);
        const QString author = e.authors[size_t(kind)];
        if (custom)
            slot.source->setText(QStringLiteral("Your image"));
        else if (art.isNull())
            slot.source->setText(QStringLiteral("None"));
        else if (author.isEmpty())
            slot.source->setText(QStringLiteral("SteamGridDB"));
        else
            slot.source->setText(QStringLiteral("SteamGridDB, by %1").arg(author));
        slot.source->setRole(art.isNull() ? Theme::Role::Text3 : Theme::Role::Text2);
        slot.reset->setVisible(custom);
    }
}

void ArtworkCard::chooseMatch()
{
    const Game game = m_game;
    auto* d = new Dialog(window(), 520);
    d->setTitle(QStringLiteral("Choose the SteamGridDB game"));
    auto* intro = new Label(QStringLiteral("Pick the game %1 takes its art from. Your choice stays when the art is refreshed.").arg(game.title()),
                            Label::Style::Small, Theme::Role::Text2);
    intro->setWordWrap(true);
    d->body()->addWidget(intro);
    auto* searchRow = new QHBoxLayout;
    searchRow->setSpacing(8);
    auto* term = new QLineEdit(CoverArtService::searchTerm(game));
    term->setProperty("framed", true);
    term->setPlaceholderText(QStringLiteral("Game name"));
    auto* go = new Button(QStringLiteral("Search"), Button::Variant::Secondary, Button::Size::Small, QStringLiteral("search"));
    searchRow->addWidget(term, 1);
    searchRow->addWidget(go);
    d->body()->addSpacing(12);
    d->body()->addLayout(searchRow);
    auto* status = new Label({}, Label::Style::Caption, Theme::Role::Text3);
    status->setWordWrap(true);
    d->body()->addSpacing(8);
    d->body()->addWidget(status);
    auto* scroll = new QScrollArea;
    scroll->setWidgetResizable(true);
    scroll->setFrameShape(QFrame::NoFrame);
    scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    scroll->setFixedHeight(240);
    auto* list = new QWidget;
    auto* results = new QVBoxLayout(list);
    results->setContentsMargins(0, 8, 0, 0);
    results->setSpacing(4);
    results->addStretch(1);
    scroll->setWidget(list);
    d->body()->addWidget(scroll);

    const auto run = [this, d, term, status, results, game] {
        while (results->count() > 1) {
            QLayoutItem* item = results->takeAt(0);
            if (item->widget())
                item->widget()->deleteLater();
            delete item;
        }
        status->setText(QStringLiteral("Searching..."));
        status->setRole(Theme::Role::Text3);
        m_s.covers.search(term->text(), d, [this, d, status, results, game](const QList<CoverArtService::Match>& matches, const QString& error) {
            if (!error.isEmpty()) {
                status->setText(QStringLiteral("The search failed: %1").arg(error));
                status->setRole(Theme::Role::Danger);
                return;
            }
            status->setText(matches.isEmpty() ? QStringLiteral("No games found. Try another name.") : QStringLiteral("%1 found").arg(matches.size()));
            for (const CoverArtService::Match& m : matches.mid(0, 10)) {
                auto* b = new Button(m.verified ? QStringLiteral("%1  (verified)").arg(m.name) : m.name, Button::Variant::Secondary, Button::Size::Small);
                results->insertWidget(results->count() - 1, b);
                connect(b, &Button::clicked, d, [this, d, game, m] {
                    m_s.covers.setMatch(game, m);
                    d->dismiss();
                });
            }
        });
    };
    connect(go, &Button::clicked, d, run);
    connect(term, &QLineEdit::returnPressed, d, run);

    if (m_s.covers.cache().entry(game.titleId()).pinned) {
        auto* automatic = new Button(QStringLiteral("Use the automatic match"), Button::Variant::Ghost, Button::Size::Small);
        d->addButton(automatic, true);
        connect(automatic, &Button::clicked, d, [this, d, game] {
            m_s.covers.clearMatch(game);
            d->dismiss();
        });
    }
    auto* cancel = new Button(QStringLiteral("Cancel"), Button::Variant::Secondary, Button::Size::Small);
    d->addButton(cancel);
    connect(cancel, &Button::clicked, d, &Dialog::dismiss);
    d->open();
    run();
}
