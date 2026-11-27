#include "FirstLaunchPage.h"

#include "app/Services.h"
#include "core/Format.h"
#include "core/HardwareInfo.h"
#include "shell/LaunchErrors.h"
#include "theme/Theme.h"
#include "widgets/Button.h"
#include "widgets/Card.h"
#include "widgets/ChoiceCard.h"
#include "widgets/CoverThumb.h"
#include "widgets/Icon.h"
#include "widgets/InfoNote.h"
#include "widgets/Text.h"
#include "widgets/Toggle.h"

#include <QButtonGroup>
#include <QDir>
#include <QDragEnterEvent>
#include <QFileInfo>
#include <QHBoxLayout>
#include <QMimeData>
#include <QPainter>
#include <QStackedWidget>
#include <QVBoxLayout>

class StepIndicator : public QWidget {
public:
    StepIndicator()
    {
        setFixedHeight(24);
        connect(&Theme::instance(), &Theme::changed, this, qOverload<>(&QWidget::update));
    }
    void setStep(int step)
    {
        m_step = step;
        update();
    }

protected:
    void paintEvent(QPaintEvent*) override
    {
        const Palette& c = Theme::instance().palette();
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        const QStringList names = {QStringLiteral("Console"), QStringLiteral("Network"), QStringLiteral("Game folders")};
        qreal x = 0;
        for (int i = 0; i < names.size(); ++i) {
            const bool current = i == m_step;
            const bool done = i < m_step;
            const QRectF dot(x + 1, 1, 22, 22);
            if (done) {
                p.setPen(Qt::NoPen);
                p.setBrush(c.accent);
                p.drawEllipse(dot);
                Icon::paint(p, dot, QStringLiteral("check"), c.onAccent, 12);
            } else {
                p.setPen(QPen(current ? c.accent : c.borderStrong, current ? 1.5 : 1));
                p.setBrush(c.surface);
                p.drawEllipse(dot.adjusted(0.5, 0.5, -0.5, -0.5));
                p.setFont(Theme::font(Metrics::TextXs, 500));
                p.setPen(current ? c.accentText : c.text3);
                p.drawText(dot, Qt::AlignCenter, QString::number(i + 1));
            }
            x += 30;
            const QFont f = Theme::font(Metrics::TextSm, current ? 600 : 500);
            p.setFont(f);
            p.setPen(current || done ? c.text : c.text3);
            const int w = QFontMetrics(f).horizontalAdvance(names[i]);
            p.drawText(QRectF(x, 0, w, 24), Qt::AlignLeft | Qt::AlignVCenter, names[i]);
            x += w + 12;
            if (i + 1 < names.size()) {
                p.fillRect(QRectF(x, 12, 30, 1), c.borderStrong);
                x += 42;
            }
        }
    }

private:
    int m_step = 0;
};

namespace {

QWidget* stepPage(int index, const QString& question, const QString& body, QVBoxLayout** content)
{
    auto* page = new QWidget;
    auto* v = new QVBoxLayout(page);
    v->setContentsMargins(0, 0, 0, 0);
    v->setSpacing(0);
    v->addWidget(new Label(QStringLiteral("Step %1 of 3").arg(index + 1), Label::Style::CapsMedium, Theme::Role::Text3));
    v->addSpacing(8);
    v->addWidget(new Label(question, Label::Style::Display));
    v->addSpacing(10);
    auto* b = new Label(body, Label::Style::Body, Theme::Role::Text2);
    b->setWordWrap(true);
    v->addWidget(b);
    v->addSpacing(28);
    auto* c = new QVBoxLayout;
    c->setSpacing(12);
    v->addLayout(c);
    v->addStretch(1);
    *content = c;
    return page;
}

QHBoxLayout* factRow(const QString& label, Label* value)
{
    auto* h = new QHBoxLayout;
    h->setContentsMargins(0, 0, 0, 0);
    h->addWidget(new Label(label, Label::Style::Small, Theme::Role::Text));
    h->addStretch(1);
    h->addWidget(value);
    return h;
}

// The dashed drop zone of the empty Game folders step.
class DropZone : public QWidget {
public:
    DropZone()
    {
        setFixedHeight(200);
        connect(&Theme::instance(), &Theme::changed, this, qOverload<>(&QWidget::update));
    }
    bool hot = false;

protected:
    void paintEvent(QPaintEvent*) override
    {
        const Palette& c = Theme::instance().palette();
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        QPen pen(hot ? c.accent : c.borderStrong, 1.5, Qt::DashLine);
        pen.setDashPattern({3, 3});
        p.setPen(pen);
        p.setBrush(hot ? c.accentSoft : c.surface);
        p.drawRoundedRect(QRectF(rect()).adjusted(0.75, 0.75, -0.75, -0.75), 12, 12);
    }
};

// 40 px accent-soft tile with the folder icon.
class FolderTile : public QWidget {
public:
    FolderTile()
    {
        setFixedSize(40, 40);
        connect(&Theme::instance(), &Theme::changed, this, qOverload<>(&QWidget::update));
    }

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        p.setPen(Qt::NoPen);
        p.setBrush(Theme::color(Theme::Role::AccentSoft));
        p.drawRoundedRect(QRectF(rect()), 10, 10);
        Icon::paint(p, QRectF(rect()), QStringLiteral("folder"), Theme::color(Theme::Role::AccentText), 16);
    }
};

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

bool under(const QString& path, const QString& root)
{
    const QString p = QDir::cleanPath(QFileInfo(path).absoluteFilePath());
    const QString r = QDir::cleanPath(QFileInfo(root).absoluteFilePath());
    return p == r || p.startsWith(r + QLatin1Char('/'));
}

}

FirstLaunchPage::FirstLaunchPage(Services& services, QWidget* parent)
    : QWidget(parent)
    , m_s(services)
    , m_steps(new StepIndicator)
    , m_stack(new QStackedWidget)
    , m_version(new Label(QStringLiteral("XSEmu Launcher %1").arg(QStringLiteral(XSEMU_VERSION)), Label::Style::Small, Theme::Role::Text3))
    , m_back(new Button(QStringLiteral("Back"), Button::Variant::Ghost, Button::Size::Large, QStringLiteral("back")))
    , m_skip(new Button(QStringLiteral("Skip for now"), Button::Variant::Ghost, Button::Size::Large))
    , m_next(new Button(QStringLiteral("Continue"), Button::Variant::Primary, Button::Size::Large))
{
    setAcceptDrops(true);
    auto* v = new QVBoxLayout(this);
    v->setContentsMargins(0, 0, 0, 0);
    v->setSpacing(0);

    auto* column = new QWidget;
    column->setFixedWidth(640);
    auto* cv = new QVBoxLayout(column);
    cv->setContentsMargins(0, 56, 0, 24);
    cv->setSpacing(0);
    auto* brand = new QHBoxLayout;
    brand->setSpacing(10);
    brand->addWidget(new BrandMark(28));
    brand->addWidget(new Label(QStringLiteral("XSEmu Launcher"), Label::Style::SectionTitle));
    brand->addStretch(1);
    cv->addLayout(brand);
    cv->addSpacing(34);
    cv->addWidget(m_steps);
    cv->addSpacing(40);
    m_stack->addWidget(buildConsole());
    m_stack->addWidget(buildNetwork());
    m_stack->addWidget(buildFolders());
    cv->addWidget(m_stack, 1);

    auto* center = new QHBoxLayout;
    center->addStretch(1);
    center->addWidget(column);
    center->addStretch(1);
    v->addLayout(center, 1);

    auto* footer = new QWidget;
    footer->setFixedHeight(68);
    auto* fh = new QHBoxLayout(footer);
    fh->setContentsMargins(0, 1, 0, 0);
    auto* fcol = new QWidget;
    fcol->setFixedWidth(640);
    auto* fch = new QHBoxLayout(fcol);
    fch->setContentsMargins(0, 0, 0, 0);
    fch->setSpacing(8);
    fch->addWidget(m_version);
    fch->addWidget(m_back);
    fch->addStretch(1);
    fch->addWidget(m_skip);
    fch->addWidget(m_next);
    fh->addStretch(1);
    fh->addWidget(fcol);
    fh->addStretch(1);
    v->addWidget(footer);

    connect(m_back, &Button::clicked, this, [this] { setStep(m_step - 1); });
    connect(m_skip, &Button::clicked, this, &FirstLaunchPage::finish);
    connect(m_next, &Button::clicked, this, [this] {
        if (m_step < 2)
            setStep(m_step + 1);
        else
            finish();
    });
    connect(&m_s.settings, &AppSettings::changed, this, [this](const QString& key) {
        if (key == QLatin1String(AppSettings::Key::GameFolders))
            refreshFolders();
    });
    connect(&m_s.library, &Library::scanStarted, this, &FirstLaunchPage::refreshFolders);
    connect(&m_s.library, &Library::gamesChanged, this, &FirstLaunchPage::refreshFolders);
    connect(&m_s.covers, &CoverArtService::artChanged, this, &FirstLaunchPage::refreshFolders);
    connect(&Theme::instance(), &Theme::changed, this, qOverload<>(&QWidget::update));
    setStep(0);
}

void FirstLaunchPage::setStep(int step)
{
    m_step = qBound(0, step, 2);
    m_steps->setStep(m_step);
    m_stack->setCurrentIndex(m_step);
    m_version->setVisible(m_step == 0);
    m_back->setVisible(m_step > 0);
    m_next->setText(m_step == 2 ? QStringLiteral("Open library") : QStringLiteral("Continue"));
    refreshFolders();
}

void FirstLaunchPage::finish()
{
    m_s.settings.setDefaultDeviceType(m_seriesS->isChecked() ? QStringLiteral("lockhart") : QStringLiteral("anaconda"));
    m_s.settings.setOffline(m_offline->isChecked());
    m_s.settings.setFirstLaunchDone(true);
    emit finished();
}

QWidget* FirstLaunchPage::buildConsole()
{
    QVBoxLayout* c = nullptr;
    QWidget* page = stepPage(0, QStringLiteral("Which console should XSEmu Core be?"),
                             QStringLiteral("Games pick their memory budget and resolution from the console model. You can change this later in Settings \u2192 Console."), &c);
    m_seriesS = new ChoiceCard(QStringLiteral("Series S"), QStringLiteral("8 GB title memory · lower resolution targets"),
                               QStringLiteral("Runs on most gaming PCs. XSEmu Core only keeps 1 GB of RAM free while a game runs."), QStringLiteral("Recommended"));
    auto* seriesX = new ChoiceCard(QStringLiteral("Series X"), QStringLiteral("13.5 GB title memory · higher targets"),
                                   QStringLiteral("Needs much more RAM and VRAM. Pick it only on a 64 GB PC."));
    auto* group = new QButtonGroup(page);
    group->addButton(m_seriesS);
    group->addButton(seriesX);
    (m_s.settings.defaultDeviceType() == QLatin1String("anaconda") ? seriesX : m_seriesS)->setChecked(true);
    auto* row = new QHBoxLayout;
    row->setSpacing(12);
    row->addWidget(m_seriesS);
    row->addWidget(seriesX);
    c->addLayout(row);

    auto* note = new InfoNote({});
    c->addWidget(note);
    // the hardware is read in the background when the launcher starts; the note fills in when it lands
    const auto showHardware = [this, note] {
        if (!m_s.hardware.ready()) {
            note->setText(QStringLiteral("Series S is the safe choice."));
            return;
        }
        const HardwareInfo& hw = m_s.hardware.info();
        const qint64 ramGb = qint64((hw.totalRamBytes + (512ULL << 20)) >> 30);
        QString text = QStringLiteral("This PC has %1 GB of RAM").arg(ramGb);
        if (hw.gpuName.isEmpty())
            text += QStringLiteral(", and XSEmu Core couldn't read its GPU.");
        else if (hw.dedicatedVramBytes > 0)
            text += QStringLiteral(" and %1 with %2.").arg(hw.gpuName, Format::gigabytes(qint64(hw.dedicatedVramBytes), 0));
        else
            text += QStringLiteral(" and %1.").arg(hw.gpuName);
        text += ramGb >= 64 ? QStringLiteral(" Either console fits; Series S is the safe choice.") : QStringLiteral(" Series S is the safe choice.");
        note->setText(text);
    };
    showHardware();
    connect(&m_s.hardware, &HardwareMonitor::changed, note, showHardware);
    return page;
}

QWidget* FirstLaunchPage::buildNetwork()
{
    QVBoxLayout* c = nullptr;
    QWidget* page = stepPage(1, QStringLiteral("Keep games offline?"),
                             QStringLiteral("Offline is how XSEmu Core is tested. Games behave as they would on a console with no internet."), &c);
    m_offlineCard = new Card(Card::Kind::Surface, 12);
    auto* oh = new QHBoxLayout(m_offlineCard);
    oh->setContentsMargins(18, 16, 18, 16);
    oh->setSpacing(16);
    auto* ov = new QVBoxLayout;
    ov->setSpacing(4);
    auto* titleRow = new QHBoxLayout;
    titleRow->setSpacing(8);
    auto* title = new Label(QStringLiteral("Offline mode"), Label::Style::Title);
    title->setFont(Theme::font(15, 600));
    titleRow->addWidget(title);
    auto* tag = new Label(QStringLiteral("Recommended"), Label::Style::Caps, Theme::Role::AccentText);
    titleRow->addWidget(tag);
    titleRow->addStretch(1);
    ov->addLayout(titleRow);
    auto* desc = new Label(QStringLiteral("Blocks all network access from the game (XCL_NETWORK=0). Matchmaking, cloud saves and telemetry fail the way they do offline."),
                           Label::Style::Small, Theme::Role::Text2);
    desc->setWordWrap(true);
    ov->addWidget(desc);
    oh->addLayout(ov, 1);
    m_offline = new Toggle;
    m_offline->setChecked(m_s.settings.offline());
    oh->addWidget(m_offline, 0, Qt::AlignVCenter);
    c->addWidget(m_offlineCard);

    auto* sees = new Card(Card::Kind::Surface, 12);
    auto* sv = new QVBoxLayout(sees);
    sv->setContentsMargins(18, 16, 18, 16);
    sv->setSpacing(10);
    sv->addWidget(new Label(QStringLiteral("What the game sees"), Label::Style::CapsMedium, Theme::Role::Text3));
    sv->addLayout(factRow(QStringLiteral("Store context"), new Label(QStringLiteral("Opens normally"), Label::Style::SmallMedium, Theme::Role::Success)));
    sv->addLayout(factRow(QStringLiteral("Store and licence queries"), new Label(QStringLiteral("Network error"), Label::Style::SmallMedium, Theme::Role::Text2)));
    m_signIn = new Label({}, Label::Style::SmallMedium, Theme::Role::Text2);
    sv->addLayout(factRow(QStringLiteral("Xbox services sign-in"), m_signIn));
    c->addWidget(sees);
    c->addWidget(new InfoNote(QStringLiteral("XSEmu Core never reports a licence it doesn't have. A game that needs a live licence check shows its own offline message.")));
    connect(m_offline, &Toggle::toggled, this, &FirstLaunchPage::refreshNetwork);
    refreshNetwork();
    return page;
}

void FirstLaunchPage::refreshNetwork()
{
    const bool offline = m_offline->isChecked();
    m_offlineCard->setSelected(offline);
    m_signIn->setText(offline ? QStringLiteral("Offline") : QStringLiteral("Not available"));
}

QWidget* FirstLaunchPage::buildFolders()
{
    QVBoxLayout* c = nullptr;
    QWidget* page = stepPage(2, QStringLiteral("Where are your games?"),
                             QStringLiteral("Pick a folder of extracted titles. XSEmu Launcher looks for games with a MicrosoftGame.config up to three folders deep."), &c);

    // empty: the drop zone
    m_dropZone = new DropZone;
    auto* dz = new QVBoxLayout(m_dropZone);
    dz->setContentsMargins(0, 0, 0, 0);
    dz->setSpacing(12);
    dz->addStretch(1);
    dz->addWidget(new FolderTile, 0, Qt::AlignHCenter);
    dz->addWidget(new Label(QStringLiteral("Drop a folder here"), Label::Style::Title), 0, Qt::AlignHCenter);
    auto* choose = new Button(QStringLiteral("Choose folder…"), Button::Variant::Secondary, Button::Size::Small);
    dz->addWidget(choose, 0, Qt::AlignHCenter);
    dz->addStretch(1);
    c->addWidget(m_dropZone);
    m_dropHint = new Label(QStringLiteral("You can add more folders later in Settings \u2192 Game folders."), Label::Style::Small, Theme::Role::Text3);
    c->addWidget(m_dropHint);

    // found: folder rows, the games, add another, the warm-up note
    m_found = new QWidget;
    auto* fv = new QVBoxLayout(m_found);
    fv->setContentsMargins(0, 0, 0, 0);
    fv->setSpacing(12);
    m_folderRows = new QVBoxLayout;
    m_folderRows->setSpacing(8);
    fv->addLayout(m_folderRows);
    m_gamesCard = new Card(Card::Kind::Surface, 12);
    new QHBoxLayout(m_gamesCard);
    fv->addWidget(m_gamesCard);
    auto* another = new Button(QStringLiteral("Add another folder…"), Button::Variant::Secondary, Button::Size::Small, QStringLiteral("plus"));
    fv->addWidget(another, 0, Qt::AlignLeft);
    m_warmNote = new InfoNote;
    fv->addWidget(m_warmNote);
    c->addWidget(m_found);

    connect(choose, &Button::clicked, this, &FirstLaunchPage::addGamesRequested);
    connect(another, &Button::clicked, this, &FirstLaunchPage::addGamesRequested);
    return page;
}

void FirstLaunchPage::refreshFolders()
{
    if (!m_folderRows)
        return;
    const QStringList folders = m_s.settings.gameFolders();
    const GameList& games = m_s.library.games();
    const bool scanning = m_s.library.isScanning();
    m_dropZone->setVisible(folders.isEmpty());
    m_dropHint->setVisible(folders.isEmpty());
    m_found->setVisible(!folders.isEmpty());
    m_skip->setVisible(m_step == 2 && games.isEmpty());
    m_next->setEnabled(m_step < 2 || !games.isEmpty());

    while (QLayoutItem* item = m_folderRows->takeAt(0)) {
        delete item->widget();
        delete item;
    }
    for (const QString& folder : folders) {
        qsizetype n = 0;
        for (const Game& g : games)
            n += under(g.folder, folder) ? 1 : 0;
        auto* row = new Card(Card::Kind::Surface, 12);
        auto* h = new QHBoxLayout(row);
        h->setContentsMargins(14, 12, 10, 12);
        h->setSpacing(12);
        h->addWidget(new IconView(QStringLiteral("folder"), Theme::Role::Text2, 16));
        auto* tv = new QVBoxLayout;
        tv->setSpacing(2);
        auto* path = new Label(QDir::toNativeSeparators(folder), Label::Style::Mono, Theme::Role::Text);
        path->setFont(Theme::mono(13));
        tv->addWidget(path);
        Label* count = nullptr;
        if (scanning)
            count = new Label(QStringLiteral("Scanning…"), Label::Style::Caption, Theme::Role::Text3);
        else if (n == 0)
            count = new Label(QStringLiteral("No games found: nothing here has a MicrosoftGame.config"), Label::Style::Caption, Theme::Role::Warning);
        else
            count = new Label(n == 1 ? QStringLiteral("1 game found") : QStringLiteral("%1 games found").arg(n), Label::Style::Caption, Theme::Role::Success);
        tv->addWidget(count);
        h->addLayout(tv, 1);
        auto* remove = new Button(QStringLiteral("Remove"), Button::Variant::Ghost, Button::Size::Tiny);
        connect(remove, &Button::clicked, this, [this, folder] {
            QStringList f = m_s.settings.gameFolders();
            f.removeAll(folder);
            m_s.settings.setGameFolders(f);
        });
        h->addWidget(remove);
        m_folderRows->addWidget(row);
    }

    QLayout* gl = m_gamesCard->layout();
    while (QLayoutItem* item = gl->takeAt(0)) {
        delete item->widget();
        delete item;
    }
    gl->setContentsMargins(16, 16, 16, 16);
    gl->setSpacing(12);
    const int shown = int(qMin<qsizetype>(games.size(), 3));
    for (int i = 0; i < shown; ++i) {
        const Game& g = games[i];
        auto* cell = new QWidget;
        auto* ch = new QHBoxLayout(cell);
        ch->setContentsMargins(0, 0, 0, 0);
        ch->setSpacing(12);
        auto* thumb = new CoverThumb(QSize(44, 66));
        thumb->setCover(m_s.art(g.titleId(), CoverCache::Kind::Grid), g.title());
        ch->addWidget(thumb);
        auto* tv = new QVBoxLayout;
        tv->setSpacing(3);
        tv->addStretch(1);
        auto* name = new Label(g.title(), Label::Style::SmallStrong);
        name->setWordWrap(true);
        tv->addWidget(name);
        tv->addWidget(new Label(g.sizeBytes < 0 ? QStringLiteral("Size not measured") : Format::bytes(g.sizeBytes), Label::Style::Caption, Theme::Role::Text3));
        tv->addStretch(1);
        ch->addLayout(tv, 1);
        static_cast<QHBoxLayout*>(gl)->addWidget(cell, 1);
    }
    if (games.size() > shown)
        static_cast<QHBoxLayout*>(gl)->addWidget(new Label(QStringLiteral("+%1 more").arg(games.size() - shown), Label::Style::SmallMedium, Theme::Role::Text2));
    m_gamesCard->setVisible(!games.isEmpty());

    QStringList cold;
    for (const Game& g : games) {
        if (LaunchErrors::shadersCold(m_s, g))
            cold << g.title();
    }
    QString note;
    if (cold.size() == 1)
        note = QStringLiteral("%1 needs a shader warm-up before it plays smoothly.").arg(cold[0]);
    else if (cold.size() == 2)
        note = QStringLiteral("%1 and %2 need a shader warm-up before they play smoothly.").arg(cold[0], cold[1]);
    else if (cold.size() > 2)
        note = QStringLiteral("These %1 games need a shader warm-up before they play smoothly.").arg(cold.size());
    m_warmNote->setText(note + QStringLiteral(" You can start it from the library."));
    m_warmNote->setVisible(!cold.isEmpty());
}

void FirstLaunchPage::dragEnterEvent(QDragEnterEvent* e)
{
    if (m_step == 2 && !droppedFolders(e->mimeData()).isEmpty())
        e->acceptProposedAction();
}

void FirstLaunchPage::dropEvent(QDropEvent* e)
{
    QStringList folders = m_s.settings.gameFolders();
    for (const QString& f : droppedFolders(e->mimeData())) {
        if (!folders.contains(f))
            folders << f;
    }
    m_s.settings.setGameFolders(folders);
    e->acceptProposedAction();
}

void FirstLaunchPage::paintEvent(QPaintEvent*)
{
    const Palette& c = Theme::instance().palette();
    QPainter p(this);
    p.fillRect(rect(), c.bg);
    const QRect footer(0, height() - 68, width(), 68);
    p.fillRect(footer, c.surface);
    p.fillRect(QRect(0, footer.top(), width(), 1), c.border);
}
