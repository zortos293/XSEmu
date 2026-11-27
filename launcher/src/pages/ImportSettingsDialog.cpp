#include "ImportSettingsDialog.h"

#include "ProfileCard.h"
#include "settings/SettingsKit.h"

#include "app/Services.h"
#include "core/SettingsProfile.h"
#include "widgets/Button.h"
#include "widgets/Dialog.h"
#include "widgets/InfoNote.h"
#include "widgets/SegmentedControl.h"
#include "widgets/Text.h"

#include <QClipboard>
#include <QDragEnterEvent>
#include <QDropEvent>
#include <QFile>
#include <QFileDialog>
#include <QFileInfo>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QMimeData>
#include <QShortcut>
#include <QVBoxLayout>

#include <algorithm>
#include <memory>

namespace {

using SettingsProfile::Line;
using SettingsProfile::Target;

constexpr qint64 kMaxFile = 256 * 1024;

bool dialogOpen(QWidget* window)
{
    for (const Dialog* d : window->findChildren<Dialog*>()) {
        if (d->isVisible())
            return true;
    }
    return false;
}

void showError(QWidget* window, const QString& error)
{
    auto* d = new Dialog(window, 460);
    d->setObjectName(QStringLiteral("importError"));
    d->setIcon(QStringLiteral("warning"), int(Theme::Tone::Danger));
    d->setTitle(QStringLiteral("Can't import these settings"));
    auto* text = new Label(error, Label::Style::Body, Theme::Role::Text2);
    text->setWordWrap(true);
    d->body()->addWidget(text);
    auto* hint = new Label(QStringLiteral("Ask for the code again, or for the .xsemu-settings file. Nothing was changed."), Label::Style::Caption, Theme::Role::Text3);
    hint->setWordWrap(true);
    d->body()->addWidget(hint);
    auto* close = new Button(QStringLiteral("Close"), Button::Variant::Primary, Button::Size::Small);
    QObject::connect(close, &Button::clicked, d, &Dialog::dismiss);
    d->addButton(close);
    d->open();
}

QString keyLabel(const QString& key)
{
    const SettingsProfile::KeySpec* spec = SettingsProfile::find(key);
    return spec ? spec->label : key;
}

QString notesText(const SettingsProfile::Parsed& in)
{
    QStringList lines;
    for (const SettingsProfile::Note& n : in.rejected)
        lines << QStringLiteral("%1: %2.").arg(keyLabel(n.key), n.text);
    if (!in.ignored.isEmpty())
        lines << (in.ignored.size() == 1 ? QStringLiteral("Ignored %1: this launcher doesn't know it.").arg(in.ignored.first())
                                         : QStringLiteral("Ignored %1: this launcher doesn't know them.").arg(in.ignored.join(QStringLiteral(", "))));
    return lines.join(QLatin1Char('\n'));
}

int changes(const QList<Line>& lines)
{
    return int(std::count_if(lines.cbegin(), lines.cend(), [](const Line& l) { return l.kind == Line::Kind::Change; }));
}

void showPreview(QWidget* window, Services& s, const SettingsProfile::Parsed& parsed, const QString& startTitleId)
{
    const auto in = std::make_shared<SettingsProfile::Parsed>(parsed);
    const SettingsProfile::Profile& p = in->profile;
    QList<QPair<QString, QVariant>> games;
    QString titleId = startTitleId;
    for (const Game& g : s.library.games()) {
        games.append({g.title(), g.titleId()});
        if (titleId.isEmpty() && g.titleId() == p.titleId)
            titleId = g.titleId();
    }

    auto* d = new Dialog(window, 600);
    d->setObjectName(QStringLiteral("importSettings"));
    d->setIcon(QStringLiteral("download"), int(Theme::Tone::Accent));
    d->setTitle(p.name.isEmpty() ? QStringLiteral("Import shared settings") : QStringLiteral("Import \u201C%1\u201D").arg(p.name));
    QVBoxLayout* b = d->body();
    b->setSpacing(0);
    QStringList meta;
    if (!p.gpuModel.isEmpty())
        meta << QStringLiteral("Made on %1").arg(p.gpuModel);
    if (!p.launcherVersion.isEmpty())
        meta << QStringLiteral("launcher %1").arg(p.launcherVersion);
    meta << (p.title.isEmpty() ? QStringLiteral("launcher defaults") : QStringLiteral("for %1").arg(p.title));
    b->addWidget(new Label(meta.join(QStringLiteral(" · ")), Label::Style::Caption, Theme::Role::Text3));
    if (!p.note.isEmpty()) {
        b->addSpacing(10);
        auto* note = new Label(QStringLiteral("\u201C%1\u201D").arg(p.note), Label::Style::Small, Theme::Role::Text);
        note->setWordWrap(true);
        b->addWidget(note);
    }
    b->addSpacing(16);

    auto* targetRow = new QHBoxLayout;
    targetRow->setSpacing(10);
    targetRow->addWidget(new Label(QStringLiteral("Compare with"), Label::Style::Small, Theme::Role::Text2));
    auto* target = new SegmentedControl({QStringLiteral("This game"), QStringLiteral("Defaults")});
    target->setObjectName(QStringLiteral("importTarget"));
    targetRow->addWidget(target);
    QComboBox* gamePick = SettingsKit::dropdown(games.isEmpty() ? QList<QPair<QString, QVariant>>{{QStringLiteral("No games yet"), QString()}} : games,
                                                titleId.isEmpty() ? games.value(0).second : QVariant(titleId), 220);
    gamePick->setObjectName(QStringLiteral("importGame"));
    gamePick->setEnabled(!games.isEmpty());
    targetRow->addWidget(gamePick);
    targetRow->addStretch(1);
    b->addLayout(targetRow);
    b->addSpacing(12);
    if (games.isEmpty()) {
        target->setItemEnabled(0, false);
        target->setCurrentIndex(1);
    } else {
        target->setCurrentIndex(titleId.isEmpty() ? 1 : 0);
    }

    auto* card = new ProfileCard(QStringLiteral("What changes"));
    card->setObjectName(QStringLiteral("importDiff"));
    b->addWidget(card);
    auto* same = new Label({}, Label::Style::Caption, Theme::Role::Text3);
    b->addSpacing(6);
    b->addWidget(same);
    const QString notes = notesText(*in);
    if (!notes.isEmpty()) {
        b->addSpacing(12);
        auto* note = new InfoNote(notes, Theme::Tone::Warning, QStringLiteral("warning"));
        note->setObjectName(QStringLiteral("importNotes"));
        b->addWidget(note);
    }

    auto* cancel = new Button(QStringLiteral("Cancel"), Button::Variant::Secondary, Button::Size::Small);
    auto* asDefaults = new Button(QStringLiteral("Apply as defaults"), Button::Variant::Secondary, Button::Size::Small);
    asDefaults->setObjectName(QStringLiteral("applyDefaults"));
    auto* toGame = new Button(QStringLiteral("Apply to this game"), Button::Variant::Primary, Button::Size::Small);
    toGame->setObjectName(QStringLiteral("applyGame"));
    d->addButton(cancel);
    d->addButton(asDefaults);
    d->addButton(toGame);
    QObject::connect(cancel, &Button::clicked, d, &Dialog::dismiss);

    const auto gameId = [gamePick] { return gamePick->currentData().toString(); };
    const auto rebuild = [d, &s, in, target, gamePick, gameId, card, same, asDefaults, toGame] {
        const bool game = target->currentIndex() == 0;
        gamePick->setVisible(game);
        const QList<Line> forGame = gameId().isEmpty() ? QList<Line>() : SettingsProfile::preview(*in, s.settings, Target::Game, gameId());
        const QList<Line> forDefaults = SettingsProfile::preview(*in, s.settings, Target::Defaults);
        const QList<Line>& lines = game ? forGame : forDefaults;
        toGame->setEnabled(changes(forGame) > 0);
        asDefaults->setEnabled(changes(forDefaults) > 0);
        toGame->setVariant(game ? Button::Variant::Primary : Button::Variant::Secondary);
        asDefaults->setVariant(game ? Button::Variant::Secondary : Button::Variant::Primary);

        card->clear();
        int changed = 0, refused = 0, matching = 0;
        for (const Line& l : lines) {
            QWidget* action = nullptr;
            QString caption;
            if (l.key == QLatin1String("runner")) {
                const QString id = l.value.toString();
                const RunnerSpec* spec = Runners::find(id);
                const RunnerManager::Status st = s.runners.status(id);
                if (!s.runners.isRunnable(id)) {
                    if (st.state == RunnerManager::State::Installing) {
                        caption = QStringLiteral("Installing %1...").arg(spec->name);
                    } else if (st.state == RunnerManager::State::Unavailable) {
                        caption = QStringLiteral("%1 can't be installed on this PC.").arg(spec->name);
                    } else {
                        auto* install = new Button(QStringLiteral("Install %1").arg(spec->name), Button::Variant::Secondary, Button::Size::Tiny, QStringLiteral("download"));
                        install->setObjectName(QStringLiteral("installRunner"));
                        QObject::connect(install, &Button::clicked, install, [&s, id, install] {
                            QString error;
                            if (!s.runners.install(id, &error))
                                install->setToolTip(error);
                        });
                        action = install;
                        caption = st.state == RunnerManager::State::Failed && !st.error.isEmpty() ? st.error : QStringLiteral("Not installed yet. Games fall back to XSEmu Core until it is.");
                    }
                }
                const QString& shared = in->profile.runnerVersion;
                if (!shared.isEmpty() && !st.pinnedVersion.isEmpty() && shared != st.pinnedVersion)
                    caption = (caption.isEmpty() ? QString() : caption + QLatin1Char(' '))
                              + QStringLiteral("Shared with %1; this launcher pins %2.").arg(shared, st.pinnedVersion);
                if (l.kind == Line::Kind::Same && !action && caption.isEmpty()) {
                    ++matching;
                    continue;
                }
            } else if (l.kind == Line::Kind::Same) {
                ++matching;
                continue;
            }
            if (l.kind == Line::Kind::NotApplied)
                ++refused;
            else if (l.kind == Line::Kind::Change)
                ++changed;
            card->addRow(l.label, l.to, l.kind == Line::Kind::Same ? QString() : l.from, l.kind == Line::Kind::NotApplied ? l.reason : QString(), action, caption);
        }
        if (card->rowCount() == 0)
            card->addMessage(lines.isEmpty() ? QStringLiteral("These settings carry nothing this launcher can import.")
                                             : QStringLiteral("Nothing changes: these settings already match."));
        QStringList trailing;
        trailing << (changed == 1 ? QStringLiteral("1 change") : QStringLiteral("%1 changes").arg(changed));
        if (refused > 0)
            trailing << QStringLiteral("%1 not applied").arg(refused);
        card->setTrailing(trailing.join(QStringLiteral(" · ")));
        same->setText(matching == 0 ? QString() : matching == 1 ? QStringLiteral("1 setting already matches.") : QStringLiteral("%1 settings already match.").arg(matching));
        same->setVisible(matching > 0);
        if (d->isVisible())
            d->open();
    };
    rebuild();
    QObject::connect(target, &SegmentedControl::currentChanged, d, rebuild);
    QObject::connect(gamePick, &QComboBox::activated, d, rebuild);
    QObject::connect(&s.runners, &RunnerManager::changed, d, rebuild);
    QObject::connect(asDefaults, &Button::clicked, d, [d, &s, in] {
        SettingsProfile::apply(*in, s.settings, Target::Defaults);
        d->dismiss();
    });
    QObject::connect(toGame, &Button::clicked, d, [d, &s, in, gameId] {
        SettingsProfile::apply(*in, s.settings, Target::Game, gameId());
        d->dismiss();
    });
    d->open();
}

class DropFilter : public QObject {
public:
    DropFilter(QWidget* window, Services& s, std::function<QString()> currentTitleId, std::function<bool()> enabled)
        : QObject(window)
        , m_window(window)
        , m_s(s)
        , m_currentTitleId(std::move(currentTitleId))
        , m_enabled(std::move(enabled))
    {
    }

protected:
    bool eventFilter(QObject* o, QEvent* e) override
    {
        if (o != m_window || (e->type() != QEvent::DragEnter && e->type() != QEvent::DragMove && e->type() != QEvent::Drop))
            return false;
        auto* drop = static_cast<QDropEvent*>(e);
        const QString path = profileIn(drop->mimeData());
        if (path.isEmpty() || !m_enabled() || dialogOpen(m_window))
            return false;
        drop->acceptProposedAction();
        if (e->type() == QEvent::Drop)
            SettingsImport::fromFile(m_window, m_s, path, m_currentTitleId());
        return true;
    }

private:
    static QString profileIn(const QMimeData* mime)
    {
        if (!mime || !mime->hasUrls())
            return {};
        for (const QUrl& url : mime->urls()) {
            if (url.isLocalFile() && SettingsImport::isProfileFile(url.toLocalFile()))
                return url.toLocalFile();
        }
        return {};
    }

    QWidget* m_window;
    Services& m_s;
    std::function<QString()> m_currentTitleId;
    std::function<bool()> m_enabled;
};

}

bool SettingsImport::isProfileFile(const QString& path)
{
    return path.endsWith(QLatin1Char('.') + QLatin1String(SettingsProfile::kFileSuffix), Qt::CaseInsensitive);
}

void SettingsImport::fromText(QWidget* window, Services& s, const QByteArray& input, const QString& titleId)
{
    const SettingsProfile::Parsed in = SettingsProfile::parse(input);
    if (!in.ok()) {
        showError(window, in.error);
        return;
    }
    showPreview(window, s, in, titleId);
}

void SettingsImport::fromClipboard(QWidget* window, Services& s, const QString& titleId)
{
    const QString text = QGuiApplication::clipboard()->text();
    if (text.trimmed().isEmpty()) {
        showError(window, QStringLiteral("The clipboard is empty. Copy the whole share code first, from XSEMU1: to its last character."));
        return;
    }
    fromText(window, s, text.toUtf8(), titleId);
}

void SettingsImport::fromFile(QWidget* window, Services& s, const QString& path, const QString& titleId)
{
    QFile f(path);
    if (f.size() > kMaxFile) {
        showError(window, QStringLiteral("%1 is too large to be an XSEmu settings profile.").arg(QFileInfo(path).fileName()));
        return;
    }
    if (!f.open(QIODevice::ReadOnly)) {
        showError(window, QStringLiteral("%1 can't be read: %2.").arg(QFileInfo(path).fileName(), f.errorString()));
        return;
    }
    fromText(window, s, f.read(kMaxFile + 1), titleId);
}

void SettingsImport::chooseFile(QWidget* window, Services& s, const QString& titleId)
{
    const QString path = QFileDialog::getOpenFileName(window, QStringLiteral("Import settings"), {},
                                                      QStringLiteral("XSEmu settings (*.%1);;All files (*)").arg(QLatin1String(SettingsProfile::kFileSuffix)));
    if (!path.isEmpty())
        fromFile(window, s, path, titleId);
}

void SettingsImport::install(QWidget* window, Services& s, std::function<QString()> currentTitleId, std::function<bool()> enabled)
{
    window->setAcceptDrops(true);
    window->installEventFilter(new DropFilter(window, s, currentTitleId, enabled));
    auto* paste = new QShortcut(QKeySequence(Qt::CTRL | Qt::SHIFT | Qt::Key_V), window);
    QObject::connect(paste, &QShortcut::activated, window, [window, &s, currentTitleId, enabled] {
        if (enabled() && !dialogOpen(window))
            fromClipboard(window, s, currentTitleId());
    });
}
