#include "ShareSettingsDialog.h"

#include "ImportSettingsDialog.h"
#include "ProfileCard.h"
#include "settings/SettingsKit.h"

#include "app/Services.h"
#include "core/SettingsProfile.h"
#include "widgets/Button.h"
#include "widgets/Card.h"
#include "widgets/Dialog.h"
#include "widgets/InfoNote.h"
#include "widgets/Text.h"

#include <QClipboard>
#include <QDir>
#include <QFile>
#include <QFileDialog>
#include <QGridLayout>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QLineEdit>
#include <QRegularExpression>
#include <QSaveFile>
#include <QTimer>
#include <QVBoxLayout>

namespace {

constexpr int kDiscordLimit = 2000;

QString madeOn(const SettingsProfile::Profile& p)
{
    QStringList parts;
    if (!p.gpuModel.isEmpty())
        parts << QStringLiteral("Made on %1").arg(p.gpuModel);
    if (!p.launcherVersion.isEmpty())
        parts << QStringLiteral("launcher %1").arg(p.launcherVersion);
    return parts.join(QStringLiteral(" \u00B7 "));
}

// The code in fixed lines for display only (it has no spaces to wrap at); Copy copies it whole.
QString lines(const QString& code)
{
    QStringList out;
    for (qsizetype i = 0; i < code.size(); i += 64)
        out << code.mid(i, 64);
    return out.join(QLatin1Char('\n'));
}

void open(QWidget* window, Services& s, SettingsProfile::Profile profile, const QString& title)
{
    const HardwareInfo& hw = s.hardware.info();
    profile.launcherVersion = QStringLiteral(XSEMU_VERSION);
    if (s.hardware.ready()) {
        profile.gpuVendor = Runners::vendorName(hw.gpuVendorId);
        profile.gpuModel = hw.gpuName;
    }

    auto* d = new Dialog(window, 560);
    d->setObjectName(QStringLiteral("shareSettings"));
    d->setIcon(QStringLiteral("external"), int(Theme::Tone::Accent));
    d->setTitle(title);
    QVBoxLayout* b = d->body();
    b->setSpacing(0);
    auto* lead = new Label(QStringLiteral("Anyone with XSEmu Launcher can paste the code or open the file. They see exactly what would change before anything is applied."),
                           Label::Style::Small, Theme::Role::Text2);
    lead->setWordWrap(true);
    b->addWidget(lead);
    b->addSpacing(16);

    auto* fields = new QGridLayout;
    fields->setHorizontalSpacing(12);
    fields->setVerticalSpacing(8);
    auto* name = new QLineEdit(profile.name);
    name->setProperty("framed", true);
    name->setMaxLength(SettingsProfile::kNameMax);
    auto* note = new QLineEdit(profile.note);
    note->setObjectName(QStringLiteral("shareNote"));
    note->setProperty("framed", true);
    note->setMaxLength(SettingsProfile::kNoteMax);
    note->setPlaceholderText(QStringLiteral("What these settings are good for (optional)"));
    fields->addWidget(new Label(QStringLiteral("Name"), Label::Style::Small, Theme::Role::Text2), 0, 0);
    fields->addWidget(name, 0, 1);
    fields->addWidget(new Label(QStringLiteral("Note"), Label::Style::Small, Theme::Role::Text2), 1, 0);
    fields->addWidget(note, 1, 1);
    fields->setColumnMinimumWidth(0, 48);
    b->addLayout(fields);
    b->addSpacing(16);

    auto* card = new ProfileCard(QStringLiteral("Included"));
    for (const SettingsProfile::KeySpec& spec : SettingsProfile::keys()) {
        const QVariant v = profile.values.value(spec.key);
        if (!v.isValid())
            continue;
        const bool pinned = spec.kind == SettingsProfile::KeySpec::Kind::Runner && !profile.runnerVersion.isEmpty();
        QString value = SettingsProfile::valueLabel(spec.key, v);
        if (pinned)
            value = QStringLiteral("%1 · pinned %2").arg(value, profile.runnerVersion);
        else if (spec.kind == SettingsProfile::KeySpec::Kind::RunProfile)
            value += QStringLiteral(" · paths on this PC stay here");
        card->addRow(spec.label, value);
    }
    const bool empty = profile.values.isEmpty();
    if (empty)
        card->addMessage(QStringLiteral("Nothing to share yet: this game follows the launcher defaults. Choose overrides in its details first."));
    card->setTrailing(empty ? QStringLiteral("None") : profile.values.size() == 1 ? QStringLiteral("1 setting") : QStringLiteral("%1 settings").arg(profile.values.size()));
    b->addWidget(card);
    if (const QString made = madeOn(profile); !made.isEmpty()) {
        b->addSpacing(8);
        b->addWidget(new Label(made, Label::Style::Caption, Theme::Role::Text3));
    }
    b->addSpacing(16);

    b->addWidget(new Label(QStringLiteral("Share code"), Label::Style::Caps, Theme::Role::Text3));
    b->addSpacing(8);
    auto* box = new Card(Card::Kind::Sunken, Metrics::RadiusLg);
    auto* bv = new QVBoxLayout(box);
    bv->setContentsMargins(12, 10, 12, 10);
    auto* codeText = new Label({}, Label::Style::Mono, Theme::Role::Text);
    codeText->setObjectName(QStringLiteral("shareCode"));
    bv->addWidget(codeText);
    b->addWidget(box);
    b->addSpacing(6);
    auto* length = new Label({}, Label::Style::Caption, Theme::Role::Text3);
    b->addWidget(length);
    b->addSpacing(14);
    b->addWidget(new InfoNote(QStringLiteral("Only the settings above are shared. Paths, keys, your library, play time and warm-up history stay on this PC.")));

    auto* save = new Button(QStringLiteral("Save file..."), Button::Variant::Secondary, Button::Size::Small, QStringLiteral("download"));
    auto* done = new Button(QStringLiteral("Done"), Button::Variant::Secondary, Button::Size::Small);
    auto* copy = new Button(QStringLiteral("Copy code"), Button::Variant::Primary, Button::Size::Small);
    copy->setObjectName(QStringLiteral("copyShareCode"));
    save->setEnabled(!empty);
    copy->setEnabled(!empty);
    d->addButton(save, true);
    d->addButton(done);
    d->addButton(copy);
    QObject::connect(done, &Button::clicked, d, &Dialog::dismiss);

    auto current = std::make_shared<SettingsProfile::Profile>(profile);
    const auto refresh = [current, name, note, codeText, length] {
        current->name = name->text();
        current->note = note->text();
        const QString code = SettingsProfile::toCode(*current);
        codeText->setText(lines(code));
        codeText->setProperty("code", code);
        length->setText(code.size() <= kDiscordLimit ? QStringLiteral("%1 characters \u00B7 fits in one chat message").arg(code.size())
                                                      : QStringLiteral("%1 characters \u00B7 too long for one chat message: save a file instead").arg(code.size()));
    };
    refresh();
    QObject::connect(name, &QLineEdit::textChanged, d, refresh);
    QObject::connect(note, &QLineEdit::textChanged, d, refresh);
    QObject::connect(copy, &Button::clicked, d, [copy, codeText] {
        QGuiApplication::clipboard()->setText(codeText->property("code").toString());
        copy->setText(QStringLiteral("Copied"));
        copy->setIconName(QStringLiteral("check"));
        QTimer::singleShot(1600, copy, [copy] {
            copy->setText(QStringLiteral("Copy code"));
            copy->setIconName({});
        });
    });
    QObject::connect(save, &Button::clicked, d, [d, current] {
        QString base = current->name.trimmed();
        base.replace(QRegularExpression(QStringLiteral("[\\\\/:*?\"<>|]+")), QStringLiteral(" "));
        if (base.isEmpty())
            base = QStringLiteral("XSEmu settings");
        const QString path = QFileDialog::getSaveFileName(d->window(), QStringLiteral("Save settings"),
                                                          QDir::home().filePath(base + QLatin1Char('.') + QLatin1String(SettingsProfile::kFileSuffix)),
                                                          QStringLiteral("XSEmu settings (*.%1)").arg(QLatin1String(SettingsProfile::kFileSuffix)));
        if (path.isEmpty())
            return;
        QSaveFile f(path);
        if (f.open(QIODevice::WriteOnly) && f.write(SettingsProfile::toJson(*current)) >= 0 && f.commit())
            d->dismiss();
    });
    d->open();
}

}

void SettingsShare::openForGame(QWidget* window, Services& s, const QString& titleId)
{
    QString title = titleId;
    for (const Game& g : s.library.games()) {
        if (g.titleId() == titleId)
            title = g.title();
    }
    open(window, s, SettingsProfile::fromGame(s.settings, titleId, title), QStringLiteral("Share %1 settings").arg(title));
}

void SettingsShare::openForDefaults(QWidget* window, Services& s)
{
    open(window, s, SettingsProfile::fromDefaults(s.settings), QStringLiteral("Share your XSEmu defaults"));
}

void SettingsShare::addSettingsRows(const SettingsKit::Section& sec, Services& s)
{
    sec.caps(QStringLiteral("Share settings"));
    auto* share = new Button(QStringLiteral("Share..."), Button::Variant::Secondary, Button::Size::Small);
    share->setObjectName(QStringLiteral("shareDefaults"));
    QObject::connect(share, &Button::clicked, share, [share, &s] { openForDefaults(share->window(), s); });
    sec.row(QStringLiteral("Share settings"),
            QStringLiteral("Your emulator defaults as a code to paste in a chat, or a .xsemu-settings file. Paths and keys are never included."), share);
    auto* paste = new Button(QStringLiteral("Paste code"), Button::Variant::Secondary, Button::Size::Small);
    paste->setToolTip(QStringLiteral("Ctrl+Shift+V"));
    auto* file = new Button(QStringLiteral("Open file..."), Button::Variant::Secondary, Button::Size::Small);
    QObject::connect(paste, &Button::clicked, paste, [paste, &s] { SettingsImport::fromClipboard(paste->window(), s); });
    QObject::connect(file, &Button::clicked, file, [file, &s] { SettingsImport::chooseFile(file->window(), s); });
    SettingsKit::Row* row = sec.row(QStringLiteral("Import settings"),
                                    QStringLiteral("Paste a share code (Ctrl+Shift+V) or open a file, or drop one on the window. You see what would change first."), paste);
    row->addControl(file);
}
