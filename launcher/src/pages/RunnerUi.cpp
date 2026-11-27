#include "RunnerUi.h"

#include "app/Services.h"
#include "core/DiskSpace.h"
#include "core/Format.h"
#include "core/HardwareInfo.h"
#include "core/Paths.h"
#include "widgets/ChoiceCard.h"
#include "widgets/Badge.h"
#include "widgets/Button.h"
#include "widgets/Card.h"
#include "widgets/Dialog.h"
#include "widgets/InfoNote.h"
#include "widgets/Text.h"

#include <QButtonGroup>
#include <QDir>
#include <QFileDialog>
#include <algorithm>
#include <QGridLayout>
#include <QHBoxLayout>
#include <memory>
#include <QLocale>
#include <QVBoxLayout>

namespace RunnerUi {

QString owner(const RunnerSpec& spec)
{
    return spec.isBuiltIn() ? QStringLiteral("built in") : QStringLiteral("by %1").arg(spec.repo.section(QLatin1Char('/'), 0, 0));
}

QString repoLine(const RunnerSpec& spec)
{
    QStringList parts{QStringLiteral("github.com/%1").arg(spec.repo)};
    if (spec.isBuiltIn())
        parts << QStringLiteral("launcher %1").arg(QStringLiteral(XSEMU_VERSION));
    else if (!spec.pinnedVersion().isEmpty())
        parts << QStringLiteral("pinned %1").arg(spec.pinnedVersion());
    if (!spec.licence.isEmpty())
        parts << spec.licence;
    return parts.join(QStringLiteral(" \u00B7 "));
}

QString provenance(const RunnerSpec& spec)
{
    if (spec.install == RunnerSpec::Install::BuildFromSource)
        return QStringLiteral("Community build, pinned version. Built from source on this PC.");
    return QStringLiteral("Community build, pinned release, checked against its SHA-256.");
}

Badge* vendorBadge(const RunnerSpec& spec)
{
    if (spec.badge.isEmpty())
        return nullptr;
    auto* b = new Badge(spec.badge, Theme::Tone::Neutral, Badge::Kind::Outline);
    b->setIcon(QStringLiteral("chip"));
    return b;
}

void updateStatusBadge(Badge* b, const RunnerManager::Status& st)
{
    using S = RunnerManager::State;
    b->setDotVisible(true);
    switch (st.state) {
    case S::BuiltIn:
    case S::Installed: b->setText(QStringLiteral("Installed")); b->setTone(Theme::Tone::Success); break;
    case S::UpdateAvailable: b->setText(QStringLiteral("Update available")); b->setTone(Theme::Tone::Accent); break;
    case S::NotInstalled: b->setText(QStringLiteral("Not installed")); b->setTone(Theme::Tone::Neutral); break;
    case S::Unavailable: b->setText(QStringLiteral("Not available on this PC")); b->setTone(Theme::Tone::Neutral); break;
    case S::Failed: b->setText(QStringLiteral("Install failed")); b->setTone(Theme::Tone::Danger); break;
    case S::Uninstalling: b->setText(QStringLiteral("Uninstalling")); b->setTone(Theme::Tone::Neutral); break;
    case S::Moving: {
        const bool known = st.progress.total > 0 && st.progress.done >= 0;
        b->setText(known ? QStringLiteral("Moving \u00B7 %1%").arg(qRound(100.0 * double(st.progress.done) / double(st.progress.total))) : QStringLiteral("Moving"));
        b->setTone(Theme::Tone::Accent);
        break;
    }
    case S::Installing: {
        const QString verb = st.progress.phase == 1 && st.progress.step == QLatin1String("Building") ? QStringLiteral("Building") : QStringLiteral("Installing");
        const bool known = st.progress.total > 0 && st.progress.done >= 0;
        b->setText(known ? QStringLiteral("%1 \u00B7 %2%").arg(verb).arg(qRound(100.0 * double(st.progress.done) / double(st.progress.total))) : verb);
        b->setTone(Theme::Tone::Accent);
        break;
    }
    }
}

Badge* statusBadge(const RunnerManager::Status& status, bool)
{
    auto* b = new Badge;
    updateStatusBadge(b, status);
    return b;
}

QString statusDetail(Services& s, const RunnerSpec& spec, const RunnerManager::Status& st)
{
    using S = RunnerManager::State;
    switch (st.state) {
    case S::BuiltIn: return QStringLiteral("Built in. Can't be uninstalled.");
    case S::Installed:
    case S::UpdateAvailable: {
        const QDateTime at = s.runners.installedAt(spec.id);
        const QString when = at.isValid() ? QStringLiteral("Installed %1").arg(QLocale().toString(at.date(), QStringLiteral("d MMM"))) : QStringLiteral("Installed");
        if (st.state == S::UpdateAvailable)
            return QStringLiteral("%1 \u2192 %2").arg(st.installedVersion, st.pinnedVersion);
        const qint64 bytes = s.runners.installedBytes(spec.id);
        return bytes < 0 ? when : QStringLiteral("%1 \u00B7 %2").arg(when, Format::bytes(bytes));
    }
    case S::Installing: return QStringLiteral("Since %1").arg(QLocale().toString(st.progress.startedAt.time(), QLocale::ShortFormat));
    case S::Uninstalling: return QStringLiteral("Deleting %1").arg(st.progress.detail);
    case S::Moving:
        return st.progress.total > 0 ? QStringLiteral("Moving to %1 \u00B7 %2 of %3").arg(st.progress.detail, Format::bytes(st.progress.done), Format::bytes(st.progress.total))
                                     : QStringLiteral("Moving to %1").arg(st.progress.detail);
    case S::NotInstalled:
        return spec.install == RunnerSpec::Install::BuildFromSource ? QStringLiteral("Builds from source on this PC; build time not measured yet.")
                                                                    : QStringLiteral("Downloads %1").arg(spec.assetName);
    case S::Unavailable: return spec.windowsOnly ? QStringLiteral("Runs on Windows only.") : QStringLiteral("No version is pinned yet.");
    case S::Failed: return st.error;
    }
    return {};
}

Runners::Recommendation recommendation(Services& s)
{
    const HardwareInfo& hw = s.hardware.info();
    return Runners::recommend(hw.gpuVendorId, hw.gpuName, hw.totalRamBytes, [&s](const QString& id) { return s.runners.isInstallable(id); });
}

QList<const RunnerSpec*> displayOrder(Services& s, const QString& first)
{
    const QString recommended = recommendation(s).runnerId;
    QList<const RunnerSpec*> order;
    for (const RunnerSpec& r : Runners::curated())
        order << &r;
    auto rank = [&](const RunnerSpec* r) {
        if (!first.isEmpty() && r->id == first)
            return 0;
        if (r->isBuiltIn())
            return 1;
        if (r->id == recommended)
            return 2;
        return s.runners.isRunnable(r->id) ? 3 : 4;
    };
    std::stable_sort(order.begin(), order.end(), [&](const RunnerSpec* a, const RunnerSpec* b) { return rank(a) < rank(b); });
    return order;
}

void confirmInstall(QWidget* window, Services& s, const QString& runnerId)
{
    const RunnerSpec* spec = Runners::find(runnerId);
    if (!spec)
        return;
    const bool build = spec->install == RunnerSpec::Install::BuildFromSource;
    const bool update = s.runners.status(runnerId).state == RunnerManager::State::UpdateAvailable;
    auto* d = new Dialog(window, 520);
    d->setIcon(QStringLiteral("download"), int(Theme::Tone::Accent));
    d->setTitle(QStringLiteral("%1 %2?").arg(update ? QStringLiteral("Update") : QStringLiteral("Install"), spec->name));
    auto* lead = new Label(build ? QStringLiteral("XSEmu Launcher downloads the pinned source and builds it on this PC. Nothing runs until the build finishes.")
                                 : QStringLiteral("XSEmu Launcher downloads the pinned release and checks it against its SHA-256 before anything runs."),
                           Label::Style::Small, Theme::Role::Text2);
    lead->setWordWrap(true);
    d->body()->addWidget(lead);
    d->body()->addSpacing(14);

    auto* table = new Card(Card::Kind::Sunken, Metrics::RadiusLg);
    auto* g = new QGridLayout(table);
    g->setContentsMargins(14, 12, 14, 12);
    g->setHorizontalSpacing(16);
    g->setVerticalSpacing(8);
    QList<QPair<QString, QString>> rows{{QStringLiteral("Source"), QStringLiteral("github.com/%1").arg(spec->repo)},
                                        {build ? QStringLiteral("Pinned commit") : QStringLiteral("Pinned release"), build ? spec->commit.left(7) : spec->releaseTag},
                                        {QStringLiteral("Licence"), spec->licence}};
    if (build)
        rows << qMakePair(QStringLiteral("Build time"), QStringLiteral("Not measured yet"));
    else
        rows << qMakePair(QStringLiteral("SHA-256"), spec->assetSha256.left(16) + QStringLiteral("\u2026"));
    rows << qMakePair(QStringLiteral("Installs to"), QDir::toNativeSeparators(spec->dataDir()));
    for (int i = 0; i < rows.size(); ++i) {
        g->addWidget(new Label(rows[i].first, Label::Style::Small, Theme::Role::Text3), i, 0);
        const bool code = i != 2 && rows[i].first != QLatin1String("Build time");
        auto* v = new Label(rows[i].second, code ? Label::Style::Mono : Label::Style::Small, Theme::Role::Text);
        v->setTextInteractionFlags(Qt::TextSelectableByMouse);
        g->addWidget(v, i, 1, Qt::AlignRight);
    }
    d->body()->addWidget(table);

    const HardwareInfo& hw = s.hardware.info();
    const Runners::Recommendation rec = recommendation(s);
    const RunnerSpec::Vendor have = hw.gpuVendorId == Runners::kVendorAmd ? RunnerSpec::Vendor::Amd
                                    : hw.gpuVendorId == Runners::kVendorNvidia ? RunnerSpec::Vendor::Nvidia
                                                                               : RunnerSpec::Vendor::Any;
    if (spec->recommendFor != RunnerSpec::Vendor::Any && spec->recommendFor != have) {
        const RunnerSpec* better = Runners::find(rec.runnerId);
        d->body()->addSpacing(12);
        d->body()->addWidget(new InfoNote(QStringLiteral("This runner is tuned for %1 GPUs. This PC has %2, so %3 is the better fit.")
                                              .arg(spec->recommendFor == RunnerSpec::Vendor::Amd ? QStringLiteral("AMD") : QStringLiteral("NVIDIA"),
                                                   hw.gpuNameOrUnknown(), better ? better->name : QStringLiteral("XSEmu Core")),
                                          Theme::Tone::Warning, QStringLiteral("warning")));
    }
    for (const QString& note : {build ? spec->buildNote : QString(), spec->note}) {
        if (note.isEmpty())
            continue;
        d->body()->addSpacing(10);
        auto* l = new Label(note, Label::Style::Caption, Theme::Role::Text3);
        l->setWordWrap(true);
        d->body()->addWidget(l);
    }
    auto* cancel = new Button(QStringLiteral("Cancel"), Button::Variant::Secondary, Button::Size::Small);
    QObject::connect(cancel, &Button::clicked, d, &Dialog::dismiss);
    d->addButton(cancel);
    auto* go = new Button(build ? QStringLiteral("Build and install") : QStringLiteral("Download and install"), Button::Variant::Primary, Button::Size::Small, QStringLiteral("download"));
    go->setObjectName(QStringLiteral("confirmInstall"));
    QObject::connect(go, &Button::clicked, d, [d, &s, runnerId] {
        s.runners.install(runnerId);
        d->dismiss();
    });
    d->addButton(go);
    d->open();
}

void changeLocation(QWidget* window, Services& s, const QString& runnerId, const QString& proposed)
{
    const RunnerSpec* spec = Runners::find(runnerId);
    if (!spec || spec->isBuiltIn())
        return;
    const RunnerManager::State state = s.runners.status(runnerId).state;
    const bool installed = state == RunnerManager::State::Installed || state == RunnerManager::State::UpdateAvailable;
    const bool build = spec->install == RunnerSpec::Install::BuildFromSource;
    auto folder = std::make_shared<QString>(proposed);

    auto* d = new Dialog(window, 540);
    d->setObjectName(QStringLiteral("changeLocation"));
    d->setIcon(QStringLiteral("folder"), int(Theme::Tone::Accent));
    d->setTitle(QStringLiteral("Change where %1 is kept").arg(spec->name));
    auto* lead = new Label(build ? QStringLiteral("Its source checkout, build folder, program, data and shader cache go into a %1 folder in the place you pick. The Rust toolchain stays in the Runners folder.").arg(spec->id)
                                 : QStringLiteral("Its program, data and shader cache go into a %1 folder in the place you pick.").arg(spec->id),
                           Label::Style::Small, Theme::Role::Text2);
    lead->setWordWrap(true);
    d->body()->addWidget(lead);
    d->body()->addSpacing(14);

    auto* table = new Card(Card::Kind::Sunken, Metrics::RadiusLg);
    auto* g = new QGridLayout(table);
    g->setContentsMargins(14, 12, 14, 12);
    g->setHorizontalSpacing(16);
    g->setVerticalSpacing(8);
    auto addRow = [g](int row, const QString& name, QWidget* value) {
        g->addWidget(new Label(name, Label::Style::Small, Theme::Role::Text3), row, 0, Qt::AlignLeft | Qt::AlignVCenter);
        g->addWidget(value, row, 1, Qt::AlignRight | Qt::AlignVCenter);
    };
    auto* now = new Label(QDir::toNativeSeparators(spec->dataDir()), Label::Style::Mono, Theme::Role::Text);
    now->setTextInteractionFlags(Qt::TextSelectableByMouse);
    addRow(0, QStringLiteral("Now"), now);
    auto* next = new Label({}, Label::Style::Mono, Theme::Role::Text);
    next->setObjectName(QStringLiteral("locationTarget"));
    next->setTextInteractionFlags(Qt::TextSelectableByMouse);
    addRow(1, QStringLiteral("Moves to"), next);
    auto* free = new Label({}, Label::Style::Small, Theme::Role::Text);
    free->setObjectName(QStringLiteral("locationFree"));
    addRow(2, QStringLiteral("Free there"), free);
    d->body()->addWidget(table);

    auto* browseRow = new QHBoxLayout;
    browseRow->setContentsMargins(0, 10, 0, 0);
    auto* browse = new Button(QStringLiteral("Choose folder..."), Button::Variant::Secondary, Button::Size::Small, QStringLiteral("folder"));
    browse->setObjectName(QStringLiteral("locationBrowse"));
    browseRow->addWidget(browse);
    browseRow->addStretch(1);
    d->body()->addLayout(browseRow);

    auto* low = new InfoNote({}, Theme::Tone::Warning, QStringLiteral("warning"));
    low->setObjectName(QStringLiteral("locationLowSpace"));
    d->body()->addSpacing(10);
    d->body()->addWidget(low);

    ChoiceCard* move = nullptr;
    ChoiceCard* rebuild = nullptr;
    if (installed) {
        d->body()->addSpacing(12);
        move = new ChoiceCard(QStringLiteral("Move"), QStringLiteral("Copies its files there, checks the sizes and the program, then deletes the old folder."),
                              QStringLiteral("Nothing is built again; on the same drive it takes a moment."), QStringLiteral("RECOMMENDED"));
        move->setObjectName(QStringLiteral("locationMove"));
        rebuild = new ChoiceCard(QStringLiteral("Rebuild there"),
                                 build ? QStringLiteral("Builds the pinned source again in the new folder, then deletes the old one.")
                                       : QStringLiteral("Downloads the pinned release again into the new folder, then deletes the old one."),
                                 QStringLiteral("If that fails, %1 keeps running from where it is now.").arg(spec->name));
        rebuild->setObjectName(QStringLiteral("locationRebuild"));
        auto* group = new QButtonGroup(d);
        group->setExclusive(true);
        for (ChoiceCard* c : {move, rebuild}) {
            c->setCheckable(true);
            group->addButton(c);
            d->body()->addWidget(c);
            d->body()->addSpacing(8);
        }
        move->setChecked(true);
    }

    const QString busy = s.runners.busyReason(runnerId);
    auto* why = new InfoNote(busy, Theme::Tone::Danger, QStringLiteral("warning"));
    why->setObjectName(QStringLiteral("locationBusy"));
    why->setVisible(!busy.isEmpty());
    d->body()->addSpacing(10);
    d->body()->addWidget(why);

    auto* cancel = new Button(QStringLiteral("Cancel"), Button::Variant::Secondary, Button::Size::Small);
    QObject::connect(cancel, &Button::clicked, d, &Dialog::dismiss);
    d->addButton(cancel);
    auto* go = new Button(installed ? QStringLiteral("Move") : QStringLiteral("Use this folder"), Button::Variant::Primary, Button::Size::Small, QStringLiteral("folder"));
    go->setObjectName(QStringLiteral("locationConfirm"));
    d->addButton(go);

    const auto refresh = [spec, folder, next, free, low, go, rebuild, busy, installed] {
        const bool chosen = !folder->isEmpty();
        next->setText(chosen ? QDir::toNativeSeparators(QDir::cleanPath(QDir(*folder).filePath(spec->id))) : QStringLiteral("Choose a folder"));
        next->setRole(chosen ? Theme::Role::Text : Theme::Role::Text3);
        const DiskSpace::Check space = DiskSpace::checkRunnerFolder(chosen ? *folder : spec->dataDir());
        free->setText(space.freeBytes < 0 ? QStringLiteral("Unknown") : QStringLiteral("%1 on %2").arg(Format::bytes(space.freeBytes), space.drive));
        free->setRole(space.low ? Theme::Role::Warning : Theme::Role::Text);
        low->setText(space.text);
        low->setVisible(space.low);
        if (installed)
            go->setText(rebuild->isChecked() ? QStringLiteral("Rebuild there") : QStringLiteral("Move"));
        go->setEnabled(chosen && busy.isEmpty());
    };
    refresh();
    if (s.settings.runnerLocations().contains(runnerId)) {
        // moved somewhere else before: back to the Runners folder in one click
        auto* home = new Button(QStringLiteral("Use the Runners folder"), Button::Variant::Ghost, Button::Size::Small);
        QObject::connect(home, &Button::clicked, d, [folder, refresh] {
            *folder = Paths::runnersDir();
            refresh();
        });
        d->addButton(home, true);
    }
    if (rebuild) {
        QObject::connect(move, &QAbstractButton::toggled, d, refresh);
        QObject::connect(rebuild, &QAbstractButton::toggled, d, refresh);
    }
    QObject::connect(browse, &Button::clicked, d, [d, spec, folder, refresh] {
        const QString dir = QFileDialog::getExistingDirectory(d->window(), QStringLiteral("Choose where to keep %1").arg(spec->name),
                                                              folder->isEmpty() ? QFileInfo(spec->dataDir()).path() : *folder);
        if (dir.isEmpty())
            return;
        *folder = dir;
        refresh();
    });
    QObject::connect(go, &Button::clicked, d, [d, &s, runnerId, folder, rebuild, why] {
        QString error;
        const auto how = rebuild && rebuild->isChecked() ? RunnerManager::Relocation::Rebuild : RunnerManager::Relocation::Move;
        if (!s.runners.relocate(runnerId, *folder, how, &error)) {
            why->setText(error);
            why->show();
            return;
        }
        d->dismiss();
    });
    d->open();
}

}
