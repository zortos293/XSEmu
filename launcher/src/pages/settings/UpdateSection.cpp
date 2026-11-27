#include "Sections.h"
#include "SettingsKit.h"

#include "app/Services.h"
#include "core/Paths.h"
#include "core/UpdateService.h"
#include "widgets/Button.h"
#include "widgets/InfoNote.h"
#include "widgets/Text.h"
#include "widgets/Toggle.h"

#include <QDir>
#include <QHBoxLayout>
#include <QPointer>

// Updates: XSEmu Launcher and XSEmu Core, separately, from the releases of zortos293/XSEmu
// (UpdateService). The switch is honoured by the startup check; Check now asks anyway.
QWidget* SettingsSections::updates(Services& s)
{
    const SettingsKit::Section sec = SettingsKit::makeSection(
        QStringLiteral("Updates"),
        QStringLiteral("The launcher and XSEmu Core update separately, from this project's GitHub releases."));

    sec.caps(QStringLiteral("XSEmu Launcher"));
    auto* launcher = SettingsKit::status({});
    auto* check = new Button(QStringLiteral("Check now"), Button::Variant::Secondary, Button::Size::Small, QStringLiteral("refresh"));
    check->setObjectName(QStringLiteral("checkForUpdates"));
    SettingsKit::Row* launcherRow = sec.row(QStringLiteral("This build"), {}, launcher);
    launcherRow->addControl(check);
    auto* updateLauncher = new Button(QStringLiteral("Update launcher"), Button::Variant::Primary, Button::Size::Small, QStringLiteral("download"));
    updateLauncher->setObjectName(QStringLiteral("updateLauncher"));
    sec.row(QStringLiteral("Launcher update"), QStringLiteral("Replaces the launcher's own files, then starts it again. A game or a warm-up must be stopped first."), updateLauncher);

    sec.caps(QStringLiteral("XSEmu Core"));
    auto* core = SettingsKit::status({});
    SettingsKit::Row* coreRow = sec.row(QStringLiteral("Emulator"), QStringLiteral("The emulator the launcher runs, in its own core folder."), core);
    coreRow->helpLabel()->setTextInteractionFlags(Qt::TextSelectableByMouse);
    auto* updateCore = new Button(QStringLiteral("Update XSEmu Core"), Button::Variant::Primary, Button::Size::Small, QStringLiteral("download"));
    updateCore->setObjectName(QStringLiteral("updateCore"));
    auto* restore = new Button(QStringLiteral("Restore previous"), Button::Variant::Secondary, Button::Size::Small);
    restore->setObjectName(QStringLiteral("restoreCore"));
    SettingsKit::Row* coreUpdateRow = sec.row(QStringLiteral("Core update"), QStringLiteral("Replaces only XSEmu.exe in the launcher's core folder. The previous copy is kept until the new one starts."), updateCore);
    coreUpdateRow->addControl(restore);
    auto* note = new InfoNote({}, Theme::Tone::Neutral);
    sec.add(note, 16);

    sec.caps(QStringLiteral("Checking"));
    auto* automatic = new Toggle;
    automatic->setChecked(s.settings.checkForUpdates());
    QObject::connect(automatic, &Toggle::toggled, &s.settings, &AppSettings::setCheckForUpdates);
    sec.row(QStringLiteral("Check for updates automatically"),
            QStringLiteral("Asks GitHub for the newest release a few seconds after the launcher starts. The update itself is never installed without you."),
            automatic);
    auto* prereleases = new Toggle;
    prereleases->setChecked(s.settings.prereleaseUpdates());
    QObject::connect(prereleases, &Toggle::toggled, &s.settings, &AppSettings::setPrereleaseUpdates);
    sec.row(QStringLiteral("Include pre-release versions"),
            QStringLiteral("Test builds are published as pre-releases (0.1.0-pre.2). Off keeps this launcher on finished releases only."),
            prereleases);
    auto* checked = SettingsKit::status({});
    sec.row(QStringLiteral("Last checked"), {}, checked);

    const auto sync = [&s, launcher, core, coreRow, checked, check, updateLauncher, updateCore, restore, note, automatic, prereleases] {
        s.updates.refresh();
        const UpdateService::Status st = s.updates.status();
        const bool busy = st.busy();
        if (st.launcherUpdate)
            launcher->setText(QStringLiteral("%1 · %2 available").arg(st.launcherVersion, st.release.launcher.version));
        else
            launcher->setText(st.launcherVersion);
        launcher->setRole(st.launcherUpdate ? Theme::Role::AccentText : Theme::Role::Text2);

        QStringList lines{QDir::toNativeSeparators(st.corePath)};
        if (st.installedCore.isEmpty())
            lines << QStringLiteral("Its version is not known yet.");
        else
            lines << QStringLiteral("XSEmu Core %1").arg(st.installedCore);
        if (!st.coreBundled)
            lines << QStringLiteral("Set to a file outside the launcher's core folder, so Core updates are off.");
        if (st.coreUpdate && !st.coreBundled)
            lines << QStringLiteral("XSEmu Core %1 is available with a launcher update.").arg(st.release.core.version);
        coreRow->setHelp(lines.join(QLatin1Char('\n')));
        core->setText(st.coreUpdate ? QStringLiteral("%1 · %2 available").arg(st.installedCore.isEmpty() ? QStringLiteral("unknown") : st.installedCore,
                                                                          st.release.core.version)
                                    : (st.installedCore.isEmpty() ? QStringLiteral("Version unknown") : st.installedCore));
        core->setRole(st.coreUpdate ? Theme::Role::AccentText : Theme::Role::Text2);

        checked->setText(st.checkedAt.isValid() ? st.checkedAt.toLocalTime().toString(QStringLiteral("d MMM yyyy, HH:mm")) : QStringLiteral("Never"));
        check->setEnabled(!busy);
        updateLauncher->setEnabled(s.updates.canInstall(UpdateService::Part::Launcher));
        updateCore->setEnabled(s.updates.canInstall(UpdateService::Part::Core));
        restore->setEnabled(st.corePreviousKept && !busy && s.gate.isFree());

        QString message = st.message;
        if (st.error.isEmpty() && message.isEmpty())
            message = QStringLiteral("Updates are checked a few seconds after the launcher starts.");
        if (st.busy() && st.total > 0 && st.done >= 0)
            message += QStringLiteral(" %1%").arg(qRound(100.0 * double(st.done) / double(st.total)));
        note->setText(message);
        note->setTone(st.error.isEmpty() ? (st.updateAvailable() ? Theme::Tone::Accent : Theme::Tone::Neutral) : Theme::Tone::Danger);
        automatic->setChecked(s.settings.checkForUpdates());
        prereleases->setChecked(s.settings.prereleaseUpdates());
    };
    sync();

    QObject::connect(&s.updates, &UpdateService::changed, sec.page, sync);
    QObject::connect(&s.gate, &ProcessGate::changed, sec.page, sync);
    QObject::connect(&s.settings, &AppSettings::changed, sec.page, [sync](const QString& key) {
        if (key == QLatin1String(AppSettings::Key::EmulatorPath) || key == QLatin1String(AppSettings::Key::CheckForUpdates)
            || key == QLatin1String(AppSettings::Key::PrereleaseUpdates))
            sync();
    });
    QObject::connect(check, &Button::clicked, check, [&s] { s.updates.check(true); });
    QObject::connect(updateLauncher, &Button::clicked, updateLauncher, [&s] { s.updates.install(UpdateService::Part::Launcher); });
    QObject::connect(updateCore, &Button::clicked, updateCore, [&s] { s.updates.install(UpdateService::Part::Core); });
    QObject::connect(restore, &Button::clicked, restore, [&s, note] {
        QString error;
        if (s.updates.restorePreviousCore(&error))
            note->setText(error.isEmpty() ? QStringLiteral("The previous XSEmu Core is back.") : error);
        else
            note->setText(error);
    });
    SettingsKit::onShow(sec.page, sync);

    sec.finish();
    return sec.page;
}
