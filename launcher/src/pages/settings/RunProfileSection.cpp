#include "Sections.h"
#include "SettingsKit.h"

#include "app/Services.h"
#include "core/LaunchConfig.h"
#include "pages/RunProfileEditor.h"
#include "widgets/Button.h"
#include "widgets/InfoNote.h"

// The global run profile (AppSettings::runProfile): the environment XSEmu gets for Play and for its
// warm-up (LaunchConfig::runEnvironment), so both fill and read one shader cache generation. Reset
// forgets the edit, so the profile follows the launcher's defaults again.
QWidget* SettingsSections::runProfile(Services& s)
{
    const SettingsKit::Section sec = SettingsKit::makeSection(
        QStringLiteral("Run profile"),
        QStringLiteral("The environment variables XSEmu Core starts with, one KEY=VALUE per line. Play and Warm-up use the same profile, so the warm-up "
                       "fills the shader cache the game reads. A game can have its own profile in its details."));
    sec.caps(QStringLiteral("Variables"));
    auto* editor = new RunProfileEditor(int(AppSettings::defaultRunProfile().size()) + 1);
    sec.add(editor, 8);
    QObject::connect(editor, &RunProfileEditor::edited, &s.settings, &AppSettings::setRunProfile);
    auto* reset = new Button(QStringLiteral("Reset to defaults"), Button::Variant::Secondary, Button::Size::Small);
    reset->setObjectName(QStringLiteral("resetRunProfile"));
    SettingsKit::Row* resetRow = sec.row(QStringLiteral("Launcher defaults"), {}, reset);
    QObject::connect(reset, &Button::clicked, &s.settings, &AppSettings::resetRunProfile);
    const auto sync = [&s, editor, reset, resetRow] {
        const QStringList lines = s.settings.runProfile();
        const bool defaults = lines == AppSettings::defaultRunProfile();
        editor->setLines(lines);
        reset->setEnabled(!defaults);
        resetRow->setHelp(defaults ? QStringLiteral("This is the profile the launcher ships with.")
                                   : QStringLiteral("Go back to the %1 variables the launcher ships with.").arg(AppSettings::defaultRunProfile().size()));
    };
    sync();
    QObject::connect(&s.settings, &AppSettings::changed, sec.page, [sync](const QString& key) {
        if (key == QLatin1String(AppSettings::Key::RunProfile))
            sync();
    });

    sec.caps(QStringLiteral("How it applies"));
    sec.add(new InfoNote(QStringLiteral("Translator switches such as XCL_SHADER_BACKEND, XCL_RECOMP_* and XCL_LOOP_* are part of the shader cache's identity. "
                                        "Changing one starts a new cache generation, so the next warm-up builds everything again.")),
            8);
    QStringList owned;
    for (const auto& k : LaunchConfig::settingsOwnedKeys())
        owned << k.first;
    sec.add(new InfoNote(QStringLiteral("The profile applies on top of each game's xcl.toml and the launcher's own environment: XCL_DISPLAY_MODE and "
                                        "XCL_ADAPTER here win over Graphics, Display mode and D3D12 adapter. %1 come from their own settings; a line "
                                        "naming one of them is not used.")
                             .arg(owned.join(QStringLiteral(", ")))),
            8);
    sec.finish();
    return sec.page;
}
