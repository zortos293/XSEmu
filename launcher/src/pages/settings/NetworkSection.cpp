#include "Sections.h"
#include "SettingsKit.h"

#include "app/Services.h"
#include "widgets/InfoNote.h"
#include "widgets/Text.h"
#include "widgets/Toggle.h"

// Offline: XCL_NETWORK=0. Store behaviour: XCL_STORE=offline-error, or unset (the store API is
// refused as unavailable). Both go into the game's environment at launch (LaunchConfig).
QWidget* SettingsSections::network(Services& s)
{
    const SettingsKit::Section sec = SettingsKit::makeSection(
        QStringLiteral("Network"), QStringLiteral("XSEmu Core runs offline by default. Nothing in the game reaches Xbox services."));
    sec.caps(QStringLiteral("Connection"));
    auto* offline = new Toggle;
    offline->setChecked(s.settings.offline());
    QObject::connect(offline, &Toggle::toggled, &s.settings, &AppSettings::setOffline);
    sec.row(QStringLiteral("Offline mode"),
            QStringLiteral("Block all network access from the game. Matchmaking, cloud saves and telemetry fail the way they do on an offline console."),
            offline);

    sec.caps(QStringLiteral("Store"));
    auto* behaviour = SettingsKit::dropdown({{QStringLiteral("Offline error"), true}, {QStringLiteral("Not available"), false}}, s.settings.storeOfflineError());
    sec.row(QStringLiteral("Store behaviour"), QStringLiteral("How store and licence queries answer while offline."), behaviour);
    auto* context = SettingsKit::status({});
    SettingsKit::Row* contextRow = sec.row(QStringLiteral("Store context"), {}, context);
    auto* queries = SettingsKit::status({});
    SettingsKit::Row* queriesRow = sec.row(QStringLiteral("Store queries"), {}, queries);
    const auto sync = [&s, context, contextRow, queries, queriesRow, offline, behaviour] {
        const bool error = s.settings.storeOfflineError();
        offline->setChecked(s.settings.offline());
        behaviour->setCurrentIndex(error ? 0 : 1);
        context->setText(error ? QStringLiteral("Succeeds") : QStringLiteral("Refused"));
        context->setRole(error ? Theme::Role::Success : Theme::Role::Text2);
        contextRow->setHelp(error ? QStringLiteral("Opens normally so the game can check entitlements.")
                                  : QStringLiteral("The store API is refused, as if the console had no store."));
        queries->setText(error ? QStringLiteral("Network error") : QStringLiteral("Refused"));
        queriesRow->setHelp(error ? QStringLiteral("Every query returns the store network error.")
                                  : QStringLiteral("Games see no store at all."));
    };
    sync();
    QObject::connect(behaviour, &QComboBox::activated, sec.page, [&s, behaviour] { s.settings.setStoreOfflineError(behaviour->currentData().toBool()); });
    QObject::connect(&s.settings, &AppSettings::changed, sec.page, [sync](const QString& key) {
        if (key == QLatin1String(AppSettings::Key::StoreOfflineError) || key == QLatin1String(AppSettings::Key::Offline))
            sync();
    });
    sec.add(new InfoNote(QStringLiteral("XSEmu Core never reports a licence it doesn't have. A game that needs a live licence check shows its own offline message.")), 20);
    sec.finish();
    return sec.page;
}
