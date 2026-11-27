#include "Sections.h"
#include "SettingsKit.h"

#include "app/Services.h"
#include "widgets/ChoiceCard.h"
#include "widgets/InfoNote.h"

#include <QButtonGroup>
#include <QHBoxLayout>

// device_type: written into each title's xcl.toml at launch (LaunchConfig), unless the game has its
// own override.
QWidget* SettingsSections::console(Services& s)
{
    const SettingsKit::Section sec = SettingsKit::makeSection(
        QStringLiteral("Console"),
        QStringLiteral("The Xbox Series model XSEmu Core reports to the game. It sets the title memory budget and the resolution targets the game picks."));
    sec.caps(QStringLiteral("Device type"));
    sec.body->addSpacing(12);
    auto* cards = new QWidget;
    auto* h = new QHBoxLayout(cards);
    h->setContentsMargins(0, 0, 0, 0);
    h->setSpacing(10);
    auto* seriesS = new ChoiceCard(QStringLiteral("Series S"), QStringLiteral("lockhart · 8 GB title memory"),
                                   QStringLiteral("Lower resolution targets and smaller allocations. The model XSEmu Core is tuned for."),
                                   QStringLiteral("DEFAULT"));
    auto* seriesX = new ChoiceCard(QStringLiteral("Series X"), QStringLiteral("anaconda · 13.5 GB title memory"),
                                   QStringLiteral("Higher resolution targets and bigger allocations. Needs much more RAM and VRAM."));
    auto* group = new QButtonGroup(cards);
    group->addButton(seriesS, 0);
    group->addButton(seriesX, 1);
    h->addWidget(seriesS, 1);
    h->addWidget(seriesX, 1);
    sec.add(cards);
    const auto sync = [&s, seriesS, seriesX] {
        const bool x = s.settings.defaultDeviceType() == QLatin1String("anaconda");
        seriesS->setChecked(!x);
        seriesX->setChecked(x);
    };
    sync();
    QObject::connect(group, &QButtonGroup::idClicked, cards, [&s](int id) {
        s.settings.setDefaultDeviceType(id == 1 ? QStringLiteral("anaconda") : QStringLiteral("lockhart"));
    });
    QObject::connect(&s.settings, &AppSettings::changed, cards, [sync](const QString& key) {
        if (key == QLatin1String(AppSettings::Key::DefaultDeviceType))
            sync();
    });
    sec.add(new InfoNote(QStringLiteral("Switching applies on the next launch. The shader cache is shared by both models, and a game's own setting in its details wins over this default.")), 20);
    sec.finish();
    return sec.page;
}
