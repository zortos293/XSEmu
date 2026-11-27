#include "Sections.h"
#include "SettingsKit.h"

#include "app/Services.h"
#include "widgets/SegmentedControl.h"
#include "widgets/Toggle.h"

// On at start: AppSettings::overlayDefault, sent to the game as {"cmd":"overlay","on":true} when its
// control pipe connects (EmulatorController). Position, size and the metrics shown are fixed in the
// emulator (a frame-time graph in the top-left corner), so they are shown unavailable.
QWidget* SettingsSections::overlay(Services& s)
{
    const SettingsKit::Section sec = SettingsKit::makeSection(
        QStringLiteral("Overlay"), QStringLiteral("XSEmu Core draws the overlay inside the game image, so it shows up in captures and costs no extra window."));
    sec.caps(QStringLiteral("Display"));
    auto* on = new Toggle;
    on->setChecked(s.settings.overlayDefault());
    QObject::connect(on, &Toggle::toggled, &s.settings, &AppSettings::setOverlayDefault);
    sec.row(QStringLiteral("Stall and performance overlay"), QStringLiteral("Shown when a game starts. Toggle it in game with F3 or from the game bar."), on);
    auto* position = new SegmentedControl({QStringLiteral("Top left"), QStringLiteral("Top right"), QStringLiteral("Bottom left"), QStringLiteral("Bottom right")});
    sec.row(QStringLiteral("Position"), {}, position)->setUnavailable(QStringLiteral("XSEmu Core draws it in the top-left corner. Moving it is not in this XSEmu Core build yet."));
    auto* size = new SegmentedControl({QStringLiteral("Small"), QStringLiteral("Medium"), QStringLiteral("Large")});
    size->setCurrentIndex(1);
    sec.row(QStringLiteral("Size"), {}, size)->setUnavailable();

    sec.caps(QStringLiteral("Show"));
    struct Metric {
        QString label, help;
        bool drawn;
    };
    const QList<Metric> metrics = {
        {QStringLiteral("Frame time graph"), QStringLiteral("The last 120 frame times: green to 60 fps, amber to 30, red slower, with a 16.7 ms line."), true},
        {QStringLiteral("Shader stalls"), QStringLiteral("Stall count and the worst frame since the last reset."), false},
        {QStringLiteral("Pipelines"), QStringLiteral("Cached pipelines and compiles in flight."), false},
        {QStringLiteral("VRAM and free RAM"), QStringLiteral("Free RAM turns amber below 2 GB."), false},
    };
    for (const Metric& m : metrics) {
        auto* t = new Toggle;
        t->setChecked(m.drawn);
        sec.row(m.label, m.help, t)->setUnavailable(m.drawn ? QStringLiteral("Always drawn. Choosing what the overlay shows is not in this XSEmu Core build yet.")
                                                            : QStringLiteral("The game bar shows this; the in-frame overlay doesn't yet."));
    }
    QObject::connect(&s.settings, &AppSettings::changed, sec.page, [&s, on](const QString& key) {
        if (key == QLatin1String(AppSettings::Key::OverlayDefault))
            on->setChecked(s.settings.overlayDefault());
    });
    sec.finish();
    return sec.page;
}
