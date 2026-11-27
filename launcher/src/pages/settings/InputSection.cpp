#include "Sections.h"
#include "SettingsKit.h"

#include "app/Services.h"
#include "core/LaunchConfig.h"
#include "widgets/SegmentedControl.h"
#include "widgets/Text.h"
#include "widgets/Toggle.h"

#include <QTimer>

#ifdef Q_OS_WIN
#include <windows.h>
#endif

namespace {

#ifdef Q_OS_WIN
// The XInput slots with a controller in them (XSEmu reads the first one, hle/gameinput.rs). Empty
// when xinput1_4.dll is missing.
QList<int> connectedControllers()
{
    QList<int> pads;
    using GetState = DWORD(WINAPI*)(DWORD, void*);
    static const GetState getState = [] {
        HMODULE h = LoadLibraryW(L"xinput1_4.dll");
        return h ? reinterpret_cast<GetState>(reinterpret_cast<void*>(GetProcAddress(h, "XInputGetState"))) : nullptr;
    }();
    if (!getState)
        return pads;
    for (DWORD i = 0; i < 4; ++i) {
        unsigned char state[16] = {};
        if (getState(i, state) == ERROR_SUCCESS)
            pads << int(i) + 1;
    }
    return pads;
}
#endif

const QList<double> kSensitivities{0.5, 0.75, 1.0, 1.5, 2.0, 3.0};

}

// keyboard_mouse and mouse_sensitivity: launcher defaults written into each title's xcl.toml at launch
// (LaunchConfig). The keys are the in-game shortcuts the emulator forwards ({"ev":"key"}) and the
// in-game view handles.
QWidget* SettingsSections::input(Services& s)
{
    const SettingsKit::Section sec = SettingsKit::makeSection(
        QStringLiteral("Input"), QStringLiteral("Keyboard, mouse and controller. In fullscreen, Esc leaves fullscreen; in a window it goes to the game."));
    sec.caps(QStringLiteral("Launcher keys"));
    sec.row(QStringLiteral("Pause and open the menu"), {}, SettingsKit::keyCap(QStringLiteral("F1")));
    sec.row(QStringLiteral("Performance overlay"), {}, SettingsKit::keyCap(QStringLiteral("F3")));
    sec.row(QStringLiteral("Fullscreen"), {}, SettingsKit::keyCap(QStringLiteral("F11")));
    sec.row(QStringLiteral("Leave fullscreen"), QStringLiteral("In a window, Esc goes to the game."), SettingsKit::keyCap(QStringLiteral("Esc")));

    sec.caps(QStringLiteral("Controller"));
    auto* pad = SettingsKit::status({});
    SettingsKit::Row* padRow = sec.row(QStringLiteral("Controller"), {}, pad);
    const auto pollPads = [pad, padRow] {
#ifdef Q_OS_WIN
        const QList<int> pads = connectedControllers();
        pad->setText(pads.isEmpty() ? QStringLiteral("Not connected") : QStringLiteral("Connected"));
        pad->setRole(pads.isEmpty() ? Theme::Role::Text3 : Theme::Role::Success);
        padRow->setHelp(pads.isEmpty() ? QStringLiteral("Connect an Xbox or other XInput controller. XSEmu Core uses the first one as the console's gamepad.")
                                        : QStringLiteral("Connected over XInput, slot %1. Used as the console's first gamepad.").arg(pads.first()));
#else
        pad->setText(QStringLiteral("Windows only"));
        pad->setRole(Theme::Role::Text3);
        padRow->setHelp(QStringLiteral("XSEmu Core reads the first XInput controller of the PC as the console's gamepad."));
#endif
    };
    pollPads();
    auto* timer = new QTimer(sec.page);
    timer->setInterval(2000);
    QObject::connect(timer, &QTimer::timeout, sec.page, pollPads);
    timer->start();

    auto* kbm = new Toggle;
    const auto kbmValue = [&s] { return LaunchConfig::typed(QStringLiteral("keyboard_mouse"), s.settings.xclDefault(QStringLiteral("keyboard_mouse"))); };
    kbm->setChecked(kbmValue().isValid() ? kbmValue().toBool() : true);
    QObject::connect(kbm, &Toggle::toggled, sec.page, [&s](bool on) { s.settings.setXclDefault(QStringLiteral("keyboard_mouse"), on); });
    sec.row(QStringLiteral("Keyboard and mouse as gamepad"),
            QStringLiteral("WASD the left stick, the mouse the right stick, the mouse buttons the triggers, the wheel the bumpers. XSEmu Core holds the cursor in the game window while this is on."),
            kbm);

    QStringList labels;
    for (double v : kSensitivities)
        labels << QStringLiteral("%1×").arg(v, 0, 'g', 3);
    auto* sens = new SegmentedControl(labels);
    const QVariant stored = LaunchConfig::typed(QStringLiteral("mouse_sensitivity"), s.settings.xclDefault(QStringLiteral("mouse_sensitivity")));
    const int at = kSensitivities.indexOf(stored.isValid() ? stored.toDouble() : 1.0);
    sens->setCurrentIndex(at >= 0 ? int(at) : 2);
    QObject::connect(sens, &SegmentedControl::currentChanged, sec.page, [&s](int i) {
        s.settings.setXclDefault(QStringLiteral("mouse_sensitivity"), kSensitivities.value(i, 1.0));
    });
    sec.row(QStringLiteral("Mouse sensitivity"), QStringLiteral("1× is full right-stick deflection at 4,000 mouse counts a second."), sens);

    sec.caps(QStringLiteral("Focus"));
    auto* pauseOnFocus = new Toggle;
    sec.row(QStringLiteral("Pause when XSEmu Core loses focus"), QStringLiteral("Stops the game when you switch windows."), pauseOnFocus)
        ->setUnavailable(QStringLiteral("Not in this launcher build yet."));
    sec.finish();
    return sec.page;
}
