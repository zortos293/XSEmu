#include "Sections.h"
#include "SettingsKit.h"

#include "app/Services.h"
#include "core/EmulatorBuild.h"
#include "core/Paths.h"
#include "pages/ShareSettingsDialog.h"
#include "shell/LaunchErrors.h"
#include "widgets/Button.h"
#include "widgets/Icon.h"
#include "widgets/Text.h"

#include <QClipboard>
#include <QDesktopServices>
#include <QDir>
#include <QFileInfo>
#include <QGuiApplication>
#include <QHBoxLayout>
#include <QPointer>
#include <QThreadPool>
#include <QUrl>
#include <QVBoxLayout>

#include <atomic>

// The launcher's version (XSEMU_VERSION), the emulator's path, its answer to --version and whether the
// launcher may start it (or why not: too old, busy), and the credits and
// licences: XCL by the xcl-dev contributors (GPL-3.0-or-later), the launcher (GPL-3.0-or-later), Qt (LGPL-3.0, linked
// dynamically), and game art from SteamGridDB with credit to its community and a link to it.
QWidget* SettingsSections::about(Services& s)
{
    const SettingsKit::Section sec = SettingsKit::makeSection({}, {});
    auto* brand = new QWidget;
    auto* bh = new QHBoxLayout(brand);
    bh->setContentsMargins(0, 0, 0, 0);
    bh->setSpacing(16);
    bh->addWidget(new BrandMark(56));
    auto* names = new QVBoxLayout;
    names->setSpacing(2);
    names->addStretch(1);
    names->addWidget(new Label(QStringLiteral("XSEmu Launcher"), Label::Style::SectionTitle));
    auto* build = new Label(QStringLiteral("launcher %1 · Qt %2").arg(QStringLiteral(XSEMU_VERSION), QString::fromLatin1(qVersion())), Label::Style::Mono, Theme::Role::Text2);
    build->setFont(Theme::mono(12));
    names->addWidget(build);
    names->addStretch(1);
    bh->addLayout(names, 1);
    sec.body->addWidget(brand);
    sec.body->addSpacing(4);

    sec.caps(QStringLiteral("Version"));
    auto* copy = new Button(QStringLiteral("Copy version"), Button::Variant::Secondary, Button::Size::Small);
    auto* choose = new Button(QStringLiteral("Choose..."), Button::Variant::Secondary, Button::Size::Small);
    choose->setObjectName(QStringLiteral("chooseEmulator"));
    SettingsKit::Row* emu = sec.row(QStringLiteral("Emulator"), {}, choose);
    emu->setObjectName(QStringLiteral("emulatorRow"));
    emu->addControl(copy);
    emu->helpLabel()->setTextInteractionFlags(Qt::TextSelectableByMouse);
    // the path, its --version answer and whether the launcher may start it. The emulator is asked on a
    // worker thread, never while another emulator (or the busy marker) holds the PC, and the question
    // is abandoned as soon as this launcher starts a game or a warm-up.
    auto held = std::make_shared<std::atomic<bool>>(!s.gate.isFree());
    QObject::connect(&s.gate, &ProcessGate::changed, emu, [&s, held] { *held = !s.gate.isFree(); });
    auto refreshFn = std::make_shared<std::function<void()>>();
    const std::weak_ptr<std::function<void()>> again = refreshFn;
    auto asking = std::make_shared<bool>(false);
    *refreshFn = [&s, emu, copy, again, asking, held] {
        const QString exe = s.settings.emulatorPath();
        const ProcessGate::External& other = s.gate.lastExternal();
        const EmulatorBuild answer = EmulatorBuild::cached(exe);
        const bool executable = QFileInfo(exe).isExecutable();
        if (executable && !answer.found && !other.busy() && s.gate.isFree() && !*asking) {
            *asking = true;
            const QPointer<QWidget> row = emu;
            QThreadPool::globalInstance()->start([exe, marker = s.gate.busyMarker(), row, again, asking, held] {
                if (!*held && !ProcessGate::externalFor(marker).busy())
                    EmulatorBuild::probe(exe, [held] { return held->load(); });
                QMetaObject::invokeMethod(qApp, [row, again, asking] {
                    *asking = false;
                    if (const auto f = again.lock(); f && row)
                        (*f)();
                });
            });
        }
        QStringList lines{QDir::toNativeSeparators(exe)};
        if (!executable)
            lines << QStringLiteral("Not found.");
        else if (!answer.found && *asking)
            lines << QStringLiteral("Asking the emulator for its version...");
        else if (!answer.found)
            lines << QStringLiteral("Version not asked yet: the emulator is busy.");
        else
            lines << (answer.version.isEmpty() ? QStringLiteral("No version reported.") : answer.version)
                  << (answer.capable ? QStringLiteral("Ready for the launcher.") : QStringLiteral("This XSEmu Core build is too old for the launcher: Play and Warm-up stay off."));
        if (other.busy())
            lines << other.reason();
        if (AppSettings::emulatorPathFromEnvironment())
            lines << QStringLiteral("Set by XSEMU_EMULATOR in the environment, which wins over Choose.");
        emu->setHelp(lines.join(QLatin1Char('\n')));
        copy->setEnabled(!answer.version.isEmpty());
        copy->setProperty("version", answer.version);
    };
    const auto refresh = [refreshFn] { (*refreshFn)(); };
    refresh();
    QObject::connect(&s.gate, &ProcessGate::changed, emu, refresh);
    QObject::connect(&s.settings, &AppSettings::changed, emu, [refresh](const QString& key) {
        if (key == QLatin1String(AppSettings::Key::EmulatorPath) || key == QLatin1String(AppSettings::Key::BusyMarker))
            refresh();
    });
    QObject::connect(choose, &Button::clicked, choose, [&s, choose] { LaunchErrors::chooseEmulator(choose->window(), s.settings); });
    QObject::connect(copy, &Button::clicked, copy, [copy] {
        QGuiApplication::clipboard()->setText(QStringLiteral("%1, launcher %2").arg(copy->property("version").toString(), QStringLiteral(XSEMU_VERSION)));
    });
    sec.row(QStringLiteral("Launcher"), QStringLiteral("%1. New code under the GNU General Public License v3 or later.").arg(QStringLiteral(XSEMU_VERSION)));
    SettingsShare::addSettingsRows(sec, s);

    sec.caps(QStringLiteral("Credits"));
    auto* credits = new Label(QStringLiteral("XSEmu Core is built on <b>XCL</b>, the Xbox Series compatibility layer by the xcl-dev contributors (GPL-3.0-or-later). "
                                             "The console model, system libraries and shader recompiler come from XCL."),
                              Label::Style::Body);
    credits->setTextFormat(Qt::RichText);
    credits->setWordWrap(true);
    sec.add(credits, 8);
    auto* licences = new Label(QStringLiteral("Like XCL, XSEmu and this launcher are licensed under the GNU General Public License v3 or later. The launcher uses Qt under the "
                                              "GNU Lesser General Public License v3, linked dynamically."),
                               Label::Style::Small, Theme::Role::Text2);
    licences->setWordWrap(true);
    sec.add(licences, 8);
    auto* art = new Label(QStringLiteral("Game art (covers, banners, logos and icons) comes from <b>SteamGridDB</b> and is made by its community; "
                                         "each image's artist is named on the game's details page. XSEmu ships no game art and is not affiliated with "
                                         "SteamGridDB. Art is fetched through XSEmu's art service, or with your own SteamGridDB key when you set one, and "
                                         "cached on this PC."),
                          Label::Style::Small, Theme::Role::Text2);
    art->setTextFormat(Qt::RichText);
    art->setWordWrap(true);
    sec.add(art, 8);
    auto* links = new QWidget;
    auto* lh = new QHBoxLayout(links);
    lh->setContentsMargins(0, 16, 0, 0);
    lh->setSpacing(6);
    const QList<QPair<QString, QString>> urls = {
        {QStringLiteral("GPL-3.0 licence"), QStringLiteral("https://www.gnu.org/licenses/gpl-3.0.html")},
        {QStringLiteral("Qt licence (LGPL-3.0)"), QStringLiteral("https://www.gnu.org/licenses/lgpl-3.0.html")},
        {QStringLiteral("XCL source"), QStringLiteral("https://github.com/ivall/xcl")},
        {QStringLiteral("SteamGridDB"), QStringLiteral("https://www.steamgriddb.com")},
    };
    for (const auto& u : urls) {
        auto* b = new Button(u.first, Button::Variant::Secondary, Button::Size::Small, QStringLiteral("external"));
        const QUrl url(u.second);
        QObject::connect(b, &Button::clicked, b, [url] { QDesktopServices::openUrl(url); });
        lh->addWidget(b);
    }
    auto* logs = new Button(QStringLiteral("Open logs folder"), Button::Variant::Secondary, Button::Size::Small);
    QObject::connect(logs, &Button::clicked, logs, [] { QDesktopServices::openUrl(QUrl::fromLocalFile(Paths::dataRoot())); });
    lh->addWidget(logs);
    lh->addStretch(1);
    sec.add(links);
    sec.finish();
    return sec.page;
}
