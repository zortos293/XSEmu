#include "app/DevLever.h"
#include "app/Services.h"
#include "shell/MainWindow.h"
#include "theme/Theme.h"
#include "widgets/Icon.h"

#include <QApplication>
#include <QTextStream>
#include <QTimer>

int main(int argc, char** argv)
{
    // The game host is a native widget; its siblings stay alien so only the host gets a window.
    QApplication::setAttribute(Qt::AA_DontCreateNativeWidgetSiblings);
    QApplication app(argc, argv);
    QApplication::setApplicationName(QStringLiteral("XSEmuLauncher"));
    QApplication::setOrganizationName(QStringLiteral("XSEmu"));
    QApplication::setApplicationDisplayName(QStringLiteral("XSEmu Launcher"));
    QApplication::setApplicationVersion(QStringLiteral(XSEMU_VERSION));

#ifdef XSEMU_DEV_TOOLS
    DevLever::Options lever;
    QString leverError;
    const bool useLever = DevLever::parse(QApplication::arguments(), &lever, &leverError);
    if (!leverError.isEmpty()) {
        QTextStream(stderr) << leverError << Qt::endl;
        return 2;
    }
    if (lever.listStates) {
        QTextStream(stdout) << DevLever::states().join(QLatin1Char('\n')) << Qt::endl;
        return 0;
    }
    if (useLever && !DevLever::prepare(&leverError)) {
        QTextStream(stderr) << leverError << Qt::endl;
        return 2;
    }
#endif

    Theme::instance().install(app);
    QApplication::setWindowIcon(Icon::appIcon());
    Services services;
    MainWindow window(services);

#ifdef XSEMU_DEV_TOOLS
    if (useLever) {
        DevLever::apply(lever, services, window);
        return QApplication::exec();
    }
#endif

    window.resize(Metrics::WindowWidth, 800);
    window.show();

    // The emulator answered the control pipe with the new Core: the copy kept in core.old is no longer
    // needed. An update found a few seconds in, so the library is usable at once, and nothing is ever
    // installed without the user asking for it.
    QObject::connect(&services.emulator, &EmulatorController::helloReceived, &services.updates, &UpdateService::noteEmulatorStarted);
    if (services.settings.checkForUpdates() && services.settings.firstLaunchDone())
        QTimer::singleShot(6000, &services.updates, [&services] { services.updates.check(false); });
    // The launcher's files are staged and its updater script runs: quit so it can replace them.
    QObject::connect(&services.updates, &UpdateService::launcherUpdateReady, &app, &QCoreApplication::quit);
    return QApplication::exec();
}
