// fake-xsemu: the XSEmu.exe command line of docs/launcher-protocol.md for launcher development.
//   fake-xsemu run [--host-hwnd <hex>] [--control-pipe <name>] <game folder> [-- args]
//   fake-xsemu warm-up [--full] [--progress-json]
// FAKE_XSEMU_SCENARIO = ok | crash | device_removed | files_missing | slow_boot | warmup_fail |
//                       warmup_partial | warmup_ram_floor | warmup_refuse
// FAKE_XSEMU_VERSION_DELAY_MS = how long --version takes to answer (default 0)
// FAKE_XSEMU_ENV_DUMP = a folder: run and warm-up write their XCL_* environment, sorted, to <folder>/<run|warm-up>.env
#include "FakeRun.h"
#include "FakeRunner.h"
#include "FakeWarmup.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QGuiApplication>
#include <QProcessEnvironment>
#include <QRegularExpression>
#include <QStringList>

#include <chrono>
#include <cstdio>
#include <memory>
#include <thread>

static int usage(FILE* to = stderr)
{
    std::fprintf(to, "usage: fake-xsemu run [--host-hwnd <hex>] [--control-pipe <name>] <game folder> [-- args]\n"
                     "       fake-xsemu warm-up [--full] [--progress-json]\n");
    return to == stderr ? 2 : 0;
}

static void dumpEnvironment(const QString& mode)
{
    const QString dir = qEnvironmentVariable("FAKE_XSEMU_ENV_DUMP");
    if (dir.isEmpty())
        return;
    QStringList lines = QProcessEnvironment::systemEnvironment().toStringList().filter(QRegularExpression(QStringLiteral("^XCL_")));
    lines.sort();
    QFile f(QDir(dir).filePath(mode + QStringLiteral(".env")));
    if (f.open(QIODevice::WriteOnly | QIODevice::Truncate))
        f.write(lines.join(QLatin1Char('\n')).toUtf8());
}

int main(int argc, char** argv)
{
    QStringList args;
    for (int i = 1; i < argc; ++i)
        args << QString::fromLocal8Bit(argv[i]);
    const QString scenario = qEnvironmentVariable("FAKE_XSEMU_SCENARIO", QStringLiteral("ok"));
    if (const QString runner = FakeRunner::kind(QString::fromLocal8Bit(argv[0]), args); !runner.isEmpty())
        return FakeRunner::run(runner, args, argc, argv);
    if (args.isEmpty())
        return usage();
    if (args.first() == QLatin1String("--version")) {
        std::this_thread::sleep_for(std::chrono::milliseconds(qEnvironmentVariableIntValue("FAKE_XSEMU_VERSION_DELAY_MS")));
        std::printf("XSEmu 0.0.0-fake\n");
        return 0;
    }
    if (args.first() == QLatin1String("--help"))
        return usage(stdout);

    if (args.first() == QLatin1String("warm-up")) {
        dumpEnvironment(args.first());
        QCoreApplication app(argc, argv);
        FakeWarmup warmup(args.contains(QStringLiteral("--full")), args.contains(QStringLiteral("--progress-json")), scenario);
        QMetaObject::invokeMethod(&warmup, &FakeWarmup::start, Qt::QueuedConnection);
        return app.exec();
    }
    if (args.first() != QLatin1String("run"))
        return usage();

    FakeRun::Options o;
    o.scenario = scenario;
    int i = 1;
    for (; i < args.size(); ++i) {
        const QString& a = args[i];
        if (a == QLatin1String("--host-hwnd") && i + 1 < args.size()) {
            QString hex = args[++i];
            if (hex.startsWith(QLatin1String("0x"), Qt::CaseInsensitive))
                hex = hex.mid(2);
            o.hostHwnd = static_cast<quintptr>(hex.toULongLong(nullptr, 16));
        } else if (a == QLatin1String("--control-pipe") && i + 1 < args.size()) {
            o.pipe = args[++i];
        } else {
            break;
        }
    }
    if (i >= args.size())
        return usage();
    o.folder = args[i];
    dumpEnvironment(args.first());

    QGuiApplication app(argc, argv);
    app.setQuitOnLastWindowClosed(true);
    FakeRun run(o);
    QMetaObject::invokeMethod(&run, &FakeRun::start, Qt::QueuedConnection);
    return app.exec();
}
