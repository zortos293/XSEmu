#include "EmulatorBuild.h"

#include "ProcessTree.h"

#include <QDateTime>
#include <QDir>
#include <QElapsedTimer>
#include <QFileInfo>
#include <QHash>
#include <QMutex>
#include <QProcess>
#include <QScopeGuard>

#include <optional>

#ifdef Q_OS_WIN
#include <windows.h>
#endif

namespace {

QString keyFor(const QFileInfo& fi)
{
    return QStringLiteral("%1|%2|%3").arg(fi.absoluteFilePath()).arg(fi.size()).arg(fi.lastModified().toMSecsSinceEpoch());
}

QMutex answersLock; // probes may run on a worker thread

QHash<QString, EmulatorBuild>& answers()
{
    static QHash<QString, EmulatorBuild> kept;
    return kept;
}

// The answer, or nullopt when abandon() turned true before or while it ran.
std::optional<QString> ask(const QString& path, const QString& option, const std::function<bool()>& abandon)
{
    if (abandon && abandon())
        return std::nullopt;
    QProcess p;
#ifdef Q_OS_WIN
    p.setCreateProcessArgumentsModifier([](QProcess::CreateProcessArguments* args) { args->flags |= CREATE_NO_WINDOW; });
#endif
    p.setProcessChannelMode(QProcess::MergedChannels);
    p.setStandardInputFile(QProcess::nullDevice());
    const qint64 pid = startHelper(p, path, {option});
    const auto forget = qScopeGuard([pid] { forgetHelper(pid); });
    if (!p.waitForStarted(3000))
        return QString();
    QElapsedTimer clock;
    clock.start();
    while (!p.waitForFinished(50)) {
        const bool abandoned = abandon && abandon();
        if (!abandoned && clock.elapsed() < 3000)
            continue;
        p.kill();
        p.waitForFinished(1000);
        return abandoned ? std::nullopt : std::optional<QString>(QString());
    }
    return QString::fromLocal8Bit(p.readAll());
}

}

QString EmulatorBuild::problem() const
{
    if (!found)
        return QStringLiteral("XSEmu Core was not found at %1.").arg(QDir::toNativeSeparators(path));
    if (capable)
        return {};
    if (version.isEmpty())
        return QStringLiteral("%1 didn't say which version it is, so the launcher won't start it.").arg(QFileInfo(path).fileName());
    return QStringLiteral("This XSEmu Core build is too old for the launcher (%1 says \"%2\"). It needs a build that supports --control-pipe and --progress-json.")
        .arg(QFileInfo(path).fileName(), version);
}

bool EmulatorBuild::speaksProtocol(const QString& versionLine, const QString& help)
{
    return versionLine.startsWith(QLatin1String("XSEmu ")) && help.contains(QLatin1String("--control-pipe"))
        && help.contains(QLatin1String("--progress-json"));
}

EmulatorBuild EmulatorBuild::cached(const QString& path)
{
    const QFileInfo fi(path);
    EmulatorBuild b;
    b.path = path;
    if (!fi.isFile() || !fi.isExecutable())
        return b;
    const QMutexLocker lock(&answersLock);
    return answers().value(keyFor(fi), b);
}

EmulatorBuild EmulatorBuild::probe(const QString& path, const std::function<bool()>& abandon)
{
    const QFileInfo fi(path);
    EmulatorBuild b;
    b.path = path;
    if (!fi.isFile() || !fi.isExecutable())
        return b;
    const QString key = keyFor(fi);
    {
        const QMutexLocker lock(&answersLock);
        if (const auto it = answers().constFind(key); it != answers().constEnd())
            return *it;
    }
    const std::optional<QString> version = ask(path, QStringLiteral("--version"), abandon);
    if (!version)
        return b;
    b.version = version->trimmed().section(QLatin1Char('\n'), 0, 0).trimmed();
    if (b.version.startsWith(QLatin1String("XSEmu "))) {
        const std::optional<QString> help = ask(path, QStringLiteral("--help"), abandon);
        if (!help)
            return b;
        b.capable = speaksProtocol(b.version, *help);
    }
    b.found = true;
    const QMutexLocker lock(&answersLock);
    answers().insert(key, b);
    return b;
}
