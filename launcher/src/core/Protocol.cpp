#include "Protocol.h"

#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonParseError>

namespace Protocol {

namespace {

std::optional<double> optDouble(const QJsonObject& o, const char* k)
{
    const QJsonValue v = o.value(QLatin1String(k));
    if (!v.isDouble())
        return std::nullopt;
    return v.toDouble();
}

std::optional<qint64> optInt(const QJsonObject& o, const char* k)
{
    const QJsonValue v = o.value(QLatin1String(k));
    if (!v.isDouble())
        return std::nullopt;
    return static_cast<qint64>(v.toDouble());
}

qint64 toInt(const QJsonObject& o, const char* k)
{
    return optInt(o, k).value_or(0);
}

QString str(const QJsonObject& o, const char* k)
{
    return o.value(QLatin1String(k)).toString();
}

bool flag(const QJsonObject& o, const char* k)
{
    return o.value(QLatin1String(k)).toBool();
}

QByteArray line(const QJsonObject& o)
{
    return QJsonDocument(o).toJson(QJsonDocument::Compact) + '\n';
}

QByteArray cmd(const char* name)
{
    return line(QJsonObject{{QStringLiteral("cmd"), QLatin1String(name)}});
}

}

GameState gameStateFromString(const QString& s)
{
    if (s == QLatin1String("booting")) return GameState::Booting;
    if (s == QLatin1String("running")) return GameState::Running;
    if (s == QLatin1String("paused")) return GameState::Paused;
    if (s == QLatin1String("stopping")) return GameState::Stopping;
    return GameState::Unknown;
}

QString toString(GameState s)
{
    switch (s) {
    case GameState::Booting: return QStringLiteral("booting");
    case GameState::Running: return QStringLiteral("running");
    case GameState::Paused: return QStringLiteral("paused");
    case GameState::Stopping: return QStringLiteral("stopping");
    case GameState::Unknown: break;
    }
    return QStringLiteral("unknown");
}

ExitReason exitReasonFromString(const QString& s)
{
    if (s == QLatin1String("stopped")) return ExitReason::Stopped;
    if (s == QLatin1String("title_exit")) return ExitReason::TitleExit;
    if (s == QLatin1String("crash")) return ExitReason::Crash;
    if (s == QLatin1String("device_removed")) return ExitReason::DeviceRemoved;
    if (s == QLatin1String("error")) return ExitReason::Error;
    return ExitReason::Unknown;
}

QString toString(ExitReason r)
{
    switch (r) {
    case ExitReason::Stopped: return QStringLiteral("stopped");
    case ExitReason::TitleExit: return QStringLiteral("title_exit");
    case ExitReason::Crash: return QStringLiteral("crash");
    case ExitReason::DeviceRemoved: return QStringLiteral("device_removed");
    case ExitReason::Error: return QStringLiteral("error");
    case ExitReason::Unknown: break;
    }
    return QStringLiteral("unknown");
}

Event parse(const QByteArray& raw)
{
    const QByteArray trimmed = raw.trimmed();
    if (trimmed.isEmpty())
        return Invalid{QStringLiteral("empty line")};
    QJsonParseError err{};
    const QJsonDocument doc = QJsonDocument::fromJson(trimmed, &err);
    if (err.error != QJsonParseError::NoError || !doc.isObject())
        return Invalid{err.error != QJsonParseError::NoError ? err.errorString() : QStringLiteral("not a JSON object")};
    const QJsonObject o = doc.object();
    const QString ev = str(o, "ev");
    if (ev.isEmpty())
        return Invalid{QStringLiteral("no \"ev\" field")};

    if (ev == QLatin1String("hello"))
        return Hello{static_cast<int>(toInt(o, "proto")), toInt(o, "pid"), str(o, "version"), str(o, "title"), str(o, "device_type"), str(o, "hwnd")};
    if (ev == QLatin1String("state")) {
        const QString s = str(o, "state");
        return State{gameStateFromString(s), s};
    }
    if (ev == QLatin1String("stats")) {
        Stats s;
        s.fps = optDouble(o, "fps");
        s.frameMs = optDouble(o, "frame_ms");
        s.ramFreeMb = optInt(o, "ram_free_mb");
        s.vramUsedMb = optInt(o, "vram_used_mb");
        s.vramBudgetMb = optInt(o, "vram_budget_mb");
        s.stalls = optInt(o, "stalls");
        s.worstMs = optDouble(o, "worst_ms");
        s.pipelines = optInt(o, "pipelines");
        s.compiling = optInt(o, "compiling");
        return s;
    }
    if (ev == QLatin1String("key"))
        return Key{str(o, "key")};
    if (ev == QLatin1String("exit")) {
        const QString r = str(o, "reason");
        return Exit{static_cast<int>(toInt(o, "code")), exitReasonFromString(r), r, str(o, "message")};
    }
    if (ev == QLatin1String("warmup_start"))
        return WarmupStart{toInt(o, "total"), flag(o, "full"), static_cast<int>(toInt(o, "threads"))};
    if (ev == QLatin1String("warmup_progress")) {
        WarmupProgress p;
        p.done = toInt(o, "done");
        p.total = toInt(o, "total");
        p.keysPerS = optDouble(o, "keys_per_s").value_or(0);
        p.etaS = optInt(o, "eta_s");
        p.ramFreeMb = optInt(o, "ram_free_mb");
        p.giantsCompiling = static_cast<int>(toInt(o, "giants_compiling"));
        p.inFlight = static_cast<int>(toInt(o, "in_flight"));
        p.failed = toInt(o, "failed");
        p.phase = str(o, "phase");
        p.pausedForRam = flag(o, "paused_for_ram");
        return p;
    }
    if (ev == QLatin1String("warmup_done")) {
        WarmupDone d;
        d.built = toInt(o, "built");
        d.failed = toInt(o, "failed");
        d.left = optInt(o, "left").value_or(-1);
        d.tooBig = toInt(o, "too_big");
        d.tookS = optDouble(o, "took_s").value_or(0);
        d.complete = flag(o, "complete");
        d.cancelled = flag(o, "cancelled");
        d.code = static_cast<int>(toInt(o, "code"));
        return d;
    }
    if (ev == QLatin1String("error"))
        return Error{str(o, "message")};
    return Unknown{ev};
}

namespace Command {
QByteArray pause() { return cmd("pause"); }
QByteArray resume() { return cmd("resume"); }
QByteArray stop() { return cmd("stop"); }
QByteArray focus() { return cmd("focus"); }
QByteArray cancel() { return cmd("cancel"); }
QByteArray overlay(bool on)
{
    return line(QJsonObject{{QStringLiteral("cmd"), QStringLiteral("overlay")}, {QStringLiteral("on"), on}});
}
}

}
