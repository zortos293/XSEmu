#include "ProcessGate.h"

#include "ProcessTree.h"

#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QPointer>
#include <QThreadPool>

ProcessGate::External ProcessGate::externalFor(const QString& marker)
{
    External e;
    e.processes = runningEmulatorProcesses(QCoreApplication::applicationPid());
    if (!marker.isEmpty() && QFileInfo::exists(marker))
        e.marker = marker;
    return e;
}

ProcessGate::ProcessGate(QObject* parent)
    : QObject(parent)
{
    m_poll.setInterval(2000);
    connect(&m_poll, &QTimer::timeout, this, &ProcessGate::poll);
    m_poll.start();
    poll();
}

void ProcessGate::poll()
{
    if (m_polling)
        return;
    m_polling = true;
    const QPointer<ProcessGate> self = this;
    QThreadPool::globalInstance()->start([self, marker = m_marker] {
        const External seen = externalFor(marker);
        QMetaObject::invokeMethod(qApp, [self, marker, seen] {
            if (!self)
                return;
            self->m_polling = false;
            if (marker != self->m_marker) {
                self->poll();
                return;
            }
            const bool differs = seen.reason() != self->m_external.reason();
            self->m_external = seen;
            if (differs)
                emit self->changed();
        });
    });
}

QString ProcessGate::External::reason() const
{
    if (!processes.isEmpty())
        return QStringLiteral("Emulator busy: %1 is running. Only one emulator runs at a time.").arg(processes.join(QStringLiteral(", ")));
    if (!marker.isEmpty())
        return QStringLiteral("Emulator busy: %1 exists, so another job holds the emulator.").arg(QDir::toNativeSeparators(marker));
    return {};
}

ProcessGate::External ProcessGate::external() const
{
    return externalFor(m_marker);
}

void ProcessGate::setBusyMarker(const QString& path)
{
    if (path == m_marker)
        return;
    m_marker = path;
    poll();
    emit changed();
}

bool ProcessGate::acquire(Owner owner, const QString& title, std::function<void()> stop)
{
    if (m_owner != Owner::None || owner == Owner::None)
        return false;
    m_owner = owner;
    m_title = title;
    m_paused = false;
    m_stop = std::move(stop);
    emit changed();
    return true;
}

void ProcessGate::release(Owner owner)
{
    if (m_owner != owner)
        return;
    m_owner = Owner::None;
    m_title.clear();
    m_paused = false;
    m_stop = nullptr;
    emit changed();
}

void ProcessGate::setPaused(bool paused)
{
    if (m_paused == paused)
        return;
    m_paused = paused;
    emit changed();
}

QString ProcessGate::blockedReason(Action action) const
{
    const QString what = m_title.isEmpty() ? QStringLiteral("A game") : m_title;
    switch (m_owner) {
    case Owner::None:
        return {};
    case Owner::Game: {
        const QString state = m_paused ? QStringLiteral("is paused") : QStringLiteral("is running");
        if (action == Action::StartWarmup)
            return QStringLiteral("%1 %2. Stop it to warm up shaders, because only one XSEmu Core process runs at a time.").arg(what, state);
        return QStringLiteral("%1 %2. Stop it to play another game, because only one XSEmu Core process runs at a time.").arg(what, state);
    }
    case Owner::Warmup:
        if (action == Action::StartWarmup)
            return QStringLiteral("A warm-up is already running.");
        return QStringLiteral("A shader warm-up is running. Cancel it or wait for it to finish, because only one XSEmu Core process runs at a time.");
    case Owner::Update:
        return QStringLiteral("%1 is running. Wait for it to finish or stop it, because it replaces the files a game would start from.").arg(what);
    }
    return {};
}

QString ProcessGate::stopActionLabel() const
{
    switch (m_owner) {
    case Owner::Game: return QStringLiteral("Stop game");
    case Owner::Warmup: return QStringLiteral("Cancel warm-up");
    case Owner::Update: return QStringLiteral("Stop update");
    case Owner::None: break;
    }
    return {};
}

void ProcessGate::stopOwner()
{
    if (m_stop)
        m_stop();
}
