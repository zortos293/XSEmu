#include "FakeWarmup.h"

#include <QCoreApplication>
#include <QDir>
#include <QFile>
#include <QJsonDocument>
#include <QJsonObject>
#include <QRandomGenerator>

#include <cstdio>
#include <iostream>
#include <string>
#include <thread>

namespace {
// The stdin reader thread may sit in a blocking read holding stdin's lock, so leave without the C
// runtime's stream cleanup (which would wait for that lock).
[[noreturn]] void leave(int code)
{
    std::fflush(stdout);
    std::fflush(stderr);
    std::_Exit(code);
}

int envInt(const char* name, int fallback)
{
    bool ok = false;
    const int v = qEnvironmentVariableIntValue(name, &ok);
    return ok ? v : fallback;
}
}

FakeWarmup::FakeWarmup(bool full, bool json, QString scenario, QObject* parent)
    : QObject(parent)
    , m_full(full)
    , m_json(json)
    , m_scenario(std::move(scenario))
{
    // what earlier runs built stays built, so running again goes on from there (docs/launcher-protocol.md)
    const QString cache = qEnvironmentVariable("XCL_DXIL_CACHE");
    if (!cache.isEmpty() && QDir(cache).exists()) {
        m_stateFile = QDir(cache).filePath(QStringLiteral("fake-warmup-built"));
        QFile f(m_stateFile);
        if (f.open(QIODevice::ReadOnly))
            m_boundBefore = f.readAll().trimmed().toLongLong();
    }
    const qint64 bound = qMax<qint64>(0, (full ? 4210 : 3120) - m_boundBefore);
    m_phases = {{"starting", 0}, {"giants", 6}, {"bound", bound}, {"compute", 412}, {"compute_blobs", 188}, {"rest", 96}};
    m_timer.setInterval(envInt("FAKE_XSEMU_TICK_MS", 50));
    connect(&m_timer, &QTimer::timeout, this, &FakeWarmup::tick);
}

void FakeWarmup::emitLine(const QJsonObject& o)
{
    if (!m_json)
        return;
    const QByteArray line = QJsonDocument(o).toJson(QJsonDocument::Compact) + '\n';
    std::fwrite(line.constData(), 1, size_t(line.size()), stdout);
    std::fflush(stdout);
}

void FakeWarmup::start()
{
    m_clock.start();
    m_phaseClock.start();
    // stdin carries {"cmd":"cancel"}; EOF counts as cancel
    std::thread([this] {
        std::string line;
        while (std::getline(std::cin, line))
            QMetaObject::invokeMethod(this, [this, l = QByteArray::fromStdString(line)] { onStdinLine(l); }, Qt::QueuedConnection);
        QMetaObject::invokeMethod(this, [this] { onStdinLine(QByteArrayLiteral("{\"cmd\":\"cancel\"}")); }, Qt::QueuedConnection);
    }).detach();

    if (m_scenario == QLatin1String("warmup_refuse")) {
        emitLine({{QStringLiteral("ev"), QStringLiteral("error")}, {QStringLiteral("message"), QStringLiteral("another XSEmu process is running (XSEmu.exe 1234)")}});
        std::fprintf(stderr, "xcl warm-up: refusing to start – another XCL process is running\n");
        leave(3);
    }
    const qint64 targets = m_phases[2].total + m_phases[5].total;
    emitLine({{QStringLiteral("ev"), QStringLiteral("warmup_start")}, {QStringLiteral("total"), targets},
              {QStringLiteral("full"), m_full}, {QStringLiteral("threads"), 16}});
    std::fprintf(stderr, "fake-xsemu warm-up: %lld pipelines, scenario %s\n", static_cast<long long>(targets), qPrintable(m_scenario));
    m_timer.start();
}

void FakeWarmup::onStdinLine(const QByteArray& line)
{
    if (QJsonDocument::fromJson(line).object().value(QStringLiteral("cmd")).toString() == QLatin1String("cancel"))
        m_cancel = true;
}

void FakeWarmup::tick()
{
    Phase& ph = m_phases[m_phase];
    const bool bound = qstrcmp(ph.id, "bound") == 0;

    if (m_cancel) {
        finish(true);
        return;
    }
    if (m_scenario == QLatin1String("warmup_ram_floor") && bound && m_done >= ph.total * 6 / 10 && m_ramTicks < envInt("FAKE_XSEMU_RAM_HOLD_TICKS", 300)) {
        m_holdForRam = true;
        ++m_ramTicks;
    } else {
        m_holdForRam = false;
        const qint64 step = ph.total <= 10 ? 1 : qMax<qint64>(1, ph.total / 40);
        const qint64 add = qMin(step, ph.total - m_done);
        m_done += add;
        m_built += add;
        if (bound)
            m_boundBuilt += add;
        if (bound && QRandomGenerator::global()->bounded(8) == 0)
            ++m_failed;
    }
    if (m_scenario == QLatin1String("warmup_fail") && bound && m_done >= ph.total * 7 / 10) {
        std::fprintf(stderr, "warm-up: the driver compile of DXIL 3f2a91c07d1e4b55 crashed (0xC0000005)\n");
        saveBuilt();
        leave(3);
    }

    const double secs = qMax(0.05, m_phaseClock.elapsed() / 1000.0);
    const double rate = m_done / secs;
    // No fixed cap: a giant compile (about 1.4 GB in the driver) is admitted while it fits in free
    // RAM above the 1 GB floor, so the number that can run at once follows free memory.
    const int ramFree = m_holdForRam ? 1010 : 12288 - int(m_done % 700);
    const int giantsFit = qMax(0, (ramFree - 1024) / 1434);
    QJsonObject o{{QStringLiteral("ev"), QStringLiteral("warmup_progress")},
                  {QStringLiteral("done"), m_done},
                  {QStringLiteral("total"), ph.total},
                  {QStringLiteral("keys_per_s"), qRound(rate * 10) / 10.0},
                  {QStringLiteral("ram_free_mb"), ramFree},
                  {QStringLiteral("giants_compiling"), qstrcmp(ph.id, "giants") == 0 ? qMin(2, giantsFit) : 0},
                  {QStringLiteral("in_flight"), m_holdForRam ? 0 : 5},
                  {QStringLiteral("failed"), m_failed},
                  {QStringLiteral("phase"), QLatin1String(ph.id)},
                  {QStringLiteral("paused_for_ram"), m_holdForRam}};
    if (rate > 0 && ph.total > m_done)
        o.insert(QStringLiteral("eta_s"), qint64((ph.total - m_done) / rate));
    emitLine(o);

    if (m_done >= ph.total) {
        if (m_phase + 1 >= m_phases.size()) {
            if (m_scenario == QLatin1String("warmup_partial"))
                m_failed += 5; // a pass that went through everything with builds it could not make
            finish(false);
            return;
        }
        ++m_phase;
        m_done = 0;
        m_phaseClock.restart();
    }
}

void FakeWarmup::saveBuilt() const
{
    if (m_stateFile.isEmpty())
        return;
    QFile f(m_stateFile);
    if (f.open(QIODevice::WriteOnly | QIODevice::Truncate))
        f.write(QByteArray::number(m_boundBefore + m_boundBuilt));
}

void FakeWarmup::finish(bool cancelled)
{
    m_timer.stop();
    saveBuilt();
    const bool complete = !cancelled;
    // what this pass did not build, as the emulator counts it: the phases it never reached and the
    // builds it failed, which the next pass goes on with
    qint64 of = 0;
    for (const Phase& ph : m_phases)
        of += ph.total;
    const qint64 left = qMax<qint64>(0, of - (m_built - m_failed));
    const bool partial = m_scenario == QLatin1String("warmup_partial");
    emitLine({{QStringLiteral("ev"), QStringLiteral("warmup_done")},
              {QStringLiteral("built"), m_built - m_failed},
              {QStringLiteral("failed"), m_failed},
              {QStringLiteral("left"), left},
              {QStringLiteral("too_big"), 0},
              {QStringLiteral("took_s"), qRound(m_clock.elapsed() / 100.0) / 10.0},
              {QStringLiteral("complete"), complete},
              {QStringLiteral("cancelled"), cancelled},
              {QStringLiteral("code"), cancelled ? 4 : partial ? 1 : 0}});
    leave(cancelled ? 4 : partial ? 1 : 0);
}
