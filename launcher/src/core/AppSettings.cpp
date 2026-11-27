#include "AppSettings.h"

#include "Paths.h"

#include <QDir>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

namespace {

QString resultName(WarmupRun::Result r)
{
    switch (r) {
    case WarmupRun::Result::Finished: return QStringLiteral("finished");
    case WarmupRun::Result::Cancelled: return QStringLiteral("cancelled");
    case WarmupRun::Result::StoppedPartWay: return QStringLiteral("stopped");
    case WarmupRun::Result::Failed: return QStringLiteral("failed");
    case WarmupRun::Result::NothingToBuild: return QStringLiteral("nothing");
    }
    return QStringLiteral("failed");
}

WarmupRun::Result resultFrom(const QString& s)
{
    if (s == QLatin1String("finished")) return WarmupRun::Result::Finished;
    if (s == QLatin1String("cancelled")) return WarmupRun::Result::Cancelled;
    if (s == QLatin1String("stopped")) return WarmupRun::Result::StoppedPartWay;
    if (s == QLatin1String("nothing")) return WarmupRun::Result::NothingToBuild;
    return WarmupRun::Result::Failed;
}

}

QVariantMap WarmupRun::toMap() const
{
    return {
        {QStringLiteral("titleId"), titleId},
        {QStringLiteral("titleName"), titleName},
        {QStringLiteral("runnerId"), runnerId},
        {QStringLiteral("done"), done},
        {QStringLiteral("total"), total},
        {QStringLiteral("failed"), failed},
        {QStringLiteral("left"), left},
        {QStringLiteral("tookS"), tookS},
        {QStringLiteral("result"), resultName(result)},
        {QStringLiteral("exitCode"), exitCode},
        {QStringLiteral("full"), full},
        {QStringLiteral("stoppedForRam"), stoppedForRam},
        {QStringLiteral("message"), message},
        {QStringLiteral("profile"), profile},
        {QStringLiteral("finishedAt"), finishedAt.toUTC().toString(Qt::ISODate)},
    };
}

WarmupRun WarmupRun::fromMap(const QVariantMap& m)
{
    WarmupRun r;
    r.titleId = m.value(QStringLiteral("titleId")).toString();
    r.titleName = m.value(QStringLiteral("titleName")).toString();
    r.runnerId = m.value(QStringLiteral("runnerId")).toString();
    r.done = m.value(QStringLiteral("done")).toLongLong();
    r.total = m.value(QStringLiteral("total")).toLongLong();
    r.failed = m.value(QStringLiteral("failed")).toLongLong();
    // a record from before the emulator said what a pass left keeps -1: unknown, not nothing
    r.left = m.contains(QStringLiteral("left")) ? m.value(QStringLiteral("left")).toLongLong() : -1;
    r.tookS = m.value(QStringLiteral("tookS")).toDouble();
    r.result = resultFrom(m.value(QStringLiteral("result")).toString());
    r.exitCode = m.value(QStringLiteral("exitCode")).toInt();
    r.full = m.value(QStringLiteral("full")).toBool();
    r.stoppedForRam = m.value(QStringLiteral("stoppedForRam")).toBool();
    r.message = m.value(QStringLiteral("message")).toString();
    r.profile = m.value(QStringLiteral("profile")).toString();
    r.finishedAt = QDateTime::fromString(m.value(QStringLiteral("finishedAt")).toString(), Qt::ISODate);
    return r;
}

QVariantMap WarmupCheckpoint::toMap() const
{
    return {
        {QStringLiteral("titleId"), titleId},
        {QStringLiteral("titleName"), titleName},
        {QStringLiteral("runnerId"), runnerId},
        {QStringLiteral("profile"), profile},
        {QStringLiteral("full"), full},
        {QStringLiteral("phase"), phase},
        {QStringLiteral("startedAt"), startedAt.toUTC().toString(Qt::ISODate)},
    };
}

WarmupCheckpoint WarmupCheckpoint::fromMap(const QVariantMap& m)
{
    WarmupCheckpoint p;
    p.titleId = m.value(QStringLiteral("titleId")).toString();
    p.titleName = m.value(QStringLiteral("titleName")).toString();
    p.runnerId = m.value(QStringLiteral("runnerId")).toString();
    p.profile = m.value(QStringLiteral("profile")).toString();
    p.full = m.value(QStringLiteral("full")).toBool();
    p.phase = m.value(QStringLiteral("phase")).toString();
    p.startedAt = QDateTime::fromString(m.value(QStringLiteral("startedAt")).toString(), Qt::ISODate);
    return p;
}

AppSettings::AppSettings(QObject* parent)
    : AppSettings(Paths::settingsFile(), parent)
{
}

AppSettings::AppSettings(const QString& iniPath, QObject* parent)
    : QObject(parent)
{
    QDir().mkpath(QFileInfo(iniPath).absolutePath());
    m_settings = std::make_unique<QSettings>(iniPath, QSettings::IniFormat);
}

AppSettings::~AppSettings() = default;

QString AppSettings::fileName() const
{
    return m_settings->fileName();
}

void AppSettings::write(const char* key, const QVariant& value)
{
    const QString k = QLatin1String(key);
    if (m_settings->contains(k) && m_settings->value(k) == value)
        return;
    m_settings->setValue(k, value);
    m_settings->sync();
    emit changed(k);
}

QVariant AppSettings::read(const char* key, const QVariant& fallback) const
{
    return m_settings->value(QLatin1String(key), fallback);
}

QVariantMap AppSettings::readMap(const char* key) const
{
    const QByteArray json = read(key).toString().toUtf8();
    return QJsonDocument::fromJson(json).object().toVariantMap();
}

QString AppSettings::toString(ThemeMode mode)
{
    switch (mode) {
    case ThemeMode::Light: return QStringLiteral("light");
    case ThemeMode::Dark: return QStringLiteral("dark");
    case ThemeMode::System: break;
    }
    return QStringLiteral("system");
}

AppSettings::ThemeMode AppSettings::themeFromString(const QString& s)
{
    if (s == QLatin1String("light")) return ThemeMode::Light;
    if (s == QLatin1String("dark")) return ThemeMode::Dark;
    return ThemeMode::System;
}

AppSettings::ThemeMode AppSettings::theme() const { return themeFromString(read(Key::Theme).toString()); }
void AppSettings::setTheme(ThemeMode mode) { write(Key::Theme, toString(mode)); }

QStringList AppSettings::gameFolders() const { return read(Key::GameFolders).toStringList(); }
void AppSettings::setGameFolders(const QStringList& folders) { write(Key::GameFolders, folders); }

bool AppSettings::firstLaunchDone() const { return read(Key::FirstLaunchDone, false).toBool(); }
void AppSettings::setFirstLaunchDone(bool done) { write(Key::FirstLaunchDone, done); }

QString AppSettings::defaultDeviceType() const { return read(Key::DefaultDeviceType, QStringLiteral("lockhart")).toString(); }
void AppSettings::setDefaultDeviceType(const QString& deviceType) { write(Key::DefaultDeviceType, deviceType); }

bool AppSettings::offline() const { return read(Key::Offline, true).toBool(); }
void AppSettings::setOffline(bool offline) { write(Key::Offline, offline); }

bool AppSettings::overlayDefault() const { return read(Key::OverlayDefault, false).toBool(); }
void AppSettings::setOverlayDefault(bool on) { write(Key::OverlayDefault, on); }

QString AppSettings::steamGridDbKey() const { return read(Key::SteamGridDbKey).toString().trimmed(); }
void AppSettings::setSteamGridDbKey(const QString& key) { write(Key::SteamGridDbKey, key.trimmed()); }
bool AppSettings::artLookups() const { return read(Key::ArtLookups, true).toBool(); }
void AppSettings::setArtLookups(bool on) { write(Key::ArtLookups, on); }

QString AppSettings::defaultRunner() const { return read(Key::DefaultRunner, QStringLiteral("xsemu")).toString(); }
void AppSettings::setDefaultRunner(const QString& id) { write(Key::DefaultRunner, id); }

QString AppSettings::runnerFor(const QString& titleId) const
{
    const QString id = gameOverrides(titleId).value(QStringLiteral("runner")).toString();
    return id.isEmpty() ? defaultRunner() : id;
}

QString AppSettings::runnersFolder() const { return read(Key::RunnersFolder).toString(); }
void AppSettings::setRunnersFolder(const QString& dir) { write(Key::RunnersFolder, dir); }

QHash<QString, QString> AppSettings::runnerLocations() const
{
    QHash<QString, QString> out;
    const QVariantMap all = readMap(Key::RunnerLocations);
    for (auto it = all.begin(); it != all.end(); ++it)
        out.insert(it.key(), it.value().toString());
    return out;
}

void AppSettings::setRunnerLocation(const QString& id, const QString& folder)
{
    QVariantMap all = readMap(Key::RunnerLocations);
    if (folder.isEmpty())
        all.remove(id);
    else
        all.insert(id, folder);
    write(Key::RunnerLocations, QString::fromUtf8(QJsonDocument(QJsonObject::fromVariantMap(all)).toJson(QJsonDocument::Compact)));
}

QHash<QString, bool> AppSettings::runnerOptions(const QString& id) const
{
    QHash<QString, bool> out;
    const QVariantMap all = readMap(Key::RunnerOptions).value(id).toMap();
    for (auto it = all.begin(); it != all.end(); ++it)
        out.insert(it.key(), it.value().toBool());
    return out;
}

void AppSettings::setRunnerOption(const QString& id, const QString& key, bool on)
{
    QVariantMap all = readMap(Key::RunnerOptions);
    QVariantMap mine = all.value(id).toMap();
    mine.insert(key, on);
    all.insert(id, mine);
    write(Key::RunnerOptions, QString::fromUtf8(QJsonDocument(QJsonObject::fromVariantMap(all)).toJson(QJsonDocument::Compact)));
}

bool AppSettings::windowGameBarAutoHide() const { return read(Key::WindowGameBarAutoHide, false).toBool(); }
void AppSettings::setWindowGameBarAutoHide(bool on) { write(Key::WindowGameBarAutoHide, on); }

bool AppSettings::storeOfflineError() const { return read(Key::StoreOfflineError, true).toBool(); }
void AppSettings::setStoreOfflineError(bool on) { write(Key::StoreOfflineError, on); }

bool AppSettings::crashDumps() const { return read(Key::CrashDumps, true).toBool(); }
void AppSettings::setCrashDumps(bool on) { write(Key::CrashDumps, on); }

QString AppSettings::shaderCacheDirOverride() const { return read(Key::ShaderCacheDir).toString().trimmed(); }
void AppSettings::setShaderCacheDirOverride(const QString& dir) { write(Key::ShaderCacheDir, dir.trimmed()); }

QVariant AppSettings::xclDefault(const QString& tomlKey) const
{
    return m_settings->value(QLatin1String(Key::XclDefaultsPrefix) + tomlKey);
}

void AppSettings::setXclDefault(const QString& tomlKey, const QVariant& value)
{
    const QString key = QLatin1String(Key::XclDefaultsPrefix) + tomlKey;
    m_settings->setValue(key, value);
    m_settings->sync();
    emit changed(key);
}

void AppSettings::clearXclDefault(const QString& tomlKey)
{
    const QString key = QLatin1String(Key::XclDefaultsPrefix) + tomlKey;
    m_settings->remove(key);
    m_settings->sync();
    emit changed(key);
}

QString AppSettings::emulatorPathOverride() const { return read(Key::EmulatorPath).toString(); }
void AppSettings::setEmulatorPathOverride(const QString& path) { write(Key::EmulatorPath, path); }

QString AppSettings::emulatorPath() const
{
    if (emulatorPathFromEnvironment())
        return qEnvironmentVariable("XSEMU_EMULATOR").trimmed();
    const QString o = emulatorPathOverride();
    return o.isEmpty() ? Paths::defaultEmulatorPath() : o;
}

bool AppSettings::emulatorPathFromEnvironment()
{
    return !qEnvironmentVariable("XSEMU_EMULATOR").trimmed().isEmpty();
}

QString AppSettings::busyMarker() const
{
    const QString env = qEnvironmentVariable("XSEMU_BUSY_MARKER").trimmed();
    return env.isEmpty() ? read(Key::BusyMarker).toString().trimmed() : env;
}

void AppSettings::setBusyMarker(const QString& path) { write(Key::BusyMarker, path); }

int AppSettings::launchNeedMb() const { return read(Key::LaunchNeedMb, kDefaultLaunchNeedMb).toInt(); }
void AppSettings::setLaunchNeedMb(int mb) { write(Key::LaunchNeedMb, mb); }

const QStringList& AppSettings::defaultRunProfile()
{
    // the run switches of the PC's play-102 runs
    static const QStringList lines{
        QStringLiteral("XCL_SHADER_BACKEND=dxil"),
        QStringLiteral("XCL_LOOP_GUARD=1024"),
        QStringLiteral("XCL_RECOMP_OUTLINE_MIN_INSTS=4096"),
        QStringLiteral("XCL_RECOMP_OUTLINE_RESULTS=static"),
        QStringLiteral("XCL_RECOMP_NO_PLAIN_LOADS=1"),
        QStringLiteral("XCL_TEXTURE_ALIAS=0"),
        QStringLiteral("XCL_CHAIN_HOLD=0"),
        QStringLiteral("XCL_FSR4_STANDIN=1"),
        QStringLiteral("XCL_BVH_NODE_FIX=1"),
        QStringLiteral("XCL_COMPUTE_PER_ACCESS_COHERENCE=0"),
        QStringLiteral("XCL_LOOP_PROFILE=67657769b50d2f8b"),
        QStringLiteral("XCL_PSO_LIBRARY=0"),
        QStringLiteral("XCL_IMAGE_WRITEBACK=0"),
        QStringLiteral("XCL_LAZY_MESH_PSO=1"),
        QStringLiteral("XCL_DISPLAY_MODE=3840x2160"),
        QStringLiteral("XCL_WAIT_TIMEOUT_MS=20000"),
        QStringLiteral("XCL_ADAPTER=rtx"),
        QStringLiteral("XCL_RT_GATE=hold"),
        QStringLiteral("XCL_SHADER_CODE_MB=320"),
        QStringLiteral("XCL_SHADER_CODE_IDLE_S=30"),
    };
    return lines;
}

QStringList AppSettings::runProfile() const
{
    return m_settings->contains(QLatin1String(Key::RunProfile)) ? read(Key::RunProfile).toStringList() : defaultRunProfile();
}

void AppSettings::setRunProfile(const QStringList& lines) { write(Key::RunProfile, lines); }

void AppSettings::resetRunProfile()
{
    const QString key = QLatin1String(Key::RunProfile);
    if (!m_settings->contains(key))
        return;
    m_settings->remove(key);
    m_settings->sync();
    emit changed(key);
}

std::optional<QStringList> AppSettings::gameRunProfile(const QString& titleId) const
{
    const QVariant own = gameOverrides(titleId).value(QStringLiteral("runProfile"));
    if (!own.isValid())
        return std::nullopt;
    return own.toStringList();
}

QVariantMap AppSettings::gameOverrides(const QString& titleId) const
{
    return readMap(Key::GameOverrides).value(titleId).toMap();
}

void AppSettings::setGameOverride(const QString& titleId, const QString& key, const QVariant& value)
{
    QVariantMap all = readMap(Key::GameOverrides);
    QVariantMap one = all.value(titleId).toMap();
    one.insert(key, value);
    all.insert(titleId, one);
    write(Key::GameOverrides, QString::fromUtf8(QJsonDocument(QJsonObject::fromVariantMap(all)).toJson(QJsonDocument::Compact)));
}

void AppSettings::clearGameOverride(const QString& titleId, const QString& key)
{
    QVariantMap all = readMap(Key::GameOverrides);
    QVariantMap one = all.value(titleId).toMap();
    one.remove(key);
    if (one.isEmpty())
        all.remove(titleId);
    else
        all.insert(titleId, one);
    write(Key::GameOverrides, QString::fromUtf8(QJsonDocument(QJsonObject::fromVariantMap(all)).toJson(QJsonDocument::Compact)));
}

bool AppSettings::xsemuWarmup() const { return read(Key::XsemuWarmup, false).toBool(); }
void AppSettings::setXsemuWarmup(bool on) { write(Key::XsemuWarmup, on); }

QList<WarmupRun> AppSettings::recentWarmups() const
{
    QList<WarmupRun> out;
    const QJsonArray arr = QJsonDocument::fromJson(read(Key::RecentWarmups).toString().toUtf8()).array();
    for (const QJsonValue& v : arr)
        out.append(WarmupRun::fromMap(v.toObject().toVariantMap()));
    return out;
}

void AppSettings::addRecentWarmup(const WarmupRun& run)
{
    QJsonArray arr = QJsonDocument::fromJson(read(Key::RecentWarmups).toString().toUtf8()).array();
    arr.prepend(QJsonObject::fromVariantMap(run.toMap()));
    while (arr.size() > kRecentWarmupLimit)
        arr.removeLast();
    write(Key::RecentWarmups, QString::fromUtf8(QJsonDocument(arr).toJson(QJsonDocument::Compact)));
}

void AppSettings::clearRecentWarmups()
{
    write(Key::RecentWarmups, QStringLiteral("[]"));
}

std::optional<WarmupCheckpoint> AppSettings::warmupInFlight() const
{
    const QVariantMap m = readMap(Key::WarmupInFlight);
    const WarmupCheckpoint pass = WarmupCheckpoint::fromMap(m);
    return pass.isValid() ? std::optional<WarmupCheckpoint>(pass) : std::nullopt;
}

void AppSettings::setWarmupInFlight(const std::optional<WarmupCheckpoint>& pass)
{
    if (!pass) {
        write(Key::WarmupInFlight, QString());
        return;
    }
    write(Key::WarmupInFlight, QString::fromUtf8(QJsonDocument(QJsonObject::fromVariantMap(pass->toMap())).toJson(QJsonDocument::Compact)));
}

qint64 AppSettings::playSeconds(const QString& titleId) const
{
    return readMap(Key::PlayStats).value(titleId).toMap().value(QStringLiteral("seconds")).toLongLong();
}

QDateTime AppSettings::lastPlayed(const QString& titleId) const
{
    return QDateTime::fromString(readMap(Key::PlayStats).value(titleId).toMap().value(QStringLiteral("last")).toString(), Qt::ISODate);
}

QVariantMap AppSettings::lastRun(const QString& titleId) const
{
    return readMap(Key::PlayStats).value(titleId).toMap().value(QStringLiteral("lastRun")).toMap();
}

void AppSettings::setLastRun(const QString& titleId, const QVariantMap& run)
{
    QVariantMap all = readMap(Key::PlayStats);
    QVariantMap one = all.value(titleId).toMap();
    one.insert(QStringLiteral("lastRun"), run);
    all.insert(titleId, one);
    write(Key::PlayStats, QString::fromUtf8(QJsonDocument(QJsonObject::fromVariantMap(all)).toJson(QJsonDocument::Compact)));
}

QString AppSettings::librarySort() const { return read(Key::LibrarySort, QStringLiteral("recent")).toString(); }
void AppSettings::setLibrarySort(const QString& sort) { write(Key::LibrarySort, sort); }

void AppSettings::addPlayTime(const QString& titleId, qint64 seconds, const QDateTime& endedAt)
{
    QVariantMap all = readMap(Key::PlayStats);
    QVariantMap one = all.value(titleId).toMap();
    one.insert(QStringLiteral("seconds"), one.value(QStringLiteral("seconds")).toLongLong() + qMax<qint64>(0, seconds));
    one.insert(QStringLiteral("last"), endedAt.toUTC().toString(Qt::ISODate));
    all.insert(titleId, one);
    write(Key::PlayStats, QString::fromUtf8(QJsonDocument(QJsonObject::fromVariantMap(all)).toJson(QJsonDocument::Compact)));
}
