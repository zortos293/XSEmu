#include "Services.h"

#include "core/LaunchConfig.h"
#include "theme/Theme.h"

Services::Services(QObject* parent)
    : QObject(parent)
    , emulator(settings, gate)
    , warmup(settings, gate)
    , covers(settings)
{
    runners.setSettings(&settings);
    runners.setInUseCheck([this](const QString& id) -> QString {
        const RunnerSpec* spec = Runners::find(id);
        const QString name = spec ? spec->name : id;
        if (emulator.isActive() && emulator.runnerId() == id)
            return QStringLiteral("%1 is running %2; stop the game first.").arg(name, emulator.game().title());
        if (warmup.isActive() && warmup.progress().runnerId == id)
            return QStringLiteral("%1 is pre-caching %2; cancel the pre-cache first.").arg(name, warmup.game().title());
        return {};
    });
    emulator.setRunners(&runners);
    warmup.setRunners(&runners);
    Theme::instance().setMode(Theme::Mode(int(settings.theme())));
    LaunchConfig::followShaderCacheDir(settings);
    library.setFolders(settings.gameFolders());
    gate.setBusyMarker(settings.busyMarker());
    shaderCache.refresh();
    connect(&warmup, &WarmupController::finished, &shaderCache, &ShaderCacheMonitor::refresh);
    connect(&emulator, &EmulatorController::exited, &shaderCache, &ShaderCacheMonitor::refresh);
    connect(&covers, &CoverArtService::artChanged, this, [this](const QString& titleId) {
        if (titleId.isEmpty()) {
            m_art.clear();
            return;
        }
        for (CoverCache::Kind kind : CoverCache::kKinds)
            m_art.remove(titleId + QLatin1Char('|') + CoverCache::kindName(kind));
    });
    connect(&settings, &AppSettings::changed, this, [this](const QString& key) {
        if (key == QLatin1String(AppSettings::Key::Theme))
            Theme::instance().setMode(Theme::Mode(int(settings.theme())));
        else if (key == QLatin1String(AppSettings::Key::GameFolders))
            library.setFolders(settings.gameFolders());
        else if (key == QLatin1String(AppSettings::Key::BusyMarker))
            gate.setBusyMarker(settings.busyMarker());
        else if (key == QLatin1String(AppSettings::Key::ShaderCacheDir))
            shaderCache.refresh();
    });
}

QPixmap Services::art(const QString& titleId, CoverCache::Kind kind)
{
    const QString key = titleId + QLatin1Char('|') + CoverCache::kindName(kind);
    if (const auto it = m_art.constFind(key); it != m_art.constEnd())
        return *it;
    const QString path = covers.artPath(titleId, kind);
    const QPixmap pm = path.isEmpty() ? QPixmap() : QPixmap(path);
    m_art.insert(key, pm);
    return pm;
}

QString Services::consoleName(const QString& deviceType)
{
    return deviceType == QLatin1String("anaconda") ? QStringLiteral("Series X") : QStringLiteral("Series S");
}
