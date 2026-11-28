#include "LaunchConfig.h"

#include "AppSettings.h"
#include "Paths.h"
#include "TitleConfig.h"

#include <QCryptographicHash>
#include <QDir>
#include <QHash>
#include <QMap>
#include <QRegularExpression>

namespace {

QString& externalCacheDir()
{
    static QString dir = qEnvironmentVariable("XCL_DXIL_CACHE").trimmed();
    return dir;
}

void applyShaderCacheDir(const AppSettings& settings)
{
    if (!externalCacheDir().isEmpty())
        return;
    const QString dir = settings.shaderCacheDirOverride();
    if (dir.isEmpty())
        qunsetenv("XCL_DXIL_CACHE");
    else
        qputenv("XCL_DXIL_CACHE", QDir::toNativeSeparators(dir).toUtf8());
}

}

namespace LaunchConfig {

const QList<XclKey>& xclKeys()
{
    static const QList<XclKey> keys{
        {"device_type", Type::String},
        {"log_level", Type::String},
        {"keyboard_mouse", Type::Bool},
        {"mouse_sensitivity", Type::Float},
        {"display_mode", Type::String},
        {"raytracing", Type::String},
        {"present", Type::String},
        {"d3d12_adapter", Type::String},
    };
    return keys;
}

QVariant typed(const QString& tomlKey, const QVariant& value)
{
    if (!value.isValid())
        return {};
    for (const XclKey& k : xclKeys()) {
        if (tomlKey != QLatin1String(k.key))
            continue;
        switch (k.type) {
        case Type::String: return value.toString();
        case Type::Bool:
            if (value.typeId() == QMetaType::QString)
                return value.toString().compare(QLatin1String("true"), Qt::CaseInsensitive) == 0;
            return value.toBool();
        case Type::Float: return value.toDouble();
        }
    }
    return value;
}

const GraphicsPreset::Preset* qualityPreset(const AppSettings& settings, const QString& titleId)
{
    if (!GraphicsPreset::supported())
        return nullptr;
    return GraphicsPreset::find(settings.gameOverrides(titleId).value(QStringLiteral("qualityPreset")).toString());
}

QVariantMap titleValues(const AppSettings& settings, const QString& titleId)
{
    const QVariantMap overrides = settings.gameOverrides(titleId);
    QVariantMap out;
    for (const XclKey& k : xclKeys()) {
        const QString key = QLatin1String(k.key);
        QVariant v = overrides.value(key);
        if (!v.isValid())
            v = key == QLatin1String("device_type") ? QVariant(settings.defaultDeviceType()) : settings.xclDefault(key);
        if (v.isValid())
            out.insert(key, typed(key, v));
    }
    // The preset's title budget, unless the game or Settings names one itself; Clear's reset (0,
    // the device's standard budget) is an override like any other, so it is written too.
    QVariant budget = overrides.value(QStringLiteral("title_memory_mb"));
    if (!budget.isValid())
        budget = settings.xclDefault(QStringLiteral("title_memory_mb"));
    if (!budget.isValid()) {
        if (const GraphicsPreset::Preset* preset = qualityPreset(settings, titleId))
            budget = preset->titleMemoryMb;
    }
    if (budget.isValid())
        out.insert(QStringLiteral("title_memory_mb"), budget.toLongLong());
    return out;
}

QString deviceType(const AppSettings& settings, const QString& titleId)
{
    return titleValues(settings, titleId).value(QStringLiteral("device_type")).toString();
}

bool prepareTitle(const AppSettings& settings, const QString& titleId)
{
    if (settings.crashDumps())
        QDir().mkpath(crashDumpDir(titleId));
    TitleConfig cfg(Paths::xclToml(titleId));
    const QVariantMap current = cfg.values();
    QVariantMap changes;
    const QVariantMap wanted = titleValues(settings, titleId);
    for (auto it = wanted.cbegin(); it != wanted.cend(); ++it) {
        if (typed(it.key(), current.value(it.key())) != it.value())
            changes.insert(it.key(), it.value());
    }
    return changes.isEmpty() || cfg.set(changes);
}

QStringList ProfileParse::lines() const
{
    QStringList out;
    for (const auto& e : entries)
        out << e.first + QLatin1Char('=') + e.second;
    return out;
}

ProfileParse parseProfile(const QString& text)
{
    static const QRegularExpression keyRe(QStringLiteral("^[A-Z_][A-Z0-9_]*$"));
    ProfileParse out;
    QHash<QString, int> seen;
    const QStringList lines = text.split(QLatin1Char('\n'));
    for (int i = 0; i < lines.size(); ++i) {
        const QString line = lines[i].trimmed();
        if (line.isEmpty())
            continue;
        const int n = i + 1;
        const qsizetype eq = line.indexOf(QLatin1Char('='));
        if (eq < 0) {
            out.errors << QStringLiteral("Line %1: write it as KEY=VALUE.").arg(n);
            continue;
        }
        const QString key = line.left(eq).trimmed();
        if (!keyRe.match(key).hasMatch()) {
            out.errors << QStringLiteral("Line %1: \"%2\" isn't a variable name. Use A-Z, 0-9 and _, not starting with a digit.").arg(n).arg(key);
            continue;
        }
        if (seen.contains(key)) {
            out.errors << QStringLiteral("Line %1: %2 is already set on line %3.").arg(n).arg(key).arg(seen.value(key));
            continue;
        }
        seen.insert(key, n);
        out.entries.append({key, line.mid(eq + 1).trimmed()});
    }
    return out;
}

const QList<QPair<QString, QString>>& settingsOwnedKeys()
{
    static const QList<QPair<QString, QString>> keys{
        {QStringLiteral("XCL_NETWORK"), QStringLiteral("Network, Offline mode")},
        {QStringLiteral("XCL_STORE"), QStringLiteral("Network, Store behaviour")},
        {QStringLiteral("XCL_FULL_DUMP"), QStringLiteral("Logs & crash reports, Crash reports")},
        {QStringLiteral("XCL_DXIL_CACHE"), QStringLiteral("Shaders, Cache folder")},
    };
    return keys;
}

QString settingOwning(const QString& key)
{
    for (const auto& k : settingsOwnedKeys()) {
        if (k.first == key)
            return k.second;
    }
    return {};
}

QStringList runProfile(const AppSettings& settings, const QString& titleId)
{
    return settings.gameRunProfile(titleId).value_or(settings.runProfile());
}

QProcessEnvironment runEnvironment(const QProcessEnvironment& base, const AppSettings& settings, const QString& titleId)
{
    QProcessEnvironment env = base;
    // the preset goes under the run profile: a profile line naming a switch is the more specific choice
    if (const GraphicsPreset::Preset* preset = qualityPreset(settings, titleId)) {
        for (const auto& [key, value] : GraphicsPreset::environment(*preset))
            env.insert(key, value);
    }
    for (const auto& [key, value] : parseProfile(runProfile(settings, titleId).join(QLatin1Char('\n'))).entries) {
        if (settingOwning(key).isEmpty())
            env.insert(key, value);
    }
    if (settings.offline())
        env.insert(QStringLiteral("XCL_NETWORK"), QStringLiteral("0"));
    else
        env.remove(QStringLiteral("XCL_NETWORK"));
    // the store answers with its network error: nothing grants a licence (hle/xgameruntime/store.rs)
    if (settings.storeOfflineError())
        env.insert(QStringLiteral("XCL_STORE"), QStringLiteral("offline-error"));
    else
        env.remove(QStringLiteral("XCL_STORE"));
    if (settings.crashDumps())
        env.insert(QStringLiteral("XCL_FULL_DUMP"), QDir::toNativeSeparators(crashDumpDir(titleId)));
    else
        env.remove(QStringLiteral("XCL_FULL_DUMP"));
    return env;
}

QString profileIdentity(const AppSettings& settings, const QString& titleId)
{
    // Keyed, so a run profile line replacing a preset's switch leaves one entry: the identity changes
    // only when the environment XSEmu gets changes.
    QMap<QString, QString> applied;
    if (const GraphicsPreset::Preset* preset = qualityPreset(settings, titleId)) {
        for (const auto& [key, value] : GraphicsPreset::environment(*preset))
            applied.insert(key, value);
    }
    for (const auto& [key, value] : parseProfile(runProfile(settings, titleId).join(QLatin1Char('\n'))).entries) {
        if (settingOwning(key).isEmpty())
            applied.insert(key, value);
    }
    QStringList lines;
    for (auto it = applied.cbegin(); it != applied.cend(); ++it)
        lines << it.key() + QLatin1Char('=') + it.value();
    return QString::fromLatin1(QCryptographicHash::hash(lines.join(QLatin1Char('\n')).toUtf8(), QCryptographicHash::Sha1).toHex().left(16));
}

QString environmentOverride(const AppSettings& settings, const QString& variable)
{
    for (const auto& [key, value] : parseProfile(settings.runProfile().join(QLatin1Char('\n'))).entries) {
        if (key == variable)
            return QStringLiteral("%1=%2 in the run profile overrides this (Settings, Run profile).").arg(key, value);
    }
    const QString v = qEnvironmentVariable(variable.toLatin1().constData()).trimmed();
    return v.isEmpty() ? QString() : QStringLiteral("%1=%2 in the environment overrides this.").arg(variable, v);
}

QString crashDumpDir(const QString& titleId)
{
    return QDir(Paths::logsDir(titleId)).filePath(QStringLiteral("dumps"));
}

void followShaderCacheDir(AppSettings& settings)
{
    externalCacheDir();
    applyShaderCacheDir(settings);
    QObject::connect(&settings, &AppSettings::changed, &settings, [&settings](const QString& key) {
        if (key == QLatin1String(AppSettings::Key::ShaderCacheDir))
            applyShaderCacheDir(settings);
    });
}

bool shaderCacheDirFromEnvironment()
{
    return !externalCacheDir().isEmpty();
}

}
