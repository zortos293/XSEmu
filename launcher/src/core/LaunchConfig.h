#pragma once

#include "GraphicsPreset.h"

#include <QProcessEnvironment>
#include <QString>
#include <QVariantMap>

class AppSettings;

// What the launcher hands XSEmu at launch besides the command line: the xcl.toml keys Settings
// owns, the XCL_* environment (the run profile and the variables Settings owns), and the folders those
// name. The emulator reads XCL_DISPLAY_MODE, XCL_ADAPTER and XCL_RAYTRACING over xcl.toml, so a run
// profile naming one wins over its xcl.toml key. Precedence for an xcl.toml key: the
// game's own launcher override (AppSettings::gameOverrides(titleId), keyed by the xcl.toml key name),
// else the launcher default from Settings, else whatever the title's xcl.toml already says.
// device_type always has a launcher default (lockhart, Series S). A quality preset adds
// XCL_UPSCALER/XCL_DLSS_MODE to the environment and title_memory_mb to the xcl.toml values; a run
// profile line naming a switch wins over the preset.
//
//   LaunchConfig::prepareTitle(settings, game.titleId());   // before XSEmu.exe run
//   env = LaunchConfig::runEnvironment(QProcessEnvironment::systemEnvironment(), settings, game.titleId());
namespace LaunchConfig {

// The xcl.toml keys Settings writes (crates/xcl/src/config.rs), with their TOML type.
enum class Type { String, Bool, Float };
struct XclKey {
    const char* key;
    Type type;
};
const QList<XclKey>& xclKeys(); // device_type first

// Converts a stored value (QSettings returns strings from the ini) to the key's TOML type.
QVariant typed(const QString& tomlKey, const QVariant& value);

// The quality preset chosen for a game (its "qualityPreset" per-game override); nullptr for Clear.
const GraphicsPreset::Preset* qualityPreset(const AppSettings& settings, const QString& titleId);

// The values to write into the title's xcl.toml before it starts (tested).
QVariantMap titleValues(const AppSettings& settings, const QString& titleId);

// The console a game runs as: its per-game override, else the launcher default ("lockhart" or
// "anaconda"). Third-party runners get it as a token; xcl.toml is XSEmu's and is not read for it.
QString deviceType(const AppSettings& settings, const QString& titleId);

// Writes titleValues() into the title's xcl.toml where they differ, and creates the crash dump folder
// when crash dumps are on. False when xcl.toml cannot be written.
bool prepareTitle(const AppSettings& settings, const QString& titleId);

// A run profile as the user writes it: one KEY=VALUE per line, blank lines ignored. A KEY matches
// [A-Z_][A-Z0-9_]* and appears once; each error names its line (tested).
struct ProfileParse {
    QList<QPair<QString, QString>> entries; // in the order written
    QStringList errors;
    bool ok() const { return errors.isEmpty(); }
    QStringList lines() const; // KEY=VALUE, as AppSettings stores them
};
ProfileParse parseProfile(const QString& text);

// The variables Settings sets, with where: a run profile line naming one is not used.
// XCL_NETWORK, XCL_STORE, XCL_FULL_DUMP and XCL_DXIL_CACHE.
const QList<QPair<QString, QString>>& settingsOwnedKeys();
QString settingOwning(const QString& key); // "Network, Offline mode"; empty when Settings doesn't own it

// The run profile a game runs with: its own when it has one, else the global one.
QStringList runProfile(const AppSettings& settings, const QString& titleId);

// The environment XSEmu gets for Play and for its warm-up, so both use the same shader cache
// generation: base, the game's run profile on top, then XCL_NETWORK, XCL_STORE and XCL_FULL_DUMP from
// Settings (XCL_DXIL_CACHE is already in base, see followShaderCacheDir) (tested).
QProcessEnvironment runEnvironment(const QProcessEnvironment& base, const AppSettings& settings, const QString& titleId);

// The run profile variables runEnvironment applies for a game, sorted and hashed to 16 hex digits:
// recorded with each XSEmu warm-up, so a warm-up made with other translator switches no longer counts.
QString profileIdentity(const AppSettings& settings, const QString& titleId);

// Why a Settings control backed by xcl.toml has no effect: the global run profile, else the launcher's
// own environment, sets the variable XSEmu reads over it. Empty when neither does.
QString environmentOverride(const AppSettings& settings, const QString& variable);

// <title>\logs\dumps: where XCL_FULL_DUMP points (crates/xcl/src/eh.rs writes xcl-{full|small}-<pid>.dmp).
QString crashDumpDir(const QString& titleId);

// The shader cache folder: XCL_DXIL_CACHE from the launcher's own environment wins; otherwise the
// user's Settings choice is put into this process's environment (XCL_DXIL_CACHE), so Paths, the warm-up
// and game runs all use it. Call once at startup; it follows later changes of the setting.
void followShaderCacheDir(AppSettings& settings);
bool shaderCacheDirFromEnvironment(); // XCL_DXIL_CACHE was set before the launcher started

}
