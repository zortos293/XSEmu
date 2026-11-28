#include "AppSettings.h"
#include "EmulatorController.h"
#include "Format.h"
#include "LaunchConfig.h"
#include "TitleConfig.h"
#include "Paths.h"
#include "ProcessGate.h"
#include "Runner.h"
#include "WarmupController.h"

#include <QSignalSpy>
#include <QTemporaryDir>
#include <QTest>

class TestSettings : public QObject {
    Q_OBJECT
private slots:
    void defaultsFollowTheHouseRules();
    void changesAreSignalledAndPersist();
    void recentWarmupsAndPlayTime();
    void paths();
    void gate();
    void runCommandLine();
    void runProfileDefaults();
    void runProfilePerGame();
    void runProfileSettingsOwnedKeysWin();
    void runProfileValidation();
    void titleConfigAtLaunch();
    void qualityPresets();
    void warmupClassification();
    void warmupResume();
    void warmupCheckpointRoundTrip();
    void formats();
};

void TestSettings::defaultsFollowTheHouseRules()
{
    QTemporaryDir dir;
    AppSettings s(dir.filePath(QStringLiteral("s.ini")));
    QCOMPARE(s.defaultDeviceType(), QStringLiteral("lockhart"));
    QVERIFY(s.offline());
    QVERIFY(!s.firstLaunchDone());
    QCOMPARE(s.theme(), AppSettings::ThemeMode::System);
    QCOMPARE(s.launchNeedMb(), 0);
    QCOMPARE(s.librarySort(), QStringLiteral("recent"));
    QVERIFY(s.artLookups());
    QVERIFY(s.steamGridDbKey().isEmpty());
}

void TestSettings::changesAreSignalledAndPersist()
{
    QTemporaryDir dir;
    const QString ini = dir.filePath(QStringLiteral("s.ini"));
    {
        AppSettings s(ini);
        QSignalSpy spy(&s, &AppSettings::changed);
        s.setOffline(false);
        s.setOffline(false);
        QCOMPARE(spy.count(), 1);
        QCOMPARE(spy.first().first().toString(), QString::fromLatin1(AppSettings::Key::Offline));
        s.setGameFolders({QStringLiteral("/a"), QStringLiteral("/b")});
        s.setTheme(AppSettings::ThemeMode::Dark);
        s.setGameOverride(QStringLiteral("T"), QStringLiteral("overlay"), true);
    }
    AppSettings s(ini);
    QVERIFY(!s.offline());
    QCOMPARE(s.gameFolders(), (QStringList{QStringLiteral("/a"), QStringLiteral("/b")}));
    QCOMPARE(s.theme(), AppSettings::ThemeMode::Dark);
    QCOMPARE(s.gameOverrides(QStringLiteral("T")).value(QStringLiteral("overlay")).toBool(), true);
    s.clearGameOverride(QStringLiteral("T"), QStringLiteral("overlay"));
    QVERIFY(s.gameOverrides(QStringLiteral("T")).isEmpty());
}

void TestSettings::recentWarmupsAndPlayTime()
{
    QTemporaryDir dir;
    AppSettings s(dir.filePath(QStringLiteral("s.ini")));
    for (int i = 0; i < AppSettings::kRecentWarmupLimit + 5; ++i) {
        WarmupRun r;
        r.titleId = QStringLiteral("T%1").arg(i);
        r.done = i;
        r.result = WarmupRun::Result::Cancelled;
        r.finishedAt = QDateTime::currentDateTime();
        s.addRecentWarmup(r);
    }
    const QList<WarmupRun> runs = s.recentWarmups();
    QCOMPARE(runs.size(), AppSettings::kRecentWarmupLimit);
    QCOMPARE(runs.first().titleId, QStringLiteral("T24"));
    QCOMPARE(runs.first().result, WarmupRun::Result::Cancelled);
    const QDateTime end = QDateTime::currentDateTime();
    s.addPlayTime(QStringLiteral("G"), 100, end);
    s.addPlayTime(QStringLiteral("G"), 50, end);
    QCOMPARE(s.playSeconds(QStringLiteral("G")), 150);
    QCOMPARE(s.lastPlayed(QStringLiteral("G")).toSecsSinceEpoch(), end.toSecsSinceEpoch());
}

void TestSettings::paths()
{
    QTemporaryDir dir;
    qputenv("XCL_DATA_ROOT", dir.path().toUtf8());
    qputenv("XCL_DXIL_CACHE", dir.filePath(QStringLiteral("dxil")).toUtf8());
    QCOMPARE(Paths::dataRoot(), dir.path());
    QCOMPARE(Paths::xclToml(QStringLiteral("A.B")), dir.filePath(QStringLiteral("A.B/xcl.toml")));
    QCOMPARE(Paths::logsDir(QStringLiteral("A.B")), dir.filePath(QStringLiteral("A.B/logs")));
    QCOMPARE(Paths::cacheDir(QStringLiteral("A.B")), dir.filePath(QStringLiteral("A.B/cache")));
    QCOMPARE(Paths::pipelineManifestDir(), dir.filePath(QStringLiteral("dxil/pipelines")));
    qunsetenv("XCL_DATA_ROOT");
    qunsetenv("XCL_DXIL_CACHE");
    QVERIFY(Paths::dataRoot().endsWith(QStringLiteral("/XCL")));
}

void TestSettings::gate()
{
    ProcessGate gate;
    int stops = 0;
    QVERIFY(gate.isFree());
    QVERIFY(gate.blockedReason(ProcessGate::Action::StartWarmup).isEmpty());
    QVERIFY(gate.acquire(ProcessGate::Owner::Game, QStringLiteral("Signal Garden"), [&] { ++stops; }));
    QVERIFY(!gate.acquire(ProcessGate::Owner::Warmup, QStringLiteral("x"), [] {}));
    gate.setPaused(true);
    QVERIFY(gate.blockedReason(ProcessGate::Action::StartWarmup).startsWith(QStringLiteral("Signal Garden is paused.")));
    QCOMPARE(gate.stopActionLabel(), QStringLiteral("Stop game"));
    gate.stopOwner();
    QCOMPARE(stops, 1);
    gate.release(ProcessGate::Owner::Warmup);
    QVERIFY(!gate.isFree());
    gate.release(ProcessGate::Owner::Game);
    QVERIFY(gate.isFree());
}

void TestSettings::runCommandLine()
{
    QCOMPARE(EmulatorController::runArguments(QStringLiteral("/games/A"), 0x1a2b3c, QStringLiteral("xsemu-1-1")),
             (QStringList{QStringLiteral("run"), QStringLiteral("--host-hwnd"), QStringLiteral("1a2b3c"), QStringLiteral("--control-pipe"),
                          QStringLiteral("xsemu-1-1"), QDir::toNativeSeparators(QStringLiteral("/games/A"))}));
    QCOMPARE(EmulatorController::runArguments(QStringLiteral("/g"), 0, QString()), (QStringList{QStringLiteral("run"), QDir::toNativeSeparators(QStringLiteral("/g"))}));
    QTemporaryDir dir;
    qputenv("XCL_DATA_ROOT", dir.filePath(QStringLiteral("xcl")).toUtf8());
    AppSettings s(dir.filePath(QStringLiteral("s.ini")));
    QProcessEnvironment base;
    base.insert(QStringLiteral("XCL_NETWORK"), QStringLiteral("1"));
    const QProcessEnvironment off = LaunchConfig::runEnvironment(base, s, QStringLiteral("T"));
    QCOMPARE(off.value(QStringLiteral("XCL_NETWORK")), QStringLiteral("0"));
    QCOMPARE(off.value(QStringLiteral("XCL_STORE")), QStringLiteral("offline-error"));
    QCOMPARE(off.value(QStringLiteral("XCL_FULL_DUMP")), QDir::toNativeSeparators(LaunchConfig::crashDumpDir(QStringLiteral("T"))));
    QVERIFY(LaunchConfig::crashDumpDir(QStringLiteral("T")).endsWith(QStringLiteral("T/logs/dumps")));
    s.setOffline(false);
    s.setStoreOfflineError(false);
    s.setCrashDumps(false);
    base.insert(QStringLiteral("XCL_STORE"), QStringLiteral("offline-error"));
    const QProcessEnvironment on = LaunchConfig::runEnvironment(base, s, QStringLiteral("T"));
    QVERIFY(!on.contains(QStringLiteral("XCL_NETWORK")));
    QVERIFY(!on.contains(QStringLiteral("XCL_STORE")));
    QVERIFY(!on.contains(QStringLiteral("XCL_FULL_DUMP")));
    QCOMPARE(WarmupController::warmupArguments(true), (QStringList{QStringLiteral("warm-up"), QStringLiteral("--full"), QStringLiteral("--progress-json")}));
}

void TestSettings::runProfileDefaults()
{
    QTemporaryDir dir;
    qputenv("XCL_DATA_ROOT", dir.filePath(QStringLiteral("xcl")).toUtf8());
    const QString ini = dir.filePath(QStringLiteral("s.ini"));
    const QStringList play102{
        QStringLiteral("XCL_SHADER_BACKEND=dxil"), QStringLiteral("XCL_LOOP_GUARD=1024"), QStringLiteral("XCL_RECOMP_OUTLINE_MIN_INSTS=4096"),
        QStringLiteral("XCL_RECOMP_OUTLINE_RESULTS=static"), QStringLiteral("XCL_RECOMP_NO_PLAIN_LOADS=1"), QStringLiteral("XCL_TEXTURE_ALIAS=0"),
        QStringLiteral("XCL_CHAIN_HOLD=0"), QStringLiteral("XCL_FSR4_STANDIN=1"), QStringLiteral("XCL_BVH_NODE_FIX=1"),
        QStringLiteral("XCL_COMPUTE_PER_ACCESS_COHERENCE=0"), QStringLiteral("XCL_LOOP_PROFILE=67657769b50d2f8b"), QStringLiteral("XCL_PSO_LIBRARY=0"),
        QStringLiteral("XCL_IMAGE_WRITEBACK=0"), QStringLiteral("XCL_LAZY_MESH_PSO=1"), QStringLiteral("XCL_DISPLAY_MODE=3840x2160"),
        QStringLiteral("XCL_WAIT_TIMEOUT_MS=20000"), QStringLiteral("XCL_ADAPTER=rtx"), QStringLiteral("XCL_RT_GATE=hold"),
        QStringLiteral("XCL_SHADER_CODE_MB=320"), QStringLiteral("XCL_SHADER_CODE_IDLE_S=30"),
    };
    {
        AppSettings s(ini);
        QCOMPARE(s.runProfile(), play102);
        QCOMPARE(AppSettings::defaultRunProfile(), play102);
        const QProcessEnvironment env = LaunchConfig::runEnvironment(QProcessEnvironment(), s, QStringLiteral("T"));
        for (const auto& [key, value] : LaunchConfig::parseProfile(play102.join(QLatin1Char('\n'))).entries)
            QCOMPARE(env.value(key), value);
        QVERIFY(!env.contains(QStringLiteral("XCL_SKIP_SHADERS_FILE")));
        QVERIFY(LaunchConfig::environmentOverride(s, QStringLiteral("XCL_DISPLAY_MODE")).contains(QLatin1String("XCL_DISPLAY_MODE=3840x2160 in the run profile")));

        QSignalSpy spy(&s, &AppSettings::changed);
        s.setRunProfile({QStringLiteral("XCL_LOOP_GUARD=2048"), QStringLiteral("XCL_LABEL=a=b,c")});
        QCOMPARE(spy.count(), 1);
        QCOMPARE(spy.first().first().toString(), QString::fromLatin1(AppSettings::Key::RunProfile));
        QVERIFY(LaunchConfig::environmentOverride(s, QStringLiteral("XCL_DISPLAY_MODE")).isEmpty());
    }
    AppSettings s(ini);
    QCOMPARE(s.runProfile(), (QStringList{QStringLiteral("XCL_LOOP_GUARD=2048"), QStringLiteral("XCL_LABEL=a=b,c")}));
    QCOMPARE(LaunchConfig::runEnvironment(QProcessEnvironment(), s, QStringLiteral("T")).value(QStringLiteral("XCL_LABEL")), QStringLiteral("a=b,c"));
    s.setRunProfile({});
    QVERIFY(AppSettings(ini).runProfile().isEmpty());
    QVERIFY(!LaunchConfig::runEnvironment(QProcessEnvironment(), s, QStringLiteral("T")).contains(QStringLiteral("XCL_LOOP_GUARD")));
    s.resetRunProfile();
    QCOMPARE(AppSettings(ini).runProfile(), play102);
}

void TestSettings::runProfilePerGame()
{
    QTemporaryDir dir;
    qputenv("XCL_DATA_ROOT", dir.filePath(QStringLiteral("xcl")).toUtf8());
    AppSettings s(dir.filePath(QStringLiteral("s.ini")));
    const QString a = QStringLiteral("Northwind.A");
    const QString b = QStringLiteral("Northwind.B");
    QVERIFY(!s.gameRunProfile(a));
    s.setGameOverride(a, QStringLiteral("runProfile"), QStringList{QStringLiteral("XCL_LOOP_GUARD=4096"), QStringLiteral("XCL_EXTRA=1")});
    QCOMPARE(*AppSettings(s.fileName()).gameRunProfile(a), (QStringList{QStringLiteral("XCL_LOOP_GUARD=4096"), QStringLiteral("XCL_EXTRA=1")}));
    QVERIFY(!s.gameRunProfile(b));

    const QProcessEnvironment envA = LaunchConfig::runEnvironment(QProcessEnvironment(), s, a);
    QCOMPARE(envA.value(QStringLiteral("XCL_LOOP_GUARD")), QStringLiteral("4096"));
    QCOMPARE(envA.value(QStringLiteral("XCL_EXTRA")), QStringLiteral("1"));
    QVERIFY(!envA.contains(QStringLiteral("XCL_SHADER_BACKEND")));
    const QProcessEnvironment envB = LaunchConfig::runEnvironment(QProcessEnvironment(), s, b);
    QCOMPARE(envB.value(QStringLiteral("XCL_LOOP_GUARD")), QStringLiteral("1024"));
    QCOMPARE(envB.value(QStringLiteral("XCL_SHADER_BACKEND")), QStringLiteral("dxil"));
    QVERIFY(!envB.contains(QStringLiteral("XCL_EXTRA")));
    // the game's own profile replaces the global one, also when that one changes
    s.setRunProfile({QStringLiteral("XCL_GLOBAL=1")});
    QVERIFY(!LaunchConfig::runEnvironment(QProcessEnvironment(), s, a).contains(QStringLiteral("XCL_GLOBAL")));
    QVERIFY(LaunchConfig::runEnvironment(QProcessEnvironment(), s, b).contains(QStringLiteral("XCL_GLOBAL")));
    // an empty own profile is still the game's own: nothing from the global one
    s.setGameOverride(a, QStringLiteral("runProfile"), QStringList{});
    QVERIFY(s.gameRunProfile(a));
    QVERIFY(!LaunchConfig::runEnvironment(QProcessEnvironment(), s, a).contains(QStringLiteral("XCL_GLOBAL")));
    s.clearGameOverride(a, QStringLiteral("runProfile"));
    QVERIFY(LaunchConfig::runEnvironment(QProcessEnvironment(), s, a).contains(QStringLiteral("XCL_GLOBAL")));

    // the identity a warm-up records follows the switches that apply, not their order or the settings-owned lines
    const QString identity = LaunchConfig::profileIdentity(s, a);
    QCOMPARE(identity.size(), 16);
    s.setRunProfile({QStringLiteral("XCL_NETWORK=1"), QStringLiteral("XCL_GLOBAL=1")});
    QCOMPARE(LaunchConfig::profileIdentity(s, a), identity);
    s.setRunProfile({QStringLiteral("XCL_GLOBAL=2")});
    QVERIFY(LaunchConfig::profileIdentity(s, a) != identity);
}

void TestSettings::runProfileSettingsOwnedKeysWin()
{
    QTemporaryDir dir;
    qputenv("XCL_DATA_ROOT", dir.filePath(QStringLiteral("xcl")).toUtf8());
    AppSettings s(dir.filePath(QStringLiteral("s.ini")));
    s.setRunProfile({QStringLiteral("XCL_NETWORK=1"), QStringLiteral("XCL_STORE=granted"), QStringLiteral("XCL_FULL_DUMP=/elsewhere"),
                     QStringLiteral("XCL_DXIL_CACHE=/profile/cache"), QStringLiteral("XCL_LOOP_GUARD=1024")});
    QProcessEnvironment base;
    base.insert(QStringLiteral("XCL_DXIL_CACHE"), QStringLiteral("/settings/cache"));
    const QProcessEnvironment on = LaunchConfig::runEnvironment(base, s, QStringLiteral("T"));
    QCOMPARE(on.value(QStringLiteral("XCL_NETWORK")), QStringLiteral("0"));
    QCOMPARE(on.value(QStringLiteral("XCL_STORE")), QStringLiteral("offline-error"));
    QCOMPARE(on.value(QStringLiteral("XCL_FULL_DUMP")), QDir::toNativeSeparators(LaunchConfig::crashDumpDir(QStringLiteral("T"))));
    QCOMPARE(on.value(QStringLiteral("XCL_DXIL_CACHE")), QStringLiteral("/settings/cache"));
    QCOMPARE(on.value(QStringLiteral("XCL_LOOP_GUARD")), QStringLiteral("1024"));
    s.setOffline(false);
    s.setStoreOfflineError(false);
    s.setCrashDumps(false);
    const QProcessEnvironment off = LaunchConfig::runEnvironment(QProcessEnvironment(), s, QStringLiteral("T"));
    for (const auto& [key, owner] : LaunchConfig::settingsOwnedKeys()) {
        QVERIFY2(!off.contains(key), qPrintable(key));
        QVERIFY(!owner.isEmpty());
        QCOMPARE(LaunchConfig::settingOwning(key), owner);
    }
    QVERIFY(LaunchConfig::settingOwning(QStringLiteral("XCL_LOOP_GUARD")).isEmpty());
}

void TestSettings::runProfileValidation()
{
    const LaunchConfig::ProfileParse ok = LaunchConfig::parseProfile(QStringLiteral("  XCL_A = 1 \n\n_B=x=y\nC2=\n"));
    QVERIFY(ok.ok());
    QCOMPARE(ok.lines(), (QStringList{QStringLiteral("XCL_A=1"), QStringLiteral("_B=x=y"), QStringLiteral("C2=")}));

    const LaunchConfig::ProfileParse bad = LaunchConfig::parseProfile(
        QStringLiteral("XCL_A=1\nxcl_b=2\n9X=1\nXCL-C=1\nXCL_A=3\nno equals\n=1\nXCL_D=4"));
    QCOMPARE(bad.lines(), (QStringList{QStringLiteral("XCL_A=1"), QStringLiteral("XCL_D=4")}));
    QCOMPARE(bad.errors.size(), 6);
    QVERIFY(bad.errors[0].startsWith(QLatin1String("Line 2: \"xcl_b\"")));
    QVERIFY(bad.errors[1].startsWith(QLatin1String("Line 3: \"9X\"")));
    QVERIFY(bad.errors[2].startsWith(QLatin1String("Line 4: \"XCL-C\"")));
    QCOMPARE(bad.errors[3], QStringLiteral("Line 5: XCL_A is already set on line 1."));
    QCOMPARE(bad.errors[4], QStringLiteral("Line 6: write it as KEY=VALUE."));
    QVERIFY(bad.errors[5].startsWith(QLatin1String("Line 7: \"\"")));
}

void TestSettings::titleConfigAtLaunch()
{
    QTemporaryDir dir;
    qputenv("XCL_DATA_ROOT", dir.filePath(QStringLiteral("xcl")).toUtf8());
    AppSettings s(dir.filePath(QStringLiteral("s.ini")));
    const QString id = QStringLiteral("Northwind.Title");

    // a fresh title gets the Series S default and nothing the user never chose
    QVariantMap v = LaunchConfig::titleValues(s, id);
    QCOMPARE(v, (QVariantMap{{QStringLiteral("device_type"), QStringLiteral("lockhart")}}));

    // launcher defaults come back with their TOML types after a round trip through the ini
    s.setXclDefault(QStringLiteral("keyboard_mouse"), false);
    s.setXclDefault(QStringLiteral("mouse_sensitivity"), 1.5);
    s.setXclDefault(QStringLiteral("log_level"), QStringLiteral("debug"));
    AppSettings reread(dir.filePath(QStringLiteral("s.ini")));
    v = LaunchConfig::titleValues(reread, id);
    QCOMPARE(v.value(QStringLiteral("keyboard_mouse")).typeId(), int(QMetaType::Bool));
    QCOMPARE(v.value(QStringLiteral("keyboard_mouse")).toBool(), false);
    QCOMPARE(v.value(QStringLiteral("mouse_sensitivity")).typeId(), int(QMetaType::Double));
    QCOMPARE(v.value(QStringLiteral("mouse_sensitivity")).toDouble(), 1.5);

    // a per-game override wins over the default
    s.setDefaultDeviceType(QStringLiteral("lockhart"));
    s.setGameOverride(id, QStringLiteral("device_type"), QStringLiteral("anaconda"));
    QCOMPARE(LaunchConfig::titleValues(s, id).value(QStringLiteral("device_type")).toString(), QStringLiteral("anaconda"));
    s.clearGameOverride(id, QStringLiteral("device_type"));

    // the file keeps the user's other keys and gets the launcher's values
    TitleConfig cfg(Paths::xclToml(id));
    QDir().mkpath(QFileInfo(cfg.path()).absolutePath());
    QFile f(cfg.path());
    QVERIFY(f.open(QIODevice::WriteOnly));
    f.write("# mine\ndevice_type = \"anaconda\"\ngamertag = \"Me\"\nmouse_sensitivity = 1.0\n");
    f.close();
    QVERIFY(LaunchConfig::prepareTitle(s, id));
    const QVariantMap after = cfg.values();
    QCOMPARE(after.value(QStringLiteral("device_type")).toString(), QStringLiteral("lockhart"));
    QCOMPARE(after.value(QStringLiteral("gamertag")).toString(), QStringLiteral("Me"));
    QCOMPARE(after.value(QStringLiteral("keyboard_mouse")).toBool(), false);
    QCOMPARE(after.value(QStringLiteral("mouse_sensitivity")).toDouble(), 1.5);
    QCOMPARE(after.value(QStringLiteral("log_level")).toString(), QStringLiteral("debug"));
    QVERIFY(QFileInfo(LaunchConfig::crashDumpDir(id)).isDir());

    // a per-game choice made on Game details (the overrides store) survives the launch rewrite;
    // launcher-only choices stay out of xcl.toml
    s.setGameOverride(id, QStringLiteral("device_type"), QStringLiteral("anaconda"));
    s.setGameOverride(id, QStringLiteral("log_level"), QStringLiteral("trace"));
    s.setGameOverride(id, QStringLiteral("runner"), QStringLiteral("kelvaroq"));
    QVERIFY(LaunchConfig::prepareTitle(s, id));
    const QVariantMap chosen = cfg.values();
    QCOMPARE(chosen.value(QStringLiteral("device_type")).toString(), QStringLiteral("anaconda"));
    QCOMPARE(chosen.value(QStringLiteral("log_level")).toString(), QStringLiteral("trace"));
    QVERIFY(!chosen.contains(QStringLiteral("runner")));
    // Use defaults clears the store; the next launch writes the defaults back
    for (const char* key : {"device_type", "log_level", "runner"})
        s.clearGameOverride(id, QLatin1String(key));
    QVERIFY(LaunchConfig::prepareTitle(s, id));
    QCOMPARE(cfg.values().value(QStringLiteral("device_type")).toString(), QStringLiteral("lockhart"));
    QCOMPARE(cfg.values().value(QStringLiteral("log_level")).toString(), QStringLiteral("debug"));

    // a runner gets the console from the same store: a Series X override with no xcl.toml at all,
    // and after Use defaults the default, whatever a stale xcl.toml still says
    const QString other = QStringLiteral("Runner.Only");
    QVERIFY(!QFileInfo::exists(Paths::xclToml(other)));
    s.setGameOverride(other, QStringLiteral("device_type"), QStringLiteral("anaconda"));
    QCOMPARE(LaunchConfig::deviceType(s, other), QStringLiteral("anaconda"));
    const RunnerSpec& kelvaroq = *Runners::find(QStringLiteral("kelvaroq"));
    const QStringList args = RunnerSpec::expand(kelvaroq.launchArgs, kelvaroq.tokens(dir.path(), dir.filePath(QStringLiteral("game.exe")), other, LaunchConfig::deviceType(s, other), 0));
    QVERIFY2(args.contains(QStringLiteral("series-x")), qPrintable(args.join(QLatin1Char(' '))));
    QVERIFY(!QFileInfo::exists(Paths::xclToml(other)));
    TitleConfig(Paths::xclToml(other)).set(QStringLiteral("device_type"), QStringLiteral("anaconda"));
    s.clearGameOverride(other, QStringLiteral("device_type"));
    QCOMPARE(LaunchConfig::deviceType(s, other), QStringLiteral("lockhart"));
}

void TestSettings::qualityPresets()
{
    qputenv("XSEMU_QUALITY_PRESETS", "1"); // the Core switches have not landed; the row is gated until they do
    QTemporaryDir dir;
    qputenv("XCL_DATA_ROOT", dir.filePath(QStringLiteral("xcl")).toUtf8());
    const QString ini = dir.filePath(QStringLiteral("s.ini"));
    const QString id = QStringLiteral("Northwind.Title");
    AppSettings s(ini);

    // Clear: neither switch, and no budget the launcher would have to explain
    QVERIFY(!LaunchConfig::qualityPreset(s, id));
    QCOMPARE(LaunchConfig::titleValues(s, id), (QVariantMap{{QStringLiteral("device_type"), QStringLiteral("lockhart")}}));
    QProcessEnvironment env = LaunchConfig::runEnvironment(QProcessEnvironment(), s, id);
    QVERIFY(!env.contains(QStringLiteral("XCL_UPSCALER")));
    QVERIFY(!env.contains(QStringLiteral("XCL_DLSS_MODE")));

    // each preset sets its DLSS mode and its title budget (the Series S standard is 8192 MiB)
    const QList<QPair<QString, QPair<QString, qint64>>> presets{
        {QStringLiteral("quality"), {QStringLiteral("quality"), 8192}},
        {QStringLiteral("balanced"), {QStringLiteral("balanced"), 6144}},
        {QStringLiteral("performance"), {QStringLiteral("performance"), 4096}},
    };
    for (const auto& [key, expected] : presets) {
        s.setGameOverride(id, QStringLiteral("qualityPreset"), key);
        env = LaunchConfig::runEnvironment(QProcessEnvironment(), s, id);
        QCOMPARE(env.value(QStringLiteral("XCL_UPSCALER")), QStringLiteral("dlss"));
        QCOMPARE(env.value(QStringLiteral("XCL_DLSS_MODE")), expected.first);
        QCOMPARE(LaunchConfig::titleValues(s, id).value(QStringLiteral("title_memory_mb")).toLongLong(), expected.second);
    }

    // the choice survives a restart, and prepareTitle writes the budget into xcl.toml
    AppSettings reread(ini);
    QVERIFY(LaunchConfig::qualityPreset(reread, id));
    QCOMPARE(LaunchConfig::titleValues(reread, id).value(QStringLiteral("title_memory_mb")).toLongLong(), 4096);
    QCOMPARE(LaunchConfig::runEnvironment(QProcessEnvironment(), reread, id).value(QStringLiteral("XCL_DLSS_MODE")), QStringLiteral("performance"));
    QVERIFY(LaunchConfig::prepareTitle(reread, id));
    QCOMPARE(TitleConfig(Paths::xclToml(id)).value(QStringLiteral("title_memory_mb")).toLongLong(), 4096);

    // a run profile line naming a switch is the more specific choice and wins over the preset
    reread.setRunProfile({QStringLiteral("XCL_DLSS_MODE=ultra-performance")});
    env = LaunchConfig::runEnvironment(QProcessEnvironment(), reread, id);
    QCOMPARE(env.value(QStringLiteral("XCL_UPSCALER")), QStringLiteral("dlss"));
    QCOMPARE(env.value(QStringLiteral("XCL_DLSS_MODE")), QStringLiteral("ultra-performance"));

    // a profile that replaces both switches leaves the warm-up identity unchanged across presets
    reread.setRunProfile({QStringLiteral("XCL_UPSCALER=dlss"), QStringLiteral("XCL_DLSS_MODE=quality")});
    const QString forced = LaunchConfig::profileIdentity(reread, id);
    reread.setGameOverride(id, QStringLiteral("qualityPreset"), QStringLiteral("quality"));
    QCOMPARE(LaunchConfig::profileIdentity(reread, id), forced);
    // with no profile override the preset is part of the identity
    reread.resetRunProfile();
    const QString quality = LaunchConfig::profileIdentity(reread, id);
    reread.setGameOverride(id, QStringLiteral("qualityPreset"), QStringLiteral("balanced"));
    QVERIFY(LaunchConfig::profileIdentity(reread, id) != quality);

    // a title_memory_mb the user set on purpose beats the preset's number
    reread.setGameOverride(id, QStringLiteral("title_memory_mb"), 2048);
    QCOMPARE(LaunchConfig::titleValues(reread, id).value(QStringLiteral("title_memory_mb")).toLongLong(), 2048);
    reread.clearGameOverride(id, QStringLiteral("title_memory_mb"));

    // an unknown stored value is Clear, not a crash or half a preset
    reread.resetRunProfile();
    reread.setGameOverride(id, QStringLiteral("qualityPreset"), QStringLiteral("ultra"));
    QVERIFY(!LaunchConfig::qualityPreset(reread, id));
    env = LaunchConfig::runEnvironment(QProcessEnvironment(), reread, id);
    QVERIFY(!env.contains(QStringLiteral("XCL_UPSCALER")) && !env.contains(QStringLiteral("XCL_DLSS_MODE")));

    // without a Core that reads the switches nothing applies, not even a stored choice
    reread.setGameOverride(id, QStringLiteral("qualityPreset"), QStringLiteral("quality"));
    qunsetenv("XSEMU_QUALITY_PRESETS");
    QVERIFY(!LaunchConfig::qualityPreset(reread, id));
    QVERIFY(!LaunchConfig::runEnvironment(QProcessEnvironment(), reread, id).contains(QStringLiteral("XCL_UPSCALER")));
    QVERIFY(!LaunchConfig::titleValues(reread, id).contains(QStringLiteral("title_memory_mb")));
}

void TestSettings::warmupClassification()
{
    WarmupController::Progress last;
    last.targets = 100;
    last.passDone = 60;
    const auto done = [](qint64 built, qint64 failed, qint64 left, bool complete, bool cancelled, int code) {
        return Protocol::WarmupDone{built, failed, left, 0, 12.5, complete, cancelled, code};
    };
    WarmupRun r = WarmupController::classify(0, false, done(98, 2, 2, true, false, 0), last, {}, false);
    QCOMPARE(r.result, WarmupRun::Result::Finished);
    QCOMPARE(r.done, 100);
    QCOMPARE(r.failed, 2);
    QCOMPARE(r.left, 2);
    // exit code 1 only says some builds failed: the pass still went through every target it meant to,
    // and what it left is what a later pass goes on with
    r = WarmupController::classify(1, false, done(90, 10, 10, true, false, 1), last, {}, false);
    QCOMPARE(r.result, WarmupRun::Result::Finished);
    QCOMPARE(r.left, 10);
    QCOMPARE(WarmupController::classify(4, false, done(40, 0, 60, false, true, 4), last, {}, true).result, WarmupRun::Result::Cancelled);
    last.pausedForRam = true;
    r = WarmupController::classify(0, false, done(40, 0, 60, false, false, 0), last, {}, false);
    QCOMPARE(r.result, WarmupRun::Result::StoppedPartWay);
    QVERIFY(r.stoppedForRam);
    QCOMPARE(r.left, 60);
    last.pausedForRam = false;
    // a crash has no done line: what the pass built is what its last progress line said, and what it
    // left is unknown rather than zero
    r = WarmupController::classify(3, true, std::nullopt, last, QStringLiteral("the driver compile crashed"), false);
    QCOMPARE(r.result, WarmupRun::Result::Failed);
    QCOMPARE(r.exitCode, 3);
    QCOMPARE(r.done, 60);
    QCOMPARE(r.left, qint64(-1));
    QCOMPARE(r.message, QStringLiteral("the driver compile crashed"));
    r = WarmupController::classify(3, false, std::nullopt, last, QStringLiteral("another XSEmu process is running"), false);
    QCOMPARE(r.result, WarmupRun::Result::Failed);
    QCOMPARE(r.message, QStringLiteral("another XSEmu process is running"));
    QCOMPARE(WarmupController::classify(9, true, std::nullopt, last, {}, true).result, WarmupRun::Result::Cancelled);
    QCOMPARE(WarmupController::phaseIndex(QStringLiteral("compute_blobs")), 4);
}

void TestSettings::warmupResume()
{
    // what Continue runs: the pass the newest unfinished run was running, and only that one
    const auto run = [](WarmupRun::Result result, qint64 done, qint64 left, bool full = false, const QString& profile = {}) {
        WarmupRun r;
        r.titleId = QStringLiteral("NorthwindInteractive.SignalGarden");
        r.runnerId = QString();
        r.result = result;
        r.done = done;
        r.left = left;
        r.full = full;
        r.profile = profile;
        return r;
    };
    const QString id = QStringLiteral("NorthwindInteractive.SignalGarden");
    const auto resumable = [&](const QList<WarmupRun>& runs, const std::optional<WarmupCheckpoint>& inFlight, const QString& profile = {}) {
        return WarmupController::resumable(runs, inFlight, id, QString(), profile);
    };
    QVERIFY(!resumable({}, std::nullopt).has_value());
    QVERIFY(!resumable({run(WarmupRun::Result::Finished, 3120, 0)}, std::nullopt).has_value());
    QVERIFY(!resumable({run(WarmupRun::Result::NothingToBuild, 0, -1)}, std::nullopt).has_value());
    // a pass that finished can still leave builds it could not make, and those are Continue's
    std::optional<WarmupController::Resume> r = resumable({run(WarmupRun::Result::Finished, 3120, 16)}, std::nullopt);
    QVERIFY(r.has_value());
    QCOMPARE(r->left, 16);
    // a failed build is not built: the count Continue names leaves it to the leftovers
    WarmupRun someFailed = run(WarmupRun::Result::Cancelled, 100, 20);
    someFailed.failed = 10;
    r = resumable({someFailed}, std::nullopt);
    QVERIFY(r.has_value());
    QCOMPARE(r->built, 90);
    QCOMPARE(r->left, 20);
    // a crash that built something: Continue goes on from it, with the same pass and no count at all,
    // because progress counts pipelines the pass dealt with instead of pipelines it built
    r = resumable({run(WarmupRun::Result::Failed, 2180, -1), run(WarmupRun::Result::Finished, 3120, 0)}, std::nullopt);
    QVERIFY(r.has_value());
    QCOMPARE(r->built, qint64(-1));
    QCOMPARE(r->left, qint64(-1));
    QVERIFY(!r->full);
    // a refusal before the pass started left nothing to continue
    QVERIFY(!resumable({run(WarmupRun::Result::Failed, 0, -1)}, std::nullopt).has_value());
    // a full pass stopped by its budget keeps --full: the next pass is the one it was
    r = resumable({run(WarmupRun::Result::StoppedPartWay, 1000, 400, true)}, std::nullopt);
    QVERIFY(r.has_value());
    QVERIFY(r->full);
    QCOMPARE(r->left, 400);
    // a changed run profile means its keys are not this profile's keys
    QVERIFY(!resumable({run(WarmupRun::Result::Cancelled, 100, 20, false, QStringLiteral("a|b"))}, std::nullopt, QStringLiteral("c|d")).has_value());
    QVERIFY(resumable({run(WarmupRun::Result::Cancelled, 100, 20, false, QStringLiteral("a|b"))}, std::nullopt, QStringLiteral("a|b")).has_value());
    QVERIFY(resumable({run(WarmupRun::Result::Cancelled, 100, 20, false, QString())}, std::nullopt, QStringLiteral("a|b")).has_value());
    // another game's run is not this game's
    WarmupRun other = run(WarmupRun::Result::Cancelled, 100, 20);
    other.titleId = QStringLiteral("Other.Title");
    QVERIFY(!resumable({other}, std::nullopt).has_value());
    // a pass the launcher did not see the end of (it closed or died) beats the recorded runs, and the
    // phase it had reached is all it can say
    WarmupCheckpoint pass;
    pass.titleId = id;
    pass.full = true;
    pass.phase = QStringLiteral("bound");
    pass.startedAt = QDateTime::currentDateTime();
    r = resumable({run(WarmupRun::Result::Cancelled, 100, 20)}, pass);
    QVERIFY(r.has_value());
    QCOMPARE(r->built, qint64(-1));
    QCOMPARE(r->phase, QStringLiteral("bound"));
    QVERIFY(r->full);
}

void TestSettings::warmupCheckpointRoundTrip()
{
    QTemporaryDir dir;
    QVERIFY(dir.isValid());
    const QString path = dir.filePath(QStringLiteral("settings.ini"));
    {
        AppSettings settings(path);
        QVERIFY(!settings.warmupInFlight().has_value());
        WarmupCheckpoint pass;
        pass.titleId = QStringLiteral("NorthwindInteractive.SignalGarden");
        pass.titleName = QStringLiteral("Signal Garden");
        pass.profile = QStringLiteral("lockhart|hlsl");
        pass.full = true;
        pass.phase = QStringLiteral("bound");
        pass.startedAt = QDateTime::currentDateTime();
        settings.setWarmupInFlight(pass);
        // a new launcher process reads it back: that is what makes Continue survive a restart
        AppSettings again(path);
        const auto back = again.warmupInFlight();
        QVERIFY(back.has_value());
        QCOMPARE(back->titleId, pass.titleId);
        QCOMPARE(back->phase, QStringLiteral("bound"));
        QCOMPARE(back->profile, QStringLiteral("lockhart|hlsl"));
        QVERIFY(back->full);
        QCOMPARE(back->startedAt.toSecsSinceEpoch(), pass.startedAt.toSecsSinceEpoch());
        again.setWarmupInFlight(std::nullopt);
        QVERIFY(!again.warmupInFlight().has_value());
    }
}

void TestSettings::formats()
{
    QCOMPARE(Format::count(41906), QStringLiteral("41,906"));
    QCOMPARE(Format::duration(60), QStringLiteral("1 min 00 s"));
    QCOMPARE(Format::shortDuration(192), QStringLiteral("3m 12s"));
    QCOMPARE(Format::shortDuration(22), QStringLiteral("22s"));
    QCOMPARE(Format::gigabytesFromMb(12288), QStringLiteral("12.0 GB"));
    QCOMPARE(Format::exitCode(3), QStringLiteral("3"));
    QCOMPARE(Format::exitCode(int(0xC0000005u)), QStringLiteral("0xC0000005"));
    const QDateTime now = QDateTime::currentDateTime();
    QCOMPARE(Format::ago(now.addSecs(-30), now), QStringLiteral("Just now"));
    QCOMPARE(Format::ago(now.addDays(-1), now), QStringLiteral("Yesterday"));
}

QTEST_GUILESS_MAIN(TestSettings)
#include "tst_settings.moc"
