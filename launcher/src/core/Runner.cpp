#include "Runner.h"

#include "Paths.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QSaveFile>
#include <QThread>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

QString RunnerSpec::pinnedVersion() const
{
    if (install == Install::ReleaseAsset)
        return releaseTag;
    if (install == Install::BuildFromSource)
        return commit.left(7);
    return {};
}

bool RunnerSpec::isPinned() const
{
#ifndef Q_OS_WIN
    if (windowsOnly)
        return false;
#endif
    switch (install) {
    case Install::BuiltIn: return true;
    case Install::ReleaseAsset: return !releaseTag.isEmpty() && !assetName.isEmpty() && assetSha256.size() == 64;
    case Install::BuildFromSource: return commit.size() == 40;
    }
    return false;
}

QString RunnerSpec::downloadUrl() const
{
    if (!assetUrl.isEmpty())
        return assetUrl;
    return QStringLiteral("https://github.com/%1/releases/download/%2/%3").arg(repo, releaseTag, assetName);
}

QString RunnerSpec::dataDir() const { return Paths::runnerDir(id); }
QString RunnerSpec::installDir() const
{
    return QDir(dataDir()).filePath(install == Install::BuildFromSource ? QStringLiteral("bin") : QStringLiteral("app"));
}

QString RunnerSpec::checkoutDir() const { return QDir(dataDir()).filePath(QStringLiteral("s")); }
QString RunnerSpec::targetDir() const { return QDir(dataDir()).filePath(QStringLiteral("t")); }

QList<QPair<QString, QString>> RunnerSpec::tokens(const QString& gameDir, const QString& gameExe, const QString& titleId,
                                                  const QString& deviceType, quintptr hostHwnd) const
{
    const auto native = [](const QString& path) { return QDir::toNativeSeparators(path); };
    return {{QStringLiteral("gameDir"), native(gameDir)},
            {QStringLiteral("gameExe"), native(gameExe)},
            {QStringLiteral("gameExeDir"), gameExe.isEmpty() ? QString() : native(QFileInfo(gameExe).absolutePath())},
            {QStringLiteral("titleId"), titleId},
            {QStringLiteral("console"), deviceType == QLatin1String("anaconda") ? QStringLiteral("series-x") : QStringLiteral("series-s")},
            {QStringLiteral("runnerDir"), native(dataDir())},
            {QStringLiteral("installDir"), native(installDir())},
            {QStringLiteral("cacheDir"), native(resolvedCacheDir())},
            {QStringLiteral("jobs"), QString::number(qMax(1, QThread::idealThreadCount()))},
            {QStringLiteral("rtFlag"), QStringLiteral("--no-rt")},
            {QStringLiteral("hostHwnd"), QStringLiteral("0x%1").arg(hostHwnd, 0, 16)}};
}

QString RunnerSpec::resolvedCacheDir() const
{
    if (cacheDir.isEmpty())
        return {};
    const QString d = QDir::cleanPath(expand(cacheDir, {{QStringLiteral("runnerDir"), dataDir()}, {QStringLiteral("installDir"), installDir()}}));
    return QDir::isAbsolutePath(d) ? d : QDir(Paths::localAppDataDir()).filePath(d);
}

QProcessEnvironment RunnerSpec::processEnvironment(const QList<QPair<QString, QString>>& values, const QHash<QString, bool>& choices) const
{
    QProcessEnvironment env = QProcessEnvironment::systemEnvironment();
    for (const QString& name : clearEnvironment)
        env.remove(name);
    QList<QPair<QString, QString>> all = environment;
    for (const Option& o : options) {
        if (optionOn(o, choices))
            all += o.environment;
    }
    for (const auto& [name, value] : all) {
        const QString v = expand(value, values);
        env.insert(name, v);
        if (name.endsWith(QLatin1String("_DIR")))
            QDir().mkpath(v);
    }
    return env;
}

QString RunnerSpec::writeConfig(const QList<QPair<QString, QString>>& values) const
{
    if (configFile.isEmpty())
        return {};
    const auto quoted = [](QString s) {
        s.replace(QLatin1Char('\\'), QLatin1String("\\\\")).replace(QLatin1Char('"'), QLatin1String("\\\""));
        return QLatin1Char('"') + s + QLatin1Char('"');
    };
    QString text = QStringLiteral("# Written by XSEmu before every run and pre-cache of %1; changes here are replaced.\n[paths]\n").arg(name);
    for (const ConfigPath& p : configPaths) {
        const QString value = expand(p.value, values);
        if (value.isEmpty())
            continue;
        const QString path = QDir::toNativeSeparators(QDir::cleanPath(value));
        if (p.create && !QDir().mkpath(path))
            return QStringLiteral("XSEmu could not create %1.").arg(path);
        text += p.key + QStringLiteral(" = ") + quoted(path) + QLatin1Char('\n');
    }
    const QString file = QDir::cleanPath(expand(configFile, values));
    QDir().mkpath(QFileInfo(file).absolutePath());
    QSaveFile f(file);
    if (!f.open(QIODevice::WriteOnly) || f.write(text.toUtf8()) < 0 || !f.commit())
        return QStringLiteral("XSEmu could not write %1: %2").arg(QDir::toNativeSeparators(file), f.errorString());
    return {};
}

QString RunnerSpec::cloneUrl() const
{
    if (repo.contains(QLatin1String("://")) || QDir::isAbsolutePath(repo))
        return repo; // a feed may point at a local repository
    return QStringLiteral("https://github.com/%1.git").arg(repo);
}

QStringList RunnerSpec::fetchUrls() const { return remotes.isEmpty() ? QStringList{cloneUrl()} : remotes; }

QString RunnerSpec::expand(const QString& arg, const QList<QPair<QString, QString>>& values)
{
    QString a = arg;
    for (const auto& [key, value] : values)
        a.replace(QLatin1Char('{') + key + QLatin1Char('}'), value);
    return a;
}

QStringList RunnerSpec::expand(const QStringList& args, const QList<QPair<QString, QString>>& values)
{
    QStringList out;
    out.reserve(args.size());
    for (const QString& a : args)
        out << expand(a, values);
    return out;
}

namespace Runners {

namespace {

QList<RunnerSpec> build()
{
    QList<RunnerSpec> list;

    RunnerSpec x;
    x.id = QLatin1String(XSEmu);
    x.name = QStringLiteral("XSEmu");
    x.summary = QStringLiteral("Built in. Embeds the game in this window.");
    x.repo = QStringLiteral("zortos293/xsemu");
    x.licence = QStringLiteral("GPL-3.0-or-later");
    x.install = RunnerSpec::Install::BuiltIn;
    x.stop = RunnerSpec::Stop::ControlPipe;
    x.embed = RunnerSpec::Embed::HostHandle;
    x.canPause = true;
    x.canOverlay = true;
    list << x;

    // Third-party runners, read from each repository at the pinned commit or release. Both VIctory
    // forks publish no releases: they are built on this PC from the pinned commit (we never
    // redistribute builds). Neither may be reparented: the game's top-level window lives on its game
    // thread, which stalls for seconds, and keyboard input is gated on it being the foreground window.
    // Both read [paths] from VICTORY_CONFIG and stop when it names a file that is not there, and both
    // refuse keys they do not know: only game_dir, cache_dir and data_dir are written. The cache and data
    // sit beside the install (bin\), so they move with the runner.
    const QString victoryCache = QStringLiteral("{runnerDir}/cache");
    const QList<QPair<QString, QString>> victoryEnvironment = {{QStringLiteral("VICTORY_CONFIG"), QStringLiteral("{runnerDir}/data/victory.toml")},
                                                               {QStringLiteral("VICTORY_CACHE_DIR"), victoryCache},
                                                               {QStringLiteral("VICTORY_DATA_DIR"), QStringLiteral("{runnerDir}/data")}};
    const QList<RunnerSpec::ConfigPath> victoryPaths = {{QStringLiteral("game_dir"), QStringLiteral("{gameDir}"), false},
                                                        {QStringLiteral("cache_dir"), victoryCache, true},
                                                        {QStringLiteral("data_dir"), QStringLiteral("{runnerDir}/data"), true}};
    RunnerSpec v1;
    v1.id = QStringLiteral("victory-7900xtx");
    v1.name = QStringLiteral("VIctory 7900 XTX");
    v1.repo = QStringLiteral("avatarmanz/VIctory-7900XTX");
    v1.badge = QStringLiteral("AMD");
    v1.recommendFor = RunnerSpec::Vendor::Amd;
    v1.recommendWhy = QStringLiteral("VIctory 7900 XTX has code paths written for Radeon GPUs");
    v1.recommendMinRamBytes = 64ULL * 1024 * 1024 * 1024;
    v1.summary = QStringLiteral("VIctory fork with code paths for Radeon GPUs.");
    v1.licence = QStringLiteral("GPL-3.0");
    v1.note = QStringLiteral("Needs 64 GB of RAM. Its GitHub repository was deleted; XSEmu fetches the audited commit from the upstream VIctory repository.");
    v1.windowsOnly = true;
    v1.install = RunnerSpec::Install::BuildFromSource;
    v1.commit = QStringLiteral("c9c932cbcc38defd8f1683d62dd298187f565031");
    v1.tree = QStringLiteral("2c208def544ad8df2b06a6da082522554b73f8cf");
    // the fork's own GitHub repository (and its author's account) is gone: the pinned commit is fetched
    // from the upstream VIctory repository, which carries it, and from a mirror of the fork
    v1.remotes = {QStringLiteral("https://github.com/Marvisak/VIctory.git"),
                  QStringLiteral("https://git.gtahub.st/github/avatarmanz--VIctory-7900XTX.git")};
    v1.toolchains = {QStringLiteral("git"), QStringLiteral("msvc"), QStringLiteral("rust")};
    v1.buildSteps = {{QStringLiteral("cargo"), QStringLiteral("build"), QStringLiteral("--locked"), QStringLiteral("--profile"), QStringLiteral("diag"),
                      QStringLiteral("-p"), QStringLiteral("victory"), QStringLiteral("--features"), QStringLiteral("diagnostics")}};
    v1.artifacts = {{QStringLiteral("diag/victory.exe"), QStringLiteral("victory.exe")}};
    v1.executable = QStringLiteral("victory.exe");
    v1.buildNote = QStringLiteral("VIctory 7900 XTX is built from source on this PC. It needs Git, Rust and the Visual Studio C++ build tools (installed if missing) and can take a while; build times have not been measured yet.");
    v1.launchArgs = {QStringLiteral("run"), QStringLiteral("{gameExe}"), QStringLiteral("--execute"), QStringLiteral("--console"), QStringLiteral("{console}")};
    // runs with VICTORY_PIPELINE_BOOK=1 write every pipeline the title creates to known-pipelines.bin, and
    // the library lists those it drew with in early-pipelines.txt; "warm --used" builds keys in both
    v1.environment = victoryEnvironment + QList<QPair<QString, QString>>{{QStringLiteral("VICTORY_PIPELINE_BOOK"), QStringLiteral("1")}};
    v1.configFile = QStringLiteral("{runnerDir}/data/victory.toml");
    v1.configPaths = victoryPaths;
    v1.precache = {{{QStringLiteral("shaders"), QStringLiteral("--jobs"), QStringLiteral("{jobs}"), QStringLiteral("{gameDir}")}, QStringLiteral("victory-shaders"), false,
                    RunnerSpec::PrecacheCancel::Kill, QStringLiteral("Translating shaders")},
                   {{QStringLiteral("warm"), QStringLiteral("--used"), QStringLiteral("--jobs"), QStringLiteral("{jobs}")}, QStringLiteral("victory-warm"), false,
                    RunnerSpec::PrecacheCancel::Kill, QStringLiteral("Building pipelines"),
                    {QStringLiteral("{cacheDir}/pipelines/known-pipelines.bin"), QStringLiteral("{cacheDir}/pipelines/early-pipelines.txt")},
                    QStringLiteral("Play once with VIctory 7900 XTX to record pipelines; the next warm-up builds them.")}};
    v1.precacheNote = QStringLiteral("Translates every shader in the game, then builds the pipelines earlier runs used. Its warm-up wants 64 GB of RAM.");
    v1.cacheDir = victoryCache;
    v1.stop = RunnerSpec::Stop::SysCommandClose;
    v1.stopGraceMs = 5000;
    v1.embed = RunnerSpec::Embed::Docked;
    list << v1;

    RunnerSpec v2 = v1;
    v2.id = QStringLiteral("victory-dexnom");
    v2.name = QStringLiteral("VIctory (DexNom)");
    v2.repo = QStringLiteral("DexNom/VIctory");
    v2.badge = QStringLiteral("AMD (tested on RX 7900 XTX)");
    v2.recommendWhy = QStringLiteral("VIctory (DexNom) is tested on a Radeon RX 7900 XTX");
    v2.recommendMinRamBytes = 0;
    v2.summary = QStringLiteral("VIctory fork tested on a Radeon RX 7900 XTX.");
    v2.note.clear();
    v2.environment = victoryEnvironment;
    v2.commit = QStringLiteral("64bd58f5fb794216894bc0e33e03a3a64d19c347");
    v2.tree = QStringLiteral("34844940e7b4325f8c7cc5a202b29460d29cfa74");
    // the BCPack format tables the build embeds, fetched from where its own setup script fetches them
    v2.fetches = {{QStringLiteral("https://raw.githubusercontent.com/Marvisak/VIctory/2849477415a46f5a079c85cf3e64a3dff32f713c/crates/victory-formats/src/bcpack-tables.bin"),
                   QStringLiteral("1f1d4760419202bb5672f0c428ce51e456b5d08fbaec8f371d043fd94e01c36b"),
                   QStringLiteral("crates/victory-formats/src/bcpack-tables.bin")}};
    v2.buildSteps = {{QStringLiteral("cargo"), QStringLiteral("build"), QStringLiteral("--locked"), QStringLiteral("--release"), QStringLiteral("-p"), QStringLiteral("victory"),
                      QStringLiteral("--features"), QStringLiteral("diagnostics"), QStringLiteral("--bins"), QStringLiteral("--target"), QStringLiteral("x86_64-pc-windows-msvc")}};
    v2.artifacts = {{QStringLiteral("x86_64-pc-windows-msvc/release/victory.exe"), QStringLiteral("victory.exe")}};
    v2.buildNote = QStringLiteral("VIctory (DexNom) is built from source on this PC. It needs Git, Rust and the Visual Studio C++ build tools (installed if missing) and can take a while; build times have not been measured yet.");
    v2.launchArgs = {QStringLiteral("run"), QStringLiteral("{gameExe}"), QStringLiteral("--execute"), QStringLiteral("--console"), QStringLiteral("{console}"), QStringLiteral("{rtFlag}")};
    v2.precache = {{{QStringLiteral("warm"), QStringLiteral("{rtFlag}"), QStringLiteral("--jobs"), QStringLiteral("{jobs}"), QStringLiteral("{gameDir}")}, QStringLiteral("victory-dexnom"), false,
                    RunnerSpec::PrecacheCancel::Kill, QStringLiteral("Translating shaders")}};
    v2.precacheNote = QStringLiteral("Translates the game's shaders and builds its compute pipelines.");
    list << v2;

    // stellarr-studio/kelvaroq v0.2.0 (0a3c30d): vendor-neutral and capability-probed, measured on an
    // RTX 5080. Its keyboard input needs its own top-level window in the foreground and its swap chain
    // binds to the first visible top-level window, so it is never reparented: it keeps its own window.
    RunnerSpec k;
    k.id = QStringLiteral("kelvaroq");
    k.name = QStringLiteral("Kelvaroq");
    k.repo = QStringLiteral("stellarr-studio/kelvaroq");
    k.badge = QStringLiteral("Vendor-neutral \u00B7 tuned on RTX 5080");
    k.recommendFor = RunnerSpec::Vendor::Nvidia;
    k.recommendWhy = QStringLiteral("Kelvaroq is vendor-neutral and was tuned on an RTX 5080");
    k.summary = QStringLiteral("Vendor-neutral; its authors measured it on a GeForce RTX 5080.");
    k.licence = QStringLiteral("GPL-3.0");
    k.note = QStringLiteral("Kelvaroq reports the game's licence as active and always runs offline; XSEmu's offline store error does not apply to it.");
    k.windowsOnly = true;
    k.install = RunnerSpec::Install::ReleaseAsset;
    k.releaseTag = QStringLiteral("v0.2.0");
    k.assetName = QStringLiteral("Kelvaroq-0.2.0-windows-x64.zip");
    k.assetSha256 = QStringLiteral("c2f99a3cdaab06bcf025d0f5a1b0c25143dd8ce7edfd4362edbfd84f9d1d5885");
    k.executable = QStringLiteral("native/kelvaroq-engine.exe");
    k.fileSha256 = {{QStringLiteral("native/kelvaroq-engine.exe"), QStringLiteral("1b3efe68185bd2dc8c7bf36fa8c087c6459dacb09dc61db4881970c9c84d6eae")}};
    k.copyFiles = {{QStringLiteral("vcruntime140.dll"), QStringLiteral("native/vcruntime140.dll")}};
    k.launchArgs = {QStringLiteral("run"), QStringLiteral("{gameExe}"), QStringLiteral("--console"), QStringLiteral("{console}"), QStringLiteral("--execute")};
    k.environment = {{QStringLiteral("VICTORY_GAME_DIR"), QStringLiteral("{gameExeDir}")},
                     {QStringLiteral("VICTORY_CACHE_DIR"), QStringLiteral("{runnerDir}/cache")},
                     {QStringLiteral("VICTORY_DATA_DIR"), QStringLiteral("{runnerDir}/data")},
                     {QStringLiteral("VICTORY_LOG"), QStringLiteral("info")}};
    k.precache = {{{QStringLiteral("precompile"), QStringLiteral("--check")}, QStringLiteral("kelvaroq"), true, RunnerSpec::PrecacheCancel::Kill, QStringLiteral("Checking what to build"),
                   {}, QStringLiteral("Play once with Kelvaroq, then warm up.")},
                  {{QStringLiteral("precompile")}, QStringLiteral("kelvaroq"), false, RunnerSpec::PrecacheCancel::CloseStdin, QStringLiteral("Building pipelines")}};
    k.precacheNote = QStringLiteral("Kelvaroq learns which pipelines a game needs while it plays, so its pre-cache has work only after the game ran once with it.");
    k.cacheDir = QStringLiteral("{runnerDir}/cache");
    k.windowReadyLine = QStringLiteral("presenting to a window");
    k.stop = RunnerSpec::Stop::SysCommandClose;
    k.stopGraceMs = 10000;
    k.embed = RunnerSpec::Embed::Docked;
    list << k;

    // codec_d's cool_fox at f4b48ad (tree 832b1ea), audited 2026-11-26: avatarmanz's VIctory 7900 XTX
    // plus six commits for a 32 GB RTX 5090 box. The GitHub original is gone, so three copies of the
    // same commit are tried in turn and the commit and tree ids are checked. Built from source only:
    // never its README's prebuilt exe or its Play-VIctory.cmd, whose environment (lines 20-37, less the
    // ambient light, which is a switch here) is set below. It reads paths from the environment and
    // no config file, and VICTORY_KEYS (scripted key presses) is never passed on.
    RunnerSpec f;
    f.id = QStringLiteral("coolfox");
    f.name = QStringLiteral("cool_fox (codec_d)");
    f.repo = QStringLiteral("codec-d/cool_fox");
    f.badge = QStringLiteral("NVIDIA \u00B7 tuned on RTX 5090 (32 GB RAM)");
    f.summary = QStringLiteral("VIctory fork tuned on a GeForce RTX 5090 with 32 GB of RAM; 64 GB recommended.");
    f.licence = QStringLiteral("GPL-3.0");
    f.note = QStringLiteral("cool_fox reports the game's licence as active. It signs the DXIL it makes without validating it, and skips some of the "
                            "title's ray-tracing shaders by name. It always runs the game as a Series S, whatever the game's console setting. "
                            "Its GitHub repository was deleted; XSEmu builds the audited commit. Needs 32 GB of RAM.");
    f.windowsOnly = true;
    f.install = RunnerSpec::Install::BuildFromSource;
    f.commit = QStringLiteral("f4b48ad766ddbec1a1867022cd42f7645b0dd55e");
    f.tree = QStringLiteral("832b1ea59576e5e2688d49c2530b67f0efae43bc");
    f.remotes = {QStringLiteral("https://github.com/cputroll/cgbj-reupload.git"), QStringLiteral("https://git.gtahub.st/github/codec-d--cool_fox.git"),
                 QStringLiteral("https://github.com/DoubleGhosty/cool_fox.git")};
    f.toolchains = {QStringLiteral("git"), QStringLiteral("msvc"), QStringLiteral("rust")};
    f.buildSteps = {{QStringLiteral("cargo"), QStringLiteral("build"), QStringLiteral("--release"), QStringLiteral("--locked"), QStringLiteral("-p"), QStringLiteral("victory")}};
    f.artifacts = {{QStringLiteral("release/victory.exe"), QStringLiteral("victory.exe")}};
    f.executable = QStringLiteral("victory.exe");
    f.buildNote = QStringLiteral("cool_fox is built from source on this PC, from a commit that was checked for malware. It needs Git, Rust and the Visual Studio C++ build tools (installed if missing) and can take a while; build times have not been measured yet.");
    f.launchArgs = {QStringLiteral("run"), QStringLiteral("{gameExe}"), QStringLiteral("--console"), QStringLiteral("series-s"), QStringLiteral("--execute")};
    f.environment = {{QStringLiteral("VICTORY_GAME_DIR"), QStringLiteral("{gameDir}")},
                     {QStringLiteral("VICTORY_CACHE_DIR"), QStringLiteral("{runnerDir}/cache")},
                     {QStringLiteral("VICTORY_DATA_DIR"), QStringLiteral("{runnerDir}/data")},
                     {QStringLiteral("VICTORY_LOG"), QStringLiteral("warn")},
                     {QStringLiteral("VICTORY_PIPELINE_BOOK"), QStringLiteral("1")}};
    for (const auto& [name, value] : QList<QPair<const char*, const char*>>{
             {"TEXTURE_NO_WAIT", "1"}, {"RELAY", "2"}, {"WIDE_BACK", "1"}, {"CARD_REHASH_RUNS", "1"}, {"JOIN_COPIES", "1"}, {"INDIRECT_WINDOW", "1"},
             {"RELAY_COPY", "1"}, {"CARD_STEPS", "2"}, {"INDIRECT_BATCH", "2"}, {"MAP_THREADS", "1"}, {"CARD_AWAKE", "1"}, {"SHOW_AFTER", "1"},
             {"THIN", "1"}, {"JOIN_PARTS", "8"}, {"UPLOAD_AHEAD", "1"}, {"LEVELS_ON_DEMAND", "1"}, {"CARD_BUFFERS_MIN_KB", "64"}})
        f.environment << qMakePair(QStringLiteral("VICTORY_") + QLatin1String(name), QString::fromLatin1(value));
    f.clearEnvironment = {QStringLiteral("VICTORY_KEYS"), QStringLiteral("VICTORY_CONFIG"), QStringLiteral("VICTORY_AMBIENT"), QStringLiteral("VICTORY_AMBIENT_AIR")};
    f.options = {{QStringLiteral("ambient"), QStringLiteral("Author's ambient light"),
                  QStringLiteral("The stand-in light codec_d plays with (VICTORY_AMBIENT=0.8, VICTORY_AMBIENT_AIR=1.5). Not the game's own lighting."), false,
                  {{QStringLiteral("VICTORY_AMBIENT"), QStringLiteral("0.8")}, {QStringLiteral("VICTORY_AMBIENT_AIR"), QStringLiteral("1.5")}}}};
    f.precache = {{{QStringLiteral("shaders"), QStringLiteral("--jobs"), QStringLiteral("{jobs}"), QStringLiteral("{gameDir}")}, QStringLiteral("coolfox-shaders"), false,
                   RunnerSpec::PrecacheCancel::Kill, QStringLiteral("Translating shaders")},
                  {{QStringLiteral("warm"), QStringLiteral("--used"), QStringLiteral("--jobs"), QStringLiteral("{jobs}")}, QStringLiteral("victory-warm"), false,
                   RunnerSpec::PrecacheCancel::Kill, QStringLiteral("Building pipelines"), {QStringLiteral("{cacheDir}/pipelines/known-pipelines.bin")},
                   QStringLiteral("Play once with cool_fox to record pipelines; the next warm-up builds them.")}};
    f.precacheNote = QStringLiteral("Translates every shader in the game (about 26,000), then builds the pipelines earlier runs drew with.");
    f.recommendMinRamBytes = 32ULL * 1024 * 1024 * 1024;
    f.cacheDir = QStringLiteral("{runnerDir}/cache");
    f.stop = RunnerSpec::Stop::SysCommandClose;
    f.stopGraceMs = 5000;
    f.embed = RunnerSpec::Embed::Docked;
    list << f;

    return list;
}

}

// %XSEMU_RUNNER_FEED% names a JSON file that re-pins runners, for tests and the screenshot lever:
// {"kelvaroq": {"tag": "...", "asset": "...", "sha256": "...", "url": "file:///...", "executable": "...",
//   "launchArgs": [...], "precache": [{"args": [...], "parser": "...", "check": true, "closeStdin": true}],
//   "cacheDir": "..."}, "victory-dexnom": {"commit": "...", "repo": "file:///...", "buildSteps": [[...]],
//   "artifacts": {"built": "in-bin"}}}
// {"keep": true} keeps the curated pin and only lifts the Windows-only limit.
static void applyFeed(QList<RunnerSpec>& list)
{
    const QString path = qEnvironmentVariable("XSEMU_RUNNER_FEED");
    if (path.isEmpty())
        return;
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly))
        return;
    const QJsonObject feed = QJsonDocument::fromJson(f.readAll()).object();
    auto strings = [](const QJsonValue& v) {
        QStringList out;
        for (const QJsonValue& s : v.toArray())
            out << s.toString();
        return out;
    };
    for (RunnerSpec& r : list) {
        const QJsonObject o = feed.value(r.id).toObject();
        if (o.isEmpty() || r.isBuiltIn())
            continue;
        r.windowsOnly = false;
        if (o.value(QLatin1String("keep")).toBool())
            continue; // the curated pin as it is, on any OS: a stand-in program plays the runner
        if (o.contains(QLatin1String("commit"))) {
            r.install = RunnerSpec::Install::BuildFromSource;
            r.commit = o.value(QLatin1String("commit")).toString();
            r.repo = o.value(QLatin1String("repo")).toString(r.repo);
            r.tree = o.value(QLatin1String("tree")).toString();
            r.remotes = strings(o.value(QLatin1String("remotes")));
        } else {
            r.install = RunnerSpec::Install::ReleaseAsset;
            r.releaseTag = o.value(QLatin1String("tag")).toString();
            r.assetName = o.value(QLatin1String("asset")).toString();
            r.assetSha256 = o.value(QLatin1String("sha256")).toString().toLower();
            r.assetUrl = o.value(QLatin1String("url")).toString();
        }
        r.executable = o.value(QLatin1String("executable")).toString(r.executable);
        if (o.contains(QLatin1String("launchArgs")))
            r.launchArgs = strings(o.value(QLatin1String("launchArgs")));
        if (o.contains(QLatin1String("precache"))) {
            r.precache.clear();
            for (const QJsonValue& step : o.value(QLatin1String("precache")).toArray()) {
                const QJsonObject st = step.toObject();
                r.precache << RunnerSpec::PrecacheStep{strings(st.value(QLatin1String("args"))), st.value(QLatin1String("parser")).toString(r.id),
                                                       st.value(QLatin1String("check")).toBool(),
                                                       st.value(QLatin1String("closeStdin")).toBool() ? RunnerSpec::PrecacheCancel::CloseStdin : RunnerSpec::PrecacheCancel::Kill};
            }
        }
        r.cacheDir = o.value(QLatin1String("cacheDir")).toString(r.cacheDir);
        r.fileSha256.clear();
        r.copyFiles.clear();
        r.fetches.clear();
        r.toolchains.clear();
        r.environment.clear();
        r.options.clear();
        r.configFile.clear();
        r.configPaths.clear();
        if (o.contains(QLatin1String("buildSteps"))) {
            r.buildSteps.clear();
            for (const QJsonValue& step : o.value(QLatin1String("buildSteps")).toArray())
                r.buildSteps << strings(step);
        }
        if (o.contains(QLatin1String("artifacts"))) {
            r.artifacts.clear();
            const QJsonObject art = o.value(QLatin1String("artifacts")).toObject();
            for (auto it = art.begin(); it != art.end(); ++it)
                r.artifacts << qMakePair(it.key(), it.value().toString());
        }
    }
}

const QList<RunnerSpec>& curated()
{
    static const QList<RunnerSpec> list = [] {
        QList<RunnerSpec> l = build();
        applyFeed(l);
        return l;
    }();
    return list;
}

const RunnerSpec* find(const QString& id)
{
    for (const RunnerSpec& r : curated()) {
        if (r.id == id)
            return &r;
    }
    return nullptr;
}

const RunnerSpec& xsemu()
{
    return curated().first();
}

QString vendorName(quint32 vendorId)
{
    switch (vendorId) {
    case kVendorNvidia: return QStringLiteral("NVIDIA");
    case kVendorAmd: return QStringLiteral("AMD");
    case kVendorIntel: return QStringLiteral("Intel");
    default: return {};
    }
}

}
