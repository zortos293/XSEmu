#include "UpdateService.h"

#include "AppSettings.h"
#include "EmulatorBuild.h"
#include "Paths.h"
#include "ProcessGate.h"

#include <QCoreApplication>
#include <QDir>
#include <QDirIterator>
#include <QEventLoop>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QMetaObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QProcess>
#include <QSaveFile>
#include <QThread>

#include <algorithm>
#include <memory>

namespace {

// One GET into memory or into a file. GitHub's API wants a User-Agent, and its asset downloads
// redirect to a storage host, so redirects are followed.
bool httpGet(const QUrl& url, const QByteArray& userAgent, QIODevice* sink, QByteArray* body,
             const std::function<bool(qint64, qint64)>& progress, int timeoutMs, QString* error)
{
    QNetworkAccessManager nam;
    QNetworkRequest req(url);
    req.setHeader(QNetworkRequest::UserAgentHeader, userAgent);
    req.setRawHeader("Accept", "application/vnd.github+json");
    req.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::NoLessSafeRedirectPolicy);
    if (timeoutMs > 0)
        req.setTransferTimeout(timeoutMs);
    QNetworkReply* reply = nam.get(req);
    QEventLoop loop;
    bool cancelled = false;
    QObject::connect(reply, &QNetworkReply::finished, &loop, &QEventLoop::quit);
    QObject::connect(reply, &QNetworkReply::readyRead, &loop, [reply, sink, body] {
        const QByteArray chunk = reply->readAll();
        if (sink)
            sink->write(chunk);
        else if (body)
            body->append(chunk);
    });
    QObject::connect(reply, &QNetworkReply::downloadProgress, &loop, [&](qint64 done, qint64 total) {
        if (progress && !cancelled && !progress(done, total)) {
            cancelled = true;
            reply->abort();
        }
    });
    loop.exec();
    const QByteArray rest = reply->readAll();
    if (sink)
        sink->write(rest);
    else if (body)
        body->append(rest);
    const QNetworkReply::NetworkError code = reply->error();
    const QString message = reply->errorString();
    reply->deleteLater();
    if (cancelled) {
        if (error)
            *error = QStringLiteral("cancelled");
        return false;
    }
    if (code != QNetworkReply::NoError) {
        if (error)
            *error = message;
        return false;
    }
    return true;
}

QByteArray fetchDefault(const QUrl& url, const QByteArray& userAgent, int timeoutMs, QString* error)
{
    QByteArray body;
    if (!httpGet(url, userAgent, nullptr, &body, {}, timeoutMs, error))
        return {};
    return body;
}

bool downloadDefault(const QUrl& url, const QByteArray& userAgent, const QString& file,
                     const std::function<bool(qint64, qint64)>& progress, QString* error)
{
    QSaveFile out(file);
    if (!out.open(QIODevice::WriteOnly)) {
        if (error)
            *error = QStringLiteral("cannot write %1").arg(QDir::toNativeSeparators(file));
        return false;
    }
    if (!httpGet(url, userAgent, &out, nullptr, progress, 0, error))
        return false;
    if (!out.commit()) {
        if (error)
            *error = QStringLiteral("cannot write %1").arg(QDir::toNativeSeparators(file));
        return false;
    }
    return true;
}

QString join(const QString& a, const QString& b) { return QDir::cleanPath(a + QLatin1Char('/') + b); }

bool samePath(const QString& a, const QString& b)
{
#ifdef Q_OS_WIN
    return QDir::cleanPath(a).compare(QDir::cleanPath(b), Qt::CaseInsensitive) == 0;
#else
    return QDir::cleanPath(a) == QDir::cleanPath(b);
#endif
}

bool removeTree(const QString& dir)
{
    QDir d(dir);
    return !d.exists() || d.removeRecursively();
}

// Unpacks a zip into dir. Windows 10 and later ship bsdtar as tar.exe, which reads zip; unzip and
// PowerShell's Expand-Archive are the fallbacks (a plain GNU tar cannot read zip, so on Linux this
// uses unzip).
bool extract(const QString& zip, const QString& dir, QString* error)
{
    if (!QDir().mkpath(dir)) {
        if (error)
            *error = QStringLiteral("cannot make %1").arg(QDir::toNativeSeparators(dir));
        return false;
    }
    const QList<QStringList> attempts = {
        {QStringLiteral("tar"), QStringLiteral("-xf"), zip, QStringLiteral("-C"), dir},
        {QStringLiteral("unzip"), QStringLiteral("-o"), QStringLiteral("-q"), zip, QStringLiteral("-d"), dir},
        {QStringLiteral("powershell"), QStringLiteral("-NoProfile"), QStringLiteral("-NonInteractive"), QStringLiteral("-Command"),
         QStringLiteral("Expand-Archive -LiteralPath '%1' -DestinationPath '%2' -Force").arg(zip, dir)},
    };
    QStringList reasons;
    for (const QStringList& args : attempts) {
        QProcess p;
        p.setProcessChannelMode(QProcess::MergedChannels);
        p.start(args.value(0), args.mid(1));
        if (!p.waitForStarted(5000)) {
            reasons << QStringLiteral("%1 did not start").arg(args.value(0));
            continue;
        }
        if (!p.waitForFinished(300000) || p.exitStatus() != QProcess::NormalExit || p.exitCode() != 0) {
            reasons << QStringLiteral("%1: %2").arg(args.value(0), QString::fromLocal8Bit(p.readAll()).trimmed().section(QLatin1Char('\n'), 0, 0));
            continue;
        }
        return true;
    }
    if (error)
        *error = QStringLiteral("cannot unpack %1 (%2)").arg(QFileInfo(zip).fileName(), reasons.join(QStringLiteral("; ")));
    return false;
}

// The directory that holds the launcher program inside an unpacked launcher zip.
QString findLauncherDir(const QString& root, const QString& program)
{
    if (QFileInfo::exists(join(root, program)))
        return root;
    QDirIterator it(root, {program}, QDir::Files, QDirIterator::Subdirectories);
    if (it.hasNext())
        return QFileInfo(it.next()).absolutePath();
    return {};
}

QByteArray userAgentFor(const QString& launcherVersion)
{
    return QStringLiteral("XSEmuLauncher/%1").arg(launcherVersion).toUtf8();
}

}

UpdateService::UpdateService(AppSettings& settings, ProcessGate& gate, QObject* parent)
    : QObject(parent)
    , m_settings(settings)
    , m_gate(gate)
{
    setDirectories(Paths::appDir(), Paths::coreDir());
    m_fetch = [version = launcherVersion()](const QUrl& url, QString* error, int timeoutMs) {
        return fetchDefault(url, userAgentFor(version), timeoutMs, error);
    };
    m_download = [version = launcherVersion()](const QUrl& url, const QString& file,
                                               const std::function<bool(qint64, qint64)>& progress, QString* error) {
        return downloadDefault(url, userAgentFor(version), file, progress, error);
    };
    refresh();
}

UpdateService::~UpdateService()
{
    m_cancel.store(true);
    if (m_thread) {
        m_thread->wait(10000);
        m_thread = nullptr;
    }
}

void UpdateService::setDirectories(const QString& appDir, const QString& coreDir)
{
    m_appDir = QDir::cleanPath(appDir);
    m_coreDir = QDir::cleanPath(coreDir);
}

void UpdateService::setTransport(Fetch fetch, Download download)
{
    if (fetch)
        m_fetch = std::move(fetch);
    if (download)
        m_download = std::move(download);
}

QString UpdateService::launcherVersion() const { return QStringLiteral(XSEMU_VERSION); }

QString UpdateService::installedCoreVersion() const
{
    // The release packaging writes core\version.json and every Core update rewrites it, so it names
    // the bundled Core exactly (the emulator's --version reports its Cargo version, without the
    // release's pre-release suffix).
    QFile f(join(m_coreDir, kCoreVersionFile));
    if (f.open(QIODevice::ReadOnly)) {
        QString parseError;
        const QJsonObject o = parseJsonObject(f.readAll(), &parseError);
        const QString v = o.value(QLatin1String("version")).toString().trimmed();
        if (!v.isEmpty())
            return v;
    }
    // No version file (a Core placed by hand, or the 0.1.0-pre.1 layout): the emulator's own answer,
    // when it has been asked already.
    QString v = EmulatorBuild::cached(m_settings.emulatorPath()).version;
    if (v.startsWith(QLatin1String("XSEmu ")))
        v = v.mid(6).trimmed();
    return v;
}

QString UpdateService::previousCoreDir() const { return m_coreDir + QLatin1String(".old"); }

void UpdateService::simulateForScreenshot(const Status& status)
{
    m_simulated = true;
    m_status = status;
    emit changed();
}

void UpdateService::refresh()
{
    if (m_status.busy() || m_simulated)
        return;
    m_status.launcherVersion = launcherVersion();
    m_status.corePath = m_settings.emulatorPath();
    m_status.coreBundled = !m_status.corePath.isEmpty() && samePath(QFileInfo(m_status.corePath).absolutePath(), m_coreDir);
    m_status.installedCore = installedCoreVersion();
    m_status.corePreviousKept = QFileInfo::exists(previousCoreDir());
}

bool UpdateService::canInstall(Part part, QString* why) const
{
    const auto no = [why](const QString& reason) {
        if (why)
            *why = reason;
        return false;
    };
    if (why)
        why->clear();
    if (m_status.busy())
        return no(QStringLiteral("An update is already running."));
    if (!m_gate.isFree())
        return no(QStringLiteral("XSEmu is busy with %1. Stop it before updating.").arg(m_gate.ownerTitle().isEmpty() ? QStringLiteral("a game") : m_gate.ownerTitle()));
    if (const ProcessGate::External other = m_gate.external(); other.busy())
        return no(other.reason());
    if (part == Part::Launcher) {
        if (!m_status.launcherUpdate)
            return no(QStringLiteral("XSEmu Launcher is up to date."));
#ifndef Q_OS_WIN
        return no(QStringLiteral("The launcher replaces itself on Windows only. On this system, build it from source."));
#else
        if (!QFileInfo(m_appDir).isWritable())
            return no(QStringLiteral("The launcher's folder is read-only, so it cannot replace itself."));
#endif
    } else {
        if (!m_status.coreBundled)
            return no(QStringLiteral("XSEmu Core is set to %1, outside the launcher's own folder. Only the bundled Core is updated.")
                          .arg(QDir::toNativeSeparators(m_status.corePath)));
        if (!m_status.coreUpdate)
            return no(QStringLiteral("XSEmu Core is up to date."));
        if (!QFileInfo(m_coreDir).isWritable())
            return no(QStringLiteral("The launcher's core folder is read-only, so it cannot be updated."));
    }
    return true;
}

void UpdateService::setState(State state, const QString& message)
{
    m_status.state = state;
    m_status.message = message;
    if (state != State::Idle && state != State::Failed)
        m_status.error.clear();
    emit changed();
}

void UpdateService::finishFailed(const QString& error)
{
    m_status.state = State::Failed;
    m_status.error = error;
    m_status.message = error;
    m_status.done = -1;
    m_status.total = -1;
    emit changed();
}

void UpdateService::post(std::function<void()> f)
{
    QMetaObject::invokeMethod(this, std::move(f), Qt::QueuedConnection);
}

void UpdateService::startWorker(std::function<void()> job)
{
    if (m_thread) {
        m_thread->wait();
        m_thread = nullptr;
    }
    m_cancel.store(false);
    QThread* t = QThread::create(std::move(job));
    m_thread = t;
    connect(t, &QThread::finished, t, &QObject::deleteLater);
    connect(t, &QThread::finished, this, [this, t] {
        if (m_thread == t)
            m_thread = nullptr;
    });
    t->start();
}

void UpdateService::cancel()
{
    if (!m_status.busy())
        return;
    m_cancel.store(true);
    m_status.message = QStringLiteral("Stopping...");
    emit changed();
}

void UpdateService::check(bool manual)
{
    if (m_status.busy())
        return;
    refresh();
    m_manual = manual;
    setState(State::Checking, QStringLiteral("Checking for updates..."));

    // Everything the worker needs, read here: QSettings is not shared with another thread.
    auto inputs = std::make_shared<CheckInputs>(CheckInputs{launcherVersion(), m_status.installedCore, m_settings.prereleaseUpdates()});
    startWorker([this, inputs] { runCheck(*inputs); });
}

void UpdateService::runCheck(const CheckInputs& in)
{
    const auto fail = [this](const QString& why) {
        post([this, why] { finishFailed(why); });
    };
    QString error;
    const QByteArray list = m_fetch(QUrl(QString::fromLatin1(kApiReleases)), &error, 30000);
    if (m_cancel.load()) {
        post([this] { setState(State::Idle, QStringLiteral("Stopped checking.")); });
        return;
    }
    if (!error.isEmpty()) {
        fail(QStringLiteral("Could not read the release list: %1").arg(error));
        return;
    }
    QJsonParseError parseError;
    const QJsonDocument doc = QJsonDocument::fromJson(list, &parseError);
    if (parseError.error != QJsonParseError::NoError || !doc.isArray()) {
        fail(QStringLiteral("The release list is not what GitHub sends (%1).").arg(parseError.errorString()));
        return;
    }
    QList<QJsonObject> candidates;
    for (const QJsonValue& v : doc.array()) {
        const QJsonObject release = v.toObject();
        if (release.value(QLatin1String("draft")).toBool())
            continue;
        if (release.value(QLatin1String("prerelease")).toBool() && !in.prereleases)
            continue;
        if (releaseAssetUrl(release, QString::fromLatin1(kManifestAsset)).isEmpty())
            continue;
        QString tag = release.value(QLatin1String("tag_name")).toString().trimmed();
        if (tag.startsWith(QLatin1Char('v')))
            tag = tag.mid(1);
        const bool newer = newerThan(tag, in.launcher) || (!in.core.isEmpty() && newerThan(tag, in.core));
        if (!newer)
            continue;
        candidates.append(release);
    }
    std::sort(candidates.begin(), candidates.end(), [](const QJsonObject& a, const QJsonObject& b) {
        return compare(ReleaseVersion::parse(a.value(QLatin1String("tag_name")).toString()),
                       ReleaseVersion::parse(b.value(QLatin1String("tag_name")).toString())) > 0;
    });

    Status found;
    QString why;
    int asked = 0;
    for (const QJsonObject& release : candidates) {
        if (asked++ >= 5 || m_cancel.load())
            break;
        const QString url = releaseAssetUrl(release, QString::fromLatin1(kManifestAsset));
        QString fetchError;
        const QByteArray bytes = m_fetch(QUrl(url), &fetchError, 30000);
        if (!fetchError.isEmpty())
            continue;
        QString parseWhy;
        const QJsonObject manifest = parseJsonObject(bytes, &parseWhy);
        if (!parseWhy.isEmpty())
            continue;
        QString releaseWhy;
        const ReleaseInfo info = parseRelease(release, manifest, &releaseWhy);
        if (!info.hasParts()) {
            why = releaseWhy;
            continue;
        }
        Status s;
        s.release = info;
        s.launcherVersion = in.launcher;
        s.installedCore = in.core;
        s.launcherUpdate = info.launcher.valid() && newerThan(info.launcher.version, in.launcher);
        // A Core whose version nobody knows (a hand-placed program) is not replaced blindly, and a
        // Core outside the launcher's folder is never replaced at all.
        s.coreUpdate = info.core.valid() && !in.core.isEmpty() && newerThan(info.core.version, in.core);
        s.checkedAt = QDateTime::currentDateTimeUtc();
        if (s.updateAvailable()) {
            found = s;
            break;
        }
    }

    post([this, found, asked] {
        const bool had = found.updateAvailable();
        m_status.release = found.release;
        m_status.launcherUpdate = found.launcherUpdate;
        m_status.coreUpdate = found.coreUpdate && m_status.coreBundled;
        m_status.checkedAt = QDateTime::currentDateTimeUtc();
        m_settings.setLastUpdateCheck(m_status.checkedAt);
        m_status.done = -1;
        m_status.total = -1;
        m_status.state = had ? State::Available : State::UpToDate;
        if (had) {
            QStringList parts;
            if (m_status.launcherUpdate)
                parts << QStringLiteral("XSEmu Launcher %1").arg(found.release.launcher.version);
            if (m_status.coreUpdate)
                parts << QStringLiteral("XSEmu Core %1").arg(found.release.core.version);
            m_status.message = QStringLiteral("%1 is available.").arg(parts.join(QStringLiteral(" and ")));
            if (found.coreUpdate && !m_status.coreBundled)
                m_status.message += QStringLiteral(" The Core in use is outside the launcher's folder, so only the launcher can be updated.");
        } else {
            m_status.message = asked == 0
                ? QStringLiteral("No newer release than %1 was found.").arg(launcherVersion())
                : QStringLiteral("XSEmu Launcher %1 and XSEmu Core %2 are up to date.")
                      .arg(launcherVersion(), m_status.installedCore.isEmpty() ? QStringLiteral("(version unknown)") : m_status.installedCore);
        }
        emit changed();
    });
}

void UpdateService::install(Part part)
{
    QString why;
    if (!canInstall(part, &why)) {
        finishFailed(why);
        return;
    }
    const ReleasePart releasePart = part == Part::Launcher ? m_status.release.launcher : m_status.release.core;
    m_status.state = State::Downloading;
    m_status.message = QStringLiteral("Downloading %1...").arg(releasePart.asset);
    m_status.error.clear();
    m_status.done = 0;
    m_status.total = releasePart.size;
    emit changed();
    startWorker([this, part, releasePart] { runInstall(part, releasePart); });
}

void UpdateService::runInstall(Part part, const ReleasePart& releasePart)
{
    const auto fail = [this](const QString& why) {
        post([this, why] { finishFailed(why); });
    };
    const QString updateDir = join(m_appDir, QString::fromLatin1(kUpdateDir));
    removeTree(updateDir);
    if (!QDir().mkpath(updateDir)) {
        fail(QStringLiteral("Could not write to %1.").arg(QDir::toNativeSeparators(m_appDir)));
        return;
    }
    const QString asset = join(updateDir, releasePart.asset);
    QString error;
    qint64 lastShown = -1;
    const bool downloaded = m_download(
        QUrl(releasePart.url), asset,
        [this, &lastShown](qint64 done, qint64 total) {
            const qint64 step = lastShown < 0 ? 0 : (total > 0 ? qMax<qint64>(1, total / 100) : 1 << 20);
            if (lastShown < 0 || done - lastShown >= step) {
                lastShown = done;
                post([this, done, total] {
                    m_status.done = done;
                    m_status.total = total;
                    emit changed();
                });
            }
            return !m_cancel.load();
        },
        &error);
    if (!downloaded) {
        fail(error == QLatin1String("cancelled") ? QStringLiteral("Stopped.") : QStringLiteral("Could not download %1: %2").arg(releasePart.asset, error));
        return;
    }
    post([this] { setState(State::Applying, QStringLiteral("Checking the download...")); });
    if (fileSha256(asset).compare(releasePart.sha256, Qt::CaseInsensitive) != 0) {
        fail(QStringLiteral("%1 does not have the SHA-256 its release says.").arg(releasePart.asset));
        return;
    }
    const QString unpacked = join(updateDir, part == Part::Launcher ? QStringLiteral("launcher") : QStringLiteral("core"));
    if (!extract(asset, unpacked, &error)) {
        fail(error);
        return;
    }

    if (part == Part::Core) {
        QString applyError;
        const QString version = releasePart.version;
        if (!applyCore(unpacked, releasePart, &applyError)) {
            fail(applyError);
            return;
        }
        post([this, version] {
            removeTree(join(m_appDir, QString::fromLatin1(kUpdateDir)));
            m_status.state = State::UpToDate;
            refresh();
            m_status.coreUpdate = false;
            m_status.launcherUpdate = m_status.release.launcher.valid() && newerThan(m_status.release.launcher.version, launcherVersion());
            m_status.state = m_status.updateAvailable() ? State::Available : State::UpToDate;
            m_status.message = QStringLiteral("XSEmu Core %1 is installed. The previous copy stays in core.old until it starts, and can be restored until then.")
                                   .arg(version);
            m_status.done = -1;
            m_status.total = -1;
            emit changed();
        });
        return;
    }

    const QString program = QStringLiteral("XSEmuLauncher.exe");
    const QString stage = findLauncherDir(unpacked, program);
    if (stage.isEmpty()) {
        fail(QStringLiteral("The launcher's zip has no %1.").arg(program));
        return;
    }
    const QString backup = join(updateDir, QStringLiteral("backup"));
    const QString log = join(QDir::tempPath(), QStringLiteral("xsemu-launcher-update.log"));
    const QString script = writeUpdaterScript(&error);
    if (script.isEmpty()) {
        fail(error);
        return;
    }
    const QStringList args = {QStringLiteral("-NoProfile"), QStringLiteral("-ExecutionPolicy"), QStringLiteral("Bypass"),
                              QStringLiteral("-WindowStyle"), QStringLiteral("Hidden"), QStringLiteral("-File"), script,
                              QStringLiteral("-WaitPid"), QString::number(QCoreApplication::applicationPid()),
                              QStringLiteral("-AppDir"), m_appDir, QStringLiteral("-Stage"), stage,
                              QStringLiteral("-Backup"), backup, QStringLiteral("-Log"), log,
                              QStringLiteral("-Program"), program};
    if (!QProcess::startDetached(QStringLiteral("powershell.exe"), args)) {
        fail(QStringLiteral("Could not start the updater."));
        return;
    }
    post([this] {
        m_status.state = State::Applying;
        m_status.message = QStringLiteral("Restarting the launcher to finish the update...");
        m_status.done = -1;
        m_status.total = -1;
        emit changed();
        emit launcherUpdateReady();
    });
}

bool UpdateService::applyCore(const QString& staged, const ReleasePart& part, QString* error)
{
    const auto fail = [error](const QString& why) {
        if (error)
            *error = why;
        return false;
    };
    // The manifest names the Core's files; the program must be one of them, and it must answer
    // --version as XSEmu before anything in the launcher's folder is touched.
    QString program;
    for (const ReleaseFile& f : part.files) {
        if (QFileInfo(f.name).fileName() == Paths::emulatorExecutableName())
            program = f.name;
    }
    if (program.isEmpty())
        return fail(QStringLiteral("The Core's manifest does not name %1.").arg(Paths::emulatorExecutableName()));
    for (const ReleaseFile& f : part.files) {
        const QString file = join(staged, f.name);
        if (!QFileInfo::exists(file))
            return fail(QStringLiteral("The Core's archive has no %1.").arg(f.name));
        if (fileSha256(file).compare(f.sha256, Qt::CaseInsensitive) != 0)
            return fail(QStringLiteral("%1 in the Core's archive does not have the SHA-256 its release says.").arg(f.name));
    }
    const EmulatorBuild probe = EmulatorBuild::probe(join(staged, program));
    if (!probe.found || !probe.version.startsWith(QLatin1String("XSEmu ")))
        return fail(QStringLiteral("The new XSEmu Core does not answer --version, so it was not installed."));

    // Keep the previous copy of every file the update replaces, instead of the whole folder: the DX
    // compiler beside the Core belongs to the launcher and is not part of a Core update.
    const QString previous = previousCoreDir();
    removeTree(previous);
    if (!QDir().mkpath(previous))
        return fail(QStringLiteral("Could not write to %1.").arg(QDir::toNativeSeparators(m_appDir)));
    const auto keepPrevious = [&]() -> bool {
        for (const ReleaseFile& f : part.files) {
            const QString from = join(m_coreDir, f.name);
            if (!QFileInfo::exists(from))
                continue;
            const QString to = join(previous, f.name);
            QDir().mkpath(QFileInfo(to).absolutePath());
            QFile::remove(to);
            if (!QFile::copy(from, to))
                return false;
        }
        return true;
    };
    const auto restore = [&] {
        for (const ReleaseFile& f : part.files) {
            const QString from = join(previous, f.name);
            if (!QFileInfo::exists(from))
                continue;
            const QString to = join(m_coreDir, f.name);
            QFile::remove(to);
            QFile::copy(from, to);
        }
    };
    if (!keepPrevious()) {
        removeTree(previous);
        return fail(QStringLiteral("Could not copy the current XSEmu Core aside, so nothing was replaced."));
    }
    if (!QDir().mkpath(m_coreDir)) {
        removeTree(previous);
        return fail(QStringLiteral("Could not write to %1.").arg(QDir::toNativeSeparators(m_coreDir)));
    }
    for (const ReleaseFile& f : part.files) {
        const QString to = join(m_coreDir, f.name);
        QDir().mkpath(QFileInfo(to).absolutePath());
        QFile::remove(to);
        if (!QFile::copy(join(staged, f.name), to) || fileSha256(to).compare(f.sha256, Qt::CaseInsensitive) != 0) {
            restore();
            return fail(QStringLiteral("Could not put %1 in place; the previous Core was kept.").arg(f.name));
        }
    }
    return true;
}

QString UpdateService::writeUpdaterScript(QString* error) const
{
    // Runs after the launcher has exited (it cannot replace the program or the Qt libraries that are
    // loaded), keeps a copy of every file it replaces, starts the new launcher, and only then removes
    // the backup. A failure restores the backup and starts the old launcher again.
    const QString script = join(QDir::tempPath(), QStringLiteral("xsemu-launcher-update.ps1"));
    QSaveFile out(script);
    if (!out.open(QIODevice::WriteOnly | QIODevice::Truncate)) {
        if (error)
            *error = QStringLiteral("Could not write the updater script to %1.").arg(QDir::toNativeSeparators(script));
        return {};
    }
    const QString text = QStringLiteral(R"([CmdletBinding()]
param([int]$WaitPid, [string]$AppDir, [string]$Stage, [string]$Backup, [string]$Log, [string]$Program)
$ErrorActionPreference = 'Stop'
function Note([string]$What) { "$(Get-Date -Format o) $What" | Add-Content -LiteralPath $Log }
function Restore-Backup {
  if (-not (Test-Path -LiteralPath $Backup)) { return }
  $root = (Resolve-Path -LiteralPath $Backup).Path
  foreach ($f in Get-ChildItem -LiteralPath $root -Recurse -File) {
    $dest = Join-Path $AppDir $f.FullName.Substring($root.Length).TrimStart('\')
    Copy-Item -LiteralPath $f.FullName -Destination $dest -Force
  }
  Note 'the previous launcher files are back'
}
try {
  Note "waiting for the launcher ($WaitPid) to exit"
  $deadline = (Get-Date).AddMinutes(3)
  while ((Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) -and ((Get-Date) -lt $deadline)) { Start-Sleep -Milliseconds 250 }
  if (Get-Process -Id $WaitPid -ErrorAction SilentlyContinue) { throw 'the launcher did not exit' }
  New-Item -ItemType Directory -Force -Path $Backup | Out-Null
  $root = (Resolve-Path -LiteralPath $Stage).Path
  $files = @(Get-ChildItem -LiteralPath $root -Recurse -File)
  Note "replacing $($files.Count) files from $root"
  foreach ($f in $files) {
    $rel = $f.FullName.Substring($root.Length).TrimStart('\')
    $dest = Join-Path $AppDir $rel
    if (Test-Path -LiteralPath $dest) {
      $bak = Join-Path $Backup $rel
      New-Item -ItemType Directory -Force -Path (Split-Path -LiteralPath $bak) | Out-Null
      Copy-Item -LiteralPath $dest -Destination $bak -Force
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -LiteralPath $dest) | Out-Null
    Copy-Item -LiteralPath $f.FullName -Destination $dest -Force
  }
  Note 'files replaced; starting the new launcher'
  $new = Start-Process -FilePath (Join-Path $AppDir $Program) -PassThru
  Start-Sleep -Seconds 8
  if ($null -eq $new -or $new.HasExited) { throw 'the new launcher exited at once' }
  Note "the new launcher ($($new.Id)) is up; removing the backup"
  Remove-Item -LiteralPath $Backup -Recurse -Force -ErrorAction SilentlyContinue
  Remove-Item -LiteralPath $Stage -Recurse -Force -ErrorAction SilentlyContinue
  Note 'done'
} catch {
  Note "failed: $($_.Exception.Message); restoring"
  Restore-Backup
  Start-Process -FilePath (Join-Path $AppDir $Program)
}
)");
    const bool wrote = out.write(text.toUtf8()) > 0 && out.commit();
    if (!wrote) {
        if (error)
            *error = QStringLiteral("Could not write the updater script to %1.").arg(QDir::toNativeSeparators(script));
        return {};
    }
    return script;
}

void UpdateService::noteEmulatorStarted()
{
    const QString previous = previousCoreDir();
    if (!QFileInfo::exists(previous))
        return;
    removeTree(previous);
    m_status.corePreviousKept = false;
    m_status.message = QStringLiteral("XSEmu Core %1 started, so the previous copy is gone.")
                           .arg(m_status.installedCore.isEmpty() ? QStringLiteral("(version unknown)") : m_status.installedCore);
    emit changed();
}

bool UpdateService::restorePreviousCore(QString* error)
{
    const QString previous = previousCoreDir();
    if (!QFileInfo::exists(previous)) {
        if (error)
            *error = QStringLiteral("There is no previous XSEmu Core to go back to.");
        return false;
    }
    if (!m_gate.isFree()) {
        if (error)
            *error = QStringLiteral("XSEmu is running. Stop it before going back to the previous Core.");
        return false;
    }
    QDirIterator it(previous, QDir::Files, QDirIterator::Subdirectories);
    while (it.hasNext()) {
        const QString from = it.next();
        const QString to = join(m_coreDir, QDir(previous).relativeFilePath(from));
        QDir().mkpath(QFileInfo(to).absolutePath());
        QFile::remove(to);
        if (!QFile::copy(from, to)) {
            if (error)
                *error = QStringLiteral("Could not put %1 back.").arg(QFileInfo(to).fileName());
            return false;
        }
    }
    removeTree(previous);
    refresh();
    m_status.message = QStringLiteral("The previous XSEmu Core (%1) is back.")
                           .arg(m_status.installedCore.isEmpty() ? QStringLiteral("version unknown") : m_status.installedCore);
    emit changed();
    return true;
}
