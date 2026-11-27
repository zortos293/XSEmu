#include "RunnerManager.h"

#include "AppSettings.h"
#include "Paths.h"
#include "RunnerInstall.h"

#include <QCoreApplication>
#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QJsonDocument>
#include <QJsonObject>
#include <QThreadPool>

namespace {

QJsonObject readJson(const QString& path)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly))
        return {};
    return QJsonDocument::fromJson(f.readAll()).object();
}

QJsonObject installRecord(const QString& id) { return readJson(QDir(Paths::runnerDir(id)).filePath(QStringLiteral("runner.json"))); }

QString clean(const QString& path) { return path.isEmpty() ? QString() : QDir::cleanPath(QDir::fromNativeSeparators(path)); }

// Paths from clean(): Windows file names (and drive letters) ignore case, so C:/Runners is c:/runners there.
#ifdef Q_OS_WIN
constexpr Qt::CaseSensitivity kPathCase = Qt::CaseInsensitive;
#else
constexpr Qt::CaseSensitivity kPathCase = Qt::CaseSensitive;
#endif
bool samePath(const QString& a, const QString& b) { return a.compare(b, kPathCase) == 0; }
bool inside(const QString& path, const QString& dir) { return samePath(path, dir) || path.startsWith(dir + QLatin1Char('/'), kPathCase); }

struct Tree {
    qint64 files = 0;
    qint64 bytes = 0;
};

Tree measure(const QString& dir)
{
    Tree t;
    QDirIterator it(dir, QDir::Files | QDir::Hidden | QDir::System | QDir::NoSymLinks, QDirIterator::Subdirectories);
    while (it.hasNext()) {
        it.next();
        ++t.files;
        t.bytes += it.fileInfo().size();
    }
    return t;
}

}

RunnerManager::RunnerManager(QObject* parent)
    : QObject(parent)
{
}

RunnerManager::~RunnerManager()
{
    if (m_task) {
        // no finished() into a half-destroyed manager (`m_settings` is a QPointer: it is null if the
        // settings were destroyed first, whichever order `Services` destroys its members in)
        m_task->disconnect(this);
        m_task->cancel();
        // a move-and-rebuild writes the new folder before it builds: stopping here must put the runner
        // back where its working copy is, or a restart looks for it in a folder that never got one
        if (m_rebuild && m_settings) {
            m_settings->disconnect(this);
            m_settings->setRunnerLocation(m_rebuild->id, m_rebuild->previous);
        }
    }
}

void RunnerManager::setSettings(AppSettings* settings)
{
    m_settings = settings;
    applyLocations();
    connect(settings, &AppSettings::changed, this, [this](const QString& key) {
        if (key == QLatin1String(AppSettings::Key::RunnersFolder) || key == QLatin1String(AppSettings::Key::RunnerLocations))
            applyLocations();
    });
}

void RunnerManager::setInUseCheck(std::function<QString(const QString& id)> check) { m_inUse = std::move(check); }

void RunnerManager::applyLocations()
{
    Paths::setRunnersFolder(m_settings->runnersFolder());
    Paths::setRunnerLocations(m_settings->runnerLocations());
    m_sizes.clear();
    for (const RunnerSpec& r : Runners::curated()) {
        if (r.isBuiltIn())
            continue;
        emit changed(r.id);
        emit stateChanged(r.id);
    }
}

void RunnerManager::setLocation(const QString& id, const QString& folder)
{
    m_settings->setRunnerLocation(id, samePath(clean(folder), clean(Paths::runnersDir())) ? QString() : clean(folder));
}

QString RunnerManager::location(const QString& id) const { return Paths::runnerLocation(id); }

QString RunnerManager::busyReason(const QString& id) const
{
    const RunnerSpec* spec = Runners::find(id);
    const QString name = spec ? spec->name : id;
    if (m_removing.contains(id))
        return QStringLiteral("%1 is being uninstalled; wait for it to finish.").arg(name);
    if (m_move) {
        const RunnerSpec* moving = Runners::find(m_move->id);
        return QStringLiteral("%1 is being moved; wait for it to finish.").arg(moving ? moving->name : m_move->id);
    }
    if (m_task && m_taskId == id) {
        const bool building = spec && spec->install == RunnerSpec::Install::BuildFromSource;
        return QStringLiteral("%1 is %2; wait for it or cancel it first.").arg(name, building ? QStringLiteral("building") : QStringLiteral("installing"));
    }
    return m_inUse ? m_inUse(id) : QString();
}

bool RunnerManager::relocate(const QString& id, const QString& folder, Relocation how, QString* error)
{
    auto refuse = [error](const QString& why) {
        if (error)
            *error = why;
        return false;
    };
    const RunnerSpec* spec = Runners::find(id);
    if (!spec || spec->isBuiltIn())
        return refuse(QStringLiteral("XSEmu Core comes with the launcher; it has no folder of its own to move."));
    if (!m_settings)
        return refuse(QStringLiteral("Runner locations need the launcher settings."));
    if (const QString busy = busyReason(id); !busy.isEmpty())
        return refuse(busy);
    const QString target = clean(folder);
    if (target.isEmpty() || !QDir::isAbsolutePath(target))
        return refuse(QStringLiteral("Pick a folder for %1.").arg(spec->name));
    const QString from = clean(Paths::runnerDir(id));
    const QString to = clean(QDir(target).filePath(id));
    if (samePath(to, from))
        return refuse(QStringLiteral("%1 is already in %2.").arg(spec->name, QDir::toNativeSeparators(target)));
    if (inside(to, from))
        return refuse(QStringLiteral("%1 can't move into its own folder.").arg(spec->name));
    if (QDir(to).exists() && !QDir(to).isEmpty(QDir::AllEntries | QDir::NoDotAndDotDot | QDir::Hidden | QDir::System))
        return refuse(QStringLiteral("%1 already has a %2 folder; pick another folder or delete that one first.").arg(QDir::toNativeSeparators(target), id));

    const QJsonObject rec = installRecord(id);
    if (how == Relocation::Rebuild && !rec.isEmpty()) {
        m_rebuild = Rebuild{id, m_settings->runnerLocations().value(id), from};
        setLocation(id, target);
        if (!install(id, error)) {
            m_settings->setRunnerLocation(id, m_rebuild->previous);
            m_rebuild.reset();
            return false;
        }
        return true;
    }
    if (!QFileInfo::exists(from)) {
        setLocation(id, target);
        emit relocated(id, true, {});
        return true;
    }

    const QString exeRel = rec.isEmpty() ? QString() : QDir(from).relativeFilePath(QDir(spec->installDir()).filePath(rec.value(QLatin1String("executable")).toString()));
    m_move = Move{id, target, to, 0, -1, QDateTime::currentDateTime()};
    emit changed(id);
    emit stateChanged(id);
    const QPointer<RunnerManager> self = this;
    QThreadPool::globalInstance()->start([self, id, from, to, exeRel] {
        const auto progress = [self, id](qint64 done, qint64 total) {
            QMetaObject::invokeMethod(qApp, [self, id, done, total] {
                if (!self || !self->m_move || self->m_move->id != id)
                    return;
                self->m_move->done = done;
                self->m_move->total = total;
                emit self->changed(id);
            });
        };
        QString why;
        QString note;
        const bool ok = moveFolder(from, to, exeRel, &why, &note, progress);
        QMetaObject::invokeMethod(qApp, [self, ok, why, note] {
            if (self)
                self->finishMove(ok, ok ? note : why);
        });
    });
    return true;
}

void RunnerManager::finishMove(bool ok, const QString& message)
{
    const Move move = *m_move;
    m_move.reset();
    if (ok)
        setLocation(move.id, move.folder);
    m_sizes.remove(move.id);
    emit changed(move.id);
    emit stateChanged(move.id);
    emit relocated(move.id, ok, message);
}

bool RunnerManager::moveFolder(const QString& from, const QString& to, const QString& exeRel, QString* error, QString* note,
                               const std::function<void(qint64, qint64)>& progress, bool allowRename)
{
    auto failed = [error, to](const QString& why) {
        QDir(to).removeRecursively();
        *error = why;
        return false;
    };
    QDir().mkpath(QFileInfo(to).absolutePath());
    if (allowRename && QDir(to).exists() && QDir(to).isEmpty(QDir::AllEntries | QDir::NoDotAndDotDot | QDir::Hidden | QDir::System))
        QDir().rmdir(to);
    if (allowRename && QDir().rename(from, to)) {
        if (!exeRel.isEmpty() && !QFileInfo::exists(QDir(to).filePath(exeRel))) {
            QDir().rename(to, from);
            *error = QStringLiteral("After the move %1 was missing, so the runner stayed where it was.").arg(QDir::toNativeSeparators(exeRel));
            return false;
        }
        return true;
    }

    if (!QFileInfo(from).isDir()) {
        *error = QStringLiteral("%1 is not there any more.").arg(QDir::toNativeSeparators(from));
        return false;
    }
    // another drive: copy every file, check the copy, then delete the original
    const Tree source = measure(from);
    qint64 done = 0;
    qint64 reported = 0;
    if (progress)
        progress(0, source.bytes);
    QDirIterator it(from, QDir::AllEntries | QDir::NoDotAndDotDot | QDir::Hidden | QDir::System | QDir::NoSymLinks, QDirIterator::Subdirectories);
    if (!QDir().mkpath(to))
        return failed(QStringLiteral("Cannot create %1.").arg(QDir::toNativeSeparators(to)));
    while (it.hasNext()) {
        it.next();
        const QFileInfo fi = it.fileInfo();
        const QString dst = QDir(to).filePath(QDir(from).relativeFilePath(fi.absoluteFilePath()));
        if (fi.isDir()) {
            if (!QDir().mkpath(dst))
                return failed(QStringLiteral("Cannot create %1.").arg(QDir::toNativeSeparators(dst)));
            continue;
        }
        QDir().mkpath(QFileInfo(dst).absolutePath());
        QFile src(fi.absoluteFilePath());
        if (!src.copy(dst))
            return failed(QStringLiteral("Cannot copy %1: %2. Nothing was moved.").arg(QDir::toNativeSeparators(fi.absoluteFilePath()), src.errorString()));
        done += fi.size();
        if (progress && (done - reported > 64LL * 1024 * 1024 || done == source.bytes)) {
            reported = done;
            progress(done, source.bytes);
        }
    }
    const Tree copy = measure(to);
    if (copy.files != source.files || copy.bytes != source.bytes)
        return failed(QStringLiteral("The copy has %1 files and %2 bytes, the original %3 and %4, so nothing was moved.")
                          .arg(copy.files).arg(copy.bytes).arg(source.files).arg(source.bytes));
    if (!exeRel.isEmpty() && (!QFileInfo::exists(QDir(to).filePath(exeRel)) || QFileInfo(QDir(to).filePath(exeRel)).size() != QFileInfo(QDir(from).filePath(exeRel)).size()))
        return failed(QStringLiteral("The copied %1 does not match the original, so nothing was moved.").arg(QDir::toNativeSeparators(exeRel)));
    if (!QDir(from).removeRecursively() && note)
        *note = QStringLiteral("Moved. Some files in %1 could not be deleted; delete that folder yourself.").arg(QDir::toNativeSeparators(from));
    return true;
}

bool RunnerManager::setRunnersFolder(const QString& folder, QString* error)
{
    auto refuse = [error](const QString& why) {
        if (error)
            *error = why;
        return false;
    };
    if (!m_settings)
        return refuse(QStringLiteral("The Runners folder needs the launcher settings."));
    if (Paths::runnersDirFromEnvironment())
        return refuse(QStringLiteral("XSEMU_RUNNERS_DIR sets the Runners folder; unset it to choose one here."));
    if (m_task) {
        const RunnerSpec* busy = Runners::find(m_taskId);
        return refuse(QStringLiteral("%1 is installing and uses the Runners folder; wait for it or cancel it first.").arg(busy ? busy->name : m_taskId));
    }
    if (m_move)
        return refuse(busyReason(m_move->id));
    const QString next = clean(folder).isEmpty() ? Paths::defaultRunnersDir() : clean(folder);
    const QString current = Paths::runnersDir();
    if (next != current) {
        const QHash<QString, QString> locations = m_settings->runnerLocations();
        for (const RunnerSpec& r : Runners::curated()) {
            if (!r.isBuiltIn() && !locations.contains(r.id) && QFileInfo::exists(Paths::runnerDir(r.id)))
                m_settings->setRunnerLocation(r.id, current);
        }
    }
    m_settings->setRunnersFolder(next == Paths::defaultRunnersDir() ? QString() : next);
    return true;
}

RunnerManager::Status RunnerManager::status(const QString& id) const
{
    if (m_simulated.contains(id))
        return m_simulated.value(id);
    Status s;
    const RunnerSpec* spec = Runners::find(id);
    if (!spec) {
        s.state = State::Unavailable;
        return s;
    }
    s.pinnedVersion = spec->pinnedVersion();
    if (spec->isBuiltIn()) {
        s.state = State::BuiltIn;
        return s;
    }
    const QJsonObject rec = installRecord(id);
    if (!rec.isEmpty()) {
        s.installedVersion = rec.value(QLatin1String("version")).toString();
        s.executable = QDir(spec->installDir()).filePath(rec.value(QLatin1String("executable")).toString());
    }
    if (m_taskId == id && m_task) {
        s.state = State::Installing;
        s.progress = m_task->progress();
        return s;
    }
    if (m_removing.contains(id)) {
        s.state = State::Uninstalling;
        s.progress.detail = QDir::toNativeSeparators(Paths::runnerDir(id));
        return s;
    }
    if (m_move && m_move->id == id) {
        s.state = State::Moving;
        s.progress.step = QStringLiteral("Moving");
        s.progress.bytes = true;
        s.progress.done = m_move->done;
        s.progress.total = m_move->total;
        s.progress.detail = QDir::toNativeSeparators(m_move->to);
        s.progress.startedAt = m_move->startedAt;
        return s;
    }
    s.error = m_errors.value(id);
    if (s.error.isEmpty())
        s.error = readJson(RunnerInstall::failurePath(id)).value(QLatin1String("error")).toString();
    if (!rec.isEmpty() && !QFileInfo::exists(s.executable)) {
        s.state = State::Failed;
        s.error = m_errors.value(id, QStringLiteral("Some files of %1 were left in %2. Install it again over them, or uninstall to delete them.")
                                         .arg(spec->name, QDir::toNativeSeparators(Paths::runnerDir(id))));
    } else if (!rec.isEmpty())
        s.state = spec->isPinned() && s.installedVersion != s.pinnedVersion ? State::UpdateAvailable : State::Installed;
    else if (!s.error.isEmpty())
        s.state = State::Failed;
    else
        s.state = spec->isPinned() ? State::NotInstalled : State::Unavailable;
    return s;
}

bool RunnerManager::isRunnable(const QString& id) const
{
    const Status s = status(id);
    return s.state == State::BuiltIn || ((s.state == State::Installed || s.state == State::UpdateAvailable) && QFileInfo::exists(s.executable));
}

bool RunnerManager::isInstallable(const QString& id) const
{
    if (isRunnable(id))
        return true;
    const RunnerSpec* spec = Runners::find(id);
    return spec && spec->isPinned();
}

QString RunnerManager::busyId() const
{
    return m_task ? m_taskId : QString();
}

bool RunnerManager::install(const QString& id, QString* error)
{
    auto refuse = [error](const QString& why) {
        if (error)
            *error = why;
        return false;
    };
    const RunnerSpec* spec = Runners::find(id);
    if (!spec || spec->isBuiltIn())
        return refuse(QStringLiteral("XSEmu Core comes with the launcher and has nothing to install."));
    if (!spec->isPinned())
        return refuse(QStringLiteral("%1 is not available yet: no release is pinned for this launcher version.").arg(spec->name));
    if (m_task) {
        const RunnerSpec* busy = Runners::find(m_taskId);
        return refuse(QStringLiteral("%1 is installing; wait for it or cancel it first.").arg(busy ? busy->name : m_taskId));
    }
    if (m_move || m_removing.contains(id))
        return refuse(busyReason(id));
    m_errors.remove(id);
    m_taskId = id;
    m_task = new RunnerInstall(*spec, this);
    connect(m_task, &RunnerInstall::progressChanged, this, [this, id] { emit changed(id); });
    connect(m_task, &RunnerInstall::finished, this, [this, id](bool ok, const QString& why) {
        // stopping it ourselves is not a failure: nothing to remember, and the status that follows is
        // the one the runner had before (NotInstalled, or the installed version with its update)
        const bool cancelled = m_task && m_task->cancelled();
        if (!ok && !cancelled)
            m_errors.insert(id, why);
        m_task->deleteLater();
        m_task.clear();
        m_taskId.clear();
        m_sizes.remove(id);
        if (m_rebuild && m_rebuild->id == id) {
            const Rebuild rebuild = *m_rebuild;
            m_rebuild.reset();
            if (ok) {
                QThreadPool::globalInstance()->start([dir = rebuild.oldDir] { QDir(dir).removeRecursively(); });
            } else {
                // the old copy still works: keep running it from where it was
                m_settings->setRunnerLocation(id, rebuild.previous);
                if (!cancelled)
                    m_errors.insert(id, QStringLiteral("%1 It still runs from %2.").arg(why, QDir::toNativeSeparators(rebuild.oldDir)));
            }
            emit relocated(id, ok, ok || cancelled ? QString() : m_errors.value(id));
        }
        emit changed(id);
        emit stateChanged(id);
        emit installFinished(id, ok, why);
    });
    m_task->start();
    emit changed(id);
    emit stateChanged(id);
    return true;
}

void RunnerManager::cancel()
{
    if (m_task)
        m_task->cancel();
}

bool RunnerManager::uninstall(const QString& id, QString* error)
{
    const RunnerSpec* spec = Runners::find(id);
    if (!spec || spec->isBuiltIn()) {
        if (error)
            *error = QStringLiteral("XSEmu Core comes with the launcher and cannot be uninstalled here.");
        return false;
    }
    if (const QString busy = busyReason(id); !busy.isEmpty()) {
        if (error)
            *error = busy;
        return false;
    }
    m_errors.remove(id);
    m_sizes.remove(id);
    m_removing.insert(id);
    emit changed(id);
    emit stateChanged(id);
    // a built runner's folder holds its whole build: delete it off the UI thread. The install record
    // goes last, so a file in use leaves it behind with the rest: status() reads a record whose program
    // is gone as Failed with the folder named, and the card offers Install over it and Uninstall again.
    const QPointer<RunnerManager> self = this;
    QThreadPool::globalInstance()->start([self, id, dir = Paths::runnerDir(id), name = spec->name] {
        bool ok = true;
        QDir d(dir);
        for (const QFileInfo& fi : d.entryInfoList(QDir::AllEntries | QDir::NoDotAndDotDot | QDir::Hidden | QDir::System)) {
            if (fi.fileName() == QLatin1String("runner.json"))
                continue;
            ok = (fi.isDir() && !fi.isSymLink() ? QDir(fi.absoluteFilePath()).removeRecursively() : QFile::remove(fi.absoluteFilePath())) && ok;
        }
        ok = ok && d.removeRecursively();
        const QString why = ok ? QString() : QStringLiteral("Some files of %1 are in use; stop it and try again.").arg(name);
        QMetaObject::invokeMethod(qApp, [self, id, ok, why] {
            if (!self)
                return;
            self->m_removing.remove(id);
            self->m_sizes.remove(id);
            emit self->changed(id);
            emit self->stateChanged(id);
            emit self->uninstalled(id, ok, why);
        });
    });
    return true;
}

void RunnerManager::simulateForScreenshot(const QString& id, const Status& status)
{
    m_simulated.insert(id, status);
    emit changed(id);
    emit stateChanged(id);
}

qint64 RunnerManager::installedBytes(const QString& id)
{
    const RunnerSpec* spec = Runners::find(id);
    if (!spec || spec->isBuiltIn())
        return 0;
    if (const auto it = m_sizes.constFind(id); it != m_sizes.constEnd())
        return *it;
    m_sizes.insert(id, -1);
    const QString dir = spec->installDir();
    const QPointer<RunnerManager> self = this;
    QThreadPool::globalInstance()->start([self, id, dir] {
        qint64 total = 0;
        QDirIterator it(dir, QDir::Files | QDir::Hidden | QDir::NoSymLinks, QDirIterator::Subdirectories);
        while (it.hasNext()) {
            it.next();
            total += it.fileInfo().size();
        }
        QMetaObject::invokeMethod(qApp, [self, id, total] {
            if (!self || self->m_sizes.value(id, 0) != -1)
                return;
            self->m_sizes.insert(id, total);
            emit self->changed(id);
        });
    });
    return -1;
}

QDateTime RunnerManager::installedAt(const QString& id) const
{
    return QDateTime::fromString(installRecord(id).value(QLatin1String("installedAt")).toString(), Qt::ISODate).toLocalTime();
}

QString RunnerManager::installLogPath(const QString& id) const { return RunnerInstall::logPath(id); }

QString RunnerManager::cachePath(const QString& id) const
{
    const RunnerSpec* spec = Runners::find(id);
    if (!spec)
        return {};
    if (spec->isBuiltIn())
        return Paths::dxilCacheDir();
    return spec->resolvedCacheDir();
}
