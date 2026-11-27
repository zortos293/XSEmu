#include "ShaderCacheInfo.h"

#include "LaunchConfig.h"

#include <QDir>
#include <QDirIterator>
#include <QFile>
#include <QFileInfo>
#include <QRegularExpression>
#include <QSet>

quint64 ShaderCacheInfo::fnv64(const char* data, qsizetype len)
{
    quint64 h = 0xcbf29ce484222325ULL;
    for (qsizetype i = 0; i < len; ++i) {
        h ^= static_cast<unsigned char>(data[i]);
        h *= 0x100000001b3ULL;
    }
    return h;
}

bool ShaderCacheInfo::readManifest(const QString& path, Manifest& m, const std::atomic<bool>* cancel)
{
    QFile file(path);
    if (!file.open(QIODevice::ReadOnly)) {
        m = Manifest{};
        return false;
    }
    const auto lengthAt = [](const char* data) {
        const auto* p = reinterpret_cast<const unsigned char*>(data);
        return quint32(p[0]) | (quint32(p[1]) << 8) | (quint32(p[2]) << 16) | (quint32(p[3]) << 24);
    };
    const qint64 size = file.size();
    const QDateTime born = QFileInfo(file).birthTime();
    const QByteArray head = file.read(64);
    bool resume = m.path == path && m.born == born && size >= m.offset && head.startsWith(m.head);
    if (resume && m.lastStart >= 0) {
        const QByteArray last = file.seek(m.lastStart) ? file.read(4 + qint64(m.lastLen)) : QByteArray();
        resume = last.size() == 4 + qint64(m.lastLen) && lengthAt(last.constData()) == m.lastLen && fnv64(last.constData() + 4, m.lastLen) == m.lastHash;
    }
    const Manifest fresh{path, born, head, 0, 0, {}, -1, 0, 0, -1};
    if (!resume)
        m = fresh;
    m.head = head;

    // false when a length prefix is longer than any entry: the parse is off the entry boundaries
    const auto parse = [&] {
        if (!file.seek(m.offset))
            return true;
        constexpr qint64 chunk = 4 << 20;
        QByteArray buf; // a partial entry and at most one chunk: entries are capped
        qint64 at = m.offset;
        while (!file.atEnd()) {
            if (cancel && cancel->load())
                return true;
            buf += file.read(chunk);
            qsizetype pos = 0;
            while (pos + 4 <= buf.size()) {
                const quint32 len = lengthAt(buf.constData() + pos);
                if (len > kMaxManifestEntry) {
                    m.badAt = at + pos;
                    m.offset = at + pos;
                    return false;
                }
                if (pos + 4 + qsizetype(len) > buf.size())
                    break; // the entry continues in the next chunk, or is a torn last entry
                const quint64 hash = fnv64(buf.constData() + pos + 4, len);
                m.keys.insert(hash | 1ULL);
                ++m.entries;
                m.lastStart = at + pos;
                m.lastLen = len;
                m.lastHash = hash;
                pos += 4 + qsizetype(len);
            }
            buf.remove(0, pos);
            at += pos;
            m.offset = at;
        }
        return true;
    };
    // a bad length where the last read already stopped is the file's own damage; anywhere else after a
    // resume it means the resume point was stale, so parse from the start, where the emulator's own
    // rule applies: the entries end at the first bad length
    const qint64 knownBad = m.badAt;
    if (!parse() && resume && m.badAt != knownBad) {
        m = fresh;
        parse();
    }
    return true;
}

ShaderCacheInfo ShaderCacheInfo::read(const QString& dxilDir)
{
    Manifest manifest;
    return read(dxilDir, manifest);
}

ShaderCacheInfo ShaderCacheInfo::read(const QString& dxilDir, Manifest& manifest, const std::atomic<bool>* cancel)
{
    ShaderCacheInfo info;
    info.dxilDir = dxilDir;
    info.scanned = true;
    const QDir root(dxilDir);
    info.exists = root.exists();
    if (!info.exists) {
        manifest = Manifest{};
        return info;
    }

    const QString manifestPath = root.filePath(QStringLiteral("pipelines/pipelines.bin"));
    if (readManifest(manifestPath, manifest, cancel)) {
        info.manifestBytes = QFileInfo(manifestPath).size();
        info.manifestEntries = manifest.entries;
        info.pipelines = manifest.keys.size();
    }

    QFile runs(root.filePath(QStringLiteral("pipelines/runs")));
    if (runs.open(QIODevice::ReadOnly)) {
        bool ok = false;
        const int n = QString::fromUtf8(runs.readAll()).trimmed().toInt(&ok);
        info.runs = ok ? n : 0;
    }

    // one walk for the folder size and the driver blobs (driver/<adapter>/**/*.pso)
    const QDir driver(root.filePath(QStringLiteral("driver")));
    const bool hasDriver = driver.exists();
    if (hasDriver) {
        info.blobs = 0;
        info.blobBytes = 0;
        info.adapters = int(driver.entryList(QDir::Dirs | QDir::NoDotAndDotDot).size());
    }
    const QString driverPrefix = driver.absolutePath() + QLatin1Char('/');
    info.totalBytes = 0;
    QDirIterator it(root.absolutePath(), QDir::Files | QDir::Hidden | QDir::NoSymLinks, QDirIterator::Subdirectories);
    for (int n = 0; it.hasNext(); ++n) {
        if ((n & 1023) == 0 && cancel && cancel->load())
            break;
        const QString path = it.next();
        const QFileInfo fi = it.fileInfo();
        info.totalBytes += fi.size();
        if (!hasDriver || !path.endsWith(QLatin1String(".pso")) || !path.startsWith(driverPrefix) || path.indexOf(QLatin1Char('/'), driverPrefix.size()) < 0)
            continue;
        ++info.blobs;
        info.blobBytes += fi.size();
        const QDateTime modified = fi.lastModified();
        if (!info.newestBlob.isValid() || modified > info.newestBlob)
            info.newestBlob = modified;
    }
    return info;
}

TitleShaderState titleShaderState(const QString& titleId, const QList<WarmupRun>& recent, const ShaderCacheInfo& cache, const QString& profile)
{
    // a finished warm-up counts only while its driver blobs are still there (Clear empties the cache)
    const bool empty = cache.scanned && (!cache.exists || cache.blobs <= 0);
    for (const WarmupRun& r : recent) {
        if (r.titleId != titleId || !r.runnerId.isEmpty()) // a runner's pre-cache fills its own cache, not XSEmu's
            continue;
        const bool otherProfile = !profile.isEmpty() && !r.profile.isEmpty() && r.profile != profile;
        switch (r.result) {
        case WarmupRun::Result::Finished:
            return empty ? TitleShaderState::NeverWarmed : otherProfile ? TitleShaderState::ProfileChanged : TitleShaderState::Ready;
        case WarmupRun::Result::Failed: return TitleShaderState::LastRunFailed;
        case WarmupRun::Result::NothingToBuild:
            return empty ? TitleShaderState::NeverWarmed : otherProfile ? TitleShaderState::ProfileChanged : TitleShaderState::Unknown;
        case WarmupRun::Result::Cancelled:
        case WarmupRun::Result::StoppedPartWay: return TitleShaderState::LastRunIncomplete;
        }
    }
    if (empty)
        return TitleShaderState::NeverWarmed;
    return TitleShaderState::Unknown;
}

TitleShaderState shownShaderState(const AppSettings& settings, const QString& titleId, const QList<WarmupRun>& recent, const ShaderCacheInfo& cache)
{
    if (!settings.xsemuWarmup())
        return TitleShaderState::Unknown;
    return titleShaderState(titleId, recent, cache, LaunchConfig::profileIdentity(settings, titleId));
}

bool ShaderCacheInfo::isCacheEntry(const QFileInfo& entry)
{
    const QString name = entry.fileName();
    if (entry.isSymLink())
        return false;
    if (entry.isDir())
        return name == QLatin1String("pipelines") || name == QLatin1String("driver") || name == QLatin1String("translated");
    static const QRegularExpression dxil(QStringLiteral("^[0-9a-fA-F]{16}[.](dxil|tmp[0-9]+)$"));
    return name == QLatin1String("used-pipelines.txt") || name == QLatin1String("skipped-pipelines.txt") || dxil.match(name).hasMatch();
}

ShaderCacheInfo::Cleared ShaderCacheInfo::clear(const QString& dxilDir)
{
    Cleared r;
    const QDir dir(dxilDir);
    if (!dir.exists())
        return r;
    if (dir.isRoot() || QDir::cleanPath(dir.absolutePath()) == QDir::cleanPath(QDir::homePath())) {
        r.ok = false;
        r.error = QStringLiteral("%1 is not a shader cache folder, so nothing was deleted.").arg(QDir::toNativeSeparators(dxilDir));
        return r;
    }
    for (const QFileInfo& fi : dir.entryInfoList(QDir::AllEntries | QDir::NoDotAndDotDot | QDir::Hidden | QDir::System, QDir::Name)) {
        if (!isCacheEntry(fi)) {
            r.kept << fi.fileName();
            continue;
        }
        r.ok = (fi.isDir() ? QDir(fi.absoluteFilePath()).removeRecursively() : QFile::remove(fi.absoluteFilePath())) && r.ok;
    }
    if (!r.ok)
        r.error = QStringLiteral("Some cache files could not be deleted. Close XSEmu Core and try again.");
    return r;
}
