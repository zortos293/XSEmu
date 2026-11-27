#include "CoverArtService.h"

#include "AppSettings.h"
#include "Paths.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QRegularExpression>
#include <QSaveFile>
#include <QTimer>
#include <QUrlQuery>

#include <algorithm>

namespace {

constexpr qint64 kMaxImageBytes = 20 * 1024 * 1024;

int httpStatus(QNetworkReply* reply)
{
    return reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
}

QString errorMessage(QNetworkReply* reply, const QByteArray& body)
{
    const QJsonArray errors = QJsonDocument::fromJson(body).object().value(QStringLiteral("errors")).toArray();
    if (!errors.isEmpty() && !errors.first().toString().isEmpty())
        return errors.first().toString();
    return reply->errorString();
}

QUrl apiPath(const QUrl& base, const QString& tail)
{
    QUrl url(base);
    QString path = url.path();
    while (path.endsWith(QLatin1Char('/')))
        path.chop(1);
    url.setPath(path + tail, QUrl::StrictMode);
    return url;
}

}

// ---- CoverCache ----

CoverCache::CoverCache(QString dir)
    : m_dir(std::move(dir))
{
    load();
}

QString CoverCache::kindName(Kind kind)
{
    switch (kind) {
    case Kind::Grid: return QStringLiteral("grid");
    case Kind::Hero: return QStringLiteral("hero");
    case Kind::Logo: return QStringLiteral("logo");
    case Kind::Icon: return QStringLiteral("icon");
    }
    return {};
}

void CoverCache::load()
{
    m_entries.clear();
    QFile f(QDir(m_dir).filePath(QStringLiteral("index.json")));
    if (!f.open(QIODevice::ReadOnly))
        return;
    const QJsonObject root = QJsonDocument::fromJson(f.readAll()).object();
    for (auto it = root.begin(); it != root.end(); ++it) {
        const QJsonObject o = it.value().toObject();
        const QJsonObject custom = o.value(QStringLiteral("custom")).toObject();
        const QJsonObject authors = o.value(QStringLiteral("authors")).toObject();
        Entry e;
        e.version = o.value(QStringLiteral("v")).toInt(1);
        e.gameId = static_cast<qint64>(o.value(QStringLiteral("gameId")).toDouble());
        e.name = o.value(QStringLiteral("name")).toString();
        e.pinned = o.value(QStringLiteral("pinned")).toBool();
        for (Kind kind : kKinds) {
            e.file(kind) = o.value(kindName(kind)).toString();
            e.customFile(kind) = custom.value(kindName(kind)).toString();
            e.authors[size_t(kind)] = authors.value(kindName(kind)).toString();
        }
        e.fetchedAt = QDateTime::fromString(o.value(QStringLiteral("fetchedAt")).toString(), Qt::ISODate);
        e.notFound = o.value(QStringLiteral("notFound")).toBool();
        m_entries.insert(it.key(), e);
    }
}

void CoverCache::save() const
{
    QDir().mkpath(m_dir);
    QJsonObject root;
    for (auto it = m_entries.cbegin(); it != m_entries.cend(); ++it) {
        const Entry& e = it.value();
        QJsonObject o{
            {QStringLiteral("v"), e.version},
            {QStringLiteral("gameId"), static_cast<double>(e.gameId)},
            {QStringLiteral("name"), e.name},
            {QStringLiteral("pinned"), e.pinned},
            {QStringLiteral("fetchedAt"), e.fetchedAt.isValid() ? e.fetchedAt.toUTC().toString(Qt::ISODate) : QString()},
            {QStringLiteral("notFound"), e.notFound},
        };
        QJsonObject custom;
        QJsonObject authors;
        for (Kind kind : kKinds) {
            o.insert(kindName(kind), e.file(kind));
            if (!e.customFile(kind).isEmpty())
                custom.insert(kindName(kind), e.customFile(kind));
            if (!e.authors[size_t(kind)].isEmpty())
                authors.insert(kindName(kind), e.authors[size_t(kind)]);
        }
        if (!custom.isEmpty())
            o.insert(QStringLiteral("custom"), custom);
        if (!authors.isEmpty())
            o.insert(QStringLiteral("authors"), authors);
        root.insert(it.key(), o);
    }
    QSaveFile f(QDir(m_dir).filePath(QStringLiteral("index.json")));
    if (f.open(QIODevice::WriteOnly)) {
        f.write(QJsonDocument(root).toJson(QJsonDocument::Indented));
        f.commit();
    }
}

void CoverCache::put(const QString& titleId, const Entry& entry)
{
    m_entries.insert(titleId, entry);
    save();
}

void CoverCache::remove(const QString& titleId)
{
    Entry e = m_entries.take(titleId);
    const QDir d(m_dir);
    for (Kind kind : kKinds) {
        if (!e.file(kind).isEmpty()) QFile::remove(d.filePath(e.file(kind)));
        if (!e.customFile(kind).isEmpty()) QFile::remove(d.filePath(e.customFile(kind)));
    }
    save();
}

void CoverCache::clear()
{
    const QDir d(m_dir);
    for (auto it = m_entries.begin(); it != m_entries.end();) {
        Entry& e = it.value();
        bool custom = false;
        for (Kind kind : kKinds) {
            if (!e.file(kind).isEmpty()) QFile::remove(d.filePath(e.file(kind)));
            custom = custom || !e.customFile(kind).isEmpty();
        }
        if (!custom) {
            it = m_entries.erase(it);
            continue;
        }
        Entry kept;
        kept.custom = e.custom;
        e = kept;
        ++it;
    }
    save();
}

QString CoverCache::existing(const QString& file) const
{
    if (file.isEmpty())
        return {};
    const QString p = QDir(m_dir).filePath(file);
    return QFileInfo(p).isFile() ? p : QString();
}

QString CoverCache::path(const QString& titleId, Kind kind) const
{
    const auto it = m_entries.constFind(titleId);
    if (it == m_entries.cend())
        return {};
    const QString custom = existing(it->customFile(kind));
    return custom.isEmpty() ? existing(it->file(kind)) : custom;
}

bool CoverCache::isCustom(const QString& titleId, Kind kind) const
{
    const auto it = m_entries.constFind(titleId);
    return it != m_entries.cend() && !existing(it->customFile(kind)).isEmpty();
}

QString CoverCache::fileName(const QString& titleId, Kind kind, const QString& extension, bool custom)
{
    static const QRegularExpression plain(QStringLiteral("^[a-z0-9]{1,5}$"));
    QString ext = extension.toLower();
    if (!plain.match(ext).hasMatch())
        ext = QStringLiteral("img");
    return QStringLiteral("%1-%2%3.%4").arg(titleId, custom ? QStringLiteral("custom-") : QString(), kindName(kind), ext);
}

// ---- CoverArtService ----

CoverArtService::CoverArtService(AppSettings& settings, QObject* parent)
    : CoverArtService(settings, Paths::coverCacheDir(), parent)
{
}

CoverArtService::CoverArtService(AppSettings& settings, const QString& cacheDir, QObject* parent)
    : QObject(parent)
    , m_settings(settings)
    , m_cache(cacheDir)
    , m_resume(new QTimer(this))
{
    m_resume->setSingleShot(true);
    connect(m_resume, &QTimer::timeout, this, &CoverArtService::pump);
    connect(&m_settings, &AppSettings::changed, this, [this](const QString& key) {
        if (key == QLatin1String(AppSettings::Key::ArtLookups))
            pump();
    });
}

CoverArtService::~CoverArtService() = default;

bool CoverArtService::canLookUp() const
{
    return lookupsUnavailableReason().isEmpty();
}

QString CoverArtService::lookupsUnavailableReason() const
{
    if (!m_settings.artLookups())
        return QStringLiteral("Art downloads are off.");
    return {};
}

bool CoverArtService::usesPersonalKey() const
{
    return !m_settings.steamGridDbKey().isEmpty();
}

QUrl CoverArtService::apiBase() const
{
    if (usesPersonalKey())
        return QUrl(QString::fromLatin1(kApiBase));
    const QString proxy = qEnvironmentVariable("XSEMU_SGDB_PROXY").trimmed();
    return QUrl(proxy.isEmpty() ? QString::fromLatin1(kProxyBase) : proxy);
}

QString CoverArtService::lastError(const QString& titleId) const
{
    const auto it = m_failures.constFind(titleId);
    if (it == m_failures.cend())
        return {};
    const QDateTime fetched = m_cache.entry(titleId).fetchedAt;
    return fetched.isValid() && fetched >= it->at ? QString() : it->message;
}

QString CoverArtService::artPath(const QString& titleId, CoverCache::Kind kind) const
{
    return m_cache.path(titleId, kind);
}

QString CoverArtService::searchTerm(const Game& game)
{
    QString term = game.title();
    term.remove(QRegularExpression(QStringLiteral("[\\x{2122}\\x{00AE}\\x{00A9}]")));
    return term.simplified();
}

QString CoverArtService::comparableName(const QString& name)
{
    QString s = name.toCaseFolded();
    s.replace(QLatin1Char('&'), QStringLiteral(" and "));
    QString out;
    for (const QChar c : std::as_const(s))
        out += c.isLetterOrNumber() ? c : QLatin1Char(' ');
    return out.simplified();
}

QUrl CoverArtService::searchUrl(const QUrl& base, const QString& term)
{
    return apiPath(base, QStringLiteral("/search/autocomplete/") + QString::fromLatin1(QUrl::toPercentEncoding(term)));
}

QUrl CoverArtService::artUrl(const QUrl& base, CoverCache::Kind kind, qint64 gameId)
{
    static const char* const kPaths[] = {"grids", "heroes", "logos", "icons"};
    QUrl url = apiPath(base, QStringLiteral("/%1/game/%2").arg(QLatin1String(kPaths[size_t(kind)])).arg(gameId));
    QUrlQuery q;
    if (kind == CoverCache::Kind::Grid)
        q.addQueryItem(QStringLiteral("dimensions"), QStringLiteral("600x900"));
    q.addQueryItem(QStringLiteral("types"), QStringLiteral("static"));
    url.setQuery(q);
    return url;
}

QList<CoverArtService::Match> CoverArtService::parseSearch(const QByteArray& body)
{
    QList<Match> out;
    const QJsonObject root = QJsonDocument::fromJson(body).object();
    if (!root.value(QStringLiteral("success")).toBool())
        return out;
    for (const QJsonValue& v : root.value(QStringLiteral("data")).toArray()) {
        const QJsonObject o = v.toObject();
        Match m;
        m.id = static_cast<qint64>(o.value(QStringLiteral("id")).toDouble());
        m.name = o.value(QStringLiteral("name")).toString();
        m.verified = o.value(QStringLiteral("verified")).toBool();
        if (m.id > 0)
            out.append(m);
    }
    return out;
}

int CoverArtService::bestMatch(const QList<Match>& matches, const QString& term)
{
    if (matches.isEmpty())
        return -1;
    const QString want = comparableName(term);
    int exact = -1;
    for (int i = 0; i < matches.size(); ++i) {
        if (comparableName(matches[i].name) != want)
            continue;
        if (matches[i].verified)
            return i;
        if (exact < 0)
            exact = i;
    }
    return exact >= 0 ? exact : 0;
}

CoverArtService::Image CoverArtService::parseFirstImage(const QByteArray& body, const QUrl& base)
{
    const QJsonObject root = QJsonDocument::fromJson(body).object();
    if (!root.value(QStringLiteral("success")).toBool())
        return {};
    for (const QJsonValue& v : root.value(QStringLiteral("data")).toArray()) {
        const QJsonObject o = v.toObject();
        const QUrl url(o.value(QStringLiteral("url")).toString());
        if (url.isValid() && (url.scheme() == QLatin1String("https") || (url.scheme() == base.scheme() && url.host() == base.host())))
            return {url, o.value(QStringLiteral("author")).toObject().value(QStringLiteral("name")).toString()};
    }
    return {};
}

QNetworkAccessManager* CoverArtService::net()
{
    if (!m_net)
        m_net = new QNetworkAccessManager(this);
    return m_net;
}

QNetworkReply* CoverArtService::apiGet(const QUrl& url)
{
    QNetworkRequest req(url);
    if (usesPersonalKey())
        req.setRawHeader("Authorization", "Bearer " + m_settings.steamGridDbKey().toUtf8());
    req.setRawHeader("Accept", "application/json");
    req.setHeader(QNetworkRequest::UserAgentHeader, QStringLiteral("XSEmuLauncher"));
    req.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::NoLessSafeRedirectPolicy);
    req.setTransferTimeout(20000);
    return net()->get(req);
}

void CoverArtService::request(const Game& game)
{
    const QString id = game.titleId();
    if (!game.isValid() || !canLookUp() || m_pending.contains(id))
        return;
    const CoverCache::Entry e = m_cache.entry(id);
    if (m_cache.contains(id) && e.fetchedAt.isValid() && e.version >= CoverCache::kVersion)
        return;
    const QDateTime failed = m_failures.value(id).at;
    if (failed.isValid() && failed.secsTo(QDateTime::currentDateTimeUtc()) < kRetryAfterFailureS)
        return;
    Job job{game, {}};
    if (e.pinned)
        job.forced = {e.gameId, e.name, false};
    enqueue(job);
}

void CoverArtService::refresh(const Game& game)
{
    if (!game.isValid() || !canLookUp() || m_pending.contains(game.titleId()))
        return;
    m_failures.remove(game.titleId());
    const CoverCache::Entry e = m_cache.entry(game.titleId());
    Job job{game, {}};
    if (e.pinned)
        job.forced = {e.gameId, e.name, false};
    enqueue(job);
}

void CoverArtService::clearCache()
{
    m_cache.clear();
    m_failures.clear();
    emit artChanged(QString());
}

void CoverArtService::search(const QString& term, QObject* context, std::function<void(const QList<Match>&, const QString& error)> done)
{
    if (!canLookUp()) {
        done({}, lookupsUnavailableReason());
        return;
    }
    QNetworkReply* reply = apiGet(searchUrl(apiBase(), term.simplified()));
    connect(reply, &QNetworkReply::finished, context, [reply, done] {
        reply->deleteLater();
        const QByteArray body = reply->readAll();
        if (httpStatus(reply) == 200)
            done(parseSearch(body), {});
        else
            done({}, errorMessage(reply, body));
    });
}

void CoverArtService::setMatch(const Game& game, const Match& match)
{
    if (!game.isValid() || match.id <= 0)
        return;
    const QString id = game.titleId();
    m_queue.erase(std::remove_if(m_queue.begin(), m_queue.end(), [&](const Job& j) { return j.game.titleId() == id; }), m_queue.end());
    m_failures.remove(id);
    enqueue({game, match});
}

void CoverArtService::clearMatch(const Game& game)
{
    const QString id = game.titleId();
    CoverCache::Entry e = m_cache.entry(id);
    if (!e.pinned)
        return;
    e.pinned = false;
    m_cache.put(id, e);
    m_queue.erase(std::remove_if(m_queue.begin(), m_queue.end(), [&](const Job& j) { return j.game.titleId() == id; }), m_queue.end());
    m_failures.remove(id);
    emit artChanged(id);
    if (canLookUp())
        enqueue({game, {}});
}

bool CoverArtService::setCustomImage(const QString& titleId, CoverCache::Kind kind, const QString& file)
{
    static const QStringList kImageTypes{QStringLiteral("png"), QStringLiteral("jpg"), QStringLiteral("jpeg"), QStringLiteral("webp"), QStringLiteral("bmp"), QStringLiteral("gif")};
    const QFileInfo src(file);
    const QString ext = src.suffix().toLower();
    if (titleId.isEmpty() || !src.isFile() || !kImageTypes.contains(ext) || src.size() > kMaxImageBytes)
        return false;
    CoverCache::Entry e = m_cache.entry(titleId);
    const QDir dir(m_cache.dir());
    QDir().mkpath(m_cache.dir());
    const QString name = CoverCache::fileName(titleId, kind, ext, true);
    const QString dest = dir.filePath(name);
    if (src.canonicalFilePath() == QFileInfo(dest).canonicalFilePath() && e.customFile(kind) == name)
        return true;
    const QString part = dest + QStringLiteral(".part");
    QFile::remove(part);
    if (!QFile::copy(src.absoluteFilePath(), part)) {
        QFile::remove(part);
        return false;
    }
    const QString backup = dest + QStringLiteral(".old");
    QFile::remove(backup);
    if (QFile::exists(dest) && !QFile::rename(dest, backup)) {
        QFile::remove(part);
        return false;
    }
    if (!QFile::rename(part, dest)) {
        QFile::rename(backup, dest);
        QFile::remove(part);
        return false;
    }
    QFile::remove(backup);
    if (!e.customFile(kind).isEmpty() && e.customFile(kind) != name)
        QFile::remove(dir.filePath(e.customFile(kind)));
    e.customFile(kind) = name;
    m_cache.put(titleId, e);
    emit artChanged(titleId);
    return true;
}

void CoverArtService::clearCustomImage(const QString& titleId, CoverCache::Kind kind)
{
    if (!m_cache.contains(titleId))
        return;
    CoverCache::Entry e = m_cache.entry(titleId);
    if (e.customFile(kind).isEmpty())
        return;
    QFile::remove(QDir(m_cache.dir()).filePath(e.customFile(kind)));
    e.customFile(kind).clear();
    m_cache.put(titleId, e);
    emit artChanged(titleId);
}

void CoverArtService::enqueue(const Job& job)
{
    m_pending.insert(job.game.titleId());
    m_queue.append(job);
    pump();
}

// Starts the next queued lookup. With art downloads off it drops the queue instead, and the lookup
// in flight stops at its next request (fetchKind).
void CoverArtService::pump()
{
    if (!canLookUp()) {
        m_resume->stop();
        for (const Job& j : std::as_const(m_queue)) {
            if (!(m_busy && m_job.game.titleId() == j.game.titleId()))
                m_pending.remove(j.game.titleId());
        }
        m_queue.clear();
        return;
    }
    if (m_busy || m_queue.isEmpty() || m_resume->isActive())
        return;
    m_job = m_queue.takeFirst();
    m_busy = true;
    m_found = {};
    m_found.fetchedAt = QDateTime::currentDateTimeUtc();
    if (m_job.forced.id > 0) {
        m_found.gameId = m_job.forced.id;
        m_found.name = m_job.forced.name;
        m_found.pinned = true;
        fetchKind(0);
        return;
    }
    startSearch();
}

// A 429 puts the job back in front and holds the queue for the Retry-After the proxy sent.
bool CoverArtService::holdOnRateLimit(QNetworkReply* reply)
{
    if (httpStatus(reply) != 429)
        return false;
    bool ok = false;
    const int s = reply->rawHeader("Retry-After").toInt(&ok);
    m_queue.prepend(m_job);
    m_busy = false;
    m_resume->start((ok ? qBound(1, s, 600) : 60) * 1000);
    return true;
}

void CoverArtService::startSearch()
{
    QNetworkReply* reply = apiGet(searchUrl(apiBase(), searchTerm(m_job.game)));
    connect(reply, &QNetworkReply::finished, this, [this, reply] {
        reply->deleteLater();
        if (holdOnRateLimit(reply))
            return;
        if (httpStatus(reply) != 200) {
            failCurrent(errorMessage(reply, reply->readAll()));
            return;
        }
        const QList<Match> matches = parseSearch(reply->readAll());
        const int best = bestMatch(matches, searchTerm(m_job.game));
        if (best < 0) {
            m_found.notFound = true;
            finishCurrent();
            return;
        }
        m_found.gameId = matches[best].id;
        m_found.name = matches[best].name;
        fetchKind(0);
    });
}

void CoverArtService::fetchKind(size_t index)
{
    if (index >= CoverCache::kKinds.size()) {
        finishCurrent();
        return;
    }
    if (!canLookUp()) {
        m_busy = false;
        m_pending.remove(m_job.game.titleId());
        emit artChanged(m_job.game.titleId());
        pump();
        return;
    }
    const CoverCache::Kind kind = CoverCache::kKinds[index];
    const QUrl base = apiBase();
    QNetworkReply* reply = apiGet(artUrl(base, kind, m_found.gameId));
    connect(reply, &QNetworkReply::finished, this, [this, reply, kind, index, base] {
        reply->deleteLater();
        if (holdOnRateLimit(reply))
            return;
        const int status = httpStatus(reply);
        if (status != 200 && status != 404) {
            failCurrent(errorMessage(reply, reply->readAll()));
            return;
        }
        const Image image = status == 200 ? parseFirstImage(reply->readAll(), base) : Image();
        downloadImage(kind, image, [this, index] { fetchKind(index + 1); });
    });
}

void CoverArtService::downloadImage(CoverCache::Kind kind, const Image& image, std::function<void()> next)
{
    if (!image.url.isValid()) {
        next();
        return;
    }
    QNetworkRequest req(image.url);
    req.setHeader(QNetworkRequest::UserAgentHeader, QStringLiteral("XSEmuLauncher"));
    req.setAttribute(QNetworkRequest::RedirectPolicyAttribute, QNetworkRequest::NoLessSafeRedirectPolicy);
    req.setTransferTimeout(30000);
    QNetworkReply* reply = net()->get(req);
    connect(reply, &QNetworkReply::finished, this, [this, reply, kind, image, next] {
        reply->deleteLater();
        const int status = httpStatus(reply);
        if (status == 0 || status == 429 || status >= 500) {
            failCurrent(errorMessage(reply, reply->readAll()));
            return;
        }
        if (status >= 400) {
            next();
            return;
        }
        const QByteArray bytes = reply->readAll();
        if (reply->error() != QNetworkReply::NoError) {
            failCurrent(reply->errorString());
            return;
        }
        if (!reply->header(QNetworkRequest::ContentTypeHeader).toString().startsWith(QLatin1String("image/")) || bytes.isEmpty() || bytes.size() > kMaxImageBytes) {
            failCurrent(QStringLiteral("SteamGridDB's image server sent something that is not an image."));
            return;
        }
        const QString name = CoverCache::fileName(m_job.game.titleId(), kind, QFileInfo(image.url.path()).suffix());
        QDir().mkpath(m_cache.dir());
        QSaveFile f(QDir(m_cache.dir()).filePath(name));
        if (!f.open(QIODevice::WriteOnly) || f.write(bytes) != bytes.size() || !f.commit()) {
            failCurrent(QStringLiteral("Couldn't save the art: %1").arg(f.errorString()));
            return;
        }
        m_found.file(kind) = name;
        m_found.authors[size_t(kind)] = image.author;
        next();
    });
}

void CoverArtService::finishCurrent()
{
    const QString id = m_job.game.titleId();
    CoverCache::Entry e = m_found;
    const CoverCache::Entry old = m_cache.entry(id);
    e.custom = old.custom;
    const QDir dir(m_cache.dir());
    for (CoverCache::Kind kind : CoverCache::kKinds) {
        const QString was = old.fetched[size_t(kind)];
        if (!was.isEmpty() && was != e.file(kind))
            QFile::remove(dir.filePath(was));
    }
    m_cache.put(id, e);
    m_failures.remove(id);
    m_busy = false;
    if (std::none_of(m_queue.cbegin(), m_queue.cend(), [&](const Job& j) { return j.game.titleId() == id; }))
        m_pending.remove(id);
    emit artChanged(id);
    pump();
}

void CoverArtService::failCurrent(const QString& message)
{
    const QString id = m_job.game.titleId();
    m_failures.insert(id, {QDateTime::currentDateTimeUtc(), message});
    m_busy = false;
    if (std::none_of(m_queue.cbegin(), m_queue.cend(), [&](const Job& j) { return j.game.titleId() == id; }))
        m_pending.remove(id);
    emit lookupFailed(id, message);
    pump();
}
