#include "AppSettings.h"
#include "CoverArtService.h"

#include <QDir>
#include <QFile>
#include <QHash>
#include <QSignalSpy>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTemporaryDir>
#include <QTest>

namespace {

// A stand-in for the proxy: answers each GET path from a table and records what it was asked.
class FakeApi : public QObject {
public:
    struct Reply {
        int status = 200;
        QByteArray type = "application/json";
        QByteArray body;
        QByteArray extra; // more header lines
    };

    FakeApi()
    {
        m_server.listen(QHostAddress::LocalHost);
        connect(&m_server, &QTcpServer::newConnection, this, [this] {
            while (QTcpSocket* s = m_server.nextPendingConnection()) {
                connect(s, &QTcpSocket::readyRead, s, [this, s] { serve(s); });
                connect(s, &QTcpSocket::disconnected, s, &QObject::deleteLater);
            }
        });
    }

    QString base() const { return QStringLiteral("http://127.0.0.1:%1/api/v2").arg(m_server.serverPort()); }
    QString imageUrl(const QString& name) const { return QStringLiteral("http://127.0.0.1:%1/img/%2").arg(m_server.serverPort()).arg(name); }

    QHash<QString, QList<Reply>> routes; // path with query -> replies, the last one repeats
    QStringList asked;
    bool sawAuthorization = false;

private:
    void serve(QTcpSocket* s)
    {
        QByteArray& buf = m_buffers[s];
        buf += s->readAll();
        const qsizetype end = buf.indexOf("\r\n\r\n");
        if (end < 0)
            return;
        const QByteArray head = buf.left(end);
        buf.clear();
        const QList<QByteArray> lines = head.split('\n');
        const QString path = QString::fromLatin1(lines.first().split(' ').value(1));
        for (const QByteArray& l : lines) {
            if (l.toLower().startsWith("authorization:"))
                sawAuthorization = true;
        }
        asked << path;
        Reply r{404, "application/json", "{\"success\":false,\"errors\":[\"not here\"]}", {}};
        auto it = routes.find(path);
        if (it != routes.end() && !it->isEmpty())
            r = it->size() > 1 ? it->takeFirst() : it->first();
        QByteArray out = "HTTP/1.1 " + QByteArray::number(r.status) + " X\r\nContent-Type: " + r.type + "\r\nContent-Length: "
                         + QByteArray::number(r.body.size()) + "\r\nConnection: close\r\n" + r.extra + "\r\n" + r.body;
        s->write(out);
        s->disconnectFromHost();
    }

    QTcpServer m_server;
    QHash<QTcpSocket*, QByteArray> m_buffers;
};

QByteArray images(const QString& url, const QString& author)
{
    return QStringLiteral("{\"success\":true,\"data\":[{\"id\":5,\"url\":\"%1\",\"author\":{\"name\":\"%2\"}}]}").arg(url, author).toUtf8();
}

Game makeGame(const QString& dir, const QString& id, const QString& name)
{
    Game g;
    g.folder = dir;
    g.identityName = id;
    g.displayName = name;
    return g;
}

}

class TestCoverArt : public QObject {
    Q_OBJECT
private slots:
    void urls();
    void parsing();
    void matching();
    void cacheRoundTrip();
    void downloadsOnByDefaultAndCanBeTurnedOff();
    void fetchesEveryKindThroughTheProxy();
    void overridesSurviveRefresh();
    void rateLimitHoldsTheQueue();
    void failureIsNotRetriedOnItsOwn();
    void turningDownloadsOffStopsTheQueue();
    void failedImageDownloadKeepsTheOldArt();
    void replacingACustomImageIsSafe();
};

void TestCoverArt::urls()
{
    const QUrl api(QString::fromLatin1(CoverArtService::kApiBase));
    QCOMPARE(CoverArtService::searchUrl(api, QStringLiteral("Signal Garden")).toString(QUrl::FullyEncoded),
             QStringLiteral("https://www.steamgriddb.com/api/v2/search/autocomplete/Signal%20Garden"));
    QCOMPARE(CoverArtService::searchUrl(api, QStringLiteral("A/B & C?")).toString(QUrl::FullyEncoded),
             QStringLiteral("https://www.steamgriddb.com/api/v2/search/autocomplete/A%2FB%20%26%20C%3F"));
    QCOMPARE(CoverArtService::artUrl(api, CoverCache::Kind::Grid, 1234).toString(QUrl::FullyEncoded),
             QStringLiteral("https://www.steamgriddb.com/api/v2/grids/game/1234?dimensions=600x900&types=static"));
    QCOMPARE(CoverArtService::artUrl(api, CoverCache::Kind::Hero, 1234).toString(QUrl::FullyEncoded),
             QStringLiteral("https://www.steamgriddb.com/api/v2/heroes/game/1234?types=static"));
    QCOMPARE(CoverArtService::artUrl(api, CoverCache::Kind::Logo, 1234).toString(QUrl::FullyEncoded),
             QStringLiteral("https://www.steamgriddb.com/api/v2/logos/game/1234?types=static"));
    QCOMPARE(CoverArtService::artUrl(QUrl(QStringLiteral("http://h/api/v2/")), CoverCache::Kind::Icon, 7).toString(QUrl::FullyEncoded),
             QStringLiteral("http://h/api/v2/icons/game/7?types=static"));

    QTemporaryDir dir;
    AppSettings settings(dir.filePath(QStringLiteral("settings.ini")));
    CoverArtService covers(settings, dir.filePath(QStringLiteral("covers")));
    qunsetenv("XSEMU_SGDB_PROXY");
    QCOMPARE(covers.apiBase(), QUrl(QString::fromLatin1(CoverArtService::kProxyBase)));
    qputenv("XSEMU_SGDB_PROXY", "http://127.0.0.1:1/api/v2");
    QCOMPARE(covers.apiBase(), QUrl(QStringLiteral("http://127.0.0.1:1/api/v2")));
    settings.setSteamGridDbKey(QStringLiteral("mine"));
    QVERIFY(covers.usesPersonalKey());
    QCOMPARE(covers.apiBase(), api);
}

void TestCoverArt::parsing()
{
    const QByteArray found = "{\"success\":true,\"data\":[{\"id\":38,\"name\":\"Signal Garden\",\"types\":[\"steam\"],\"verified\":true},{\"id\":0},{\"id\":9,\"name\":\"Other\"}]}";
    const QByteArray empty = "{\"success\":true,\"data\":[]}";
    const QByteArray failed = "{\"success\":false,\"errors\":[\"bad key\"]}";
    const QByteArray imgs = "{\"success\":true,\"data\":[{\"id\":1,\"url\":\"http://insecure/x.png\"},{\"id\":2,\"url\":\"https://cdn.example/grid/a.png\",\"author\":{\"name\":\"Painter\"}}]}";
    const QList<CoverArtService::Match> m = CoverArtService::parseSearch(found);
    QCOMPARE(m.size(), 2);
    QCOMPARE(m[0].id, 38);
    QCOMPARE(m[0].name, QStringLiteral("Signal Garden"));
    QVERIFY(m[0].verified);
    QVERIFY(!m[1].verified);
    QVERIFY(CoverArtService::parseSearch(empty).isEmpty());
    QVERIFY(CoverArtService::parseSearch(failed).isEmpty());
    QVERIFY(CoverArtService::parseSearch("garbage").isEmpty());
    const QUrl api(QString::fromLatin1(CoverArtService::kApiBase));
    const CoverArtService::Image img = CoverArtService::parseFirstImage(imgs, api);
    QCOMPARE(img.url, QUrl(QStringLiteral("https://cdn.example/grid/a.png")));
    QCOMPARE(img.author, QStringLiteral("Painter"));
    QVERIFY(!CoverArtService::parseFirstImage(empty, api).url.isValid());
    QCOMPARE(CoverArtService::parseFirstImage(imgs, QUrl(QStringLiteral("http://insecure/api/v2"))).url, QUrl(QStringLiteral("http://insecure/x.png")));
}

void TestCoverArt::matching()
{
    using M = CoverArtService::Match;
    const QList<M> list{{1, QStringLiteral("Signal Garden: Remastered"), true}, {2, QStringLiteral("signal garden"), false}, {3, QStringLiteral("Signal Garden"), true}};
    QCOMPARE(CoverArtService::bestMatch(list, QStringLiteral("Signal Garden")), 2);
    QCOMPARE(CoverArtService::bestMatch(list.mid(0, 2), QStringLiteral("SIGNAL-GARDEN")), 1);
    QCOMPARE(CoverArtService::bestMatch(list, QStringLiteral("Unrelated")), 0);
    QCOMPARE(CoverArtService::bestMatch({}, QStringLiteral("x")), -1);
    QCOMPARE(CoverArtService::comparableName(QStringLiteral("Ratchet & Clank: Rift Apart")), QStringLiteral("ratchet and clank rift apart"));
    QCOMPARE(CoverArtService::searchTerm(makeGame(QStringLiteral("/g"), QStringLiteral("T"), QStringLiteral("  Signal\u2122  Garden\u00AE "))), QStringLiteral("Signal Garden"));
}

void TestCoverArt::cacheRoundTrip()
{
    QTemporaryDir dir;
    {
        CoverCache cache(dir.path());
        QVERIFY(!cache.contains(QStringLiteral("T.One")));
        for (const QString& name : {CoverCache::fileName(QStringLiteral("T.One"), CoverCache::Kind::Grid, QStringLiteral("PNG")),
                                    CoverCache::fileName(QStringLiteral("T.One"), CoverCache::Kind::Logo, QStringLiteral("png"), true)}) {
            QFile img(dir.filePath(name));
            QVERIFY(img.open(QIODevice::WriteOnly));
            img.write("png");
        }
        CoverCache::Entry e;
        e.gameId = 99;
        e.name = QStringLiteral("One");
        e.pinned = true;
        e.file(CoverCache::Kind::Grid) = QStringLiteral("T.One-grid.png");
        e.customFile(CoverCache::Kind::Logo) = QStringLiteral("T.One-custom-logo.png");
        e.authors[size_t(CoverCache::Kind::Grid)] = QStringLiteral("Painter");
        e.fetchedAt = QDateTime::currentDateTimeUtc();
        cache.put(QStringLiteral("T.One"), e);
        CoverCache::Entry missing;
        missing.notFound = true;
        cache.put(QStringLiteral("T.Two"), missing);
    }
    QFile legacy(dir.filePath(QStringLiteral("index.json")));
    QVERIFY(legacy.open(QIODevice::ReadWrite));
    QByteArray json = legacy.readAll();
    json.replace("{\n", "{\n    \"T.Old\": { \"gameId\": 5, \"grid\": \"T.Old-grid.png\", \"fetchedAt\": \"2026-01-01T00:00:00Z\" },\n");
    legacy.resize(0);
    legacy.write(json);
    legacy.close();

    CoverCache again(dir.path());
    const CoverCache::Entry one = again.entry(QStringLiteral("T.One"));
    QCOMPARE(one.gameId, 99);
    QVERIFY(one.pinned);
    QCOMPARE(one.version, CoverCache::kVersion);
    QCOMPARE(one.authors[size_t(CoverCache::Kind::Grid)], QStringLiteral("Painter"));
    QCOMPARE(again.path(QStringLiteral("T.One"), CoverCache::Kind::Grid), dir.filePath(QStringLiteral("T.One-grid.png")));
    QCOMPARE(again.path(QStringLiteral("T.One"), CoverCache::Kind::Logo), dir.filePath(QStringLiteral("T.One-custom-logo.png")));
    QVERIFY(again.isCustom(QStringLiteral("T.One"), CoverCache::Kind::Logo));
    QVERIFY(again.path(QStringLiteral("T.One"), CoverCache::Kind::Hero).isEmpty());
    QVERIFY(again.entry(QStringLiteral("T.Two")).notFound);
    QCOMPARE(again.entry(QStringLiteral("T.Old")).version, 1);
    QCOMPARE(again.entry(QStringLiteral("T.Old")).file(CoverCache::Kind::Grid), QStringLiteral("T.Old-grid.png"));

    again.clear();
    QVERIFY(!QFile::exists(dir.filePath(QStringLiteral("T.One-grid.png"))));
    QVERIFY(QFile::exists(dir.filePath(QStringLiteral("T.One-custom-logo.png"))));
    QVERIFY(!again.contains(QStringLiteral("T.Two")));
    QVERIFY(!again.entry(QStringLiteral("T.One")).fetchedAt.isValid());
    again.remove(QStringLiteral("T.One"));
    QVERIFY(!QFile::exists(dir.filePath(QStringLiteral("T.One-custom-logo.png"))));
    QVERIFY(!CoverCache(dir.path()).contains(QStringLiteral("T.One")));
}

void TestCoverArt::downloadsOnByDefaultAndCanBeTurnedOff()
{
    qputenv("XSEMU_SGDB_PROXY", "http://127.0.0.1:9/api/v2");
    QTemporaryDir dir;
    AppSettings settings(dir.filePath(QStringLiteral("settings.ini")));
    CoverArtService covers(settings, dir.filePath(QStringLiteral("covers")));
    const Game g = makeGame(dir.path(), QStringLiteral("T.Game"), QStringLiteral("Game"));
    QVERIFY(settings.artLookups());
    QVERIFY(covers.canLookUp());
    QVERIFY(!covers.usesPersonalKey());
    settings.setArtLookups(false);
    QVERIFY(covers.lookupsUnavailableReason().contains(QStringLiteral("off")));
    covers.request(g);
    covers.refresh(g);
    QVERIFY(!covers.isPending(g.titleId()));
    settings.setOffline(true);
    settings.setArtLookups(true);
    QVERIFY(covers.canLookUp());
}

void TestCoverArt::fetchesEveryKindThroughTheProxy()
{
    FakeApi api;
    qputenv("XSEMU_SGDB_PROXY", api.base().toUtf8());
    api.routes[QStringLiteral("/api/v2/search/autocomplete/Signal%20Garden")] = {
        {200, "application/json", "{\"success\":true,\"data\":[{\"id\":1,\"name\":\"Signal Garden 2\"},{\"id\":38,\"name\":\"Signal Garden\",\"verified\":true}]}", {}}};
    api.routes[QStringLiteral("/api/v2/grids/game/38?dimensions=600x900&types=static")] = {{200, "application/json", images(api.imageUrl(QStringLiteral("g.png")), QStringLiteral("Ann")), {}}};
    api.routes[QStringLiteral("/api/v2/heroes/game/38?types=static")] = {{200, "application/json", images(api.imageUrl(QStringLiteral("h.jpg")), QStringLiteral("Bo")), {}}};
    api.routes[QStringLiteral("/api/v2/icons/game/38?types=static")] = {{200, "application/json", images(api.imageUrl(QStringLiteral("i.png")), {}), {}}};
    for (const QString& f : {QStringLiteral("g.png"), QStringLiteral("h.jpg"), QStringLiteral("i.png")})
        api.routes[QStringLiteral("/img/") + f] = {{200, "image/png", "bytes-of-" + f.toLatin1(), {}}};

    QTemporaryDir dir;
    AppSettings settings(dir.filePath(QStringLiteral("settings.ini")));
    CoverArtService covers(settings, dir.filePath(QStringLiteral("covers")));
    QSignalSpy changed(&covers, &CoverArtService::artChanged);
    const Game g = makeGame(dir.path(), QStringLiteral("T.Signal"), QStringLiteral("Signal Garden\u2122"));
    covers.request(g);
    QVERIFY(covers.isPending(g.titleId()));
    QTRY_COMPARE_WITH_TIMEOUT(changed.size(), 1, 10000);
    QVERIFY(!covers.isPending(g.titleId()));
    QVERIFY(!api.sawAuthorization);

    const CoverCache::Entry e = covers.cache().entry(g.titleId());
    QCOMPARE(e.gameId, 38);
    QCOMPARE(e.name, QStringLiteral("Signal Garden"));
    QVERIFY(!e.pinned);
    QCOMPARE(e.authors[size_t(CoverCache::Kind::Grid)], QStringLiteral("Ann"));
    QFile grid(covers.artPath(g.titleId(), CoverCache::Kind::Grid));
    QVERIFY(grid.open(QIODevice::ReadOnly));
    QCOMPARE(grid.readAll(), QByteArray("bytes-of-g.png"));
    QVERIFY(covers.artPath(g.titleId(), CoverCache::Kind::Hero).endsWith(QStringLiteral("T.Signal-hero.jpg")));
    QVERIFY(!covers.artPath(g.titleId(), CoverCache::Kind::Icon).isEmpty());
    QVERIFY(covers.artPath(g.titleId(), CoverCache::Kind::Logo).isEmpty());

    const qsizetype asked = api.asked.size();
    covers.request(g);
    QVERIFY(!covers.isPending(g.titleId()));
    QCOMPARE(api.asked.size(), asked);
}

void TestCoverArt::overridesSurviveRefresh()
{
    FakeApi api;
    qputenv("XSEMU_SGDB_PROXY", api.base().toUtf8());
    api.routes[QStringLiteral("/api/v2/search/autocomplete/Quiet")] = {{200, "application/json", "{\"success\":true,\"data\":[{\"id\":3,\"name\":\"Quiet\"}]}", {}}};
    api.routes[QStringLiteral("/api/v2/heroes/game/77?types=static")] = {{200, "application/json", images(api.imageUrl(QStringLiteral("h77.png")), QStringLiteral("Cy")), {}}};
    api.routes[QStringLiteral("/img/h77.png")] = {{200, "image/png", "hero77", {}}};

    QTemporaryDir dir;
    AppSettings settings(dir.filePath(QStringLiteral("settings.ini")));
    CoverArtService covers(settings, dir.filePath(QStringLiteral("covers")));
    QSignalSpy changed(&covers, &CoverArtService::artChanged);
    const Game g = makeGame(dir.path(), QStringLiteral("T.Quiet"), QStringLiteral("Quiet"));

    const QString mine = dir.filePath(QStringLiteral("mine.png"));
    QFile f(mine);
    QVERIFY(f.open(QIODevice::WriteOnly));
    f.write("my-cover");
    f.close();
    QVERIFY(!covers.setCustomImage(g.titleId(), CoverCache::Kind::Grid, dir.filePath(QStringLiteral("missing.png"))));
    QVERIFY(covers.setCustomImage(g.titleId(), CoverCache::Kind::Grid, mine));
    QVERIFY(covers.cache().isCustom(g.titleId(), CoverCache::Kind::Grid));

    covers.request(g); // a custom image alone does not count as looked up
    QTRY_VERIFY_WITH_TIMEOUT(!covers.isPending(g.titleId()), 10000);
    QCOMPARE(covers.cache().entry(g.titleId()).gameId, 3);

    covers.setMatch(g, {77, QStringLiteral("The Quiet"), false});
    QTRY_VERIFY_WITH_TIMEOUT(!covers.isPending(g.titleId()), 10000);
    CoverCache::Entry e = covers.cache().entry(g.titleId());
    QCOMPARE(e.gameId, 77);
    QVERIFY(e.pinned);
    QFile hero(covers.artPath(g.titleId(), CoverCache::Kind::Hero));
    QVERIFY(hero.open(QIODevice::ReadOnly));
    QCOMPARE(hero.readAll(), QByteArray("hero77"));
    QFile cover(covers.artPath(g.titleId(), CoverCache::Kind::Grid));
    QVERIFY(cover.open(QIODevice::ReadOnly));
    QCOMPARE(cover.readAll(), QByteArray("my-cover"));

    const int searches = api.asked.filter(QStringLiteral("/search/")).size();
    covers.refresh(g);
    QTRY_VERIFY_WITH_TIMEOUT(!covers.isPending(g.titleId()), 10000);
    QCOMPARE(api.asked.filter(QStringLiteral("/search/")).size(), searches);
    QCOMPARE(covers.cache().entry(g.titleId()).gameId, 77);

    covers.clearCustomImage(g.titleId(), CoverCache::Kind::Grid);
    QVERIFY(covers.artPath(g.titleId(), CoverCache::Kind::Grid).isEmpty());
    covers.clearMatch(g);
    QTRY_VERIFY_WITH_TIMEOUT(!covers.isPending(g.titleId()), 10000);
    QCOMPARE(covers.cache().entry(g.titleId()).gameId, 3);
    QVERIFY(!covers.cache().entry(g.titleId()).pinned);
}

void TestCoverArt::rateLimitHoldsTheQueue()
{
    FakeApi api;
    qputenv("XSEMU_SGDB_PROXY", api.base().toUtf8());
    api.routes[QStringLiteral("/api/v2/search/autocomplete/Busy")] = {
        {429, "application/json", "{\"success\":false,\"errors\":[\"slow down\"]}", "Retry-After: 1\r\n"},
        {200, "application/json", "{\"success\":true,\"data\":[]}", {}}};
    QTemporaryDir dir;
    AppSettings settings(dir.filePath(QStringLiteral("settings.ini")));
    CoverArtService covers(settings, dir.filePath(QStringLiteral("covers")));
    QSignalSpy failed(&covers, &CoverArtService::lookupFailed);
    const Game g = makeGame(dir.path(), QStringLiteral("T.Busy"), QStringLiteral("Busy"));
    QElapsedTimer t;
    t.start();
    covers.request(g);
    QTRY_VERIFY_WITH_TIMEOUT(!covers.isPending(g.titleId()), 10000);
    QVERIFY(t.elapsed() >= 900);
    QCOMPARE(failed.size(), 0);
    QVERIFY(covers.cache().entry(g.titleId()).notFound);
}

void TestCoverArt::failureIsNotRetriedOnItsOwn()
{
    FakeApi api;
    qputenv("XSEMU_SGDB_PROXY", api.base().toUtf8());
    api.routes[QStringLiteral("/api/v2/search/autocomplete/Down")] = {
        {503, "application/json", "{\"success\":false,\"errors\":[\"The proxy has no SteamGridDB key configured.\"]}", {}}};
    QTemporaryDir dir;
    AppSettings settings(dir.filePath(QStringLiteral("settings.ini")));
    CoverArtService covers(settings, dir.filePath(QStringLiteral("covers")));
    QSignalSpy failed(&covers, &CoverArtService::lookupFailed);
    const Game g = makeGame(dir.path(), QStringLiteral("T.Down"), QStringLiteral("Down"));
    covers.request(g);
    QTRY_COMPARE_WITH_TIMEOUT(failed.size(), 1, 10000);
    QCOMPARE(covers.lastError(g.titleId()), QStringLiteral("The proxy has no SteamGridDB key configured."));
    QVERIFY(!covers.cache().contains(g.titleId()));
    covers.request(g);
    QVERIFY(!covers.isPending(g.titleId()));
    covers.refresh(g);
    QVERIFY(covers.isPending(g.titleId()));
    QTRY_COMPARE_WITH_TIMEOUT(failed.size(), 2, 10000);
}

void TestCoverArt::turningDownloadsOffStopsTheQueue()
{
    FakeApi api;
    qputenv("XSEMU_SGDB_PROXY", api.base().toUtf8());
    api.routes[QStringLiteral("/api/v2/search/autocomplete/First")] = {{200, "application/json", "{\"success\":true,\"data\":[{\"id\":11,\"name\":\"First\"}]}", {}}};
    QTemporaryDir dir;
    AppSettings settings(dir.filePath(QStringLiteral("settings.ini")));
    CoverArtService covers(settings, dir.filePath(QStringLiteral("covers")));
    const Game first = makeGame(dir.path(), QStringLiteral("T.First"), QStringLiteral("First"));
    const Game second = makeGame(dir.path(), QStringLiteral("T.Second"), QStringLiteral("Second"));
    covers.request(first);
    covers.request(second);
    QVERIFY(covers.isPending(second.titleId()));
    settings.setArtLookups(false);
    QVERIFY(!covers.isPending(second.titleId()));
    QTRY_VERIFY_WITH_TIMEOUT(!covers.isPending(first.titleId()), 10000);
    QTest::qWait(200);
    QCOMPARE(api.asked, QStringList{QStringLiteral("/api/v2/search/autocomplete/First")});
    QVERIFY(!covers.cache().contains(first.titleId()));
}

void TestCoverArt::failedImageDownloadKeepsTheOldArt()
{
    FakeApi api;
    qputenv("XSEMU_SGDB_PROXY", api.base().toUtf8());
    api.routes[QStringLiteral("/api/v2/search/autocomplete/Kept")] = {{200, "application/json", "{\"success\":true,\"data\":[{\"id\":21,\"name\":\"Kept\"}]}", {}}};
    api.routes[QStringLiteral("/api/v2/grids/game/21?dimensions=600x900&types=static")] = {{200, "application/json", images(api.imageUrl(QStringLiteral("k.png")), QStringLiteral("Di")), {}}};
    api.routes[QStringLiteral("/img/k.png")] = {{200, "image/png", "kept-cover", {}}, {503, "text/plain", "busy", {}}, {200, "text/html", "<html>oops</html>", {}}};
    QTemporaryDir dir;
    AppSettings settings(dir.filePath(QStringLiteral("settings.ini")));
    CoverArtService covers(settings, dir.filePath(QStringLiteral("covers")));
    QSignalSpy failed(&covers, &CoverArtService::lookupFailed);
    const Game g = makeGame(dir.path(), QStringLiteral("T.Kept"), QStringLiteral("Kept"));
    covers.request(g);
    QTRY_VERIFY_WITH_TIMEOUT(!covers.isPending(g.titleId()), 10000);
    const QDateTime fetched = covers.cache().entry(g.titleId()).fetchedAt;
    covers.refresh(g);
    QTRY_COMPARE_WITH_TIMEOUT(failed.size(), 1, 10000);
    QCOMPARE(covers.cache().entry(g.titleId()).fetchedAt, fetched);
    QFile cover(covers.artPath(g.titleId(), CoverCache::Kind::Grid));
    QVERIFY(cover.open(QIODevice::ReadOnly));
    QCOMPARE(cover.readAll(), QByteArray("kept-cover"));
    QVERIFY(!covers.lastError(g.titleId()).isEmpty());

    covers.refresh(g); // a 200 that is not an image fails the lookup too
    QTRY_COMPARE_WITH_TIMEOUT(failed.size(), 2, 10000);
    QCOMPARE(covers.cache().entry(g.titleId()).fetchedAt, fetched);
    QVERIFY(QFile::exists(covers.artPath(g.titleId(), CoverCache::Kind::Grid)));
    QCOMPARE(QDir(covers.cache().dir()).entryList({QStringLiteral("*.part")}, QDir::Files), QStringList());
}

void TestCoverArt::replacingACustomImageIsSafe()
{
    qputenv("XSEMU_SGDB_PROXY", "http://127.0.0.1:9/api/v2");
    QTemporaryDir dir;
    AppSettings settings(dir.filePath(QStringLiteral("settings.ini")));
    CoverArtService covers(settings, dir.filePath(QStringLiteral("covers")));
    const QString id = QStringLiteral("T.Own");
    const auto write = [&](const QString& name, const QByteArray& bytes) {
        QFile f(dir.filePath(name));
        f.open(QIODevice::WriteOnly);
        f.write(bytes);
        return f.fileName();
    };
    QVERIFY(covers.setCustomImage(id, CoverCache::Kind::Hero, write(QStringLiteral("a.png"), "first")));
    const QString kept = covers.artPath(id, CoverCache::Kind::Hero);
    QVERIFY(covers.setCustomImage(id, CoverCache::Kind::Hero, kept));
    QFile same(kept);
    QVERIFY(same.open(QIODevice::ReadOnly));
    QCOMPARE(same.readAll(), QByteArray("first"));
    same.close();
    QVERIFY(covers.setCustomImage(id, CoverCache::Kind::Hero, write(QStringLiteral("b.jpg"), "second")));
    QVERIFY(!QFile::exists(kept));
    QFile now(covers.artPath(id, CoverCache::Kind::Hero));
    QVERIFY(now.open(QIODevice::ReadOnly));
    QCOMPARE(now.readAll(), QByteArray("second"));
    QVERIFY(!covers.setCustomImage(id, CoverCache::Kind::Hero, dir.filePath(QStringLiteral("gone.png"))));
    QVERIFY(QFile::exists(covers.artPath(id, CoverCache::Kind::Hero)));
}

QTEST_GUILESS_MAIN(TestCoverArt)
#include "tst_coverart.moc"
