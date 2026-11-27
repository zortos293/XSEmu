#pragma once

#include "Game.h"

#include <QDateTime>
#include <QHash>
#include <QList>
#include <QObject>
#include <QSet>
#include <QUrl>

#include <array>
#include <functional>

class AppSettings;
class QNetworkAccessManager;
class QNetworkReply;
class QTimer;

// Cover art cache: <cover cache>\index.json plus one image file per title and kind.
// index.json: { "<titleId>": { "v": 2, "gameId": 123, "name": "...", "pinned": false,
//               "grid": "<file>", "hero": "<file>", "logo": "<file>", "icon": "<file>",
//               "custom": { "grid": "<file>" }, "authors": { "grid": "<artist>" },
//               "fetchedAt": "<ISO>", "notFound": false } }
// The plain kind keys are what SteamGridDB gave; "custom" holds the user's own images, which win.
class CoverCache {
public:
    enum class Kind { Grid, Hero, Logo, Icon };
    static constexpr std::array<Kind, 4> kKinds{Kind::Grid, Kind::Hero, Kind::Logo, Kind::Icon};
    static constexpr int kVersion = 2; // entries from before logos and icons are fetched again once

    struct Entry {
        int version = kVersion;
        qint64 gameId = 0;
        QString name;  // the SteamGridDB game name matched
        bool pinned = false; // the user chose this match; refreshes keep it
        std::array<QString, 4> fetched; // file names in the cache folder by Kind, empty when none
        std::array<QString, 4> custom;
        std::array<QString, 4> authors; // the SteamGridDB artist of each fetched image
        QDateTime fetchedAt;
        bool notFound = false;

        QString& file(Kind kind) { return fetched[size_t(kind)]; }
        const QString& file(Kind kind) const { return fetched[size_t(kind)]; }
        QString& customFile(Kind kind) { return custom[size_t(kind)]; }
        const QString& customFile(Kind kind) const { return custom[size_t(kind)]; }
    };

    explicit CoverCache(QString dir);

    QString dir() const { return m_dir; }
    bool contains(const QString& titleId) const { return m_entries.contains(titleId); }
    Entry entry(const QString& titleId) const { return m_entries.value(titleId); }
    void put(const QString& titleId, const Entry& entry); // saves the index
    void remove(const QString& titleId);
    void clear();

    // Absolute path of the image shown for that kind (the user's own first), or empty.
    QString path(const QString& titleId, Kind kind) const;
    bool isCustom(const QString& titleId, Kind kind) const;
    static QString kindName(Kind kind); // "grid", "hero", "logo", "icon"
    // The file name an image of that kind gets (titleId is already a safe file name).
    static QString fileName(const QString& titleId, Kind kind, const QString& extension, bool custom = false);

private:
    void load();
    void save() const;
    QString existing(const QString& file) const;

    QString m_dir;
    QHash<QString, Entry> m_entries;
};

// Cover, banner, logo and icon art from SteamGridDB (API v2). Without a key of the user's own it
// goes through XSEmu's proxy (tools/sgdb-proxy), which holds a SteamGridDB key as a Worker secret, so
// nobody has to enter one; with a personal key it calls SteamGridDB directly. Art downloads are on by
// default and the user can turn them off (AppSettings::artLookups). The offline setting is for the
// title, not the launcher, so it does not block them.
//
//   covers.request(game);                       // looks the game up unless it is cached
//   connect(&covers, &CoverArtService::artChanged, this, [&](const QString& titleId) { ... });
//   QString file = covers.artPath(game.titleId(), CoverCache::Kind::Grid);   // empty: no art
//
// Lookups run one game at a time; a 429 from the proxy pauses the queue for its Retry-After.
class CoverArtService : public QObject {
    Q_OBJECT
public:
    static constexpr const char* kApiBase = "https://www.steamgriddb.com/api/v2";
    static constexpr const char* kProxyBase = "https://xsemu-sgdb.zortosgames293.workers.dev/api/v2";
    static constexpr int kRetryAfterFailureS = 600; // a failed lookup is not retried on its own sooner

    struct Match {
        qint64 id = 0;
        QString name;
        bool verified = false;
    };
    struct Image {
        QUrl url;
        QString author;
    };

    explicit CoverArtService(AppSettings& settings, QObject* parent = nullptr);
    CoverArtService(AppSettings& settings, const QString& cacheDir, QObject* parent = nullptr);
    ~CoverArtService() override;

    bool canLookUp() const;
    QString lookupsUnavailableReason() const; // empty when lookups can run
    bool usesPersonalKey() const;
    // The API base used: SteamGridDB with a personal key, else XSEMU_SGDB_PROXY, else kProxyBase.
    QUrl apiBase() const;

    QString artPath(const QString& titleId, CoverCache::Kind kind) const;
    bool isPending(const QString& titleId) const { return m_pending.contains(titleId); }
    // Why the last lookup of this title failed, when it failed after the art it has was fetched.
    QString lastError(const QString& titleId) const;

    void request(const Game& game);  // look up unless cached (a "not found" is cached too)
    void refresh(const Game& game);  // fetch again: the chosen match when there is one, else a new search
    void clearCache();

    // Choosing art by hand. A match the user picks is kept by refreshes until clearMatch.
    void search(const QString& term, QObject* context, std::function<void(const QList<Match>&, const QString& error)> done);
    void setMatch(const Game& game, const Match& match);
    void clearMatch(const Game& game);
    // Copies an image file into the cache as this title's own art of that kind; false when it cannot.
    bool setCustomImage(const QString& titleId, CoverCache::Kind kind, const QString& file);
    void clearCustomImage(const QString& titleId, CoverCache::Kind kind);

    static QString searchTerm(const Game& game);
    static QUrl searchUrl(const QUrl& base, const QString& term);
    static QUrl artUrl(const QUrl& base, CoverCache::Kind kind, qint64 gameId);
    // An API v2 response body, read (tested).
    static QList<Match> parseSearch(const QByteArray& body);
    // The result to use for term: an exact name match (verified first), else SteamGridDB's first.
    static int bestMatch(const QList<Match>& matches, const QString& term);
    static QString comparableName(const QString& name);
    // The first image served over https (or from the API's own host), with its artist.
    static Image parseFirstImage(const QByteArray& body, const QUrl& base);

    CoverCache& cache() { return m_cache; }

signals:
    void artChanged(const QString& titleId);
    void lookupFailed(const QString& titleId, const QString& message);

private:
    struct Job {
        Game game;
        Match forced; // id 0: search by name
    };

    void enqueue(const Job& job);
    void pump();
    void startSearch();
    void fetchKind(size_t index);
    void downloadImage(CoverCache::Kind kind, const Image& image, std::function<void()> next);
    bool holdOnRateLimit(QNetworkReply* reply);
    void finishCurrent();
    void dropStaged();
    void failCurrent(const QString& message);
    QNetworkReply* apiGet(const QUrl& url);
    QNetworkAccessManager* net();

    AppSettings& m_settings;
    CoverCache m_cache;
    QNetworkAccessManager* m_net = nullptr;
    QTimer* m_resume = nullptr;
    QList<Job> m_queue;
    bool m_busy = false;
    Job m_job;
    CoverCache::Entry m_found; // what the current job found so far; its files are staged as <name>.part until it finishes
    struct Failure {
        QDateTime at;
        QString message;
    };
    QSet<QString> m_pending;
    QHash<QString, Failure> m_failures;
};
