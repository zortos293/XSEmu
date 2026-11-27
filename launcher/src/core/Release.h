#pragma once

#include <QDateTime>
#include <QJsonObject>
#include <QList>
#include <QString>
#include <QStringList>

// The releases of zortos293/XSEmu, as the launcher's updater reads them: a version string, and per
// part (the launcher and the XSEmu Core) one release asset with its SHA-256 and the files inside it.
// .github/workflows/release.yml and scripts/package_release.py produce xsemu-manifest.json.
//
//   ReleaseVersion v = ReleaseVersion::parse("0.1.0-pre.2");   // numbers {0,1,0}, prerelease {"pre","2"}
//   if (newerThan(candidate, installed)) ...
struct ReleaseVersion {
    QList<int> numbers;     // a release of 0.1.0 is {0, 1, 0}
    QStringList prerelease; // 0.1.0-pre.2 is {"pre", "2"}; empty for a release
    bool valid() const { return !numbers.isEmpty(); }
    static ReleaseVersion parse(const QString& text); // a leading "v" is dropped ("v0.1.0")
    QString toString() const;
};

// <0 when a is older, 0 when they are the same version, >0 when a is newer.
int compare(const ReleaseVersion& a, const ReleaseVersion& b);

// Whether candidate replaces installed, by version order alone: 0.1.1 above 0.1.0, and a release above
// its own pre-releases (0.1.0 above 0.1.0-pre.2). No exception is made for a version that only names
// its release family, such as the 0.1.0-pre.1 build whose --version answers "0.1.0": tag a release
// above it (v0.1.1-pre.1) and it updates; treating 0.1.0-pre.2 as newer than 0.1.0 would move a
// finished 0.1.0 install onto a pre-release of the version it already has.
bool newerThan(const QString& candidate, const QString& installed);

struct ReleaseFile {
    QString name;   // relative path inside the archive
    QString sha256; // lowercase hex
};

struct ReleasePart {
    QString version; // the part's own version; the launcher's and the Core's may differ
    QString asset;   // file name of the release asset
    QString url;     // its download URL
    QString sha256;  // lowercase hex of the whole asset
    qint64 size = -1;
    QList<ReleaseFile> files; // what the archive must hold (the Core's program and its version file)
    bool valid() const;
};

struct ReleaseInfo {
    QString tag;     // "v0.1.0-pre.2"
    QString version; // "0.1.0-pre.2"
    bool prerelease = false;
    QDateTime publishedAt;
    ReleasePart launcher;
    ReleasePart core;
    bool hasParts() const { return launcher.valid() || core.valid(); }
};

// The download URL of one asset in a GitHub release object (each entry of .../releases), or empty.
QString releaseAssetUrl(const QJsonObject& release, const QString& assetName);

// A GitHub release object plus its xsemu-manifest.json asset. *why says what was wrong when the
// result has no parts. A release without a manifest (or with a manifest for another tag) has none.
ReleaseInfo parseRelease(const QJsonObject& release, const QJsonObject& manifest, QString* why);

// The manifest for the GitHub release at tag. Side-effect free: the caller fetches the bytes.
QJsonObject parseJsonObject(const QByteArray& bytes, QString* error);

// The SHA-256 of a file's bytes, lowercase hex; empty when the file cannot be read.
QString fileSha256(const QString& path);
