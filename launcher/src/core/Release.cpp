#include "Release.h"

#include <QCryptographicHash>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonValue>

namespace {

// "pre", "2" -> compared part by part: numbers numerically, words as text.
int comparePrerelease(const QStringList& a, const QStringList& b)
{
    if (a.isEmpty() && b.isEmpty())
        return 0;
    if (a.isEmpty())
        return 1; // a release outranks any pre-release of it
    if (b.isEmpty())
        return -1;
    for (int i = 0; i < a.size() && i < b.size(); ++i) {
        bool aNum = false, bNum = false;
        const int an = a[i].toInt(&aNum);
        const int bn = b[i].toInt(&bNum);
        int c = 0;
        if (aNum && bNum)
            c = an < bn ? -1 : an > bn ? 1 : 0;
        else
            c = QString::compare(a[i], b[i], Qt::CaseInsensitive);
        if (c != 0)
            return c < 0 ? -1 : 1;
    }
    return a.size() == b.size() ? 0 : a.size() < b.size() ? -1 : 1;
}

QString digestHex(const QJsonValue& v)
{
    // GitHub writes "sha256:<hex>" in an asset's digest; the manifest writes the bare hex.
    QString s = v.toString().trimmed().toLower();
    if (s.startsWith(QLatin1String("sha256:")))
        s = s.mid(7);
    return s.size() == 64 ? s : QString();
}

ReleasePart partFrom(const QJsonObject& manifestPart, const QJsonObject& release, QString* why)
{
    ReleasePart p;
    p.version = manifestPart.value(QLatin1String("version")).toString().trimmed();
    p.asset = manifestPart.value(QLatin1String("asset")).toString().trimmed();
    p.sha256 = manifestPart.value(QLatin1String("sha256")).toString().trimmed().toLower();
    p.size = qint64(manifestPart.value(QLatin1String("size")).toDouble(-1));
    const QJsonObject files = manifestPart.value(QLatin1String("files")).toObject();
    for (auto it = files.begin(); it != files.end(); ++it) {
        const QString name = it.key();
        const QString hash = digestHex(it.value());
        if (name.isEmpty() || hash.isEmpty())
            continue;
        p.files.append({name, hash});
    }
    if (p.version.isEmpty() || p.asset.isEmpty() || p.sha256.size() != 64) {
        if (why && why->isEmpty())
            *why = QStringLiteral("the manifest's %1 entry has no version, asset or sha256").arg(manifestPart.isEmpty() ? QStringLiteral("part") : p.asset);
        return {};
    }
    p.url = releaseAssetUrl(release, p.asset);
    if (p.url.isEmpty()) {
        if (why)
            *why = QStringLiteral("the release has no %1 asset").arg(p.asset);
        return {};
    }
    // The release's own digest is the authority for the bytes; a manifest that disagrees is wrong.
    const QJsonArray assets = release.value(QLatin1String("assets")).toArray();
    for (const QJsonValue& a : assets) {
        const QJsonObject o = a.toObject();
        if (o.value(QLatin1String("name")).toString() != p.asset)
            continue;
        const QString digest = digestHex(o.value(QLatin1String("digest")));
        if (!digest.isEmpty() && digest != p.sha256) {
            if (why)
                *why = QStringLiteral("%1's manifest sha256 does not match the asset's digest").arg(p.asset);
            return {};
        }
        if (p.size < 0)
            p.size = qint64(o.value(QLatin1String("size")).toDouble(-1));
    }
    return p;
}

}

bool ReleasePart::valid() const
{
    return !version.isEmpty() && !url.isEmpty() && sha256.size() == 64;
}

ReleaseVersion ReleaseVersion::parse(const QString& text)
{
    ReleaseVersion v;
    QString s = text.trimmed();
    if (s.startsWith(QLatin1Char('v')) || s.startsWith(QLatin1Char('V')))
        s = s.mid(1);
    if (s.isEmpty())
        return v;
    const QStringList halves = s.split(QLatin1Char('-'), Qt::KeepEmptyParts);
    const QStringList dotted = halves.value(0).split(QLatin1Char('.'), Qt::KeepEmptyParts);
    for (const QString& part : dotted) {
        bool ok = false;
        const int n = part.toInt(&ok);
        if (!ok || n < 0) {
            v.numbers.clear();
            return v;
        }
        v.numbers.append(n);
    }
    for (int i = 1; i < halves.size(); ++i) {
        for (const QString& part : halves[i].split(QLatin1Char('.'), Qt::SkipEmptyParts))
            v.prerelease.append(part);
    }
    return v;
}

QString ReleaseVersion::toString() const
{
    QStringList parts;
    for (int n : numbers)
        parts << QString::number(n);
    QString s = parts.join(QLatin1Char('.'));
    if (!prerelease.isEmpty())
        s += QLatin1Char('-') + prerelease.join(QLatin1Char('.'));
    return s;
}

int compare(const ReleaseVersion& a, const ReleaseVersion& b)
{
    if (!a.valid() && !b.valid())
        return 0;
    if (!a.valid())
        return -1;
    if (!b.valid())
        return 1;
    const int n = qMax(a.numbers.size(), b.numbers.size());
    for (int i = 0; i < n; ++i) {
        const int an = i < a.numbers.size() ? a.numbers[i] : 0;
        const int bn = i < b.numbers.size() ? b.numbers[i] : 0;
        if (an != bn)
            return an < bn ? -1 : 1;
    }
    return comparePrerelease(a.prerelease, b.prerelease);
}

bool newerThan(const QString& candidate, const QString& installed)
{
    const ReleaseVersion c = ReleaseVersion::parse(candidate);
    const ReleaseVersion i = ReleaseVersion::parse(installed);
    if (!c.valid() || !i.valid())
        return false;
    if (c.toString().compare(i.toString(), Qt::CaseInsensitive) == 0)
        return false;
    if (compare(c, i) > 0)
        return true;
    // The installed build only reports its release family ("0.1.0" for the 0.1.0-pre.1 build), so a
    // pre-release of that same family is newer than it.
    return !i.prerelease.isEmpty() ? false : candidate.trimmed().toLower().startsWith(installed.trimmed().toLower() + QLatin1Char('-'));
}

QString releaseAssetUrl(const QJsonObject& release, const QString& assetName)
{
    const QJsonArray assets = release.value(QLatin1String("assets")).toArray();
    for (const QJsonValue& a : assets) {
        const QJsonObject o = a.toObject();
        if (o.value(QLatin1String("name")).toString() != assetName)
            continue;
        return o.value(QLatin1String("browser_download_url")).toString();
    }
    return {};
}

QJsonObject parseJsonObject(const QByteArray& bytes, QString* error)
{
    QJsonParseError e;
    const QJsonDocument doc = QJsonDocument::fromJson(bytes, &e);
    if (e.error != QJsonParseError::NoError || !doc.isObject()) {
        if (error)
            *error = QStringLiteral("not a JSON object: %1").arg(e.errorString());
        return {};
    }
    return doc.object();
}

ReleaseInfo parseRelease(const QJsonObject& release, const QJsonObject& manifest, QString* why)
{
    ReleaseInfo r;
    if (why)
        why->clear();
    r.tag = release.value(QLatin1String("tag_name")).toString().trimmed();
    r.prerelease = release.value(QLatin1String("prerelease")).toBool();
    r.publishedAt = QDateTime::fromString(release.value(QLatin1String("published_at")).toString(), Qt::ISODate);
    r.version = manifest.value(QLatin1String("version")).toString().trimmed();
    const QString manifestTag = manifest.value(QLatin1String("tag")).toString().trimmed();
    if (r.version.isEmpty())
        r.version = r.tag.startsWith(QLatin1Char('v')) ? r.tag.mid(1) : r.tag;
    if (r.tag.isEmpty()) {
        if (why)
            *why = QStringLiteral("the release has no tag");
        return r;
    }
    if (!manifestTag.isEmpty() && manifestTag != r.tag) {
        if (why)
            *why = QStringLiteral("the manifest is for %1, not %2").arg(manifestTag, r.tag);
        return r;
    }
    const int schema = manifest.value(QLatin1String("schema")).toInt(0);
    if (schema != 1) {
        if (why)
            *why = QStringLiteral("the manifest schema is %1, not 1").arg(schema);
        return r;
    }
    r.launcher = partFrom(manifest.value(QLatin1String("launcher")).toObject(), release, why);
    r.core = partFrom(manifest.value(QLatin1String("core")).toObject(), release, why);
    return r;
}

QString fileSha256(const QString& path)
{
    QFile f(path);
    if (!f.open(QIODevice::ReadOnly))
        return {};
    QCryptographicHash hash(QCryptographicHash::Sha256);
    if (!hash.addData(&f))
        return {};
    return QString::fromLatin1(hash.result().toHex());
}
