#include "SettingsProfile.h"

#include "AppSettings.h"
#include "LaunchConfig.h"
#include "Runner.h"

#include <QJsonDocument>
#include <QJsonObject>
#include <QRegularExpression>
#include <QSet>
#include <QtEndian>

#include <algorithm>
#include <cmath>

namespace SettingsProfile {

namespace {

constexpr qsizetype kMaxInput = 256 * 1024;
constexpr quint32 kMaxJson = 64 * 1024;
constexpr int kInfoMax = 120;

QString s(const char* text) { return QString::fromUtf8(text); }

bool isXclKey(const QString& key)
{
    for (const LaunchConfig::XclKey& k : LaunchConfig::xclKeys()) {
        if (key == QLatin1String(k.key))
            return true;
    }
    return false;
}

bool isList(const QVariant& v)
{
    return v.typeId() == QMetaType::QStringList || v.typeId() == QMetaType::QVariantList;
}

bool same(const QVariant& a, const QVariant& b)
{
    if (!a.isValid() || !b.isValid())
        return a.isValid() == b.isValid();
    if (isList(a) || isList(b))
        return isList(a) && isList(b) && a.toStringList() == b.toStringList();
    if (a.typeId() == QMetaType::QString || b.typeId() == QMetaType::QString)
        return a.toString().compare(b.toString(), Qt::CaseInsensitive) == 0;
    if (a.typeId() == QMetaType::Bool || b.typeId() == QMetaType::Bool)
        return a.typeId() == b.typeId() && a.toBool() == b.toBool();
    return std::abs(a.toDouble() - b.toDouble()) < 1e-6;
}

bool isNumber(const QVariant& v)
{
    switch (v.typeId()) {
    case QMetaType::Double:
    case QMetaType::Float:
    case QMetaType::Int:
    case QMetaType::LongLong:
    case QMetaType::UInt:
    case QMetaType::ULongLong:
        return true;
    default:
        return false;
    }
}

QString shown(const QVariant& v)
{
    if (v.typeId() == QMetaType::Bool)
        return v.toBool() ? QStringLiteral("true") : QStringLiteral("false");
    if (isNumber(v))
        return QString::number(v.toDouble(), 'g', 6);
    if (v.typeId() == QMetaType::QString)
        return v.toString().left(40);
    if (isList(v))
        return QStringLiteral("a list of %1").arg(v.toList().size());
    return QStringLiteral("a value of the wrong type");
}

// The canonical value, or an invalid QVariant with *why set.
QVariant canonical(const KeySpec& spec, const QVariant& v, QString* why)
{
    switch (spec.kind) {
    case KeySpec::Kind::Bool:
        if (v.typeId() == QMetaType::Bool)
            return v;
        *why = QStringLiteral("must be true or false");
        return {};
    case KeySpec::Kind::Choice: {
        const bool numeric = isNumber(spec.choices.first().first);
        if (numeric ? !isNumber(v) : v.typeId() != QMetaType::QString) {
            *why = numeric ? QStringLiteral("must be a number") : QStringLiteral("must be text");
            return {};
        }
        for (const auto& c : spec.choices) {
            if (same(v, c.first))
                return c.first;
        }
        if (numeric)
            *why = QStringLiteral("%1 is outside what the launcher offers (%2 to %3)")
                       .arg(shown(v), spec.choices.first().second, spec.choices.last().second);
        else
            *why = QStringLiteral("\"%1\" isn't a value this launcher knows").arg(shown(v));
        return {};
    }
    case KeySpec::Kind::DisplayMode: {
        if (v.typeId() != QMetaType::QString) {
            *why = QStringLiteral("must be text");
            return {};
        }
        const QString mode = v.toString().trimmed().toLower();
        if (mode == QLatin1String("auto") || mode == QLatin1String("host"))
            return mode;
        // the modes XSEmu accepts (crates/xcl/src/config.rs DisplaySetting::parse)
        static const QRegularExpression re(QStringLiteral("^(\\d{1,5})x(\\d{1,5})(?:@(\\d{1,3}))?$"));
        const QRegularExpressionMatch m = re.match(mode);
        const int w = m.captured(1).toInt();
        const int h = m.captured(2).toInt();
        const int hz = m.captured(3).isEmpty() ? 60 : m.captured(3).toInt();
        if (m.hasMatch() && w >= 1 && w <= 16384 && h >= 1 && h <= 16384 && (hz == 60 || hz == 120))
            return mode;
        *why = QStringLiteral("\"%1\" isn't a display mode (auto, host, or WIDTHxHEIGHT at 60 or 120 Hz)").arg(shown(v));
        return {};
    }
    case KeySpec::Kind::Runner:
        if (v.typeId() == QMetaType::QString && Runners::find(v.toString()))
            return v.toString();
        *why = QStringLiteral("unknown runner \"%1\"; ignored").arg(shown(v));
        return {};
    case KeySpec::Kind::RunProfile: {
        const QVariantList items = isList(v) ? v.toList() : QVariantList();
        if (!isList(v) || std::any_of(items.cbegin(), items.cend(), [](const QVariant& i) { return i.typeId() != QMetaType::QString; })) {
            *why = QStringLiteral("must be a list of KEY=VALUE lines");
            return {};
        }
        const LaunchConfig::ProfileParse parsed = LaunchConfig::parseProfile(v.toStringList().join(QLatin1Char('\n')));
        if (!parsed.ok()) {
            *why = parsed.errors.first();
            return {};
        }
        return parsed.lines();
    }
    }
    return {};
}

QList<QPair<QString, QString>> profileEntries(const QVariant& v)
{
    return LaunchConfig::parseProfile(v.toStringList().join(QLatin1Char('\n'))).entries;
}

// The run profile without its path lines; paths gets their keys.
QStringList withoutPaths(const QVariant& profile, QStringList* paths = nullptr)
{
    QStringList out;
    for (const auto& [key, value] : profileEntries(profile)) {
        if (!looksLikePath(value))
            out << key + QLatin1Char('=') + value;
        else if (paths)
            *paths << key;
    }
    return out;
}

// An imported run profile with this PC's own path lines kept where the import doesn't name them.
QStringList withLocalPaths(const QVariant& imported, const QStringList& current)
{
    QStringList out = imported.toStringList();
    QSet<QString> named;
    for (const auto& e : profileEntries(imported))
        named.insert(e.first);
    for (const auto& [key, value] : profileEntries(current)) {
        if (looksLikePath(value) && !named.contains(key))
            out << key + QLatin1Char('=') + value;
    }
    return out;
}

// "changes XCL_LOOP_GUARD, adds XCL_FOO, removes XCL_BAR"
QString profileChanges(const QVariant& from, const QVariant& to)
{
    QHash<QString, QString> before;
    for (const auto& [key, value] : profileEntries(from))
        before.insert(key, value);
    QStringList changed, added;
    for (const auto& [key, value] : profileEntries(to)) {
        if (!before.contains(key))
            added << key;
        else if (before.take(key) != value)
            changed << key;
    }
    QStringList removed = before.keys();
    removed.sort();
    QStringList parts;
    const auto part = [&parts](const char* verb, const QStringList& keys) {
        if (keys.isEmpty())
            return;
        const QString shownKeys = keys.size() > 3 ? QStringLiteral("%1 and %2 more").arg(keys.mid(0, 3).join(QStringLiteral(", "))).arg(keys.size() - 3)
                                                  : keys.join(QStringLiteral(", "));
        parts << QStringLiteral("%1 %2").arg(QLatin1String(verb), shownKeys);
    };
    part("changes", changed);
    part("adds", added);
    part("removes", removed);
    return parts.join(QStringLiteral(", "));
}

QString forbiddenReason(const KeySpec& spec, const QVariant& v)
{
    if (!spec.unavailable.isEmpty())
        return spec.unavailable;
    if (spec.key == QLatin1String("offline") && !v.toBool())
        return QStringLiteral("XSEmu Core runs games offline. Online play isn't available.");
    if (spec.key == QLatin1String("store") && v.toString() != QLatin1String("offline-error"))
        return QStringLiteral("Games always get the store's offline error. XSEmu Core never reports a licence it doesn't have.");
    return {};
}

// Settings marks these rows unavailable while the environment sets them (Graphics section). The run
// profile travels with the import, so a variable it sets doesn't refuse the key here.
QString environmentReason(const QString& key)
{
    const char* var = key == QLatin1String("display_mode") ? "XCL_DISPLAY_MODE" : key == QLatin1String("raytracing") ? "XCL_RAYTRACING" : nullptr;
    if (!var)
        return {};
    const QString v = qEnvironmentVariable(var).trimmed();
    return v.isEmpty() ? QString() : QStringLiteral("%1=%2 in the environment overrides this.").arg(QLatin1String(var), v);
}

QVariant valid(const KeySpec& spec, const QVariant& v)
{
    QString why;
    return v.isValid() ? canonical(spec, v, &why) : QVariant();
}

QVariant currentDefault(const AppSettings& st, const KeySpec& spec)
{
    const QString& k = spec.key;
    if (k == QLatin1String("device_type"))
        return st.defaultDeviceType();
    if (k == QLatin1String("overlay"))
        return st.overlayDefault();
    if (k == QLatin1String("runner"))
        return st.defaultRunner();
    if (k == QLatin1String("offline"))
        return st.offline();
    if (k == QLatin1String("store"))
        return st.storeOfflineError() ? QStringLiteral("offline-error") : QStringLiteral("not-available");
    if (k == QLatin1String("crash_dumps"))
        return st.crashDumps();
    if (k == QLatin1String("run_profile"))
        return st.runProfile();
    if (isXclKey(k))
        return LaunchConfig::typed(k, st.xclDefault(k));
    return {};
}

QVariant currentForGame(const AppSettings& st, const QString& titleId, const KeySpec& spec)
{
    const QString& k = spec.key;
    if (k == QLatin1String("runner"))
        return st.runnerFor(titleId);
    if (isXclKey(k))
        return LaunchConfig::titleValues(st, titleId).value(k);
    const QVariant own = st.gameOverrides(titleId).value(k);
    if (own.isValid())
        return own;
    if (k == QLatin1String("overlay"))
        return st.overlayDefault();
    if (spec.kind == KeySpec::Kind::Bool && spec.game && !spec.defaults)
        return false;
    return {};
}

void writeDefault(AppSettings& st, const QString& k, const QVariant& v)
{
    if (k == QLatin1String("device_type"))
        st.setDefaultDeviceType(v.toString());
    else if (k == QLatin1String("overlay"))
        st.setOverlayDefault(v.toBool());
    else if (k == QLatin1String("runner"))
        st.setDefaultRunner(v.toString());
    else if (k == QLatin1String("offline"))
        st.setOffline(v.toBool());
    else if (k == QLatin1String("store"))
        st.setStoreOfflineError(v.toString() == QLatin1String("offline-error"));
    else if (k == QLatin1String("crash_dumps"))
        st.setCrashDumps(v.toBool());
    else if (k == QLatin1String("run_profile"))
        st.setRunProfile(v.toStringList());
    else if (isXclKey(k))
        st.setXclDefault(k, v);
}

QString clean(QString text, int max, bool multiLine)
{
    for (QChar& c : text) {
        if (c.category() == QChar::Other_Control && !(multiLine && c == QLatin1Char('\n')))
            c = QLatin1Char(' ');
    }
    return text.trimmed().left(max);
}

QJsonObject toObject(const Profile& p)
{
    QJsonObject o{{QStringLiteral("schema"), QLatin1String(kSchema)}, {QStringLiteral("version"), kVersion}};
    if (!p.name.isEmpty())
        o.insert(QStringLiteral("name"), clean(p.name, kNameMax, false));
    if (!p.note.isEmpty())
        o.insert(QStringLiteral("note"), clean(p.note, kNoteMax, true));
    QJsonObject made;
    if (!p.launcherVersion.isEmpty())
        made.insert(QStringLiteral("launcher"), p.launcherVersion);
    if (!p.gpuVendor.isEmpty())
        made.insert(QStringLiteral("gpuVendor"), p.gpuVendor);
    if (!p.gpuModel.isEmpty())
        made.insert(QStringLiteral("gpu"), p.gpuModel);
    if (!made.isEmpty())
        o.insert(QStringLiteral("madeWith"), made);
    if (!p.titleId.isEmpty())
        o.insert(QStringLiteral("game"), QJsonObject{{QStringLiteral("titleId"), p.titleId}, {QStringLiteral("title"), p.title}});
    QJsonObject values;
    for (const KeySpec& spec : keys()) {
        const QVariant v = valid(spec, p.values.value(spec.key));
        if (!v.isValid())
            continue;
        if (spec.kind == KeySpec::Kind::Runner) {
            QJsonObject runner{{QStringLiteral("id"), v.toString()}};
            if (!p.runnerVersion.isEmpty())
                runner.insert(QStringLiteral("version"), p.runnerVersion);
            o.insert(QStringLiteral("runner"), runner);
        } else {
            values.insert(spec.key, QJsonValue::fromVariant(v));
        }
    }
    o.insert(QStringLiteral("settings"), values);
    return o;
}

Parsed failed(const QString& error)
{
    Parsed p;
    p.error = error;
    return p;
}

QString infoString(const QJsonObject& o, const char* key, int max = kInfoMax)
{
    return clean(o.value(QLatin1String(key)).toString(), max, false);
}

}

const QList<KeySpec>& keys()
{
    using K = KeySpec::Kind;
    static const QString notInBuild = s("Not in this XSEmu Core build yet.");
    static const QList<KeySpec> list{
        {s("device_type"), s("Console"), K::Choice, {{s("lockhart"), s("Series S")}, {s("anaconda"), s("Series X")}}},
        {s("display_mode"), s("Display mode"), K::DisplayMode, {}},
        {s("raytracing"), s("Raytracing"), K::Choice, {{s("auto"), s("Auto")}, {s("tier1_1"), s("Tier 1.1")}, {s("tier1_0"), s("Tier 1.0")}, {s("off"), s("Off")}}},
        {s("keyboard_mouse"), s("Keyboard & mouse"), K::Bool, {}},
        {s("mouse_sensitivity"), s("Mouse speed"), K::Choice,
         {{0.5, s("0.5×")}, {0.75, s("0.75×")}, {1.0, s("1×")}, {1.5, s("1.5×")}, {2.0, s("2×")}, {3.0, s("3×")}}},
        {s("log_level"), s("Log level"), K::Choice, {{s("info"), s("Info")}, {s("warn"), s("Warn")}, {s("debug"), s("Debug")}, {s("trace"), s("Trace")}}},
        {s("present"), s("Show the game image"), K::Choice, {{s("auto"), s("On")}, {s("off"), s("Off")}}, false, true},
        {s("overlay"), s("Overlay"), K::Bool, {}},
        {s("runner"), s("Runs with"), K::Runner, {}},
        {s("alwaysWarmUp"), s("Always warm up first"), K::Bool, {}, true, false},
        {s("startIn"), s("Start in"), K::Choice, {{s("window"), s("Window")}, {s("fullscreen"), s("Fullscreen")}}, true, false, notInBuild},
        {s("lowLatency"), s("Low latency"), K::Bool, {}, true, false, notInBuild},
        {s("offline"), s("Offline mode"), K::Bool, {}, false, true},
        {s("store"), s("Store behaviour"), K::Choice, {{s("offline-error"), s("Offline error")}, {s("not-available"), s("Not available")}}, false, true},
        {s("crash_dumps"), s("Crash reports"), K::Bool, {}, false, true},
        {s("run_profile"), s("Run profile"), K::RunProfile, {}, false, true},
    };
    return list;
}

const KeySpec* find(const QString& key)
{
    for (const KeySpec& spec : keys()) {
        if (spec.key == key)
            return &spec;
    }
    return nullptr;
}

QString valueLabel(const QString& key, const QVariant& value)
{
    const KeySpec* spec = find(key);
    const QVariant v = spec ? valid(*spec, value) : QVariant();
    if (!v.isValid())
        return QStringLiteral("Not set");
    switch (spec->kind) {
    case KeySpec::Kind::Bool: return v.toBool() ? QStringLiteral("On") : QStringLiteral("Off");
    case KeySpec::Kind::Choice:
        for (const auto& c : spec->choices) {
            if (same(v, c.first))
                return c.second;
        }
        break;
    case KeySpec::Kind::DisplayMode: {
        const QString mode = v.toString();
        if (mode == QLatin1String("auto"))
            return QStringLiteral("Auto");
        if (mode == QLatin1String("host"))
            return QStringLiteral("This display");
        const QStringList parts = mode.split(QLatin1Char('@'));
        QString size = parts.first();
        size.replace(QLatin1Char('x'), QChar(0x00D7));
        return parts.size() > 1 ? QStringLiteral("%1 · %2 Hz").arg(size, parts[1]) : size;
    }
    case KeySpec::Kind::Runner: return Runners::find(v.toString())->name;
    case KeySpec::Kind::RunProfile: {
        const qsizetype n = v.toStringList().size();
        const QString count = n == 1 ? QStringLiteral("1 variable") : QStringLiteral("%1 variables").arg(n);
        return v.toStringList() == AppSettings::defaultRunProfile() ? QStringLiteral("Defaults, %1").arg(count) : count;
    }
    }
    return v.toString();
}

bool looksLikePath(const QString& value)
{
    static const QRegularExpression re(QStringLiteral("^(?:[A-Za-z]:[\\\\/]|[\\\\/]|~[\\\\/])"));
    return re.match(value.trimmed()).hasMatch();
}

Profile fromGame(const AppSettings& settings, const QString& titleId, const QString& title)
{
    Profile p;
    p.titleId = titleId;
    p.title = title;
    p.name = clean(QStringLiteral("%1 settings").arg(title), kNameMax, false);
    const QVariantMap own = settings.gameOverrides(titleId);
    for (const KeySpec& spec : keys()) {
        if (!spec.game || !spec.unavailable.isEmpty())
            continue;
        const QVariant v = valid(spec, isXclKey(spec.key) ? LaunchConfig::typed(spec.key, own.value(spec.key)) : own.value(spec.key));
        if (v.isValid())
            p.values.insert(spec.key, v);
    }
    if (const RunnerSpec* r = Runners::find(p.values.value(QStringLiteral("runner")).toString()))
        p.runnerVersion = r->pinnedVersion();
    return p;
}

Profile fromDefaults(const AppSettings& settings)
{
    Profile p;
    p.name = QStringLiteral("XSEmu defaults");
    for (const KeySpec& spec : keys()) {
        if (!spec.defaults || !spec.unavailable.isEmpty())
            continue;
        QVariant v = valid(spec, currentDefault(settings, spec));
        if (spec.kind == KeySpec::Kind::RunProfile && v.isValid())
            v = withoutPaths(v);
        if (v.isValid())
            p.values.insert(spec.key, v);
    }
    if (const RunnerSpec* r = Runners::find(p.values.value(QStringLiteral("runner")).toString()))
        p.runnerVersion = r->pinnedVersion();
    return p;
}

QByteArray toJson(const Profile& profile, bool compact)
{
    return QJsonDocument(toObject(profile)).toJson(compact ? QJsonDocument::Compact : QJsonDocument::Indented);
}

QString toCode(const Profile& profile)
{
    const QByteArray packed = qCompress(toJson(profile, true), 9);
    return QLatin1String(kCodePrefix) + QString::fromLatin1(packed.toBase64(QByteArray::Base64UrlEncoding | QByteArray::OmitTrailingEquals));
}

Parsed parse(const QByteArray& input)
{
    if (input.size() > kMaxInput)
        return failed(QStringLiteral("This is too large to be an XSEmu settings profile."));
    QByteArray json = input.trimmed();
    if (json.startsWith("\xEF\xBB\xBF"))
        json = json.mid(3).trimmed();
    const bool fromCode = !json.startsWith('{');
    if (fromCode) {
        static const QRegularExpression codeRe(QStringLiteral("XSEMU(\\d+):([A-Za-z0-9_\\-=\\r\\n]*)"));
        const QString text = QString::fromUtf8(input);
        const QRegularExpressionMatch m = codeRe.match(text);
        if (!m.hasMatch())
            return failed(QStringLiteral("This isn't an XSEmu settings code or file. A share code starts with XSEMU1:"));
        if (m.captured(1).toInt() > 1)
            return failed(QStringLiteral("These settings were made with a newer launcher. Update XSEmu to import them."));
        QString payload = m.captured(2);
        payload.remove(QLatin1Char('\r')).remove(QLatin1Char('\n'));
        const qsizetype end = m.capturedEnd(0);
        if (end < text.size() && !text.at(end).isSpace() && !QStringLiteral(".,;!?)\"'").contains(text.at(end)))
            return failed(QStringLiteral("This share code is damaged: it has characters a share code never contains."));
        const QByteArray::FromBase64Result decoded = QByteArray::fromBase64Encoding(
            payload.toLatin1(), QByteArray::Base64UrlEncoding | QByteArray::AbortOnBase64DecodingErrors);
        if (!decoded || payload.size() % 4 == 1)
            return failed(QStringLiteral("This share code is incomplete or damaged. Copy the whole code, from XSEMU1: to its last character."));
        const QByteArray& packed = *decoded;
        const quint32 expected = packed.size() >= 4 ? qFromBigEndian<quint32>(packed.constData()) : 0;
        json = expected > 0 && expected <= kMaxJson && packed.size() > 4 ? qUncompress(packed) : QByteArray();
        if (json.isEmpty())
            return failed(QStringLiteral("This share code is incomplete or damaged. Copy the whole code, from XSEMU1: to its last character."));
    }

    QJsonParseError error;
    const QJsonDocument doc = QJsonDocument::fromJson(json, &error);
    if (error.error != QJsonParseError::NoError || !doc.isObject())
        return failed(fromCode ? QStringLiteral("This share code is damaged: what it holds isn't readable.")
                               : error.error != QJsonParseError::NoError
                               ? QStringLiteral("This file isn't readable: %1 at character %2.").arg(error.errorString()).arg(error.offset)
                               : QStringLiteral("This file isn't an XSEmu settings profile."));
    const QJsonObject o = doc.object();
    if (o.value(QStringLiteral("schema")).toString() != QLatin1String(kSchema))
        return failed(QStringLiteral("This isn't an XSEmu settings profile."));
    const QJsonValue version = o.value(QStringLiteral("version"));
    if (!version.isDouble() || version.toDouble() != std::floor(version.toDouble()) || version.toDouble() < 1)
        return failed(QStringLiteral("This settings profile is malformed: \"version\" must be a whole number from 1."));
    if (version.toDouble() > kVersion)
        return failed(QStringLiteral("These settings were made with a newer launcher. Update XSEmu to import them."));
    const QJsonValue settings = o.value(QStringLiteral("settings"));
    if (!settings.isUndefined() && !settings.isObject())
        return failed(QStringLiteral("This settings profile is malformed: \"settings\" must be an object."));

    Parsed out;
    Profile& p = out.profile;
    static const QStringList known{QStringLiteral("schema"), QStringLiteral("version"), QStringLiteral("name"), QStringLiteral("note"),
                                   QStringLiteral("madeWith"), QStringLiteral("game"), QStringLiteral("settings"), QStringLiteral("runner")};
    for (auto it = o.constBegin(); it != o.constEnd(); ++it) {
        if (!known.contains(it.key()))
            out.ignored << it.key();
    }
    const auto text = [&](const char* key, int max, bool multiLine) {
        const QJsonValue v = o.value(QLatin1String(key));
        if (v.isUndefined())
            return QString();
        if (!v.isString()) {
            out.rejected << Note{QLatin1String(key), shown(v.toVariant()), QStringLiteral("must be text")};
            return QString();
        }
        return clean(v.toString(), max, multiLine);
    };
    p.name = text("name", kNameMax, false);
    p.note = text("note", kNoteMax, true);
    const QJsonObject made = o.value(QStringLiteral("madeWith")).toObject();
    p.launcherVersion = infoString(made, "launcher", 40);
    p.gpuVendor = infoString(made, "gpuVendor", 40);
    p.gpuModel = infoString(made, "gpu");
    const QJsonObject game = o.value(QStringLiteral("game")).toObject();
    p.titleId = infoString(game, "titleId");
    p.title = infoString(game, "title");

    const QJsonObject values = settings.toObject();
    for (auto it = values.constBegin(); it != values.constEnd(); ++it) {
        const KeySpec* spec = find(it.key());
        if (!spec || spec->kind == KeySpec::Kind::Runner) {
            out.ignored << it.key();
            continue;
        }
        QString why;
        const QVariant v = canonical(*spec, it.value().toVariant(), &why);
        if (!v.isValid()) {
            out.rejected << Note{spec->key, shown(it.value().toVariant()), why};
            continue;
        }
        if (spec->kind == KeySpec::Kind::RunProfile) {
            QStringList paths;
            const QStringList kept = withoutPaths(v, &paths);
            for (const QString& key : paths)
                out.rejected << Note{spec->key, key, QStringLiteral("is a path on the PC it came from; left out")};
            p.values.insert(spec->key, kept);
            continue;
        }
        if (const QString reason = forbiddenReason(*spec, v); !reason.isEmpty()) {
            out.forbidden << Note{spec->key, valueLabel(spec->key, v), reason};
            out.forbiddenValues.insert(spec->key, v);
            continue;
        }
        p.values.insert(spec->key, v);
    }

    const QJsonValue runner = o.value(QStringLiteral("runner"));
    if (!runner.isUndefined()) {
        const QJsonValue id = runner.toObject().value(QStringLiteral("id"));
        QString why;
        const QVariant v = runner.isObject() ? canonical(*find(QStringLiteral("runner")), id.toVariant(), &why) : QVariant();
        if (!runner.isObject())
            out.rejected << Note{QStringLiteral("runner"), shown(runner.toVariant()), QStringLiteral("must be an object with the runner's id")};
        else if (!v.isValid())
            out.rejected << Note{QStringLiteral("runner"), shown(id.toVariant()), why};
        else {
            p.values.insert(QStringLiteral("runner"), v);
            p.runnerVersion = clean(runner.toObject().value(QStringLiteral("version")).toString(), 40, false);
        }
    }
    return out;
}

QList<Line> preview(const Parsed& parsed, const AppSettings& settings, Target target, const QString& titleId)
{
    QList<Line> lines;
    for (const KeySpec& spec : keys()) {
        const bool allowed = parsed.profile.values.contains(spec.key);
        if (!allowed && !parsed.forbiddenValues.contains(spec.key))
            continue;
        Line l;
        l.key = spec.key;
        l.label = spec.label;
        l.value = allowed ? parsed.profile.values.value(spec.key) : parsed.forbiddenValues.value(spec.key);
        const QVariant current = valid(spec, target == Target::Game ? currentForGame(settings, titleId, spec) : currentDefault(settings, spec));
        if (spec.kind == KeySpec::Kind::RunProfile)
            l.value = withLocalPaths(l.value, current.toStringList());
        l.from = valueLabel(spec.key, current);
        l.to = valueLabel(spec.key, l.value);
        if (spec.kind == KeySpec::Kind::RunProfile && target == Target::Defaults && !same(current, l.value))
            l.to = QStringLiteral("%1 (%2)").arg(l.to, profileChanges(current, l.value));
        if (!allowed) {
            for (const Note& n : parsed.forbidden) {
                if (n.key == spec.key)
                    l.reason = n.text;
            }
        } else if (target == Target::Game && !spec.game) {
            l.reason = QStringLiteral("Only applies as a launcher default.");
        } else if (target == Target::Defaults && !spec.defaults) {
            l.reason = QStringLiteral("Only applies to a game.");
        } else if (target == Target::Defaults) {
            l.reason = environmentReason(spec.key);
        }
        l.kind = !l.reason.isEmpty() ? Line::Kind::NotApplied : same(current, l.value) ? Line::Kind::Same : Line::Kind::Change;
        lines << l;
    }
    return lines;
}

int apply(const Parsed& parsed, AppSettings& settings, Target target, const QString& titleId)
{
    if (!parsed.ok() || (target == Target::Game && titleId.isEmpty()))
        return 0;
    int changed = 0;
    for (const Line& l : preview(parsed, settings, target, titleId)) {
        if (l.kind != Line::Kind::Change)
            continue;
        if (target == Target::Game)
            settings.setGameOverride(titleId, l.key, l.value);
        else
            writeDefault(settings, l.key, l.value);
        ++changed;
    }
    return changed;
}

}
