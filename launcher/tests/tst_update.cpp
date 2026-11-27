// The update service against a fake release: the version rules, the manifest parse, and a Core update
// end to end (download, SHA-256, the staged program asked --version, the swap, the kept previous
// copy, and the way back). Transport is injected, so no network and no GitHub is involved.

#include "core/AppSettings.h"
#include "core/EmulatorBuild.h"
#include "core/Paths.h"
#include "core/ProcessGate.h"
#include "core/Release.h"
#include "core/UpdateService.h"

#include <QCryptographicHash>
#include <QHash>
#include <QDir>
#include <QElapsedTimer>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QProcess>
#include <QTemporaryDir>
#include <QTest>

#include <functional>

namespace {

quint32 crc32(const QByteArray& data)
{
    static quint32 table[256];
    static bool built = false;
    if (!built) {
        for (quint32 i = 0; i < 256; ++i) {
            quint32 c = i;
            for (int k = 0; k < 8; ++k)
                c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
            table[i] = c;
        }
        built = true;
    }
    quint32 c = 0xFFFFFFFFu;
    for (char b : data)
        c = table[(c ^ quint8(b)) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

void put16(QByteArray& out, quint16 v) { out.append(char(v & 0xFF)); out.append(char(v >> 8)); }
void put32(QByteArray& out, quint32 v)
{
    for (int i = 0; i < 4; ++i)
        out.append(char((v >> (8 * i)) & 0xFF));
}

// A stored (uncompressed) zip, which tar, unzip and Expand-Archive all read, so the test needs no zip
// tool of its own. The entries are marked 0755 so the extracted program can be asked --version.
bool writeZip(const QString& path, const QList<QPair<QString, QByteArray>>& entries)
{
    QByteArray local, central;
    for (const auto& [name, bytes] : entries) {
        const QByteArray n = name.toUtf8();
        const quint32 crc = crc32(bytes);
        const quint32 offset = quint32(local.size());
        put32(local, 0x04034b50);
        put16(local, 20);
        put16(local, 0);
        put16(local, 0); // stored
        put32(local, 0); // time/date
        put32(local, crc);
        put32(local, quint32(bytes.size()));
        put32(local, quint32(bytes.size()));
        put16(local, quint16(n.size()));
        put16(local, 0);
        local.append(n);
        local.append(bytes);

        put32(central, 0x02014b50);
        put16(central, 0x031E); // made by unix
        put16(central, 20);
        put16(central, 0);
        put16(central, 0);
        put32(central, 0);
        put32(central, crc);
        put32(central, quint32(bytes.size()));
        put32(central, quint32(bytes.size()));
        put16(central, quint16(n.size()));
        put16(central, 0);
        put16(central, 0);
        put16(central, 0);
        put16(central, 0);
        put32(central, 0100755u << 16);
        put32(central, offset);
        central.append(n);
    }
    QByteArray eocd;
    put32(eocd, 0x06054b50);
    put16(eocd, 0);
    put16(eocd, 0);
    put16(eocd, quint16(entries.size()));
    put16(eocd, quint16(entries.size()));
    put32(eocd, quint32(central.size()));
    put32(eocd, quint32(local.size()));
    put16(eocd, 0);

    QFile f(path);
    if (!f.open(QIODevice::WriteOnly))
        return false;
    f.write(local);
    f.write(central);
    f.write(eocd);
    return true;
}

QByteArray bytesOf(const QString& path)
{
    QFile f(path);
    return f.open(QIODevice::ReadOnly) ? f.readAll() : QByteArray();
}

QString shaOf(const QByteArray& bytes)
{
    return QString::fromLatin1(QCryptographicHash::hash(bytes, QCryptographicHash::Sha256).toHex());
}

QJsonObject releaseJson(const QString& tag, bool prerelease, const QStringList& assets)
{
    QJsonArray list;
    for (const QString& name : assets) {
        list.append(QJsonObject{{QStringLiteral("name"), name},
                                {QStringLiteral("browser_download_url"), QStringLiteral("https://example.invalid/%1/%2").arg(tag, name)},
                                {QStringLiteral("digest"), QString()}});
    }
    return QJsonObject{{QStringLiteral("tag_name"), tag},
                       {QStringLiteral("prerelease"), prerelease},
                       {QStringLiteral("draft"), false},
                       {QStringLiteral("published_at"), QStringLiteral("2026-11-27T09:00:00Z")},
                       {QStringLiteral("assets"), list}};
}

QJsonObject partJson(const QString& version, const QString& asset, const QString& sha, qint64 size,
                     const QJsonObject& files = {})
{
    QJsonObject o{{QStringLiteral("version"), version},
                  {QStringLiteral("asset"), asset},
                  {QStringLiteral("sha256"), sha},
                  {QStringLiteral("size"), double(size)}};
    if (!files.isEmpty())
        o.insert(QStringLiteral("files"), files);
    return o;
}

QJsonObject manifestJson(const QString& tag, const QJsonObject& launcher, const QJsonObject& core)
{
    return QJsonObject{{QStringLiteral("schema"), 1},
                       {QStringLiteral("tag"), tag},
                       {QStringLiteral("version"), tag.mid(1)},
                       {QStringLiteral("launcher"), launcher},
                       {QStringLiteral("core"), core}};
}

bool waitFor(const std::function<bool()>& done, int timeoutMs = 30000)
{
    QElapsedTimer clock;
    clock.start();
    while (clock.elapsed() < timeoutMs) {
        QCoreApplication::processEvents(QEventLoop::AllEvents, 20);
        if (done())
            return true;
        QTest::qWait(10);
    }
    return done();
}

// One launcher folder: <tmp>/launcher with a core/ holding the program in use and its version file.
struct Fixture {
    QTemporaryDir tmp;
    QString appDir;
    QString coreDir;
    QString appSettingsIni;

    Fixture()
    {
        appDir = tmp.filePath(QStringLiteral("launcher"));
        coreDir = QDir(appDir).filePath(QStringLiteral("core"));
        QDir().mkpath(coreDir);
        appSettingsIni = tmp.filePath(QStringLiteral("settings.ini"));
        installCore(QStringLiteral("0.1.0-pre.1"));
    }

    void installCore(const QString& version, bool withVersionFile = true)
    {
        // fake-xsemu stands in for XSEmu.exe: it answers --version as XSEmu does.
        const QString program = QDir(coreDir).filePath(Paths::emulatorExecutableName());
        QFile::remove(program);
        QFile::copy(QStringLiteral(XSEMU_FAKE_EMULATOR_PATH), program);
        QFile::setPermissions(program, QFile::ReadOwner | QFile::WriteOwner | QFile::ExeOwner | QFile::ReadGroup | QFile::ExeGroup
                                           | QFile::ReadOther | QFile::ExeOther);
        QFile::remove(QDir(coreDir).filePath(QStringLiteral("version.json")));
        if (!withVersionFile)
            return;
        QFile v(QDir(coreDir).filePath(QStringLiteral("version.json")));
        if (v.open(QIODevice::WriteOnly))
            v.write(QJsonDocument(QJsonObject{{QStringLiteral("version"), version}}).toJson());
    }

    QString coreProgram() const { return QDir(coreDir).filePath(Paths::emulatorExecutableName()); }
    QString previousProgram() const { return QDir(appDir).filePath(QStringLiteral("core.old/") + Paths::emulatorExecutableName()); }
};

}

class TestUpdate : public QObject {
    Q_OBJECT

private slots:
    void versions_data()
    {
        QTest::addColumn<QString>("candidate");
        QTest::addColumn<QString>("installed");
        QTest::addColumn<bool>("newer");
        QTest::newRow("patch") << "0.1.1" << "0.1.0" << true;
        QTest::newRow("same") << "0.1.0" << "0.1.0" << false;
        QTest::newRow("older") << "0.0.9" << "0.1.0" << false;
        QTest::newRow("tag prefix") << "v0.2.0" << "0.1.0" << true;
        QTest::newRow("pre after pre") << "0.1.0-pre.2" << "0.1.0-pre.1" << true;
        QTest::newRow("same pre") << "0.1.0-pre.1" << "0.1.0-pre.1" << false;
        QTest::newRow("release beats its pre") << "0.1.1" << "0.1.1-pre.1" << true;
        // A finished release is not replaced by a pre-release of the same version, whatever the
        // installed build reports: the next release has to be tagged above it (v0.1.1-pre.1).
        QTest::newRow("pre of a finished version") << "0.1.0-pre.2" << "0.1.0" << false;
        QTest::newRow("next family pre") << "0.1.1-pre.1" << "0.1.0" << true;
        QTest::newRow("unparseable installed") << "0.1.0" << "" << false;
        QTest::newRow("numbers past ten") << "0.1.10" << "0.1.9" << true;
    }

    void versions()
    {
        QFETCH(QString, candidate);
        QFETCH(QString, installed);
        QFETCH(bool, newer);
        QCOMPARE(newerThan(candidate, installed), newer);
    }

    void manifestParses()
    {
        const QJsonObject release = releaseJson(QStringLiteral("v0.1.1-pre.1"), true,
                                                {QStringLiteral("XSEmuLauncher-windows.zip"), QStringLiteral("XSEmuCore-windows.zip"),
                                                 QStringLiteral("xsemu-manifest.json")});
        const QJsonObject manifest = manifestJson(
            QStringLiteral("v0.1.1-pre.1"),
            partJson(QStringLiteral("0.1.1-pre.1"), QStringLiteral("XSEmuLauncher-windows.zip"), QString(64, QLatin1Char('a')), 10),
            partJson(QStringLiteral("0.1.1-pre.1"), QStringLiteral("XSEmuCore-windows.zip"), QString(64, QLatin1Char('b')), 20,
                     QJsonObject{{QStringLiteral("fake-xsemu"), QString(64, QLatin1Char('c'))}}));
        QString why;
        const ReleaseInfo info = parseRelease(release, manifest, &why);
        QVERIFY2(info.hasParts(), qPrintable(why));
        QCOMPARE(info.version, QStringLiteral("0.1.1-pre.1"));
        QVERIFY(info.prerelease);
        QVERIFY(info.launcher.valid());
        QVERIFY(info.core.valid());
        QCOMPARE(info.core.url, QStringLiteral("https://example.invalid/v0.1.1-pre.1/XSEmuCore-windows.zip"));
        QCOMPARE(info.core.files.size(), 1);

        // An asset the release does not carry is a broken manifest, not an update.
        QJsonObject wrongAsset = manifest;
        wrongAsset.insert(QStringLiteral("core"),
                          partJson(QStringLiteral("0.1.1-pre.1"), QStringLiteral("missing.zip"), QString(64, QLatin1Char('b')), 20));
        QVERIFY(!parseRelease(release, wrongAsset, &why).core.valid());
        QVERIFY(why.contains(QStringLiteral("missing.zip")));

        // A manifest for another tag is not this release's.
        QJsonObject otherTag = manifest;
        otherTag.insert(QStringLiteral("tag"), QStringLiteral("v0.1.0-pre.3"));
        QVERIFY(!parseRelease(release, otherTag, &why).hasParts());
        QVERIFY(why.contains(QStringLiteral("v0.1.0-pre.3")));
    }

    void checkFindsTheUpdateAndCoreUpdate()
    {
        Fixture f;
        Fake fake(f);
        AppSettings settings(f.appSettingsIni);
        settings.setPrereleaseUpdates(true);
        settings.setEmulatorPathOverride(f.coreProgram());
        ProcessGate gate;
        UpdateService updates(settings, gate);
        updates.setDirectories(f.appDir, f.coreDir);
        fake.attach(updates);

        updates.check(true);
        QVERIFY(waitFor([&] { return !updates.status().busy(); }));
        const UpdateService::Status st = updates.status();
        QCOMPARE(st.state, UpdateService::State::Available);
        QVERIFY2(st.launcherUpdate, qPrintable(st.message));
        QVERIFY2(st.coreUpdate, qPrintable(st.message));
        QVERIFY(st.coreBundled);
        QCOMPARE(st.release.core.version, QStringLiteral("0.1.1-pre.1"));
        QVERIFY(!settings.lastUpdateCheck().isNull());
    }

    void checkIgnoresPrereleasesWhenOff()
    {
        Fixture f;
        Fake fake(f);
        AppSettings settings(f.appSettingsIni);
        settings.setPrereleaseUpdates(false);
        settings.setEmulatorPathOverride(f.coreProgram());
        ProcessGate gate;
        UpdateService updates(settings, gate);
        updates.setDirectories(f.appDir, f.coreDir);
        fake.attach(updates);

        updates.check(true);
        QVERIFY(waitFor([&] { return !updates.status().busy(); }));
        QCOMPARE(updates.status().state, UpdateService::State::UpToDate);
        QVERIFY(!updates.status().updateAvailable());
    }

    void coreInstallsAndKeepsTheOldCopy()
    {
        Fixture f;
        Fake fake(f);
        AppSettings settings(f.appSettingsIni);
        settings.setPrereleaseUpdates(true);
        settings.setEmulatorPathOverride(f.coreProgram());
        ProcessGate gate;
        UpdateService updates(settings, gate);
        updates.setDirectories(f.appDir, f.coreDir);
        fake.attach(updates);

        updates.check(true);
        QVERIFY(waitFor([&] { return !updates.status().busy(); }));
        QVERIFY(updates.status().coreUpdate);

        updates.install(UpdateService::Part::Core);
        QCOMPARE(gate.owner(), ProcessGate::Owner::Update);
        QVERIFY(waitFor([&] { return !updates.status().busy(); }));
        QVERIFY(gate.isFree());
        QVERIFY2(updates.status().state != UpdateService::State::Failed, qPrintable(updates.status().error));
        QCOMPARE(updates.installedCoreVersion(), QStringLiteral("0.1.1-pre.1"));
        QCOMPARE(shaOf(bytesOf(f.coreProgram())), fake.newProgramSha);
        QVERIFY2(QFileInfo::exists(f.previousProgram()), "the previous copy stays until the new Core starts");
        QVERIFY2(QFileInfo::exists(QDir(f.coreDir).filePath(QStringLiteral("version.json"))), "the other files in core stay");

        updates.noteEmulatorStarted();
        QVERIFY2(!QFileInfo::exists(QDir(f.appDir).filePath(QStringLiteral("core.old"))), "a started Core retires the previous copy");
        QVERIFY(!updates.status().corePreviousKept);
    }

    void coreRestoresThePreviousCopy()
    {
        Fixture f;
        Fake fake(f);
        AppSettings settings(f.appSettingsIni);
        settings.setPrereleaseUpdates(true);
        settings.setEmulatorPathOverride(f.coreProgram());
        ProcessGate gate;
        UpdateService updates(settings, gate);
        updates.setDirectories(f.appDir, f.coreDir);
        fake.attach(updates);

        const QByteArray oldProgram = bytesOf(f.coreProgram());
        updates.check(true);
        QVERIFY(waitFor([&] { return !updates.status().busy(); }));
        updates.install(UpdateService::Part::Core);
        QVERIFY(waitFor([&] { return !updates.status().busy(); }));
        QVERIFY(updates.status().corePreviousKept);
        QVERIFY(updates.status().installedCore != QStringLiteral("0.1.0-pre.1"));

        QString error;
        QVERIFY2(updates.restorePreviousCore(&error), qPrintable(error));
        QCOMPARE(shaOf(bytesOf(f.coreProgram())), shaOf(oldProgram));
        QCOMPARE(updates.installedCoreVersion(), QStringLiteral("0.1.0-pre.1"));
        QVERIFY(!QFileInfo::exists(QDir(f.appDir).filePath(QStringLiteral("core.old"))));
    }

    void coreRollbackDropsTheVersionFileTheUpdateAdded()
    {
        Fixture f;
        f.installCore(QString(), false); // a Core placed by hand: no version file beside it
        Fake fake(f);
        AppSettings settings(f.appSettingsIni);
        settings.setPrereleaseUpdates(true);
        settings.setEmulatorPathOverride(f.coreProgram());
        ProcessGate gate;
        UpdateService updates(settings, gate);
        updates.setDirectories(f.appDir, f.coreDir);
        fake.attach(updates);

        // Without a version file the Core's own answer stands ("XSEmu 0.0.0-fake"), which a newer
        // release does replace; the update brings the version file the rollback has to take away.
        EmulatorBuild::probe(f.coreProgram());
        updates.refresh();
        const QByteArray handPlaced = bytesOf(f.coreProgram());
        updates.check(true);
        QVERIFY(waitFor([&] { return !updates.status().busy(); }));
        QVERIFY(updates.status().coreUpdate);
        updates.install(UpdateService::Part::Core);
        QVERIFY(waitFor([&] { return !updates.status().busy(); }));
        QCOMPARE(updates.installedCoreVersion(), QStringLiteral("0.1.1-pre.1"));

        QString error;
        QVERIFY2(updates.restorePreviousCore(&error), qPrintable(error));
        QCOMPARE(shaOf(bytesOf(f.coreProgram())), shaOf(handPlaced));
        QVERIFY2(!QFileInfo::exists(QDir(f.coreDir).filePath(QStringLiteral("version.json"))),
                 "the version file the update brought says nothing about the Core that is back");
        QVERIFY(updates.installedCoreVersion().isEmpty());
    }

    void aBadChecksumInstallsNothing()
    {
        Fixture f;
        // The manifest names another file than the asset the release serves.
        Fake fake(f, {}, QString(64, QLatin1Char('0')));
        AppSettings settings(f.appSettingsIni);
        settings.setPrereleaseUpdates(true);
        settings.setEmulatorPathOverride(f.coreProgram());
        ProcessGate gate;
        UpdateService updates(settings, gate);
        updates.setDirectories(f.appDir, f.coreDir);
        fake.attach(updates);

        const QByteArray oldProgram = bytesOf(f.coreProgram());
        updates.check(true);
        QVERIFY(waitFor([&] { return !updates.status().busy(); }));
        updates.install(UpdateService::Part::Core);
        QVERIFY(waitFor([&] { return !updates.status().busy(); }));
        QCOMPARE(updates.status().state, UpdateService::State::Failed);
        QVERIFY(updates.status().error.contains(QStringLiteral("SHA-256")));
        QCOMPARE(shaOf(bytesOf(f.coreProgram())), shaOf(oldProgram));
        QVERIFY(!QFileInfo::exists(QDir(f.appDir).filePath(QStringLiteral("core.old"))));
    }

    void aProgramThatDoesNotAnswerIsNotInstalled()
    {
        Fixture f;
        Fake fake(f, "not a program");
        AppSettings settings(f.appSettingsIni);
        settings.setPrereleaseUpdates(true);
        settings.setEmulatorPathOverride(f.coreProgram());
        ProcessGate gate;
        UpdateService updates(settings, gate);
        updates.setDirectories(f.appDir, f.coreDir);
        fake.attach(updates);

        const QByteArray oldProgram = bytesOf(f.coreProgram());
        updates.check(true);
        QVERIFY(waitFor([&] { return !updates.status().busy(); }));
        updates.install(UpdateService::Part::Core);
        QVERIFY(waitFor([&] { return !updates.status().busy(); }, 60000));
        QCOMPARE(updates.status().state, UpdateService::State::Failed);
        QCOMPARE(shaOf(bytesOf(f.coreProgram())), shaOf(oldProgram));
        QVERIFY(!QFileInfo::exists(QDir(f.appDir).filePath(QStringLiteral("core.old"))));
    }

    void aCustomCoreIsNeverReplaced()
    {
        Fixture f;
        Fake fake(f);
        AppSettings settings(f.appSettingsIni);
        settings.setPrereleaseUpdates(true);
        settings.setEmulatorPathOverride(f.tmp.filePath(QStringLiteral("somewhere/XSEmu.exe")));
        ProcessGate gate;
        UpdateService updates(settings, gate);
        updates.setDirectories(f.appDir, f.coreDir);
        fake.attach(updates);

        updates.check(true);
        QVERIFY(waitFor([&] { return !updates.status().busy(); }));
        QVERIFY(updates.status().launcherUpdate);
        QVERIFY(!updates.status().coreUpdate);
        QVERIFY(!updates.status().coreBundled);
        QString why;
        QVERIFY(!updates.canInstall(UpdateService::Part::Core, &why));
        QVERIFY2(why.contains(QStringLiteral("outside")), qPrintable(why));
    }

    void aRunningGameBlocksInstalling()
    {
        Fixture f;
        Fake fake(f);
        AppSettings settings(f.appSettingsIni);
        settings.setPrereleaseUpdates(true);
        settings.setEmulatorPathOverride(f.coreProgram());
        ProcessGate gate;
        UpdateService updates(settings, gate);
        updates.setDirectories(f.appDir, f.coreDir);
        fake.attach(updates);

        updates.check(true);
        QVERIFY(waitFor([&] { return !updates.status().busy(); }));
        QVERIFY(updates.status().coreUpdate);
        QVERIFY(gate.acquire(ProcessGate::Owner::Game, QStringLiteral("A game"), [] {}));
        QString why;
        QVERIFY(!updates.canInstall(UpdateService::Part::Core, &why));
        QVERIFY2(why.contains(QStringLiteral("Stop it")), qPrintable(why));
        gate.release(ProcessGate::Owner::Game);
        QVERIFY(updates.canInstall(UpdateService::Part::Core));
    }

private:
    // One fake release: v0.1.1-pre.1 with a launcher zip and a Core zip, served from local files.
    class Fake {
    public:
        explicit Fake(Fixture& f, const QByteArray& program = {}, const QString& shaOverride = {})
            : m_dir(f.tmp.filePath(QStringLiteral("release")))
        {
            QDir().mkpath(m_dir);
            m_program = program.isEmpty() ? bytesOf(f.coreProgram()) : program;
            const QByteArray versionFile = QJsonDocument(QJsonObject{{QStringLiteral("version"), QStringLiteral("0.1.1-pre.1")}}).toJson();
            const QString coreZip = QDir(m_dir).filePath(QStringLiteral("XSEmuCore-windows.zip"));
            writeZip(coreZip, {{Paths::emulatorExecutableName(), m_program}, {QStringLiteral("version.json"), versionFile}});
            const QByteArray coreBytes = bytesOf(coreZip);
            newProgramSha = shaOf(m_program);
            coreSha = shaOverride.isEmpty() ? shaOf(coreBytes) : shaOverride;

            const QJsonObject release = releaseJson(QStringLiteral("v0.1.1-pre.1"), true,
                                                    {QStringLiteral("XSEmuCore-windows.zip"), QStringLiteral("xsemu-manifest.json"),
                                                     QStringLiteral("XSEmuLauncher-windows.zip")});
            const QJsonObject manifest = manifestJson(
                QStringLiteral("v0.1.1-pre.1"),
                partJson(QStringLiteral("0.1.1-pre.1"), QStringLiteral("XSEmuLauncher-windows.zip"), QString(64, QLatin1Char('a')), 10),
                partJson(QStringLiteral("0.1.1-pre.1"), QStringLiteral("XSEmuCore-windows.zip"), coreSha, coreBytes.size(),
                         QJsonObject{{Paths::emulatorExecutableName(), shaOf(m_program)},
                                     {QStringLiteral("version.json"), shaOf(versionFile)}}));

            bodies.insert(QString::fromLatin1(UpdateService::kApiReleases),
                          QJsonDocument(QJsonArray{release}).toJson());
            bodies.insert(QStringLiteral("https://example.invalid/v0.1.1-pre.1/xsemu-manifest.json"),
                          QJsonDocument(manifest).toJson());
            downloads.insert(QStringLiteral("https://example.invalid/v0.1.1-pre.1/XSEmuCore-windows.zip"), coreZip);
        }

        void attach(UpdateService& updates)
        {
            updates.setTransport(
                [this](const QUrl& url, QString* error, int) -> QByteArray {
                    if (!bodies.contains(url.toString())) {
                        if (error)
                            *error = QStringLiteral("no fixture for %1").arg(url.toString());
                        return {};
                    }
                    return bodies.value(url.toString());
                },
                [this](const QUrl& url, const QString& file, const std::function<bool(qint64, qint64)>& progress, QString* error) {
                    const QString from = downloads.value(url.toString());
                    if (from.isEmpty() || !QFile::exists(from)) {
                        if (error)
                            *error = QStringLiteral("no download for %1").arg(url.toString());
                        return false;
                    }
                    if (progress && !progress(0, 1))
                        return false;
                    QFile::remove(file);
                    const bool ok = QFile::copy(from, file);
                    if (!ok && error)
                        *error = QStringLiteral("cannot copy the fixture");
                    return ok;
                });
        }

        QString newProgramSha;
        QString coreSha;

    private:
        QString m_dir;
        QByteArray m_program;
        QHash<QString, QByteArray> bodies;
        QHash<QString, QString> downloads;
    };
};

QTEST_GUILESS_MAIN(TestUpdate)
#include "tst_update.moc"
