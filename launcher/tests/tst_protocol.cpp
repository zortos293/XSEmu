#include "Protocol.h"

#include <QFile>
#include <QRegularExpression>
#include <QTest>

using namespace Protocol;

#define QCOMPARE_ENUM(a, b) QCOMPARE(static_cast<int>(a), static_cast<int>(b))

class TestProtocol : public QObject {
    Q_OBJECT
private slots:
    void everyExampleInTheDocParses();
    void hello();
    void stateAndStats();
    void statsLeavesOutWhatIsMissing();
    void exitReasons();
    void warmupLines();
    void invalidAndUnknown();
    void commands();
};

static QList<QByteArray> docExamples()
{
    QFile f(QStringLiteral(XSEMU_PROTOCOL_DOC));
    if (!f.open(QIODevice::ReadOnly))
        return {};
    const QString doc = QString::fromUtf8(f.readAll());
    QList<QByteArray> out;
    static const QRegularExpression re(QStringLiteral("`(\\{\"ev\":[^`]*\\})`"));
    auto it = re.globalMatch(doc);
    while (it.hasNext())
        out << it.next().captured(1).toUtf8();
    return out;
}

void TestProtocol::everyExampleInTheDocParses()
{
    const QList<QByteArray> lines = docExamples();
    QVERIFY2(lines.size() >= 9, qPrintable(QStringLiteral("found %1 examples").arg(lines.size())));
    for (const QByteArray& l : lines) {
        const Event e = parse(l);
        QVERIFY2(!std::holds_alternative<Invalid>(e), l.constData());
        QVERIFY2(!std::holds_alternative<Unknown>(e), l.constData());
    }
}

void TestProtocol::hello()
{
    const Event e = parse(R"j({"ev":"hello","proto":1,"pid":4242,"version":"0.1.0","title":"Some.Title","device_type":"lockhart","hwnd":"1a2b3c"})j");
    const auto* h = std::get_if<Hello>(&e);
    QVERIFY(h);
    QCOMPARE(h->proto, 1);
    QCOMPARE(h->pid, 4242);
    QCOMPARE(h->version, QStringLiteral("0.1.0"));
    QCOMPARE(h->deviceType, QStringLiteral("lockhart"));
    QCOMPARE(h->hwnd, QStringLiteral("1a2b3c"));
    const Event noHwnd = parse(R"j({"ev":"hello","proto":1,"pid":1})j");
    QVERIFY(std::get<Hello>(noHwnd).hwnd.isEmpty());
}

void TestProtocol::stateAndStats()
{
    QCOMPARE_ENUM(std::get<State>(parse(R"j({"ev":"state","state":"booting"})j")).state, GameState::Booting);
    QCOMPARE_ENUM(std::get<State>(parse(R"j({"ev":"state","state":"running"})j")).state, GameState::Running);
    QCOMPARE_ENUM(std::get<State>(parse(R"j({"ev":"state","state":"paused"})j")).state, GameState::Paused);
    QCOMPARE_ENUM(std::get<State>(parse(R"j({"ev":"state","state":"stopping"})j")).state, GameState::Stopping);
    QCOMPARE_ENUM(std::get<State>(parse(R"j({"ev":"state","state":"dancing"})j")).state, GameState::Unknown);
    const Stats s = std::get<Stats>(parse(R"j({"ev":"stats","fps":58.2,"frame_ms":17.2,"ram_free_mb":9830,"vram_used_mb":5222,"vram_budget_mb":8192,"stalls":0,"worst_ms":34.0,"pipelines":41906,"compiling":0})j"));
    QCOMPARE(*s.fps, 58.2);
    QCOMPARE(*s.frameMs, 17.2);
    QCOMPARE(*s.ramFreeMb, 9830);
    QCOMPARE(*s.vramUsedMb, 5222);
    QCOMPARE(*s.vramBudgetMb, 8192);
    QCOMPARE(*s.stalls, 0);
    QCOMPARE(*s.pipelines, 41906);
}

void TestProtocol::statsLeavesOutWhatIsMissing()
{
    const Stats s = std::get<Stats>(parse(R"j({"ev":"stats","fps":30,"extra":"ignored"})j"));
    QVERIFY(s.fps.has_value());
    QVERIFY(!s.frameMs.has_value());
    QVERIFY(!s.vramUsedMb.has_value());
    QVERIFY(!s.ramFreeMb.has_value());
}

void TestProtocol::exitReasons()
{
    const Exit stopped = std::get<Exit>(parse(R"j({"ev":"exit","code":0,"reason":"stopped","message":""})j"));
    QCOMPARE_ENUM(stopped.reason, ExitReason::Stopped);
    QCOMPARE_ENUM(std::get<Exit>(parse(R"j({"ev":"exit","code":0,"reason":"title_exit"})j")).reason, ExitReason::TitleExit);
    const Exit crash = std::get<Exit>(parse(R"j({"ev":"exit","code":-1073741819,"reason":"crash","message":"access violation in game.exe+0x1234"})j"));
    QCOMPARE_ENUM(crash.reason, ExitReason::Crash);
    QCOMPARE(crash.code, -1073741819);
    QCOMPARE(crash.message, QStringLiteral("access violation in game.exe+0x1234"));
    QCOMPARE_ENUM(std::get<Exit>(parse(R"j({"ev":"exit","code":5,"reason":"device_removed","message":"DXGI_ERROR_DEVICE_HUNG"})j")).reason, ExitReason::DeviceRemoved);
    QCOMPARE_ENUM(std::get<Exit>(parse(R"j({"ev":"exit","code":2,"reason":"error","message":"game files missing"})j")).reason, ExitReason::Error);
    QCOMPARE_ENUM(std::get<Exit>(parse(R"j({"ev":"exit","code":2,"reason":"later"})j")).reason, ExitReason::Unknown);
}

void TestProtocol::warmupLines()
{
    const WarmupStart s = std::get<WarmupStart>(parse(R"j({"ev":"warmup_start","total":41906,"full":false,"threads":16})j"));
    QCOMPARE(s.total, 41906);
    QCOMPARE(s.threads, 16);
    const WarmupProgress p = std::get<WarmupProgress>(parse(R"j({"ev":"warmup_progress","done":27292,"total":41906,"keys_per_s":243.0,"eta_s":60,"ram_free_mb":12288,"giants_compiling":2,"in_flight":5,"failed":0,"phase":"bound","paused_for_ram":false})j"));
    QCOMPARE(p.done, 27292);
    QCOMPARE(p.total, 41906);
    QCOMPARE(p.keysPerS, 243.0);
    QCOMPARE(*p.etaS, 60);
    QCOMPARE(*p.ramFreeMb, 12288);
    QCOMPARE(p.giantsCompiling, 2);
    QCOMPARE(p.inFlight, 5);
    QCOMPARE(p.phase, QStringLiteral("bound"));
    QVERIFY(!p.pausedForRam);
    const WarmupProgress noEta = std::get<WarmupProgress>(parse(R"j({"ev":"warmup_progress","done":0,"total":10,"keys_per_s":0,"phase":"giants","paused_for_ram":true})j"));
    QVERIFY(!noEta.etaS.has_value());
    QVERIFY(noEta.pausedForRam);
    const WarmupDone d = std::get<WarmupDone>(parse(R"j({"ev":"warmup_done","built":41890,"failed":16,"left":16,"too_big":0,"took_s":812.4,"complete":true,"cancelled":false,"code":0})j"));
    QCOMPARE(d.built, 41890);
    QCOMPARE(d.failed, 16);
    QCOMPARE(d.left, 16);
    QCOMPARE(d.tookS, 812.4);
    QVERIFY(d.complete);
    QVERIFY(!d.cancelled);
    // an emulator older than the field leaves `left` out: what is left of the pass is then unknown, not zero
    const WarmupDone noLeft = std::get<WarmupDone>(parse(R"j({"ev":"warmup_done","built":400,"failed":0,"too_big":0,"took_s":12.5,"complete":true,"cancelled":false,"code":0})j"));
    QCOMPARE(noLeft.left, qint64(-1));
    QCOMPARE(std::get<Error>(parse(R"j({"ev":"error","message":"another XSEmu process is running (XSEmu.exe 1234)"})j")).message,
             QStringLiteral("another XSEmu process is running (XSEmu.exe 1234)"));
}

void TestProtocol::invalidAndUnknown()
{
    QVERIFY(std::holds_alternative<Invalid>(parse("")));
    QVERIFY(std::holds_alternative<Invalid>(parse("not json")));
    QVERIFY(std::holds_alternative<Invalid>(parse("[1,2]")));
    QVERIFY(std::holds_alternative<Invalid>(parse(R"j({"cmd":"pause"})j")));
    QCOMPARE(std::get<Unknown>(parse(R"j({"ev":"future_thing","x":1})j")).ev, QStringLiteral("future_thing"));
    QVERIFY(std::holds_alternative<Stats>(parse("  {\"ev\":\"stats\"}\r\n")));
}

void TestProtocol::commands()
{
    QCOMPARE(Command::pause(), QByteArray(R"j({"cmd":"pause"})j" "\n"));
    QCOMPARE(Command::resume(), QByteArray(R"j({"cmd":"resume"})j" "\n"));
    QCOMPARE(Command::stop(), QByteArray(R"j({"cmd":"stop"})j" "\n"));
    QCOMPARE(Command::focus(), QByteArray(R"j({"cmd":"focus"})j" "\n"));
    QCOMPARE(Command::cancel(), QByteArray(R"j({"cmd":"cancel"})j" "\n"));
    QCOMPARE(Command::overlay(true), QByteArray(R"j({"cmd":"overlay","on":true})j" "\n"));
    QCOMPARE(Command::overlay(false), QByteArray(R"j({"cmd":"overlay","on":false})j" "\n"));
}

QTEST_GUILESS_MAIN(TestProtocol)
#include "tst_protocol.moc"
