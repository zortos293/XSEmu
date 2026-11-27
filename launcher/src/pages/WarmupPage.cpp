#include "WarmupPage.h"

#include "PageScaffold.h"
#include "app/Services.h"
#include "core/Format.h"
#include "core/LaunchConfig.h"
#include "core/Runner.h"
#include "core/ShaderCacheInfo.h"
#include "settings/SettingsKit.h"
#include "shell/MainWindow.h"
#include "widgets/Badge.h"
#include "widgets/Button.h"
#include "widgets/CoverThumb.h"
#include "widgets/Icon.h"
#include "widgets/PageHeader.h"
#include "widgets/ProgressBlock.h"
#include "widgets/Text.h"

#include <QDesktopServices>
#include <QFileInfo>
#include <QGridLayout>
#include <QHBoxLayout>
#include <QPainter>
#include <QPainterPath>
#include <QUrl>
#include <QVBoxLayout>

namespace {

class Rule : public QWidget {
public:
    Rule() { setFixedHeight(1); }

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.fillRect(rect(), Theme::color(Theme::Role::Border));
    }
};

// The amber banner while a game holds the one XSEmu slot (Warm-up: blocked).
class Banner : public QWidget {
public:
    Banner() { connect(&Theme::instance(), &Theme::changed, this, qOverload<>(&QWidget::update)); }

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        QPainterPath path;
        path.addRoundedRect(QRectF(rect()), 12, 12);
        p.fillPath(path, Theme::toneSoft(Theme::Tone::Warning));
        const QRectF disc(20, (height() - 22) / 2.0, 22, 22);
        p.setPen(Qt::NoPen);
        p.setBrush(Theme::withAlpha(Theme::color(Theme::Role::Warning), 0.14));
        p.drawEllipse(disc);
        Icon::paint(p, disc, QStringLiteral("lock"), Theme::color(Theme::Role::Warning), 12);
    }
};

QString phaseText(const WarmupController::Progress& pr)
{
    const int i = WarmupController::phaseIndex(pr.phase);
    if (i < 0)
        return WarmupController::phaseLabel(pr.phase);
    return QStringLiteral("%1 · phase %2 of %3").arg(WarmupController::phaseLabel(pr.phase)).arg(i + 1).arg(WarmupController::phases().size());
}

QString ramText(qint64 mb)
{
    return mb < 1024 ? QStringLiteral("%1 MB").arg(mb) : Format::gigabytesFromMb(mb);
}

// The emulator's message as a sentence: capital first letter, full stop.
QString sentence(QString s)
{
    s = s.trimmed();
    if (s.isEmpty())
        return s;
    s[0] = s[0].toUpper();
    if (!s.endsWith(QLatin1Char('.')))
        s += QLatin1Char('.');
    return s;
}

// "27,292 / 41,906", or just the count when the pass reported no comparable total.
QString builtOf(qint64 done, qint64 total)
{
    return total > 0 && done <= total ? QStringLiteral("%1 / %2").arg(Format::count(done), Format::count(total)) : Format::count(done);
}

double fractionOf(qint64 done, qint64 total)
{
    return total > 0 ? qMin(1.0, double(done) / double(total)) : 0.0;
}

struct RunResult {
    QString text;
    Theme::Tone tone;
};

Theme::Role toneText(Theme::Tone tone)
{
    switch (tone) {
    case Theme::Tone::Neutral: return Theme::Role::Text2;
    case Theme::Tone::Accent: return Theme::Role::AccentText;
    case Theme::Tone::Success: return Theme::Role::Success;
    case Theme::Tone::Warning: return Theme::Role::Warning;
    case Theme::Tone::Danger: return Theme::Role::Danger;
    }
    return Theme::Role::Text2;
}

RunResult runResult(const WarmupRun& run)
{
    if (const RunnerSpec* spec = run.runnerId.isEmpty() ? nullptr : Runners::find(run.runnerId)) {
        WarmupRun plain = run;
        plain.runnerId.clear();
        RunResult r = runResult(plain);
        r.text += QStringLiteral(" \u00B7 ") + spec->name;
        return r;
    }
    switch (run.result) {
    case WarmupRun::Result::Finished: return {QStringLiteral("Finished"), Theme::Tone::Success};
    case WarmupRun::Result::Cancelled: return {QStringLiteral("Cancelled"), Theme::Tone::Neutral};
    case WarmupRun::Result::StoppedPartWay:
        return {run.stoppedForRam ? QStringLiteral("Stopped · free RAM below 1 GB") : QStringLiteral("Stopped part way"), Theme::Tone::Warning};
    case WarmupRun::Result::Failed: return {QStringLiteral("Failed · exit code %1").arg(Format::exitCode(run.exitCode)), Theme::Tone::Danger};
    case WarmupRun::Result::NothingToBuild: return {QStringLiteral("Nothing to build yet"), Theme::Tone::Warning};
    }
    return {};
}

QString stateSentence(const QString& name, TitleShaderState st)
{
    switch (st) {
    case TitleShaderState::Ready: return QStringLiteral("%1 is ready").arg(name);
    case TitleShaderState::NeverWarmed: return QStringLiteral("%1 has never been warmed up").arg(name);
    case TitleShaderState::LastRunIncomplete: return QStringLiteral("%1's last warm-up stopped part way").arg(name);
    case TitleShaderState::LastRunFailed: return QStringLiteral("%1's last warm-up failed").arg(name);
    case TitleShaderState::ProfileChanged: return QStringLiteral("%1's run profile changed since its last warm-up").arg(name);
    case TitleShaderState::Unknown: break;
    }
    return {};
}

QString joinSentences(const QStringList& parts)
{
    if (parts.isEmpty())
        return {};
    if (parts.size() == 1)
        return parts.first() + QLatin1Char('.');
    return QStringList(parts.mid(0, parts.size() - 1)).join(QStringLiteral(", ")) + QStringLiteral(", and ") + parts.last() + QLatin1Char('.');
}

}

WarmupPage::WarmupPage(Services& services, QWidget* parent)
    : QWidget(parent)
    , m_s(services)
    , m_game(SettingsKit::dropdown({}, {}, 200))
    , m_start(new Button(QStringLiteral("Start warm-up"), Button::Variant::Primary, Button::Size::Large, QStringLiteral("bolt")))
    , m_banner(new Banner)
    , m_bannerTitle(new Label({}, Label::Style::Title))
    , m_bannerText(new Label({}, Label::Style::Small, Theme::Role::Text2))
    , m_stopGame(new Button(QStringLiteral("Stop game"), Button::Variant::DangerSecondary, Button::Size::Small, QStringLiteral("stop")))
    , m_block(new ProgressBlock)
    , m_play(new Button({}, Button::Variant::Primary, Button::Size::Small, QStringLiteral("play")))
    , m_warm(new Button({}, Button::Variant::Primary, Button::Size::Small, QStringLiteral("bolt")))
    , m_resume(new Button({}, Button::Variant::Secondary, Button::Size::Small, QStringLiteral("bolt")))
    , m_log(new Button(QStringLiteral("Open log"), Button::Variant::Secondary, Button::Size::Small, QStringLiteral("file")))
    , m_showLog(new Button(QStringLiteral("Show log"), Button::Variant::Link, Button::Size::Tiny))
    , m_dismiss(new Button(QStringLiteral("Dismiss"), Button::Variant::Ghost, Button::Size::Small))
    , m_cancel(new Button(QStringLiteral("Cancel"), Button::Variant::Ghost, Button::Size::Small))
    , m_runs(nullptr)
{
    const PageScaffold page = makePageScaffold(this, QStringLiteral("Shader warm-up"));
    page.header->setSubtitle(QStringLiteral("Pre-builds every pipeline with the game's runner so the first run doesn't stall"));
    m_game->setFixedHeight(Metrics::PageAction);
    page.header->addAction(m_game);
    page.header->addAction(m_start);

    auto* bh = new QHBoxLayout(m_banner);
    bh->setContentsMargins(58, 16, 20, 16);
    bh->setSpacing(16);
    auto* bt = new QVBoxLayout;
    bt->setSpacing(4);
    bt->addWidget(m_bannerTitle);
    m_bannerText->setWordWrap(true);
    bt->addWidget(m_bannerText);
    bh->addLayout(bt, 1);
    bh->addWidget(m_stopGame, 0, Qt::AlignVCenter);
    page.content->addWidget(m_banner);

    m_block->setActionText({});
    for (Button* b : {m_warm, m_play, m_resume, m_log, m_dismiss, m_cancel})
        m_block->addHeaderWidget(b);
    m_block->addNoteWidget(m_showLog);
    page.content->addWidget(m_block);

    auto* runsHead = makeSectionLabel(QStringLiteral("Recent runs"), new Label(QStringLiteral("The last 20, kept by the launcher"), Label::Style::Caption, Theme::Role::Text3));
    page.content->addSpacing(4);
    page.content->addWidget(runsHead);
    auto* runsHost = new QWidget;
    m_runs = new QGridLayout(runsHost);
    m_runs->setContentsMargins(0, 0, 0, 0);
    m_runs->setHorizontalSpacing(24);
    m_runs->setVerticalSpacing(0);
    page.content->addWidget(runsHost);
    page.content->addStretch(1);

    connect(m_start, &Button::clicked, this, [this] { start(false); });
    connect(m_warm, &Button::clicked, this, [this] { start(m_fullPass); });
    connect(m_resume, &Button::clicked, this, [this] { start(m_continueFull); });
    connect(m_play, &Button::clicked, this, [this] {
        if (auto* w = qobject_cast<MainWindow*>(window()))
            w->play(m_s.warmup.game());
    });
    for (Button* b : {m_log, m_showLog})
        connect(b, &Button::clicked, this, [] { QDesktopServices::openUrl(QUrl::fromLocalFile(WarmupController::logPath())); });
    connect(m_dismiss, &Button::clicked, this, [this] {
        m_dismissed = true;
        refresh();
    });
    connect(m_cancel, &Button::clicked, &m_s.warmup, &WarmupController::cancel);
    connect(m_stopGame, &Button::clicked, &m_s.emulator, &EmulatorController::stop);
    connect(m_game, &QComboBox::activated, this, [this] {
        if (!m_s.warmup.isActive())
            m_dismissed = true;
        refresh();
    });
    connect(&m_s.warmup, &WarmupController::stateChanged, this, [this](WarmupController::State st) {
        if (st == WarmupController::State::Running)
            m_dismissed = false;
        refresh();
        refreshRuns();
    });
    connect(&m_s.warmup, &WarmupController::progressChanged, this, [this] {
        refresh();
        refreshRuns();
    });
    connect(&m_s.gate, &ProcessGate::changed, this, &WarmupPage::refresh);
    connect(&m_s.runners, &RunnerManager::stateChanged, this, &WarmupPage::refresh);
    connect(&m_s.shaderCache, &ShaderCacheMonitor::changed, this, &WarmupPage::refresh);
    connect(&m_s.emulator, &EmulatorController::statsChanged, this, &WarmupPage::refresh);
    connect(&m_s.library, &Library::gamesChanged, this, [this] {
        refreshGames();
        refreshRuns();
    });
    connect(&m_s.covers, &CoverArtService::artChanged, this, &WarmupPage::refreshRuns);
    connect(&m_s.covers, &CoverArtService::artChanged, this, &WarmupPage::refreshGames);
    connect(&m_s.settings, &AppSettings::changed, this, [this](const QString& key) {
        if (key == QLatin1String(AppSettings::Key::RecentWarmups)) {
            refreshRuns();
            refresh();
        } else if (key == QLatin1String(AppSettings::Key::RunProfile) || key == QLatin1String(AppSettings::Key::GameOverrides)) {
            refresh();
        }
    });
    refreshGames();
    refresh();
    refreshRuns();
}

void WarmupPage::start(bool full)
{
    m_startError.clear();
    if (!m_s.warmup.start(full, m_s.library.game(m_game->currentData().toString()), &m_startError))
        refresh();
}

void WarmupPage::refreshGames()
{
    const QString current = m_game->currentData().toString();
    m_game->clear();
    m_game->setIconSize(QSize(14, 20));
    const qreal dpr = devicePixelRatioF();
    for (const Game& g : m_s.library.games()) {
        QPixmap cover(QSize(14, 20) * dpr);
        cover.setDevicePixelRatio(dpr);
        cover.fill(Qt::transparent);
        QPainter p(&cover);
        p.setRenderHint(QPainter::Antialiasing);
        CoverThumb::paint(p, QRectF(0, 0, 14, 20), m_s.art(g.titleId(), CoverCache::Kind::Grid), g.title(), 2);
        p.end();
        m_game->addItem(QIcon(cover), g.title(), g.titleId());
    }
    const int i = m_game->findData(current);
    if (i >= 0)
        m_game->setCurrentIndex(i);
    m_game->setVisible(m_game->count() > 0);
    refresh();
}

void WarmupPage::refresh()
{
    using State = WarmupController::State;
    const State st = m_s.warmup.state();
    const WarmupController::Progress pr = m_s.warmup.progress();
    const bool active = m_s.warmup.isActive();
    const bool gameHolds = m_s.gate.owner() == ProcessGate::Owner::Game;
    if (active || (st != State::Idle && !m_dismissed)) {
        const int i = m_game->findData(m_s.warmup.game().titleId());
        if (i >= 0)
            m_game->setCurrentIndex(i);
    }
    m_game->setEnabled(!active);
    const Game picked = m_s.library.game(m_game->currentData().toString());
    const QString pickedName = picked.isValid() ? picked.title() : QString();
    const QString cannot = active ? QString() : m_s.warmup.checkStart(picked);
    const bool showResult = !active && st != State::Idle && !m_dismissed;
    // the runner this block is about: the running or last pre-cache's, else the picked game's
    const QString runnerId = active ? pr.runnerId : showResult ? m_s.warmup.lastRun().value_or(WarmupRun{}).runnerId : m_s.warmup.runnerFor(picked).id;
    const RunnerSpec* runner = Runners::find(runnerId);
    if (runner && runner->isBuiltIn())
        runner = nullptr;

    // XSEmu's own warm-up turned off: one line instead of its buttons, banner and states
    if (!runner && !active && !m_s.settings.xsemuWarmup()) {
        m_start->hide();
        m_banner->hide();
        for (Button* b : {m_warm, m_play, m_resume, m_log, m_showLog, m_dismiss, m_cancel})
            b->hide();
        m_block->setVisible(true);
        m_block->setState(ProgressBlock::State::Idle);
        m_block->setFraction(-1);
        m_block->setColumns({});
        m_block->setTitle(cannot);
        m_block->setSubtitle({});
        m_block->setNote({});
        return;
    }
    m_start->show();
    m_start->setEnabled(cannot.isEmpty() && !active);
    m_start->setText(active ? (pr.runnerId.isEmpty() ? QStringLiteral("Warming up...") : QStringLiteral("Pre-caching..."))
                            : (runner ? QStringLiteral("Start pre-cache") : QStringLiteral("Start warm-up")));
    m_start->setIconName(gameHolds ? QStringLiteral("lock") : QStringLiteral("bolt"));
    m_start->setToolTip(cannot);

    // the banner: a game holds the one XSEmu slot, or something else keeps the warm-up from starting
    const QString blocker = gameHolds ? QString() : (!m_startError.isEmpty() ? m_startError : cannot);
    const bool bannerShown = gameHolds || (!blocker.isEmpty() && !active);
    m_banner->setVisible(bannerShown);
    m_stopGame->setVisible(gameHolds);
    if (gameHolds) {
        const Game playing = m_s.emulator.game();
        const bool paused = m_s.emulator.state() == EmulatorController::State::Paused;
        const qint64 held = m_s.emulator.stats().heldBytes;
        m_bannerTitle->setText(QStringLiteral("Warm-up waits for %1").arg(playing.title()));
        const QString holds = held > 0 ? QStringLiteral("The %1 game still holds %2, and XSEmu Core runs one process at a time.").arg(paused ? QStringLiteral("paused") : QStringLiteral("running"), Format::gigabytes(held))
                                       : QStringLiteral("XSEmu Core runs one process at a time.");
        m_bannerText->setText(holds + (pickedName.isEmpty() ? QStringLiteral(" Stop the game to warm up.") : QStringLiteral(" Stop the game to warm up %1.").arg(pickedName)));
    } else if (bannerShown) {
        m_bannerTitle->setText(QStringLiteral("Warm-up can't start"));
        m_bannerText->setText(blocker);
    }

    for (Button* b : {m_warm, m_play, m_resume, m_log, m_showLog, m_dismiss, m_cancel})
        b->hide();
    m_block->setNote({});
    const bool blockShown = !gameHolds && (active || showResult || picked.isValid() || m_game->count() == 0);
    m_block->setVisible(blockShown);
    if (!blockShown)
        return;
    if (runner) {
        refreshRunner(*runner, picked, showResult, cannot);
        return;
    }

    const QString game = m_s.warmup.game().title();
    const auto withGame = [&game](const QString& s) { return game.isEmpty() ? s : game + QStringLiteral(" · ") + s; };
    QList<ProgressBlock::Column> progressColumns{
        {QStringLiteral("Done"), QStringLiteral("%1 / %2").arg(Format::count(pr.done), Format::count(pr.total))},
        {QStringLiteral("Speed"), QStringLiteral("%1 keys/s").arg(Format::count(qRound64(pr.keysPerS)))},
    };
    if (st == State::PausedForRam)
        progressColumns.append({QStringLiteral("Time left"), QStringLiteral("Paused")});
    else if (pr.etaS)
        progressColumns.append({QStringLiteral("Time left"), Format::duration(*pr.etaS)});
    progressColumns.append({QStringLiteral("Free RAM"), pr.ramFreeMb ? ramText(*pr.ramFreeMb) : QStringLiteral("Unknown")});
    progressColumns.append({QStringLiteral("Giant compiles"), QString::number(pr.giantsCompiling)});

    if (st == State::Running) {
        m_block->setState(ProgressBlock::State::Running);
        m_block->setTitle(QStringLiteral("Warming up shaders"));
        m_block->setSubtitle(withGame(phaseText(pr)));
        m_block->setFraction(pr.phaseFraction());
        m_block->setColumns(progressColumns);
        m_cancel->setText(m_s.warmup.cancelRequested() ? QStringLiteral("Cancelling...") : QStringLiteral("Cancel"));
        m_cancel->setEnabled(!m_s.warmup.cancelRequested());
        m_cancel->show();
        if (m_s.warmup.cancelRequested())
            m_block->setNote(QStringLiteral("No new compiles start. The ones in flight finish first, which can take a few minutes for giant shaders."));
        return;
    }
    if (st == State::PausedForRam) {
        m_block->setState(ProgressBlock::State::Paused);
        m_block->setTitle(QStringLiteral("Paused at the RAM floor"));
        m_block->setSubtitle(withGame(phaseText(pr)));
        m_block->setFraction(pr.phaseFraction());
        m_block->setColumns(progressColumns);
        m_block->setColumnTone(int(progressColumns.size()) - 2, Theme::Role::Warning);
        const QString ram = pr.ramFreeMb ? QStringLiteral("Free RAM fell to %1, below the 1 GB floor. ").arg(ramText(*pr.ramFreeMb)) : QStringLiteral("Free RAM is below the 1 GB floor. ");
        m_block->setNote(ram + QStringLiteral("Compiles wait and continue on their own when memory frees; closing other apps helps. Finished pipelines are kept."));
        m_cancel->setText(m_s.warmup.cancelRequested() ? QStringLiteral("Cancelling...") : QStringLiteral("Cancel"));
        m_cancel->setEnabled(!m_s.warmup.cancelRequested());
        m_cancel->show();
        return;
    }

    if (showResult) {
        WarmupRun fromProgress;
        fromProgress.done = pr.done;
        fromProgress.total = pr.total;
        fromProgress.failed = pr.failed;
        fromProgress.tookS = pr.elapsedS;
        const WarmupRun run = m_s.warmup.lastRun().value_or(fromProgress);
        const Game runGame = m_s.warmup.game();
        // Continue applies while the launcher knows what the pass left: not to a refusal that never
        // started one, not to a pass that finished, and not across a run profile change
        const auto resume = WarmupController::resumable(m_s.settings.recentWarmups(), m_s.settings.warmupInFlight(),
                                                        runGame.titleId(), QString(),
                                                        LaunchConfig::profileIdentity(m_s.settings, runGame.titleId()));
        const QString leftCount = run.left > 0 ? Format::count(run.left) : QString();
        const QString atOf = run.total > 0 && run.done <= run.total ? QStringLiteral("at %1 of %2").arg(Format::count(run.done), Format::count(run.total)) : game;
        // A failed build is not built: `done` counts what the pass settled with, failures included, so
        // take them out and built + left is the whole pass. Without a done line there is no honest
        // whole-pass count, so the phase position says where it stopped instead.
        const qint64 built = qMax<qint64>(0, run.done - run.failed);
        const QString sofar = run.left > 0 ? QStringLiteral("%1 built · %2 left").arg(Format::count(built), leftCount)
                            : run.left == 0 ? QStringLiteral("%1 built · nothing left").arg(Format::count(built))
                            : pr.total > 0  ? QStringLiteral("stopped in %1 at %2 of %3").arg(WarmupController::phaseLabel(pr.phase).toLower(), Format::count(pr.done), Format::count(pr.total))
                                            : atOf;
        const double fraction = run.left > 0  ? fractionOf(built, built + run.left)
                              : run.left == 0 ? 1.0
                              : pr.total > 0  ? pr.phaseFraction()
                              : run.total > 0 && run.done <= run.total ? fractionOf(run.done, run.total)
                              : -1;
        const bool canPlay = runGame.isValid() && m_s.gate.isFree();
        // what an unfinished pass leaves behind: go on with the leftovers, play anyway, or build everything
        const auto unfinished = [&] {
            m_continueFull = resume ? resume->full : run.full;
            m_resume->setVariant(resume ? Button::Variant::Primary : Button::Variant::Secondary);
            m_resume->setText(resume ? QStringLiteral("Continue") : QStringLiteral("Retry"));
            m_resume->setToolTip(!resume ? QString()
                                 : !leftCount.isEmpty() ? QStringLiteral("Builds the %1 pipelines this pass left. Everything it built is kept.").arg(leftCount)
                                                        : QStringLiteral("Goes on from what this pass built. Everything it built is kept."));
            m_resume->show();
            m_fullPass = !run.full && st == State::StoppedPartWay;
            m_warm->setVisible(m_fullPass);
            m_warm->setText(QStringLiteral("Full pass"));
            m_warm->setToolTip(QStringLiteral("Every pipeline with no time limit, including the ones a budgeted pass leaves."));
            m_warm->setEnabled(cannot.isEmpty());
            if (canPlay) {
                m_play->setVariant(Button::Variant::Secondary);
                m_play->setText(runGame.title().size() > 24 ? QStringLiteral("Play") : QStringLiteral("Play %1").arg(runGame.title()));
                m_play->show();
            }
            // the log as a note-row link, so the header keeps room for Continue and Play
            if (QFileInfo::exists(WarmupController::logPath()))
                m_showLog->show();
            m_dismiss->show();
        };
        // what the pass kept, and what Continue would build from it
        const QString keeps = QStringLiteral("XSEmu Core keeps every pipeline it built. ");
        const QString then = resume ? (leftCount.isEmpty() ? QStringLiteral("Continue goes on from where this pass stopped.")
                                                           : QStringLiteral("Continue builds only the %1 that are left.").arg(leftCount))
                                    : (leftCount.isEmpty() ? QStringLiteral("Nothing is built twice.")
                                                           : QStringLiteral("The %1 that are left build again when the game needs them.").arg(leftCount));
        switch (st) {
        case State::Finished: {
            m_block->setState(ProgressBlock::State::Finished);
            m_block->setTitle(QStringLiteral("Warm-up finished"));
            m_block->setSubtitle(QStringLiteral("%1 pipelines in %2").arg(Format::count(run.done), Format::duration(qint64(run.tookS + 0.5))));
            m_block->setFraction(1.0);
            const double avg = run.tookS > 0 ? run.done / run.tookS : 0;
            m_block->setColumns({
                {QStringLiteral("Done"), builtOf(run.done, run.total)},
                {QStringLiteral("Speed"), QStringLiteral("%1 keys/s avg").arg(Format::count(qRound64(avg)))},
                {QStringLiteral("Time left"), QStringLiteral("Done")},
                {QStringLiteral("Free RAM"), pr.ramFreeMb ? Format::gigabytesFromMb(*pr.ramFreeMb) : QStringLiteral("Unknown")},
            });
            if (run.failed > 0)
                m_block->setNote(QStringLiteral("%1 pipelines failed to build; the game compiles them when it needs them. The log names them.").arg(Format::count(run.failed))
                                 + (resume ? QStringLiteral(" Continue tries them again.") : QString()));
            m_fullPass = false;
            m_continueFull = resume && resume->full;
            if (resume) {
                m_resume->setVariant(Button::Variant::Secondary);
                m_resume->setText(QStringLiteral("Continue"));
                m_resume->setToolTip(QStringLiteral("Tries the %1 pipelines this pass could not build again. Everything it built is kept.").arg(leftCount));
                m_resume->show();
            }
            if (canPlay) {
                m_play->setVariant(Button::Variant::Primary);
                m_play->setText(runGame.title().size() > 24 ? QStringLiteral("Play") : QStringLiteral("Play %1").arg(runGame.title()));
                m_play->show();
            }
            if (run.failed > 0)
                m_log->show();
            m_dismiss->show();
            return;
        }
        case State::Cancelled: {
            m_block->setState(ProgressBlock::State::Cancelled);
            m_block->setTitle(QStringLiteral("Warm-up cancelled"));
            m_block->setSubtitle(sofar);
            m_block->setFraction(fraction);
            m_block->setColumns({});
            m_block->setNote(keeps + then);
            unfinished();
            return;
        }
        case State::StoppedPartWay: {
            m_block->setState(ProgressBlock::State::Paused);
            m_block->setTitle(QStringLiteral("Warm-up stopped part way"));
            m_block->setSubtitle(sofar);
            m_block->setFraction(fraction);
            m_block->setColumns({});
            m_block->setNote((run.stoppedForRam ? QStringLiteral("Free RAM stayed below the 1 GB floor; close other apps, then continue. ")
                                                : QStringLiteral("The 20-minute budget ran out, and a full pass has no limit. "))
                             + keeps + then);
            unfinished();
            return;
        }
        case State::Failed: {
            m_block->setState(ProgressBlock::State::Failed);
            m_block->setTitle(QStringLiteral("Warm-up failed"));
            m_block->setSubtitle(sofar);
            m_block->setFraction(fraction);
            m_block->setColumns({});
            const QString why = m_s.warmup.errorMessage().isEmpty() ? QString() : QStringLiteral(" ") + sentence(m_s.warmup.errorMessage());
            const QString code = QStringLiteral("XSEmu Core exited with code %1.").arg(Format::exitCode(run.exitCode));
            // a refusal before the pass started built nothing, so only the code and the reason are worth saying
            m_block->setNote(built > 0 ? code + QLatin1Char(' ') + keeps + then + why : code + why);
            unfinished();
            return;
        }
        default:
            break;
        }
    }

    // idle: what a warm-up would do for the picked game
    const ShaderCacheInfo& cache = m_s.shaderCache.snapshot();
    const QList<WarmupRun> recent = m_s.settings.recentWarmups();
    m_block->setState(ProgressBlock::State::Idle);
    m_block->setFraction(0);
    m_block->setColumns({});
    const TitleShaderState pickedState = picked.isValid() ? shownShaderState(m_s.settings, picked.titleId(), recent, cache) : TitleShaderState::Unknown;
    switch (pickedState) {
    case TitleShaderState::Ready:
        m_block->setTitle(QStringLiteral("%1 is warmed up").arg(pickedName));
        break;
    case TitleShaderState::LastRunIncomplete:
        m_block->setTitle(QStringLiteral("%1's last warm-up stopped part way").arg(pickedName));
        break;
    case TitleShaderState::LastRunFailed:
        m_block->setTitle(QStringLiteral("%1's last warm-up failed").arg(pickedName));
        break;
    case TitleShaderState::ProfileChanged:
        m_block->setTitle(QStringLiteral("%1's run profile changed since its last warm-up").arg(pickedName));
        break;
    case TitleShaderState::NeverWarmed:
    case TitleShaderState::Unknown:
        m_block->setTitle(pickedName.isEmpty() ? QStringLiteral("Warm up shaders") : QStringLiteral("%1 needs a warm-up").arg(pickedName));
        break;
    }
    QString lastRunText;
    for (const WarmupRun& r : recent) {
        if (picked.isValid() && r.titleId == picked.titleId()) {
            lastRunText = QStringLiteral("last run %1").arg(Format::ago(r.finishedAt).toLower());
            break;
        }
    }
    // a pass that stopped without finishing (or one the launcher did not see the end of) is what this
    // button runs again, and the leftovers are what it says it has
    const auto resume = WarmupController::resumable(recent, m_s.settings.warmupInFlight(), picked.titleId(), QString(),
                                                    LaunchConfig::profileIdentity(m_s.settings, picked.titleId()));
    if (resume) {
        lastRunText = resume->built >= 0 && resume->left > 0 ? QStringLiteral("%1 built · %2 left").arg(Format::count(resume->built), Format::count(resume->left))
                     : resume->built >= 0 && resume->left == 0 ? QStringLiteral("%1 built · nothing left").arg(Format::count(resume->built))
                     : resume->left > 0                       ? QStringLiteral("%1 left").arg(Format::count(resume->left))
                     : !resume->phase.isEmpty()               ? QStringLiteral("a pass stopped in %1").arg(WarmupController::phaseLabel(resume->phase).toLower())
                                                              : QStringLiteral("a pass was interrupted");
    }
    m_block->setSubtitle(pickedState == TitleShaderState::NeverWarmed ? QStringLiteral("never warmed up") : lastRunText);

    double built = 0, took = 0;
    for (const WarmupRun& r : recent) {
        if (r.result == WarmupRun::Result::Finished && r.tookS > 0) {
            built += r.done;
            took += r.tookS;
        }
    }
    QString note;
    if (cache.pipelines > 0) {
        note = QStringLiteral("%1 pipelines recorded").arg(Format::count(cache.pipelines));
        if (built > 0 && took > 0)
            note += QStringLiteral(", about %1 on this PC").arg(Format::duration(qint64(cache.pipelines / (built / took) + 0.5)));
        note += QStringLiteral(". ");
    } else {
        note = QStringLiteral("XSEmu Core records each pipeline a game builds; play a game once and a warm-up builds them all ahead of time. ");
    }
    QStringList others;
    for (const Game& g : m_s.library.games()) {
        if (g.titleId() == picked.titleId() || others.size() >= 3)
            continue;
        const QString s = stateSentence(g.title(), shownShaderState(m_s.settings, g.titleId(), recent, cache));
        if (!s.isEmpty())
            others << s;
    }
    m_block->setNote((note + joinSentences(others)).trimmed());
    const bool shortName = pickedName.isEmpty() || pickedName.size() > 24;
    m_warm->setText(resume ? (shortName ? QStringLiteral("Continue") : QStringLiteral("Continue %1").arg(pickedName))
                           : (shortName ? QStringLiteral("Warm up") : QStringLiteral("Warm up %1").arg(pickedName)));
    m_fullPass = resume && resume->full;
    m_warm->setEnabled(cannot.isEmpty());
    m_warm->setToolTip(!cannot.isEmpty() ? cannot
                       : resume        ? QStringLiteral("Builds only what the last pass left; everything it built is kept.")
                                       : QString());
    m_warm->show();
}

void WarmupPage::refreshRunner(const RunnerSpec& spec, const Game& picked, bool showResult, const QString& cannot)
{
    using State = WarmupController::State;
    const State st = m_s.warmup.state();
    const WarmupController::Progress pr = m_s.warmup.progress();
    const QString version = QStringLiteral("%1 %2").arg(spec.name, spec.pinnedVersion()).trimmed();
    const auto joined = [](const QStringList& parts) {
        QStringList out;
        for (const QString& p : parts) {
            if (!p.isEmpty())
                out << p;
        }
        return out.join(QStringLiteral(" · "));
    };
    const QString freeRam = pr.ramFreeMb ? ramText(*pr.ramFreeMb) : QStringLiteral("Unknown");

    if (m_s.warmup.isActive()) {
        m_block->setState(ProgressBlock::State::Running);
        m_block->setTitle(QStringLiteral("Pre-caching shaders with %1").arg(spec.name));
        const QString step = pr.stepCount > 1 ? QStringLiteral("%1, step %2 of %3").arg(pr.phase).arg(pr.step).arg(pr.stepCount) : pr.phase;
        m_block->setSubtitle(joined({m_s.warmup.game().title(), version, step}));
        m_block->setFraction(pr.total > 0 ? fractionOf(pr.done, pr.total) : -1);
        m_block->setColumns({
            {QStringLiteral("Done"), pr.total > 0      ? QStringLiteral("%1 / %2").arg(Format::count(pr.done), Format::count(pr.total))
                                     : pr.estimate > 0 ? QStringLiteral("%1 / about %2").arg(Format::count(pr.done), Format::count(pr.estimate))
                                                       : Format::count(pr.done)},
            {QStringLiteral("Speed"), pr.keysPerS > 0 ? QStringLiteral("%1/s").arg(Format::count(qRound64(pr.keysPerS))) : QStringLiteral("Measuring")},
            {QStringLiteral("Time left"), pr.etaS ? Format::duration(*pr.etaS) : QStringLiteral("Unknown")},
            {QStringLiteral("Free RAM"), freeRam},
        });
        const bool gentle = pr.step > 0 && pr.step <= spec.precache.size() && spec.precache.at(pr.step - 1).cancel == RunnerSpec::PrecacheCancel::CloseStdin;
        m_block->setNote(!m_s.warmup.cancelRequested() ? QStringLiteral("%1 runs in the background with no console window. XSEmu Core reads its progress; its output goes to the logs folder.").arg(spec.name)
                         : gentle                      ? QStringLiteral("%1 finishes the pipelines in flight, then stops.").arg(spec.name)
                                                       : QStringLiteral("Stopping %1.").arg(spec.name));
        m_cancel->setText(m_s.warmup.cancelRequested() ? QStringLiteral("Cancelling...") : QStringLiteral("Cancel"));
        m_cancel->setEnabled(!m_s.warmup.cancelRequested());
        m_cancel->show();
        m_showLog->show();
        return;
    }

    if (showResult) {
        const WarmupRun run = m_s.warmup.lastRun().value_or(WarmupRun{});
        const bool comparable = run.total > 0 && run.done <= run.total;
        const QString at = QStringLiteral("at %1 of %2").arg(Format::count(run.done), Format::count(run.total));
        m_block->setColumns({});
        switch (st) {
        case State::Finished:
            if (run.result == WarmupRun::Result::NothingToBuild) {
                m_block->setState(ProgressBlock::State::Warning);
                m_block->setTitle(QStringLiteral("Nothing to build yet"));
                m_block->setSubtitle(joined({m_s.warmup.game().title(), version}));
                m_block->setFraction(-1);
                m_block->setNote(run.message);
                m_continueFull = false;
                m_resume->setText(QStringLiteral("Run again"));
                m_resume->show();
                m_log->setEnabled(QFileInfo::exists(WarmupController::logPath()));
                m_log->show();
                m_dismiss->show();
                return;
            }
            m_block->setState(ProgressBlock::State::Finished);
            if (run.total <= 0) {
                m_block->setTitle(QStringLiteral("Nothing to pre-cache yet"));
                m_block->setSubtitle(joined({m_s.warmup.game().title(), version}));
                m_block->setFraction(1.0);
                m_block->setNote(spec.precacheNote);
            } else {
                m_block->setTitle(QStringLiteral("Pre-cache finished"));
                m_block->setSubtitle(QStringLiteral("%1 pipelines in %2 with %3").arg(Format::count(run.done), Format::duration(qint64(run.tookS + 0.5)), version));
                m_block->setFraction(1.0);
                m_block->setColumns({{QStringLiteral("Done"), builtOf(run.done, run.total)},
                                     {QStringLiteral("Speed"), run.tookS > 0 ? QStringLiteral("%1/s avg").arg(Format::count(qRound64(run.done / run.tookS))) : QStringLiteral("Unknown")},
                                     {QStringLiteral("Free RAM"), freeRam}});
                if (run.failed > 0)
                    m_block->setNote(QStringLiteral("%1 pipelines failed to build; %2 builds them when the game needs them. The log names them.").arg(Format::count(run.failed), spec.name));
                else if (!run.message.isEmpty())
                    m_block->setNote(run.message);
            }
            if (m_s.warmup.game().isValid() && m_s.gate.isFree()) {
                const QString name = m_s.warmup.game().title();
                m_play->setText(name.size() > 24 ? QStringLiteral("Play") : QStringLiteral("Play %1").arg(name));
                m_play->show();
            }
            m_dismiss->show();
            return;
        case State::Failed:
            m_block->setState(ProgressBlock::State::Failed);
            m_block->setTitle(QStringLiteral("Pre-cache failed"));
            m_block->setSubtitle(joined({comparable ? at : m_s.warmup.game().title(), version}));
            m_block->setFraction(comparable ? fractionOf(run.done, run.total) : -1);
            m_block->setNote(QStringLiteral("%1 exited with code %2.").arg(spec.name, Format::exitCode(run.exitCode))
                             + (run.message.isEmpty() ? QString() : QStringLiteral(" ") + sentence(run.message)));
            m_continueFull = false;
            m_resume->setText(QStringLiteral("Retry"));
            m_resume->show();
            m_log->setEnabled(QFileInfo::exists(WarmupController::logPath()));
            m_log->show();
            return;
        default:
            m_block->setState(ProgressBlock::State::Cancelled);
            m_block->setTitle(QStringLiteral("Pre-cache cancelled"));
            m_block->setSubtitle(joined({comparable ? at : m_s.warmup.game().title(), version}));
            m_block->setFraction(comparable ? fractionOf(run.done, run.total) : -1);
            m_block->setNote(QStringLiteral("What %1 built is kept in its own cache.").arg(spec.name));
            m_continueFull = false;
            m_resume->setText(QStringLiteral("Run again"));
            m_resume->show();
            m_dismiss->show();
            return;
        }
    }

    // idle: what a pre-cache would do for the picked game with its runner
    m_block->setState(ProgressBlock::State::Idle);
    m_block->setFraction(0);
    m_block->setColumns({});
    std::optional<WarmupRun> last;
    for (const WarmupRun& r : m_s.settings.recentWarmups()) {
        if (r.titleId == picked.titleId() && r.runnerId == spec.id) {
            last = r;
            break;
        }
    }
    const QString name = picked.title();
    if (last && last->result == WarmupRun::Result::Finished && last->total > 0)
        m_block->setTitle(QStringLiteral("%1 is pre-cached with %2").arg(name, spec.name));
    else if (last && last->result == WarmupRun::Result::Failed)
        m_block->setTitle(QStringLiteral("%1's last pre-cache with %2 failed").arg(name, spec.name));
    else if (last && last->result == WarmupRun::Result::NothingToBuild)
        m_block->setTitle(QStringLiteral("%1's last pre-cache with %2 had nothing to build").arg(name, spec.name));
    else
        m_block->setTitle(QStringLiteral("%1 runs with %2").arg(name, spec.name));
    m_block->setSubtitle(last ? QStringLiteral("last run %1").arg(Format::ago(last->finishedAt).toLower()) : version);
    m_block->setNote(spec.precache.isEmpty()                                                    ? QStringLiteral("%1 has no shader pre-cache; it builds what it needs while the game runs.").arg(spec.name)
                     : last && last->result == WarmupRun::Result::NothingToBuild && !last->message.isEmpty() ? last->message
                                                                                                             : spec.precacheNote);
    m_warm->setText(QStringLiteral("Pre-cache with %1").arg(spec.name));
    m_fullPass = false;
    m_warm->setEnabled(cannot.isEmpty());
    m_warm->setToolTip(cannot);
    m_warm->show();
}

void WarmupPage::refreshRuns()
{
    while (QLayoutItem* item = m_runs->takeAt(0)) {
        delete item->widget();
        delete item;
    }
    QList<WarmupRun> runs = m_s.settings.recentWarmups();
    const bool active = m_s.warmup.isActive();
    if (runs.isEmpty() && !active) {
        m_runs->addWidget(new Label(QStringLiteral("No warm-ups yet."), Label::Style::Small, Theme::Role::Text3), 0, 0);
        return;
    }
    const QStringList heads = {QStringLiteral("Game"), QStringLiteral("Pipelines"), QStringLiteral("Took"), QStringLiteral("Result"), QStringLiteral("When")};
    for (int c = 0; c < heads.size(); ++c) {
        auto* h = new Label(heads[c], Label::Style::Caption, Theme::Role::Text2);
        h->setFixedHeight(32);
        if (c == 1 || c == 2 || c == 4)
            h->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        m_runs->addWidget(h, 0, c);
    }
    m_runs->addWidget(new Rule, 1, 0, 1, heads.size());
    int row = 2;
    const auto addRow = [&](const QString& titleId, const QString& titleName, const QString& count, const QString& took, const RunResult& rr, const QString& when) {
        auto* nameCell = new QWidget;
        nameCell->setFixedHeight(47);
        auto* nh = new QHBoxLayout(nameCell);
        nh->setContentsMargins(0, 0, 0, 0);
        nh->setSpacing(10);
        auto* thumb = new CoverThumb(QSize(22, 30));
        thumb->setCover(titleId.isEmpty() ? QPixmap() : m_s.art(titleId, CoverCache::Kind::Grid), titleName);
        nh->addWidget(thumb);
        nh->addWidget(new Label(titleName.isEmpty() ? QStringLiteral("All games") : titleName, Label::Style::BodyMedium), 1);
        m_runs->addWidget(nameCell, row, 0);
        auto* countLabel = new Label(count, Label::Style::Number, Theme::Role::Text2);
        countLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        m_runs->addWidget(countLabel, row, 1);
        auto* tookLabel = new Label(took, Label::Style::Number, Theme::Role::Text2);
        tookLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        m_runs->addWidget(tookLabel, row, 2);
        auto* resultCell = new QWidget;
        auto* rh = new QHBoxLayout(resultCell);
        rh->setContentsMargins(0, 0, 0, 0);
        rh->setSpacing(0);
        rh->addWidget(new Badge({}, rr.tone, Badge::Kind::Plain));
        rh->addWidget(new Label(rr.text, Label::Style::Small, toneText(rr.tone)));
        m_runs->addWidget(resultCell, row, 3, Qt::AlignLeft | Qt::AlignVCenter);
        auto* whenLabel = new Label(when, Label::Style::Small, Theme::Role::Text3);
        whenLabel->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
        m_runs->addWidget(whenLabel, row, 4);
        m_runs->addWidget(new Rule, row + 1, 0, 1, heads.size());
        row += 2;
    };
    if (active) {
        const WarmupController::Progress pr = m_s.warmup.progress();
        const Game g = m_s.warmup.game();
        addRow(g.titleId(), g.title(), QStringLiteral("%1 / %2").arg(Format::count(pr.done), Format::count(pr.total)), Format::shortDuration(pr.elapsedS),
               {m_s.warmup.state() == WarmupController::State::PausedForRam ? QStringLiteral("Paused · RAM floor") : QStringLiteral("Running"),
                m_s.warmup.state() == WarmupController::State::PausedForRam ? Theme::Tone::Warning : Theme::Tone::Accent},
               QStringLiteral("Now"));
    }
    for (const WarmupRun& run : runs)
        addRow(run.titleId, run.titleName, builtOf(run.done, run.total), Format::shortDuration(run.tookS), runResult(run), Format::ago(run.finishedAt));
    m_runs->setColumnStretch(0, 1);
}
