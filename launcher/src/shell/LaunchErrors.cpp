#include "LaunchErrors.h"

#include "MainWindow.h"
#include "app/Services.h"
#include "core/Format.h"
#include "core/HardwareInfo.h"
#include "core/Paths.h"
#include "core/ShaderCacheInfo.h"
#include "widgets/Button.h"
#include "widgets/Card.h"
#include "widgets/CoverThumb.h"
#include "widgets/Dialog.h"
#include "widgets/Icon.h"
#include "widgets/Text.h"

#include <QAbstractButton>
#include <algorithm>
#include <optional>
#include <QCoreApplication>
#include <QDesktopServices>
#include <QDir>
#include <QFileDialog>
#include <QFileInfo>
#include <QHash>
#include <QHBoxLayout>
#include <QPainter>
#include <QLabel>
#include <QPointer>
#include <QRegularExpression>
#include <QTimer>
#include <QUrl>
#include <QVBoxLayout>

#ifdef Q_OS_WIN
#include <windows.h>
#include <psapi.h>
#include <tlhelp32.h>
#endif

namespace {

Label* paragraph(const QString& text, Theme::Role role = Theme::Role::Text2, Label::Style style = Label::Style::Body)
{
    auto* l = new Label(text, style, role);
    l->setWordWrap(true);
    return l;
}

// Icon tile, title and the design's spacing (16 under the icon, 8 under the title); the caller adds
// the rest with explicit spacing.
void begin(Dialog* d, const QString& icon, Theme::Tone tone, const QString& title)
{
    d->setIcon(icon, int(tone));
    d->setTitle(title);
    QVBoxLayout* b = d->body();
    b->setSpacing(0);
    if (QSpacerItem* s = b->itemAt(1)->spacerItem())
        s->changeSize(0, 16, QSizePolicy::Fixed, QSizePolicy::Fixed);
    if (auto* t = qobject_cast<QLabel*>(b->itemAt(2)->widget()))
        t->setWordWrap(true);
    b->addSpacing(8);
}

QWidget* pathBox(const QString& path, const QString& note = {})
{
    auto* box = new Card(Card::Kind::Sunken, Metrics::RadiusLg);
    auto* v = new QVBoxLayout(box);
    v->setContentsMargins(12, 10, 12, 10);
    v->setSpacing(4);
    auto* l = new Label(QDir::toNativeSeparators(path), Label::Style::Mono, Theme::Role::Text);
    l->setFont(Theme::mono(12));
    l->setWordWrap(true);
    l->setTextInteractionFlags(Qt::TextSelectableByMouse);
    v->addWidget(l);
    if (!note.isEmpty()) {
        auto* n = new Label(note, Label::Style::Caption, Theme::Role::Text3);
        n->setWordWrap(true);
        v->addWidget(n);
    }
    return box;
}

Button* closeButton(Dialog* d, const QString& text, Button::Variant variant = Button::Variant::Secondary)
{
    auto* b = new Button(text, variant, Button::Size::Large);
    QObject::connect(b, &Button::clicked, d, &Dialog::dismiss);
    return b;
}

Button* actionButton(Dialog* d, const QString& text, std::function<void()> action,
                     Button::Variant variant = Button::Variant::Primary, const QString& icon = {})
{
    auto* b = new Button(text, variant, Button::Size::Large, icon);
    QObject::connect(b, &Button::clicked, d, [d, action] {
        d->dismiss();
        if (action)
            action();
    });
    return b;
}

// The way out of a missing or too old emulator: choose another one, then launch again. XSEMU_EMULATOR
// wins over any choice, so with it set the dialog says where to change it instead.
void offerChooseEmulator(Dialog* d, QWidget* window, Services& s, std::function<void()> retry)
{
    if (AppSettings::emulatorPathFromEnvironment()) {
        d->body()->addSpacing(8);
        d->body()->addWidget(paragraph(QStringLiteral("XSEMU_EMULATOR in the launcher's environment sets this path. Change it there, or start the launcher "
                                                      "without it to choose the emulator here."),
                                       Theme::Role::Text3, Label::Style::Small));
        return;
    }
    Button* choose = actionButton(d, QStringLiteral("Choose emulator..."), [window, &s, retry] {
        if (LaunchErrors::chooseEmulator(window, s.settings))
            retry();
    }, Button::Variant::Primary, QStringLiteral("folder"));
    choose->setObjectName(QStringLiteral("chooseEmulator"));
    d->addButton(choose);
}

// A label/value table on a sunken card ("Needed 9.6 GB").
struct FactsBox {
    Card* card;
    QVBoxLayout* rows;
    Label* add(const QString& label, const QString& value, Theme::Role role = Theme::Role::Text)
    {
        auto* row = new QHBoxLayout;
        row->setSpacing(12);
        row->addWidget(new Label(label, Label::Style::Small, Theme::Role::Text2));
        row->addStretch(1);
        auto* v = new Label(value, Label::Style::SmallStrong, role);
        row->addWidget(v);
        rows->addLayout(row);
        return v;
    }
};

FactsBox factsBox()
{
    auto* card = new Card(Card::Kind::Sunken, 10);
    auto* v = new QVBoxLayout(card);
    v->setContentsMargins(14, 12, 14, 12);
    v->setSpacing(8);
    return {card, v};
}

// The design's 16 px checkbox with a label (cold-cache prompt).
class CheckRow : public QAbstractButton {
public:
    explicit CheckRow(const QString& text)
    {
        setText(text);
        setCheckable(true);
        setCursor(Qt::PointingHandCursor);
        setFocusPolicy(Qt::StrongFocus);
        connect(&Theme::instance(), &Theme::changed, this, qOverload<>(&QWidget::update));
    }
    QSize sizeHint() const override
    {
        return {26 + QFontMetrics(Theme::font(Metrics::TextSm)).horizontalAdvance(text()) + 2, 18};
    }

protected:
    void paintEvent(QPaintEvent*) override
    {
        const Palette& c = Theme::instance().palette();
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        const QRectF box(0.5, 1.5, 15, 15);
        if (isChecked()) {
            p.setPen(Qt::NoPen);
            p.setBrush(c.accent);
            p.drawRoundedRect(box, 4, 4);
            Icon::paint(p, box, QStringLiteral("check"), c.onAccent, 12);
        } else {
            p.setPen(QPen(c.borderStrong, 1));
            p.setBrush(c.surface);
            p.drawRoundedRect(box, 4, 4);
        }
        p.setFont(Theme::font(Metrics::TextSm));
        p.setPen(c.text);
        p.drawText(QRectF(26, 0, width() - 26, height()), Qt::AlignLeft | Qt::AlignVCenter, text());
    }
};

// The newest warm-up the launcher recorded for the title, if any.
std::optional<WarmupRun> lastWarmup(Services& s, const Game& game)
{
    for (const WarmupRun& r : s.settings.recentWarmups()) {
        if (r.titleId == game.titleId())
            return r;
    }
    return std::nullopt;
}

// After stop: run next() once the gate is free (a game stops within its grace period, or is killed).
void whenGateFree(Services& s, std::function<void()> next)
{
    if (s.gate.isFree()) {
        next();
        return;
    }
    auto* conn = new QMetaObject::Connection;
    *conn = QObject::connect(&s.gate, &ProcessGate::changed, &s.gate, [&s, conn, next] {
        if (!s.gate.isFree())
            return;
        QObject::disconnect(*conn);
        delete conn;
        QTimer::singleShot(0, &s.gate, next);
    });
}

void showStopFirst(QWidget* window, Services& s, const Game& game, std::function<void()> retry)
{
    auto* d = new Dialog(window, 460);
    const QString owner = s.gate.ownerTitle();
    if (s.gate.owner() == ProcessGate::Owner::Warmup) {
        begin(d, QStringLiteral("bolt"), Theme::Tone::Warning, QStringLiteral("Cancel the warm-up first?"));
        d->body()->addWidget(paragraph(QStringLiteral("%1 can't start while the shader warm-up for %2 runs, because XSEmu Core runs one process at a time. Compiles in flight finish before it stops.").arg(game.title(), owner)));
        d->addButton(closeButton(d, QStringLiteral("Keep warming up")));
        d->addButton(actionButton(d, QStringLiteral("Cancel and play %1").arg(game.title()), [&s, retry] {
            s.gate.stopOwner();
            whenGateFree(s, retry);
        }));
        d->open();
        return;
    }
    const Game held = s.emulator.game();
    const bool paused = s.emulator.state() == EmulatorController::State::Paused;
    begin(d, QStringLiteral("stop"), Theme::Tone::Warning, QStringLiteral("Stop %1 first?").arg(owner));
    const qint64 heldBytes = s.emulator.stats().heldBytes;
    QString why = QStringLiteral("%1 can't start while %2 is %3.").arg(game.title(), owner, paused ? QStringLiteral("paused") : QStringLiteral("running"));
    why += heldBytes > 0 ? QStringLiteral(" It still holds %1, and XSEmu Core runs one game at a time.").arg(Format::gigabytes(heldBytes))
                         : QStringLiteral(" XSEmu Core runs one game at a time.");
    d->body()->addWidget(paragraph(why));
    d->body()->addSpacing(16);

    auto* box = new Card(Card::Kind::Sunken, 10);
    auto* bv = new QVBoxLayout(box);
    bv->setContentsMargins(14, 14, 14, 14);
    bv->setSpacing(8);
    auto* row = new QHBoxLayout;
    row->setSpacing(10);
    auto* thumb = new CoverThumb(QSize(24, 36));
    thumb->setCover(s.art(held.titleId(), CoverCache::Kind::Grid), held.title());
    row->addWidget(thumb);
    row->addWidget(new Label(owner, Label::Style::Small, Theme::Role::Text), 1);
    const qint64 session = s.emulator.sessionSeconds();
    if (session > 0)
        row->addWidget(new Label(QStringLiteral("%1 this session").arg(Format::duration(session).replace(QRegularExpression(QStringLiteral(" \\d+ s$")), QString())), Label::Style::Caption, Theme::Role::Text2));
    bv->addLayout(row);
    auto* note = new Label(QStringLiteral("Stopping quits the game without saving. Use the game's own save first if you need it."), Label::Style::Caption, Theme::Role::Text3);
    note->setWordWrap(true);
    bv->addWidget(note);
    d->body()->addWidget(box);

    d->addButton(closeButton(d, QStringLiteral("Cancel")));
    // the design names both games on the buttons; long names fall back to the verb alone
    const bool shortLabels = owner.size() + game.title().size() > 26;
    const QString resumeLabel = paused ? QStringLiteral("Resume") : QStringLiteral("Back to game");
    QPointer<QWidget> w = window;
    d->addButton(actionButton(d, shortLabels || !paused ? resumeLabel : QStringLiteral("Resume %1").arg(owner), [&s, w, paused] {
        if (paused)
            s.emulator.resume();
        if (auto* mw = qobject_cast<MainWindow*>(w.data()))
            mw->enterInGame();
    }, Button::Variant::Secondary));
    d->addButton(actionButton(d, shortLabels ? QStringLiteral("Stop and play") : QStringLiteral("Stop and play %1").arg(game.title()), [&s, retry] {
        s.gate.stopOwner();
        whenGateFree(s, retry);
    }));
    d->open();
}

void showFilesMissing(QWidget* window, Services& s, const Game& game, std::function<void()> retry)
{
    auto* d = new Dialog(window, 500);
    begin(d, QStringLiteral("folder"), Theme::Tone::Danger, QStringLiteral("%1 can't start").arg(game.title()));
    const bool folderGone = !QFileInfo(game.folder).isDir() || !QFileInfo::exists(game.configPath);
    if (folderGone) {
        d->body()->addWidget(paragraph(QStringLiteral("Its folder is gone or a drive is disconnected. XSEmu Launcher looked for MicrosoftGame.config here:")));
        d->body()->addSpacing(12);
        d->body()->addWidget(pathBox(game.folder));
    } else {
        const QString exe = game.exe.isEmpty() ? game.executables.value(0) : game.exe;
        d->body()->addWidget(paragraph(exe.isEmpty() ? QStringLiteral("Its MicrosoftGame.config names no executable, so XSEmu Launcher has nothing to start in:")
                                                     : QStringLiteral("Its MicrosoftGame.config names %1, but that file isn't in the game folder. XSEmu Launcher looked here:").arg(QDir::toNativeSeparators(exe))));
        d->body()->addSpacing(12);
        d->body()->addWidget(pathBox(exe.isEmpty() ? game.folder : QDir(game.folder).filePath(exe)));
    }
    d->body()->addSpacing(12);
    d->body()->addWidget(paragraph(QStringLiteral("Reconnect the drive and try again, or point XSEmu Launcher at the new folder. The shader cache stays either way."),
                                   Theme::Role::Text3, Label::Style::Small));

    const QStringList roots = s.settings.gameFolders();
    const QString canonical = QDir::cleanPath(game.folder);
    const bool isRoot = std::any_of(roots.cbegin(), roots.cend(), [&](const QString& r) { return QDir::cleanPath(r) == canonical; });
    if (folderGone || isRoot) {
        d->addButton(actionButton(d, QStringLiteral("Remove from library"), [&s, canonical] {
            QStringList folders = s.settings.gameFolders();
            folders.erase(std::remove_if(folders.begin(), folders.end(), [&](const QString& r) { return QDir::cleanPath(r) == canonical; }), folders.end());
            s.settings.setGameFolders(folders);
            s.library.rescanNow();
        }, Button::Variant::Ghost), true);
    }
    QPointer<QWidget> w = window;
    d->addButton(actionButton(d, QStringLiteral("Locate folder\u2026"), [&s, w, game, canonical, retry] {
        const QString dir = QFileDialog::getExistingDirectory(w, QStringLiteral("Locate %1").arg(game.title()));
        if (dir.isEmpty())
            return;
        const Game found = Library::readGame(dir);
        QStringList folders = s.settings.gameFolders();
        folders.erase(std::remove_if(folders.begin(), folders.end(), [&](const QString& r) { return QDir::cleanPath(r) == canonical; }), folders.end());
        if (!folders.contains(dir))
            folders << dir;
        s.settings.setGameFolders(folders);
        s.library.rescanNow();
        if (found.isValid() && w) {
            if (auto* mw = qobject_cast<MainWindow*>(w.data()))
                mw->play(s.library.game(found.titleId()).isValid() ? s.library.game(found.titleId()) : found);
        }
    }, Button::Variant::Secondary));
    d->addButton(actionButton(d, QStringLiteral("Try again"), [&s, retry] {
        s.library.rescanNow();
        retry();
    }));
    d->open();
}

void showNotEnoughRam(QWidget* window, const Game& game, const EmulatorController::LaunchCheck& check, std::function<void()> retry)
{
    auto* d = new Dialog(window, 480);
    begin(d, QStringLiteral("warning"), Theme::Tone::Warning, QStringLiteral("Not enough free memory to start"));
    const qint64 floorMb = EmulatorController::kRamFloorMb;
    const qint64 extraMb = qMax<qint64>(0, check.needMb - floorMb);
    QString body = extraMb > 0
        ? QStringLiteral("%1 needs %2 free to start: the %3 you set in Settings, plus the 1 GB floor XSEmu Core keeps free.").arg(game.title(), Format::gigabytesFromMb(check.needMb), Format::gigabytesFromMb(extraMb))
        : QStringLiteral("XSEmu Core keeps 1 GB of RAM free while %1 runs, and stops the game below that.").arg(game.title());
    body += QStringLiteral(" This PC has %1 free right now.").arg(Format::gigabytesFromMb(check.freeMb));
    auto* text = paragraph(body);
    d->body()->addWidget(text);
    d->body()->addSpacing(14);

    FactsBox facts = factsBox();
    facts.add(QStringLiteral("Needed"), Format::gigabytesFromMb(check.needMb));
    Label* freeNow = facts.add(QStringLiteral("Free now"), Format::gigabytesFromMb(check.freeMb), Theme::Role::Warning);
    const LaunchErrors::BiggestApp big = LaunchErrors::biggestOtherApp();
    if (!big.name.isEmpty())
        facts.add(QStringLiteral("Biggest other app"), QStringLiteral("%1 \u00B7 %2").arg(big.name, Format::gigabytes(big.bytes)));
    d->body()->addWidget(facts.card);
    d->body()->addSpacing(12);
    auto* note = paragraph(QStringLiteral("Close other apps and XSEmu Launcher starts the game as soon as enough memory is free."), Theme::Role::Text3, Label::Style::Small);
    d->body()->addWidget(note);

    auto* timer = new QTimer(d);
    timer->setInterval(1000);
    const qint64 need = check.needMb;
    QObject::connect(timer, &QTimer::timeout, d, [freeNow, need] {
        const qint64 mb = qint64(HardwareInfo::availableRam() / (1024 * 1024));
        freeNow->setText(Format::gigabytesFromMb(mb));
        freeNow->setRole(mb >= need ? Theme::Role::Success : Theme::Role::Warning);
    });
    timer->start();

    d->addButton(closeButton(d, QStringLiteral("Cancel")));
    auto* wait = new Button(QStringLiteral("Start when memory frees"), Button::Variant::Primary, Button::Size::Large);
    QObject::connect(wait, &Button::clicked, d, [d, wait, note, timer, need, retry] {
        wait->setEnabled(false);
        wait->setText(QStringLiteral("Waiting for memory\u2026"));
        note->setText(QStringLiteral("Waiting. The game starts by itself once %1 is free; Cancel stops waiting.").arg(Format::gigabytesFromMb(need)));
        QObject::connect(timer, &QTimer::timeout, d, [d, need, retry] {
            if (qint64(HardwareInfo::availableRam() / (1024 * 1024)) < need)
                return;
            d->dismiss();
            retry();
        });
    });
    d->addButton(wait);
    d->open();
}

}

bool LaunchErrors::shadersCold(Services& s, const Game& game)
{
    if (!s.emulator.runnerFor(game).isBuiltIn())
        return false; // a third-party runner keeps its own cache; its pre-cache is optional
    switch (shownShaderState(s.settings, game.titleId(), s.settings.recentWarmups(), s.shaderCache.snapshot())) {
    case TitleShaderState::NeverWarmed:
    case TitleShaderState::LastRunIncomplete:
    case TitleShaderState::LastRunFailed:
    case TitleShaderState::ProfileChanged:
        return true;
    case TitleShaderState::Ready:
    case TitleShaderState::Unknown:
        break;
    }
    return false;
}

void LaunchErrors::confirmPlay(QWidget* window, Services& s, const Game& game, std::function<void()> play, std::function<void()> warmUp)
{
    if (!s.gate.isFree() || !shadersCold(s, game)) {
        play();
        return;
    }
    const QString alwaysKey = QStringLiteral("alwaysWarmUp");
    if (s.settings.gameOverrides(game.titleId()).value(alwaysKey).toBool()) {
        warmUp();
        return;
    }
    const TitleShaderState state = shownShaderState(s.settings, game.titleId(), s.settings.recentWarmups(), s.shaderCache.snapshot());
    const std::optional<WarmupRun> last = lastWarmup(s, game);
    auto* d = new Dialog(window, 460);
    begin(d, QStringLiteral("bolt"), Theme::Tone::Warning, QStringLiteral("Warm up %1 first?").arg(game.title()));
    QString why;
    switch (state) {
    case TitleShaderState::LastRunIncomplete:
        why = last && last->left > 0   ? QStringLiteral("The last warm-up for %1 stopped part way, with %2 pipelines left.").arg(game.title(), Format::count(last->left))
            : last && last->total > 0  ? QStringLiteral("The last warm-up for %1 stopped part way, at %2 of %3 pipelines.").arg(game.title(), Format::count(last->done), Format::count(last->total))
                                       : QStringLiteral("The last warm-up for %1 stopped part way.").arg(game.title());
        break;
    case TitleShaderState::ProfileChanged:
        why = QStringLiteral("The run profile for %1 changed since its last warm-up, so its shaders build again.").arg(game.title());
        break;
    case TitleShaderState::LastRunFailed:
        why = QStringLiteral("The last warm-up for %1 failed%2.").arg(game.title(), last ? QStringLiteral(" (exit code %1)").arg(Format::exitCode(last->exitCode)) : QString());
        break;
    default:
        why = QStringLiteral("%1 has no shaders on this PC yet.").arg(game.title());
        break;
    }
    why += QStringLiteral(" You can play now, but expect stutter and long stalls while each shader compiles the first time the game needs it.");
    d->body()->addWidget(paragraph(why));

    if (last && last->total > 0 && last->done > 0 && last->tookS > 0) {
        // what is left of that pass, at the rate it built the rest
        const qint64 rest = last->left > 0 ? last->left : last->total;
        const double estimate = last->tookS / double(last->done) * double(rest);
        auto* row = new QHBoxLayout;
        row->setSpacing(8);
        row->addWidget(new IconView(QStringLiteral("clock"), Theme::Role::Text3, 14));
        const qint64 secs = qint64(estimate + 0.5);
        const QString about = secs < 90 ? QStringLiteral("%1 seconds").arg(secs) : QStringLiteral("%1 minutes").arg((secs + 30) / 60);
        row->addWidget(new Label(QStringLiteral("About %1 on this PC · %2%3").arg(about, Format::count(rest), last->left > 0 ? QStringLiteral(" left") : QStringLiteral(" pipelines")),
                                 Label::Style::Small, Theme::Role::Text2), 1);
        d->body()->addSpacing(16);
        d->body()->addLayout(row);
    }
    auto* always = new CheckRow(QStringLiteral("Always warm up first for this game"));
    d->body()->addSpacing(18);
    d->body()->addWidget(always);

    const QString titleId = game.titleId();
    auto remember = [&s, always, titleId, alwaysKey] {
        if (always->isChecked())
            s.settings.setGameOverride(titleId, alwaysKey, true);
    };
    d->addButton(actionButton(d, QStringLiteral("Play anyway"), [remember, play] {
        remember();
        play();
    }, Button::Variant::Secondary));
    d->addButton(actionButton(d, QStringLiteral("Warm up first"), [remember, warmUp] {
        remember();
        warmUp();
    }, Button::Variant::Primary, QStringLiteral("bolt")));
    d->open();
}

bool LaunchErrors::chooseEmulator(QWidget* parent, AppSettings& settings)
{
    const QString picked = QFileDialog::getOpenFileName(parent, QStringLiteral("Choose the emulator"), QFileInfo(settings.emulatorPath()).absolutePath(),
                                                        QStringLiteral("XSEmu Core (XSEmu.exe xcl.exe);;Programs (*.exe);;All files (*)"));
    if (picked.isEmpty())
        return false;
    settings.setEmulatorPathOverride(picked);
    return true;
}

void LaunchErrors::showLaunchProblem(QWidget* window, Services& s, const Game& game,
                                     const EmulatorController::LaunchCheck& check, std::function<void()> retry)
{
    using P = EmulatorController::LaunchCheck::Problem;
    switch (check.problem) {
    case P::None:
        return;
    case P::Busy:
        showStopFirst(window, s, game, retry);
        return;
    case P::GameFilesMissing:
        showFilesMissing(window, s, game, retry);
        return;
    case P::NotEnoughRam:
        showNotEnoughRam(window, game, check, retry);
        return;
    case P::EmulatorMissing: {
        auto* d = new Dialog(window, 500);
        begin(d, QStringLiteral("file"), Theme::Tone::Danger, QStringLiteral("XSEmu.exe is missing"));
        d->body()->addWidget(paragraph(check.reason));
        d->body()->addSpacing(12);
        d->body()->addWidget(pathBox(s.settings.emulatorPath()));
        d->body()->addSpacing(12);
        d->body()->addWidget(paragraph(QStringLiteral("Keep XSEmu.exe next to the launcher, or choose where it is. The choice is kept, so the launcher "
                                                      "starts from its normal shortcut."),
                                       Theme::Role::Text3, Label::Style::Small));
        d->addButton(closeButton(d, QStringLiteral("Close")));
        offerChooseEmulator(d, window, s, retry);
        d->open();
        return;
    }
    case P::EmulatorTooOld: {
        auto* d = new Dialog(window, 500);
        begin(d, QStringLiteral("warning"), Theme::Tone::Danger, QStringLiteral("This XSEmu Core build is too old for the launcher"));
        d->body()->addWidget(paragraph(check.reason));
        d->body()->addSpacing(12);
        d->body()->addWidget(pathBox(s.settings.emulatorPath()));
        d->body()->addSpacing(12);
        d->body()->addWidget(paragraph(QStringLiteral("The launcher didn't start it. Put a current XSEmu.exe next to the launcher, or choose another build."), Theme::Role::Text3, Label::Style::Small));
        d->addButton(closeButton(d, QStringLiteral("Close")));
        offerChooseEmulator(d, window, s, retry);
        d->open();
        return;
    }
    case P::RunnerNotInstalled: {
        const RunnerSpec& runner = s.emulator.runnerFor(game);
        auto* d = new Dialog(window, 500);
        begin(d, QStringLiteral("download"), Theme::Tone::Accent, QStringLiteral("%1 is not installed").arg(runner.name));
        d->body()->addWidget(paragraph(check.reason));
        d->addButton(actionButton(d, QStringLiteral("Play with XSEmu Core"), [&s, game, retry] {
            s.settings.setGameOverride(game.titleId(), QStringLiteral("runner"), QString::fromLatin1(Runners::XSEmu));
            retry();
        }, Button::Variant::Secondary));
        d->addButton(actionButton(d, QStringLiteral("Install %1").arg(runner.name), [&s, id = runner.id] { s.runners.install(id); }, Button::Variant::Primary, QStringLiteral("download")));
        d->open();
        return;
    }
    case P::OtherEmulatorRunning: {
        auto* d = new Dialog(window, 500);
        begin(d, QStringLiteral("warning"), Theme::Tone::Warning, QStringLiteral("Emulator busy"));
        d->body()->addWidget(paragraph(check.reason));
        QStringList holders = check.processes;
        if (!check.marker.isEmpty())
            holders << QDir::toNativeSeparators(check.marker);
        if (!holders.isEmpty()) {
            d->body()->addSpacing(12);
            d->body()->addWidget(pathBox(holders.join(QLatin1Char('\n'))));
        }
        d->body()->addSpacing(12);
        d->body()->addWidget(paragraph(check.processes.isEmpty() ? QStringLiteral("Play and Warm-up wait until the file is gone.")
                                                                 : QStringLiteral("Only one emulator process runs at a time. Let it finish or close it, then try again."),
                                       Theme::Role::Text3, Label::Style::Small));
        d->addButton(closeButton(d, QStringLiteral("Close")));
        d->addButton(actionButton(d, QStringLiteral("Try again"), retry));
        d->open();
        return;
    }
    }
}

QString LaunchErrors::logsFolder(const Game& game)
{
    return Paths::logsDir(game.titleId());
}

QString LaunchErrors::crashReport(const Game& game)
{
    const QString report = Paths::reportJson(game.titleId());
    return QFileInfo::exists(report) ? report : logsFolder(game);
}

void LaunchErrors::openCrashReport(const Game& game)
{
    QDesktopServices::openUrl(QUrl::fromLocalFile(crashReport(game)));
}

LaunchErrors::ExitReport LaunchErrors::describeExit(const EmulatorController::ExitInfo& info)
{
    using K = EmulatorController::ExitInfo::Kind;
    ExitReport r;
    const QString title = info.game.title();
    const QString after = info.playedSeconds >= 60 ? QStringLiteral(" after %1").arg(Format::playTime(info.playedSeconds)) : QString();
    const QString saved = QStringLiteral(" Play time was saved and the memory is free again.");
    switch (info.kind) {
    case K::Stopped:
    case K::TitleExit:
        return r;
    case K::DeviceRemoved: {
        r.icon = QStringLiteral("warning");
        r.status = QStringLiteral("Stopped \u00B7 GPU device removed");
        r.title = QStringLiteral("The graphics driver reset");
        const QString reason = info.message.isEmpty() ? QStringLiteral("DXGI_ERROR_DEVICE_REMOVED") : info.message;
        r.body = QStringLiteral("Windows removed the GPU device (%1)%2. This is usually a driver timeout.").arg(reason, after) + saved;
        break;
    }
    case K::Crash:
        r.icon = QStringLiteral("warning");
        r.status = QStringLiteral("Stopped \u00B7 crashed");
        r.title = QStringLiteral("%1 crashed").arg(title);
        r.body = (info.message.isEmpty() ? QStringLiteral("XSEmu Core stopped unexpectedly%1.").arg(after) : info.message) + saved;
        break;
    case K::Error:
        r.icon = QStringLiteral("warning");
        r.status = QStringLiteral("Couldn't start");
        r.title = QStringLiteral("%1 couldn't start").arg(title);
        r.body = info.message.isEmpty() ? QStringLiteral("XSEmu Core reported an error while starting the game.") : info.message;
        break;
    }
    const bool report = QFileInfo::exists(Paths::reportJson(info.game.titleId()));
    r.detailPath = QDir::toNativeSeparators(report ? Paths::reportJson(info.game.titleId()) : logsFolder(info.game));
    r.detailNote = report ? QStringLiteral("Exit code %1. The log and this crash report stay on this PC; nothing is uploaded.").arg(Format::exitCode(info.exitCode))
                          : QStringLiteral("Exit code %1. The log stays on this PC; nothing is uploaded.").arg(Format::exitCode(info.exitCode));
    r.openLabel = report ? QStringLiteral("Open crash report") : QStringLiteral("Open logs");
    return r;
}

void LaunchErrors::showExit(QWidget* window, const EmulatorController::ExitInfo& info, std::function<void()> playAgain)
{
    const ExitReport r = describeExit(info);
    if (r.title.isEmpty())
        return;
    auto* d = new Dialog(window, 520);
    begin(d, r.icon, r.tone, r.title);
    d->body()->addWidget(paragraph(r.body));
    d->body()->addSpacing(16);
    d->body()->addWidget(pathBox(r.detailPath, r.detailNote));
    d->addButton(actionButton(d, QStringLiteral("Relaunch"), playAgain, Button::Variant::Primary, QStringLiteral("play")), true);
    const Game game = info.game;
    auto* open = new Button(r.openLabel, Button::Variant::Secondary, Button::Size::Large, QStringLiteral("file"));
    QObject::connect(open, &Button::clicked, d, [game] { openCrashReport(game); });
    d->addButton(open, true);
    d->addButton(closeButton(d, QStringLiteral("Back to library"), Button::Variant::Ghost));
    d->open();
}

LaunchErrors::BiggestApp LaunchErrors::biggestOtherApp()
{
    QHash<QString, qint64> byName;
    const qint64 self = QCoreApplication::applicationPid();
#ifdef Q_OS_WIN
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);
    if (snap == INVALID_HANDLE_VALUE)
        return {};
    PROCESSENTRY32W pe{};
    pe.dwSize = sizeof(pe);
    for (BOOL ok = Process32FirstW(snap, &pe); ok; ok = Process32NextW(snap, &pe)) {
        if (qint64(pe.th32ProcessID) == self || pe.th32ProcessID == 0)
            continue;
        HANDLE h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE, pe.th32ProcessID);
        if (!h)
            continue;
        PROCESS_MEMORY_COUNTERS pmc{};
        pmc.cb = sizeof(pmc);
        if (K32GetProcessMemoryInfo(h, &pmc, sizeof(pmc)))
            byName[QString::fromWCharArray(pe.szExeFile)] += qint64(pmc.WorkingSetSize);
        CloseHandle(h);
    }
    CloseHandle(snap);
#else
    const QDir proc(QStringLiteral("/proc"));
    const qint64 page = 4096;
    for (const QString& pid : proc.entryList(QDir::Dirs | QDir::NoDotAndDotDot)) {
        bool numeric = false;
        const qint64 id = pid.toLongLong(&numeric);
        if (!numeric || id == self)
            continue;
        QFile comm(proc.filePath(pid + QStringLiteral("/comm")));
        QFile statm(proc.filePath(pid + QStringLiteral("/statm")));
        if (!comm.open(QIODevice::ReadOnly) || !statm.open(QIODevice::ReadOnly))
            continue;
        const QList<QByteArray> fields = statm.readAll().split(' ');
        if (fields.size() < 2)
            continue;
        byName[QString::fromUtf8(comm.readAll()).trimmed()] += fields[1].toLongLong() * page;
    }
#endif
    BiggestApp best;
    for (auto it = byName.cbegin(); it != byName.cend(); ++it) {
        if (it.value() > best.bytes) {
            best.bytes = it.value();
            best.name = it.key();
        }
    }
    if (best.name.endsWith(QLatin1String(".exe"), Qt::CaseInsensitive))
        best.name.chop(4);
    if (!best.name.isEmpty())
        best.name[0] = best.name.at(0).toUpper();
    return best;
}
