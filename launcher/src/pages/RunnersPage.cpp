#include "RunnersPage.h"

#include "PageScaffold.h"
#include "RunnerUi.h"
#include "app/Services.h"
#include "core/Format.h"
#include "core/HardwareInfo.h"
#include "widgets/Badge.h"
#include "widgets/Button.h"
#include "widgets/Card.h"
#include "widgets/Icon.h"
#include "widgets/PageHeader.h"
#include "widgets/Spinner.h"
#include "widgets/Text.h"
#include "widgets/Toggle.h"

#include <QDesktopServices>
#include <QDir>
#include <QFileInfo>
#include <QUrl>
#include <algorithm>
#include <QHBoxLayout>
#include <QPainter>
#include <QVBoxLayout>

namespace {

// A thin progress bar; a negative fraction draws a moving segment (no total known).
class Bar : public QWidget {
public:
    Bar() { setFixedHeight(6); }
    void setFraction(double f)
    {
        m_fraction = f;
        if (f < 0 && !m_timer)
            m_timer = startTimer(40);
        else if (f >= 0 && m_timer) {
            killTimer(m_timer);
            m_timer = 0;
        }
        update();
    }

protected:
    void timerEvent(QTimerEvent*) override
    {
        m_phase = (m_phase + 2) % 200;
        update();
    }
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        p.setPen(Qt::NoPen);
        p.setBrush(Theme::color(Theme::Role::Hover));
        p.drawRoundedRect(QRectF(rect()), 3, 3);
        p.setBrush(Theme::color(Theme::Role::Accent));
        if (m_fraction >= 0) {
            p.drawRoundedRect(QRectF(0, 0, width() * qBound(0.0, m_fraction, 1.0), height()), 3, 3);
        } else {
            const double w = width() * 0.25;
            const double x = (width() + w) * (m_phase / 200.0) - w;
            p.setClipRect(rect());
            p.drawRoundedRect(QRectF(x, 0, w, height()), 3, 3);
        }
    }

private:
    double m_fraction = 0;
    int m_timer = 0;
    int m_phase = 0;
};

// One step row of an install: a glyph (done, working, waiting), the step name and what it is doing.
struct StepRow {
    QWidget* row;
    IconView* done;
    Spinner* spin;
    IconView* wait;
    Label* name;
    Label* detail;
};

StepRow makeStep(const QString& name)
{
    StepRow r;
    r.row = new QWidget;
    auto* h = new QHBoxLayout(r.row);
    h->setContentsMargins(0, 0, 0, 0);
    h->setSpacing(10);
    r.done = new IconView(QStringLiteral("check"), Theme::Role::Success, 14);
    r.spin = new Spinner(14);
    r.wait = new IconView(QStringLiteral("pending"), Theme::Role::Text3, 14);
    r.name = new Label(name, Label::Style::SmallMedium);
    r.name->setFixedWidth(170);
    r.detail = new Label({}, Label::Style::Small, Theme::Role::Text2);
    h->addWidget(r.done);
    h->addWidget(r.spin);
    h->addWidget(r.wait);
    h->addWidget(r.name);
    h->addWidget(r.detail, 1);
    return r;
}

}

class RunnerCard : public Card {
public:
    RunnerCard(Services& s, const RunnerSpec& spec)
        : Card(Card::Kind::Surface, Metrics::RadiusXl)
        , m_s(s)
        , m_spec(spec)
    {
        setObjectName(QStringLiteral("runner-") + spec.id);
        auto* v = new QVBoxLayout(this);
        v->setContentsMargins(20, 18, 20, 18);
        v->setSpacing(14);
        auto* top = new QHBoxLayout;
        top->setSpacing(16);
        v->addLayout(top);

        QWidget* icon;
        if (spec.isBuiltIn()) {
            icon = new BrandMark(40);
        } else {
            auto* tile = new Card(Card::Kind::Sunken, Metrics::RadiusLg);
            tile->setFixedSize(40, 40);
            auto* tl = new QVBoxLayout(tile);
            tl->setContentsMargins(0, 0, 0, 0);
            tl->addWidget(new IconView(QStringLiteral("runner"), Theme::Role::Text2, 18), 0, Qt::AlignCenter);
            icon = tile;
        }
        top->addWidget(icon, 0, Qt::AlignTop);

        auto* mid = new QVBoxLayout;
        mid->setSpacing(6);
        auto* title = new QHBoxLayout;
        title->setSpacing(8);
        auto* name = new Label(spec.name, Label::Style::Title);
        name->setFont(Theme::font(16, 600, -0.01));
        title->addWidget(name);
        title->addWidget(new Label(RunnerUi::owner(spec), Label::Style::Small, Theme::Role::Text3));
        if (Badge* b = RunnerUi::vendorBadge(spec))
            title->addWidget(b);
        m_recommended = new Badge(QStringLiteral("Recommended for your GPU"), Theme::Tone::Accent);
        m_recommended->setIcon(QStringLiteral("check"));
        title->addWidget(m_recommended);
        title->addStretch(1);
        mid->addLayout(title);
        auto* summary = new Label(spec.summary, Label::Style::Small, Theme::Role::Text2);
        summary->setWordWrap(true);
        mid->addWidget(summary);
        auto* repo = new QHBoxLayout;
        repo->setSpacing(6);
        repo->addWidget(new IconView(QStringLiteral("external"), Theme::Role::Text3, 12));
        repo->addWidget(new Label(RunnerUi::repoLine(spec), Label::Style::Mono, Theme::Role::Text2));
        repo->addStretch(1);
        mid->addLayout(repo);
        if (!spec.isBuiltIn()) {
            auto* prov = new QHBoxLayout;
            prov->setSpacing(6);
            prov->addWidget(new IconView(QStringLiteral("info"), Theme::Role::Text3, 12));
            prov->addWidget(new Label(RunnerUi::provenance(spec), Label::Style::Caption, Theme::Role::Text3));
            prov->addStretch(1);
            mid->addLayout(prov);
        }
        if (!spec.note.isEmpty()) {
            auto* note = new Label(spec.note, Label::Style::Caption, Theme::Role::Warning);
            note->setWordWrap(true);
            mid->addWidget(note);
        }
        const QHash<QString, bool> choices = m_s.settings.runnerOptions(spec.id);
        for (const RunnerSpec::Option& option : spec.options) {
            auto* row = new QHBoxLayout;
            row->setSpacing(8);
            auto* toggle = new Toggle;
            toggle->setObjectName(QStringLiteral("option-%1-%2").arg(spec.id, option.key));
            toggle->setChecked(RunnerSpec::optionOn(option, choices));
            toggle->setToolTip(option.description);
            connect(toggle, &Toggle::toggled, this, [this, key = option.key](bool on) { m_s.settings.setRunnerOption(m_spec.id, key, on); });
            row->addWidget(toggle, 0, Qt::AlignTop);
            auto* text = new QVBoxLayout;
            text->setSpacing(2);
            text->addWidget(new Label(option.label, Label::Style::Small, Theme::Role::Text));
            auto* what = new Label(option.description, Label::Style::Caption, Theme::Role::Text3);
            what->setWordWrap(true);
            text->addWidget(what);
            row->addLayout(text, 1);
            mid->addLayout(row);
        }
        top->addLayout(mid, 1);

        auto* right = new QVBoxLayout;
        right->setSpacing(8);
        auto* badges = new QHBoxLayout;
        badges->setSpacing(6);
        badges->addStretch(1);
        m_status = new Badge;
        badges->addWidget(m_status);
        m_default = new Badge(QStringLiteral("Default"), Theme::Tone::Neutral);
        m_default->setDotVisible(false);
        badges->addWidget(m_default);
        right->addLayout(badges);
        m_detail = new Label({}, Label::Style::Caption, Theme::Role::Text3);
        m_detail->setAlignment(Qt::AlignRight);
        m_detail->setWordWrap(true);
        m_detail->setFixedWidth(240);
        right->addWidget(m_detail, 0, Qt::AlignRight);
        m_location = new Button(QStringLiteral("Change location..."), Button::Variant::Link, Button::Size::Tiny, QStringLiteral("folder"));
        m_location->setObjectName(QStringLiteral("location-") + spec.id);
        right->addWidget(m_location, 0, Qt::AlignRight);
        m_note = new Label({}, Label::Style::Caption, Theme::Role::Danger);
        m_note->setObjectName(QStringLiteral("runnerNote-") + spec.id);
        m_note->setAlignment(Qt::AlignRight);
        m_note->setWordWrap(true);
        m_note->setFixedWidth(240);
        m_note->hide();
        right->addWidget(m_note, 0, Qt::AlignRight);
        right->addStretch(1);
        auto* actions = new QHBoxLayout;
        actions->setSpacing(8);
        actions->addStretch(1);
        m_makeDefault = new Button(QStringLiteral("Make default"), Button::Variant::Ghost, Button::Size::Small);
        m_makeDefault->setObjectName(QStringLiteral("default-") + spec.id);
        m_makeDefault->setToolTip(QStringLiteral("Games without their own choice run with the default runner"));
        m_uninstall = new Button(QStringLiteral("Uninstall"), Button::Variant::Ghost, Button::Size::Small);
        m_uninstall->setObjectName(QStringLiteral("uninstall-") + spec.id);
        m_cancel = new Button(QStringLiteral("Cancel"), Button::Variant::Secondary, Button::Size::Small);
        m_install = new Button(QStringLiteral("Install"), Button::Variant::Primary, Button::Size::Small, QStringLiteral("download"));
        for (Button* b : {m_makeDefault, m_uninstall, m_cancel, m_install})
            actions->addWidget(b);
        m_install->setObjectName(QStringLiteral("install-") + spec.id);
        right->addLayout(actions);
        top->addLayout(right);

        // the install steps, shown while installing and after a failure
        m_steps = new Card(Card::Kind::Sunken, Metrics::RadiusLg);
        auto* sv = new QVBoxLayout(m_steps);
        sv->setContentsMargins(18, 16, 18, 14);
        sv->setSpacing(10);
        const bool build = spec.install == RunnerSpec::Install::BuildFromSource;
        for (const QString& n : {build ? QStringLiteral("Download pinned source") : QStringLiteral("Download release"),
                                 build ? QStringLiteral("Build from source") : QStringLiteral("Unpack and check"), QStringLiteral("Install")}) {
            m_rows << makeStep(n);
            sv->addWidget(m_rows.last().row);
            m_bars << new Bar;
            m_bars.last()->setContentsMargins(24, 0, 0, 0);
            auto* barRow = new QHBoxLayout;
            barRow->setContentsMargins(24, 0, 0, 0);
            barRow->addWidget(m_bars.last());
            sv->addLayout(barRow);
        }
        m_foot = new QWidget;
        auto* fh = new QHBoxLayout(m_foot);
        fh->setContentsMargins(0, 4, 0, 0);
        fh->setSpacing(6);
        fh->addWidget(new IconView(QStringLiteral("info"), Theme::Role::Text3, 12), 0, Qt::AlignTop);
        auto* footText = new Label(QStringLiteral("Runs in the background; you can keep using XSEmu Launcher."), Label::Style::Caption, Theme::Role::Text3);
        footText->setWordWrap(true);
        fh->addWidget(footText, 1);
        sv->addWidget(m_foot);
        m_error = new Label({}, Label::Style::Small, Theme::Role::Danger);
        m_error->setWordWrap(true);
        m_error->setTextInteractionFlags(Qt::TextSelectableByMouse);
        sv->addWidget(m_error);
        m_log = new Button(QStringLiteral("Open install log"), Button::Variant::Link, Button::Size::Tiny, QStringLiteral("external"));
        m_log->setObjectName(QStringLiteral("installLog-") + spec.id);
        auto* logRow = new QHBoxLayout;
        logRow->addWidget(m_log);
        logRow->addStretch(1);
        sv->addLayout(logRow);
        v->addWidget(m_steps);

        connect(m_install, &Button::clicked, this, [this] { RunnerUi::confirmInstall(window(), m_s, m_spec.id); });
        connect(m_cancel, &Button::clicked, this, [this] { m_s.runners.cancel(); });
        connect(m_location, &Button::clicked, this, [this] { RunnerUi::changeLocation(window(), m_s, m_spec.id); });
        connect(&m_s.runners, &RunnerManager::relocated, this, [this](const QString& id, bool ok, const QString& message) {
            if (id == m_spec.id)
                showNote(ok, message);
        });
        connect(&m_s.runners, &RunnerManager::uninstalled, this, [this](const QString& id, bool ok, const QString& error) {
            if (id == m_spec.id)
                showNote(ok, error);
        });
        connect(m_log, &Button::clicked, this, [this] { QDesktopServices::openUrl(QUrl::fromLocalFile(m_s.runners.installLogPath(m_spec.id))); });
        connect(m_uninstall, &Button::clicked, this, [this] {
            QString error;
            showNote(m_s.runners.uninstall(m_spec.id, &error), error);
        });
        connect(m_makeDefault, &Button::clicked, this, [this] { m_s.settings.setDefaultRunner(m_spec.id); });
        refresh();
    }

    void refresh()
    {
        using S = RunnerManager::State;
        const RunnerManager::Status st = m_s.runners.status(m_spec.id);
        RunnerUi::updateStatusBadge(m_status, st);
        m_default->setVisible(m_s.settings.defaultRunner() == m_spec.id);
        m_recommended->setVisible(!m_spec.isBuiltIn() && RunnerUi::recommendation(m_s).runnerId == m_spec.id);
        m_detail->setText(RunnerUi::statusDetail(m_s, m_spec, st));
        m_detail->setVisible(st.state != S::Failed);
        const bool leftOver = st.state == S::Failed && !st.installedVersion.isEmpty();
        m_uninstall->setVisible(st.state == S::Installed || st.state == S::UpdateAvailable || leftOver);
        const QString busy = m_s.runners.busyReason(m_spec.id);
        m_uninstall->setEnabled(busy.isEmpty());
        m_uninstall->setToolTip(busy);
        m_makeDefault->setVisible(m_s.runners.isRunnable(m_spec.id) && m_s.settings.defaultRunner() != m_spec.id);
        m_cancel->setVisible(st.state == S::Installing);
        m_location->setVisible(!m_spec.isBuiltIn()
                               && (st.state == S::NotInstalled || st.state == S::Failed || st.state == S::Installed || st.state == S::UpdateAvailable));
        m_location->setToolTip(QStringLiteral("In %1. Keep its source, build, program and cache in another folder or drive.").arg(QDir::toNativeSeparators(m_spec.dataDir())));
        m_install->setVisible(st.state == S::NotInstalled || st.state == S::UpdateAvailable || st.state == S::Failed);
        m_install->setEnabled(m_s.runners.busyId().isEmpty());
        m_install->setText(st.state == S::UpdateAvailable ? QStringLiteral("Update")
                           : leftOver                      ? QStringLiteral("Reinstall")
                           : st.state == S::Failed         ? QStringLiteral("Try again")
                                                           : QStringLiteral("Install"));
        m_install->setIconName(st.state == S::UpdateAvailable || st.state == S::Failed ? QStringLiteral("refresh") : QStringLiteral("download"));

        const bool failed = st.state != S::Installing && !st.error.isEmpty();
        m_steps->setVisible(st.state == S::Installing || failed);
        m_error->setVisible(failed);
        m_error->setText(st.state == S::Failed ? st.error : QStringLiteral("The last update failed: %1").arg(st.error));
        m_log->setVisible((st.state == S::Installing || failed) && QFileInfo::exists(m_s.runners.installLogPath(m_spec.id)));
        m_foot->setVisible(st.state == S::Installing);
        const int phase = st.state == S::Failed ? m_lastPhase : st.progress.phase;
        if (st.state == S::Installing)
            m_lastPhase = phase;
        for (int i = 0; i < m_rows.size(); ++i) {
            const bool done = i < phase;
            const bool active = i == phase;
            m_rows[i].done->setVisible(done);
            m_rows[i].spin->setVisible(active && st.state == S::Installing);
            m_rows[i].wait->setVisible(!done && !(active && st.state == S::Installing));
            m_rows[i].name->setRole(done || active ? Theme::Role::Text : Theme::Role::Text3);
            m_bars[i]->setVisible(active && st.state == S::Installing);
            if (!active || st.state != S::Installing) {
                if (i == 0)
                    m_rows[i].detail->setText(done ? m_spec.pinnedVersion() : QString());
                else if (i == 2)
                    m_rows[i].detail->setText(QStringLiteral("In ") + QDir::toNativeSeparators(QFileInfo(m_spec.dataDir()).path()));
                else if (!done)
                    m_rows[i].detail->clear();
                continue;
            }
            const RunnerManager::Progress& p = st.progress;
            QString text = p.step;
            if (p.bytes && p.done >= 0)
                text = p.total > 0 ? QStringLiteral("%1 of %2").arg(Format::bytes(p.done), Format::bytes(p.total)) : Format::bytes(p.done);
            else if (p.step == QLatin1String("Building") && p.done >= 0)
                text = QStringLiteral("Compiled %1 crates").arg(p.done);
            else if (p.total > 0)
                text = QStringLiteral("%1 \u00B7 %2 of %3").arg(p.step).arg(p.done).arg(p.total);
            if (!p.detail.isEmpty())
                text += QStringLiteral(" \u00B7 ") + p.detail;
            m_rows[i].detail->setText(text.left(110));
            m_bars[i]->setFraction(p.total > 0 && p.done >= 0 ? double(p.done) / double(p.total) : -1);
        }
    }

private:
    // what the last move or uninstall left to say; an error in danger
    void showNote(bool ok, const QString& message)
    {
        m_note->setText(message);
        m_note->setRole(ok ? Theme::Role::Text3 : Theme::Role::Danger);
        m_note->setVisible(!message.isEmpty());
    }

    Services& m_s;
    const RunnerSpec& m_spec;
    Badge* m_recommended;
    Badge* m_status;
    Badge* m_default;
    Label* m_detail;
    Button* m_makeDefault;
    Button* m_uninstall;
    Button* m_cancel;
    Button* m_install;
    Button* m_location;
    Label* m_note;
    Card* m_steps;
    QList<StepRow> m_rows;
    QList<Bar*> m_bars;
    QWidget* m_foot;
    Label* m_error;
    Button* m_log;
    int m_lastPhase = 0;
};

RunnersPage::RunnersPage(Services& services, QWidget* parent)
    : QWidget(parent)
    , m_s(services)
{
    const PageScaffold page = makePageScaffold(this, QStringLiteral("Runners"));
    page.header->setSubtitle(QStringLiteral("Programs that can run your games"));
    auto* gpu = new Card(Card::Kind::Surface, Metrics::RadiusLg);
    gpu->setObjectName(QStringLiteral("detectedGpu"));
    auto* gl = new QHBoxLayout(gpu);
    gl->setContentsMargins(12, 7, 14, 7);
    gl->setSpacing(8);
    gl->addWidget(new IconView(QStringLiteral("chip"), Theme::Role::Text3, 14));
    gl->addWidget(new Label(QStringLiteral("Detected GPU"), Label::Style::Small, Theme::Role::Text2));
    auto* gpuName = new Label(m_s.hardware.gpuLabel(), Label::Style::SmallStrong);
    gl->addWidget(gpuName);
    page.header->addAction(gpu);

    m_list = new QVBoxLayout;
    m_list->setSpacing(12);
    page.content->addLayout(m_list);
    for (const RunnerSpec& r : Runners::curated()) {
        auto* card = new RunnerCard(m_s, r);
        m_cards.insert(r.id, card);
    }
    const Runners::Recommendation rec = RunnerUi::recommendation(m_s);
    auto* why = new Label(rec.reason, Label::Style::Caption, Theme::Role::Text2);
    why->setObjectName(QStringLiteral("recommendationReason"));
    why->setWordWrap(true);
    page.content->addSpacing(12);
    page.content->addWidget(why);
    auto* foot = new Label(QStringLiteral("Pick a runner per game in its details. Community runners are projects by their authors, run as separate programs; XSEmu Launcher installs the pinned version and never updates past it."),
                           Label::Style::Caption, Theme::Role::Text3);
    foot->setWordWrap(true);
    page.content->addSpacing(4);
    page.content->addWidget(foot);
    page.content->addStretch(1);
    reorder();

    connect(&m_s.runners, &RunnerManager::changed, this, [this](const QString& id) {
        if (installingId() != m_busy)
            reorder();
        for (RunnerCard* c : std::as_const(m_cards))
            c->refresh();
        Q_UNUSED(id);
    });
    // a game or a pre-cache starting or ending changes what a runner may do (Uninstall waits for it)
    const auto refreshCards = [this] {
        for (RunnerCard* c : std::as_const(m_cards))
            c->refresh();
    };
    connect(&m_s.emulator, &EmulatorController::stateChanged, this, refreshCards);
    connect(&m_s.warmup, &WarmupController::stateChanged, this, refreshCards);
    connect(&m_s.settings, &AppSettings::changed, this, [this] {
        for (RunnerCard* c : std::as_const(m_cards))
            c->refresh();
    });
    connect(&m_s.hardware, &HardwareMonitor::changed, this, [this, gpuName, why] {
        gpuName->setText(m_s.hardware.gpuLabel());
        why->setText(RunnerUi::recommendation(m_s).reason);
        reorder();
        for (RunnerCard* c : std::as_const(m_cards))
            c->refresh();
    });
}

QString RunnersPage::installingId() const
{
    for (const RunnerSpec& r : Runners::curated())
        if (m_s.runners.status(r.id).state == RunnerManager::State::Installing)
            return r.id;
    return {};
}

void RunnersPage::reorder()
{
    m_busy = installingId();
    for (RunnerCard* c : std::as_const(m_cards))
        m_list->removeWidget(c);
    for (const RunnerSpec* r : RunnerUi::displayOrder(m_s, m_busy))
        m_list->addWidget(m_cards.value(r->id));
}
