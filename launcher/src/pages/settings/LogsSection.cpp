#include "Sections.h"
#include "SettingsKit.h"

#include "app/Services.h"
#include "core/Format.h"
#include "core/LaunchConfig.h"
#include "core/Paths.h"
#include "widgets/Button.h"
#include "widgets/SegmentedControl.h"
#include "widgets/Text.h"
#include "widgets/Toggle.h"

#include <QDesktopServices>
#include <QDir>
#include <QHBoxLayout>
#include <QPainter>
#include <QUrl>
#include <QVBoxLayout>

#include <algorithm>

namespace {

struct Dump {
    QString titleDir;
    QFileInfo file;
};

// Every dump XCL_FULL_DUMP left: <data root>\<title>\logs\dumps\xcl-{full|small}-<pid>.dmp, newest first.
QList<Dump> findDumps()
{
    QList<Dump> out;
    const QDir root(Paths::dataRoot());
    for (const QString& title : root.entryList(QDir::Dirs | QDir::NoDotAndDotDot)) {
        const QDir dumps(LaunchConfig::crashDumpDir(title));
        for (const QFileInfo& fi : dumps.entryInfoList({QStringLiteral("*.dmp")}, QDir::Files))
            out.append({title, fi});
    }
    std::sort(out.begin(), out.end(), [](const Dump& a, const Dump& b) { return a.file.lastModified() > b.file.lastModified(); });
    return out;
}

class Dot : public QWidget {
public:
    Dot() { setFixedSize(6, 6); }

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        p.setPen(Qt::NoPen);
        p.setBrush(Theme::color(Theme::Role::Danger));
        p.drawEllipse(rect());
    }
};

QString when(const QDateTime& t)
{
    const QDate today = QDate::currentDate();
    const QString time = t.toString(QStringLiteral("HH:mm"));
    if (t.date() == today)
        return QStringLiteral("Today ") + time;
    if (t.date() == today.addDays(-1))
        return QStringLiteral("Yesterday ") + time;
    return t.toString(QStringLiteral("d MMM yyyy HH:mm"));
}

}

// log_level: a launcher default written into each title's xcl.toml at launch (info or debug). Crash
// dumps: XCL_FULL_DUMP=<title>\logs\dumps in the game's environment (LaunchConfig). The logs live in
// <data root>\<title>\logs (crates/xcl/src/config.rs Boot::logs_dir).
QWidget* SettingsSections::logs(Services& s)
{
    const SettingsKit::Section sec = SettingsKit::makeSection(
        QStringLiteral("Logs & crash reports"), QStringLiteral("Everything stays on this PC. XSEmu Launcher never uploads logs or crash reports."));
    sec.caps(QStringLiteral("Logs"));
    auto* pathRow = new QWidget;
    auto* ph = new QHBoxLayout(pathRow);
    ph->setContentsMargins(0, 12, 0, 12);
    ph->setSpacing(8);
    ph->addWidget(new SettingsKit::PathField(QDir(Paths::dataRoot()).filePath(QStringLiteral("<game>/logs"))), 1);
    auto* open = new Button(QStringLiteral("Open folder"), Button::Variant::Secondary, Button::Size::Small);
    ph->addWidget(open);
    QObject::connect(open, &Button::clicked, pathRow, [] {
        QDir().mkpath(Paths::dataRoot());
        QDesktopServices::openUrl(QUrl::fromLocalFile(Paths::dataRoot()));
    });
    sec.add(pathRow);
    auto* detail = new SegmentedControl({QStringLiteral("Normal"), QStringLiteral("Verbose")});
    detail->setCurrentIndex(s.settings.xclDefault(QStringLiteral("log_level")).toString() == QLatin1String("debug") ? 1 : 0);
    QObject::connect(detail, &SegmentedControl::currentChanged, sec.page, [&s](int i) {
        s.settings.setXclDefault(QStringLiteral("log_level"), i == 1 ? QStringLiteral("debug") : QStringLiteral("info"));
    });
    sec.row(QStringLiteral("Log detail"), QStringLiteral("Verbose writes debug lines too. Logs grow much faster."), detail);
    auto* keep = SettingsKit::dropdown({{QStringLiteral("Last 2 runs"), 2}}, 2);
    sec.row(QStringLiteral("Keep logs for"), {}, keep)->setUnavailable(QStringLiteral("XSEmu Core keeps the newest log and the one before it (xcl.log, xcl.prev.log)."));

    sec.caps(QStringLiteral("Crash reports"));
    auto* dumpsOn = new Toggle;
    dumpsOn->setChecked(s.settings.crashDumps());
    QObject::connect(dumpsOn, &Toggle::toggled, &s.settings, &AppSettings::setCrashDumps);
    sec.row(QStringLiteral("Write a crash report when a game crashes"),
            QStringLiteral("A minidump of the game process in its logs folder, next to the log of that run."), dumpsOn);
    auto* list = new QWidget;
    auto* lv = new QVBoxLayout(list);
    lv->setContentsMargins(0, 0, 0, 0);
    lv->setSpacing(0);
    sec.add(list);
    const auto refresh = [&s, lv] {
        while (QLayoutItem* item = lv->takeAt(0)) {
            delete item->widget();
            delete item;
        }
        const QList<Dump> dumps = findDumps();
        if (dumps.isEmpty()) {
            auto* none = new Label(QStringLiteral("No crash reports yet."), Label::Style::Small, Theme::Role::Text3);
            none->setContentsMargins(0, 14, 0, 0);
            lv->addWidget(none);
        }
        for (const Dump& d : dumps.mid(0, 10)) {
            QString title = d.titleDir;
            for (const Game& g : s.library.games()) {
                if (g.titleId() == d.titleDir)
                    title = g.title();
            }
            auto* row = new SettingsKit::Row(title, QStringLiteral("%1 · %2 · %3").arg(when(d.file.lastModified()),
                                                                                          d.file.fileName().contains(QLatin1String("-full-")) ? QStringLiteral("full dump") : QStringLiteral("small dump"),
                                                                                          Format::bytes(d.file.size())));
            auto* dotHost = new QWidget;
            auto* dh = new QHBoxLayout(dotHost);
            dh->setContentsMargins(0, 6, 0, 0);
            dh->addWidget(new Dot);
            row->layout()->setContentsMargins(0, 10, 0, 10);
            row->layout()->setSpacing(14);
            static_cast<QHBoxLayout*>(row->layout())->insertWidget(0, dotHost, 0, Qt::AlignTop);
            auto* show = new Button(QStringLiteral("Show in folder"), Button::Variant::Link, Button::Size::Tiny);
            const QString dir = d.file.absolutePath();
            QObject::connect(show, &Button::clicked, row, [dir] { QDesktopServices::openUrl(QUrl::fromLocalFile(dir)); });
            row->addControl(show);
            lv->addWidget(row);
        }
        if (QLayoutItem* last = lv->itemAt(lv->count() - 1); last && qobject_cast<SettingsKit::Row*>(last->widget()))
            static_cast<SettingsKit::Row*>(last->widget())->setBorder(false);
    };
    refresh();
    QObject::connect(&s.emulator, &EmulatorController::exited, list, refresh);
    SettingsKit::onShow(sec.page, refresh);
    QObject::connect(&s.library, &Library::gamesChanged, list, refresh);
    QObject::connect(&s.settings, &AppSettings::changed, sec.page, [&s, dumpsOn](const QString& key) {
        if (key == QLatin1String(AppSettings::Key::CrashDumps))
            dumpsOn->setChecked(s.settings.crashDumps());
    });
    sec.finish();
    return sec.page;
}
