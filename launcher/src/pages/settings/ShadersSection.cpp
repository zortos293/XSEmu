#include "Sections.h"
#include "SettingsKit.h"

#include "app/Services.h"
#include "core/Format.h"
#include "core/LaunchConfig.h"
#include "core/Paths.h"
#include "core/ShaderCacheInfo.h"
#include "widgets/Button.h"
#include "widgets/Dialog.h"
#include "widgets/InfoNote.h"
#include "widgets/Text.h"
#include "widgets/Toggle.h"

#include <QDesktopServices>
#include <QDir>
#include <QCoreApplication>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QPointer>
#include <QThreadPool>
#include <QUrl>
#include <QVBoxLayout>

namespace {

// A folder the cache may move to: empty, or one that already holds the emulator's cache folders.
bool isShaderCacheFolder(const QDir& dir)
{
    if (!dir.exists())
        return false;
    if (dir.isRoot() || QDir::cleanPath(dir.absolutePath()) == QDir::cleanPath(QDir::homePath()))
        return false;
    if (dir.isEmpty(QDir::AllEntries | QDir::NoDotAndDotDot | QDir::Hidden))
        return true;
    for (const char* sub : {"pipelines", "driver", "translated"}) {
        if (QFileInfo(dir.filePath(QLatin1String(sub))).isDir())
            return true;
    }
    return false;
}

}

// Warm-up when the translator changes: XSEmu reports no translator id to the launcher (it is a
// build-time constant, XCL_TRANSLATOR_SOURCE_ID), so it is shown unavailable. Giant compiles: no knob
// (the admission rule is shown instead). Cache folder: XCL_DXIL_CACHE (LaunchConfig). Clear: deletes
// the emulator's own entries in it after the confirm dialog (ShaderCacheInfo::clear) and names the rest.
QWidget* SettingsSections::shaders(Services& s)
{
    const SettingsKit::Section sec = SettingsKit::makeSection(
        QStringLiteral("Shaders"), QStringLiteral("XSEmu Core translates console shaders to DXIL and keeps the result, so each one compiles once."));
    sec.caps(QStringLiteral("Warm-up"));
    auto* onChange = new Toggle;
    sec.row(QStringLiteral("Warm up when the translator changes"),
            QStringLiteral("Runs the blocking warm-up before launch after an XSEmu Core update. Skipped when nothing changed."), onChange)
        ->setUnavailable(QStringLiteral("XSEmu Core doesn't report its translator version to the launcher yet."));
    sec.row(QStringLiteral("RAM floor"), QStringLiteral("Compiles wait while less than this is free, in the warm-up and in games."),
            SettingsKit::status(QStringLiteral("1 GB free"), Theme::Role::Text));
    sec.add(new InfoNote(QStringLiteral("XSEmu Core uses as much memory as it needs. Each compile starts when its estimated peak fits in free RAM minus the 1 GB floor, and waits in a queue until memory frees otherwise. There is no fixed limit on giant compiles.")), 12);

    sec.caps(QStringLiteral("Cache"));
    auto* pathRow = new QWidget;
    auto* ph = new QHBoxLayout(pathRow);
    ph->setContentsMargins(0, 12, 0, 12);
    ph->setSpacing(8);
    auto* path = new SettingsKit::PathField;
    auto* change = new Button(QStringLiteral("Change..."), Button::Variant::Secondary, Button::Size::Small);
    auto* reset = new Button(QStringLiteral("Use default"), Button::Variant::Ghost, Button::Size::Small);
    ph->addWidget(path, 1);
    ph->addWidget(reset);
    ph->addWidget(change);
    sec.add(pathRow);
    auto* pathNote = new Label({}, Label::Style::Caption, Theme::Role::Text3);
    pathNote->setWordWrap(true);
    sec.add(pathNote);

    auto* size = new Label({}, Label::Style::SmallStrong);
    auto* open = new Button(QStringLiteral("Open folder"), Button::Variant::Secondary, Button::Size::Small);
    auto* clear = new Button(QStringLiteral("Clear cache..."), Button::Variant::DangerSecondary, Button::Size::Small);
    clear->setObjectName(QStringLiteral("clearShaderCache"));
    SettingsKit::Row* sizeRow = sec.row(QStringLiteral("Cache size"));
    sizeRow->addControl(size);
    sizeRow->addControl(open);
    sizeRow->addControl(clear);

    const QPointer<QWidget> page = sec.page;
    const auto refresh = [&s, path, pathNote, change, reset, size, sizeRow, clear] {
        const ShaderCacheInfo& info = s.shaderCache.snapshot();
        const QString dir = Paths::dxilCacheDir();
        path->setPath(dir);
        const bool fromEnv = LaunchConfig::shaderCacheDirFromEnvironment();
        change->setEnabled(!fromEnv);
        reset->setVisible(!fromEnv && !s.settings.shaderCacheDirOverride().isEmpty());
        pathNote->setText(fromEnv ? QStringLiteral("Set by XCL_DXIL_CACHE in the environment.") : QString());
        pathNote->setVisible(fromEnv);
        if (!s.shaderCache.hasSnapshot() || info.dxilDir != dir) {
            sizeRow->setHelp(QStringLiteral("Reading the cache..."));
            size->setText(QStringLiteral("Measuring..."));
            clear->setEnabled(false);
            return;
        }
        const QString pipelines = info.pipelines < 0 ? QStringLiteral("No pipelines recorded yet")
                                                     : QStringLiteral("%1 pipelines recorded").arg(Format::count(info.pipelines));
        const QString blobs = info.blobs < 0 ? QString() : QStringLiteral(", %1 driver blobs").arg(Format::count(info.blobs));
        sizeRow->setHelp(pipelines + blobs + QStringLiteral(", shared by all your games."));
        size->setText(info.exists ? Format::bytes(info.totalBytes) : QStringLiteral("Empty"));
        const QString blocked = s.gate.isFree() ? QString() : s.gate.blockedReason(ProcessGate::Action::StartWarmup);
        clear->setEnabled(info.exists && blocked.isEmpty());
        clear->setToolTip(blocked);
    };
    refresh();

    QObject::connect(open, &Button::clicked, sec.page, [] {
        const QString dir = Paths::dxilCacheDir();
        QDir().mkpath(dir);
        QDesktopServices::openUrl(QUrl::fromLocalFile(dir));
    });
    QObject::connect(change, &Button::clicked, sec.page, [&s, page, pathNote] {
        const QString dir = QFileDialog::getExistingDirectory(page->window(), QStringLiteral("Choose a folder for the shader cache"));
        if (dir.isEmpty())
            return;
        if (!isShaderCacheFolder(QDir(dir))) {
            pathNote->setText(QStringLiteral("Pick an empty folder or an existing XSEmu Core shader cache."));
            pathNote->show();
            return;
        }
        s.settings.setShaderCacheDirOverride(dir);
    });
    QObject::connect(reset, &Button::clicked, sec.page, [&s] { s.settings.setShaderCacheDirOverride({}); });
    QObject::connect(clear, &Button::clicked, sec.page, [&s, page] {
        const ShaderCacheInfo& info = s.shaderCache.snapshot();
        auto* d = new Dialog(page->window(), 460);
        d->setIcon(QStringLiteral("trash"), int(Theme::Tone::Danger));
        d->setTitle(QStringLiteral("Clear the shader cache?"));
        const QString what = info.pipelines > 0 ? QStringLiteral("%1 pipelines").arg(Format::count(info.pipelines)) : QStringLiteral("every compiled shader");
        const QString amount = info.totalBytes >= 0 ? QStringLiteral(" (%1)").arg(Format::bytes(info.totalBytes)) : QString();
        auto* text = new Label(QStringLiteral("This deletes %1%2 for all your games. Each game stalls on its next run until you warm it up again.").arg(what, amount),
                               Label::Style::Body, Theme::Role::Text2);
        text->setWordWrap(true);
        d->body()->addWidget(text);
        auto* driver = new Label(QStringLiteral("The graphics driver's own cache isn't touched."), Label::Style::Small, Theme::Role::Text3);
        driver->setWordWrap(true);
        d->body()->addWidget(driver);
        auto* cancel = new Button(QStringLiteral("Cancel"), Button::Variant::Secondary, Button::Size::Large);
        auto* confirm = new Button(QStringLiteral("Clear cache"), Button::Variant::Danger, Button::Size::Large);
        d->addButton(cancel);
        d->addButton(confirm);
        QObject::connect(cancel, &Button::clicked, d, &Dialog::dismiss);
        QObject::connect(confirm, &Button::clicked, d, [&s, d, driver, confirm, cancel] {
            QString error;
            if (!s.gate.isFree())
                error = s.gate.blockedReason(ProcessGate::Action::StartWarmup);
            else if (const ProcessGate::External other = s.gate.external(); other.busy())
                error = other.reason();
            if (!error.isEmpty()) {
                driver->setText(error);
                driver->setRole(Theme::Role::Danger);
                return;
            }
            // deleting a large cache takes a while, so it runs off the UI thread
            confirm->setEnabled(false);
            cancel->setEnabled(false);
            confirm->setText(QStringLiteral("Clearing..."));
            const QPointer<Dialog> dialog = d;
            const QPointer<ShaderCacheMonitor> monitor = &s.shaderCache;
            const QString dir = Paths::dxilCacheDir();
            QThreadPool::globalInstance()->start([dialog, monitor, dir, driver, confirm, cancel] {
                const ShaderCacheInfo::Cleared cleared = ShaderCacheInfo::clear(dir);
                QMetaObject::invokeMethod(qApp, [dialog, monitor, cleared, driver, confirm, cancel] {
                    if (monitor)
                        monitor->refresh();
                    if (!dialog)
                        return;
                    cancel->setEnabled(true);
                    if (!cleared.error.isEmpty()) {
                        driver->setText(cleared.error);
                        driver->setRole(Theme::Role::Danger);
                        confirm->setText(QStringLiteral("Clear cache"));
                        confirm->setEnabled(true);
                        return;
                    }
                    if (!cleared.kept.isEmpty()) {
                        driver->setText(QStringLiteral("Cleared. These aren't XSEmu Core's, so they were kept: %1.").arg(cleared.kept.join(QStringLiteral(", "))));
                        driver->setRole(Theme::Role::Text2);
                        confirm->hide();
                        cancel->setText(QStringLiteral("Close"));
                        return;
                    }
                    dialog->dismiss();
                });
            });
        });
        d->open();
    });
    QObject::connect(&s.settings, &AppSettings::changed, sec.page, [refresh](const QString& key) {
        if (key == QLatin1String(AppSettings::Key::ShaderCacheDir) || key == QLatin1String(AppSettings::Key::RecentWarmups))
            refresh();
    });
    QObject::connect(&s.gate, &ProcessGate::changed, sec.page, refresh);
    QObject::connect(&s.shaderCache, &ShaderCacheMonitor::changed, sec.page, refresh);
    SettingsKit::onShow(sec.page, [&s] { s.shaderCache.refreshIfStale(); });
    sec.finish();
    return sec.page;
}
