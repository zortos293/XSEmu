#include "Sections.h"
#include "SettingsKit.h"

#include "app/Services.h"
#include "core/CoverArtService.h"
#include "core/DiskSpace.h"
#include "core/Format.h"
#include "core/Paths.h"
#include "widgets/Button.h"
#include "widgets/Icon.h"
#include "widgets/InfoNote.h"
#include "widgets/Text.h"
#include "widgets/Toggle.h"

#include <QDateTime>
#include <QDir>
#include <QFileDialog>
#include <QHBoxLayout>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLineEdit>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QPainter>
#include <QPainterPath>
#include <QPointer>
#include <QVBoxLayout>

namespace {

// The bordered list of folders (Settings: Game folders).
class FolderList : public QWidget {
public:
    FolderList()
    {
        m_rows = new QVBoxLayout(this);
        m_rows->setContentsMargins(1, 1, 1, 1);
        m_rows->setSpacing(0);
        connect(&Theme::instance(), &Theme::changed, this, qOverload<>(&QWidget::update));
    }
    QVBoxLayout* rows() const { return m_rows; }

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        const QRectF r = QRectF(rect()).adjusted(0.5, 0.5, -0.5, -0.5);
        p.setPen(QPen(Theme::color(Theme::Role::Border), 1));
        p.setBrush(Theme::color(Theme::Role::Surface));
        p.drawRoundedRect(r, 11.5, 11.5);
        for (int i = 0; i + 1 < m_rows->count(); ++i) {
            if (QWidget* w = m_rows->itemAt(i)->widget())
                p.fillRect(QRect(1, w->geometry().bottom(), width() - 2, 1), Theme::color(Theme::Role::Border));
        }
    }

private:
    QVBoxLayout* m_rows;
};

bool inFolder(const QString& path, const QString& folder)
{
    const QString f = QDir::cleanPath(QDir(folder).absolutePath());
    const QString p = QDir::cleanPath(QDir(path).absolutePath());
    return p == f || p.startsWith(f + QLatin1Char('/'));
}

}

// Folders: AppSettings::gameFolders, scanned and watched by Library. Rescan: Library::rescan. The
// library always rescans at start and watches its folders, so those two rows say so. Cover art: the
// SteamGridDB key and the art lookups switch (CoverArtService); Test does a real search request.
QWidget* SettingsSections::gameFolders(Services& s, std::function<void()> addFolder)
{
    const SettingsKit::Section sec = SettingsKit::makeSection(
        QStringLiteral("Game folders"), QStringLiteral("XSEmu Launcher scans these folders for extracted titles: one game per subfolder, each with a MicrosoftGame.config."));
    sec.caps(QStringLiteral("Folders"));
    sec.body->addSpacing(8);
    auto* list = new FolderList;
    sec.add(list);
    auto* add = new Button(QStringLiteral("Add folder..."), Button::Variant::Secondary, Button::Size::Small);
    auto* addRow = new QWidget;
    auto* ah = new QHBoxLayout(addRow);
    ah->setContentsMargins(0, 10, 0, 0);
    ah->addWidget(add);
    ah->addStretch(1);
    sec.add(addRow);
    QObject::connect(add, &Button::clicked, addRow, [addFolder] { addFolder(); });

    auto scannedAt = std::make_shared<QDateTime>(QDateTime::currentDateTime());
    const auto refresh = [&s, list, scannedAt] {
        QVBoxLayout* rows = list->rows();
        while (QLayoutItem* item = rows->takeAt(0)) {
            delete item->widget();
            delete item;
        }
        const QStringList folders = s.settings.gameFolders();
        list->setVisible(!folders.isEmpty());
        for (const QString& folder : folders) {
            int count = 0;
            for (const Game& g : s.library.games())
                count += inFolder(g.folder, folder) ? 1 : 0;
            auto* row = new QWidget;
            auto* h = new QHBoxLayout(row);
            h->setContentsMargins(16, 12, 12, 12);
            h->setSpacing(12);
            h->addWidget(new IconView(QStringLiteral("folder"), Theme::Role::Text2, 16));
            auto* text = new QVBoxLayout;
            text->setSpacing(2);
            auto* name = new Label(QDir::toNativeSeparators(folder), Label::Style::Mono, Theme::Role::Text);
            name->setFont(Theme::mono(12));
            text->addWidget(name);
            const bool exists = QDir(folder).exists();
            const QString when = s.library.isScanning() ? QStringLiteral("scanning...") : QStringLiteral("scanned %1").arg(Format::ago(*scannedAt));
            const QString facts = !exists ? QStringLiteral("folder not found")
                                : count == 0 ? QStringLiteral("0 games · no MicrosoftGame.config found")
                                             : QStringLiteral("%1 · %2").arg(count == 1 ? QStringLiteral("1 game") : QStringLiteral("%1 games").arg(count), when);
            text->addWidget(new Label(facts, Label::Style::Caption, count == 0 ? Theme::Role::Warning : Theme::Role::Text3));
            h->addLayout(text, 1);
            auto* rescan = new Button(QStringLiteral("Rescan"), Button::Variant::Ghost, Button::Size::Tiny);
            auto* remove = new Button(QStringLiteral("Remove"), Button::Variant::Ghost, Button::Size::Tiny);
            h->addWidget(rescan);
            h->addWidget(remove);
            QObject::connect(rescan, &Button::clicked, row, [&s] { s.library.rescan(); });
            QObject::connect(remove, &Button::clicked, row, [&s, folder] {
                QStringList f = s.settings.gameFolders();
                f.removeAll(folder);
                s.settings.setGameFolders(f);
            });
            rows->addWidget(row);
        }
        list->update();
    };
    refresh();
    QObject::connect(&s.library, &Library::gamesChanged, list, [refresh, scannedAt] {
        *scannedAt = QDateTime::currentDateTime();
        refresh();
    });
    QObject::connect(&s.library, &Library::scanStarted, list, refresh);
    QObject::connect(&s.settings, &AppSettings::changed, list, [refresh](const QString& key) {
        if (key == QLatin1String(AppSettings::Key::GameFolders))
            refresh();
    });

    sec.caps(QStringLiteral("Scanning"));
    auto* atStart = new Toggle;
    atStart->setChecked(true);
    sec.row(QStringLiteral("Rescan when XSEmu Launcher starts"), QStringLiteral("Picks up games you copied in while it was closed."), atStart)
        ->setUnavailable(QStringLiteral("Always on in this launcher."));
    auto* watch = new Toggle;
    watch->setChecked(true);
    sec.row(QStringLiteral("Watch folders for changes"), QStringLiteral("Adds and removes games without a restart."), watch)
        ->setUnavailable(QStringLiteral("Always on in this launcher."));

    sec.caps(QStringLiteral("Runners"));
    auto* runnersRow = new QWidget;
    runnersRow->setObjectName(QStringLiteral("runnersFolder"));
    auto* rh = new QHBoxLayout(runnersRow);
    rh->setContentsMargins(0, 12, 0, 6);
    rh->setSpacing(8);
    auto* runnersPath = new SettingsKit::PathField;
    auto* runnersReset = new Button(QStringLiteral("Use default"), Button::Variant::Ghost, Button::Size::Small);
    auto* runnersChange = new Button(QStringLiteral("Change..."), Button::Variant::Secondary, Button::Size::Small);
    runnersChange->setObjectName(QStringLiteral("runnersFolderChange"));
    rh->addWidget(runnersPath, 1);
    rh->addWidget(runnersReset);
    rh->addWidget(runnersChange);
    sec.add(runnersRow);
    auto* runnersHelp = new Label({}, Label::Style::Caption, Theme::Role::Text3);
    runnersHelp->setWordWrap(true);
    sec.add(runnersHelp);
    auto* runnersSpace = new InfoNote({}, Theme::Tone::Neutral, QStringLiteral("info"));
    runnersSpace->setObjectName(QStringLiteral("runnersFolderSpace"));
    sec.add(runnersSpace, 10);
    const auto refreshRunners = [&s, runnersPath, runnersReset, runnersChange, runnersHelp, runnersSpace] {
        const QString dir = Paths::runnersDir();
        runnersPath->setPath(QDir::toNativeSeparators(dir));
        const bool fromEnv = Paths::runnersDirFromEnvironment();
        runnersChange->setEnabled(!fromEnv);
        runnersReset->setVisible(!fromEnv && !s.settings.runnersFolder().isEmpty());
        runnersHelp->setText(fromEnv ? QStringLiteral("Set by XSEMU_RUNNERS_DIR in the environment.")
                                     : QStringLiteral("Community runners are downloaded and built here, with the Rust toolchain their builds need. Runners already installed stay where they are; use Change location on the Runners page to move one."));
        const DiskSpace::Check space = DiskSpace::checkRunnerFolder(dir);
        runnersSpace->setText(space.text);
        runnersSpace->setTone(space.low ? Theme::Tone::Warning : Theme::Tone::Neutral);
    };
    refreshRunners();
    const auto chooseRunners = [&s, runnersSpace](const QString& dir) {
        QString error;
        if (!s.runners.setRunnersFolder(dir, &error)) {
            runnersSpace->setText(error);
            runnersSpace->setTone(Theme::Tone::Danger);
        }
    };
    QObject::connect(runnersChange, &Button::clicked, runnersRow, [runnersRow, chooseRunners] {
        const QString dir = QFileDialog::getExistingDirectory(runnersRow->window(), QStringLiteral("Choose a folder for runners"), Paths::runnersDir());
        if (!dir.isEmpty())
            chooseRunners(dir);
    });
    QObject::connect(runnersReset, &Button::clicked, runnersRow, [chooseRunners] { chooseRunners({}); });
    QObject::connect(&s.settings, &AppSettings::changed, runnersRow, [refreshRunners](const QString& key) {
        if (key == QLatin1String(AppSettings::Key::RunnersFolder))
            refreshRunners();
    });
    SettingsKit::onShow(sec.page, refreshRunners);

    sec.caps(QStringLiteral("Cover art"));
    auto* key = new QLineEdit(s.settings.steamGridDbKey());
    key->setProperty("framed", true);
    key->setEchoMode(QLineEdit::Password);
    key->setPlaceholderText(QStringLiteral("API key"));
    key->setFixedWidth(200);
    auto* test = new Button(QStringLiteral("Test"), Button::Variant::Secondary, Button::Size::Small);
    auto* clearKey = new Button(QStringLiteral("Clear"), Button::Variant::Ghost, Button::Size::Small);
    SettingsKit::Row* keyRow = sec.row(QStringLiteral("SteamGridDB API key"),
                                       QStringLiteral("Covers are fetched from SteamGridDB with your own key and cached on this PC. XSEmu ships no game art."));
    keyRow->addControl(key);
    keyRow->addControl(test);
    keyRow->addControl(clearKey);
    auto* result = new Label({}, Label::Style::Caption, Theme::Role::Text3);
    result->setWordWrap(true);
    result->hide();
    sec.add(result);
    QObject::connect(key, &QLineEdit::editingFinished, key, [&s, key] { s.settings.setSteamGridDbKey(key->text()); });
    QObject::connect(clearKey, &Button::clicked, key, [&s, key, result] {
        key->clear();
        s.settings.setSteamGridDbKey({});
        result->hide();
    });
    auto* net = new QNetworkAccessManager(key);
    QObject::connect(test, &Button::clicked, key, [&s, key, test, result, net] {
        const QString k = key->text().trimmed();
        if (k.isEmpty()) {
            result->setText(QStringLiteral("Enter a key first."));
            result->setRole(Theme::Role::Warning);
            result->show();
            return;
        }
        s.settings.setSteamGridDbKey(k);
        QNetworkRequest req(CoverArtService::searchUrl(QStringLiteral("test")));
        req.setRawHeader("Authorization", "Bearer " + k.toUtf8());
        req.setTransferTimeout(10000);
        test->setEnabled(false);
        result->setText(QStringLiteral("Asking SteamGridDB..."));
        result->setRole(Theme::Role::Text3);
        result->show();
        QNetworkReply* reply = net->get(req);
        QObject::connect(reply, &QNetworkReply::finished, test, [reply, test, result] {
            reply->deleteLater();
            test->setEnabled(true);
            const int code = reply->attribute(QNetworkRequest::HttpStatusCodeAttribute).toInt();
            const QJsonObject body = QJsonDocument::fromJson(reply->readAll()).object();
            if (code == 200 && body.value(QStringLiteral("success")).toBool()) {
                result->setText(QStringLiteral("The key works: SteamGridDB answered a search."));
                result->setRole(Theme::Role::Success);
            } else if (code == 401 || code == 403) {
                result->setText(QStringLiteral("SteamGridDB refused this key."));
                result->setRole(Theme::Role::Danger);
            } else {
                result->setText(QStringLiteral("SteamGridDB could not be reached: %1").arg(reply->errorString()));
                result->setRole(Theme::Role::Danger);
            }
        });
    });
    auto* lookups = new Toggle;
    lookups->setChecked(s.settings.artLookups());
    QObject::connect(lookups, &Toggle::toggled, &s.settings, &AppSettings::setArtLookups);
    sec.row(QStringLiteral("Art lookups"), QStringLiteral("Off by default. On, with your key: XSEmu asks SteamGridDB for cover and banner art. Off: it never contacts SteamGridDB and shows the No cover art tile."), lookups);
    sec.finish();
    return sec.page;
}
