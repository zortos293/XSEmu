#pragma once

#include <QWidget>

class QAbstractButton;
class PageHeader;
class QStackedWidget;
class Services;

// Settings (Settings pages): a 208 px sub-navigation of ten sections beside the section's page.
// The sections live in pages/settings/, one file each.
class SettingsPage : public QWidget {
    Q_OBJECT
public:
    enum class Section { Console, Network, Shaders, RunProfile, Overlay, GameFolders, Input, Graphics, Logs, Updates, About };
    explicit SettingsPage(Services& services, QWidget* parent = nullptr);
    void showSection(Section section);
    QWidget* sectionPage(Section section) const;

signals:
    void addGamesRequested();

private:
    QList<QAbstractButton*> m_nav;
    QStringList m_names;
    PageHeader* m_header;
    QStackedWidget* m_stack;
};
