#include "SettingsPage.h"

#include "app/Services.h"
#include "settings/Sections.h"
#include "widgets/Icon.h"
#include "widgets/PageHeader.h"
#include "widgets/Text.h"

#include <QAbstractButton>
#include <QButtonGroup>
#include <QHBoxLayout>
#include <QPainter>
#include <QScrollArea>
#include <QStackedWidget>
#include <QVBoxLayout>

namespace {

// The sub-navigation column with its hairline on the right.
class SubNav : public QWidget {
public:
    SubNav()
    {
        setFixedWidth(208);
        connect(&Theme::instance(), &Theme::changed, this, qOverload<>(&QWidget::update));
    }

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.fillRect(QRect(width() - 1, 0, 1, height()), Theme::color(Theme::Role::Border));
    }
};

// A 34 px sub-navigation row: 16 px icon and 13 px label, semibold on the sunken fill when checked.
class SubNavItem : public QAbstractButton {
public:
    SubNavItem(const QString& text, QString icon)
        : m_icon(std::move(icon))
    {
        setText(text);
        setCheckable(true);
        setFixedHeight(34);
        setCursor(Qt::PointingHandCursor);
        connect(&Theme::instance(), &Theme::changed, this, qOverload<>(&QWidget::update));
    }

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        if (isChecked() || underMouse()) {
            p.setPen(Qt::NoPen);
            p.setBrush(Theme::color(isChecked() ? Theme::Role::Sunken : Theme::Role::Hover));
            p.drawRoundedRect(QRectF(rect()), 7, 7);
        }
        const Theme::Role role = isChecked() ? Theme::Role::Text : Theme::Role::Text2;
        Icon::paint(p, QRectF(10, (height() - 16) / 2.0, 16, 16), m_icon, Theme::color(role), 16);
        p.setFont(Theme::font(13, isChecked() ? 600 : 500));
        p.setPen(Theme::color(role));
        p.drawText(rect().adjusted(36, 0, -8, 0), Qt::AlignVCenter | Qt::AlignLeft, text());
    }
    void enterEvent(QEnterEvent*) override { update(); }
    void leaveEvent(QEvent*) override { update(); }

private:
    QString m_icon;
};

}

SettingsPage::SettingsPage(Services& services, QWidget* parent)
    : QWidget(parent)
    , m_header(new PageHeader(QStringLiteral("Settings")))
    , m_stack(new QStackedWidget)
{
    auto* v = new QVBoxLayout(this);
    v->setContentsMargins(0, 0, 0, 0);
    v->setSpacing(0);
    v->addWidget(m_header);

    auto* row = new QHBoxLayout;
    row->setContentsMargins(0, 0, 0, 0);
    row->setSpacing(0);
    auto* nav = new SubNav;
    auto* nv = new QVBoxLayout(nav);
    nv->setContentsMargins(12, 16, 13, 16);
    nv->setSpacing(2);
    auto* group = new QButtonGroup(this);
    const QList<QPair<QString, QString>> items = {
        {QStringLiteral("Console"), QStringLiteral("gamepad")},
        {QStringLiteral("Network"), QStringLiteral("network-off")},
        {QStringLiteral("Shaders"), QStringLiteral("bolt")},
        {QStringLiteral("Run profile"), QStringLiteral("chip")},
        {QStringLiteral("Overlay"), QStringLiteral("overlay")},
        {QStringLiteral("Game folders"), QStringLiteral("folder")},
        {QStringLiteral("Input"), QStringLiteral("keyboard")},
        {QStringLiteral("Graphics"), QStringLiteral("monitor")},
        {QStringLiteral("Logs & crash reports"), QStringLiteral("file")},
        {QStringLiteral("Updates"), QStringLiteral("refresh")},
        {QStringLiteral("About"), QStringLiteral("info")},
    };
    for (int i = 0; i < items.size(); ++i) {
        auto* item = new SubNavItem(items[i].first, items[i].second);
        group->addButton(item, i);
        nv->addWidget(item);
        m_nav << item;
        m_names << items[i].first;
        connect(item, &QAbstractButton::clicked, this, [this, i] { showSection(Section(i)); });
    }
    nv->addStretch(1);
    auto* footer = new Label(QStringLiteral("Changes save as you make them. Console and graphics changes apply on the next launch."),
                             Label::Style::Caption, Theme::Role::Text3);
    footer->setWordWrap(true);
    footer->setContentsMargins(10, 0, 10, 4);
    nv->addWidget(footer);
    row->addWidget(nav);

    const QList<QWidget*> pages = {
        SettingsSections::console(services), SettingsSections::network(services), SettingsSections::shaders(services),
        SettingsSections::runProfile(services), SettingsSections::overlay(services), SettingsSections::gameFolders(services, [this] { emit addGamesRequested(); }),
        SettingsSections::input(services), SettingsSections::graphics(services), SettingsSections::logs(services),
        SettingsSections::updates(services), SettingsSections::about(services),
    };
    for (QWidget* page : pages) {
        auto* scroll = new QScrollArea;
        scroll->setWidgetResizable(true);
        scroll->setFrameShape(QFrame::NoFrame);
        scroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        page->setObjectName(QStringLiteral("scrollContents"));
        scroll->setWidget(page);
        m_stack->addWidget(scroll);
    }
    row->addWidget(m_stack, 1);
    v->addLayout(row, 1);
    showSection(Section::Console);
}

void SettingsPage::showSection(Section section)
{
    m_nav[int(section)]->setChecked(true);
    m_stack->setCurrentIndex(int(section));
    m_header->setSubtitle(m_names[int(section)]);
}

QWidget* SettingsPage::sectionPage(Section section) const
{
    return static_cast<QScrollArea*>(m_stack->widget(int(section)))->widget();
}
