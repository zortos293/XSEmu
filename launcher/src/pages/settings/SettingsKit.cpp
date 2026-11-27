#include "SettingsKit.h"

#include "widgets/Icon.h"
#include "widgets/Text.h"

#include <QAbstractItemView>
#include <QDir>
#include <QEvent>
#include <QHBoxLayout>
#include <QPainter>
#include <QPainterPath>
#include <QVBoxLayout>

namespace SettingsKit {

QString notInBuild()
{
    return QStringLiteral("Not in this XSEmu Core build yet.");
}

Row::Row(const QString& label, const QString& help, QWidget* parent)
    : QWidget(parent)
    , m_row(new QHBoxLayout(this))
    , m_help(new Label(help, Label::Style::Small, Theme::Role::Text2))
    , m_why(new Label({}, Label::Style::Caption, Theme::Role::Text3))
{
    m_row->setContentsMargins(0, 14, 0, 14);
    m_row->setSpacing(24);
    auto* text = new QVBoxLayout;
    text->setSpacing(3);
    auto* title = new Label(label, Label::Style::BodyMedium);
    title->setWordWrap(true);
    text->addWidget(title);
    m_help->setWordWrap(true);
    m_help->setVisible(!help.isEmpty());
    text->addWidget(m_help);
    m_why->setWordWrap(true);
    m_why->hide();
    text->addWidget(m_why);
    m_row->addLayout(text, 1);
}

void Row::addControl(QWidget* w)
{
    m_row->addWidget(w, 0, Qt::AlignVCenter);
}

void Row::setHelp(const QString& help)
{
    m_help->setText(help);
    m_help->setVisible(!help.isEmpty());
}

void Row::setUnavailable(const QString& why)
{
    for (int i = 1; i < m_row->count(); ++i) {
        if (QWidget* w = m_row->itemAt(i)->widget())
            w->setEnabled(false);
    }
    m_why->setText(why);
    m_why->show();
}

void Row::setAvailable()
{
    for (int i = 1; i < m_row->count(); ++i) {
        if (QWidget* w = m_row->itemAt(i)->widget())
            w->setEnabled(true);
    }
    m_why->hide();
}

void Row::setBorder(bool on)
{
    m_border = on;
    update();
}

void Row::paintEvent(QPaintEvent*)
{
    if (!m_border)
        return;
    QPainter p(this);
    p.fillRect(QRect(0, height() - 1, width(), 1), Theme::color(Theme::Role::Border));
}

void Section::caps(const QString& text) const
{
    auto* w = new QWidget;
    auto* l = new QVBoxLayout(w);
    l->setContentsMargins(0, 24, 0, 4);
    l->addWidget(new Label(text, Label::Style::CapsMedium, Theme::Role::Text3));
    body->addWidget(w);
}

Row* Section::row(const QString& label, const QString& help, QWidget* control) const
{
    auto* r = new Row(label, help);
    if (control)
        r->addControl(control);
    body->addWidget(r);
    return r;
}

void Section::add(QWidget* w, int spacingBefore) const
{
    if (spacingBefore > 0)
        body->addSpacing(spacingBefore);
    body->addWidget(w);
}

void Section::finish() const
{
    for (int i = body->count() - 1; i >= 0; --i) {
        QWidget* w = body->itemAt(i)->widget();
        if (auto* r = qobject_cast<Row*>(w)) {
            r->setBorder(false);
            break;
        }
        if (w)
            break;
    }
    body->addStretch(1);
}

Section makeSection(const QString& title, const QString& description)
{
    auto* page = new QWidget;
    auto* v = new QVBoxLayout(page);
    v->setContentsMargins(32, 28, 32 + 120, 32);
    v->setSpacing(0);
    if (title.isEmpty())
        return {page, v};
    auto* head = new QVBoxLayout;
    head->setContentsMargins(0, 0, 0, 8);
    head->setSpacing(6);
    head->addWidget(new Label(title, Label::Style::SectionTitle));
    auto* d = new Label(description, Label::Style::Small, Theme::Role::Text2);
    d->setWordWrap(true);
    head->addWidget(d);
    v->addLayout(head);
    return {page, v};
}

void Dropdown::paintEvent(QPaintEvent*)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    if (!isEnabled())
        p.setOpacity(Metrics::DisabledOpacity);
    const QRectF r = QRectF(rect()).adjusted(0.5, 0.5, -0.5, -1.5);
    p.setPen(Qt::NoPen);
    p.setBrush(QColor(0, 0, 0, 0x0A));
    p.drawRoundedRect(r.translated(0, 1), 7, 7);
    p.setPen(QPen(Theme::color(hasFocus() ? Theme::Role::Accent : Theme::Role::BorderStrong), 1));
    p.setBrush(Theme::color(Theme::Role::Surface));
    p.drawRoundedRect(r, 7, 7);
    p.setFont(Theme::font(13));
    p.setPen(Theme::color(Theme::Role::Text));
    QRect text = rect().adjusted(12, 0, -30, -1);
    if (const QIcon icon = itemIcon(currentIndex()); !icon.isNull()) {
        const QSize s = iconSize();
        icon.paint(&p, QRect(QPoint(text.left(), (height() - s.height()) / 2), s));
        text.setLeft(text.left() + s.width() + 8);
    }
    p.drawText(text, Qt::AlignVCenter | Qt::AlignLeft, QFontMetrics(p.font()).elidedText(currentText(), Qt::ElideRight, text.width()));
    Icon::paint(p, QRectF(width() - 22, (height() - 12) / 2.0 - 0.5, 12, 12), QStringLiteral("chevron-down"), Theme::color(Theme::Role::Text2), 12);
}

QComboBox* dropdown(const QList<QPair<QString, QVariant>>& items, const QVariant& current, int width)
{
    auto* c = new Dropdown;
    for (const auto& it : items)
        c->addItem(it.first, it.second);
    const int i = c->findData(current);
    c->setCurrentIndex(i >= 0 ? i : 0);
    c->setFixedSize(width, 32);
    c->setCursor(Qt::PointingHandCursor);
    return c;
}

Label* status(const QString& text, Theme::Role role)
{
    auto* l = new Label(text, Label::Style::SmallMedium, role);
    l->setAlignment(Qt::AlignRight | Qt::AlignVCenter);
    return l;
}

namespace {

class KeyCap : public QWidget {
public:
    explicit KeyCap(QString key)
        : m_key(std::move(key))
    {
        const QFontMetrics fm(Theme::mono(11));
        setFixedSize(fm.horizontalAdvance(m_key) + 16, 24);
        connect(&Theme::instance(), &Theme::changed, this, qOverload<>(&QWidget::update));
    }

protected:
    void paintEvent(QPaintEvent*) override
    {
        QPainter p(this);
        p.setRenderHint(QPainter::Antialiasing);
        const QRectF r = QRectF(rect()).adjusted(0.5, 0.5, -0.5, -1.5);
        p.setPen(QPen(Theme::color(Theme::Role::BorderStrong), 1));
        p.setBrush(Theme::color(Theme::Role::Surface));
        p.drawRoundedRect(r.translated(0, 1), 5, 5);
        p.drawRoundedRect(r, 5, 5);
        p.setFont(Theme::mono(11));
        p.setPen(Theme::color(Theme::Role::Text));
        p.drawText(r, Qt::AlignCenter, m_key);
    }

private:
    QString m_key;
};

}

namespace {

class ShowWatcher : public QObject {
public:
    ShowWatcher(QWidget* w, std::function<void()> f)
        : QObject(w)
        , m_f(std::move(f))
    {
        w->installEventFilter(this);
    }

protected:
    bool eventFilter(QObject*, QEvent* e) override
    {
        if (e->type() == QEvent::Show)
            m_f();
        return false;
    }

private:
    std::function<void()> m_f;
};

}

void onShow(QWidget* w, std::function<void()> f)
{
    new ShowWatcher(w, std::move(f));
}

QWidget* keyCap(const QString& key)
{
    return new KeyCap(key);
}

PathField::PathField(const QString& path, QWidget* parent)
    : QWidget(parent)
{
    setFixedHeight(32);
    setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed);
    setPath(path);
    connect(&Theme::instance(), &Theme::changed, this, qOverload<>(&QWidget::update));
}

void PathField::setPath(const QString& path)
{
    m_path = QDir::toNativeSeparators(path);
    setToolTip(m_path);
    update();
}

void PathField::paintEvent(QPaintEvent*)
{
    QPainter p(this);
    p.setRenderHint(QPainter::Antialiasing);
    QPainterPath path;
    path.addRoundedRect(QRectF(rect()), 7, 7);
    p.fillPath(path, Theme::color(Theme::Role::Sunken));
    p.setFont(Theme::mono(12));
    p.setPen(Theme::color(Theme::Role::Text));
    const QRect text = rect().adjusted(12, 0, -12, 0);
    p.drawText(text, Qt::AlignVCenter | Qt::AlignLeft, QFontMetrics(p.font()).elidedText(m_path, Qt::ElideMiddle, text.width()));
}

}
