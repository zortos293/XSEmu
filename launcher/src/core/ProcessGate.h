#pragma once

#include <QObject>
#include <QString>
#include <QStringList>
#include <QTimer>

#include <functional>

// The one-process rule: at most one XSEmu process tree (a game or a warm-up) at a time. Both
// controllers acquire the gate before starting XSEmu.exe and release it when the tree is gone; every
// other UI asks it why an action is blocked and how to clear the block.
//
//   if (!gate.isFree()) {
//       note->setText(gate.blockedReason(ProcessGate::Action::StartWarmup));
//       stopButton->setText(gate.stopActionLabel());   // "Stop game" / "Cancel warm-up"
//       connect(stopButton, &QPushButton::clicked, &gate, &ProcessGate::stopOwner);
//   }
class ProcessGate : public QObject {
    Q_OBJECT
public:
    // Update: the launcher is replacing its own files or the Core's, which needs the same idle slot a
    // game or a warm-up would take (UpdateService holds it for the whole update).
    enum class Owner { None, Game, Warmup, Update };
    enum class Action { LaunchGame, StartWarmup };

    explicit ProcessGate(QObject* parent = nullptr);

    Owner owner() const { return m_owner; }
    bool isFree() const { return m_owner == Owner::None; }
    QString ownerTitle() const { return m_title; } // the game playing or being warmed up for
    bool ownerPaused() const { return m_paused; }

    // Controllers only.
    bool acquire(Owner owner, const QString& title, std::function<void()> stop);
    void release(Owner owner);
    void setPaused(bool paused);

    // Something outside this launcher holds the PC's one emulator slot: an XSEmu.exe, xcl.exe or
    // game_scarlett_master.exe running (not counting the launcher and its own helpers, such as the
    // emulator asked --version), or the busy marker file
    // existing. external() looks now, for the check right before starting something. A worker thread
    // looks every 2 s; lastExternal() is what it saw, for display, and changed() follows a difference.
    struct External {
        QStringList processes; // "xcl.exe 1234"
        QString marker;        // the busy marker file, when it exists
        bool busy() const { return !processes.isEmpty() || !marker.isEmpty(); }
        QString reason() const; // "Emulator busy: ..." or empty
    };
    External external() const;
    static External externalFor(const QString& marker); // safe from any thread
    const External& lastExternal() const { return m_external; }
    // The file whose existence means "busy" (AppSettings::busyMarker()); empty: none.
    void setBusyMarker(const QString& path);
    QString busyMarker() const { return m_marker; }

    // A human sentence for why the action is blocked now, or empty when it is allowed.
    QString blockedReason(Action action) const;
    // The label of the action that clears the block ("Stop game", "Cancel warm-up"), or empty.
    QString stopActionLabel() const;

public slots:
    // Ask the owner to stop (game: stop with grace then kill; warm-up: cancel).
    void stopOwner();

signals:
    void changed();

private:
    Owner m_owner = Owner::None;
    QString m_title;
    bool m_paused = false;
    std::function<void()> m_stop;
    QString m_marker;
    void poll();
    External m_external;
    bool m_polling = false;
    QTimer m_poll;
};
