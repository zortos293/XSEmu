#pragma once

#include <QByteArray>
#include <QString>

#include <optional>
#include <variant>

// The JSON lines of docs/launcher-protocol.md (v1), in one place. Parse emulator lines with
// Protocol::parse; build launcher lines with the Command helpers. Unknown fields are ignored,
// unknown messages parse to Protocol::Unknown, lines that are not JSON objects to Protocol::Invalid.
namespace Protocol {

constexpr int kVersion = 1;

// ---- game control channel (run --control-pipe) ----

struct Hello {
    int proto = 0;
    qint64 pid = 0;
    QString version;
    QString title;
    QString deviceType;
    QString hwnd; // hex, empty without --host-hwnd
};

enum class GameState { Booting, Running, Paused, Stopping, Unknown };

struct State {
    GameState state = GameState::Unknown;
    QString raw;
};

// Fields the emulator cannot measure are left out of the line, so each is optional.
struct Stats {
    std::optional<double> fps;
    std::optional<double> frameMs;
    std::optional<qint64> ramFreeMb;
    std::optional<qint64> vramUsedMb;
    std::optional<qint64> vramBudgetMb;
    std::optional<qint64> stalls;
    std::optional<double> worstMs;
    std::optional<qint64> pipelines;
    std::optional<qint64> compiling;
};

struct Key {
    QString key; // F1, F3, F11, Escape
};

enum class ExitReason { Stopped, TitleExit, Crash, DeviceRemoved, Error, Unknown };

struct Exit {
    int code = 0;
    ExitReason reason = ExitReason::Unknown;
    QString rawReason;
    QString message;
};

// ---- warm-up (warm-up --progress-json) ----

struct WarmupStart {
    qint64 total = 0;
    bool full = false;
    int threads = 0;
};

struct WarmupProgress {
    qint64 done = 0;
    qint64 total = 0;
    double keysPerS = 0;
    std::optional<qint64> etaS;
    std::optional<qint64> ramFreeMb;
    int giantsCompiling = 0;
    int inFlight = 0;
    qint64 failed = 0;
    QString phase;
    bool pausedForRam = false;
};

struct WarmupDone {
    qint64 built = 0;
    qint64 failed = 0;
    qint64 left = -1; // pipelines this pass did not build; -1: the emulator did not say
    qint64 tooBig = 0;
    double tookS = 0;
    bool complete = false;
    bool cancelled = false;
    int code = 0;
};

struct Error {
    QString message;
};

struct Unknown {
    QString ev;
};

struct Invalid {
    QString reason;
};

using Event = std::variant<Invalid, Unknown, Hello, State, Stats, Key, Exit, WarmupStart, WarmupProgress, WarmupDone, Error>;

Event parse(const QByteArray& line);

GameState gameStateFromString(const QString& s);
QString toString(GameState s);
ExitReason exitReasonFromString(const QString& s);
QString toString(ExitReason r);

// Launcher -> emulator lines, each ending in '\n'.
namespace Command {
QByteArray pause();
QByteArray resume();
QByteArray stop();
QByteArray overlay(bool on);
QByteArray focus();
QByteArray cancel(); // warm-up stdin
}

}
