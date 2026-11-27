#pragma once

#include <QWidget>

class Button;
class Label;
class ProgressBlock;
class QComboBox;
class QGridLayout;
class Services;
struct Game;
struct RunnerSpec;

// Shader warm-up (Warm-up page). Every state comes from WarmupController and the --progress-json
// protocol: the idle card for the picked game, running with the phase named, paused at the RAM
// floor, finished, cancelled (running again goes on from there), stopped part way (a full pass
// builds the rest), failed (exit code and the log), and the banner while a game blocks it. Recent
// runs are AppSettings::recentWarmups.
class WarmupPage : public QWidget {
    Q_OBJECT
public:
    explicit WarmupPage(Services& services, QWidget* parent = nullptr);

private:
    void refresh();
    void refreshGames();
    void refreshRuns();
    void refreshRunner(const RunnerSpec& spec, const Game& picked, bool showResult, const QString& cannot);
    void start(bool full);

    Services& m_s;
    QComboBox* m_game;
    Button* m_start;
    QWidget* m_banner;
    Label* m_bannerTitle;
    Label* m_bannerText;
    Button* m_stopGame;
    ProgressBlock* m_block;
    Button* m_play;
    Button* m_warm;   // idle: "Warm up <game>" (primary)
    Button* m_resume; // cancelled: "Resume warm-up"; failed: "Retry"; stopped: "Run full pass"
    Button* m_log;
    Button* m_showLog; // running: in the note line
    Button* m_dismiss;
    Button* m_cancel;
    QGridLayout* m_runs;
    bool m_dismissed = false;
    bool m_fullPass = false;     // the warm-up button starts a full pass (a result block), not the picked game's
    bool m_continueFull = false; // Continue runs the pass that stopped, which may have been a full one
    QString m_startError;
};
