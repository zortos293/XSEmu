#pragma once

#include "RunnerManager.h"

#include <QCryptographicHash>
#include <QFile>
#include <QObject>
#include <QPointer>
#include <QProcessEnvironment>

#include <functional>

class ProcessTree;
class QNetworkAccessManager;
class QNetworkReply;

// One install or update of a third-party runner, step by step, every process hidden:
//   release asset: download (SHA-256 checked) -> unpack -> swap into place
//   source:        toolchain bootstrap -> fetch the pinned commit (each remote in turn until one
//                  has it, its commit and tree ids checked) -> build steps -> swap into place
// The new copy is assembled in <runner>\staging and only replaces <runner>\app when every step
// succeeded, so a cancelled or failed update leaves the installed version working. Each run writes
// <runner>\install.log; a failed run also leaves <runner>\install-failed.json with the reason, so the
// failure survives a restart (RunnerManager::status), and a successful one removes it.
class RunnerInstall : public QObject {
    Q_OBJECT
public:
    RunnerInstall(const RunnerSpec& spec, QObject* parent = nullptr);
    ~RunnerInstall() override;

    void start();
    void cancel();
    RunnerManager::Progress progress() const { return m_progress; }
    // A person stopped this install (cancel()): finished(false, "Cancelled.") follows, but it is not a
    // failure and nothing is recorded as one.
    bool cancelled() const { return m_cancelled; }

    // A download is tried this many times in all, so a dropped connection costs a retry rather than the
    // whole install; a SHA-256 that does not match is never retried.
    static constexpr int kDownloadAttempts = 3;

    // The unpack command for an archive (Windows: the system tar.exe, which reads zip; elsewhere
    // unzip for .zip and tar for the rest), tested.
    static QStringList unpackCommand(const QString& archive, const QString& intoDir);
    // The program a step runs, found on the PATH the step gets (env), not the launcher's: Windows looks
    // a bare name up on the parent's PATH, which lacks the Rust the install bootstrapped. Unchanged when
    // it is already a path or is not found there.
    static QString resolveProgram(const QString& program, const QProcessEnvironment& env);
    // <runner>\install.log (the last run) and <runner>\install-failed.json (the last run failed).
    static QString logPath(const QString& id);
    static QString failurePath(const QString& id);

signals:
    void progressChanged();
    void finished(bool ok, const QString& error);

private:
    struct Step {
        QString name;
        std::function<void()> run;
    };
    void next();
    void fail(const QString& error);
    void setStep(const QString& detail = {});
    void download(const QString& url, const QString& to, const QString& sha256, int attempt = 0);
    // then (when set) gets whether it exited 0 instead of the run going on or failing
    void runProcess(const QStringList& command, const QString& workDir, const QString& parser, std::function<void(bool)> then = {});
    void fetchFrom(const QString& src, int remote, QStringList refused);
    // Why the checkout's commit or tree is not the pinned one (empty: it is).
    QString pinMismatch(const QString& src, const QString& rev);
    void ensureGit();
    void ensureMsvc();
    void ensureRust();
    void finishInstall(const QString& appDir);
    void log(const QString& line);

    RunnerSpec m_spec;
    QString m_dir;
    QList<Step> m_steps;
    int m_index = -1;
    RunnerManager::Progress m_progress;
    QNetworkAccessManager* m_net = nullptr;
    QPointer<QNetworkReply> m_reply;
    QFile m_file;
    QFile m_log; // <runner>\install.log: every step, command and output line of this run
    QCryptographicHash m_hash{QCryptographicHash::Sha256};
    QPointer<ProcessTree> m_process;
    QProcessEnvironment m_env;
    QString m_lastLine;
    QString m_git = QStringLiteral("git");
    bool m_cancelled = false;
    bool m_done = false;
};
