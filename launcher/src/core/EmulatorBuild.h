#pragma once

#include <QString>

#include <functional>

// What the emulator executable is: its --version line and whether it speaks the launcher protocol.
// The launcher starts a game or a warm-up with XSEmu only on a capable build: an older xcl.exe would
// read --host-hwnd and the pipe name as the game folder and start the game in its own window.
//
//   const EmulatorBuild build = EmulatorBuild::probe(settings.emulatorPath());
//   if (!build.capable)
//       note->setText(build.problem());
struct EmulatorBuild {
    QString path;
    bool found = false;   // the file exists and is executable
    QString version;      // the first line of --version ("XSEmu 0.9.0", "xcl 0.1.0"); empty: no answer
    bool capable = false; // version starts with "XSEmu " and --help names --control-pipe and --progress-json

    // Why the launcher can't use it, empty when capable.
    QString problem() const;

    // Ask the executable --version and --help (hidden, 3 s each). The answer is kept per path, size and
    // modification time, so each build is asked once; safe from any thread. Never call it while another
    // emulator runs. When abandon() turns true before or while it asks (checked every 50 ms), the child
    // is killed and the result is not found and not kept.
    static EmulatorBuild probe(const QString& path, const std::function<bool()>& abandon = {});
    // The kept answer for the file as it is now, without starting anything; found is false when none.
    static EmulatorBuild cached(const QString& path);
    // Drop the kept answers for a path, for when the file behind it is not the one that was asked.
    static void forget(const QString& path);
    static bool speaksProtocol(const QString& versionLine, const QString& help);
};
