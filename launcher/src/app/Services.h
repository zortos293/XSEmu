#pragma once

#include "core/AppSettings.h"
#include "core/CoverArtService.h"
#include "core/EmulatorController.h"
#include "core/HardwareInfo.h"
#include "core/Library.h"
#include "core/ProcessGate.h"
#include "core/RunnerManager.h"
#include "core/ShaderCacheMonitor.h"
#include "core/UpdateService.h"
#include "core/WarmupController.h"

#include <QHash>
#include <QPixmap>

// Every long-lived service of the launcher, created once in main() and handed to the shell by
// reference. The gate is shared so one XSEmu process tree runs at a time (a game or a warm-up).
// Nothing the UI thread reads here touches the disk on the way: the shader cache, the hardware and
// runner sizes are snapshots kept fresh in the background.
class Services : public QObject {
    Q_OBJECT
public:
    explicit Services(QObject* parent = nullptr);

    AppSettings settings;
    ProcessGate gate;
    RunnerManager runners;
    Library library;
    EmulatorController emulator;
    WarmupController warmup;
    CoverArtService covers;
    ShaderCacheMonitor shaderCache;
    UpdateService updates;
    HardwareMonitor hardware;

    // A game's cover art decoded once and kept in memory (null when there is none), so refreshes and
    // paints never decode from disk. Dropped when the art changes.
    QPixmap art(const QString& titleId, CoverCache::Kind kind);

    // Settings → Console: "lockhart" is Series S, "anaconda" Series X.
    static QString consoleName(const QString& deviceType);

private:
    QHash<QString, QPixmap> m_art;
};
