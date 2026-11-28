#pragma once

#include <QList>
#include <QPair>
#include <QString>

// The quality presets of Per-game overrides: one choice sets the upscaler switches and the title
// memory budget together.
//
//   Quality      XCL_UPSCALER=dlss  XCL_DLSS_MODE=quality       title_memory_mb = 8192
//   Balanced     XCL_UPSCALER=dlss  XCL_DLSS_MODE=balanced      title_memory_mb = 6144
//   Performance  XCL_UPSCALER=dlss  XCL_DLSS_MODE=performance   title_memory_mb = 4096
//
// Clear (no stored value) sets neither switch and writes no budget: XSEmu Core's blit default
// applies. The budgets stay inside the Series S 8 GiB title budget (crates/xcl/src/config.rs:
// title_memory_mb 0 is the device's standard budget, 8 192 MiB on Series S).
namespace GraphicsPreset {

struct Preset {
    QString key;           // the value AppSettings::gameOverrides stores
    QString label;         // the name the control shows
    QString dlssMode;      // XCL_DLSS_MODE
    qint64 titleMemoryMb;  // title_memory_mb
};

// Quality, Balanced, Performance, in that order.
const QList<Preset>& all();

// The preset a stored value names; nullptr for Clear and any value this build does not know.
const Preset* find(const QString& key);

// True when the Core this launcher starts reads XCL_UPSCALER and XCL_DLSS_MODE. No Core build does
// yet (crates/xcl has neither switch), so the row stays disabled and no title budget is written;
// XSEMU_QUALITY_PRESETS=1 enables it for tests and for a Core built from that patch.
bool supported();

// The environment one preset adds; empty for Clear.
QList<QPair<QString, QString>> environment(const Preset& preset);

}
