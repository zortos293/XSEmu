#include "GraphicsPreset.h"

#include <QStringLiteral>

namespace GraphicsPreset {

const QList<Preset>& all()
{
    static const QList<Preset> presets{
        {QStringLiteral("quality"), QStringLiteral("Quality"), QStringLiteral("quality"), 8192},
        {QStringLiteral("balanced"), QStringLiteral("Balanced"), QStringLiteral("balanced"), 6144},
        {QStringLiteral("performance"), QStringLiteral("Performance"), QStringLiteral("performance"), 4096},
    };
    return presets;
}

const Preset* find(const QString& key)
{
    for (const Preset& preset : all()) {
        if (preset.key == key)
            return &preset;
    }
    return nullptr;
}

QList<QPair<QString, QString>> environment(const Preset& preset)
{
    return {{QStringLiteral("XCL_UPSCALER"), QStringLiteral("dlss")}, {QStringLiteral("XCL_DLSS_MODE"), preset.dlssMode}};
}

}
