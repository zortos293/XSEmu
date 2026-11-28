// xsemu-dlss-shim.dll - the C ABI the emulator loads, Streamline's C++ API underneath.
//
// xcl gives this DLL one D3D12 device at startup and, per frame, the four resources the title's
// FSR4 chain would have consumed plus the jitter it was rendered with. Everything else - optimal
// settings, constants, tags, the evaluation itself - happens here.
//
// It creates no device, owns no queue and submits nothing: the evaluation is recorded on the
// command list xcl hands in, between the title's own submissions, so the frame stays one stream.

#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>

#include <cstdio>
#include <cstring>
#include <mutex>

#include "sl.h"
#include "sl_dlss.h"

namespace {

std::mutex g_lock;
bool g_up = false;
sl::ViewportHandle g_viewport{ 0 };
sl::FrameToken* g_token = nullptr;

void logline(const char* what, sl::Result r)
{
    // One line to the process console, the same place xcl logs; never a message box.
    std::fprintf(stderr, "xsemu-dlss-shim: %s: %d\n", what, static_cast<int>(r));
}

// A resource tag array is rebuilt every frame: Streamline reads it during the evaluate call only.
struct Tags {
    sl::ResourceTag tags[5]{};
    uint32_t n = 0;
    void add(sl::Resource* r, sl::BufferType type)
    {
        tags[n++] = sl::ResourceTag{ r, type, nullptr, sl::ResourceLifecycle::eValidUntilEvaluate };
    }
    const sl::BaseStructure** base() const { return reinterpret_cast<const sl::BaseStructure**>(tags); }
};

sl::Resource wrap(ID3D12Resource* r, uint32_t state)
{
    sl::Resource res{ sl::ResourceType::eTex2d, r };
    res.state = state;
    return res;
}

} // namespace

extern "C" __declspec(dllexport) int xsemu_dlss_init(void* device, int mode)
{
    std::lock_guard<std::mutex> hold(g_lock);
    if (g_up) return 0;
    auto* dev = static_cast<ID3D12Device*>(device);
    if (dev == nullptr) return 1;

    sl::Preferences pref{};
    pref.renderAPI = sl::RenderAPI::eD3D12;
    pref.logLevel = sl::LogLevel::eDefault;
    pref.pathsToPlugins = nullptr; // plugins sit beside sl.interposer.dll, which is beside us
    pref.numPathsToPlugins = 0;

    sl::Result r = slInit(pref, sl::kSDKVersion);
    if (r != sl::Result::eOk) { logline("slInit", r); return 2; }

    r = slSetD3DDevice(dev);
    if (r != sl::Result::eOk) { logline("slSetD3DDevice", r); return 3; }

    sl::AdapterInfo info{};
    info.deviceLUID = nullptr;
    r = slIsFeatureSupported(sl::kFeatureDLSS, &info);
    if (r != sl::Result::eOk) { logline("slIsFeatureSupported(DLSS)", r); return 4; }

    // The feature's own functions first: without slDLSSSetOptions nothing below is worth doing.
    using SetOptionsFn = sl::Result (*)(const sl::ViewportHandle&, const sl::DLSSOptions&);
    using GetOptimalFn = sl::Result (*)(const sl::ViewportHandle&, sl::DLSSOptimalSettings&);
    void* setOptions = nullptr;
    r = slGetFeatureFunction(sl::kFeatureDLSS, "slDLSSSetOptions", setOptions);
    if (r != sl::Result::eOk) { logline("slGetFeatureFunction(slDLSSSetOptions)", r); return 5; }
    void* getOptimal = nullptr;
    r = slGetFeatureFunction(sl::kFeatureDLSS, "slDLSSGetOptimalSettings", getOptimal);
    if (r != sl::Result::eOk) { logline("slGetFeatureFunction(slDLSSGetOptimalSettings)", r); return 6; }

    sl::DLSSOptions options{};
    options.mode = mode == 1 ? sl::DLSSMode::eBalanced
                  : mode == 2 ? sl::DLSSMode::eMaxPerformance
                              : sl::DLSSMode::eMaxQuality;
    // Sizes come from the resources at evaluate time; the emulator's display size is fixed for the
    // run, so the optimal settings are asked once, here, from the descriptor the title uses.
    sl::DLSSOptimalSettings optimal{};
    if (reinterpret_cast<GetOptimalFn>(getOptimal)(g_viewport, optimal) == sl::Result::eOk) {
        options.outputWidth = optimal.optimalRenderWidth;
        options.outputHeight = optimal.optimalRenderHeight;
    }

    r = reinterpret_cast<SetOptionsFn>(setOptions)(g_viewport, options);
    if (r != sl::Result::eOk) { logline("slDLSSSetOptions", r); return 7; }

    g_up = true;
    std::fprintf(stderr, "xsemu-dlss-shim: Streamline up, DLSS mode %d\n", mode);
    return 0;
}

extern "C" __declspec(dllexport) int xsemu_dlss_evaluate(void* cmd_list, void* colour, void* depth,
    void* motion, void* output, float jitter_x, float jitter_y, int reset)
{
    if (!g_up) return 1;
    auto* list = static_cast<ID3D12GraphicsCommandList*>(cmd_list);
    auto* col = static_cast<ID3D12Resource*>(colour);
    auto* dep = static_cast<ID3D12Resource*>(depth);
    auto* mvec = static_cast<ID3D12Resource*>(motion);
    auto* out = static_cast<ID3D12Resource*>(output);
    if (list == nullptr || col == nullptr || dep == nullptr || mvec == nullptr || out == nullptr) return 2;

    sl::Result r = slGetNewFrameToken(g_token, nullptr);
    if (r != sl::Result::eOk) { logline("slGetNewFrameToken", r); return 3; }

    sl::Constants constants{};
    // DLSS takes pixel-space jitter in its own field and demands jitter-free matrices. The title's
    // matrices are not readable from here, so the camera matrices stay identity and the plugin gets
    // the jitter, depth flags and motion scale - what an upscaler can use without the camera.
    constants.jitterOffset = sl::float2{ jitter_x, jitter_y };
    constants.mvecScale = sl::float2{ 1.0f, 1.0f };
    constants.depthInverted = sl::Boolean::eFalse;
    constants.cameraMotionIncluded = sl::Boolean::eTrue;
    constants.motionVectors3D = sl::Boolean::eFalse;
    constants.reset = reset != 0 ? sl::Boolean::eTrue : sl::Boolean::eFalse;
    constants.cameraViewToClip = sl::float4x4{};
    constants.clipToCameraView = sl::float4x4{};
    constants.clipToPrevClip = sl::float4x4{};
    constants.prevClipToClip = sl::float4x4{};
    for (int i = 0; i < 4; ++i) {
        constants.cameraViewToClip[i][i] = 1.0f;
        constants.clipToCameraView[i][i] = 1.0f;
        constants.clipToPrevClip[i][i] = 1.0f;
        constants.prevClipToClip[i][i] = 1.0f;
    }

    r = slSetConstants(constants, *g_token, g_viewport);
    if (r != sl::Result::eOk) { logline("slSetConstants", r); return 4; }

    sl::Resource c = wrap(col, static_cast<uint32_t>(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
    sl::Resource d = wrap(dep, static_cast<uint32_t>(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
    sl::Resource m = wrap(mvec, static_cast<uint32_t>(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE));
    sl::Resource o = wrap(out, static_cast<uint32_t>(D3D12_RESOURCE_STATE_UNORDERED_ACCESS));

    Tags tags;
    tags.add(&c, sl::BufferType::eBufferTypeScalingInputColor);
    tags.add(&d, sl::BufferType::eBufferTypeDepth);
    tags.add(&m, sl::BufferType::eBufferTypeMotionVectors);
    tags.add(&o, sl::BufferType::eBufferTypeScalingOutputColor);

    auto* cmd = reinterpret_cast<sl::CommandBuffer*>(list);
    r = slEvaluateFeature(sl::kFeatureDLSS, *g_token, tags.base(), tags.n, cmd);
    if (r != sl::Result::eOk) { logline("slEvaluateFeature(DLSS)", r); return 5; }
    return 0;
}

extern "C" __declspec(dllexport) void xsemu_dlss_shutdown(void)
{
    std::lock_guard<std::mutex> hold(g_lock);
    if (!g_up) return;
    slFreeResources(sl::kFeatureDLSS, g_viewport);
    slShutdown();
    g_up = false;
}
