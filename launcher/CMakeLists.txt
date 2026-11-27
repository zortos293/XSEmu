cmake_minimum_required(VERSION 3.21)

# The version the About page, the update check and the release it came from agree on.
# .github/workflows/release.yml passes the tag's version (-DXSEMU_LAUNCHER_VERSION=0.1.0-pre.2), which
# may carry a pre-release suffix; project(VERSION) takes numbers only, so it keeps the release family
# and XSEMU_VERSION carries the whole string.
set(XSEMU_LAUNCHER_VERSION "0.1.0" CACHE STRING "Version reported by the launcher and compared with releases")
string(REGEX MATCH "^[0-9]+(\\.[0-9]+)*" XSEMU_LAUNCHER_FAMILY "${XSEMU_LAUNCHER_VERSION}")
if(XSEMU_LAUNCHER_FAMILY STREQUAL "")
    message(FATAL_ERROR "XSEMU_LAUNCHER_VERSION '${XSEMU_LAUNCHER_VERSION}' does not start with a number")
endif()

project(XSEmuLauncher VERSION ${XSEMU_LAUNCHER_FAMILY} DESCRIPTION "XSEmu launcher" LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

if(WIN32)
    set(_xsemu_dev_default OFF)
    set(_xsemu_fake_default OFF)
else()
    set(_xsemu_dev_default ON)
    set(_xsemu_fake_default ON)
endif()
option(XSEMU_LAUNCHER_TESTS "Build the launcher unit tests" ${_xsemu_fake_default})
option(XSEMU_FAKE_EMULATOR "Build fake-xsemu, a stand-in for XSEmu.exe that speaks the launcher protocol" ${_xsemu_fake_default})
option(XSEMU_DEV_TOOLS "Build the screenshot lever (--state/--grab) and the widget gallery" ${_xsemu_dev_default})
option(XSEMU_DEPLOY_QT "Run windeployqt after building and installing (Windows)" ON)

find_package(Qt6 6.4 REQUIRED COMPONENTS Core Gui Widgets Network Svg)

function(xsemu_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE /W4 /permissive- /utf-8 /Zc:__cplusplus)
        target_compile_definitions(${target} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN UNICODE _UNICODE)
    else()
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Wshadow=local -Wno-missing-field-initializers)
    endif()
endfunction()

# ---- core: services without any UI (tested) ----
add_library(xsemu_core STATIC
    src/core/AppSettings.cpp src/core/AppSettings.h
    src/core/CoverArtService.cpp src/core/CoverArtService.h
    src/core/DevFixtures.cpp src/core/DevFixtures.h
    src/core/DiskSpace.cpp src/core/DiskSpace.h
    src/core/EmulatorController.cpp src/core/EmulatorController.h
    src/core/Format.cpp src/core/Format.h
    src/core/Game.cpp src/core/Game.h
    src/core/GameManifest.cpp src/core/GameManifest.h
    src/core/HardwareInfo.cpp src/core/HardwareInfo.h
    src/core/LaunchConfig.cpp src/core/LaunchConfig.h
    src/core/NativeWindow.cpp src/core/NativeWindow.h
    src/core/Library.cpp src/core/Library.h
    src/core/Paths.cpp src/core/Paths.h
    src/core/ProcessGate.cpp src/core/ProcessGate.h
    src/core/EmulatorBuild.cpp src/core/EmulatorBuild.h
    src/core/ProcessTree.cpp src/core/ProcessTree.h
    src/core/Release.cpp src/core/Release.h
    src/core/UpdateService.cpp src/core/UpdateService.h
    src/core/ProgressParsers.cpp src/core/ProgressParsers.h
    src/core/Runner.cpp src/core/Runner.h
    src/core/RunnerInstall.cpp src/core/RunnerInstall.h
    src/core/RunnerManager.cpp src/core/RunnerManager.h
    src/core/Protocol.cpp src/core/Protocol.h
    src/core/ShaderCacheInfo.cpp src/core/ShaderCacheInfo.h
    src/core/SettingsProfile.cpp src/core/SettingsProfile.h
    src/core/ShaderCacheMonitor.cpp src/core/ShaderCacheMonitor.h
    src/core/TitleConfig.cpp src/core/TitleConfig.h
    src/core/WarmupController.cpp src/core/WarmupController.h
)
target_include_directories(xsemu_core PUBLIC src src/core)
target_compile_definitions(xsemu_core PUBLIC XSEMU_VERSION="${XSEMU_LAUNCHER_VERSION}")
target_link_libraries(xsemu_core PUBLIC Qt6::Core Qt6::Network)
if(WIN32)
    target_link_libraries(xsemu_core PUBLIC dxgi psapi user32)
elseif(UNIX AND NOT APPLE)
    # other processes' windows (NativeWindow) on X11
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(XCB QUIET IMPORTED_TARGET xcb)
        if(XCB_FOUND)
            target_link_libraries(xsemu_core PUBLIC PkgConfig::XCB)
            target_compile_definitions(xsemu_core PRIVATE XSEMU_XCB)
        endif()
    endif()
endif()
xsemu_warnings(xsemu_core)

# ---- the launcher ----
file(GLOB_RECURSE XSEMU_UI_SOURCES CONFIGURE_DEPENDS
    src/app/*.cpp src/app/*.h
    src/theme/*.cpp src/theme/*.h
    src/widgets/*.cpp src/widgets/*.h
    src/shell/*.cpp src/shell/*.h
    src/pages/*.cpp src/pages/*.h
)
# An object library so the in-game end-to-end test drives the same widgets the launcher ships.
add_library(xsemu_ui OBJECT ${XSEMU_UI_SOURCES} resources/launcher.qrc)
target_include_directories(xsemu_ui PUBLIC src)
target_link_libraries(xsemu_ui PUBLIC xsemu_core Qt6::Gui Qt6::Widgets Qt6::Svg)
if(XSEMU_DEV_TOOLS)
    target_compile_definitions(xsemu_ui PUBLIC XSEMU_DEV_TOOLS XSEMU_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests/fixtures")
endif()
xsemu_warnings(xsemu_ui)

set(XSEMU_RESOURCES)
if(WIN32)
    list(APPEND XSEMU_RESOURCES resources/branding/app.rc)
endif()

add_executable(XSEmuLauncher WIN32 src/main.cpp ${XSEMU_RESOURCES})
target_link_libraries(XSEmuLauncher PRIVATE xsemu_ui)
xsemu_warnings(XSEmuLauncher)

# ---- fake-xsemu ----
if(XSEMU_FAKE_EMULATOR)
    add_subdirectory(tools/fake-xsemu)
    add_dependencies(XSEmuLauncher fake-xsemu)
endif()

# ---- install / deploy ----
include(GNUInstallDirs)
install(TARGETS XSEmuLauncher RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR} BUNDLE DESTINATION .)
if(WIN32 AND XSEMU_DEPLOY_QT)
    get_target_property(_qmake Qt6::qmake IMPORTED_LOCATION)
    get_filename_component(_qtbin "${_qmake}" DIRECTORY)
    find_program(WINDEPLOYQT windeployqt HINTS "${_qtbin}")
    if(WINDEPLOYQT)
        add_custom_command(TARGET XSEmuLauncher POST_BUILD
            COMMAND "${WINDEPLOYQT}" --no-translations --no-system-d3d-compiler --no-opengl-sw "$<TARGET_FILE:XSEmuLauncher>"
            COMMENT "windeployqt (build tree)")
        install(CODE "execute_process(COMMAND \"${WINDEPLOYQT}\" --no-translations --no-system-d3d-compiler --no-opengl-sw \"\${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_BINDIR}/XSEmuLauncher.exe\" COMMAND_ERROR_IS_FATAL ANY)")
    else()
        message(WARNING "windeployqt not found next to qmake; the launcher will not be deployed")
    endif()
endif()

# ---- tests ----
if(XSEMU_LAUNCHER_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
