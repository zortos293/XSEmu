cmake_minimum_required(VERSION 3.20)
project(xsemu-dlss-shim LANGUAGES CXX)

# Build: point STREAMLINE_DIR at a Streamline SDK 2.x checkout (scripts/fetch-dlss.ps1 unpacks one).
#   cmake -B build -A x64 -DSTREAMLINE_DIR=C:/streamline-sdk
#   cmake --build build --config Release
# The result is xsemu-dlss-shim.dll; scripts/fetch-dlss.ps1 copies it beside the emulator.
set(STREAMLINE_DIR "" CACHE PATH "Streamline SDK root (contains include/ and lib/x64/)")
if(NOT STREAMLINE_DIR)
    message(FATAL_ERROR "set -DSTREAMLINE_DIR=<Streamline SDK root>")
endif()

add_library(xsemu-dlss-shim SHARED xsemu-dlss-shim.cpp)
target_include_directories(xsemu-dlss-shim PRIVATE "${STREAMLINE_DIR}/include")
# sl.interposer.lib is the loader stub; the DLLs are resolved at run time from the folder xcl points at.
target_link_directories(xsemu-dlss-shim PRIVATE "${STREAMLINE_DIR}/lib/x64")
target_link_libraries(xsemu-dlss-shim PRIVATE d3d12 dxgi sl.interposer)
target_compile_features(xsemu-dlss-shim PRIVATE cxx_std_17)
set_target_properties(xsemu-dlss-shim PROPERTIES OUTPUT_NAME "xsemu-dlss-shim")
