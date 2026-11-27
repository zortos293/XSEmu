find_package(Qt6 REQUIRED COMPONENTS Test)

function(xsemu_test name)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} PRIVATE xsemu_core Qt6::Test)
    target_compile_definitions(${name} PRIVATE
        XSEMU_FIXTURES_DIR="${CMAKE_CURRENT_SOURCE_DIR}/fixtures"
        XSEMU_PROTOCOL_DOC="${CMAKE_CURRENT_SOURCE_DIR}/../../docs/launcher-protocol.md")
    xsemu_warnings(${name})
    # -o -,txt: the plain-text log on stdout, so ctest --output-on-failure shows it on Windows too
    add_test(NAME ${name} COMMAND ${name} -o -,txt)
    # no test reaches the real art proxy: lookups for fixture games fail fast on a closed port
    set_tests_properties(${name} PROPERTIES ENVIRONMENT "QT_QPA_PLATFORM=offscreen;XSEMU_SGDB_PROXY=http://127.0.0.1:9/api/v2")
endfunction()

xsemu_test(tst_protocol)
xsemu_test(tst_titleconfig)
xsemu_test(tst_library)
xsemu_test(tst_coverart)
xsemu_test(tst_shadercache)
xsemu_test(tst_settings)
xsemu_test(tst_runners)
xsemu_test(tst_settingsprofile)

# a DexNom-shaped source build read back through every runner surface, then after a restart
# (cargo is a small program here, so the build step starts the same way on every OS)
add_executable(fake-cargo fake-cargo.cpp)
xsemu_warnings(fake-cargo)
xsemu_test(tst_runnerstate)
target_link_libraries(tst_runnerstate PRIVATE xsemu_ui)
add_dependencies(tst_runnerstate fake-cargo)
target_compile_definitions(tst_runnerstate PRIVATE XSEMU_FAKE_CARGO="$<TARGET_FILE:fake-cargo>")

if(WIN32)
    # a window from another process reparented into a Qt host (the test picks the real windows platform)
    add_executable(reparent-helper WIN32 reparent-helper.cpp)
    xsemu_warnings(reparent-helper)
    xsemu_test(tst_reparent)
    target_link_libraries(tst_reparent PRIVATE Qt6::Widgets user32)
    add_dependencies(tst_reparent reparent-helper)
    target_compile_definitions(tst_reparent PRIVATE XSEMU_REPARENT_HELPER="$<TARGET_FILE:reparent-helper>")
endif()

if(XSEMU_FAKE_EMULATOR)
    xsemu_test(tst_controllers)
    add_dependencies(tst_controllers fake-xsemu)
    target_compile_definitions(tst_controllers PRIVATE XSEMU_FAKE_EMULATOR_PATH="$<TARGET_FILE:fake-xsemu>")

    # The in-game view end to end: real clicks on the shipped widgets, fake-xsemu hosted in the view.
    # Runs offscreen by default; CI also runs it on Xvfb so the game window is a real embedded child.
    xsemu_test(tst_ingame)
    target_link_libraries(tst_ingame PRIVATE xsemu_ui)
    add_dependencies(tst_ingame fake-xsemu)
    target_compile_definitions(tst_ingame PRIVATE XSEMU_FAKE_EMULATOR_PATH="$<TARGET_FILE:fake-xsemu>")

    # page switches and stats ticks timed on a generated PC-shaped shader cache and library
    xsemu_test(tst_perf)
    target_link_libraries(tst_perf PRIVATE xsemu_ui)
    add_dependencies(tst_perf fake-xsemu)
    target_compile_definitions(tst_perf PRIVATE XSEMU_FAKE_EMULATOR_PATH="$<TARGET_FILE:fake-xsemu>")
    set_tests_properties(tst_perf PROPERTIES TIMEOUT 600)

    # The launcher allows one emulator per PC, and tst_controllers runs a stand-in named xcl.exe to
    # prove it; tests that start an emulator therefore take turns under ctest -j.
    set_tests_properties(tst_controllers tst_ingame tst_perf PROPERTIES RESOURCE_LOCK emulator_slot)
endif()
