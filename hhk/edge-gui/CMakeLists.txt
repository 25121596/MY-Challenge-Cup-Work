cmake_minimum_required(VERSION 3.16)
project(edge-gui LANGUAGES CXX C)

# ── Dependencies ──────────────────────────────────────────────

find_package(OpenGL REQUIRED)
find_package(Threads REQUIRED)

# GLFW
find_package(glfw3 QUIET)
if(NOT glfw3_FOUND)
    include(FetchContent)
    FetchContent_Declare(glfw
        GIT_REPOSITORY https://github.com/glfw/glfw.git
        GIT_TAG 3.4
        GIT_SHALLOW TRUE
    )
    set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(glfw)
endif()

# Dear ImGui
include(FetchContent)
FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.91.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(imgui)

add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
)
target_include_directories(imgui PUBLIC ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends)
target_link_libraries(imgui PUBLIC glfw OpenGL::GL)

# ── Link llama.cpp libraries ──────────────────────────────────

if(TARGET llama AND TARGET llama-common)
    message(STATUS "edge-gui: building as part of llama.cpp")
else()
    find_package(llama REQUIRED)
    find_package(llama-common REQUIRED)
endif()

# ── Edge GUI executable ───────────────────────────────────────

add_executable(edge-gui
    main.cpp
    app.cpp
    app.h
    engine.cpp
    engine.h
    cloud.cpp
    cloud.h
    edge_cloud.cpp
    edge_cloud.h
    device_monitor.cpp
    device_monitor.h
    heartbeat.cpp
    heartbeat.h
    task_scheduler.cpp
    task_scheduler.h
    model_sync.cpp
    model_sync.h
    p2p_mesh.cpp
    p2p_mesh.h
    conflict_detector.cpp
    conflict_detector.h
    file_dialog.cpp
    file_dialog.h
)

# Optional multimodal support (Phase 3)
option(EDGE_GUI_MTMD "edge-gui: enable multimodal input" OFF)
if(EDGE_GUI_MTMD)
    target_sources(edge-gui PRIVATE media_source.cpp media_source.h)
    target_link_libraries(edge-gui PRIVATE mtmd)
    target_compile_definitions(edge-gui PRIVATE EDGE_GUI_HAS_MTMD)
endif()

target_link_libraries(edge-gui PRIVATE
    imgui
    llama
    llama-common
    cpp-httplib
    Threads::Threads
)

target_compile_features(edge-gui PRIVATE cxx_std_17)

# Platform-specific libraries
if(WIN32)
    target_link_libraries(edge-gui PRIVATE winmm comdlg32 ws2_32)
endif()

install(TARGETS edge-gui RUNTIME DESTINATION bin)