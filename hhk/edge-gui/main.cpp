// Edge GUI - Lightweight LLM Inference GUI for edge devices
//
// Uses Dear ImGui + GLFW + OpenGL3 for rendering,
// libllama for local inference,
// cpp-httplib for cloud offloading.

#include "app.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"

#include <GLFW/glfw3.h>
#include <cstdio>
#include <cstdlib>
#include <clocale>

static void glfw_error_callback(int error, const char * description) {
    fprintf(stderr, "GLFW error %d: %s\n", error, description);
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // ── Parse command-line arguments ───────────────────────────
    std::string model_path;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            model_path = argv[++i];
        }
    }

    // ── Initialize GLFW + OpenGL ───────────────────────────────
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return 1;
    }

    const char * glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    GLFWwindow * window = glfwCreateWindow(900, 650,
                                           "Edge LLM",
                                           nullptr, nullptr);
    if (!window) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return 1;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);  // vsync

    // ── Initialize Dear ImGui ─────────────────────────────────
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO & io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.IniFilename = "edge-gui.ini";

    // Try to load a CJK-capable TTF font from the system.
    // Platform-specific font paths for CJK rendering.
    {
        const char * font_candidates[] = {
#ifdef _WIN32
            "C:/Windows/Fonts/msyh.ttc",   // Microsoft YaHei (Win10/11)
            "C:/Windows/Fonts/msyh.ttf",   // Older Windows
            "C:/Windows/Fonts/simhei.ttf", // SimHei fallback
            "C:/Windows/Fonts/simsun.ttc", // SimSun fallback
#elif defined(__APPLE__)
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/STHeiti Light.ttc",
            "/Library/Fonts/Arial Unicode.ttf",
#else
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
            "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
#endif
        };
        ImFont * font = nullptr;
        for (const char * path : font_candidates) {
            font = io.Fonts->AddFontFromFileTTF(path, 20.0f, nullptr,
                                                io.Fonts->GetGlyphRangesChineseFull());
            if (font) break;
        }
        if (!font) {
            // No CJK font available - use default and just scale it up
            io.Fonts->AddFontDefault();
            io.FontGlobalScale = 1.6f;
        }
    }

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // ── Initialize application ─────────────────────────────────
    EdgeApp app;
    if (!model_path.empty()) {
        app.load_model(model_path);
    }

    // ── Main loop ──────────────────────────────────────────────
    while (!glfwWindowShouldClose(window)) {
        glfwPollEvents();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        app.render();

        ImGui::Render();

        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(0.12f, 0.12f, 0.14f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glfwSwapBuffers(window);
    }

    // ── Cleanup ────────────────────────────────────────────────
    app.shutdown();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();

    return 0;
}
