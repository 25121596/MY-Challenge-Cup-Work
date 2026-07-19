#include "file_dialog.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

// ── Windows ───────────────────────────────────────────────────
#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#  include <commdlg.h>

std::string file_dialog_open(const std::string & title,
                             const std::string & filter_patterns) {
    // Build the filter string: "desc\0*.ext\0desc\0*.ext\0\0"
    std::wstring wide_filter;
    {
        std::string copy = filter_patterns;
        for (size_t i = 0; i < copy.size(); i++) {
            if (copy[i] == ';') copy[i] = '\0';
        }
        if (!copy.empty() && copy.back() != '\0') copy += '\0';
        // Convert to wide: "GGUF models\0*.gguf\0All files\0*.*\0"
        std::string final_filter = "GGUF models";
        final_filter += '\0';
        final_filter += copy;
        final_filter += '\0';
        final_filter += "All files (*.*)";
        final_filter += '\0';
        final_filter += "*.*";
        final_filter += '\0';
        int wlen = MultiByteToWideChar(CP_UTF8, 0,
                                       final_filter.data(),
                                       (int)final_filter.size(),
                                       nullptr, 0);
        wide_filter.resize(wlen + 1);
        MultiByteToWideChar(CP_UTF8, 0,
                            final_filter.data(),
                            (int)final_filter.size(),
                            wide_filter.data(), wlen);
        // Replace embedded \0 with actual nulls
        for (int k = 0; k < wlen; k++) {
            if (wide_filter[k] == L'\0' && k + 1 < wlen)
                continue; // wide_filter already has \0 from the conversion
        }
    }

    wchar_t file_buf[MAX_PATH] = {0};

    std::wstring wtitle;
    {
        int wlen = MultiByteToWideChar(CP_UTF8, 0, title.c_str(), -1, nullptr, 0);
        wtitle.resize(wlen);
        MultiByteToWideChar(CP_UTF8, 0, title.c_str(), -1, wtitle.data(), wlen);
    }

    OPENFILENAMEW ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner   = nullptr;
    ofn.lpstrFilter = wide_filter.c_str();
    ofn.lpstrFile   = file_buf;
    ofn.nMaxFile    = MAX_PATH;
    ofn.lpstrTitle  = wtitle.c_str();
    ofn.Flags       = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST | OFN_NOCHANGEDIR;

    if (!GetOpenFileNameW(&ofn)) {
        return {};
    }

    int len = WideCharToMultiByte(CP_UTF8, 0, file_buf, -1, nullptr, 0, nullptr, nullptr);
    if (len <= 0) return {};
    std::string result(len - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, file_buf, -1, result.data(), len, nullptr, nullptr);
    return result;
}

std::string file_dialog_save(const std::string & title,
                             const std::string & default_name) {
    wchar_t file_buf[MAX_PATH] = {0};
    {
        int wlen = MultiByteToWideChar(CP_UTF8, 0, default_name.c_str(), -1, nullptr, 0);
        if (wlen > 0 && wlen <= MAX_PATH) {
            MultiByteToWideChar(CP_UTF8, 0, default_name.c_str(), -1, file_buf, wlen);
        }
    }

    std::wstring wtitle;
    {
        int wlen = MultiByteToWideChar(CP_UTF8, 0, title.c_str(), -1, nullptr, 0);
        wtitle.resize(wlen);
        MultiByteToWideChar(CP_UTF8, 0, title.c_str(), -1, wtitle.data(), wlen);
    }

    OPENFILENAMEW ofn = {};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner   = nullptr;
    ofn.lpstrFilter = L"All files (*.*)\0*.*\0";
    ofn.lpstrFile   = file_buf;
    ofn.nMaxFile    = MAX_PATH;
    ofn.lpstrTitle  = wtitle.c_str();
    ofn.Flags       = OFN_OVERWRITEPROMPT | OFN_NOCHANGEDIR;

    if (!GetSaveFileNameW(&ofn)) {
        return {};
    }

    int len = WideCharToMultiByte(CP_UTF8, 0, file_buf, -1, nullptr, 0, nullptr, nullptr);
    if (len <= 0) return {};
    std::string result(len - 1, '\0');
    WideCharToMultiByte(CP_UTF8, 0, file_buf, -1, result.data(), len, nullptr, nullptr);
    return result;
}

// ── macOS ─────────────────────────────────────────────────────
#elif defined(__APPLE__)

#include <cstdlib>

static std::string popen_read(const std::string & cmd) {
    std::string result;
    FILE * fp = popen(cmd.c_str(), "r");
    if (!fp) return {};
    char buf[4096];
    while (fgets(buf, sizeof(buf), fp)) {
        result += buf;
    }
    pclose(fp);
    // Strip trailing newline
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
        result.pop_back();
    return result;
}

std::string file_dialog_open(const std::string & title,
                             const std::string & /*filter_patterns*/) {
    std::string cmd =
        "osascript -e 'POSIX path of (choose file with prompt \""
        + title + "\")' 2>/dev/null";
    return popen_read(cmd);
}

std::string file_dialog_save(const std::string & title,
                             const std::string & default_name) {
    std::string cmd =
        "osascript -e 'POSIX path of (choose file name with prompt \""
        + title + "\" default name \"" + default_name + "\")' 2>/dev/null";
    return popen_read(cmd);
}

// ── Linux / BSD ───────────────────────────────────────────────
#else

#include <cstdlib>

static std::string popen_read(const std::string & cmd) {
    std::string result;
    FILE * fp = popen(cmd.c_str(), "r");
    if (!fp) return {};
    char buf[4096];
    while (fgets(buf, sizeof(buf), fp)) {
        result += buf;
    }
    pclose(fp);
    while (!result.empty() && (result.back() == '\n' || result.back() == '\r'))
        result.pop_back();
    return result;
}

// Check if a command is available in PATH
static bool command_exists(const char * name) {
    std::string cmd = std::string("command -v ") + name + " >/dev/null 2>&1";
    return system(cmd.c_str()) == 0;
}

std::string file_dialog_open(const std::string & title,
                             const std::string & filter_patterns) {
    // Build a file-filter argument for zenity / kdialog
    std::string filter_arg;
    {
        std::string copy = filter_patterns;
        // Convert "*.gguf;*.bin" to zenity format: --file-filter="GGUF | *.gguf *.bin"
        filter_arg = "--file-filter=\"GGUF | " + copy + "\"";
        // Replace ; with space for the filter
        for (auto & c : filter_arg) {
            if (c == ';') c = ' ';
        }
    }

    if (command_exists("zenity")) {
        std::string cmd = "zenity --file-selection --title=\"" + title + "\" "
                          + filter_arg + " 2>/dev/null";
        return popen_read(cmd);
    }
    if (command_exists("kdialog")) {
        std::string cmd = "kdialog --getopenfilename . \"" +
                          std::string(filter_patterns) + "\" "
                          "--title \"" + title + "\" 2>/dev/null";
        return popen_read(cmd);
    }
    return {};
}

std::string file_dialog_save(const std::string & title,
                             const std::string & default_name) {
    if (command_exists("zenity")) {
        std::string cmd = "zenity --file-selection --save --confirm-overwrite "
                          "--title=\"" + title + "\" --filename=\"" +
                          default_name + "\" 2>/dev/null";
        return popen_read(cmd);
    }
    if (command_exists("kdialog")) {
        std::string cmd = "kdialog --getsavefilename . \"" + default_name + "\" "
                          "--title \"" + title + "\" 2>/dev/null";
        return popen_read(cmd);
    }
    return {};
}

#endif
