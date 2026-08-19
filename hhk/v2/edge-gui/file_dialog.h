#pragma once

#include <string>

// Cross-platform native file-open dialog.
// Returns the selected file path, or an empty string on cancel/error.
//
// Windows    : COMCTL32 GetOpenFileNameW
// macOS      : osascript NSOpenPanel
// Linux/BSD  : zenity / kdialog
std::string file_dialog_open(
    const std::string & title,
    const std::string & filter_patterns);  // e.g. "*.gguf;*.bin"

// Cross-platform native file-save dialog.
// Returns the chosen path, or an empty string on cancel/error.
std::string file_dialog_save(
    const std::string & title,
    const std::string & default_name);
