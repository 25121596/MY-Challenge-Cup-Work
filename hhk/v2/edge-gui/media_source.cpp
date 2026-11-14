#include "media_source.h"

// stb_image is vendored in llama.cpp
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <cstdio>
#include <cstring>

// ── ImageSource ───────────────────────────────────────────────

ImageSource::ImageSource(const std::string & filepath)
    : _filepath(filepath) {}

ImageSource::~ImageSource() { close(); }

bool ImageSource::open() {
    if (_opened) return true;

    int w = 0, h = 0, c = 0;
    unsigned char * pixels = stbi_load(_filepath.c_str(), &w, &h, &c, 3); // force RGB
    if (!pixels) {
        fprintf(stderr, "ImageSource: failed to load %s: %s\n",
                _filepath.c_str(), stbi_failure_reason());
        return false;
    }

    _frame.type     = MediaType::Image;
    _frame.width    = w;
    _frame.height   = h;
    _frame.channels = 3;
    _frame.data.assign(pixels, pixels + (size_t)(w * h * 3));
    stbi_image_free(pixels);

    // Guess mime type from extension
    if (_filepath.find(".png") != std::string::npos ||
        _filepath.find(".PNG") != std::string::npos)
        _frame.mime_type = "image/png";
    else if (_filepath.find(".jpg") != std::string::npos ||
             _filepath.find(".JPG") != std::string::npos ||
             _filepath.find(".jpeg") != std::string::npos ||
             _filepath.find(".JPEG") != std::string::npos)
        _frame.mime_type = "image/jpeg";
    else if (_filepath.find(".webp") != std::string::npos ||
             _filepath.find(".WEBP") != std::string::npos)
        _frame.mime_type = "image/webp";
    else if (_filepath.find(".bmp") != std::string::npos ||
             _filepath.find(".BMP") != std::string::npos)
        _frame.mime_type = "image/bmp";

    _opened   = true;
    _consumed = false;
    return true;
}

void ImageSource::close() {
    _frame = MediaFrame{};
    _opened   = false;
    _consumed = false;
}

MediaFrame ImageSource::next_frame() {
    if (!_opened || _consumed) return MediaFrame{};
    _consumed = true;
    return _frame;
}

bool ImageSource::has_next() const {
    return _opened && !_consumed;
}

std::string ImageSource::description() const {
    return std::string("Image: ") + _filepath;
}

// ── VideoSource (stub) ────────────────────────────────────────

VideoSource::VideoSource(const std::string & filepath)
    : _filepath(filepath) {}
VideoSource::~VideoSource() {}
bool VideoSource::open()           { return false; }
void VideoSource::close()          {}
MediaFrame VideoSource::next_frame() { return MediaFrame{}; }
bool VideoSource::has_next() const { return false; }
std::string VideoSource::description() const {
    return std::string("Video: ") + _filepath + " (not implemented)";
}

// ── CameraSource (stub) ───────────────────────────────────────

CameraSource::CameraSource(int device_id) : _device_id(device_id) {}
CameraSource::~CameraSource() {}
bool CameraSource::open()           { return false; }
void CameraSource::close()          {}
MediaFrame CameraSource::next_frame() { return MediaFrame{}; }
bool CameraSource::has_next() const { return false; }
std::string CameraSource::description() const {
    return "Camera device " + std::to_string(_device_id) + " (not implemented)";
}

// ── DataStreamSource (stub) ───────────────────────────────────

DataStreamSource::DataStreamSource(const std::string & filepath, int batch_size)
    : _filepath(filepath), _batch_size(batch_size) {}
DataStreamSource::~DataStreamSource() {}
bool DataStreamSource::open()           { return false; }
void DataStreamSource::close()          {}
MediaFrame DataStreamSource::next_frame() { return MediaFrame{}; }
bool DataStreamSource::has_next() const { return false; }
std::string DataStreamSource::description() const {
    return std::string("Data: ") + _filepath + " (not implemented)";
}

// ── TextStreamSource (stub) ───────────────────────────────────

TextStreamSource::TextStreamSource(const std::string & filepath, int chunk_tokens)
    : _filepath(filepath), _chunk_tokens(chunk_tokens) {}
TextStreamSource::~TextStreamSource() {}
bool TextStreamSource::open()           { return false; }
void TextStreamSource::close()          {}
MediaFrame TextStreamSource::next_frame() { return MediaFrame{}; }
bool TextStreamSource::has_next() const { return false; }
std::string TextStreamSource::description() const {
    return std::string("Text: ") + _filepath + " (not implemented)";
}
