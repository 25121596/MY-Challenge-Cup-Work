#pragma once

// ── Unified multimodal input abstraction ──────────────────────
//
// MediaSource provides a common interface for feeding different kinds
// of media into the inference pipeline.  Each concrete source produces
// a sequence of MediaFrame objects; the engine consumes them via
// tokenize_with_media().
//
// Currently implemented:
//   ImageSource  — static images (PNG, JPG, WEBP, BMP)
//
// Reserved for future phases:
//   VideoSource       — video file frame extraction
//   CameraSource      — live camera capture
//   DataStreamSource  — structured data (CSV, JSON) → text injection
//   TextStreamSource  — large document chunked injection
//   AudioSource       — audio input for speech models

#include <cstdint>
#include <string>
#include <vector>

// ── Media types ───────────────────────────────────────────────

enum class MediaType {
    Image,
    Video,
    Audio,
    Data,
    Text
};

// ── A single media frame ──────────────────────────────────────

struct MediaFrame {
    MediaType type = MediaType::Image;

    // Raw pixel data (for Image/Video) or sample data (for Audio)
    std::vector<uint8_t> data;

    int width    = 0;       // image/frame width in pixels
    int height   = 0;       // image/frame height in pixels
    int channels = 3;       // RGB=3, RGBA=4, Gray=1

    int64_t timestamp_us = 0;   // for video/audio sync
    std::string mime_type;      // "image/png", "video/mp4", etc.
    std::string annotation;     // optional alt-text / caption

    bool empty() const { return data.empty() && width == 0 && height == 0; }
};

// ── Abstract media source ─────────────────────────────────────

class MediaSource {
public:
    virtual ~MediaSource() = default;

    // Open the source. Returns false on failure.
    virtual bool open() = 0;

    // Close and release resources.
    virtual void close() = 0;

    // Get the next frame. Returns an empty MediaFrame when exhausted.
    virtual MediaFrame next_frame() = 0;

    // Whether more frames are available (without consuming them).
    virtual bool has_next() const = 0;

    // What type of media this source produces.
    virtual MediaType type() const = 0;

    // Human-readable description for UI display.
    virtual std::string description() const = 0;
};

// ── ImageSource — static image file ───────────────────────────
//
// Reads a single image from disk.  Uses stb_image.h internally,
// which is already vendored in llama.cpp's vendor/stb/ directory.

class ImageSource : public MediaSource {
public:
    explicit ImageSource(const std::string & filepath);
    ~ImageSource() override;

    bool        open() override;
    void        close() override;
    MediaFrame  next_frame() override;
    bool        has_next() const override;
    MediaType   type() const override { return MediaType::Image; }
    std::string description() const override;

private:
    std::string _filepath;
    bool        _opened    = false;
    bool        _consumed  = false;
    MediaFrame  _frame;
};

// ── Reserved interfaces (empty stubs for future phases) ───────
//
// These are declared but not implemented.  They document the
// interface contract for future contributors.

class VideoSource : public MediaSource {
public:
    explicit VideoSource(const std::string & filepath);
    ~VideoSource() override;
    bool        open() override;      // TODO: integrate FFmpeg / libmpv
    void        close() override;
    MediaFrame  next_frame() override;
    bool        has_next() const override;
    MediaType   type() const override { return MediaType::Video; }
    std::string description() const override;
private:
    std::string _filepath;
};

class CameraSource : public MediaSource {
public:
    explicit CameraSource(int device_id = 0);
    ~CameraSource() override;
    bool        open() override;      // TODO: platform-specific capture
    void        close() override;
    MediaFrame  next_frame() override;
    bool        has_next() const override;
    MediaType   type() const override { return MediaType::Video; }
    std::string description() const override;
private:
    int _device_id = 0;
};

class DataStreamSource : public MediaSource {
public:
    // filepath: path to a CSV or JSON-lines file
    // batch_size: how many rows/lines to inject per frame
    explicit DataStreamSource(const std::string & filepath, int batch_size = 1);
    ~DataStreamSource() override;
    bool        open() override;
    void        close() override;
    MediaFrame  next_frame() override;  // each frame = serialized batch as text
    bool        has_next() const override;
    MediaType   type() const override { return MediaType::Data; }
    std::string description() const override;
private:
    std::string _filepath;
    int _batch_size = 1;
};

class TextStreamSource : public MediaSource {
public:
    // filepath: large text file to inject chunk-by-chunk
    // chunk_tokens: approx tokens per chunk
    explicit TextStreamSource(const std::string & filepath, int chunk_tokens = 512);
    ~TextStreamSource() override;
    bool        open() override;
    void        close() override;
    MediaFrame  next_frame() override;  // each frame = one text chunk
    bool        has_next() const override;
    MediaType   type() const override { return MediaType::Text; }
    std::string description() const override;
private:
    std::string _filepath;
    int _chunk_tokens = 512;
};
