#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <set>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>

#include "core/infra/sha256.hpp"
#include "core/output/durable_file.hpp"

namespace tmv::checkpoint::wire
{

inline constexpr std::uint64_t max_elements = 1ULL << 34;
inline constexpr std::uint32_t max_string_bytes = 1U << 20;
inline constexpr std::uint64_t max_file_bytes = max_elements * 32ULL;
inline constexpr std::uint32_t max_sections = 4096;

inline std::uint64_t checksum(const std::uint8_t* data,
                              std::size_t size) noexcept
{
    constexpr std::uint64_t offset = 1469598103934665603ULL;
    constexpr std::uint64_t prime = 1099511628211ULL;
    std::uint64_t value = offset;
    for (std::size_t i = 0; i < size; ++i)
    {
        value ^= data[i];
        value *= prime;
    }
    return value;
}

inline std::uint64_t checksum_append(std::uint64_t value,
                                     const std::uint8_t* data,
                                     std::size_t size) noexcept
{
    constexpr std::uint64_t prime = 1099511628211ULL;
    for (std::size_t i = 0; i < size; ++i)
    {
        value ^= data[i];
        value *= prime;
    }
    return value;
}

template <typename UInt>
std::array<std::uint8_t, sizeof(UInt)> little_endian_bytes(UInt value) noexcept
{
    static_assert(std::is_unsigned<UInt>::value, "unsigned type required");
    std::array<std::uint8_t, sizeof(UInt)> bytes{};
    for (std::size_t byte = 0; byte < sizeof(UInt); ++byte)
    {
        bytes[byte] = static_cast<std::uint8_t>(
            (value >> (8U * byte)) & static_cast<UInt>(0xffU));
    }
    return bytes;
}

class Encoder
{
public:
    using Sink = std::function<bool(const std::uint8_t*, std::size_t)>;

    Encoder() = default;
    explicit Encoder(Sink sink) : sink_(std::move(sink)) {}

    template <typename UInt>
    void unsigned_value(UInt value)
    {
        static_assert(std::is_unsigned<UInt>::value, "unsigned type required");
        std::array<std::uint8_t, sizeof(UInt)> encoded{};
        for (std::size_t byte = 0; byte < sizeof(UInt); ++byte)
        {
            encoded[byte] = static_cast<std::uint8_t>(
                (value >> (8U * byte)) & static_cast<UInt>(0xffU));
        }
        append(encoded.data(), encoded.size());
    }

    void boolean(bool value)
    {
        unsigned_value<std::uint8_t>(value ? 1U : 0U);
    }

    void integer(std::int32_t value)
    {
        unsigned_value<std::uint32_t>(static_cast<std::uint32_t>(value));
    }

    void floating(float value)
    {
        static_assert(std::numeric_limits<float>::is_iec559,
                      "checkpoint format requires IEEE-754 float");
        std::uint32_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        unsigned_value(bits);
    }

    void floating(double value)
    {
        static_assert(std::numeric_limits<double>::is_iec559,
                      "checkpoint format requires IEEE-754 double");
        std::uint64_t bits = 0;
        std::memcpy(&bits, &value, sizeof(bits));
        unsigned_value(bits);
    }

    bool string(const std::string& value)
    {
        if (value.size() > max_string_bytes)
        {
            error_ = "string exceeds checkpoint limit";
            return false;
        }
        unsigned_value<std::uint32_t>(
            static_cast<std::uint32_t>(value.size()));
        append(reinterpret_cast<const std::uint8_t*>(value.data()), value.size());
        return true;
    }

    void floating_array(const float* values, std::size_t count)
    {
        static_assert(sizeof(float) == sizeof(std::uint32_t) && std::numeric_limits<float>::is_iec559,
                      "checkpoint format requires IEEE-754 binary32");
        if (!good() || count == 0) return;
        if (values == nullptr || count > max_file_bytes / sizeof(float) ||
            count > std::numeric_limits<std::size_t>::max() / sizeof(float))
        {
            error_ = "invalid checkpoint float array";
            return;
        }
        const std::uint32_t endian_probe = 1;
        if (*reinterpret_cast<const unsigned char*>(&endian_probe) == 1)
            append(reinterpret_cast<const std::uint8_t*>(values), count * sizeof(float));
        else
            for (std::size_t index = 0; index < count && good(); ++index)
                floating(values[index]);
    }

    void raw(const std::vector<std::uint8_t>& value)
    {
        append(value.data(), value.size());
    }

    void raw(const std::uint8_t* value, std::size_t size) { append(value, size); }

    bool good() const noexcept { return error_.empty(); }
    const std::string& error() const noexcept { return error_; }
    std::uint64_t size() const noexcept { return size_; }

    std::vector<std::uint8_t> bytes;

private:
    void append(const std::uint8_t* data, std::size_t size)
    {
        if (!error_.empty() || size == 0) return;
        if (size_ > max_file_bytes || size > max_file_bytes - size_)
        {
            error_ = "checkpoint encoder exceeds size limit";
            return;
        }
        if (sink_)
        {
            if (!sink_(data, size))
            {
                error_ = "checkpoint encoder sink failed";
                return;
            }
        }
        else
        {
            bytes.insert(bytes.end(), data, data + size);
        }
        size_ += static_cast<std::uint64_t>(size);
    }

    Sink sink_;
    std::uint64_t size_ = 0;
    std::string error_;
};

struct ByteView
{
    const std::uint8_t* begin = nullptr;
    std::size_t length = 0;

    const std::uint8_t* data() const noexcept { return begin; }
    std::size_t size() const noexcept { return length; }
    void assign(const std::uint8_t* first, const std::uint8_t* last) noexcept
    {
        begin = first;
        length = static_cast<std::size_t>(last - first);
    }
};

class Decoder
{
public:
    Decoder(const std::uint8_t* begin, std::size_t size)
        : cursor_(begin), remaining_(size)
    {
    }

    template <typename UInt>
    bool unsigned_value(UInt& value)
    {
        static_assert(std::is_unsigned<UInt>::value, "unsigned type required");
        if (remaining_ < sizeof(UInt)) return fail("truncated scalar");
        value = 0;
        for (std::size_t byte = 0; byte < sizeof(UInt); ++byte)
        {
            value |= static_cast<UInt>(cursor_[byte]) << (8U * byte);
        }
        cursor_ += sizeof(UInt);
        remaining_ -= sizeof(UInt);
        return true;
    }

    bool boolean(bool& value)
    {
        std::uint8_t encoded = 0;
        if (!unsigned_value(encoded)) return false;
        if (encoded > 1U) return fail("invalid boolean");
        value = encoded != 0;
        return true;
    }

    bool integer(std::int32_t& value)
    {
        std::uint32_t encoded = 0;
        if (!unsigned_value(encoded)) return false;
        value = static_cast<std::int32_t>(encoded);
        return true;
    }

    bool floating(float& value)
    {
        std::uint32_t bits = 0;
        if (!unsigned_value(bits)) return false;
        std::memcpy(&value, &bits, sizeof(value));
        return true;
    }

    bool floating(double& value)
    {
        std::uint64_t bits = 0;
        if (!unsigned_value(bits)) return false;
        std::memcpy(&value, &bits, sizeof(value));
        return true;
    }

    bool floating_array(float* values, std::size_t count)
    {
        static_assert(sizeof(float) == sizeof(std::uint32_t) && std::numeric_limits<float>::is_iec559,
                      "checkpoint format requires IEEE-754 binary32");
        if (!error_.empty()) return false;
        if (count == 0) return true;
        if (values == nullptr) return fail("invalid checkpoint float destination");
        if (count > remaining_ / sizeof(float)) return fail("truncated float array");
        const std::uint32_t endian_probe = 1;
        if (*reinterpret_cast<const unsigned char*>(&endian_probe) == 1)
        {
            const auto bytes = count * sizeof(float);
            std::memcpy(values, cursor_, bytes);
            cursor_ += bytes;
            remaining_ -= bytes;
            return true;
        }
        for (std::size_t index = 0; index < count; ++index)
            if (!floating(values[index])) return false;
        return true;
    }

    bool string(std::string& value)
    {
        std::uint32_t size = 0;
        if (!unsigned_value(size)) return false;
        if (size > max_string_bytes || remaining_ < size)
        {
            return fail("invalid string length");
        }
        value.assign(reinterpret_cast<const char*>(cursor_), size);
        cursor_ += size;
        remaining_ -= size;
        return true;
    }

    bool raw(std::size_t size, std::vector<std::uint8_t>& value)
    {
        if (remaining_ < size) return fail("truncated byte block");
        value.assign(cursor_, cursor_ + size);
        cursor_ += size;
        remaining_ -= size;
        return true;
    }

    bool raw(std::size_t size, ByteView& value)
    {
        if (remaining_ < size) return fail("truncated byte block");
        value = {cursor_, size};
        cursor_ += size;
        remaining_ -= size;
        return true;
    }

    bool finished() const noexcept { return remaining_ == 0; }
    std::size_t remaining() const noexcept { return remaining_; }
    const std::string& error() const noexcept { return error_; }

private:
    bool fail(const char* message)
    {
        error_ = message;
        return false;
    }

    const std::uint8_t* cursor_;
    std::size_t remaining_;
    std::string error_;
};

inline constexpr std::uint32_t section_required = 1U << 0;

template <typename Payload>
struct SectionRecord
{
    std::uint32_t id = 0;
    std::uint32_t version = 0;
    std::uint32_t flags = section_required;
    Payload payload;
};

using Section = SectionRecord<std::vector<std::uint8_t>>;
using SectionView = SectionRecord<ByteView>;

struct PublishedFileIntegrity
{
    std::uint64_t bytes = 0;
    std::uint64_t fnv1a64 = 14695981039346656037ULL;
    std::string sha256;
};

struct StreamingSection
{
    using encode_type = std::function<bool(Encoder&, std::string&)>;

    std::uint32_t id = 0;
    std::uint32_t version = 0;
    std::uint32_t flags = section_required;
    encode_type encode;
};

inline bool section_payload_size(const std::vector<Section>& sections,
                                 std::uint64_t& payload_size,
                                 std::string& error)
{
    error.clear();
    if (sections.size() > max_sections)
    {
        error = "too many checkpoint sections";
        return false;
    }
    constexpr std::uint64_t section_count_bytes = 4U;
    constexpr std::uint64_t section_header_bytes = 4U + 4U + 4U + 8U + 8U;
    payload_size = section_count_bytes;
    std::set<std::uint32_t> ids;
    for (const Section& section : sections)
    {
        if (section.id == 0 || !ids.insert(section.id).second)
        {
            error = "invalid or duplicate checkpoint section id";
            return false;
        }
        if ((section.flags & ~section_required) != 0)
        {
            error = "unsupported checkpoint section flags";
            return false;
        }
        if (section.payload.size() > max_file_bytes ||
            payload_size > max_file_bytes - section_header_bytes ||
            section.payload.size() >
                max_file_bytes - payload_size - section_header_bytes)
        {
            error = "checkpoint section exceeds size limit";
            return false;
        }
        payload_size += section_header_bytes + section.payload.size();
    }
    return true;
}

template <std::size_t N>
bool wrapped_sections_file_size(const std::vector<Section>& sections,
                                std::uint64_t& file_size,
                                std::string& error)
{
    std::uint64_t payload_size = 0;
    if (!section_payload_size(sections, payload_size, error)) return false;
    constexpr std::uint64_t outer_scalars = 4U + 8U + 8U;
    if (payload_size > max_file_bytes ||
        static_cast<std::uint64_t>(N) > max_file_bytes - payload_size ||
        outer_scalars > max_file_bytes - payload_size - static_cast<std::uint64_t>(N))
    {
        error = "checkpoint envelope exceeds size limit";
        return false;
    }
    file_size = static_cast<std::uint64_t>(N) + outer_scalars + payload_size;
    return true;
}

// Sections make the full-run format evolvable without coupling every module's
// payload version to the outer file version.
inline bool encode_sections(const std::vector<Section>& sections,
                            std::vector<std::uint8_t>& bytes,
                            std::string& error)
{
    error.clear();
    if (sections.size() > max_sections)
    {
        error = "too many checkpoint sections";
        return false;
    }
    std::set<std::uint32_t> ids;
    Encoder out;
    out.unsigned_value<std::uint32_t>(
        static_cast<std::uint32_t>(sections.size()));
    for (const Section& section : sections)
    {
        if (section.id == 0 || !ids.insert(section.id).second)
        {
            error = "invalid or duplicate checkpoint section id";
            return false;
        }
        if ((section.flags & ~section_required) != 0)
        {
            error = "unsupported checkpoint section flags";
            return false;
        }
        if (section.payload.size() > max_file_bytes)
        {
            error = "checkpoint section exceeds size limit";
            return false;
        }
        out.unsigned_value(section.id);
        out.unsigned_value(section.version);
        out.unsigned_value(section.flags);
        out.unsigned_value<std::uint64_t>(section.payload.size());
        out.unsigned_value<std::uint64_t>(
            checksum(section.payload.data(), section.payload.size()));
        out.raw(section.payload);
    }
    bytes = std::move(out.bytes);
    return true;
}

// Build the section payload directly inside its outer envelope. This preserves
// the wire bytes produced by encode_sections()+wrap() while avoiding a second
// full-size payload allocation during large simulation checkpoints.
template <std::size_t N>
bool wrap_sections(const std::array<std::uint8_t, N>& magic,
                   std::uint32_t version,
                   const std::vector<Section>& sections,
                   std::vector<std::uint8_t>& bytes,
                   std::string& error)
{
    constexpr std::uint64_t outer_scalars = 4U + 8U + 8U;
    std::uint64_t payload_size = 0;
    if (!section_payload_size(sections, payload_size, error)) return false;
    const std::uint64_t total_size =
        static_cast<std::uint64_t>(N) + outer_scalars + payload_size;
    if (total_size > std::numeric_limits<std::size_t>::max())
    {
        error = "checkpoint exceeds addressable size";
        return false;
    }

    Encoder out;
    out.bytes.reserve(static_cast<std::size_t>(total_size));
    out.bytes.insert(out.bytes.end(), magic.begin(), magic.end());
    out.unsigned_value(version);
    out.unsigned_value(payload_size);
    const std::size_t checksum_offset = out.bytes.size();
    out.unsigned_value<std::uint64_t>(0U);
    const std::size_t payload_offset = out.bytes.size();
    out.unsigned_value<std::uint32_t>(
        static_cast<std::uint32_t>(sections.size()));
    for (const Section& section : sections)
    {
        out.unsigned_value(section.id);
        out.unsigned_value(section.version);
        out.unsigned_value(section.flags);
        out.unsigned_value<std::uint64_t>(section.payload.size());
        out.unsigned_value<std::uint64_t>(
            checksum(section.payload.data(), section.payload.size()));
        out.raw(section.payload);
    }
    const std::uint64_t payload_checksum = checksum(
        out.bytes.data() + payload_offset,
        static_cast<std::size_t>(payload_size));
    for (std::size_t byte = 0; byte < sizeof(payload_checksum); ++byte)
    {
        out.bytes[checksum_offset + byte] = static_cast<std::uint8_t>(
            (payload_checksum >> (8U * byte)) & 0xffU);
    }
    bytes = std::move(out.bytes);
    return true;
}

namespace detail
{
class CheckpointFileSink
{
public:
    CheckpointFileSink(const std::filesystem::path& path, const char* label, std::string& error)
        : path_(path), label_(label), error_(error)
    {
        errno = 0;
        stream_.open(path_, std::ios::binary | std::ios::trunc);
        const int code = errno;
        if (!stream_) fail("open", code, 0);
    }

    bool good() const noexcept { return !failed_; }

    bool write(const std::uint8_t* data, std::size_t size)
    {
        if (failed_) return false;
        while (size > 0)
        {
            const auto chunk = std::min(size, static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max()));
            errno = 0;
            stream_.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(chunk));
            const int code = errno;
            if (!stream_)
            {
                fail("write", code, chunk);
                return false;
            }
            accepted_bytes_ += chunk;
            data += chunk;
            size -= chunk;
        }
        return true;
    }

    bool close()
    {
        if (stream_.is_open())
        {
            errno = 0;
            stream_.close();
            const int code = errno;
            if (!stream_ && !failed_) fail("close", code, 0);
        }
        return !failed_;
    }

private:
    void fail(const char* operation, int code, std::size_t requested)
    {
        failed_ = true;
        error_ = std::string(label_) + " checkpoint " + operation + " failed: path='" + path_.string() +
            "', stream_bytes_accepted=" + std::to_string(accepted_bytes_) +
            ", request_bytes=" + std::to_string(requested) + ", errno=" + std::to_string(code) +
            " (" + (code ? std::generic_category().message(code) : "OS error unavailable") +
            "), stream_state=" + std::to_string(static_cast<unsigned>(stream_.rdstate()));
    }

    std::filesystem::path path_;
    const char* label_;
    std::string& error_;
    std::ofstream stream_;
    std::uint64_t accepted_bytes_ = 0;
    bool failed_ = false;
};
} // namespace detail

// Publish an already encoded section set without materializing a second
// full-size envelope. The byte stream is exactly the one produced by
// wrap_sections(); only the ownership of the final write is different.
template <std::size_t N>
bool write_sections_atomic(const std::filesystem::path& path,
                           const std::array<std::uint8_t, N>& magic,
                           std::uint32_t version,
                           const std::vector<Section>& sections,
                           std::string& error,
                           const char* label,
                           PublishedFileIntegrity* published_integrity = nullptr,
                           std::uint64_t minimum_free_bytes_after_publication = 0)
{
    std::uint64_t payload_size = 0;
    if (!section_payload_size(sections, payload_size, error)) return false;

    std::uint64_t file_size = 0;
    if (!wrapped_sections_file_size<N>(sections, file_size, error)) return false;
    if (minimum_free_bytes_after_publication > max_file_bytes - file_size)
    {
        error = std::string(label) + " checkpoint disk requirement exceeds the representable limit";
        return false;
    }
    const std::uint64_t required_free_bytes = file_size + minimum_free_bytes_after_publication;
    std::error_code space_error;
    const auto destination = path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
    const auto space = std::filesystem::space(destination, space_error);
    if (space_error)
    {
        error = std::string("cannot inspect free space for ") + label + " checkpoint: " + space_error.message();
        return false;
    }
    if (space.available < required_free_bytes)
    {
        error = std::string("insufficient free space for ") + label +
                " checkpoint atomic publication: required=" +
                std::to_string(required_free_bytes) + ", available=" +
                std::to_string(space.available) + ", artifact=" +
                std::to_string(file_size) + ", retained_reserve=" +
                std::to_string(minimum_free_bytes_after_publication);
        return false;
    }

    constexpr std::uint64_t checksum_offset = 1469598103934665603ULL;
    std::uint64_t payload_checksum = checksum_offset;
    std::array<std::uint64_t, max_sections> section_checksums{};
    const auto section_count = little_endian_bytes<std::uint32_t>(
        static_cast<std::uint32_t>(sections.size()));
    payload_checksum = checksum_append(
        payload_checksum, section_count.data(), section_count.size());
    for (std::size_t index = 0; index < sections.size(); ++index)
    {
        const Section& section = sections[index];
        section_checksums[index] = checksum(
            section.payload.data(), section.payload.size());
        const auto id = little_endian_bytes(section.id);
        const auto section_version = little_endian_bytes(section.version);
        const auto flags = little_endian_bytes(section.flags);
        const auto size = little_endian_bytes<std::uint64_t>(section.payload.size());
        const auto section_checksum = little_endian_bytes<std::uint64_t>(
            section_checksums[index]);
        for (const auto& scalar : {id, section_version, flags})
            payload_checksum = checksum_append(
                payload_checksum, scalar.data(), scalar.size());
        payload_checksum = checksum_append(
            payload_checksum, size.data(), size.size());
        payload_checksum = checksum_append(
            payload_checksum, section_checksum.data(), section_checksum.size());
        payload_checksum = checksum_append(
            payload_checksum, section.payload.data(), section.payload.size());
    }

    const std::filesystem::path temporary = path.string() + ".tmp";
    detail::CheckpointFileSink out(temporary, label, error);
    if (!out.good()) return false;
    std::uint64_t published_bytes = 0;
    std::uint64_t published_fnv1a64 = 14695981039346656037ULL;
    nimbus::crypto::Sha256 published_sha256;
    const auto write_bytes = [&](const std::uint8_t* data,
                                 std::size_t size) {
        constexpr std::size_t max_chunk =
            static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max());
        while (size > 0)
        {
            const std::size_t chunk = std::min(size, max_chunk);
            if (!out.write(data, chunk)) return false;
            for (std::size_t index = 0; index < chunk; ++index)
            {
                published_fnv1a64 ^= data[index];
                published_fnv1a64 *= 1099511628211ULL;
            }
            published_sha256.update(data, chunk);
            published_bytes += static_cast<std::uint64_t>(chunk);
            data += chunk;
            size -= chunk;
        }
        return true;
    };
    const auto encoded_version = little_endian_bytes(version);
    const auto encoded_payload_size = little_endian_bytes(payload_size);
    const auto encoded_payload_checksum = little_endian_bytes(payload_checksum);
    bool wrote = write_bytes(magic.data(), magic.size()) &&
                 write_bytes(encoded_version.data(), encoded_version.size()) &&
                 write_bytes(encoded_payload_size.data(), encoded_payload_size.size()) &&
                 write_bytes(encoded_payload_checksum.data(), encoded_payload_checksum.size()) &&
                 write_bytes(section_count.data(), section_count.size());
    for (std::size_t index = 0; index < sections.size(); ++index)
    {
        const Section& section = sections[index];
        const auto id = little_endian_bytes(section.id);
        const auto section_version = little_endian_bytes(section.version);
        const auto flags = little_endian_bytes(section.flags);
        const auto size = little_endian_bytes<std::uint64_t>(section.payload.size());
        const auto section_checksum = little_endian_bytes<std::uint64_t>(
            section_checksums[index]);
        wrote = wrote && write_bytes(id.data(), id.size()) &&
                write_bytes(section_version.data(), section_version.size()) &&
                write_bytes(flags.data(), flags.size()) &&
                write_bytes(size.data(), size.size()) &&
                write_bytes(section_checksum.data(), section_checksum.size()) &&
                write_bytes(section.payload.data(), section.payload.size());
        if (!wrote) break;
    }
    const bool closed = out.close();
    if (!wrote || !closed)
    {
        if (error.empty()) error = std::string("failed writing ") + label + " checkpoint";
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return false;
    }
    if (!nimbus::output::sync_file_to_disk(temporary, error))
    {
        error = std::string("cannot make ") + label
              + " checkpoint durable: " + error;
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return false;
    }
    if (!nimbus::output::atomic_replace_file(temporary, path, error))
    {
        error = std::string("cannot publish ") + label
              + " checkpoint: " + error;
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return false;
    }
    const auto parent = path.has_parent_path()
        ? path.parent_path() : std::filesystem::path(".");
    if (!nimbus::output::sync_directory_to_disk(parent, error))
    {
        error = std::string("cannot make published ") + label
              + " checkpoint durable: " + error;
        return false;
    }
    if (published_integrity != nullptr)
    {
        published_integrity->bytes = published_bytes;
        published_integrity->fnv1a64 = published_fnv1a64;
        published_integrity->sha256 = published_sha256.hex_digest();
    }
    return true;
}

template <std::size_t N>
bool write_streaming_sections_atomic(
    const std::filesystem::path& path,
    const std::array<std::uint8_t, N>& magic,
    std::uint32_t version,
    const std::vector<StreamingSection>& sections,
    std::string& error,
    const char* label,
    PublishedFileIntegrity* published_integrity = nullptr,
    std::uint64_t minimum_free_bytes_after_publication = 0)
{
    error.clear();
    if (sections.size() > max_sections)
    {
        error = "too many checkpoint sections";
        return false;
    }
    std::set<std::uint32_t> ids;
    std::array<std::uint64_t, max_sections> section_sizes{};
    std::array<std::uint64_t, max_sections> section_checksums{};
    constexpr std::uint64_t fnv_offset = 1469598103934665603ULL;
    constexpr std::uint64_t section_header_bytes = 4U + 4U + 4U + 8U + 8U;
    std::uint64_t payload_size = 4U;
    for (std::size_t index = 0; index < sections.size(); ++index)
    {
        const auto& section = sections[index];
        if (section.id == 0 || !ids.insert(section.id).second)
        {
            error = "invalid or duplicate checkpoint section id";
            return false;
        }
        if ((section.flags & ~section_required) != 0 || !section.encode)
        {
            error = "unsupported checkpoint section flags or encoder";
            return false;
        }
        std::uint64_t measured_checksum = fnv_offset;
        Encoder measured([&](const std::uint8_t* data, std::size_t size) {
            measured_checksum = checksum_append(measured_checksum, data, size);
            return true;
        });
        if (!section.encode(measured, error) || !measured.good())
        {
            if (error.empty()) error = measured.error();
            return false;
        }
        section_sizes[index] = measured.size();
        section_checksums[index] = measured_checksum;
        if (section_sizes[index] > max_file_bytes ||
            payload_size > max_file_bytes - section_header_bytes ||
            section_sizes[index] > max_file_bytes - payload_size - section_header_bytes)
        {
            error = "checkpoint section exceeds size limit";
            return false;
        }
        payload_size += section_header_bytes + section_sizes[index];
    }

    const auto section_count = little_endian_bytes<std::uint32_t>(
        static_cast<std::uint32_t>(sections.size()));
    std::uint64_t payload_checksum = checksum_append(
        fnv_offset, section_count.data(), section_count.size());
    for (std::size_t index = 0; index < sections.size(); ++index)
    {
        const auto& section = sections[index];
        const auto id = little_endian_bytes(section.id);
        const auto section_version = little_endian_bytes(section.version);
        const auto flags = little_endian_bytes(section.flags);
        const auto size = little_endian_bytes(section_sizes[index]);
        const auto section_checksum = little_endian_bytes(section_checksums[index]);
        for (const auto& scalar : {id, section_version, flags})
            payload_checksum = checksum_append(payload_checksum, scalar.data(), scalar.size());
        payload_checksum = checksum_append(payload_checksum, size.data(), size.size());
        payload_checksum = checksum_append(
            payload_checksum, section_checksum.data(), section_checksum.size());
        std::uint64_t observed_size = 0;
        std::uint64_t observed_checksum = fnv_offset;
        Encoder checksum_encoder([&](const std::uint8_t* data, std::size_t chunk) {
            observed_size += static_cast<std::uint64_t>(chunk);
            auto section_hash = observed_checksum;
            auto payload_hash = payload_checksum;
            for (std::size_t byte = 0; byte < chunk; ++byte)
            {
                const auto value = data[byte];
                section_hash = (section_hash ^ value) * 1099511628211ULL;
                payload_hash = (payload_hash ^ value) * 1099511628211ULL;
            }
            observed_checksum = section_hash;
            payload_checksum = payload_hash;
            return true;
        });
        if (!section.encode(checksum_encoder, error) || !checksum_encoder.good() ||
            observed_size != section_sizes[index] ||
            observed_checksum != section_checksums[index])
        {
            if (error.empty()) error = checksum_encoder.good()
                ? "checkpoint section encoder is not deterministic"
                : checksum_encoder.error();
            return false;
        }
    }

    constexpr std::uint64_t outer_scalars = 4U + 8U + 8U;
    if (payload_size > max_file_bytes || static_cast<std::uint64_t>(N) > max_file_bytes - payload_size ||
        outer_scalars > max_file_bytes - payload_size - static_cast<std::uint64_t>(N))
    {
        error = "checkpoint envelope exceeds size limit";
        return false;
    }
    const std::uint64_t file_size = static_cast<std::uint64_t>(N) + outer_scalars + payload_size;
    if (minimum_free_bytes_after_publication > max_file_bytes - file_size)
    {
        error = std::string(label) + " checkpoint disk requirement exceeds the representable limit";
        return false;
    }
    const std::uint64_t required_free_bytes = file_size + minimum_free_bytes_after_publication;
    std::error_code space_error;
    const auto destination = path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
    const auto space = std::filesystem::space(destination, space_error);
    if (space_error || space.available < required_free_bytes)
    {
        error = space_error
            ? std::string("cannot inspect free space for ") + label + " checkpoint: " + space_error.message()
            : std::string("insufficient free space for ") + label +
                " checkpoint atomic publication: required=" + std::to_string(required_free_bytes) +
                ", available=" + std::to_string(space.available) +
                ", artifact=" + std::to_string(file_size) +
                ", retained_reserve=" + std::to_string(minimum_free_bytes_after_publication);
        return false;
    }

    const std::filesystem::path temporary = path.string() + ".tmp";
    detail::CheckpointFileSink out(temporary, label, error);
    if (!out.good()) return false;
    std::uint64_t published_bytes = 0;
    std::uint64_t published_fnv1a64 = 14695981039346656037ULL;
    nimbus::crypto::Sha256 published_sha256;
    const auto write_bytes = [&](const std::uint8_t* data, std::size_t size,
                                 std::uint64_t* section_checksum = nullptr) {
        constexpr std::size_t max_chunk =
            static_cast<std::size_t>(std::numeric_limits<std::streamsize>::max());
        while (size > 0)
        {
            const std::size_t chunk = std::min(size, max_chunk);
            if (!out.write(data, chunk)) return false;
            if (section_checksum != nullptr)
            {
                auto section_hash = *section_checksum;
                auto file_hash = published_fnv1a64;
                for (std::size_t byte = 0; byte < chunk; ++byte)
                {
                    const auto value = data[byte];
                    section_hash = (section_hash ^ value) * 1099511628211ULL;
                    file_hash = (file_hash ^ value) * 1099511628211ULL;
                }
                *section_checksum = section_hash;
                published_fnv1a64 = file_hash;
            }
            else
                published_fnv1a64 = checksum_append(published_fnv1a64, data, chunk);
            published_sha256.update(data, chunk);
            published_bytes += static_cast<std::uint64_t>(chunk);
            data += chunk;
            size -= chunk;
        }
        return true;
    };
    const auto encoded_version = little_endian_bytes(version);
    const auto encoded_payload_size = little_endian_bytes(payload_size);
    const auto encoded_payload_checksum = little_endian_bytes(payload_checksum);
    bool wrote = write_bytes(magic.data(), magic.size()) &&
                 write_bytes(encoded_version.data(), encoded_version.size()) &&
                 write_bytes(encoded_payload_size.data(), encoded_payload_size.size()) &&
                 write_bytes(encoded_payload_checksum.data(), encoded_payload_checksum.size()) &&
                 write_bytes(section_count.data(), section_count.size());
    for (std::size_t index = 0; wrote && index < sections.size(); ++index)
    {
        const auto& section = sections[index];
        const auto id = little_endian_bytes(section.id);
        const auto section_version = little_endian_bytes(section.version);
        const auto flags = little_endian_bytes(section.flags);
        const auto size = little_endian_bytes(section_sizes[index]);
        const auto section_checksum = little_endian_bytes(section_checksums[index]);
        wrote = write_bytes(id.data(), id.size()) &&
                write_bytes(section_version.data(), section_version.size()) &&
                write_bytes(flags.data(), flags.size()) &&
                write_bytes(size.data(), size.size()) &&
                write_bytes(section_checksum.data(), section_checksum.size());
        std::uint64_t observed_size = 0;
        std::uint64_t observed_checksum = fnv_offset;
        Encoder writer([&](const std::uint8_t* data, std::size_t chunk) {
            observed_size += static_cast<std::uint64_t>(chunk);
            return write_bytes(data, chunk, &observed_checksum);
        });
        wrote = wrote && section.encode(writer, error) && writer.good() &&
                observed_size == section_sizes[index] &&
                observed_checksum == section_checksums[index];
        if (!wrote && error.empty())
            error = writer.good() ? "checkpoint section encoder changed during publication" : writer.error();
    }
    const bool closed = out.close();
    if (!wrote || !closed)
    {
        if (error.empty()) error = std::string("failed writing ") + label + " checkpoint";
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return false;
    }
    if (!nimbus::output::sync_file_to_disk(temporary, error) ||
        !nimbus::output::atomic_replace_file(temporary, path, error))
    {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return false;
    }
    const auto parent = path.has_parent_path() ? path.parent_path() : std::filesystem::path(".");
    if (!nimbus::output::sync_directory_to_disk(parent, error)) return false;
    if (published_integrity != nullptr)
    {
        published_integrity->bytes = published_bytes;
        published_integrity->fnv1a64 = published_fnv1a64;
        published_integrity->sha256 = published_sha256.hex_digest();
    }
    return true;
}

template <typename Bytes, typename Payload>
bool decode_sections(const Bytes& bytes,
                            std::vector<SectionRecord<Payload>>& sections,
                            std::string& error)
{
    error.clear();
    Decoder in(bytes.data(), bytes.size());
    std::uint32_t count = 0;
    if (!in.unsigned_value(count) || count > max_sections)
    {
        error = in.error().empty() ? "invalid checkpoint section count"
                                   : in.error();
        return false;
    }

    std::set<std::uint32_t> ids;
    std::vector<SectionRecord<Payload>> restored;
    restored.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index)
    {
        SectionRecord<Payload> section;
        std::uint64_t size = 0;
        std::uint64_t expected_checksum = 0;
        if (!in.unsigned_value(section.id)
            || !in.unsigned_value(section.version)
            || !in.unsigned_value(section.flags)
            || !in.unsigned_value(size)
            || !in.unsigned_value(expected_checksum))
        {
            error = in.error();
            return false;
        }
        if (section.id == 0 || !ids.insert(section.id).second)
        {
            error = "invalid or duplicate checkpoint section id";
            return false;
        }
        if ((section.flags & ~section_required) != 0)
        {
            error = "unsupported checkpoint section flags";
            return false;
        }
        if (size > max_file_bytes
            || size > std::numeric_limits<std::size_t>::max()
            || !in.raw(static_cast<std::size_t>(size), section.payload))
        {
            error = in.error().empty() ? "invalid checkpoint section size"
                                       : in.error();
            return false;
        }
        if (checksum(section.payload.data(), section.payload.size())
            != expected_checksum)
        {
            error = "checkpoint section checksum mismatch";
            return false;
        }
        restored.push_back(std::move(section));
    }
    if (!in.finished())
    {
        error = "trailing checkpoint section data";
        return false;
    }
    sections = std::move(restored);
    return true;
}

template <std::size_t N>
void wrap(const std::array<std::uint8_t, N>& magic,
          std::uint32_t version,
          const std::vector<std::uint8_t>& payload,
          std::vector<std::uint8_t>& bytes)
{
    Encoder file;
    file.bytes.insert(file.bytes.end(), magic.begin(), magic.end());
    file.unsigned_value(version);
    file.unsigned_value<std::uint64_t>(payload.size());
    file.unsigned_value<std::uint64_t>(
        checksum(payload.data(), payload.size()));
    file.bytes.insert(file.bytes.end(), payload.begin(), payload.end());
    bytes = std::move(file.bytes);
}

template <std::size_t N, typename Payload>
bool unwrap(const std::vector<std::uint8_t>& bytes,
            const std::array<std::uint8_t, N>& magic,
            std::uint32_t supported_version,
            Payload& payload,
            std::string& error,
            const char* label)
{
    constexpr std::size_t scalar_header = 4 + 8 + 8;
    const std::size_t header_size = N + scalar_header;
    if (bytes.size() < header_size
        || !std::equal(magic.begin(), magic.end(), bytes.begin()))
    {
        error = std::string("invalid ") + label + " checkpoint magic";
        return false;
    }

    Decoder header(bytes.data() + N, bytes.size() - N);
    std::uint32_t version = 0;
    std::uint64_t payload_size = 0;
    std::uint64_t expected_checksum = 0;
    if (!header.unsigned_value(version)
        || !header.unsigned_value(payload_size)
        || !header.unsigned_value(expected_checksum))
    {
        error = header.error();
        return false;
    }
    if (version != supported_version)
    {
        error = std::string("unsupported ") + label + " checkpoint version";
        return false;
    }
    if (payload_size != bytes.size() - header_size)
    {
        error = std::string(label) + " checkpoint length mismatch";
        return false;
    }
    const std::uint8_t* payload_begin = bytes.data() + header_size;
    if (checksum(payload_begin, static_cast<std::size_t>(payload_size))
        != expected_checksum)
    {
        error = std::string(label) + " checkpoint checksum mismatch";
        return false;
    }
    payload.assign(payload_begin, payload_begin + payload_size);
    return true;
}

namespace detail
{
template <bool WithIntegrity, std::size_t N>
bool unwrap_section_views_impl(const std::vector<std::uint8_t>& bytes,
                               const std::array<std::uint8_t, N>& magic,
                               std::uint32_t supported_version,
                               std::vector<SectionView>& sections,
                               std::string& error,
                               const char* label,
                               PublishedFileIntegrity* integrity)
{
    error.clear();
    constexpr std::size_t header_size = N + 4 + 8 + 8;
    if (bytes.size() < header_size || !std::equal(magic.begin(), magic.end(), bytes.begin()))
    {
        error = std::string("invalid ") + label + " checkpoint magic";
        return false;
    }
    Decoder header(bytes.data() + N, bytes.size() - N);
    std::uint32_t version = 0;
    std::uint64_t payload_size = 0, expected_payload_hash = 0;
    if (!header.unsigned_value(version) || !header.unsigned_value(payload_size) ||
        !header.unsigned_value(expected_payload_hash))
    {
        error = header.error();
        return false;
    }
    if (version != supported_version || payload_size != bytes.size() - header_size)
    {
        error = version != supported_version ? std::string("unsupported ") + label + " checkpoint version"
                                            : std::string(label) + " checkpoint length mismatch";
        return false;
    }
    const auto* payload = bytes.data() + header_size;
    Decoder in(payload, static_cast<std::size_t>(payload_size));
    std::uint32_t count = 0;
    if (!in.unsigned_value(count) || count > max_sections)
    {
        error = in.error().empty() ? "invalid checkpoint section count" : in.error();
        return false;
    }
    // Wire envelope/section hashes retain their historical seed; publication uses standard FNV-1a.
    constexpr std::uint64_t wire_seed = 1469598103934665603ULL;
    constexpr std::uint64_t publication_seed = 14695981039346656037ULL;
    constexpr std::uint64_t prime = 1099511628211ULL;
    std::uint64_t payload_hash = checksum_append(wire_seed, payload, 4);
    [[maybe_unused]] std::uint64_t whole_hash = 0;
    if constexpr (WithIntegrity)
        whole_hash = checksum_append(publication_seed, bytes.data(), header_size + 4);
    std::set<std::uint32_t> ids;
    std::vector<SectionView> restored;
    restored.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index)
    {
        SectionView section;
        std::uint64_t size = 0, expected_section_hash = 0;
        const auto offset = static_cast<std::size_t>(payload_size) - in.remaining();
        if (!in.unsigned_value(section.id) || !in.unsigned_value(section.version) ||
            !in.unsigned_value(section.flags) || !in.unsigned_value(size) || !in.unsigned_value(expected_section_hash))
        {
            error = in.error();
            return false;
        }
        if (section.id == 0 || !ids.insert(section.id).second)
        {
            error = "invalid or duplicate checkpoint section id";
            return false;
        }
        if ((section.flags & ~section_required) != 0)
        {
            error = "unsupported checkpoint section flags";
            return false;
        }
        if (size > max_file_bytes || size > std::numeric_limits<std::size_t>::max() ||
            !in.raw(static_cast<std::size_t>(size), section.payload))
        {
            error = in.error().empty() ? "invalid checkpoint section size" : in.error();
            return false;
        }
        constexpr std::size_t section_header_size = 4 + 4 + 4 + 8 + 8;
        payload_hash = checksum_append(payload_hash, payload + offset, section_header_size);
        if constexpr (WithIntegrity)
            whole_hash = checksum_append(whole_hash, payload + offset, section_header_size);
        std::uint64_t section_hash = wire_seed;
        for (std::size_t n = 0; n < section.payload.size(); ++n)
        {
            const auto byte = section.payload.data()[n];
            payload_hash = (payload_hash ^ byte) * prime;
            section_hash = (section_hash ^ byte) * prime;
            if constexpr (WithIntegrity)
                whole_hash = (whole_hash ^ byte) * prime;
        }
        if (section_hash != expected_section_hash)
        {
            error = "checkpoint section checksum mismatch";
            return false;
        }
        restored.push_back(section);
    }
    if (!in.finished())
    {
        error = "trailing checkpoint section data";
        return false;
    }
    if (payload_hash != expected_payload_hash)
    {
        error = std::string(label) + " checkpoint checksum mismatch";
        return false;
    }
    if constexpr (WithIntegrity)
    {
        nimbus::crypto::Sha256 digest;
        digest.update(bytes.data(), bytes.size());
        PublishedFileIntegrity candidate{static_cast<std::uint64_t>(bytes.size()), whole_hash, digest.hex_digest()};
        static_assert(std::is_nothrow_move_assignable_v<PublishedFileIntegrity>);
        *integrity = std::move(candidate);
    }
    sections = std::move(restored);
    return true;
}
} // namespace detail

// Returned views borrow the input. Failed validation preserves both outputs.
template <std::size_t N>
bool unwrap_section_views(const std::vector<std::uint8_t>& bytes,
                          const std::array<std::uint8_t, N>& magic,
                          std::uint32_t supported_version,
                          std::vector<SectionView>& sections,
                          std::string& error,
                          const char* label,
                          PublishedFileIntegrity* integrity = nullptr)
{
    return integrity != nullptr
        ? detail::unwrap_section_views_impl<true>(bytes, magic, supported_version, sections, error, label, integrity)
        : detail::unwrap_section_views_impl<false>(bytes, magic, supported_version, sections, error, label, nullptr);
}

inline bool write_atomic(const std::filesystem::path& path,
                         const std::vector<std::uint8_t>& bytes,
                         std::string& error,
                         const char* label)
{
    const std::filesystem::path temporary = path.string() + ".tmp";
    detail::CheckpointFileSink out(temporary, label, error);
    if (!out.good()) return false;
    const bool wrote = out.write(bytes.data(), bytes.size());
    const bool closed = out.close();
    if (!wrote || !closed)
    {
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return false;
    }
    if (!nimbus::output::atomic_replace_file(temporary, path, error))
    {
        error = std::string("cannot publish ") + label
              + " checkpoint: " + error;
        std::error_code ignored;
        std::filesystem::remove(temporary, ignored);
        return false;
    }
    return true;
}

inline bool read_file(const std::filesystem::path& path,
                      std::vector<std::uint8_t>& bytes,
                      std::string& error,
                      const char* label,
                      PublishedFileIntegrity* integrity = nullptr)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
    {
        error = std::string("cannot open ") + label + " checkpoint";
        return false;
    }
    const std::streamoff length = in.tellg();
    if (length < 0 || static_cast<std::uint64_t>(length) > max_file_bytes)
    {
        error = std::string("invalid ") + label + " checkpoint file size";
        return false;
    }
    bytes.resize(static_cast<std::size_t>(length));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(bytes.data()), length);
    if (!in)
    {
        error = std::string("failed reading ") + label + " checkpoint";
        return false;
    }
    if (integrity != nullptr)
    {
        std::uint64_t fnv1a64 = 14695981039346656037ULL;
        for (const std::uint8_t byte : bytes)
        {
            fnv1a64 ^= byte;
            fnv1a64 *= 1099511628211ULL;
        }
        nimbus::crypto::Sha256 sha256;
        sha256.update(bytes.data(), bytes.size());
        integrity->bytes = static_cast<std::uint64_t>(bytes.size());
        integrity->fnv1a64 = fnv1a64;
        integrity->sha256 = sha256.hex_digest();
    }
    return true;
}

} // namespace tmv::checkpoint::wire
