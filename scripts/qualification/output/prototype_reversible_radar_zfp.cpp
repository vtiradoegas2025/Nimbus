#include <zfp.h>
#include <zfp/bitstream.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <sys/resource.h>

namespace
{

constexpr std::size_t k_dim0 = 12;
constexpr std::size_t k_dim1 = 12;
constexpr std::size_t k_dim2 = 48;
constexpr std::size_t k_brick_depth = 32;

std::vector<float> read_exact_fixture(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("cannot open NPY reference");
    input.seekg(0, std::ios::end);
    const auto file_bytes = input.tellg();
    constexpr auto raw_bytes = k_dim0 * k_dim1 * k_dim2 * sizeof(float);
    if (file_bytes < static_cast<std::streamoff>(raw_bytes + 16))
        throw std::runtime_error("NPY reference is too short");
    input.seekg(0);
    std::array<char, 256> header{};
    input.read(header.data(), static_cast<std::streamsize>(header.size()));
    const std::string prefix(header.data(), static_cast<std::size_t>(input.gcount()));
    if (prefix.compare(0, 6, "\x93NUMPY") != 0 ||
        prefix.find("'<f4'") == std::string::npos ||
        prefix.find("False") == std::string::npos ||
        prefix.find("(12, 12, 48)") == std::string::npos)
        throw std::runtime_error("NPY reference differs from exact active fixture");
    input.clear();
    input.seekg(-static_cast<std::streamoff>(raw_bytes), std::ios::end);
    std::vector<float> values(k_dim0 * k_dim1 * k_dim2);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(raw_bytes));
    if (!input)
        throw std::runtime_error("NPY reference has truncated payload");
    return values;
}

struct Result
{
    std::size_t payload_bytes = 0;
    std::size_t maximum_size_api_bytes = 0;
    std::size_t maximum_buffer_bytes = 0;
    std::size_t brick_count = 0;
    bool byte_identical = false;
    bool within_api_maximum_size = true;
};

Result reversible_bricks(const std::vector<float>& source,
                         std::vector<std::vector<unsigned char>>* emitted = nullptr)
{
    std::vector<float> reconstructed(source.size(), 0.0f);
    Result result;
    for (std::size_t origin = 0; origin < k_dim2; origin += k_brick_depth)
    {
        const std::size_t depth = std::min(k_brick_depth, k_dim2 - origin);
        auto* field = zfp_field_3d(
            const_cast<float*>(source.data() + origin), zfp_type_float,
            depth, k_dim1, k_dim0);
        auto* stream = zfp_stream_open(nullptr);
        if (!field || !stream)
            throw std::runtime_error("cannot allocate reversible ZFP controls");
        zfp_field_set_stride_3d(field, 1, k_dim2, k_dim1 * k_dim2);
        zfp_stream_set_reversible(stream);
        const std::size_t maximum = zfp_stream_maximum_size(stream, field);
        const std::size_t capacity = std::max(maximum, depth * k_dim0 * k_dim1 * sizeof(float) * 4 + 1024);
        std::vector<unsigned char> buffer(capacity);
        auto* bits = stream_open(buffer.data(), buffer.size());
        if (!bits)
            throw std::runtime_error("cannot allocate reversible ZFP bitstream");
        zfp_stream_set_bit_stream(stream, bits);
        zfp_stream_rewind(stream);
        const std::size_t compressed = zfp_compress(stream, field);
        if (compressed == 0 || compressed > capacity)
            throw std::runtime_error("reversible ZFP compression failed");
        if (emitted != nullptr)
            emitted->emplace_back(buffer.begin(), buffer.begin() + compressed);
        zfp_field_set_pointer(field, reconstructed.data() + origin);
        zfp_stream_rewind(stream);
        if (zfp_decompress(stream, field) == 0)
            throw std::runtime_error("reversible ZFP decompression failed");
        stream_close(bits);
        zfp_stream_close(stream);
        zfp_field_free(field);
        result.payload_bytes += compressed;
        result.maximum_size_api_bytes = std::max(result.maximum_size_api_bytes, maximum);
        result.maximum_buffer_bytes = std::max(result.maximum_buffer_bytes, capacity);
        result.within_api_maximum_size = result.within_api_maximum_size && compressed <= maximum;
        ++result.brick_count;
    }
    result.byte_identical = std::memcmp(
        source.data(), reconstructed.data(), source.size() * sizeof(float)) == 0;
    return result;
}

int full_brick_bound()
{
    constexpr std::size_t depth = 32;
    constexpr std::size_t rows = 64;
    constexpr std::size_t columns = 64;
    std::vector<float> source(depth * rows * columns, 1.0f);
    std::vector<float> decoded(source.size(), 0.0f);
    auto* field = zfp_field_3d(source.data(), zfp_type_float, depth, rows, columns);
    auto* stream = zfp_stream_open(nullptr);
    if (!field || !stream)
        throw std::runtime_error("cannot allocate full-brick controls");
    zfp_stream_set_reversible(stream);
    const std::size_t maximum = zfp_stream_maximum_size(stream, field);
    std::vector<unsigned char> buffer(maximum);
    auto* bits = stream_open(buffer.data(), buffer.size());
    if (!bits)
        throw std::runtime_error("cannot allocate full-brick bitstream");
    zfp_stream_set_bit_stream(stream, bits);
    zfp_stream_rewind(stream);
    const std::size_t compressed = zfp_compress(stream, field);
    if (compressed == 0 || compressed > maximum)
        throw std::runtime_error("full-brick reversible compression exceeded bound");
    zfp_field_set_pointer(field, decoded.data());
    zfp_stream_rewind(stream);
    if (zfp_decompress(stream, field) == 0)
        throw std::runtime_error("full-brick reversible decode failed");
    const bool exact = std::memcmp(source.data(), decoded.data(),
                                   source.size() * sizeof(float)) == 0;
    stream_close(bits);
    zfp_stream_close(stream);
    zfp_field_free(field);
    std::cout << "{\"zfpVersion\":\"" << ZFP_VERSION_STRING
              << "\",\"shape\":[64,64,32],\"rawBytes\":"
              << source.size() * sizeof(float)
              << ",\"maximumSizeApiBytes\":" << maximum
              << ",\"constantPayloadBytes\":" << compressed
              << ",\"readerBrickLimitBytes\":" << 4ULL * 1024ULL * 1024ULL
              << ",\"parallelWorkerCodecReserveBytes\":" << 2ULL * 1024ULL * 1024ULL
              << ",\"byteIdentical\":" << (exact ? "true" : "false") << "}\n";
    return exact ? 0 : 1;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc == 6 && std::strcmp(argv[1], "--emit-bricks") == 0)
    {
        try
        {
            const auto source = read_exact_fixture(argv[4]);
            std::vector<std::vector<unsigned char>> payloads;
            const auto result = reversible_bricks(source, &payloads);
            if (!result.byte_identical || payloads.size() != 2)
                throw std::runtime_error("emitted reversible bricks are not bitwise exact");
            const std::filesystem::path directory(argv[5]);
            if (!std::filesystem::create_directories(directory))
                throw std::runtime_error("emitted brick directory already exists");
            for (std::size_t index = 0; index < payloads.size(); ++index)
            {
                std::ofstream file(directory / ("brick_" + std::to_string(index) + ".bin"),
                                   std::ios::binary);
                file.write(reinterpret_cast<const char*>(payloads[index].data()),
                           static_cast<std::streamsize>(payloads[index].size()));
                if (!file)
                    throw std::runtime_error("cannot write reversible test brick");
            }
            std::cout << "{\"field\":\"" << argv[2] << "\",\"step\":\"" << argv[3]
                      << "\",\"brickCount\":" << payloads.size()
                      << ",\"payloadBytes\":" << result.payload_bytes
                      << ",\"byteIdentical\":true}\n";
            return 0;
        }
        catch (const std::exception& error)
        {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
    if (argc == 2 && std::strcmp(argv[1], "--full-brick-bound") == 0)
    {
        try
        {
            return full_brick_bound();
        }
        catch (const std::exception& error)
        {
            std::cerr << error.what() << '\n';
            return 1;
        }
    }
    if (argc != 4 && argc != 5)
    {
        std::cerr << "usage: prototype_reversible_radar_zfp FIELD STEP NPY [PREVIOUS_NPY]\n";
        return 2;
    }
    try
    {
        const auto source = read_exact_fixture(argv[3]);
        const auto result = reversible_bricks(source);
        std::size_t simple_delta_mismatch_count = 0;
        if (argc == 5)
        {
            const auto previous = read_exact_fixture(argv[4]);
            for (std::size_t index = 0; index < source.size(); ++index)
            {
                const float residual = static_cast<float>(
                    static_cast<double>(source[index]) - static_cast<double>(previous[index]));
                const float reconstructed = static_cast<float>(
                    static_cast<double>(previous[index]) + static_cast<double>(residual));
                if (std::memcmp(&source[index], &reconstructed, sizeof(float)) != 0)
                    ++simple_delta_mismatch_count;
            }
        }
        rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) != 0)
            throw std::runtime_error("cannot read prototype peak resident memory");
#if defined(__APPLE__)
        const auto peak_resident_bytes = static_cast<std::size_t>(usage.ru_maxrss);
#else
        const auto peak_resident_bytes = static_cast<std::size_t>(usage.ru_maxrss) * 1024;
#endif
        std::cout << "{\"field\":\"" << argv[1] << "\",\"step\":\"" << argv[2]
                  << "\",\"zfpVersion\":\"" << ZFP_VERSION_STRING
                  << "\",\"rawBytes\":" << source.size() * sizeof(float)
                  << ",\"brickCount\":" << result.brick_count
                  << ",\"reversiblePayloadBytes\":" << result.payload_bytes
                  << ",\"estimatedNimbusV3ContainerBytes\":"
                  << 44 + 8 * result.brick_count + result.payload_bytes
                  << ",\"maximumSizeApiBytes\":" << result.maximum_size_api_bytes
                  << ",\"maximumBufferBytes\":" << result.maximum_buffer_bytes
                  << ",\"estimatedPeakOwnedBytes\":"
                  << 2 * source.size() * sizeof(float) + result.maximum_buffer_bytes
                  << ",\"peakResidentBytes\":" << peak_resident_bytes
                  << ",\"withinApiMaximumSize\":"
                  << (result.within_api_maximum_size ? "true" : "false")
                  << ",\"simpleDeltaFloatMismatchCount\":" << simple_delta_mismatch_count
                  << ",\"byteIdentical\":" << (result.byte_identical ? "true" : "false")
                  << "}\n";
        return result.byte_identical ? 0 : 1;
    }
    catch (const std::exception& error)
    {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
