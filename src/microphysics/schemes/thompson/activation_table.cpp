#include "activation_table.hpp"

#include "core/infra/sha256.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace tmv::thompson
{
namespace
{

constexpr char magic[] = "NTCACT1";
constexpr std::size_t digest_size = 32;
constexpr std::array<double, ActivationTableMetadata::aerosol_counts> aerosol_axis =
    {10.0, 31.6, 100.0, 316.0, 1000.0, 3160.0, 10000.0};
constexpr std::array<double, ActivationTableMetadata::updraft_speeds> updraft_axis =
    {0.01, 0.0316, 0.1, 0.316, 1.0, 3.16, 10.0, 31.6, 100.0};
constexpr std::array<double, ActivationTableMetadata::temperatures> temperature_axis =
    {243.15, 253.15, 263.15, 273.15, 283.15, 293.15, 303.15};

std::uint32_t big_endian_u32(const std::uint8_t* bytes)
{
    return (static_cast<std::uint32_t>(bytes[0]) << 24U)
           | (static_cast<std::uint32_t>(bytes[1]) << 16U)
           | (static_cast<std::uint32_t>(bytes[2]) << 8U)
           | static_cast<std::uint32_t>(bytes[3]);
}

std::uint32_t little_endian_u32(const std::uint8_t* bytes)
{
    return static_cast<std::uint32_t>(bytes[0])
           | (static_cast<std::uint32_t>(bytes[1]) << 8U)
           | (static_cast<std::uint32_t>(bytes[2]) << 16U)
           | (static_cast<std::uint32_t>(bytes[3]) << 24U);
}

float float_from_bits(std::uint32_t bits)
{
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

template<typename Consumer>
void consume_u32(std::uint32_t value, Consumer&& consume)
{
    const std::array<std::uint8_t, 4> bytes = {
        static_cast<std::uint8_t>(value),
        static_cast<std::uint8_t>(value >> 8U),
        static_cast<std::uint8_t>(value >> 16U),
        static_cast<std::uint8_t>(value >> 24U)};
    consume(bytes.data(), bytes.size());
}

template<typename Consumer>
void consume_float(float value, Consumer&& consume)
{
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    consume_u32(bits, std::forward<Consumer>(consume));
}

std::string payload_hash(const ActivationTable& table)
{
    nimbus::crypto::Sha256 hash;
    for (float value : table.fractions)
        consume_float(value, [&](const void* bytes, std::size_t count) {
            hash.update(bytes, count);
        });
    return hash.hex_digest();
}

std::array<std::uint8_t, digest_size> digest_bytes(const char* hex)
{
    std::array<std::uint8_t, digest_size> result{};
    auto nibble = [](char value) -> int {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        if (value >= 'A' && value <= 'F') return value - 'A' + 10;
        return -1;
    };
    if (std::strlen(hex) != result.size() * 2)
        throw std::invalid_argument("invalid activation-table SHA-256 length");
    for (std::size_t index = 0; index < result.size(); ++index)
    {
        const int high = nibble(hex[index * 2]);
        const int low = nibble(hex[index * 2 + 1]);
        if (high < 0 || low < 0)
            throw std::invalid_argument("invalid activation-table SHA-256");
        result[index] = static_cast<std::uint8_t>((high << 4) | low);
    }
    return result;
}

std::string hex_digest(const std::array<std::uint8_t, digest_size>& bytes)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string result(bytes.size() * 2, '0');
    for (std::size_t index = 0; index < bytes.size(); ++index)
    {
        result[index * 2] = digits[bytes[index] >> 4];
        result[index * 2 + 1] = digits[bytes[index] & 15U];
    }
    return result;
}

bool read_exact(std::istream& in, void* destination, std::size_t size)
{
    in.read(static_cast<char*>(destination), static_cast<std::streamsize>(size));
    return in.good();
}

bool read_u32(std::istream& in, std::uint32_t& value)
{
    std::array<std::uint8_t, 4> bytes{};
    if (!read_exact(in, bytes.data(), bytes.size())) return false;
    value = little_endian_u32(bytes.data());
    return true;
}

std::vector<std::uint8_t> read_file(const std::filesystem::path& path,
                                    std::string& error)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
    {
        error = "unable to open WRF Thompson activation table";
        return {};
    }
    const auto end = in.tellg();
    if (end < 0)
    {
        error = "unable to size WRF Thompson activation table";
        return {};
    }
    std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
    in.seekg(0);
    if (!read_exact(in, bytes.data(), bytes.size()))
    {
        error = "unable to read WRF Thompson activation table";
        return {};
    }
    return bytes;
}

std::string bytes_hash(const std::vector<std::uint8_t>& bytes)
{
    nimbus::crypto::Sha256 hash;
    hash.update(bytes.data(), bytes.size());
    return hash.hex_digest();
}

} // namespace

std::size_t activation_flatten(std::size_t aerosol_count,
                               std::size_t updraft_speed,
                               std::size_t temperature,
                               std::size_t mean_radius,
                               std::size_t hygroscopicity)
{
    if (aerosol_count >= ActivationTableMetadata::aerosol_counts
        || updraft_speed >= ActivationTableMetadata::updraft_speeds
        || temperature >= ActivationTableMetadata::temperatures
        || mean_radius >= ActivationTableMetadata::mean_radii
        || hygroscopicity >= ActivationTableMetadata::hygroscopicities)
    {
        throw std::out_of_range("Thompson activation-table index out of range");
    }
    std::size_t index = hygroscopicity;
    index = index * ActivationTableMetadata::mean_radii + mean_radius;
    index = index * ActivationTableMetadata::temperatures + temperature;
    index = index * ActivationTableMetadata::updraft_speeds + updraft_speed;
    return index * ActivationTableMetadata::aerosol_counts + aerosol_count;
}

bool validate_activation_table(const ActivationTable& table, std::string& error)
{
    if (table.fractions.size() != ActivationTableMetadata::element_count)
    {
        error = "Thompson activation-table dimensions do not match WRF mp=28";
        return false;
    }
    for (float value : table.fractions)
    {
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f)
        {
            error = "Thompson activation table contains an invalid fraction";
            return false;
        }
    }
    const std::string computed = payload_hash(table);
    if (computed != ActivationTableMetadata::payload_sha256
        || (!table.payload_sha256.empty() && table.payload_sha256 != computed))
    {
        error = "Thompson activation-table payload hash mismatch";
        return false;
    }
    error.clear();
    return true;
}

bool import_wrf_activation_table(const std::filesystem::path& source,
                                 ActivationTable& table,
                                 std::string& error)
{
    const auto bytes = read_file(source, error);
    constexpr std::size_t payload_bytes = ActivationTableMetadata::element_count * sizeof(float);
    if (bytes.size() != payload_bytes + 8)
    {
        error = "WRF Thompson activation-table byte count mismatch";
        return false;
    }
    const std::uint32_t leading = big_endian_u32(bytes.data());
    const std::uint32_t trailing = big_endian_u32(bytes.data() + bytes.size() - 4);
    if (leading != payload_bytes || trailing != payload_bytes)
    {
        error = "WRF Thompson activation-table record markers are invalid";
        return false;
    }
    if (bytes_hash(bytes) != ActivationTableMetadata::upstream_sha256)
    {
        error = "WRF Thompson activation-table source hash mismatch";
        return false;
    }

    ActivationTable candidate;
    candidate.fractions.resize(ActivationTableMetadata::element_count);
    for (std::size_t index = 0; index < candidate.fractions.size(); ++index)
        candidate.fractions[index] = float_from_bits(big_endian_u32(bytes.data() + 4 + index * 4));
    candidate.payload_sha256 = payload_hash(candidate);
    if (!validate_activation_table(candidate, error)) return false;
    table = std::move(candidate);
    error.clear();
    return true;
}

bool write_activation_table(const std::filesystem::path& path,
                            const ActivationTable& table,
                            std::string& error)
{
    if (!validate_activation_table(table, error)) return false;
    const std::string hash = payload_hash(table);
    const auto temporary = path.string() + ".tmp";
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out)
    {
        error = "unable to open Thompson activation-table temporary file";
        return false;
    }
    auto write = [&](const void* bytes, std::size_t count) {
        out.write(static_cast<const char*>(bytes), static_cast<std::streamsize>(count));
    };
    write(magic, sizeof(magic));
    consume_u32(ActivationTableMetadata::format_version, write);
    for (std::uint32_t dimension : {
             static_cast<std::uint32_t>(ActivationTableMetadata::aerosol_counts),
             static_cast<std::uint32_t>(ActivationTableMetadata::updraft_speeds),
             static_cast<std::uint32_t>(ActivationTableMetadata::temperatures),
             static_cast<std::uint32_t>(ActivationTableMetadata::mean_radii),
             static_cast<std::uint32_t>(ActivationTableMetadata::hygroscopicities)})
        consume_u32(dimension, write);
    const auto source = digest_bytes(ActivationTableMetadata::upstream_sha256);
    const auto payload = digest_bytes(hash.c_str());
    write(source.data(), source.size());
    write(payload.data(), payload.size());
    for (float value : table.fractions) consume_float(value, write);
    out.flush();
    if (!out)
    {
        error = "failed writing Thompson activation-table payload";
        return false;
    }
    out.close();
    std::error_code ec;
    std::filesystem::rename(temporary, path, ec);
    if (ec)
    {
        error = "failed atomically publishing Thompson activation table: " + ec.message();
        return false;
    }
    error.clear();
    return true;
}

bool load_activation_table(const std::filesystem::path& path,
                           ActivationTable& table,
                           std::string& error,
                           std::size_t maximum_resident_bytes)
{
    const std::size_t required_bytes =
        ActivationTableMetadata::element_count * sizeof(float);
    if (required_bytes > maximum_resident_bytes)
    {
        error = "Thompson activation table exceeds the configured resident-memory budget";
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        error = "unable to open Thompson activation table";
        return false;
    }
    std::array<char, sizeof(magic)> actual_magic{};
    if (!read_exact(in, actual_magic.data(), actual_magic.size())
        || !std::equal(actual_magic.begin(), actual_magic.end(), std::begin(magic)))
    {
        error = "invalid Thompson activation-table magic";
        return false;
    }
    std::uint32_t version = 0;
    if (!read_u32(in, version) || version != ActivationTableMetadata::format_version)
    {
        error = "unsupported Thompson activation-table format";
        return false;
    }
    for (std::uint32_t expected : {
             static_cast<std::uint32_t>(ActivationTableMetadata::aerosol_counts),
             static_cast<std::uint32_t>(ActivationTableMetadata::updraft_speeds),
             static_cast<std::uint32_t>(ActivationTableMetadata::temperatures),
             static_cast<std::uint32_t>(ActivationTableMetadata::mean_radii),
             static_cast<std::uint32_t>(ActivationTableMetadata::hygroscopicities)})
    {
        std::uint32_t actual = 0;
        if (!read_u32(in, actual) || actual != expected)
        {
            error = "Thompson activation-table metadata mismatch";
            return false;
        }
    }
    std::array<std::uint8_t, digest_size> source{};
    std::array<std::uint8_t, digest_size> expected_payload{};
    if (!read_exact(in, source.data(), source.size())
        || !read_exact(in, expected_payload.data(), expected_payload.size())
        || source != digest_bytes(ActivationTableMetadata::upstream_sha256))
    {
        error = "Thompson activation-table upstream identity mismatch";
        return false;
    }
    ActivationTable candidate;
    candidate.fractions.resize(ActivationTableMetadata::element_count);
    for (float& value : candidate.fractions)
    {
        std::uint32_t bits = 0;
        if (!read_u32(in, bits))
        {
            error = "truncated Thompson activation-table payload";
            return false;
        }
        value = float_from_bits(bits);
    }
    char trailing = 0;
    if (in.read(&trailing, 1))
    {
        error = "unexpected trailing bytes in Thompson activation table";
        return false;
    }
    candidate.payload_sha256 = hex_digest(expected_payload);
    if (!validate_activation_table(candidate, error)) return false;
    table = std::move(candidate);
    error.clear();
    return true;
}

double activated_cloud_number_m3(const ActivationTable& table,
                                 double temperature_k,
                                 double updraft_m_s,
                                 double available_aerosol_m3)
{
    if (table.fractions.size() != ActivationTableMetadata::element_count
        || !std::isfinite(temperature_k) || !std::isfinite(updraft_m_s)
        || !std::isfinite(available_aerosol_m3) || available_aerosol_m3 < 0.0)
    {
        throw std::invalid_argument("invalid Thompson activation input");
    }
    double aerosol_cm3 = available_aerosol_m3 * 1.0e-6;
    if (aerosol_cm3 >= aerosol_axis.back()) aerosol_cm3 = aerosol_axis.back() - 1.0;
    else if (aerosol_cm3 <= aerosol_axis.front()) aerosol_cm3 = aerosol_axis.front() + 1.0;
    double updraft = updraft_m_s;
    if (updraft >= updraft_axis.back()) updraft = updraft_axis.back() - 1.0;
    else if (updraft <= updraft_axis.front()) updraft = updraft_axis.front() + 0.001;

    const auto aerosol_upper = static_cast<std::size_t>(
        std::upper_bound(aerosol_axis.begin(), aerosol_axis.end(), aerosol_cm3)
        - aerosol_axis.begin());
    const auto updraft_upper = static_cast<std::size_t>(
        std::upper_bound(updraft_axis.begin(), updraft_axis.end(), updraft)
        - updraft_axis.begin());
    const double temperature_coordinate = std::clamp((temperature_k - temperature_axis.front()) * 0.1,
                                                     0.0,
                                                     static_cast<double>(ActivationTableMetadata::temperatures - 1));
    const long   rounded_temperature    = std::lround(temperature_coordinate);
    const std::size_t temperature = static_cast<std::size_t>(
        std::clamp(rounded_temperature,
                   0L,
                   static_cast<long>(ActivationTableMetadata::temperatures - 1)));
    constexpr std::size_t mean_radius =
        ActivationTableMetadata::selected_mean_radius_index;
    constexpr std::size_t hygroscopicity =
        ActivationTableMetadata::selected_hygroscopicity_index;
    const auto fraction = [&](std::size_t aerosol, std::size_t velocity) {
        return table.fractions[activation_flatten(
            aerosol, velocity, temperature, mean_radius, hygroscopicity)];
    };
    const double x1 = std::log(aerosol_axis[aerosol_upper - 1]);
    const double x2 = std::log(aerosol_axis[aerosol_upper]);
    const double y1 = std::log(updraft_axis[updraft_upper - 1]);
    const double y2 = std::log(updraft_axis[updraft_upper]);
    const double t = (std::log(aerosol_cm3) - x1) / (x2 - x1);
    const double u = (std::log(updraft) - y1) / (y2 - y1);
    const double a = fraction(aerosol_upper - 1, updraft_upper - 1);
    const double b = fraction(aerosol_upper, updraft_upper - 1);
    const double c = fraction(aerosol_upper, updraft_upper);
    const double d = fraction(aerosol_upper - 1, updraft_upper);
    const double activated_fraction = (1.0 - t) * (1.0 - u) * a
                                      + t * (1.0 - u) * b + t * u * c
                                      + (1.0 - t) * u * d;
    return available_aerosol_m3 * activated_fraction;
}

bool ActivationTableStore::publish_wrf_file(const std::filesystem::path& path,
                                             std::string& error)
{
    auto candidate = std::make_shared<ActivationTable>();
    if (!import_wrf_activation_table(path, *candidate, error)) return false;
    std::atomic_store_explicit(&table_,
                               std::static_pointer_cast<const ActivationTable>(candidate),
                               std::memory_order_release);
    error.clear();
    return true;
}

bool ActivationTableStore::publish_file(const std::filesystem::path& path,
                                        std::string& error,
                                        std::size_t maximum_resident_bytes)
{
    auto candidate = std::make_shared<ActivationTable>();
    if (!load_activation_table(path, *candidate, error, maximum_resident_bytes))
        return false;
    std::atomic_store_explicit(&table_,
                               std::static_pointer_cast<const ActivationTable>(candidate),
                               std::memory_order_release);
    error.clear();
    return true;
}

std::shared_ptr<const ActivationTable> ActivationTableStore::snapshot() const noexcept
{
    return std::atomic_load_explicit(&table_, std::memory_order_acquire);
}

} // namespace tmv::thompson
