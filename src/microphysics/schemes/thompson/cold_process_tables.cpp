#include "microphysics/schemes/thompson/cold_process_tables.hpp"
#include "microphysics/schemes/thompson/freezing_factors.hpp"

#include "core/infra/sha256.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>

namespace tmv::thompson
{
namespace
{

constexpr double pi = 3.1415926536;
constexpr double rho_water = 1000.0;
constexpr double rho_ice = 890.0;
constexpr double water_mass_coefficient = pi * rho_water / 6.0;
constexpr double ice_mass_coefficient = pi * rho_ice / 6.0;
constexpr double minimum_cloud_diameter = 1.0e-6;
constexpr double minimum_rain_diameter = 50.0e-6;
constexpr double minimum_snow_diameter = 300.0e-6;
constexpr double minimum_ice_diameter =
    1.2898438095973233e-5;
constexpr char magic[] = "NTCOLD1";
constexpr std::size_t digest_bytes = 32;
constexpr char constants_identity[] =
    "WRF-v4.7.1-mp28-cold|water=37|rain_n0=37|cloud_n=100|temperature=45|"
    "IN=55|ice=64|ice_n=55|bins=100|rho_w=1000|rho_i=890|"
    "am_r=pi*rho_w/6|bm_r=3|mu_r=0|am_i=pi*rho_i/6|bm_i=3|mu_i=0|"
    "xm0i=1e-12|D0c=1e-6|D0r=50e-6|D0s=300e-6|rain_max=0.005|"
    "cloud_max=100e-6|ice_max=600e-6|Bigg=120*V*5.2e-4|NtIN=1..1e6";

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs)
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
        throw std::overflow_error("Thompson cold-table dimension overflow");
    return lhs * rhs;
}

template<std::size_t N>
std::array<double, N> decade_grid(int first_exponent)
{
    static_assert((N - 1) % 9 == 0);
    std::array<double, N> values{};
    std::size_t cursor = 0;
    const int decades = static_cast<int>((N - 1) / 9);
    for (int decade = 0; decade < decades; ++decade)
    {
        const double scale = std::pow(10.0, first_exponent + decade);
        for (int digit = 1; digit <= 9; ++digit)
            values[cursor++] = digit * scale;
    }
    values[cursor] = std::pow(10.0, first_exponent + decades);
    return values;
}

template<std::size_t N>
std::array<double, N> logarithmic_centers(double minimum, double maximum)
{
    std::array<double, N + 1> edges{};
    std::array<double, N> values{};
    edges.front() = minimum;
    edges.back() = maximum;
    for (std::size_t index = 1; index < N; ++index)
        edges[index] = std::exp(
            static_cast<double>(index) / static_cast<double>(N)
                * std::log(maximum / minimum)
            + std::log(minimum));
    for (std::size_t index = 0; index < N; ++index)
        values[index] = std::sqrt(edges[index] * edges[index + 1]);
    return values;
}

template<std::size_t N>
std::array<double, N> logarithmic_widths(double minimum, double maximum)
{
    std::array<double, N + 1> edges{};
    std::array<double, N> values{};
    edges.front() = minimum;
    edges.back() = maximum;
    for (std::size_t index = 1; index < N; ++index)
        edges[index] = std::exp(
            static_cast<double>(index) / static_cast<double>(N)
                * std::log(maximum / minimum)
            + std::log(minimum));
    for (std::size_t index = 0; index < N; ++index)
        values[index] = edges[index + 1] - edges[index];
    return values;
}

struct Context
{
    std::array<double, ColdProcessTableMetadata::water_contents> water_content =
        decade_grid<ColdProcessTableMetadata::water_contents>(-6);
    std::array<double, ColdProcessTableMetadata::rain_intercepts> rain_intercept =
        decade_grid<ColdProcessTableMetadata::rain_intercepts>(6);
    std::array<double, ColdProcessTableMetadata::ice_contents> ice_content =
        decade_grid<ColdProcessTableMetadata::ice_contents>(-10);
    std::array<double, ColdProcessTableMetadata::ice_numbers> ice_number =
        decade_grid<ColdProcessTableMetadata::ice_numbers>(0);
    std::array<double, ColdProcessTableMetadata::ice_nuclei> ice_nuclei =
        decade_grid<ColdProcessTableMetadata::ice_nuclei>(0);
    std::array<double, ColdProcessTableMetadata::particle_bins> rain_diameter =
        logarithmic_centers<ColdProcessTableMetadata::particle_bins>(
            minimum_rain_diameter, 0.005);
    std::array<double, ColdProcessTableMetadata::particle_bins> rain_width =
        logarithmic_widths<ColdProcessTableMetadata::particle_bins>(
            minimum_rain_diameter, 0.005);
    std::array<double, ColdProcessTableMetadata::particle_bins> ice_diameter =
        logarithmic_centers<ColdProcessTableMetadata::particle_bins>(
            minimum_ice_diameter, 2.0 * minimum_snow_diameter);
    std::array<double, ColdProcessTableMetadata::particle_bins> ice_width =
        logarithmic_widths<ColdProcessTableMetadata::particle_bins>(
            minimum_ice_diameter, 2.0 * minimum_snow_diameter);
    std::array<double, ColdProcessTableMetadata::cloud_numbers> cloud_number{};

    Context()
    {
        const auto centers = logarithmic_centers<
            ColdProcessTableMetadata::cloud_numbers>(1.0, 3000.0);
        for (std::size_t index = 0; index < cloud_number.size(); ++index)
            cloud_number[index] = centers[index] * 1.0e6;
    }
};

const Context& context()
{
    static const Context value;
    return value;
}

void require_index(bool valid)
{
    if (!valid)
        throw std::out_of_range("Thompson cold-table index out of range");
}

std::string constants_hash()
{
    return nimbus::crypto::sha256_hex(constants_identity);
}

template<typename Consumer>
void consume_u32(std::uint32_t value, Consumer&& consume)
{
    std::array<std::uint8_t, 4> bytes{};
    for (std::size_t index = 0; index < bytes.size(); ++index)
        bytes[index] = static_cast<std::uint8_t>(value >> (8U * index));
    consume(bytes.data(), bytes.size());
}

template<typename Consumer>
void consume_u64(std::uint64_t value, Consumer&& consume)
{
    std::array<std::uint8_t, 8> bytes{};
    for (std::size_t index = 0; index < bytes.size(); ++index)
        bytes[index] = static_cast<std::uint8_t>(value >> (8U * index));
    consume(bytes.data(), bytes.size());
}

template<typename Consumer>
void consume_double(double value, Consumer&& consume)
{
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    consume_u64(bits, std::forward<Consumer>(consume));
}

template<typename Consumer>
void consume_payload(const ColdProcessTables& tables, Consumer&& consume)
{
    for (const auto& cell : tables.rain_freezing)
    {
        consume_double(cell.ice_mass, consume);
        consume_double(cell.graupel_mass, consume);
        consume_double(cell.ice_number, consume);
        consume_double(cell.rain_number, consume);
    }
    for (const auto& cell : tables.cloud_freezing)
    {
        consume_double(cell.ice_mass, consume);
        consume_double(cell.ice_number, consume);
    }
    for (const auto& cell : tables.ice_snow)
    {
        consume_double(cell.snow_mass, consume);
        consume_double(cell.ice_number, consume);
        consume_double(cell.ice_deposition_fraction, consume);
    }
}

std::string payload_hash(const ColdProcessTables& tables)
{
    nimbus::crypto::Sha256 hash;
    consume_payload(tables, [&](const void* data, std::size_t size) {
        hash.update(data, size);
    });
    return hash.hex_digest();
}

std::array<std::uint8_t, digest_bytes> digest_from_hex(const std::string& hex)
{
    std::array<std::uint8_t, digest_bytes> result{};
    if (hex.size() != result.size() * 2)
        throw std::invalid_argument("invalid Thompson cold-table digest");
    const auto nibble = [](char value) {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        if (value >= 'A' && value <= 'F') return value - 'A' + 10;
        return -1;
    };
    for (std::size_t index = 0; index < result.size(); ++index)
    {
        const int high = nibble(hex[index * 2]);
        const int low = nibble(hex[index * 2 + 1]);
        if (high < 0 || low < 0)
            throw std::invalid_argument("invalid Thompson cold-table digest");
        result[index] = static_cast<std::uint8_t>((high << 4) | low);
    }
    return result;
}

std::string hex_digest(const std::array<std::uint8_t, digest_bytes>& digest)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string result(digest.size() * 2, '0');
    for (std::size_t index = 0; index < digest.size(); ++index)
    {
        result[index * 2] = digits[digest[index] >> 4];
        result[index * 2 + 1] = digits[digest[index] & 15];
    }
    return result;
}

bool read_exact(std::istream& input, void* destination, std::size_t size)
{
    input.read(static_cast<char*>(destination), static_cast<std::streamsize>(size));
    return input.good();
}

bool read_u32(std::istream& input, std::uint32_t& value)
{
    std::array<std::uint8_t, 4> bytes{};
    if (!read_exact(input, bytes.data(), bytes.size())) return false;
    value = 0;
    for (std::size_t index = 0; index < bytes.size(); ++index)
        value |= static_cast<std::uint32_t>(bytes[index]) << (8U * index);
    return true;
}

bool read_u64(std::istream& input, std::uint64_t& value)
{
    std::array<std::uint8_t, 8> bytes{};
    if (!read_exact(input, bytes.data(), bytes.size())) return false;
    value = 0;
    for (std::size_t index = 0; index < bytes.size(); ++index)
        value |= static_cast<std::uint64_t>(bytes[index]) << (8U * index);
    return true;
}

bool read_double(std::istream& input, double& value)
{
    std::uint64_t bits = 0;
    if (!read_u64(input, bits)) return false;
    std::memcpy(&value, &bits, sizeof(value));
    return true;
}

bool finite_nonnegative(double value)
{
    return std::isfinite(value) && value >= 0.0;
}

} // namespace

std::size_t checked_rain_freezing_size()
{
    std::size_t size = ColdProcessTableMetadata::water_contents;
    size = checked_multiply(size, ColdProcessTableMetadata::rain_intercepts);
    size = checked_multiply(size, ColdProcessTableMetadata::temperatures);
    return checked_multiply(size, ColdProcessTableMetadata::ice_nuclei);
}

std::size_t checked_cloud_freezing_size()
{
    std::size_t size = ColdProcessTableMetadata::water_contents;
    size = checked_multiply(size, ColdProcessTableMetadata::cloud_numbers);
    size = checked_multiply(size, ColdProcessTableMetadata::temperatures);
    return checked_multiply(size, ColdProcessTableMetadata::ice_nuclei);
}

std::size_t checked_ice_snow_size()
{
    return checked_multiply(ColdProcessTableMetadata::ice_contents,
                            ColdProcessTableMetadata::ice_numbers);
}

std::size_t checked_cold_payload_bytes()
{
    std::size_t values = checked_multiply(checked_rain_freezing_size(), 4);
    values += checked_multiply(checked_cloud_freezing_size(), 2);
    values += checked_multiply(checked_ice_snow_size(), 3);
    return checked_multiply(values, sizeof(double));
}

std::size_t cold_resident_bytes(const ColdProcessTables& tables) noexcept
{
    return tables.rain_freezing.capacity() * sizeof(RainFreezingCell)
        + tables.cloud_freezing.capacity() * sizeof(CloudFreezingCell)
        + tables.ice_snow.capacity() * sizeof(IceSnowCell);
}

std::size_t flatten(RainFreezingIndex index)
{
    require_index(index.rain_content < ColdProcessTableMetadata::water_contents
                  && index.rain_intercept
                      < ColdProcessTableMetadata::rain_intercepts
                  && index.temperature < ColdProcessTableMetadata::temperatures
                  && index.ice_nuclei < ColdProcessTableMetadata::ice_nuclei);
    return (((index.ice_nuclei * ColdProcessTableMetadata::temperatures
              + index.temperature)
             * ColdProcessTableMetadata::rain_intercepts
             + index.rain_intercept)
            * ColdProcessTableMetadata::water_contents)
        + index.rain_content;
}

std::size_t flatten(CloudFreezingIndex index)
{
    require_index(index.cloud_content < ColdProcessTableMetadata::water_contents
                  && index.cloud_number < ColdProcessTableMetadata::cloud_numbers
                  && index.temperature < ColdProcessTableMetadata::temperatures
                  && index.ice_nuclei < ColdProcessTableMetadata::ice_nuclei);
    return (((index.ice_nuclei * ColdProcessTableMetadata::temperatures
              + index.temperature)
             * ColdProcessTableMetadata::cloud_numbers
             + index.cloud_number)
            * ColdProcessTableMetadata::water_contents)
        + index.cloud_content;
}

std::size_t flatten(IceSnowIndex index)
{
    require_index(index.ice_content < ColdProcessTableMetadata::ice_contents
                  && index.ice_number < ColdProcessTableMetadata::ice_numbers);
    return index.ice_number * ColdProcessTableMetadata::ice_contents
        + index.ice_content;
}

FreezingFactors generate_freezing_factors()
{
    const auto&     data = context();
    FreezingFactors factors;
    factors.water_content  = data.water_content;
    factors.rain_intercept = data.rain_intercept;
    factors.ice_nuclei     = data.ice_nuclei;
    factors.cloud_number   = data.cloud_number;
    factors.rain_bin_number.resize(37 * 37 * 100);
    factors.cloud_bin_number.resize(100 * 37 * 100);
    factors.rain_probability.resize(55 * 45 * 100);
    factors.cloud_probability.resize(55 * 45 * 100);
    for (std::size_t bin = 0; bin < 100; ++bin)
    {
        const double rain_diameter  = data.rain_diameter[bin];
        const double cloud_diameter = minimum_cloud_diameter * (bin + 1);
        factors.rain_mass[bin]      = water_mass_coefficient * rain_diameter * rain_diameter * rain_diameter;
        factors.cloud_mass[bin]     = water_mass_coefficient * cloud_diameter * cloud_diameter * cloud_diameter;
    }
    for (std::size_t intercept_index = 0; intercept_index < 37; ++intercept_index)
        for (std::size_t content_index = 0; content_index < 37; ++content_index)
        {
            const double content = data.water_content[content_index];
            const double exponential =
                std::pow(data.rain_intercept[intercept_index] * water_mass_coefficient * 6.0 / content, 0.25);
            const double lambda    = exponential * std::cbrt(6.0);
            const double intercept = data.rain_intercept[intercept_index] / exponential * lambda;
            for (std::size_t bin = 0; bin < 100; ++bin)
            {
                factors.rain_bin_number[(intercept_index * 37 + content_index) * 100 + bin] =
                    intercept * std::exp(-lambda * data.rain_diameter[bin]) * data.rain_width[bin];
            }
        }
    for (std::size_t number_index = 0; number_index < 100; ++number_index)
        for (std::size_t content_index = 0; content_index < 37; ++content_index)
        {
            const double number    = data.cloud_number[number_index];
            const int    shape     = std::min(15, static_cast<int>(std::floor(1000e6 / number + .5)) + 2);
            const double lambda    = std::cbrt(number * water_mass_coefficient * std::tgamma(shape + 4.0) /
                                               std::tgamma(shape + 1.0) / data.water_content[content_index]);
            const double intercept = number / std::tgamma(shape + 1.0) * std::pow(lambda, shape + 1.0);
            for (std::size_t bin = 0; bin < 100; ++bin)
            {
                const double diameter = minimum_cloud_diameter * (bin + 1);
                factors.cloud_bin_number[(number_index * 37 + content_index) * 100 + bin] =
                    intercept * std::pow(diameter, shape) * std::exp(-lambda * diameter) * minimum_cloud_diameter;
            }
        }
    for (std::size_t nuclei = 0; nuclei < 55; ++nuclei)
        for (std::size_t temperature = 0; temperature < 45; ++temperature)
        {
            const double adjustment = std::clamp(3.0 - std::log10(data.ice_nuclei[nuclei]), -3.0, 3.0);
            const double thermal    = std::exp(static_cast<double>(temperature + 1) - adjustment) - 1.0;
            for (std::size_t bin = 0; bin < 100; ++bin)
            {
                const auto index = (nuclei * 45 + temperature) * 100 + bin;
                factors.rain_probability[index] =
                    std::max(0.0, 1.0 - std::exp(-120.0 * factors.rain_mass[bin] / rho_water * 5.2e-4 * thermal));
                factors.cloud_probability[index] =
                    std::max(0.0, 1.0 - std::exp(-120.0 * factors.cloud_mass[bin] / rho_water * 5.2e-4 * thermal));
            }
        }
    return factors;
}

namespace
{
const FreezingFactors& cached_freezing_factors()
{
    static const auto factors = generate_freezing_factors();
    return factors;
}
} // namespace

RainFreezingCell generate_rain_freezing_cell(RainFreezingIndex index)
{
    static_cast<void>(flatten(index));
    const auto&      data         = cached_freezing_factors();
    const double     rain_content = data.water_content[index.rain_content];
    RainFreezingCell result;
    for (std::size_t bin = 100; bin-- > 0;)
    {
        const double mass        = data.rain_mass[bin];
        const double number      = data.rain_bin_number[(index.rain_intercept * 37 + index.rain_content) * 100 + bin];
        const double probability = data.rain_probability[(index.ice_nuclei * 45 + index.temperature) * 100 + bin];
        if (mass < ice_mass_coefficient * minimum_snow_diameter * minimum_snow_diameter * minimum_snow_diameter)
        {
            result.ice_mass += probability * number * mass;
            result.ice_number += probability * number;
        }
        else
        {
            result.graupel_mass += probability * number * mass;
            result.rain_number += probability * number;
        }
        if (result.ice_mass + result.graupel_mass >= rain_content)
            break;
    }
    return result;
}

double freezing_water_content(std::size_t index)
{
    return context().water_content.at(index);
}

double freezing_cloud_number(std::size_t index)
{
    return context().cloud_number.at(index);
}

double ice_table_content(std::size_t index)
{
    return context().ice_content.at(index);
}

double ice_table_number(std::size_t index)
{
    return context().ice_number.at(index);
}

CloudFreezingCell generate_cloud_freezing_cell(CloudFreezingIndex index)
{
    static_cast<void>(flatten(index));
    return generate_cloud_freezing_cell(index, cached_freezing_factors().water_content[index.cloud_content]);
}

CloudFreezingCell generate_cloud_freezing_cell(CloudFreezingIndex index, double cloud_content)
{
    static_cast<void>(flatten(index));
    const auto&  data              = cached_freezing_factors();
    const double tabulated_content = data.water_content[index.cloud_content];
    if (!std::isfinite(cloud_content) || cloud_content <= 0 || cloud_content > tabulated_content)
        throw std::invalid_argument("invalid cloud-freezing continuation content");
    const double cloud_number      = data.cloud_number[index.cloud_number];
    const bool   continuation      = cloud_content < tabulated_content;
    double       lambda_difference = 0, log_scale = 0;
    if (continuation)
    {
        const int    shape  = std::min(15, static_cast<int>(std::floor(1000e6 / cloud_number + .5)) + 2);
        const double factor = cloud_number * water_mass_coefficient * double((shape + 1) * (shape + 2) * (shape + 3));
        const double reference_lambda = std::cbrt(factor / tabulated_content);
        const double lambda           = std::cbrt(factor / cloud_content);
        lambda_difference             = lambda - reference_lambda;
        log_scale                     = (shape + 1) * std::log(lambda / reference_lambda);
    }
    CloudFreezingCell result;
    for (std::size_t bin = 0; bin < 100; ++bin)
    {
        const double mass   = data.cloud_mass[bin];
        double       number = data.cloud_bin_number[(index.cloud_number * 37 + index.cloud_content) * 100 + bin];
        if (continuation && number > 0)
            number = std::exp(std::log(number) + log_scale - lambda_difference * minimum_cloud_diameter * (bin + 1));
        const double probability = data.cloud_probability[(index.ice_nuclei * 45 + index.temperature) * 100 + bin];
        result.ice_number        = std::min(cloud_number, result.ice_number + probability * number);
        result.ice_mass += probability * number * mass;
        if (result.ice_mass >= cloud_content)
            break;
    }
    return result;
}

IceSnowCell generate_ice_snow_cell(IceSnowIndex index)
{
    static_cast<void>(flatten(index));
    const auto&  data          = context();
    const double content       = data.ice_content[index.ice_content];
    const double number        = data.ice_number[index.ice_number];
    const double lambda        = std::cbrt(ice_mass_coefficient * 6.0 * number / content);
    const double mean_diameter = 4.0 / lambda;
    IceSnowCell  result;
    if (mean_diameter > 5.0 * minimum_snow_diameter)
    {
        result.snow_mass  = content;
        result.ice_number = number;
        return result;
    }
    if (mean_diameter < minimum_ice_diameter)
    {
        result.ice_deposition_fraction = 1.0;
        return result;
    }
    const double limit             = lambda * minimum_snow_diameter;
    result.ice_deposition_fraction = 1.0 - std::exp(-limit) * (1.0 + limit);
    const double intercept         = number * lambda;
    for (std::size_t bin = 0; bin < data.ice_diameter.size(); ++bin)
    {
        const double diameter = data.ice_diameter[bin];
        if (diameter < minimum_snow_diameter)
            continue;
        const double bin_number = intercept * std::exp(-lambda * diameter)
            * data.ice_width[bin];
        result.snow_mass += bin_number * ice_mass_coefficient
            * diameter * diameter * diameter;
        result.ice_number += bin_number;
    }
    return result;
}

ColdProcessTables generate_cold_process_tables()
{
    ColdProcessTables tables;
    tables.rain_freezing.resize(checked_rain_freezing_size());
    tables.cloud_freezing.resize(checked_cloud_freezing_size());
    tables.ice_snow.resize(checked_ice_snow_size());
    #pragma omp parallel for schedule(static)
    for (std::size_t index = 0; index < tables.rain_freezing.size(); ++index)
    {
        std::size_t value = index;
        RainFreezingIndex decoded;
        decoded.rain_content = value % ColdProcessTableMetadata::water_contents;
        value /= ColdProcessTableMetadata::water_contents;
        decoded.rain_intercept = value % ColdProcessTableMetadata::rain_intercepts;
        value /= ColdProcessTableMetadata::rain_intercepts;
        decoded.temperature = value % ColdProcessTableMetadata::temperatures;
        decoded.ice_nuclei = value / ColdProcessTableMetadata::temperatures;
        tables.rain_freezing[index] = generate_rain_freezing_cell(decoded);
    }
    #pragma omp parallel for schedule(static)
    for (std::size_t index = 0; index < tables.cloud_freezing.size(); ++index)
    {
        std::size_t value = index;
        CloudFreezingIndex decoded;
        decoded.cloud_content = value % ColdProcessTableMetadata::water_contents;
        value /= ColdProcessTableMetadata::water_contents;
        decoded.cloud_number = value % ColdProcessTableMetadata::cloud_numbers;
        value /= ColdProcessTableMetadata::cloud_numbers;
        decoded.temperature = value % ColdProcessTableMetadata::temperatures;
        decoded.ice_nuclei = value / ColdProcessTableMetadata::temperatures;
        tables.cloud_freezing[index] = generate_cloud_freezing_cell(decoded);
    }
    #pragma omp parallel for schedule(static)
    for (std::size_t index = 0; index < tables.ice_snow.size(); ++index)
    {
        tables.ice_snow[index] = generate_ice_snow_cell({
            index % ColdProcessTableMetadata::ice_contents,
            index / ColdProcessTableMetadata::ice_contents});
    }
    tables.payload_sha256 = payload_hash(tables);
    return tables;
}

bool validate_cold_process_tables(const ColdProcessTables& tables,
                                  std::string& error)
{
    if (tables.rain_freezing.size() != checked_rain_freezing_size()
        || tables.cloud_freezing.size() != checked_cloud_freezing_size()
        || tables.ice_snow.size() != checked_ice_snow_size())
    {
        error = "Thompson cold-table dimension mismatch";
        return false;
    }
    for (const auto& cell : tables.rain_freezing)
        if (!finite_nonnegative(cell.ice_mass)
            || !finite_nonnegative(cell.graupel_mass)
            || !finite_nonnegative(cell.ice_number)
            || !finite_nonnegative(cell.rain_number))
        {
            error = "invalid Thompson rain-freezing table value";
            return false;
        }
    for (const auto& cell : tables.cloud_freezing)
        if (!finite_nonnegative(cell.ice_mass)
            || !finite_nonnegative(cell.ice_number))
        {
            error = "invalid Thompson cloud-freezing table value";
            return false;
        }
    for (const auto& cell : tables.ice_snow)
        if (!finite_nonnegative(cell.snow_mass)
            || !finite_nonnegative(cell.ice_number)
            || !finite_nonnegative(cell.ice_deposition_fraction)
            || cell.ice_deposition_fraction > 1.0)
        {
            error = "invalid Thompson ice-snow table value";
            return false;
        }
    const std::string computed = payload_hash(tables);
    if (!tables.payload_sha256.empty() && tables.payload_sha256 != computed)
    {
        error = "Thompson cold-table payload hash mismatch";
        return false;
    }
    error.clear();
    return true;
}

bool write_cold_process_tables(const std::filesystem::path& path,
                               const ColdProcessTables& tables,
                               std::string& error)
{
    if (!validate_cold_process_tables(tables, error)) return false;
    const std::filesystem::path temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output)
    {
        error = "unable to create Thompson cold tables";
        return false;
    }
    const std::string hash = payload_hash(tables);
    const auto source_digest = digest_from_hex(
        ColdProcessTableMetadata::upstream_source_sha256);
    const auto constant_digest = digest_from_hex(constants_hash());
    const auto payload_digest = digest_from_hex(hash);
    output.write(magic, sizeof(magic));
    auto write = [&](const void* data, std::size_t size) {
        output.write(static_cast<const char*>(data),
                     static_cast<std::streamsize>(size));
    };
    consume_u32(ColdProcessTableMetadata::format_version, write);
    for (std::uint32_t dimension : {
             37U, 37U, 100U, 45U, 55U, 64U, 55U})
        consume_u32(dimension, write);
    write(source_digest.data(), source_digest.size());
    write(constant_digest.data(), constant_digest.size());
    write(payload_digest.data(), payload_digest.size());
    consume_payload(tables, write);
    output.close();
    if (!output)
    {
        error = "failed writing Thompson cold tables";
        std::filesystem::remove(temporary);
        return false;
    }
    std::error_code ec;
    std::filesystem::rename(temporary, path, ec);
    if (ec)
    {
        std::filesystem::remove(path, ec);
        ec.clear();
        std::filesystem::rename(temporary, path, ec);
    }
    if (ec)
    {
        error = "failed publishing Thompson cold tables";
        std::filesystem::remove(temporary);
        return false;
    }
    error.clear();
    return true;
}

bool load_cold_process_tables(const std::filesystem::path& path,
                              ColdProcessTables& tables,
                              std::string& error,
                              std::size_t maximum_resident_bytes)
{
    if (checked_cold_payload_bytes() > maximum_resident_bytes)
    {
        error = "Thompson cold tables exceed resident-memory budget";
        return false;
    }
    std::ifstream input(path, std::ios::binary);
    if (!input)
    {
        error = "unable to open Thompson cold tables";
        return false;
    }
    std::array<char, sizeof(magic)> file_magic{};
    std::uint32_t version = 0;
    if (!read_exact(input, file_magic.data(), file_magic.size())
        || std::memcmp(file_magic.data(), magic, sizeof(magic)) != 0
        || !read_u32(input, version)
        || version != ColdProcessTableMetadata::format_version)
    {
        error = "Thompson cold-table header mismatch";
        return false;
    }
    const std::array<std::uint32_t, 7> expected = {
        37U, 37U, 100U, 45U, 55U, 64U, 55U};
    for (std::uint32_t value : expected)
    {
        std::uint32_t actual = 0;
        if (!read_u32(input, actual) || actual != value)
        {
            error = "Thompson cold-table dimension mismatch";
            return false;
        }
    }
    std::array<std::uint8_t, digest_bytes> source{};
    std::array<std::uint8_t, digest_bytes> constants{};
    std::array<std::uint8_t, digest_bytes> payload{};
    if (!read_exact(input, source.data(), source.size())
        || !read_exact(input, constants.data(), constants.size())
        || !read_exact(input, payload.data(), payload.size())
        || hex_digest(source) != ColdProcessTableMetadata::upstream_source_sha256
        || hex_digest(constants) != constants_hash())
    {
        error = "Thompson cold-table source identity mismatch";
        return false;
    }
    ColdProcessTables candidate;
    try
    {
        candidate.rain_freezing.resize(checked_rain_freezing_size());
        candidate.cloud_freezing.resize(checked_cloud_freezing_size());
        candidate.ice_snow.resize(checked_ice_snow_size());
    }
    catch (...)
    {
        error = "unable to allocate Thompson cold tables";
        return false;
    }
    auto read_values = [&](auto& cells, auto... members) {
        for (auto& cell : cells)
            if (!(read_double(input, cell.*members) && ...)) return false;
        return true;
    };
    if (!read_values(candidate.rain_freezing,
                     &RainFreezingCell::ice_mass,
                     &RainFreezingCell::graupel_mass,
                     &RainFreezingCell::ice_number,
                     &RainFreezingCell::rain_number)
        || !read_values(candidate.cloud_freezing,
                        &CloudFreezingCell::ice_mass,
                        &CloudFreezingCell::ice_number)
        || !read_values(candidate.ice_snow,
                        &IceSnowCell::snow_mass,
                        &IceSnowCell::ice_number,
                        &IceSnowCell::ice_deposition_fraction)
        || input.peek() != std::char_traits<char>::eof())
    {
        error = "truncated or trailing Thompson cold-table payload";
        return false;
    }
    candidate.payload_sha256 = hex_digest(payload);
    if (!validate_cold_process_tables(candidate, error)) return false;
    tables = std::move(candidate);
    return true;
}

bool ColdProcessTableStore::publish_file(const std::filesystem::path& path,
                                         std::string& error,
                                         std::size_t maximum_resident_bytes)
{
    auto candidate = std::make_shared<ColdProcessTables>();
    if (!load_cold_process_tables(
            path, *candidate, error, maximum_resident_bytes))
        return false;
    tables_ = std::move(candidate);
    return true;
}

bool ColdProcessTableStore::publish_packaged_file(
    const std::filesystem::path& path, std::string& error,
    std::size_t maximum_resident_bytes)
{
    auto candidate = std::make_shared<ColdProcessTables>();
    if (!load_cold_process_tables(
            path, *candidate, error, maximum_resident_bytes))
        return false;
    if (candidate->payload_sha256
        != ColdProcessTableMetadata::packaged_payload_sha256)
    {
        error = "packaged Thompson cold-table payload mismatch";
        return false;
    }
    std::ifstream input(path, std::ios::binary);
    nimbus::crypto::Sha256 hash;
    std::array<char, 1 << 16> buffer{};
    while (input)
    {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        hash.update(buffer.data(), static_cast<std::size_t>(input.gcount()));
    }
    if (hash.hex_digest() != ColdProcessTableMetadata::packaged_sha256)
    {
        error = "packaged Thompson cold-table container mismatch";
        return false;
    }
    tables_ = std::move(candidate);
    return true;
}

std::shared_ptr<const ColdProcessTables>
ColdProcessTableStore::snapshot() const noexcept
{
    return tables_;
}

} // namespace tmv::thompson
