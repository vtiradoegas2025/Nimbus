#include "collection_tables.hpp"

#include "core/infra/sha256.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <chrono>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <utility>

namespace tmv::thompson
{
namespace
{

constexpr double pi = 3.1415926536;
constexpr double rho_water = 1000.0;
constexpr double rho_graupel = 400.0;
constexpr double am_rain = pi * rho_water / 6.0;
constexpr double am_snow = 0.069;
constexpr double am_graupel = pi * rho_graupel / 6.0;
constexpr double bm_rain = 3.0;
constexpr double bm_snow = 2.0;
constexpr double bm_graupel = 3.0;
constexpr double mu_rain = 0.0;
constexpr double mu_snow = 0.6357;
constexpr double mu_graupel = 0.0;
constexpr double av_snow = 40.0;
constexpr double bv_snow = 0.55;
constexpr double fv_snow = 100.0;
constexpr double av_graupel = 442.0;
constexpr double bv_graupel = 0.89;
constexpr double rain_snow_efficiency = 0.95;
constexpr double rain_graupel_efficiency = 0.75;
constexpr double kap0 = 490.6;
constexpr double kap1 = 17.46;
constexpr double lam0 = 20.78;
constexpr double lam1 = 3.29;

constexpr std::array<double, 10> snow_a = {
    5.065339, -0.062659, -3.032362, 0.029469, -0.000285,
    0.31255, 0.000204, 0.003199, 0.0, -0.015952};
constexpr std::array<double, 10> snow_b = {
    0.476221, -0.015896, 0.165977, 0.007468, -0.000141,
    0.060366, 0.000079, 0.000594, 0.0, -0.003577};
constexpr std::array<double, 9> temperatures = {
    -0.01, -5.0, -10.0, -15.0, -20.0, -25.0, -30.0, -35.0, -40.0};
constexpr char constants_identity[] =
    "WRF-v4.7.1-mp28|bins=100|r=37|r1=37|s=37|t=9|g=37|g1=37|gd=1|"
    "D0r=5e-5|Dmaxr=5e-3|D0s=3e-4|Dmaxs=2e-2|D0g=3.5e-4|Dmaxg=5e-2|"
    "rho_w=1000|rho_g=400|am_s=0.069|bm_r=3|bm_s=2|bm_g=3|mu_r=0|"
    "mu_s=0.6357|mu_g=0|av_s=40|bv_s=0.55|fv_s=100|av_g=442|"
    "bv_g=0.89|Ef_rs=0.95|Ef_rg=0.75|Kap0=490.6|Kap1=17.46|"
    "Lam0=20.78|Lam1=3.29|sa=5.065339,-0.062659,-3.032362,0.029469,"
    "-0.000285,0.31255,0.000204,0.003199,0,-0.015952|sb=0.476221,"
    "-0.015896,0.165977,0.007468,-0.000141,0.060366,0.000079,0.000594,"
    "0,-0.003577|Tc=-0.01,-5,-10,-15,-20,-25,-30,-35,-40";

constexpr char magic[] = "NTCTBL1";
constexpr std::size_t digest_bytes = 32;

std::size_t checked_multiply(std::size_t lhs, std::size_t rhs)
{
    if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
        throw std::overflow_error("Thompson collection-table dimension overflow");
    return lhs * rhs;
}

template<std::size_t N>
std::array<double, N> logarithmic_bins(double minimum, double maximum)
{
    std::array<double, N> centers{};
    std::array<double, N + 1> edges{};
    edges.front() = minimum;
    edges.back() = maximum;
    for (std::size_t index = 1; index < N; ++index)
    {
        const double fraction = static_cast<double>(index) / static_cast<double>(N);
        edges[index] = std::exp(fraction * std::log(maximum / minimum) + std::log(minimum));
    }
    for (std::size_t index = 0; index < N; ++index)
        centers[index] = std::sqrt(edges[index] * edges[index + 1]);
    return centers;
}

template<std::size_t N>
std::array<double, N> logarithmic_widths(double minimum, double maximum)
{
    std::array<double, N> widths{};
    std::array<double, N + 1> edges{};
    edges.front() = minimum;
    edges.back() = maximum;
    for (std::size_t index = 1; index < N; ++index)
    {
        const double fraction = static_cast<double>(index) / static_cast<double>(N);
        edges[index] = std::exp(fraction * std::log(maximum / minimum) + std::log(minimum));
    }
    for (std::size_t index = 0; index < N; ++index)
        widths[index] = edges[index + 1] - edges[index];
    return widths;
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
            values[cursor++] = static_cast<double>(digit) * scale;
    }
    values[cursor] = std::pow(10.0, first_exponent + decades);
    return values;
}

struct Context
{
    std::array<double, CollectionTableMetadata::bins> rain_diameter =
        logarithmic_bins<CollectionTableMetadata::bins>(50.0e-6, 0.005);
    std::array<double, CollectionTableMetadata::bins> rain_width =
        logarithmic_widths<CollectionTableMetadata::bins>(50.0e-6, 0.005);
    std::array<double, CollectionTableMetadata::bins> snow_diameter =
        logarithmic_bins<CollectionTableMetadata::bins>(300.0e-6, 0.02);
    std::array<double, CollectionTableMetadata::bins> snow_width =
        logarithmic_widths<CollectionTableMetadata::bins>(300.0e-6, 0.02);
    std::array<double, CollectionTableMetadata::bins> graupel_diameter =
        logarithmic_bins<CollectionTableMetadata::bins>(350.0e-6, 0.05);
    std::array<double, CollectionTableMetadata::bins> graupel_width =
        logarithmic_widths<CollectionTableMetadata::bins>(350.0e-6, 0.05);
    std::array<double, CollectionTableMetadata::rain_contents> rain_content = decade_grid<37>(-6);
    std::array<double, CollectionTableMetadata::snow_contents> snow_content = decade_grid<37>(-6);
    std::array<double, CollectionTableMetadata::graupel_contents> graupel_content = decade_grid<37>(-6);
    std::array<double, CollectionTableMetadata::rain_intercepts> rain_intercept = decade_grid<37>(6);
    std::array<double, CollectionTableMetadata::graupel_intercepts> graupel_intercept = decade_grid<37>(2);
    std::array<double, CollectionTableMetadata::bins> rain_velocity{};
    std::array<double, CollectionTableMetadata::bins> snow_velocity{};
    std::array<double, CollectionTableMetadata::bins> graupel_velocity{};

    Context()
    {
        for (std::size_t index = 0; index < CollectionTableMetadata::bins; ++index)
        {
            const double rain = rain_diameter[index];
            rain_velocity[index] = -0.1021 + 4.932e3 * rain - 0.9551e6 * rain * rain
                                   + 0.07934e9 * rain * rain * rain
                                   - 0.002362e12 * rain * rain * rain * rain;
            const double snow = snow_diameter[index];
            snow_velocity[index] = 1.5 * av_snow * std::pow(snow, bv_snow)
                                   * std::exp(-fv_snow * snow);
            graupel_velocity[index] = av_graupel
                                      * std::pow(graupel_diameter[index], bv_graupel);
        }
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
        throw std::out_of_range("Thompson collection-table index out of range");
}

double snow_polynomial(const std::array<double, 10>& coefficients,
                       double temperature, double moment)
{
    const double t2 = temperature * temperature;
    const double m2 = moment * moment;
    return coefficients[0] + coefficients[1] * temperature + coefficients[2] * moment
           + coefficients[3] * temperature * moment + coefficients[4] * t2
           + coefficients[5] * m2 + coefficients[6] * t2 * moment
           + coefficients[7] * temperature * m2
           + coefficients[8] * t2 * temperature + coefficients[9] * m2 * moment;
}

std::array<std::uint8_t, digest_bytes> digest_bytes_from_hex(const char* hex)
{
    std::array<std::uint8_t, digest_bytes> bytes{};
    auto nibble = [](char value) -> int {
        if (value >= '0' && value <= '9') return value - '0';
        if (value >= 'a' && value <= 'f') return value - 'a' + 10;
        if (value >= 'A' && value <= 'F') return value - 'A' + 10;
        return -1;
    };
    if (std::strlen(hex) != digest_bytes * 2)
        throw std::invalid_argument("invalid SHA-256 text length");
    for (std::size_t index = 0; index < bytes.size(); ++index)
    {
        const int high = nibble(hex[index * 2]);
        const int low = nibble(hex[index * 2 + 1]);
        if (high < 0 || low < 0)
            throw std::invalid_argument("invalid SHA-256 text");
        bytes[index] = static_cast<std::uint8_t>((high << 4) | low);
    }
    return bytes;
}

std::string hex_digest(const std::array<std::uint8_t, digest_bytes>& bytes)
{
    constexpr char digits[] = "0123456789abcdef";
    std::string result(bytes.size() * 2, '0');
    for (std::size_t index = 0; index < bytes.size(); ++index)
    {
        result[index * 2] = digits[bytes[index] >> 4];
        result[index * 2 + 1] = digits[bytes[index] & 0x0f];
    }
    return result;
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
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    consume_u64(bits, std::forward<Consumer>(consume));
}

template<typename Consumer>
void consume_payload(const CollectionTables& tables, Consumer&& consume)
{
    for (const auto& cell : tables.rain_graupel)
    {
        consume_double(cell.graupel_mass_by_rain, consume);
        consume_double(cell.rain_mass_by_rain, consume);
        consume_double(cell.rain_mass_by_graupel, consume);
        consume_double(cell.graupel_number_by_rain, consume);
        consume_double(cell.rain_number_by_graupel, consume);
    }
    for (const auto& cell : tables.rain_snow)
        for (double value : cell.values)
            consume_double(value, consume);
}

std::string payload_hash(const CollectionTables& tables)
{
    nimbus::crypto::Sha256 hash;
    consume_payload(tables, [&](const void* bytes, std::size_t count) {
        hash.update(bytes, count);
    });
    return hash.hex_digest();
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
    value = 0;
    for (std::size_t index = 0; index < bytes.size(); ++index)
        value |= static_cast<std::uint32_t>(bytes[index]) << (8U * index);
    return true;
}

bool read_u64(std::istream& in, std::uint64_t& value)
{
    std::array<std::uint8_t, 8> bytes{};
    if (!read_exact(in, bytes.data(), bytes.size())) return false;
    value = 0;
    for (std::size_t index = 0; index < bytes.size(); ++index)
        value |= static_cast<std::uint64_t>(bytes[index]) << (8U * index);
    return true;
}

bool read_double(std::istream& in, double& value)
{
    std::uint64_t bits = 0;
    if (!read_u64(in, bits)) return false;
    std::memcpy(&value, &bits, sizeof(value));
    return true;
}

bool finite_nonnegative(double value)
{
    return std::isfinite(value) && value >= 0.0;
}

} // namespace

std::size_t checked_rain_graupel_size()
{
    std::size_t size = CollectionTableMetadata::graupel_intercepts;
    size = checked_multiply(size, CollectionTableMetadata::graupel_contents);
    size = checked_multiply(size, CollectionTableMetadata::graupel_densities);
    size = checked_multiply(size, CollectionTableMetadata::rain_intercepts);
    return checked_multiply(size, CollectionTableMetadata::rain_contents);
}

std::size_t checked_rain_snow_size()
{
    std::size_t size = CollectionTableMetadata::snow_contents;
    size = checked_multiply(size, CollectionTableMetadata::temperatures);
    size = checked_multiply(size, CollectionTableMetadata::rain_intercepts);
    return checked_multiply(size, CollectionTableMetadata::rain_contents);
}

std::size_t checked_payload_bytes()
{
    const std::size_t graupel = checked_multiply(
        checked_rain_graupel_size(), sizeof(RainGraupelCollection));
    const std::size_t snow = checked_multiply(
        checked_rain_snow_size(), sizeof(RainSnowCollection));
    if (snow > std::numeric_limits<std::size_t>::max() - graupel)
        throw std::overflow_error("Thompson collection-table payload overflow");
    return graupel + snow;
}

std::uint64_t checked_full_pair_evaluations()
{
    const std::uint64_t cells = static_cast<std::uint64_t>(checked_rain_graupel_size())
                                + static_cast<std::uint64_t>(checked_rain_snow_size());
    if (cells > std::numeric_limits<std::uint64_t>::max()
                    / CollectionTableMetadata::pair_evaluations_per_cell)
    {
        throw std::overflow_error("Thompson generator work-count overflow");
    }
    return cells * CollectionTableMetadata::pair_evaluations_per_cell;
}

std::size_t resident_bytes(const CollectionTables& tables) noexcept
{
    return tables.rain_graupel.capacity() * sizeof(RainGraupelCollection)
           + tables.rain_snow.capacity() * sizeof(RainSnowCollection);
}

std::size_t flatten(RainGraupelIndex index)
{
    require_index(index.graupel_intercept < CollectionTableMetadata::graupel_intercepts
                  && index.graupel_content < CollectionTableMetadata::graupel_contents
                  && index.graupel_density < CollectionTableMetadata::graupel_densities
                  && index.rain_intercept < CollectionTableMetadata::rain_intercepts
                  && index.rain_content < CollectionTableMetadata::rain_contents);
    std::size_t flat = index.graupel_intercept;
    flat = flat * CollectionTableMetadata::graupel_contents + index.graupel_content;
    flat = flat * CollectionTableMetadata::graupel_densities + index.graupel_density;
    flat = flat * CollectionTableMetadata::rain_intercepts + index.rain_intercept;
    return flat * CollectionTableMetadata::rain_contents + index.rain_content;
}

std::size_t flatten(RainSnowIndex index)
{
    require_index(index.snow_content < CollectionTableMetadata::snow_contents
                  && index.temperature < CollectionTableMetadata::temperatures
                  && index.rain_intercept < CollectionTableMetadata::rain_intercepts
                  && index.rain_content < CollectionTableMetadata::rain_contents);
    std::size_t flat = index.snow_content;
    flat = flat * CollectionTableMetadata::temperatures + index.temperature;
    flat = flat * CollectionTableMetadata::rain_intercepts + index.rain_intercept;
    return flat * CollectionTableMetadata::rain_contents + index.rain_content;
}

RainGraupelCollection generate_rain_graupel_cell(RainGraupelIndex index)
{
    (void)flatten(index);
    const auto& c = context();
    const double rain_content = c.rain_content[index.rain_content];
    const double rain_intercept = c.rain_intercept[index.rain_intercept];
    const double rain_mass_gamma = std::tgamma(bm_rain + mu_rain + 1.0);
    const double rain_number_gamma = std::tgamma(mu_rain + 1.0);
    const double rain_lambda_exponential = std::pow(
        rain_intercept * am_rain * std::tgamma(bm_rain + 1.0) / rain_content,
        1.0 / (bm_rain + 1.0));
    const double lambda_rain = rain_lambda_exponential
                               * std::pow(rain_mass_gamma
                                              / (rain_number_gamma * std::tgamma(bm_rain + 1.0)),
                                          1.0 / bm_rain);
    const double rain_n0 = rain_intercept
                           / (rain_number_gamma * rain_lambda_exponential)
                           * std::pow(lambda_rain, mu_rain + 1.0);
    std::array<double, CollectionTableMetadata::bins> rain_number{};
    for (std::size_t rain = 0; rain < rain_number.size(); ++rain)
        rain_number[rain] = rain_n0 * std::pow(c.rain_diameter[rain], mu_rain)
                            * std::exp(-lambda_rain * c.rain_diameter[rain])
                            * c.rain_width[rain];

    const double graupel_content = c.graupel_content[index.graupel_content];
    const double graupel_intercept = c.graupel_intercept[index.graupel_intercept];
    const double graupel_mass_gamma = std::tgamma(bm_graupel + mu_graupel + 1.0);
    const double graupel_number_gamma = std::tgamma(mu_graupel + 1.0);
    const double graupel_lambda_exponential = std::pow(
        graupel_intercept * am_graupel * std::tgamma(bm_graupel + 1.0) / graupel_content,
        1.0 / (bm_graupel + 1.0));
    const double lambda_graupel = graupel_lambda_exponential
                                  * std::pow(graupel_mass_gamma
                                                 / (graupel_number_gamma
                                                    * std::tgamma(bm_graupel + 1.0)),
                                             1.0 / bm_graupel);
    const double graupel_n0 = graupel_intercept
                              / (graupel_number_gamma * graupel_lambda_exponential)
                              * std::pow(lambda_graupel, mu_graupel + 1.0);
    std::array<double, CollectionTableMetadata::bins> graupel_number{};
    for (std::size_t graupel = 0; graupel < graupel_number.size(); ++graupel)
    {
        graupel_number[graupel] = graupel_n0
                                   * std::pow(c.graupel_diameter[graupel], mu_graupel)
                                   * std::exp(-lambda_graupel * c.graupel_diameter[graupel])
                                   * c.graupel_width[graupel];
    }

    RainGraupelCollection result;
    for (std::size_t rain = 0; rain < CollectionTableMetadata::bins; ++rain)
    {
        const double rain_mass = am_rain * std::pow(c.rain_diameter[rain], bm_rain);
        for (std::size_t graupel = 0; graupel < CollectionTableMetadata::bins; ++graupel)
        {
            const double graupel_mass = am_graupel
                                         * std::pow(c.graupel_diameter[graupel], bm_graupel);
            const double cross_section = pi * 0.25 * rain_graupel_efficiency
                                         * std::pow(c.graupel_diameter[graupel]
                                                        + c.rain_diameter[rain],
                                                    2.0)
                                         * graupel_number[graupel] * rain_number[rain];
            const double rain_faster = std::max(
                0.0, c.rain_velocity[rain] - c.graupel_velocity[graupel]);
            const double graupel_faster = std::max(
                0.0, c.graupel_velocity[graupel] - c.rain_velocity[rain]);
            result.graupel_mass_by_rain += cross_section * rain_faster * graupel_mass;
            result.rain_mass_by_rain += cross_section * rain_faster * rain_mass;
            result.graupel_number_by_rain += cross_section * rain_faster;
            result.rain_mass_by_graupel += cross_section * graupel_faster * rain_mass;
            result.rain_number_by_graupel += cross_section * graupel_faster;
        }
    }
    result.rain_mass_by_rain = std::min(result.rain_mass_by_rain, rain_content);
    return result;
}

RainSnowCollection generate_rain_snow_cell(RainSnowIndex index)
{
    (void)flatten(index);
    const auto& c = context();
    const double rain_content = c.rain_content[index.rain_content];
    const double rain_intercept = c.rain_intercept[index.rain_intercept];
    const double rain_mass_gamma = std::tgamma(bm_rain + mu_rain + 1.0);
    const double rain_number_gamma = std::tgamma(mu_rain + 1.0);
    const double rain_lambda_exponential = std::pow(
        rain_intercept * am_rain * std::tgamma(bm_rain + 1.0) / rain_content,
        1.0 / (bm_rain + 1.0));
    const double lambda_rain = rain_lambda_exponential
                               * std::pow(rain_mass_gamma
                                              / (rain_number_gamma * std::tgamma(bm_rain + 1.0)),
                                          1.0 / bm_rain);
    const double rain_n0 = rain_intercept
                           / (rain_number_gamma * rain_lambda_exponential)
                           * std::pow(lambda_rain, mu_rain + 1.0);
    std::array<double, CollectionTableMetadata::bins> rain_number{};
    for (std::size_t rain = 0; rain < rain_number.size(); ++rain)
        rain_number[rain] = rain_n0 * std::pow(c.rain_diameter[rain], mu_rain)
                            * std::exp(-lambda_rain * c.rain_diameter[rain])
                            * c.rain_width[rain];

    const double temperature = temperatures[index.temperature];
    const double m2 = c.snow_content[index.snow_content] / am_snow;
    const double a2 = std::pow(10.0, snow_polynomial(snow_a, temperature, bm_snow));
    const double b2 = snow_polynomial(snow_b, temperature, bm_snow);
    const double second = std::pow(m2 / a2, 1.0 / b2);
    const double next_moment = bm_snow + 1.0;
    const double a3 = std::pow(10.0, snow_polynomial(snow_a, temperature, next_moment));
    const double b3 = snow_polynomial(snow_b, temperature, next_moment);
    const double m3 = a3 * std::pow(second, b3);
    const double ratio = m2 / m3;
    const double multiplier = m2 * ratio * ratio * ratio;
    const double snow_shape = std::pow(ratio, mu_snow);
    const double lambda_snow_1 = ratio * lam0;
    const double lambda_snow_2 = ratio * lam1;
    std::array<double, CollectionTableMetadata::bins> snow_number{};
    for (std::size_t snow = 0; snow < snow_number.size(); ++snow)
    {
        const double diameter = c.snow_diameter[snow];
        snow_number[snow] = multiplier
                            * (kap0 * std::exp(-lambda_snow_1 * diameter)
                               + kap1 * snow_shape * std::pow(diameter, mu_snow)
                                     * std::exp(-lambda_snow_2 * diameter))
                            * c.snow_width[snow];
    }

    double t1 = 0.0;
    double z1 = 0.0;
    double y1 = 0.0;
    double t3 = 0.0;
    double z3 = 0.0;
    double y3 = 0.0;
    double t2 = 0.0;
    double z2 = 0.0;
    double y2 = 0.0;
    double t4 = 0.0;
    double z4 = 0.0;
    double y4 = 0.0;
    for (std::size_t rain = 0; rain < CollectionTableMetadata::bins; ++rain)
    {
        const double rain_mass = am_rain * std::pow(c.rain_diameter[rain], bm_rain);
        for (std::size_t snow = 0; snow < CollectionTableMetadata::bins; ++snow)
        {
            const double snow_mass = am_snow * std::pow(c.snow_diameter[snow], bm_snow);
            const double cross_section = pi * 0.25 * rain_snow_efficiency
                                         * std::pow(c.snow_diameter[snow]
                                                        + c.rain_diameter[rain],
                                                    2.0)
                                         * snow_number[snow] * rain_number[rain];
            const double rain_faster = std::max(
                0.0, c.rain_velocity[rain] - c.snow_velocity[snow]);
            const double snow_faster = std::max(
                0.0, c.snow_velocity[snow] - c.rain_velocity[rain]);
            const bool rain_dominates = rain_mass > 1.5 * snow_mass;
            if (rain_dominates)
            {
                t1 += cross_section * rain_faster * snow_mass;
                z1 += cross_section * rain_faster * rain_mass;
                y1 += cross_section * rain_faster;
                t2 += cross_section * snow_faster * rain_mass;
                z2 += cross_section * snow_faster * snow_mass;
                y2 += cross_section * snow_faster;
            }
            else
            {
                t3 += cross_section * rain_faster * snow_mass;
                z3 += cross_section * rain_faster * rain_mass;
                y3 += cross_section * rain_faster;
                t4 += cross_section * snow_faster * rain_mass;
                z4 += cross_section * snow_faster * snow_mass;
                y4 += cross_section * snow_faster;
            }
        }
    }
    RainSnowCollection result;
    result.values = {t1, std::min(z1, rain_content), t3, z3,
                     t2, z2, t4, z4, y1, y3, y2, y4};
    return result;
}

CollectionTables generate_collection_tables()
{
    CollectionTables tables;
    tables.rain_graupel.resize(checked_rain_graupel_size());
    tables.rain_snow.resize(checked_rain_snow_size());

    #pragma omp parallel for
    for (std::int64_t rain_content = 0;
         rain_content < static_cast<std::int64_t>(CollectionTableMetadata::rain_contents);
         ++rain_content)
    {
        for (std::size_t rain_intercept = 0;
             rain_intercept < CollectionTableMetadata::rain_intercepts; ++rain_intercept)
        {
            for (std::size_t graupel_content = 0;
                 graupel_content < CollectionTableMetadata::graupel_contents; ++graupel_content)
            {
                for (std::size_t graupel_intercept = 0;
                     graupel_intercept < CollectionTableMetadata::graupel_intercepts;
                     ++graupel_intercept)
                {
                    const RainGraupelIndex index{graupel_intercept, graupel_content, 0,
                                                 rain_intercept,
                                                 static_cast<std::size_t>(rain_content)};
                    tables.rain_graupel[flatten(index)] = generate_rain_graupel_cell(index);
                }
            }
            for (std::size_t temperature = 0;
                 temperature < CollectionTableMetadata::temperatures; ++temperature)
            {
                for (std::size_t snow_content = 0;
                     snow_content < CollectionTableMetadata::snow_contents; ++snow_content)
                {
                    const RainSnowIndex index{snow_content, temperature, rain_intercept,
                                              static_cast<std::size_t>(rain_content)};
                    tables.rain_snow[flatten(index)] = generate_rain_snow_cell(index);
                }
            }
        }
    }
    tables.payload_sha256 = payload_hash(tables);
    return tables;
}

GeneratorMeasurement measure_generator_cost(std::size_t sampled_cells)
{
    if (sampled_cells == 0)
        throw std::invalid_argument("Thompson generator measurement needs samples");
    GeneratorMeasurement measurement;
    measurement.sampled_cells = sampled_cells;
    const auto started = std::chrono::steady_clock::now();
    for (std::size_t sample = 0; sample < sampled_cells; ++sample)
    {
        if ((sample & 1U) == 0)
        {
            const std::size_t n = (sample * 17U) % CollectionTableMetadata::rain_contents;
            const auto cell = generate_rain_graupel_cell(
                {n, (sample * 13U) % CollectionTableMetadata::graupel_contents, 0,
                 (sample * 11U) % CollectionTableMetadata::rain_intercepts, n});
            measurement.checksum += cell.graupel_mass_by_rain
                                    + cell.rain_mass_by_graupel;
        }
        else
        {
            const auto cell = generate_rain_snow_cell(
                {(sample * 13U) % CollectionTableMetadata::snow_contents,
                 (sample * 7U) % CollectionTableMetadata::temperatures,
                 (sample * 11U) % CollectionTableMetadata::rain_intercepts,
                 (sample * 17U) % CollectionTableMetadata::rain_contents});
            measurement.checksum += cell.values[0] + cell.values[11];
        }
    }
    measurement.elapsed_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    measurement.sampled_pair_evaluations =
        static_cast<std::uint64_t>(sampled_cells)
        * CollectionTableMetadata::pair_evaluations_per_cell;
    measurement.projected_full_generation_seconds = measurement.elapsed_seconds
        * static_cast<double>(checked_full_pair_evaluations())
        / static_cast<double>(measurement.sampled_pair_evaluations);
    return measurement;
}

bool validate_collection_tables(const CollectionTables& tables, std::string& error)
{
    if (nimbus::crypto::sha256_hex(constants_identity)
        != CollectionTableMetadata::constants_sha256)
    {
        error = "Thompson collection-table constants identity mismatch";
        return false;
    }
    if (tables.rain_graupel.size() != checked_rain_graupel_size()
        || tables.rain_snow.size() != checked_rain_snow_size())
    {
        error = "Thompson collection-table dimensions do not match WRF v4.7.1 mp=28";
        return false;
    }
    for (const auto& cell : tables.rain_graupel)
    {
        if (!finite_nonnegative(cell.graupel_mass_by_rain)
            || !finite_nonnegative(cell.rain_mass_by_rain)
            || !finite_nonnegative(cell.rain_mass_by_graupel)
            || !finite_nonnegative(cell.graupel_number_by_rain)
            || !finite_nonnegative(cell.rain_number_by_graupel))
        {
            error = "Thompson rain-graupel table contains invalid values";
            return false;
        }
    }
    for (const auto& cell : tables.rain_snow)
    {
        if (!std::all_of(cell.values.begin(), cell.values.end(), finite_nonnegative))
        {
            error = "Thompson rain-snow table contains invalid values";
            return false;
        }
    }
    const std::string computed = payload_hash(tables);
    if (!tables.payload_sha256.empty() && tables.payload_sha256 != computed)
    {
        error = "Thompson collection-table payload hash mismatch";
        return false;
    }
    error.clear();
    return true;
}

bool write_collection_tables(const std::filesystem::path& path,
                             const CollectionTables& tables,
                             std::string& error)
{
    if (!validate_collection_tables(tables, error)) return false;
    const std::string hash = payload_hash(tables);
    const auto temporary = path.string() + ".tmp";
    std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
    if (!out)
    {
        error = "unable to open Thompson collection-table temporary file";
        return false;
    }
    auto write = [&](const void* bytes, std::size_t count) {
        out.write(static_cast<const char*>(bytes), static_cast<std::streamsize>(count));
    };
    write(magic, sizeof(magic));
    consume_u32(CollectionTableMetadata::format_version, write);
    for (std::uint32_t dimension : {
             static_cast<std::uint32_t>(CollectionTableMetadata::bins),
             static_cast<std::uint32_t>(CollectionTableMetadata::rain_contents),
             static_cast<std::uint32_t>(CollectionTableMetadata::rain_intercepts),
             static_cast<std::uint32_t>(CollectionTableMetadata::snow_contents),
             static_cast<std::uint32_t>(CollectionTableMetadata::temperatures),
             static_cast<std::uint32_t>(CollectionTableMetadata::graupel_contents),
             static_cast<std::uint32_t>(CollectionTableMetadata::graupel_intercepts),
             static_cast<std::uint32_t>(CollectionTableMetadata::graupel_densities)})
        consume_u32(dimension, write);
    consume_u64(static_cast<std::uint64_t>(tables.rain_graupel.size()), write);
    consume_u64(static_cast<std::uint64_t>(tables.rain_snow.size()), write);
    const auto source = digest_bytes_from_hex(CollectionTableMetadata::upstream_source_sha256);
    const auto license = digest_bytes_from_hex(CollectionTableMetadata::upstream_license_sha256);
    const auto constants = digest_bytes_from_hex(CollectionTableMetadata::constants_sha256);
    const auto payload = digest_bytes_from_hex(hash.c_str());
    write(source.data(), source.size());
    write(license.data(), license.size());
    write(constants.data(), constants.size());
    write(payload.data(), payload.size());
    consume_payload(tables, write);
    out.flush();
    if (!out)
    {
        error = "failed writing Thompson collection-table payload";
        return false;
    }
    out.close();
    std::error_code ec;
    std::filesystem::rename(temporary, path, ec);
    if (ec)
    {
        error = "failed atomically publishing Thompson collection tables: " + ec.message();
        return false;
    }
    error.clear();
    return true;
}

bool load_collection_tables(const std::filesystem::path& path,
                            CollectionTables& tables,
                            std::string& error,
                            std::size_t maximum_resident_bytes)
{
    if (checked_payload_bytes() > maximum_resident_bytes)
    {
        error = "Thompson collection tables exceed the configured resident-memory budget";
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in)
    {
        error = "unable to open Thompson collection tables";
        return false;
    }
    std::array<char, sizeof(magic)> actual_magic{};
    if (!read_exact(in, actual_magic.data(), actual_magic.size())
        || !std::equal(actual_magic.begin(), actual_magic.end(), std::begin(magic)))
    {
        error = "invalid Thompson collection-table magic";
        return false;
    }
    std::uint32_t version = 0;
    if (!read_u32(in, version) || version != CollectionTableMetadata::format_version)
    {
        error = "unsupported Thompson collection-table format";
        return false;
    }
    const std::array<std::uint32_t, 8> expected_dimensions = {
        static_cast<std::uint32_t>(CollectionTableMetadata::bins),
        static_cast<std::uint32_t>(CollectionTableMetadata::rain_contents),
        static_cast<std::uint32_t>(CollectionTableMetadata::rain_intercepts),
        static_cast<std::uint32_t>(CollectionTableMetadata::snow_contents),
        static_cast<std::uint32_t>(CollectionTableMetadata::temperatures),
        static_cast<std::uint32_t>(CollectionTableMetadata::graupel_contents),
        static_cast<std::uint32_t>(CollectionTableMetadata::graupel_intercepts),
        static_cast<std::uint32_t>(CollectionTableMetadata::graupel_densities)};
    for (std::uint32_t expected : expected_dimensions)
    {
        std::uint32_t actual = 0;
        if (!read_u32(in, actual) || actual != expected)
        {
            error = "Thompson collection-table metadata mismatch";
            return false;
        }
    }
    std::uint64_t graupel_count = 0;
    std::uint64_t snow_count = 0;
    if (!read_u64(in, graupel_count) || !read_u64(in, snow_count)
        || graupel_count != checked_rain_graupel_size()
        || snow_count != checked_rain_snow_size())
    {
        error = "Thompson collection-table element count mismatch";
        return false;
    }
    std::array<std::uint8_t, digest_bytes> source{};
    std::array<std::uint8_t, digest_bytes> license{};
    std::array<std::uint8_t, digest_bytes> constants{};
    std::array<std::uint8_t, digest_bytes> expected_payload{};
    if (!read_exact(in, source.data(), source.size())
        || !read_exact(in, license.data(), license.size())
        || !read_exact(in, constants.data(), constants.size())
        || !read_exact(in, expected_payload.data(), expected_payload.size())
        || source != digest_bytes_from_hex(CollectionTableMetadata::upstream_source_sha256)
        || license != digest_bytes_from_hex(CollectionTableMetadata::upstream_license_sha256)
        || constants != digest_bytes_from_hex(CollectionTableMetadata::constants_sha256))
    {
        error = "Thompson collection-table upstream identity mismatch";
        return false;
    }

    CollectionTables candidate;
    candidate.rain_graupel.resize(static_cast<std::size_t>(graupel_count));
    candidate.rain_snow.resize(static_cast<std::size_t>(snow_count));
    for (auto& cell : candidate.rain_graupel)
    {
        if (!read_double(in, cell.graupel_mass_by_rain)
            || !read_double(in, cell.rain_mass_by_rain)
            || !read_double(in, cell.rain_mass_by_graupel)
            || !read_double(in, cell.graupel_number_by_rain)
            || !read_double(in, cell.rain_number_by_graupel))
        {
            error = "truncated Thompson rain-graupel table";
            return false;
        }
    }
    for (auto& cell : candidate.rain_snow)
    {
        for (double& value : cell.values)
        {
            if (!read_double(in, value))
            {
                error = "truncated Thompson rain-snow table";
                return false;
            }
        }
    }
    char trailing = 0;
    if (in.read(&trailing, 1))
    {
        error = "unexpected trailing bytes in Thompson collection tables";
        return false;
    }
    candidate.payload_sha256 = hex_digest(expected_payload);
    if (!validate_collection_tables(candidate, error)) return false;
    tables = std::move(candidate);
    error.clear();
    return true;
}

bool CollectionTableStore::publish_file(const std::filesystem::path& path,
                                        std::string& error,
                                        std::size_t maximum_resident_bytes)
{
    auto candidate = std::make_shared<CollectionTables>();
    if (!load_collection_tables(path, *candidate, error, maximum_resident_bytes))
        return false;
    std::atomic_store_explicit(&tables_,
                               std::static_pointer_cast<const CollectionTables>(candidate),
                               std::memory_order_release);
    error.clear();
    return true;
}

bool CollectionTableStore::publish_packaged_file(
    const std::filesystem::path& path,
    std::string& error,
    std::size_t maximum_resident_bytes)
{
    auto candidate = std::make_shared<CollectionTables>();
    if (!load_collection_tables(path, *candidate, error, maximum_resident_bytes))
        return false;
    if (candidate->payload_sha256
        != CollectionTableMetadata::packaged_payload_sha256)
    {
        error = "Thompson packaged collection-table payload mismatch";
        return false;
    }
    std::atomic_store_explicit(&tables_,
                               std::static_pointer_cast<const CollectionTables>(candidate),
                               std::memory_order_release);
    error.clear();
    return true;
}

std::shared_ptr<const CollectionTables> CollectionTableStore::snapshot() const noexcept
{
    return std::atomic_load_explicit(&tables_, std::memory_order_acquire);
}

} // namespace tmv::thompson
