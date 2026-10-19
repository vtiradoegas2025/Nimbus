#include "chaos/control_grid_geometry.hpp"
#include "earth_correlated_noise.hpp"
#include "compute/backend_interface.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tmv::chaos
{
namespace
{

constexpr double two_pi = 6.283185307179586476925286766559;

std::uint64_t splitmix64(std::uint64_t value) noexcept
{
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::uint64_t fold(std::uint64_t state, std::uint64_t value) noexcept
{
    return splitmix64(state ^ splitmix64(value));
}

double open_unit(std::uint64_t bits) noexcept
{
    // Midpoints of the 2^53 equiprobable bins exclude both singular endpoints.
    constexpr double inverse_2_pow_53 = 1.0 / 9007199254740992.0;
    return (static_cast<double>(bits >> 11U) + 0.5) * inverse_2_pow_53;
}

std::array<double, 4> cubic_weights(double fraction) noexcept
{
    const double one_minus = 1.0 - fraction;
    const double fraction2 = fraction * fraction;
    const double fraction3 = fraction2 * fraction;
    return {{
        one_minus * one_minus * one_minus / 6.0,
        (3.0 * fraction3 - 6.0 * fraction2 + 4.0) / 6.0,
        (-3.0 * fraction3 + 3.0 * fraction2 + 3.0 * fraction + 1.0) / 6.0,
        fraction3 / 6.0,
    }};
}

bool finite_field(const Field3D& field)
{
    for (std::size_t index = 0; index < field.size(); ++index)
        if (!std::isfinite(static_cast<double>(field.data()[index])))
            return false;
    return true;
}

} // namespace

void EarthCorrelatedNoise::configure_native_evolution(
    ComputeBackend* backend, bool required) noexcept
{
    native_backend_ = backend;
    native_evolution_required_ = required;
}

bool EarthCorrelatedNoise::dispatch_native_evolution(
    Field3D& destination,
    const Field3D& previous,
    std::uint64_t epoch,
    double rho,
    EarthControlEvolutionMode mode,
    std::int64_t control_i_min,
    std::int64_t control_j_min) const
{
    if (native_backend_ == nullptr)
        return false;
    EarthControlEvolutionDispatch dispatch;
    dispatch.previous = previous.data();
    dispatch.destination = destination.data();
    dispatch.seed = seed_;
    dispatch.member_id = member_id_;
    dispatch.initial_process_key = initial_process_key_;
    dispatch.innovation_process_key = innovation_process_key_;
    dispatch.innovation_epoch = epoch;
    dispatch.control_i_min = control_i_min;
    dispatch.control_j_min = control_j_min;
    dispatch.control_i_count = destination.size_r();
    dispatch.control_j_count = destination.size_th();
    dispatch.nz = destination.size_z();
    dispatch.rho = rho;
    dispatch.mode = mode;
    return native_backend_->dispatch_earth_control_evolution(dispatch);
}

void EarthCorrelatedNoise::configure(
    std::uint64_t seed,
    int member_id,
    std::uint64_t initial_process_key,
    std::uint64_t innovation_process_key,
    double correlation_length_x_m,
    double correlation_length_y_m,
    double dx_m,
    double dy_m)
{
    if (initialized_)
        throw std::logic_error(
            "initialized Earth-correlated chaos cannot be reconfigured");
    if (!std::isfinite(correlation_length_x_m)
        || correlation_length_x_m <= 0.0
        || !std::isfinite(correlation_length_y_m)
        || correlation_length_y_m <= 0.0
        || !std::isfinite(dx_m) || dx_m <= 0.0
        || !std::isfinite(dy_m) || dy_m <= 0.0)
        throw std::invalid_argument(
            "Earth-correlated chaos requires positive finite lengths and spacing");

    const double control_dx = earth_control_spacing(dx_m, correlation_length_x_m);
    const double control_dy = earth_control_spacing(dy_m, correlation_length_y_m);
    seed_ = seed;
    member_id_ = member_id;
    initial_process_key_ = initial_process_key;
    innovation_process_key_ = innovation_process_key;
    dx_m_ = dx_m;
    dy_m_ = dy_m;
    // Below-grid correlation lengths asymptote to cell-independent noise.
    // A quarter-cell control spacing makes adjacent cubic supports disjoint
    // without allowing an unresolved requested scale to explode memory.
    control_spacing_x_m_ = control_dx;
    control_spacing_y_m_ = control_dy;
    std::uint64_t phase_key = splitmix64(seed_);
    phase_key = fold(phase_key, static_cast<std::uint64_t>(
                                   static_cast<std::int64_t>(member_id_)));
    phase_key = fold(phase_key, initial_process_key_);
    control_phase_x_ = open_unit(splitmix64(
        phase_key ^ 0xe7037ed1a0b428dbULL));
    control_phase_y_ = open_unit(splitmix64(
        phase_key ^ 0x8ebc6af09c88c6e3ULL));
    configured_ = true;
}

EarthCorrelatedNoise::ControlBounds
EarthCorrelatedNoise::required_control_bounds(
    std::int64_t origin_i,
    std::int64_t origin_j,
    int nr,
    int nth,
    double continuous_offset_x_m,
    double continuous_offset_y_m) const
{
    constexpr std::int64_t largest_exact_double_integer =
        static_cast<std::int64_t>(1ULL << 52U);
    auto axis_bounds = [=](std::int64_t origin, int count, double spacing,
                          double control_spacing, double control_phase,
                          double continuous_offset_m) {
        if (count <= 0
            || origin > std::numeric_limits<std::int64_t>::max()
                            - static_cast<std::int64_t>(count - 1))
            throw std::overflow_error(
                "Earth-correlated chaos cell address overflow");
        const std::int64_t last_cell =
            origin + static_cast<std::int64_t>(count - 1);
        if (origin < -largest_exact_double_integer
            || origin > largest_exact_double_integer
            || last_cell < -largest_exact_double_integer
            || last_cell > largest_exact_double_integer)
            throw std::overflow_error(
                "Earth-correlated chaos cell address exceeds exact metric range");
        const double first =
            ((static_cast<double>(origin) + 0.5) * spacing
             + continuous_offset_m) / control_spacing
            + control_phase;
        const double last =
            ((static_cast<double>(last_cell) + 0.5) * spacing
             + continuous_offset_m) / control_spacing
            + control_phase;
        const double lower = std::min(first, last);
        const double upper = std::max(first, last);
        return std::array<std::int64_t, 2>{{
            static_cast<std::int64_t>(std::floor(lower)) - 1,
            static_cast<std::int64_t>(std::floor(upper)) + 2,
        }};
    };
    const auto i = axis_bounds(
        origin_i, nr, dx_m_, control_spacing_x_m_, control_phase_x_,
        continuous_offset_x_m);
    const auto j = axis_bounds(
        origin_j, nth, dy_m_, control_spacing_y_m_, control_phase_y_,
        continuous_offset_y_m);
    return {i[0], i[1], j[0], j[1]};
}

double EarthCorrelatedNoise::gaussian(
    std::uint64_t process_key,
    std::uint64_t epoch,
    std::int64_t control_i,
    std::int64_t control_j,
    int k) const noexcept
{
    std::uint64_t address = splitmix64(seed_);
    address = fold(address, static_cast<std::uint64_t>(
                                static_cast<std::int64_t>(member_id_)));
    address = fold(address, process_key);
    address = fold(address, epoch);
    address = fold(address, static_cast<std::uint64_t>(control_i));
    address = fold(address, static_cast<std::uint64_t>(control_j));
    address = fold(address, static_cast<std::uint64_t>(k));
    const double u1 = open_unit(splitmix64(address));
    const double u2 = open_unit(splitmix64(address ^ 0xd1b54a32d192ed03ULL));
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(two_pi * u2);
}

double EarthCorrelatedNoise::reconstruct(
    std::int64_t control_i,
    std::int64_t control_j,
    int k,
    std::uint64_t epoch,
    double rho) const noexcept
{
    float value = static_cast<float>(gaussian(
        initial_process_key_, 0, control_i, control_j, k));
    if (epoch == 0 || rho == 1.0)
        return static_cast<double>(value);
    if (rho == 0.0)
        return gaussian(
            innovation_process_key_, epoch - 1,
            control_i, control_j, k);
    const double innovation_scale =
        std::sqrt(std::max(0.0, 1.0 - rho * rho));
    for (std::uint64_t step = 0; step < epoch; ++step)
        value = static_cast<float>(
            rho * static_cast<double>(value)
            + innovation_scale * gaussian(
                innovation_process_key_, step, control_i, control_j, k));
    return static_cast<double>(value);
}

void EarthCorrelatedNoise::initialize_domain(int nr, int nth, int nz)
{
    EarthCorrelatedNoise candidate(*this);
    candidate.initialize_domain_in_place(nr, nth, nz);
    swap_window_state(candidate);
}

void EarthCorrelatedNoise::initialize_domain_in_place(int nr, int nth, int nz)
{
    if (!configured_)
        throw std::logic_error("Earth-correlated chaos is not configured");
    if (nr <= 0 || nth <= 0 || nz <= 0)
        throw std::invalid_argument(
            "Earth-correlated chaos requires a positive domain");
    nr_ = nr;
    nth_ = nth;
    nz_ = nz;
    const ControlBounds bounds = required_control_bounds(
        earth_origin_i_cells_, earth_origin_j_cells_, nr_, nth_,
        continuous_offset_x_m_, continuous_offset_y_m_);
    control_i_min_ = bounds.i_min;
    control_j_min_ = bounds.j_min;
    const std::int64_t ci_span = bounds.i_max - bounds.i_min + 1;
    const std::int64_t cj_span = bounds.j_max - bounds.j_min + 1;
    if (ci_span <= 0 || cj_span <= 0
        || ci_span > std::numeric_limits<int>::max()
        || cj_span > std::numeric_limits<int>::max())
        throw std::overflow_error(
            "Earth-correlated chaos control lattice is too large");
    const auto ci_count = static_cast<int>(ci_span);
    const auto cj_count = static_cast<int>(cj_span);
    current_controls_.resize(ci_count, cj_count, nz_);
    if (native_backend_ != nullptr || native_evolution_required_)
    {
        Field3D zero(ci_count, cj_count, nz_, 0.0f);
        if (dispatch_native_evolution(
                current_controls_, zero, 0, 0.0,
                EarthControlEvolutionMode::Initialize,
                control_i_min_, control_j_min_))
        {
            previous_controls_ = current_controls_;
            initialized_ = true;
            rebuild_stencils();
            return;
        }
        if (native_evolution_required_)
            throw std::runtime_error(
                "native Earth-control initialization dispatch failed");
    }
#pragma omp parallel for collapse(2) schedule(static) \
    if(static_cast<long long>(ci_count) * cj_count * nz_ >= 4096)
    for (int ci = 0; ci < ci_count; ++ci)
        for (int cj = 0; cj < cj_count; ++cj)
            for (int k = 0; k < nz_; ++k)
                current_controls_(ci, cj, k) = static_cast<float>(gaussian(
                    initial_process_key_, 0,
                    control_i_min_ + ci, control_j_min_ + cj, k));
    previous_controls_ = current_controls_;
    initialized_ = true;
    rebuild_stencils();
}

void EarthCorrelatedNoise::initialize_domain_from(
    const EarthCorrelatedNoise& source,
    std::int64_t earth_origin_i_cells,
    std::int64_t earth_origin_j_cells,
    int nr,
    int nth,
    int nz,
    std::uint64_t source_epoch,
    std::uint64_t target_epoch,
    double rho)
{
    EarthCorrelatedNoise candidate(*this);
    candidate.initialize_domain_from_in_place(
        source, earth_origin_i_cells, earth_origin_j_cells,
        nr, nth, nz, source_epoch, target_epoch, rho);
    swap_window_state(candidate);
}

void EarthCorrelatedNoise::initialize_domain_from_in_place(
    const EarthCorrelatedNoise& source,
    std::int64_t earth_origin_i_cells,
    std::int64_t earth_origin_j_cells,
    int nr,
    int nth,
    int nz,
    std::uint64_t source_epoch,
    std::uint64_t target_epoch,
    double rho)
{
    if (!configured_ || !source.configured_ || !source.initialized_)
        throw std::logic_error(
            "nested Earth-correlated chaos requires initialized source controls");
    if (!std::isfinite(rho) || rho < 0.0 || rho > 1.0)
        throw std::invalid_argument(
            "nested Earth-correlated chaos AR(1) rho is invalid");
    if (target_epoch > source_epoch)
        throw std::invalid_argument(
            "nested Earth-correlated chaos cannot initialize from a future epoch");
    if (seed_ != source.seed_ || member_id_ != source.member_id_ ||
        initial_process_key_ != source.initial_process_key_ ||
        innovation_process_key_ != source.innovation_process_key_ ||
        control_spacing_x_m_ != source.control_spacing_x_m_ ||
        control_spacing_y_m_ != source.control_spacing_y_m_ ||
        control_phase_x_ != source.control_phase_x_ ||
        control_phase_y_ != source.control_phase_y_)
        throw std::invalid_argument(
            "nested Earth-correlated chaos lattices are not identical");
    if (nr <= 0 || nth <= 0 || nz <= 0 || nz != source.nz_)
        throw std::invalid_argument(
            "nested Earth-correlated chaos requires shared positive levels");

    earth_origin_i_cells_ = earth_origin_i_cells;
    earth_origin_j_cells_ = earth_origin_j_cells;
    nr_ = nr;
    nth_ = nth;
    nz_ = nz;
    const ControlBounds bounds = required_control_bounds(
        earth_origin_i_cells_, earth_origin_j_cells_, nr_, nth_);
    const std::int64_t ci_span = bounds.i_max - bounds.i_min + 1;
    const std::int64_t cj_span = bounds.j_max - bounds.j_min + 1;
    if (ci_span <= 0 || cj_span <= 0 ||
        ci_span > std::numeric_limits<int>::max() ||
        cj_span > std::numeric_limits<int>::max())
        throw std::overflow_error(
            "nested Earth-correlated chaos control lattice is too large");
    const int ci_count = static_cast<int>(ci_span);
    const int cj_count = static_cast<int>(cj_span);
    Field3D current(ci_count, cj_count, nz_);
    Field3D previous(ci_count, cj_count, nz_);
    bool native_reconstructed = false;
    if (native_backend_ != nullptr || native_evolution_required_)
    {
        Field3D zero(ci_count, cj_count, nz_, 0.0f);
        native_reconstructed = dispatch_native_evolution(
            current, zero, target_epoch, rho,
            EarthControlEvolutionMode::Reconstruct,
            bounds.i_min, bounds.j_min);
        if (native_reconstructed)
            native_reconstructed = dispatch_native_evolution(
                previous, zero, target_epoch == 0 ? 0 : target_epoch - 1,
                rho, EarthControlEvolutionMode::Reconstruct,
                bounds.i_min, bounds.j_min);
        if (!native_reconstructed && native_evolution_required_)
            throw std::runtime_error(
                "native nested Earth-control reconstruction dispatch failed");
    }
    const std::int64_t source_i_max =
        source.control_i_min_ + source.current_controls_.size_r() - 1;
    const std::int64_t source_j_max =
        source.control_j_min_ + source.current_controls_.size_th() - 1;
    const std::uint64_t previous_target_epoch =
        target_epoch == 0 ? 0 : target_epoch - 1;

#pragma omp parallel for collapse(2) schedule(static) \
    if(static_cast<long long>(ci_count) * cj_count * nz_ >= 4096)
    for (int ci = 0; ci < ci_count; ++ci)
        for (int cj = 0; cj < cj_count; ++cj)
        {
            const std::int64_t earth_ci = bounds.i_min + ci;
            const std::int64_t earth_cj = bounds.j_min + cj;
            const bool resident =
                earth_ci >= source.control_i_min_ &&
                earth_ci <= source_i_max &&
                earth_cj >= source.control_j_min_ &&
                earth_cj <= source_j_max;
            for (int k = 0; k < nz_; ++k)
            {
                const int source_ci = resident
                    ? static_cast<int>(earth_ci - source.control_i_min_) : 0;
                const int source_cj = resident
                    ? static_cast<int>(earth_cj - source.control_j_min_) : 0;
                if (resident && target_epoch == source_epoch)
                    current(ci, cj, k) =
                        source.current_controls_(source_ci, source_cj, k);
                else if (resident && source_epoch > 0 &&
                         target_epoch + 1 == source_epoch)
                    current(ci, cj, k) =
                        source.previous_controls_(source_ci, source_cj, k);
                else if (!native_reconstructed)
                    current(ci, cj, k) = static_cast<float>(reconstruct(
                        earth_ci, earth_cj, k, target_epoch, rho));

                if (target_epoch == 0)
                    previous(ci, cj, k) = current(ci, cj, k);
                else if (resident && previous_target_epoch == source_epoch)
                    previous(ci, cj, k) =
                        source.current_controls_(source_ci, source_cj, k);
                else if (resident && source_epoch > 0 &&
                         previous_target_epoch + 1 == source_epoch)
                    previous(ci, cj, k) =
                        source.previous_controls_(source_ci, source_cj, k);
                else if (!native_reconstructed)
                    previous(ci, cj, k) = static_cast<float>(reconstruct(
                        earth_ci, earth_cj, k, previous_target_epoch, rho));
            }
        }

    current_controls_ = std::move(current);
    previous_controls_ = std::move(previous);
    control_i_min_ = bounds.i_min;
    control_j_min_ = bounds.j_min;
    initialized_ = true;
    rebuild_stencils();
}

void EarthCorrelatedNoise::advance(double rho, std::uint64_t innovation_epoch)
{
    if (!initialized_)
        return;
    if (!std::isfinite(rho) || rho < 0.0 || rho > 1.0)
        throw std::invalid_argument("Earth-correlated chaos AR(1) rho is invalid");
    Field3D next_previous(current_controls_);
    Field3D next_current(
        current_controls_.size_r(), current_controls_.size_th(), nz_);
    if (native_backend_ != nullptr || native_evolution_required_)
    {
        if (dispatch_native_evolution(
                next_current, next_previous, innovation_epoch, rho,
                EarthControlEvolutionMode::Advance,
                control_i_min_, control_j_min_))
        {
            previous_controls_ = std::move(next_previous);
            current_controls_ = std::move(next_current);
            return;
        }
        if (native_evolution_required_)
            throw std::runtime_error(
                "native Earth-control advance dispatch failed");
    }
    const double innovation_scale =
        std::sqrt(std::max(0.0, 1.0 - rho * rho));
#pragma omp parallel for collapse(2) schedule(static) \
    if(static_cast<long long>(next_current.size_r()) \
       * next_current.size_th() * nz_ >= 4096)
    for (int ci = 0; ci < next_current.size_r(); ++ci)
        for (int cj = 0; cj < next_current.size_th(); ++cj)
            for (int k = 0; k < nz_; ++k)
            {
                const double innovation = gaussian(
                    innovation_process_key_, innovation_epoch,
                    control_i_min_ + ci, control_j_min_ + cj, k);
                next_current(ci, cj, k) = static_cast<float>(
                    rho * static_cast<double>(next_previous(ci, cj, k))
                    + innovation_scale * innovation);
            }
    previous_controls_ = std::move(next_previous);
    current_controls_ = std::move(next_current);
}

void EarthCorrelatedNoise::translate(
    int shift_i_cells,
    int shift_j_cells,
    std::uint64_t current_epoch,
    double rho)
{
    EarthCorrelatedNoise candidate;
    prepare_translated_window_in_place(
        candidate, shift_i_cells, shift_j_cells, current_epoch, rho,
        continuous_offset_x_m_, continuous_offset_y_m_);
    swap_window_state(candidate);
}

void EarthCorrelatedNoise::set_continuous_earth_origin(
    double origin_x_m,
    double origin_y_m,
    std::uint64_t current_epoch,
    double rho)
{
    if (!configured_ || !std::isfinite(origin_x_m) ||
        !std::isfinite(origin_y_m) || !std::isfinite(rho) ||
        rho < 0.0 || rho > 1.0)
        throw std::invalid_argument(
            "continuous Earth-correlated chaos origin is invalid");
    const double cell_i_value = std::floor(origin_x_m / dx_m_);
    const double cell_j_value = std::floor(origin_y_m / dy_m_);
    constexpr double int64_upper_exclusive = 9223372036854775808.0;
    if (cell_i_value < -int64_upper_exclusive ||
        cell_i_value >= int64_upper_exclusive ||
        cell_j_value < -int64_upper_exclusive ||
        cell_j_value >= int64_upper_exclusive)
        throw std::overflow_error(
            "continuous Earth-correlated chaos origin exceeds cell range");
    const auto target_i = static_cast<std::int64_t>(cell_i_value);
    const auto target_j = static_cast<std::int64_t>(cell_j_value);
    const double offset_x = origin_x_m - cell_i_value * dx_m_;
    const double offset_y = origin_y_m - cell_j_value * dy_m_;
    if (target_i == earth_origin_i_cells_ &&
        target_j == earth_origin_j_cells_ &&
        offset_x == continuous_offset_x_m_ &&
        offset_y == continuous_offset_y_m_)
        return;
    if (!initialized_)
    {
        earth_origin_i_cells_ = target_i;
        earth_origin_j_cells_ = target_j;
        continuous_offset_x_m_ = offset_x;
        continuous_offset_y_m_ = offset_y;
        return;
    }
    const ControlBounds target_bounds = required_control_bounds(
        target_i, target_j, nr_, nth_, offset_x, offset_y);
    if (target_bounds.i_min == control_i_min_ &&
        target_bounds.j_min == control_j_min_ &&
        target_bounds.i_max - target_bounds.i_min + 1 ==
            current_controls_.size_r() &&
        target_bounds.j_max - target_bounds.j_min + 1 ==
            current_controls_.size_th())
    {
        earth_origin_i_cells_ = target_i;
        earth_origin_j_cells_ = target_j;
        continuous_offset_x_m_ = offset_x;
        continuous_offset_y_m_ = offset_y;
        rebuild_stencils();
        return;
    }
    const long double shift_i_value =
        static_cast<long double>(target_i) - earth_origin_i_cells_;
    const long double shift_j_value =
        static_cast<long double>(target_j) - earth_origin_j_cells_;
    if (shift_i_value < std::numeric_limits<int>::min() ||
        shift_i_value > std::numeric_limits<int>::max() ||
        shift_j_value < std::numeric_limits<int>::min() ||
        shift_j_value > std::numeric_limits<int>::max())
        throw std::overflow_error(
            "continuous Earth-correlated chaos shift exceeds transaction range");
    const int shift_i = static_cast<int>(shift_i_value);
    const int shift_j = static_cast<int>(shift_j_value);
    EarthCorrelatedNoise candidate;
    prepare_translated_window_in_place(
        candidate, shift_i, shift_j,
        current_epoch, rho, offset_x, offset_y);
    swap_window_state(candidate);
}

void EarthCorrelatedNoise::prepare_translated_window(
    EarthCorrelatedNoise& destination,
    int shift_i_cells,
    int shift_j_cells,
    std::uint64_t current_epoch,
    double rho) const
{
    if (&destination == this)
        throw std::invalid_argument(
            "Earth-correlated chaos translation workspace aliases live state");

    const bool shifted_origin_in_range =
        !((shift_i_cells > 0
           && earth_origin_i_cells_
                  > std::numeric_limits<std::int64_t>::max() - shift_i_cells)
          || (shift_i_cells < 0
              && earth_origin_i_cells_
                     < std::numeric_limits<std::int64_t>::min() - shift_i_cells)
          || (shift_j_cells > 0
              && earth_origin_j_cells_
                     > std::numeric_limits<std::int64_t>::max() - shift_j_cells)
          || (shift_j_cells < 0
              && earth_origin_j_cells_
                     < std::numeric_limits<std::int64_t>::min() - shift_j_cells));
    if (initialized_ && native_backend_ == nullptr
        && destination.native_backend_ == nullptr
        && shifted_origin_in_range && std::isfinite(rho)
        && rho >= 0.0 && rho <= 1.0)
    {
        const std::int64_t shifted_origin_i =
            earth_origin_i_cells_ + shift_i_cells;
        const std::int64_t shifted_origin_j =
            earth_origin_j_cells_ + shift_j_cells;
        const ControlBounds bounds = required_control_bounds(
            shifted_origin_i, shifted_origin_j, nr_, nth_,
            continuous_offset_x_m_, continuous_offset_y_m_);
        const std::int64_t ci_span = bounds.i_max - bounds.i_min + 1;
        const std::int64_t cj_span = bounds.j_max - bounds.j_min + 1;
        const bool reusable_window = destination.initialized_
            && destination.earth_origin_i_cells_ == shifted_origin_i
            && destination.earth_origin_j_cells_ == shifted_origin_j
            && destination.continuous_offset_x_m_ == continuous_offset_x_m_
            && destination.continuous_offset_y_m_ == continuous_offset_y_m_
            && destination.nr_ == nr_ && destination.nth_ == nth_
            && destination.nz_ == nz_ && ci_span > 0 && cj_span > 0
            && ci_span <= std::numeric_limits<int>::max()
            && cj_span <= std::numeric_limits<int>::max()
            && destination.current_controls_.size_r() == ci_span
            && destination.current_controls_.size_th() == cj_span
            && destination.current_controls_.size_z() == nz_
            && destination.previous_controls_.size_r() == ci_span
            && destination.previous_controls_.size_th() == cj_span
            && destination.previous_controls_.size_z() == nz_
            && destination.i_stencils_.size()
                   == static_cast<std::size_t>(nr_)
            && destination.j_stencils_.size()
                   == static_cast<std::size_t>(nth_);
        if (reusable_window)
        {
            // Every potentially allocating container already has its exact
            // final size, and CPU reconstruction is noexcept. Reusing this
            // detached window preserves failure atomicity and avoids replacing
            // its storage on an identical relocation retry.
            prepare_translated_window_in_place(
                destination, shift_i_cells, shift_j_cells,
                current_epoch, rho,
                continuous_offset_x_m_, continuous_offset_y_m_);
            return;
        }
    }

    EarthCorrelatedNoise candidate;
    prepare_translated_window_in_place(
        candidate, shift_i_cells, shift_j_cells, current_epoch, rho,
        continuous_offset_x_m_, continuous_offset_y_m_);
    destination.swap_window_state(candidate);
}

void EarthCorrelatedNoise::prepare_translated_window_in_place(
    EarthCorrelatedNoise& destination,
    int shift_i_cells,
    int shift_j_cells,
    std::uint64_t current_epoch,
    double rho,
    double target_offset_x_m,
    double target_offset_y_m) const
{
    if ((shift_i_cells > 0
         && earth_origin_i_cells_
                > std::numeric_limits<std::int64_t>::max() - shift_i_cells)
        || (shift_i_cells < 0
            && earth_origin_i_cells_
                < std::numeric_limits<std::int64_t>::min() - shift_i_cells)
        || (shift_j_cells > 0
            && earth_origin_j_cells_
                > std::numeric_limits<std::int64_t>::max() - shift_j_cells)
        || (shift_j_cells < 0
            && earth_origin_j_cells_
                < std::numeric_limits<std::int64_t>::min() - shift_j_cells))
        throw std::overflow_error(
            "Earth-correlated chaos translated origin overflow");
    if (initialized_ &&
        (!std::isfinite(rho) || rho < 0.0 || rho > 1.0))
        throw std::invalid_argument(
            "Earth-correlated chaos AR(1) rho is invalid");
    destination.seed_ = seed_;
    destination.member_id_ = member_id_;
    destination.initial_process_key_ = initial_process_key_;
    destination.innovation_process_key_ = innovation_process_key_;
    destination.dx_m_ = dx_m_;
    destination.dy_m_ = dy_m_;
    destination.control_spacing_x_m_ = control_spacing_x_m_;
    destination.control_spacing_y_m_ = control_spacing_y_m_;
    destination.control_phase_x_ = control_phase_x_;
    destination.control_phase_y_ = control_phase_y_;
    destination.nr_ = nr_;
    destination.nth_ = nth_;
    destination.nz_ = nz_;
    destination.earth_origin_i_cells_ =
        earth_origin_i_cells_ + shift_i_cells;
    destination.earth_origin_j_cells_ =
        earth_origin_j_cells_ + shift_j_cells;
    destination.continuous_offset_x_m_ = target_offset_x_m;
    destination.continuous_offset_y_m_ = target_offset_y_m;
    destination.configured_ = configured_;
    destination.initialized_ = initialized_;
    destination.native_backend_ = native_backend_;
    destination.native_evolution_required_ = native_evolution_required_;
    if (!initialized_)
        return;

    const ControlBounds bounds = required_control_bounds(
        destination.earth_origin_i_cells_,
        destination.earth_origin_j_cells_, nr_, nth_,
        target_offset_x_m, target_offset_y_m);
    const std::int64_t ci_span = bounds.i_max - bounds.i_min + 1;
    const std::int64_t cj_span = bounds.j_max - bounds.j_min + 1;
    if (ci_span <= 0 || cj_span <= 0 ||
        ci_span > std::numeric_limits<int>::max() ||
        cj_span > std::numeric_limits<int>::max())
        throw std::overflow_error(
            "Earth-correlated chaos translated control lattice is too large");
    const int ci_count = static_cast<int>(ci_span);
    const int cj_count = static_cast<int>(cj_span);
    destination.current_controls_.resize(ci_count, cj_count, nz_);
    destination.previous_controls_.resize(ci_count, cj_count, nz_);
    bool native_reconstructed = false;
    if (destination.native_backend_ != nullptr || destination.native_evolution_required_)
    {
        Field3D zero(ci_count, cj_count, nz_, 0.0f);
        native_reconstructed = destination.dispatch_native_evolution(
            destination.current_controls_, zero, current_epoch, rho,
            EarthControlEvolutionMode::Reconstruct,
            bounds.i_min, bounds.j_min);
        if (native_reconstructed)
            native_reconstructed = destination.dispatch_native_evolution(
                destination.previous_controls_, zero,
                current_epoch == 0 ? 0 : current_epoch - 1,
                rho, EarthControlEvolutionMode::Reconstruct,
                bounds.i_min, bounds.j_min);
        if (!native_reconstructed && native_evolution_required_)
            throw std::runtime_error(
                "native prepared Earth-control reconstruction dispatch failed");
    }
    const std::int64_t old_i_max =
        control_i_min_ + current_controls_.size_r() - 1;
    const std::int64_t old_j_max =
        control_j_min_ + current_controls_.size_th() - 1;
    const std::uint64_t previous_epoch =
        current_epoch == 0 ? 0 : current_epoch - 1;

#pragma omp parallel for collapse(2) schedule(static) \
    if(static_cast<long long>(ci_count) * cj_count * nz_ >= 4096)
    for (int ci = 0; ci < ci_count; ++ci)
        for (int cj = 0; cj < cj_count; ++cj)
        {
            const std::int64_t earth_ci = bounds.i_min + ci;
            const std::int64_t earth_cj = bounds.j_min + cj;
            const bool retained =
                earth_ci >= control_i_min_ && earth_ci <= old_i_max &&
                earth_cj >= control_j_min_ && earth_cj <= old_j_max;
            for (int k = 0; k < nz_; ++k)
            {
                if (retained)
                {
                    const int old_ci =
                        static_cast<int>(earth_ci - control_i_min_);
                    const int old_cj =
                        static_cast<int>(earth_cj - control_j_min_);
                    destination.current_controls_(ci, cj, k) =
                        current_controls_(old_ci, old_cj, k);
                    destination.previous_controls_(ci, cj, k) =
                        previous_controls_(old_ci, old_cj, k);
                }
                else if (!native_reconstructed)
                {
                    destination.current_controls_(ci, cj, k) =
                        static_cast<float>(reconstruct(
                            earth_ci, earth_cj, k, current_epoch, rho));
                    destination.previous_controls_(ci, cj, k) =
                        static_cast<float>(reconstruct(
                            earth_ci, earth_cj, k, previous_epoch, rho));
                }
            }
        }

    destination.control_i_min_ = bounds.i_min;
    destination.control_j_min_ = bounds.j_min;
    destination.rebuild_stencils();
}

void EarthCorrelatedNoise::swap_window_state(
    EarthCorrelatedNoise& other) noexcept
{
    using std::swap;
    swap(seed_, other.seed_);
    swap(member_id_, other.member_id_);
    swap(initial_process_key_, other.initial_process_key_);
    swap(innovation_process_key_, other.innovation_process_key_);
    swap(dx_m_, other.dx_m_);
    swap(dy_m_, other.dy_m_);
    swap(control_spacing_x_m_, other.control_spacing_x_m_);
    swap(control_spacing_y_m_, other.control_spacing_y_m_);
    swap(control_phase_x_, other.control_phase_x_);
    swap(control_phase_y_, other.control_phase_y_);
    swap(nr_, other.nr_);
    swap(nth_, other.nth_);
    swap(nz_, other.nz_);
    swap(earth_origin_i_cells_, other.earth_origin_i_cells_);
    swap(earth_origin_j_cells_, other.earth_origin_j_cells_);
    swap(continuous_offset_x_m_, other.continuous_offset_x_m_);
    swap(continuous_offset_y_m_, other.continuous_offset_y_m_);
    swap(control_i_min_, other.control_i_min_);
    swap(control_j_min_, other.control_j_min_);
    swap(configured_, other.configured_);
    swap(initialized_, other.initialized_);
    swap(current_controls_, other.current_controls_);
    swap(previous_controls_, other.previous_controls_);
    swap(i_stencils_, other.i_stencils_);
    swap(j_stencils_, other.j_stencils_);
    swap(native_backend_, other.native_backend_);
    swap(native_evolution_required_, other.native_evolution_required_);
}

void EarthCorrelatedNoise::rebuild_stencils()
{
    auto build = [](std::vector<AxisStencil>& stencils,
                    int count,
                    std::int64_t earth_origin,
                    double spacing,
                    double control_spacing,
                    double control_phase,
                    std::int64_t control_min,
                    double continuous_offset_m) {
        stencils.resize(static_cast<std::size_t>(count));
        for (int index = 0; index < count; ++index)
        {
            const double coordinate =
                ((static_cast<double>(earth_origin + index) + 0.5)
                 * spacing + continuous_offset_m) /
                    control_spacing + control_phase;
            const double coordinate_floor = std::floor(coordinate);
            const auto base = static_cast<std::int64_t>(coordinate_floor);
            const auto raw = cubic_weights(coordinate - coordinate_floor);
            double norm_sq = 0.0;
            for (double weight : raw) norm_sq += weight * weight;
            const double inverse_norm = 1.0 / std::sqrt(norm_sq);
            AxisStencil& stencil = stencils[static_cast<std::size_t>(index)];
            for (int point = 0; point < 4; ++point)
            {
                stencil.control_indices[static_cast<std::size_t>(point)] =
                    static_cast<int>(base - 1 + point - control_min);
                stencil.normalized_weights[static_cast<std::size_t>(point)] =
                    raw[static_cast<std::size_t>(point)] * inverse_norm;
            }
        }
    };
    build(i_stencils_, nr_, earth_origin_i_cells_, dx_m_,
          control_spacing_x_m_, control_phase_x_, control_i_min_,
          continuous_offset_x_m_);
    build(j_stencils_, nth_, earth_origin_j_cells_, dy_m_,
          control_spacing_y_m_, control_phase_y_, control_j_min_,
          continuous_offset_y_m_);
}

void EarthCorrelatedNoise::render(
    const Field3D& controls,
    Field3D& output) const
{
    if (!initialized_)
    {
        output = Field3D();
        return;
    }
    if (output.size_r() != nr_ || output.size_th() != nth_
        || output.size_z() != nz_)
        output.resize(nr_, nth_, nz_);

    render_preallocated(controls, output);
}

void EarthCorrelatedNoise::render_preallocated(
    const Field3D& controls,
    Field3D& output) const noexcept
{
    assert(initialized_);
    assert(output.size_r() == nr_ && output.size_th() == nth_ &&
           output.size_z() == nz_);

#pragma omp parallel for collapse(2) schedule(static) \
    if(static_cast<long long>(nr_) * nth_ * nz_ >= 32768)
    for (int i = 0; i < nr_; ++i)
    {
        for (int j = 0; j < nth_; ++j)
        {
            const AxisStencil& si = i_stencils_[static_cast<std::size_t>(i)];
            const AxisStencil& sj = j_stencils_[static_cast<std::size_t>(j)];
            for (int k = 0; k < nz_; ++k)
            {
                double value = 0.0;
                for (int ai = 0; ai < 4; ++ai)
                    for (int aj = 0; aj < 4; ++aj)
                    {
                        const int ci = si.control_indices[
                            static_cast<std::size_t>(ai)];
                        const int cj = sj.control_indices[
                            static_cast<std::size_t>(aj)];
                        value += si.normalized_weights[static_cast<std::size_t>(ai)]
                            * sj.normalized_weights[static_cast<std::size_t>(aj)]
                            * static_cast<double>(controls(ci, cj, k));
                    }
                output(i, j, k) = static_cast<float>(value);
            }
        }
    }
}

bool EarthCorrelatedNoise::render_matches(
    const Field3D& controls,
    const Field3D& field) const noexcept
{
    if (!initialized_)
        return field.empty();
    if (field.size_r() != nr_ || field.size_th() != nth_ ||
        field.size_z() != nz_)
        return false;
    int mismatch = 0;
#pragma omp parallel for collapse(2) schedule(static) reduction(|:mismatch) \
    if(static_cast<long long>(nr_) * nth_ * nz_ >= 32768)
    for (int i = 0; i < nr_; ++i)
        for (int j = 0; j < nth_; ++j)
        {
            const AxisStencil& si = i_stencils_[static_cast<std::size_t>(i)];
            const AxisStencil& sj = j_stencils_[static_cast<std::size_t>(j)];
            for (int k = 0; k < nz_; ++k)
            {
                double value = 0.0;
                for (int ai = 0; ai < 4; ++ai)
                    for (int aj = 0; aj < 4; ++aj)
                    {
                        const int ci = si.control_indices[
                            static_cast<std::size_t>(ai)];
                        const int cj = sj.control_indices[
                            static_cast<std::size_t>(aj)];
                        value += si.normalized_weights[
                                     static_cast<std::size_t>(ai)] *
                            sj.normalized_weights[
                                static_cast<std::size_t>(aj)] *
                            static_cast<double>(controls(ci, cj, k));
                    }
                const float expected = static_cast<float>(value);
                const float actual = field(i, j, k);
                if (std::memcmp(&expected, &actual, sizeof(float)) != 0)
                    mismatch = 1;
            }
        }
    return mismatch == 0;
}

void EarthCorrelatedNoise::render_current(Field3D& output) const
{
    render(current_controls_, output);
}

void EarthCorrelatedNoise::render_previous(Field3D& output) const
{
    render(previous_controls_, output);
}

void EarthCorrelatedNoise::render_current_preallocated(
    Field3D& output) const noexcept
{
    render_preallocated(current_controls_, output);
}

void EarthCorrelatedNoise::render_previous_preallocated(
    Field3D& output) const noexcept
{
    render_preallocated(previous_controls_, output);
}

bool EarthCorrelatedNoise::current_render_matches(
    const Field3D& field) const noexcept
{
    return render_matches(current_controls_, field);
}

bool EarthCorrelatedNoise::previous_render_matches(
    const Field3D& field) const noexcept
{
    return render_matches(previous_controls_, field);
}

bool EarthCorrelatedNoise::restore(
    const Field3D& current_controls,
    const Field3D& previous_controls,
    int nr,
    int nth,
    int nz,
    std::int64_t earth_origin_i_cells,
    std::int64_t earth_origin_j_cells,
    std::int64_t control_i_min,
    std::int64_t control_j_min,
    double continuous_offset_x_m,
    double continuous_offset_y_m,
    std::string& error)
{
    if (!configured_ || nr <= 0 || nth <= 0 || nz <= 0
        || current_controls.empty()
        || current_controls.size_r() != previous_controls.size_r()
        || current_controls.size_th() != previous_controls.size_th()
        || current_controls.size_z() != previous_controls.size_z()
        || current_controls.size_z() != nz
        || !finite_field(current_controls) || !finite_field(previous_controls)
        || !std::isfinite(continuous_offset_x_m)
        || !std::isfinite(continuous_offset_y_m)
        || continuous_offset_x_m < 0.0 || continuous_offset_x_m >= dx_m_
        || continuous_offset_y_m < 0.0 || continuous_offset_y_m >= dy_m_)
    {
        error = "invalid Earth-correlated chaos control state";
        return false;
    }
    const ControlBounds expected = required_control_bounds(
        earth_origin_i_cells, earth_origin_j_cells, nr, nth,
        continuous_offset_x_m, continuous_offset_y_m);
    if (expected.i_min != control_i_min || expected.j_min != control_j_min
        || expected.i_max - expected.i_min + 1 != current_controls.size_r()
        || expected.j_max - expected.j_min + 1 != current_controls.size_th())
    {
        error = "Earth-correlated chaos control bounds do not match the window";
        return false;
    }
    EarthCorrelatedNoise candidate(*this);
    candidate.nr_ = nr;
    candidate.nth_ = nth;
    candidate.nz_ = nz;
    candidate.earth_origin_i_cells_ = earth_origin_i_cells;
    candidate.earth_origin_j_cells_ = earth_origin_j_cells;
    candidate.continuous_offset_x_m_ = continuous_offset_x_m;
    candidate.continuous_offset_y_m_ = continuous_offset_y_m;
    candidate.control_i_min_ = control_i_min;
    candidate.control_j_min_ = control_j_min;
    candidate.current_controls_ = current_controls;
    candidate.previous_controls_ = previous_controls;
    candidate.initialized_ = true;
    candidate.rebuild_stencils();
    swap_window_state(candidate);
    error.clear();
    return true;
}

bool EarthCorrelatedNoise::restore_inferred_control_bounds(
    const Field3D& current_controls,
    const Field3D& previous_controls,
    int nr,
    int nth,
    int nz,
    std::int64_t earth_origin_i_cells,
    std::int64_t earth_origin_j_cells,
    std::string& error)
{
    try
    {
        const ControlBounds expected = required_control_bounds(
            earth_origin_i_cells, earth_origin_j_cells, nr, nth);
        return restore(
            current_controls, previous_controls, nr, nth, nz,
            earth_origin_i_cells, earth_origin_j_cells,
            expected.i_min, expected.j_min, error);
    }
    catch (const std::exception& exception)
    {
        error = exception.what();
        return false;
    }
}

bool EarthCorrelatedNoise::restore_uninitialized_origin(
    std::int64_t earth_origin_i_cells,
    std::int64_t earth_origin_j_cells,
    std::string& error)
{
    if (!configured_ || initialized_)
    {
        error = "invalid uninitialized Earth-correlated chaos restore";
        return false;
    }
    earth_origin_i_cells_ = earth_origin_i_cells;
    earth_origin_j_cells_ = earth_origin_j_cells;
    error.clear();
    return true;
}

const char* EarthCorrelatedNoise::algorithm_id() noexcept
{
    return "earth_control_lattice_cubic_ar1_v1";
}

} // namespace tmv::chaos
