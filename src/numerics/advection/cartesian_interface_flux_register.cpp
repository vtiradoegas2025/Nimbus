#include "numerics/advection/cartesian_interface_flux_register.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tmv
{

void CartesianDryMassInterfaceFluxRegister::configure(
    int nr, int nth, int nz, CartesianInterfaceFootprint footprint,
    int x_face_tangential_ratio, int y_face_tangential_ratio)
{
    if (nr < 3 || nth < 3 || nz < 1 || footprint.i_begin < 1 ||
        footprint.i_end >= nr || footprint.j_begin < 1 ||
        footprint.j_end >= nth || footprint.k_begin < 0 ||
        footprint.k_end > nz || footprint.i_begin >= footprint.i_end ||
        footprint.j_begin >= footprint.j_end ||
        footprint.k_begin >= footprint.k_end ||
        x_face_tangential_ratio <= 0 || y_face_tangential_ratio <= 0 ||
        (footprint.j_end - footprint.j_begin) %
                x_face_tangential_ratio != 0 ||
        (footprint.i_end - footprint.i_begin) %
                y_face_tangential_ratio != 0)
        throw std::invalid_argument(
            "Cartesian dry-mass flux register requires a non-empty interior footprint");

    CartesianDryMassInterfaceFluxRegister candidate;
    candidate.nr_ = nr;
    candidate.nth_ = nth;
    candidate.nz_ = nz;
    candidate.x_ratio_ = x_face_tangential_ratio;
    candidate.y_ratio_ = y_face_tangential_ratio;
    candidate.footprint_ = footprint;
    const std::size_t levels = static_cast<std::size_t>(
        footprint.k_end - footprint.k_begin);
    const std::size_t x_count = static_cast<std::size_t>(
        (footprint.j_end - footprint.j_begin) / x_face_tangential_ratio) *
        levels;
    const std::size_t y_count = static_cast<std::size_t>(
        (footprint.i_end - footprint.i_begin) / y_face_tangential_ratio) *
        levels;
    candidate.fluxes_.west.assign(x_count, 0.0);
    candidate.fluxes_.east.assign(x_count, 0.0);
    candidate.fluxes_.south.assign(y_count, 0.0);
    candidate.fluxes_.north.assign(y_count, 0.0);
    candidate.configured_ = true;
    *this = std::move(candidate);
}

void CartesianDryMassInterfaceFluxRegister::reset_interval() noexcept
{
    std::fill(fluxes_.west.begin(), fluxes_.west.end(), 0.0);
    std::fill(fluxes_.east.begin(), fluxes_.east.end(), 0.0);
    std::fill(fluxes_.south.begin(), fluxes_.south.end(), 0.0);
    std::fill(fluxes_.north.begin(), fluxes_.north.end(), 0.0);
    fluxes_.accumulated_dt_s = 0.0;
    fluxes_.contribution_count = 0;
}

void CartesianDryMassInterfaceFluxRegister::accumulate(
    const float* mass_flux_x, const float* mass_flux_y,
    double dt_s, double dx_m, double dy_m, double dz_m)
{
    if (!configured_ || mass_flux_x == nullptr || mass_flux_y == nullptr ||
        !std::isfinite(dt_s) || !std::isfinite(dx_m) ||
        !std::isfinite(dy_m) || !std::isfinite(dz_m) || dt_s <= 0.0 ||
        dx_m <= 0.0 || dy_m <= 0.0 || dz_m <= 0.0)
        throw std::invalid_argument(
            "Cartesian dry-mass flux accumulation requires finite fluxes and metrics");

    std::vector<double> west(fluxes_.west.size(), 0.0);
    std::vector<double> east(fluxes_.east.size(), 0.0);
    std::vector<double> south(fluxes_.south.size(), 0.0);
    std::vector<double> north(fluxes_.north.size(), 0.0);
    const double x_scale = dt_s * dy_m * dz_m;
    const double y_scale = dt_s * dx_m * dz_m;
    const int west_face = footprint_.i_begin - 1;
    const int east_face = footprint_.i_end - 1;
    const int south_face = footprint_.j_begin - 1;
    const int north_face = footprint_.j_end - 1;
    const int levels = footprint_.k_end - footprint_.k_begin;
    const auto index = [this](int i, int j, int k) {
        return (static_cast<std::size_t>(i) *
                    static_cast<std::size_t>(nth_) +
                static_cast<std::size_t>(j)) *
                   static_cast<std::size_t>(nz_) +
               static_cast<std::size_t>(k);
    };
    for (int j = footprint_.j_begin; j < footprint_.j_end; ++j)
        for (int k = footprint_.k_begin; k < footprint_.k_end; ++k)
        {
            const std::size_t bin = static_cast<std::size_t>(
                (j - footprint_.j_begin) / x_ratio_ * levels +
                k - footprint_.k_begin);
            west[bin] += x_scale * mass_flux_x[index(west_face, j, k)];
            east[bin] += x_scale * mass_flux_x[index(east_face, j, k)];
        }
    for (int i = footprint_.i_begin; i < footprint_.i_end; ++i)
        for (int k = footprint_.k_begin; k < footprint_.k_end; ++k)
        {
            const std::size_t bin = static_cast<std::size_t>(
                (i - footprint_.i_begin) / y_ratio_ * levels +
                k - footprint_.k_begin);
            south[bin] += y_scale * mass_flux_y[index(i, south_face, k)];
            north[bin] += y_scale * mass_flux_y[index(i, north_face, k)];
        }

    const auto validate = [](const std::vector<double>& committed,
                             const std::vector<double>& staged) {
        for (std::size_t bin = 0; bin < committed.size(); ++bin)
            if (!std::isfinite(staged[bin]) ||
                !std::isfinite(committed[bin] + staged[bin]))
                throw std::overflow_error(
                    "Cartesian dry-mass interface flux overflowed");
    };
    validate(fluxes_.west, west);
    validate(fluxes_.east, east);
    validate(fluxes_.south, south);
    validate(fluxes_.north, north);
    if (!std::isfinite(fluxes_.accumulated_dt_s + dt_s))
        throw std::overflow_error(
            "Cartesian dry-mass interface duration overflowed");
    if (fluxes_.contribution_count ==
        std::numeric_limits<std::uint64_t>::max())
        throw std::overflow_error(
            "Cartesian dry-mass interface contribution count overflowed");

    for (std::size_t bin = 0; bin < west.size(); ++bin)
    {
        fluxes_.west[bin] += west[bin];
        fluxes_.east[bin] += east[bin];
    }
    for (std::size_t bin = 0; bin < south.size(); ++bin)
    {
        fluxes_.south[bin] += south[bin];
        fluxes_.north[bin] += north[bin];
    }
    fluxes_.accumulated_dt_s += dt_s;
    ++fluxes_.contribution_count;
}

} // namespace tmv
