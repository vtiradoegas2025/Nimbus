/**
 * @file trajectories.cpp
 * @brief Online Lagrangian parcel-trajectory integration (see trajectories.hpp).
 */

#include "diagnostics/trajectories.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <type_traits>

#include "core/infra/grid_coordinates.hpp"
#include "diagnostics/physical_derivative_kernel.hpp"
#include "diagnostics/vorticity_budget_kernel.hpp" // shared Cartesian C-grid budget
#include "terrain/conservative_transport.hpp"

namespace tmv
{
namespace
{

static_assert(std::is_standard_layout<TrajectorySample>::value);
static_assert(sizeof(TrajectorySample) ==
              sizeof(float) * static_cast<std::size_t>(kTrajectoryVarCount));

bool write_trajectory_npy_header(std::ofstream& out,
                                 std::size_t parcels,
                                 std::size_t samples)
{
    const std::string dict =
        "{'descr': '<f4', 'fortran_order': False, 'shape': (" +
        std::to_string(parcels) + ", " + std::to_string(samples) + ", " +
        std::to_string(kTrajectoryVarCount) + "), }";
    constexpr std::size_t preamble_size = 10;
    const std::size_t dict_plus_newline = dict.size() + 1;
    const std::size_t padding =
        (16 - ((preamble_size + dict_plus_newline) % 16)) % 16;
    const std::size_t header_size = dict_plus_newline + padding;
    if (header_size > std::numeric_limits<std::uint16_t>::max())
        return false;

    out.write("\x93NUMPY", 6);
    out.put(static_cast<char>(1));
    out.put(static_cast<char>(0));
    const auto encoded_size = static_cast<std::uint16_t>(header_size);
    out.put(static_cast<char>(encoded_size & 0xffU));
    out.put(static_cast<char>((encoded_size >> 8U) & 0xffU));
    out.write(dict.data(), static_cast<std::streamsize>(dict.size()));
    for (std::size_t index = 0; index < padding; ++index)
        out.put(' ');
    out.put('\n');
    return out.good();
}

/**
 * @brief Trilinear sample of `f` at fractional grid indices (fi, fj, fk).
 *
 * The caller supplies indices already offset for the field's staggering (e.g.
 * fi - 0.5 for an x-face velocity). i and k clamp at the domain edge; j wraps
 * when `j_periodic` (periodic y), else clamps.
 */
double trilerp(const Field3D& f, int nr, int nth, int nz, double fi, double fj, double fk, bool j_periodic)
{
    if (nr <= 0 || nth <= 0 || nz <= 0)
        return 0.0;

    // i — clamp.
    fi     = std::clamp(fi, 0.0, static_cast<double>(nr - 1));
    int i0 = static_cast<int>(std::floor(fi));
    if (i0 > nr - 2)
        i0 = std::max(0, nr - 2);
    const int    i1 = std::min(i0 + 1, nr - 1);
    const double ti = fi - i0;

    // k — clamp.
    fk     = std::clamp(fk, 0.0, static_cast<double>(nz - 1));
    int k0 = static_cast<int>(std::floor(fk));
    if (k0 > nz - 2)
        k0 = std::max(0, nz - 2);
    const int    k1 = std::min(k0 + 1, nz - 1);
    const double tk = fk - k0;

    // j — periodic (periodic-y lateral BC) or clamp.
    int    j0, j1;
    double tj;
    if (j_periodic && nth > 1)
    {
        double fjw = std::fmod(fj, static_cast<double>(nth));
        if (fjw < 0.0)
            fjw += nth;
        const double base = std::floor(fjw);
        j0                = static_cast<int>(base) % nth;
        j1                = (j0 + 1) % nth;
        tj                = fjw - base;
    }
    else
    {
        fj = std::clamp(fj, 0.0, static_cast<double>(nth - 1));
        j0 = static_cast<int>(std::floor(fj));
        if (j0 > nth - 2)
            j0 = std::max(0, nth - 2);
        j1 = std::min(j0 + 1, nth - 1);
        tj = fj - j0;
    }

    const auto V = [&](int i, int j, int k) -> double { return static_cast<double>(f[i][j][k]); };

    const double c00 = V(i0, j0, k0) * (1.0 - ti) + V(i1, j0, k0) * ti;
    const double c10 = V(i0, j1, k0) * (1.0 - ti) + V(i1, j1, k0) * ti;
    const double c01 = V(i0, j0, k1) * (1.0 - ti) + V(i1, j0, k1) * ti;
    const double c11 = V(i0, j1, k1) * (1.0 - ti) + V(i1, j1, k1) * ti;

    const double c0 = c00 * (1.0 - tj) + c10 * tj;
    const double c1 = c01 * (1.0 - tj) + c11 * tj;
    return c0 * (1.0 - tk) + c1 * tk;
}

} // namespace

void TrajectorySet::configure(const TrajectoryConfig&            cfg,
                              CoordinateSystem                   coord,
                              bool                               staggered,
                              int                                nr,
                              int                                nth,
                              int                                nz,
                              double                             dr,
                              double                             dtheta,
                              double                             dz,
                              tmv::domain::HorizontalDomainFrame frame,
                              const TerrainMetrics3D*            terrain_metrics,
                              bool                               earth_fixed_terrain)
{
    if (cfg.enabled)
    {
        if (cfg.nx <= 0 || cfg.ny <= 0 || cfg.nz <= 0)
            throw std::invalid_argument(
                "trajectory seed counts must be positive");
        const std::size_t nx = static_cast<std::size_t>(cfg.nx);
        const std::size_t ny = static_cast<std::size_t>(cfg.ny);
        const std::size_t nz = static_cast<std::size_t>(cfg.nz);
        if (nx > kMaximumTrajectoryParcels / ny ||
            nx * ny > kMaximumTrajectoryParcels / nz)
            throw std::invalid_argument(
                "trajectory seed lattice exceeds the supported parcel limit");
        if (!std::isfinite(cfg.sample_interval_s) ||
            cfg.sample_interval_s < 0.0)
            throw std::invalid_argument(
                "trajectory sample interval must be finite and non-negative");
        if (cfg.sample_interval_s > 0.0 &&
            !(cfg.seed_time_s + cfg.sample_interval_s > cfg.seed_time_s))
            throw std::invalid_argument(
                "trajectory sample interval is too small to advance the cadence clock");
        if (cfg.max_samples_per_parcel <= 0)
            throw std::invalid_argument(
                "trajectory retained-sample bound must be positive");
        const std::size_t parcels = nx * ny * nz;
        if (static_cast<std::size_t>(cfg.max_samples_per_parcel) >
            kMaximumTrajectorySamples / parcels)
            throw std::invalid_argument(
                "trajectory retained-sample capacity exceeds the supported total limit");
    }
    cfg_       = cfg;
    coord_     = coord;
    staggered_ = staggered;
    nr_        = nr;
    nth_       = nth;
    nz_        = nz;
    domain_    = physical_scalar_domain(nr, nth, nz, staggered ? StaggerType::CGrid : StaggerType::Collocated);
    dr_        = (dr > 0.0) ? dr : 1.0;
    dtheta_    = (dtheta > 0.0) ? dtheta : 1.0;
    dz_        = (dz > 0.0) ? dz : 1.0;
    terrain_metrics_ = terrain_metrics;
    surface_velocity_ms_ = earth_fixed_terrain
        ? domain::HorizontalVelocity{} : frame.velocity_ms;
    if (terrain_metrics_ != nullptr)
    {
        if (coord_ != CoordinateSystem::Cartesian || !staggered_)
            throw std::invalid_argument(
                "terrain trajectories require the Cartesian C-grid");
        if (nr_ < 3 || nth_ < 3 || nz_ < 3)
            throw std::invalid_argument(
                "terrain trajectories require at least three cells on every axis");
        diagnostics::validate_physical_derivative_geometry(
            nr_, nth_, nz_, dr_, dtheta_, dz_, terrain_metrics_);
        // The lower-boundary physical velocity uses the complete conservative
        // metric set. Validate it once at configuration instead of discovering
        // a missing face metric during a near-ground parcel step.
        terrain_transport::validate_geometry(*terrain_metrics_);
        const auto spacing_matches = [](double metric, double configured)
        {
            return std::abs(metric - configured)
                <= 64.0 * std::numeric_limits<double>::epsilon()
                    * std::max({1.0, std::abs(metric), std::abs(configured)});
        };
        if (!spacing_matches(terrain_metrics_->dx, dr_)
            || !spacing_matches(terrain_metrics_->dy, dtheta_)
            || !spacing_matches(terrain_metrics_->dzeta, dz_))
        {
            throw std::invalid_argument(
                "terrain trajectory geometry spacing does not match the model grid");
        }
        for (int i = 0; i < nr_; ++i)
            for (int j = 0; j < nth_; ++j)
                for (int k = 0; k < nz_; ++k)
                {
                    const double lower = terrain_metrics_->z_face(i, j, k);
                    const double upper = terrain_metrics_->z_face(i, j, k + 1);
                    if (!std::isfinite(lower) || !std::isfinite(upper)
                        || !(upper > lower))
                    {
                        throw std::invalid_argument(
                            "terrain trajectories require finite increasing interface heights");
                    }
                }
    }
    frame.validate();
    frame_ = frame;
    prepared_dt_ = 0.0;
    midpoints_.clear();
}

bool TrajectorySet::restore_checkpoint_state(TrajectoryCheckpointState state,
                                             double checkpoint_time_s,
                                             std::string& error)
{
    error.clear();
    const auto same_floating = [](double left, double right)
    {
        return left == right || (std::isnan(left) && std::isnan(right));
    };
    const auto& saved = state.configuration;
    if (!cfg_.enabled || !saved.enabled || cfg_.nx != saved.nx ||
        cfg_.ny != saved.ny || cfg_.nz != saved.nz ||
        !same_floating(cfg_.seed_time_s, saved.seed_time_s) ||
        !same_floating(cfg_.sample_interval_s, saved.sample_interval_s) ||
        cfg_.max_samples_per_parcel != saved.max_samples_per_parcel ||
        !same_floating(cfg_.half_x_m, saved.half_x_m) ||
        !same_floating(cfg_.half_y_m, saved.half_y_m) ||
        !same_floating(cfg_.half_z_m, saved.half_z_m) ||
        !same_floating(cfg_.center_x_m, saved.center_x_m) ||
        !same_floating(cfg_.center_y_m, saved.center_y_m) ||
        !same_floating(cfg_.center_z_m, saved.center_z_m))
    {
        error = "trajectory checkpoint configuration does not exactly match the requested branch";
        return false;
    }
    if (!std::isfinite(checkpoint_time_s) || checkpoint_time_s < 0.0)
    {
        error = "trajectory checkpoint has an invalid simulation time";
        return false;
    }
    if ((!state.seeded && !std::isnan(state.next_sample_time_s)) ||
        (state.seeded && cfg_.sample_interval_s == 0.0 &&
         !std::isnan(state.next_sample_time_s)) ||
        (state.seeded && cfg_.sample_interval_s > 0.0 &&
         (!std::isfinite(state.next_sample_time_s) ||
          state.next_sample_time_s <= checkpoint_time_s)))
    {
        error = "trajectory checkpoint has an invalid sampling cadence state";
        return false;
    }
    if (!state.seeded && !state.parcels.empty())
    {
        error = "unseeded trajectory checkpoint contains parcels";
        return false;
    }
    for (std::size_t index = 0; index < state.parcels.size(); ++index)
    {
        const Parcel& parcel = state.parcels[index];
        if (parcel.id != static_cast<int>(index) || !std::isfinite(parcel.x) ||
            !std::isfinite(parcel.y) || !std::isfinite(parcel.z))
        {
            error = "trajectory checkpoint contains an invalid parcel";
            return false;
        }
        if (parcel.active && !in_horizontal_domain(parcel.x, parcel.y))
        {
            error = "trajectory checkpoint contains an active parcel outside the restored domain";
            return false;
        }
        if (parcel.active &&
            (parcel.z < local_z_lower(parcel.x, parcel.y) ||
             parcel.z > local_z_upper(parcel.x, parcel.y)))
        {
            error = "trajectory checkpoint contains an active parcel outside the restored vertical domain";
            return false;
        }
        float previous_time = -std::numeric_limits<float>::infinity();
        for (const TrajectorySample& sample : parcel.path)
        {
            if (!std::isfinite(sample.t) || !std::isfinite(sample.x) ||
                !std::isfinite(sample.y) || !std::isfinite(sample.z) ||
                sample.t < previous_time ||
                static_cast<double>(sample.t) > checkpoint_time_s + 1.0e-3)
            {
                error = "trajectory checkpoint contains an invalid or future path sample";
                return false;
            }
            previous_time = sample.t;
        }
        if (state.seeded && parcel.path.empty())
        {
            error = "seeded trajectory checkpoint contains a parcel without provenance";
            return false;
        }
        if (parcel.path.size() >
            static_cast<std::size_t>(cfg_.max_samples_per_parcel))
        {
            error = "trajectory checkpoint exceeds its retained-sample bound";
            return false;
        }
    }

    seeded_ = state.seeded;
    next_sample_time_s_ = state.next_sample_time_s;
    parcels_ = std::move(state.parcels);
    prepared_dt_ = 0.0;
    midpoints_.clear();
    return true;
}

void TrajectorySet::retain_sample(Parcel& parcel,
                                  const TrajectorySample& sample)
{
    parcel.path.push_back(sample);
    while (parcel.path.size() >
           static_cast<std::size_t>(cfg_.max_samples_per_parcel))
        parcel.path.pop_front();
}

bool TrajectorySet::in_horizontal_domain(double x, double y) const
{
    return !domain_.empty() && x >= x_lower() && x <= x_upper() && y >= y_lower() && y <= y_upper();
}

double TrajectorySet::terrain_height_at(double x, double y, int level,
                                        bool interface) const
{
    if (terrain_metrics_ == nullptr)
        return interface
            ? static_cast<double>(level) * dz_
            : (static_cast<double>(level) + 0.5) * dz_;
    const int level_count = interface ? nz_ + 1 : nz_;
    if (level < 0 || level >= level_count)
        throw std::out_of_range("terrain trajectory height level is outside the grid");

    const auto axis = [](double fractional, int count,
                         int& lower, int& upper, double& weight)
    {
        if (count <= 1)
        {
            lower = upper = 0;
            weight = 0.0;
            return;
        }
        fractional = std::clamp(fractional, 0.0,
                                static_cast<double>(count - 1));
        lower = static_cast<int>(std::floor(fractional));
        if (lower >= count - 1)
            lower = count - 2;
        upper = lower + 1;
        weight = fractional - static_cast<double>(lower);
    };
    int i0 = 0, i1 = 0, j0 = 0, j1 = 0;
    double tx = 0.0, ty = 0.0;
    axis(grid_coordinates::scalar_fractional_index(x, dr_), nr_, i0, i1, tx);
    axis(grid_coordinates::scalar_fractional_index(y, dtheta_), nth_, j0, j1, ty);
    const auto value = [&](int i, int j)
    {
        return interface
            ? terrain_metrics_->z_face(i, j, level)
            : terrain_metrics_->z(i, j, level);
    };
    const double south = value(i0, j0) * (1.0 - tx) + value(i1, j0) * tx;
    const double north = value(i0, j1) * (1.0 - tx) + value(i1, j1) * tx;
    return south * (1.0 - ty) + north * ty;
}

double TrajectorySet::scalar_vertical_fractional_index(
    double x, double y, double z) const
{
    if (terrain_metrics_ == nullptr)
        return grid_coordinates::scalar_fractional_index(z, dz_);
    if (nz_ <= 1)
        return 0.0;
    const double first = terrain_height_at(x, y, 0, false);
    const double last = terrain_height_at(x, y, nz_ - 1, false);
    if (z <= first) return 0.0;
    if (z >= last) return static_cast<double>(nz_ - 1);
    int lower = 0;
    int upper = nz_ - 1;
    while (upper - lower > 1)
    {
        const int middle = lower + (upper - lower) / 2;
        if (terrain_height_at(x, y, middle, false) <= z)
            lower = middle;
        else
            upper = middle;
    }
    const double z0 = terrain_height_at(x, y, lower, false);
    const double z1 = terrain_height_at(x, y, upper, false);
    return static_cast<double>(lower) + (z - z0) / (z1 - z0);
}

double TrajectorySet::upper_face_vertical_fractional_index(
    double x, double y, double z) const
{
    if (terrain_metrics_ == nullptr)
        return grid_coordinates::upper_face_fractional_index(z, dz_);
    if (nz_ <= 1)
        return 0.0;
    // Stored C-grid w[k] owns physical interface k+1.
    const double first = terrain_height_at(x, y, 1, true);
    const double last = terrain_height_at(x, y, nz_, true);
    if (z <= first) return 0.0;
    if (z >= last) return static_cast<double>(nz_ - 1);
    int lower = 0;
    int upper = nz_ - 1;
    while (upper - lower > 1)
    {
        const int middle = lower + (upper - lower) / 2;
        if (terrain_height_at(x, y, middle + 1, true) <= z)
            lower = middle;
        else
            upper = middle;
    }
    const double z0 = terrain_height_at(x, y, lower + 1, true);
    const double z1 = terrain_height_at(x, y, upper + 1, true);
    return static_cast<double>(lower) + (z - z0) / (z1 - z0);
}

double TrajectorySet::terrain_ground_vertical_velocity_at(
    double x, double y, const Field3D& u, const Field3D& v) const
{
    if (terrain_metrics_ == nullptr)
        return 0.0;
    const double fi = std::clamp(
        grid_coordinates::scalar_fractional_index(x, dr_),
        0.0, static_cast<double>(nr_ - 1));
    const double fj = std::clamp(
        grid_coordinates::scalar_fractional_index(y, dtheta_),
        0.0, static_cast<double>(nth_ - 1));
    const int i0_raw = std::min(static_cast<int>(std::floor(fi)), nr_ - 2);
    const int j0_raw = std::min(static_cast<int>(std::floor(fj)), nth_ - 2);
    const int i1_raw = i0_raw + 1;
    const int j1_raw = j0_raw + 1;
    const double tx = fi - static_cast<double>(i0_raw);
    const double ty = fj - static_cast<double>(j0_raw);
    const auto sample = [&](int i, int j)
    {
        i = std::clamp(i, 1, nr_ - 1);
        j = std::clamp(j, 1, nth_ - 1);
        return terrain_transport::cgrid_ground_vertical_velocity(
            *terrain_metrics_, u, v, i, j,
            surface_velocity_ms_.u_ms, surface_velocity_ms_.v_ms);
    };
    const double south = sample(i0_raw, j0_raw) * (1.0 - tx)
                       + sample(i1_raw, j0_raw) * tx;
    const double north = sample(i0_raw, j1_raw) * (1.0 - tx)
                       + sample(i1_raw, j1_raw) * tx;
    return south * (1.0 - ty) + north * ty;
}

double TrajectorySet::local_z_lower(double x, double y) const
{
    return terrain_metrics_ == nullptr
        ? z_lower()
        : terrain_height_at(x, y, domain_.k_begin, true);
}

double TrajectorySet::local_z_upper(double x, double y) const
{
    return terrain_metrics_ == nullptr
        ? z_upper()
        : terrain_height_at(x, y, domain_.k_end, true);
}

double TrajectorySet::clamp_vertical(double x, double y, double z) const
{
    return std::clamp(z, local_z_lower(x, y), local_z_upper(x, y));
}

void TrajectorySet::velocity_at(double         x,
                                double         y,
                                double         z,
                                const Field3D& u,
                                const Field3D& v,
                                const Field3D& w,
                                double&        vx,
                                double&        vy,
                                double&        vz) const
{
    const double sx = grid_coordinates::scalar_fractional_index(x, dr_);
    const double sy = grid_coordinates::scalar_fractional_index(y, dtheta_);
    const double sz = scalar_vertical_fractional_index(x, y, z);
    if (!staggered_)
    {
        vx = trilerp(u, nr_, nth_, nz_, sx, sy, sz, false) - frame_.velocity_ms.u_ms;
        vy = trilerp(v, nr_, nth_, nz_, sx, sy, sz, false) - frame_.velocity_ms.v_ms;
        vz = trilerp(w, nr_, nth_, nz_, sx, sy, sz, false);
        return;
    }

    // C-grid component index i/j/k denotes the upper face of scalar cell
    // i/j/k. Cross-component axes remain scalar-centred.
    const double ux       = grid_coordinates::upper_face_fractional_index(x, dr_);
    const double vy_index = grid_coordinates::upper_face_fractional_index(y, dtheta_);
    const double wz       = upper_face_vertical_fractional_index(x, y, z);
    vx                    = trilerp(u, nr_, nth_, nz_, ux, sy, sz, false) - frame_.velocity_ms.u_ms;
    vy                    = trilerp(v, nr_, nth_, nz_, sx, vy_index, sz, false) - frame_.velocity_ms.v_ms;
    if (terrain_metrics_ != nullptr
        && z < terrain_height_at(x, y, 1, true))
    {
        const double ground_z = terrain_height_at(x, y, 0, true);
        const double upper_z = terrain_height_at(x, y, 1, true);
        const double weight = std::clamp(
            (z - ground_z) / (upper_z - ground_z), 0.0, 1.0);
        const double ground_w =
            terrain_ground_vertical_velocity_at(x, y, u, v);
        const double upper_w = trilerp(
            w, nr_, nth_, nz_, sx, sy, 0.0, false);
        vz = ground_w * (1.0 - weight) + upper_w * weight;
    }
    else
    {
        vz = trilerp(w, nr_, nth_, nz_, sx, sy, wz, false);
    }
}

float TrajectorySet::scalar_at(double x, double y, double z, const Field3D& f) const
{
    return static_cast<float>(trilerp(f,
                                      nr_,
                                      nth_,
                                      nz_,
                                      grid_coordinates::scalar_fractional_index(x, dr_),
                                      grid_coordinates::scalar_fractional_index(y, dtheta_),
                                      scalar_vertical_fractional_index(x, y, z),
                                      false));
}

void TrajectorySet::sample_fields(double            x,
                                  double            y,
                                  double            z,
                                  const Field3D&    u,
                                  const Field3D&    v,
                                  const Field3D&    w,
                                  const Field3D&    rho,
                                  const Field3D&    pres,
                                  const Field3D&    vort_z,
                                  TrajectorySample& s) const
{
    double sampled_u = 0.0, sampled_v = 0.0, sampled_w = 0.0;
    velocity_at(x, y, z, u, v, w, sampled_u, sampled_v, sampled_w);
    s.w = static_cast<float>(sampled_w);

    const float nanf = std::numeric_limits<float>::quiet_NaN();

    // Online vorticity budget — Cartesian C-grid or collocated A-grid. The
    // shared kernel needs the i+/-1, j+/-1, k+/-1 neighbours. Evaluate it at the
    // surrounding interior scalar centres, then interpolate the pointwise
    // budget to the parcel. Computing it here from fields that update every
    // step keeps the budget current rather than frozen at the last output frame
    // (the gridded vort_z / *_term diagnostics run only at output cadence).
    // zeta comes from the same interpolation so the path's vorticity remains
    // consistent with its budget.
    const bool cartesian_online = coord_ == CoordinateSystem::Cartesian;
    if (cartesian_online && nr_ >= 3 && nth_ >= 3 && nz_ >= 3)
    {
        struct InteriorBracket
        {
            int lower = 1;
            int upper = 1;
            double upper_weight = 0.0;
        };
        const auto bracket = [](double fractional, int count)
        {
            InteriorBracket result;
            const int last = count - 2;
            fractional = std::clamp(fractional, 1.0,
                                    static_cast<double>(last));
            result.lower = static_cast<int>(std::floor(fractional));
            result.upper = std::min(result.lower + 1, last);
            result.upper_weight = result.upper == result.lower
                ? 0.0
                : fractional - static_cast<double>(result.lower);
            return result;
        };
        const InteriorBracket ib = bracket(
            grid_coordinates::scalar_fractional_index(x, dr_), nr_);
        const InteriorBracket jb = bracket(
            grid_coordinates::scalar_fractional_index(y, dtheta_), nth_);
        const InteriorBracket kb = bracket(
            scalar_vertical_fractional_index(x, y, z), nz_);
        VorticityBudgetPoint b{0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        for (int di = 0; di < 2; ++di)
        {
            const int ii = di == 0 ? ib.lower : ib.upper;
            const double wi = di == 0
                ? 1.0 - ib.upper_weight : ib.upper_weight;
            if (wi == 0.0)
                continue;
            for (int dj = 0; dj < 2; ++dj)
            {
                const int jj = dj == 0 ? jb.lower : jb.upper;
                const double wj = dj == 0
                    ? 1.0 - jb.upper_weight : jb.upper_weight;
                if (wj == 0.0)
                    continue;
                for (int dk = 0; dk < 2; ++dk)
                {
                    const int kk = dk == 0 ? kb.lower : kb.upper;
                    const double wk = dk == 0
                        ? 1.0 - kb.upper_weight : kb.upper_weight;
                    const double weight = wi * wj * wk;
                    if (weight == 0.0)
                        continue;
                    const VorticityBudgetPoint point = terrain_metrics_ != nullptr
                        ? cartesian_cgrid_mapped_vorticity_budget_at(
                            u, v, w, rho, pres, *terrain_metrics_,
                            ii, jj, kk, dr_, dtheta_, dz_,
                            surface_velocity_ms_.u_ms,
                            surface_velocity_ms_.v_ms)
                        : (staggered_
                            ? cartesian_cgrid_vorticity_budget_at(
                                u, v, w, rho, pres, ii, jj, kk,
                                dr_, dtheta_, dz_)
                            : cartesian_collocated_vorticity_budget_at(
                                u, v, w, rho, pres, ii, jj, kk,
                                dr_, dtheta_, dz_));
                    b.omega_x += weight * point.omega_x;
                    b.omega_y += weight * point.omega_y;
                    b.omega_z += weight * point.omega_z;
                    b.stretching += weight * point.stretching;
                    b.tilting += weight * point.tilting;
                    b.baroclinic += weight * point.baroclinic;
                }
            }
        }
        s.zeta       = static_cast<float>(b.omega_z);
        s.tilting    = static_cast<float>(b.tilting);
        s.stretching = static_cast<float>(b.stretching);
        s.baroclinic = static_cast<float>(b.baroclinic);
    }
    else
    {
        // Unsupported coordinate systems retain the explicit gridded fallback:
        // sample output-cadence vorticity_z and leave the budget terms NaN.
        s.zeta       = scalar_at(x, y, z, vort_z);
        s.tilting    = nanf;
        s.stretching = nanf;
        s.baroclinic = nanf;
    }
}

void TrajectorySet::seed(double         sim_time_s,
                         const Field3D& u,
                         const Field3D& v,
                         const Field3D& w,
                         const Field3D& rho,
                         const Field3D& pres,
                         const Field3D& vort_z)
{
    if (!cfg_.enabled || seeded_)
        return;
    if (domain_.empty())
    {
        seeded_ = true;
        return;
    }

    double     cx                          = cfg_.center_x_m;
    double     cy                          = cfg_.center_y_m;
    double     cz                          = cfg_.center_z_m;
    const bool automatic_horizontal_center = std::isnan(cx) || std::isnan(cy);
    const bool automatic_vertical_center   = std::isnan(cz);

    // Auto-centre on the developing low-level mesocyclone: the (i, j) column of
    // strongest |vorticity_z| over the ~0.3-2 km AGL band.
    if (automatic_horizontal_center)
    {
        double best_vort = -1.0, best_w = -1.0;
        int    vi = nr_ / 2, vj = nth_ / 2, wi = nr_ / 2, wj = nth_ / 2;
        for (int i = domain_.i_begin; i < domain_.i_end; ++i)
        {
            for (int j = domain_.j_begin; j < domain_.j_end; ++j)
            {
                const double ground = terrain_metrics_ == nullptr
                    ? 0.0
                    : terrain_metrics_->z_face(i, j, domain_.k_begin);
                for (int k = domain_.k_begin; k < domain_.k_end; ++k)
                {
                    const double height = terrain_metrics_ == nullptr
                        ? grid_coordinates::scalar_center(k, dz_)
                        : terrain_metrics_->z(i, j, k);
                    const double agl = height - ground;
                    if (agl < 300.0 || agl > 2000.0)
                        continue;
                    const double mv = std::fabs(static_cast<double>(vort_z[i][j][k]));
                    if (mv > best_vort)
                    {
                        best_vort = mv;
                        vi        = i;
                        vj        = j;
                    }
                    double sampled_w = static_cast<double>(w[i][j][k]);
                    if (staggered_)
                    {
                        const double lower_w = k == domain_.k_begin
                            ? (terrain_metrics_ == nullptr
                                   ? 0.0
                                   : terrain_transport::cgrid_ground_vertical_velocity(
                                         *terrain_metrics_, u, v, i, j,
                                         surface_velocity_ms_.u_ms,
                                         surface_velocity_ms_.v_ms))
                            : static_cast<double>(w[i][j][k - 1]);
                        sampled_w = 0.5 * (lower_w + sampled_w);
                    }
                    const double mw = std::fabs(sampled_w);
                    if (mw > best_w)
                    {
                        best_w = mw;
                        wi     = i;
                        wj     = j;
                    }
                }
            }
        }
        // Prefer the rotation maximum (the developing mesocyclone); fall back to
        // the strongest updraft when there is no meaningful low-level vorticity
        // yet (e.g. early in the run, before genesis).
        const int bi = (best_vort > 1.0e-4) ? vi : wi;
        const int bj = (best_vort > 1.0e-4) ? vj : wj;
        cx           = grid_coordinates::scalar_center(bi, dr_);
        cy           = grid_coordinates::scalar_center(bj, dtheta_);
    }
    else
    {
        const auto grid_center = frame_.earth_to_grid({cx, cy}, sim_time_s);
        cx                     = grid_center.x_m;
        cy                     = grid_center.y_m;
    }
    if (automatic_vertical_center)
        cz = clamp_vertical(
            cx, cy, local_z_lower(cx, cy) + 1000.0);

    auto axis_frac = [](int idx, int n) -> double
    {
        return (n <= 1) ? 0.0 : (static_cast<double>(idx) / (n - 1)) * 2.0 - 1.0; // [-1, 1]
    };

    // The default automatic 10 km x 10 km seed box must remain usable on
    // student-size domains and when the detected storm is near a boundary.
    // Intersect the requested box with the physical domain, then distribute
    // the configured count across the retained interval. Explicit Earth-frame
    // boxes retain their exact semantics (including an intentionally empty
    // intersection) because silently relocating a requested experiment would
    // corrupt provenance.
    const double automatic_x_min     = std::max(x_lower(), cx - cfg_.half_x_m);
    const double automatic_x_max     = std::min(x_upper(), cx + cfg_.half_x_m);
    const double automatic_y_min     = std::max(y_lower(), cy - cfg_.half_y_m);
    const double automatic_y_max     = std::min(y_upper(), cy + cfg_.half_y_m);
    const auto   interval_coordinate = [](int idx, int n, double lower, double upper) -> double
    {
        return (n <= 1) ? 0.5 * (lower + upper)
                        : lower + (upper - lower) * static_cast<double>(idx) / static_cast<double>(n - 1);
    };

    int id = 0;
    for (int ix = 0; ix < cfg_.nx; ++ix)
    {
        const double x = automatic_horizontal_center
                             ? interval_coordinate(ix, cfg_.nx, automatic_x_min, automatic_x_max)
                             : cx + axis_frac(ix, cfg_.nx) * cfg_.half_x_m;
        for (int iy = 0; iy < cfg_.ny; ++iy)
        {
            const double y = automatic_horizontal_center
                                 ? interval_coordinate(iy, cfg_.ny, automatic_y_min, automatic_y_max)
                                 : cy + axis_frac(iy, cfg_.ny) * cfg_.half_y_m;
            for (int iz = 0; iz < cfg_.nz; ++iz)
            {
                if (!in_horizontal_domain(x, y))
                    continue;
                // An automatic terrain seed is specified in AGL so the entire
                // lattice follows the same atmospheric layer. An explicit
                // centre_z_m remains an absolute geometric (MSL) height.
                const double column_center_z = automatic_vertical_center
                    ? clamp_vertical(x, y, local_z_lower(x, y) + 1000.0)
                    : cz;
                double z = column_center_z
                    + axis_frac(iz, cfg_.nz) * cfg_.half_z_m;
                z = clamp_vertical(x, y, z);
                Parcel p;
                p.id     = id++;
                p.active = true;
                p.x      = x;
                p.y      = y;
                p.z      = z;
                TrajectorySample s;
                s.t                       = static_cast<float>(sim_time_s);
                const auto earth_position = frame_.grid_to_earth({x, y}, sim_time_s);
                s.x                       = static_cast<float>(earth_position.x_m);
                s.y                       = static_cast<float>(earth_position.y_m);
                s.z                       = static_cast<float>(z);
                sample_fields(x, y, z, u, v, w, rho, pres, vort_z, s);
                retain_sample(p, s);
                parcels_.push_back(std::move(p));
            }
        }
    }
    seeded_ = true;
    next_sample_time_s_ = cfg_.sample_interval_s > 0.0
        ? sim_time_s + cfg_.sample_interval_s
        : std::numeric_limits<double>::quiet_NaN();
}

void TrajectorySet::prepare_advance(double start_time_s, double dt,
                                    const Field3D& u, const Field3D& v, const Field3D& w)
{
    prepared_dt_ = 0.0;
    if (!seeded_) return;
    if (!std::isfinite(start_time_s) || !std::isfinite(dt) || dt <= 0.0 ||
        !std::isfinite(start_time_s + dt) || start_time_s + dt <= start_time_s)
        throw std::invalid_argument("trajectory step requires a finite advancing clock");
    for (const Field3D* field : {&u, &v, &w})
        if (field->size_r() != nr_ || field->size_th() != nth_ || field->size_z() != nz_)
            throw std::invalid_argument("trajectory wind shape differs from the configured grid");
    midpoints_.resize(parcels_.size());
    for (std::size_t index = 0; index < parcels_.size(); ++index)
    {
        const auto& parcel = parcels_[index];
        if (!parcel.active) continue;
        auto& midpoint = midpoints_[index];
        double vx, vy, vz;
        velocity_at(parcel.x, parcel.y, parcel.z, u, v, w, vx, vy, vz);
        if (!std::isfinite(vx) || !std::isfinite(vy) || !std::isfinite(vz))
            throw std::invalid_argument("trajectory beginning wind must be finite");
        midpoint.x = parcel.x + 0.5 * dt * vx;
        midpoint.y = parcel.y + 0.5 * dt * vy;
        const double z = parcel.z + 0.5 * dt * vz;
        if (!std::isfinite(midpoint.x) || !std::isfinite(midpoint.y) || !std::isfinite(z))
            throw std::invalid_argument("trajectory midpoint is not finite");
        midpoint.z = clamp_vertical(midpoint.x, midpoint.y, z);
        velocity_at(midpoint.x, midpoint.y, midpoint.z, u, v, w,
                    midpoint.vx, midpoint.vy, midpoint.vz);
        if (!std::isfinite(midpoint.vx) || !std::isfinite(midpoint.vy) || !std::isfinite(midpoint.vz))
            throw std::invalid_argument("trajectory midpoint wind must be finite");
    }
    prepared_end_time_s_ = start_time_s + dt;
    prepared_dt_ = dt;
}

void TrajectorySet::advance(double sim_time_s, double dt,
                            const Field3D& u, const Field3D& v, const Field3D& w,
                            const Field3D& rho, const Field3D& pres, const Field3D& vort_z)
{
    if (!seeded_ || dt <= 0.0) return;
    prepare_advance(sim_time_s - dt, dt, u, v, w);
    finish_advance(sim_time_s, dt, u, v, w, rho, pres, vort_z);
}

void TrajectorySet::finish_advance(double         sim_time_s,
                            double         dt,
                            const Field3D& u,
                            const Field3D& v,
                            const Field3D& w,
                            const Field3D& rho,
                            const Field3D& pres,
                            const Field3D& vort_z)
{
    if (!seeded_ || dt <= 0.0)
        return;

    if (!std::isfinite(sim_time_s) || !std::isfinite(dt) || dt != prepared_dt_ ||
        std::abs(sim_time_s - prepared_end_time_s_) >
            16.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(sim_time_s)) ||
        midpoints_.size() != parcels_.size())
        throw std::invalid_argument("trajectory endpoint does not match its prepared step");
    for (const Field3D* field : {&u, &v, &w})
        if (field->size_r() != nr_ || field->size_th() != nth_ || field->size_z() != nz_)
            throw std::invalid_argument("trajectory wind shape differs from the configured grid");

    const double cadence_tolerance =
        16.0 * std::numeric_limits<double>::epsilon() *
        std::max(1.0, std::abs(sim_time_s));
    const bool record_sample = cfg_.sample_interval_s == 0.0 ||
        sim_time_s + cadence_tolerance >= next_sample_time_s_;

    for (std::size_t index = 0; index < parcels_.size(); ++index)
    {
        auto& p = parcels_[index];
        if (!p.active) continue;
        const auto& midpoint = midpoints_[index];
        double v2x, v2y, v2z;
        velocity_at(midpoint.x, midpoint.y, midpoint.z, u, v, w, v2x, v2y, v2z);
        v2x = 0.5 * (midpoint.vx + v2x);
        v2y = 0.5 * (midpoint.vy + v2y);
        v2z = 0.5 * (midpoint.vz + v2z);

        p.x += dt * v2x;
        p.y += dt * v2y;
        p.z = clamp_vertical(p.x, p.y, p.z + dt * v2z);

        if (!in_horizontal_domain(p.x, p.y))
        {
            p.active = false; // left the domain — stop tracking
            continue;
        }

        if (record_sample)
        {
            TrajectorySample s;
            s.t                       = static_cast<float>(sim_time_s);
            const auto earth_position = frame_.grid_to_earth({p.x, p.y}, sim_time_s);
            s.x                       = static_cast<float>(earth_position.x_m);
            s.y                       = static_cast<float>(earth_position.y_m);
            s.z                       = static_cast<float>(p.z);
            sample_fields(p.x, p.y, p.z, u, v, w, rho, pres, vort_z, s);
            retain_sample(p, s);
        }
    }

    if (record_sample && cfg_.sample_interval_s > 0.0)
    {
        const double elapsed = std::max(0.0, sim_time_s - next_sample_time_s_);
        const double crossed =
            std::floor(elapsed / cfg_.sample_interval_s) + 1.0;
        next_sample_time_s_ += crossed * cfg_.sample_interval_s;
        if (!(next_sample_time_s_ > sim_time_s))
            next_sample_time_s_ = std::nextafter(
                sim_time_s, std::numeric_limits<double>::infinity());
    }
    prepared_dt_ = 0.0;
}

void TrajectorySet::preflight_domain_recycling(double                                    displacement_x_m,
                                               double                                    displacement_y_m,
                                               const tmv::domain::HorizontalDomainFrame& post_shift_frame) const
{
    if (!std::isfinite(displacement_x_m) || !std::isfinite(displacement_y_m))
        throw std::invalid_argument("trajectory domain-recycling displacement must be finite");
    if (prepared_dt_ != 0.0)
        throw std::logic_error("trajectory recycling requires a completed integration step");
    post_shift_frame.validate();
    if (post_shift_frame.velocity_ms.u_ms != frame_.velocity_ms.u_ms ||
        post_shift_frame.velocity_ms.v_ms != frame_.velocity_ms.v_ms)
        throw std::invalid_argument("trajectory domain recycling cannot change frame velocity");
    for (const Parcel& parcel : parcels_)
        if (!std::isfinite(parcel.x) || !std::isfinite(parcel.y) || !std::isfinite(parcel.z))
            throw std::invalid_argument("trajectory domain recycling found a non-finite parcel");
}

void TrajectorySet::commit_domain_recycling(double                                    displacement_x_m,
                                            double                                    displacement_y_m,
                                            const tmv::domain::HorizontalDomainFrame& post_shift_frame) noexcept
{
    frame_ = post_shift_frame;
    for (Parcel& parcel : parcels_)
    {
        parcel.x -= displacement_x_m;
        parcel.y -= displacement_y_m;
        if (parcel.active && !in_horizontal_domain(parcel.x, parcel.y))
            parcel.active = false;
    }
}

std::size_t TrajectorySet::total_samples() const
{
    std::size_t n = 0;
    for (const auto& p : parcels_)
        n += p.path.size();
    return n;
}

bool TrajectorySet::write(const std::string& path) const
{
    const std::size_t P = parcels_.size();
    if (P == 0)
        return false;
    std::size_t S = 0;
    for (const auto& p : parcels_)
        S = std::max(S, p.path.size());
    if (S == 0)
        return false;

    if (P > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        S > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        P > std::numeric_limits<std::size_t>::max() / S ||
        P * S > std::numeric_limits<std::size_t>::max() /
                    static_cast<std::size_t>(kTrajectoryVarCount))
        return false;

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out || !write_trajectory_npy_header(out, P, S))
        return false;

    constexpr std::size_t padding_chunk_values = 4096;
    std::array<float, padding_chunk_values> padding{};
    padding.fill(std::numeric_limits<float>::quiet_NaN());
    for (std::size_t pi = 0; pi < P; ++pi)
    {
        const auto& path_pts = parcels_[pi].path;
        for (const TrajectorySample& sample : path_pts)
        {
            out.write(reinterpret_cast<const char*>(&sample),
                      static_cast<std::streamsize>(sizeof(sample)));
        }
        std::size_t remaining =
            (S - path_pts.size()) * static_cast<std::size_t>(kTrajectoryVarCount);
        while (remaining > 0)
        {
            const std::size_t count = std::min(remaining, padding.size());
            out.write(reinterpret_cast<const char*>(padding.data()),
                      static_cast<std::streamsize>(count * sizeof(float)));
            remaining -= count;
        }
        if (!out)
            return false;
    }
    out.close();
    return out.good();
}

} // namespace tmv
