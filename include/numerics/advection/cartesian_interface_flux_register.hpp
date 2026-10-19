#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tmv
{

struct CartesianInterfaceFootprint
{
    int i_begin = 0;
    int i_end   = 0;
    int j_begin = 0;
    int j_end   = 0;
    int k_begin = 0;
    int k_end   = 0;
};

struct CartesianThetaInterfaceFluxes
{
    std::vector<double> west;
    std::vector<double> east;
    std::vector<double> south;
    std::vector<double> north;
    double              accumulated_dt_s   = 0.0;
    std::uint64_t       contribution_count = 0;
};

using CartesianScalarInterfaceFluxes = CartesianThetaInterfaceFluxes;

struct CartesianDryMassInterfaceFluxes
{
    std::vector<double> west;
    std::vector<double> east;
    std::vector<double> south;
    std::vector<double> north;
    double              accumulated_dt_s   = 0.0;
    std::uint64_t       contribution_count = 0;
};

class CartesianDryMassInterfaceFluxRegister
{
public:
    void configure(int                         nr,
                   int                         nth,
                   int                         nz,
                   CartesianInterfaceFootprint footprint,
                   int                         x_face_tangential_ratio = 1,
                   int                         y_face_tangential_ratio = 1);
    void reset_interval() noexcept;
    void
    accumulate(const float* mass_flux_x, const float* mass_flux_y, double dt_s, double dx_m, double dy_m, double dz_m);

    [[nodiscard]] bool interval_empty() const noexcept
    {
        return fluxes_.contribution_count == 0 && fluxes_.accumulated_dt_s == 0;
    }
    [[nodiscard]] bool configured() const noexcept
    {
        return configured_;
    }
    [[nodiscard]] const CartesianInterfaceFootprint& footprint() const noexcept
    {
        return footprint_;
    }
    [[nodiscard]] const CartesianDryMassInterfaceFluxes& fluxes() const noexcept
    {
        return fluxes_;
    }

private:
    bool                            configured_ = false;
    int                             nr_         = 0;
    int                             nth_        = 0;
    int                             nz_         = 0;
    int                             x_ratio_    = 1;
    int                             y_ratio_    = 1;
    CartesianInterfaceFootprint     footprint_{};
    CartesianDryMassInterfaceFluxes fluxes_{};
};

class CartesianInterfaceFluxRegister
{
public:
    void configure(int                         nr,
                   int                         nth,
                   int                         nz,
                   CartesianInterfaceFootprint footprint,
                   int                         x_face_tangential_ratio = 1,
                   int                         y_face_tangential_ratio = 1)
    {
        if (nr < 3 || nth < 3 || nz < 1 || footprint.i_begin < 1 || footprint.i_end >= nr || footprint.j_begin < 1 ||
            footprint.j_end >= nth || footprint.k_begin < 0 || footprint.k_end > nz ||
            footprint.i_begin >= footprint.i_end || footprint.j_begin >= footprint.j_end ||
            footprint.k_begin >= footprint.k_end || x_face_tangential_ratio <= 0 || y_face_tangential_ratio <= 0 ||
            (footprint.j_end - footprint.j_begin) % x_face_tangential_ratio != 0 ||
            (footprint.i_end - footprint.i_begin) % y_face_tangential_ratio != 0)
        {
            throw std::invalid_argument("Cartesian interface flux register requires a non-empty interior footprint");
        }
        CartesianInterfaceFluxRegister candidate;
        candidate.nr_                      = nr;
        candidate.nth_                     = nth;
        candidate.nz_                      = nz;
        candidate.footprint_               = footprint;
        candidate.x_face_tangential_ratio_ = x_face_tangential_ratio;
        candidate.y_face_tangential_ratio_ = y_face_tangential_ratio;
        const std::size_t x_count =
            static_cast<std::size_t>((footprint.j_end - footprint.j_begin) / x_face_tangential_ratio) *
            static_cast<std::size_t>(footprint.k_end - footprint.k_begin);
        const std::size_t y_count =
            static_cast<std::size_t>((footprint.i_end - footprint.i_begin) / y_face_tangential_ratio) *
            static_cast<std::size_t>(footprint.k_end - footprint.k_begin);
        candidate.theta_.west.assign(x_count, 0.0);
        candidate.theta_.east.assign(x_count, 0.0);
        candidate.theta_.south.assign(y_count, 0.0);
        candidate.theta_.north.assign(y_count, 0.0);
        candidate.staged_west_.assign(x_count, 0.0);
        candidate.staged_east_.assign(x_count, 0.0);
        candidate.staged_south_.assign(y_count, 0.0);
        candidate.staged_north_.assign(y_count, 0.0);
        candidate.configured_ = true;
        *this                 = std::move(candidate);
    }

    void reset_interval() noexcept
    {
        std::fill(theta_.west.begin(), theta_.west.end(), 0.0);
        std::fill(theta_.east.begin(), theta_.east.end(), 0.0);
        std::fill(theta_.south.begin(), theta_.south.end(), 0.0);
        std::fill(theta_.north.begin(), theta_.north.end(), 0.0);
        theta_.accumulated_dt_s   = 0.0;
        theta_.contribution_count = 0;
        staged_dt_s_              = 0.0;
        staged_                   = false;
        staged_validated_         = false;
    }

    [[nodiscard]] bool interval_empty() const noexcept
    {
        return !staged_ && theta_.contribution_count == 0 && theta_.accumulated_dt_s == 0;
    }

    void require_compatible(int nr, int nth, int nz) const
    {
        if (!configured_ || nr != nr_ || nth != nth_ || nz != nz_)
        {
            throw std::invalid_argument("Cartesian interface flux register is not configured for the active level");
        }
    }

    template <class FaceSampler>
    void stage_scalar_flux(const FaceSampler& sample, double dt_s, double dx_m, double dy_m, double dz_m)
    {
        require_compatible(nr_, nth_, nz_);
        if (!std::isfinite(dt_s) || !std::isfinite(dx_m) || !std::isfinite(dy_m) || !std::isfinite(dz_m) ||
            dt_s <= 0.0 || dx_m <= 0.0 || dy_m <= 0.0 || dz_m <= 0.0)
        {
            throw std::invalid_argument(
                "Cartesian interface scalar flux accumulation requires finite fluxes and metrics");
        }

        if (staged_)
            throw std::logic_error("Cartesian scalar interface register already owns a staged contribution");
        const double x_scale    = dt_s * dy_m * dz_m;
        const double y_scale    = dt_s * dx_m * dz_m;
        const int    west_face  = footprint_.i_begin - 1;
        const int    east_face  = footprint_.i_end - 1;
        const int    south_face = footprint_.j_begin - 1;
        const int    north_face = footprint_.j_end - 1;
        std::fill(staged_west_.begin(), staged_west_.end(), 0.0);
        std::fill(staged_east_.begin(), staged_east_.end(), 0.0);
        std::fill(staged_south_.begin(), staged_south_.end(), 0.0);
        std::fill(staged_north_.begin(), staged_north_.end(), 0.0);
        const std::size_t k_count = static_cast<std::size_t>(footprint_.k_end - footprint_.k_begin);
        for (int j = footprint_.j_begin; j < footprint_.j_end; ++j)
            for (int k = footprint_.k_begin; k < footprint_.k_end; ++k)
            {
                const std::size_t bin =
                    static_cast<std::size_t>((j - footprint_.j_begin) / x_face_tangential_ratio_) * k_count +
                    static_cast<std::size_t>(k - footprint_.k_begin);
                const double west = x_scale * sample(0, west_face, j, k);
                const double east = x_scale * sample(0, east_face, j, k);
                if (!std::isfinite(west) || !std::isfinite(east) ||
                    !std::isfinite(theta_.west[bin] + staged_west_[bin] + west) ||
                    !std::isfinite(theta_.east[bin] + staged_east_[bin] + east))
                {
                    throw std::runtime_error("Cartesian interface scalar flux contribution is non-finite");
                }
                staged_west_[bin] += west;
                staged_east_[bin] += east;
            }
        for (int i = footprint_.i_begin; i < footprint_.i_end; ++i)
            for (int k = footprint_.k_begin; k < footprint_.k_end; ++k)
            {
                const std::size_t bin =
                    static_cast<std::size_t>((i - footprint_.i_begin) / y_face_tangential_ratio_) * k_count +
                    static_cast<std::size_t>(k - footprint_.k_begin);
                const double south = y_scale * sample(1, i, south_face, k);
                const double north = y_scale * sample(1, i, north_face, k);
                if (!std::isfinite(south) || !std::isfinite(north) ||
                    !std::isfinite(theta_.south[bin] + staged_south_[bin] + south) ||
                    !std::isfinite(theta_.north[bin] + staged_north_[bin] + north))
                {
                    throw std::runtime_error("Cartesian interface scalar flux contribution is non-finite");
                }
                staged_south_[bin] += south;
                staged_north_[bin] += north;
            }
        if (!std::isfinite(theta_.accumulated_dt_s + dt_s))
            throw std::runtime_error("Cartesian interface scalar flux duration is non-finite");
        staged_dt_s_      = dt_s;
        staged_           = true;
        staged_validated_ = false;
    }

    void stage_theta_flux(const float* flux_x, const float* flux_y, double dt_s, double dx_m, double dy_m, double dz_m)
    {
        if (!flux_x || !flux_y)
            throw std::invalid_argument("Cartesian interface flux storage is missing");
        stage_scalar_flux(
            [&](int axis, int i, int j, int k)
            {
                const auto index = (std::size_t(i) * nth_ + j) * nz_ + k;
                return double(axis == 0 ? flux_x[index] : flux_y[index]);
            },
            dt_s,
            dx_m,
            dy_m,
            dz_m);
    }

    void validate_staged_scalar_commit()
    {
        if (!staged_)
            throw std::logic_error("Cartesian interface scalar flux commit has no staged contribution");
        const auto validate = [](const std::vector<double>& committed, const std::vector<double>& staged)
        {
            if (committed.size() != staged.size())
                throw std::logic_error("Cartesian interface scalar staged strip shape changed");
            for (std::size_t bin = 0; bin < committed.size(); ++bin)
                if (!std::isfinite(committed[bin] + staged[bin]))
                    throw std::overflow_error("Cartesian interface scalar cumulative strip overflowed");
        };
        validate(theta_.west, staged_west_);
        validate(theta_.east, staged_east_);
        validate(theta_.south, staged_south_);
        validate(theta_.north, staged_north_);
        if (!std::isfinite(theta_.accumulated_dt_s + staged_dt_s_))
            throw std::overflow_error("Cartesian interface scalar cumulative duration overflowed");
        if (theta_.contribution_count == std::numeric_limits<std::uint64_t>::max())
            throw std::overflow_error("Cartesian interface scalar contribution count overflowed");
        staged_validated_ = true;
    }

    void commit_staged_scalar() noexcept
    {
        if (!staged_ || !staged_validated_)
            return;
        for (std::size_t bin = 0; bin < theta_.west.size(); ++bin)
        {
            theta_.west[bin] += staged_west_[bin];
            theta_.east[bin] += staged_east_[bin];
        }
        for (std::size_t bin = 0; bin < theta_.south.size(); ++bin)
        {
            theta_.south[bin] += staged_south_[bin];
            theta_.north[bin] += staged_north_[bin];
        }
        theta_.accumulated_dt_s += staged_dt_s_;
        ++theta_.contribution_count;
        staged_           = false;
        staged_validated_ = false;
        staged_dt_s_      = 0.0;
    }

    void discard_staged_scalar() noexcept
    {
        std::fill(staged_west_.begin(), staged_west_.end(), 0.0);
        std::fill(staged_east_.begin(), staged_east_.end(), 0.0);
        std::fill(staged_south_.begin(), staged_south_.end(), 0.0);
        std::fill(staged_north_.begin(), staged_north_.end(), 0.0);
        staged_dt_s_      = 0.0;
        staged_           = false;
        staged_validated_ = false;
    }

    void validate_staged_theta_commit()
    {
        validate_staged_scalar_commit();
    }
    void commit_staged_theta() noexcept
    {
        commit_staged_scalar();
    }
    void discard_staged_theta() noexcept
    {
        discard_staged_scalar();
    }
    [[nodiscard]] const CartesianScalarInterfaceFluxes& scalar() const noexcept
    {
        return theta_;
    }

    [[nodiscard]] bool configured() const noexcept
    {
        return configured_;
    }
    [[nodiscard]] const CartesianInterfaceFootprint& footprint() const noexcept
    {
        return footprint_;
    }
    [[nodiscard]] const CartesianThetaInterfaceFluxes& theta() const noexcept
    {
        return theta_;
    }

private:
    bool                          configured_              = false;
    int                           nr_                      = 0;
    int                           nth_                     = 0;
    int                           nz_                      = 0;
    int                           x_face_tangential_ratio_ = 1;
    int                           y_face_tangential_ratio_ = 1;
    CartesianInterfaceFootprint   footprint_{};
    CartesianThetaInterfaceFluxes theta_{};
    std::vector<double>           staged_west_;
    std::vector<double>           staged_east_;
    std::vector<double>           staged_south_;
    std::vector<double>           staged_north_;
    double                        staged_dt_s_      = 0.0;
    bool                          staged_           = false;
    bool                          staged_validated_ = false;
};

struct CartesianMoistInterfaceCapture
{
    bool                                          enabled = false;
    std::array<CartesianInterfaceFluxRegister, 7> water;
    std::array<double, 8>                         committed_physics{};
    void configure(int nr, int nth, int nz, CartesianInterfaceFootprint footprint, int ratio)
    {
        CartesianMoistInterfaceCapture candidate;
        for (auto& field : candidate.water)
            field.configure(nr, nth, nz, footprint, ratio, ratio);
        candidate.enabled = true;
        *this             = std::move(candidate);
    }
    void reset_interval() noexcept
    {
        for (auto& field : water)
            field.reset_interval();
        committed_physics = {};
    }
};

} // namespace tmv
