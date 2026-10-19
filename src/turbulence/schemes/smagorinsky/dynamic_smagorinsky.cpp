#include "dynamic_smagorinsky.hpp"

#include "core/infra/physical_cell_domain.hpp"
#include "core/runtime/runtime_config.hpp"
#include "turbulence/base/eddy_viscosity.hpp"
#include "util/grid_metric_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tmv::dynamic_smagorinsky
{
namespace
{

using Tensor = std::array<double, 6>;
constexpr std::array<std::pair<int, int>, 6> kTensorPairs{{{0, 0}, {1, 1}, {2, 2}, {0, 1}, {0, 2}, {1, 2}}};

struct ResolvedFields
{
    Field3D u;
    Field3D v;
    Field3D w;
    Field3D rho;
};

int wrap(int index, int begin, int end)
{
    if (index < begin)
        return end - 1;
    if (index >= end)
        return begin;
    return index;
}

std::vector<double> vertical_centers(const GridMetrics& grid, int nz)
{
    std::vector<double> centers(static_cast<std::size_t>(nz));
    if (grid.z_int.size() == static_cast<std::size_t>(nz + 1))
    {
        for (int k = 0; k < nz; ++k)
            centers[static_cast<std::size_t>(k)] =
                0.5 * (grid.z_int[static_cast<std::size_t>(k)] + grid.z_int[static_cast<std::size_t>(k + 1)]);
        return centers;
    }
    double height = 0.0;
    for (int k = 0; k < nz; ++k)
    {
        const double thickness               = grid_metric::vertical_cell_thickness(grid, 0, 0, k, nz);
        centers[static_cast<std::size_t>(k)] = height + 0.5 * thickness;
        height += thickness;
    }
    return centers;
}

ResolvedFields reconstruct_centers(const TurbulenceStateView& state, const PhysicalCellDomain& domain)
{
    ResolvedFields fields{Field3D(state.NR, state.NTH, state.NZ, 0.0f),
                          Field3D(state.NR, state.NTH, state.NZ, 0.0f),
                          Field3D(state.NR, state.NTH, state.NZ, 0.0f),
                          Field3D(state.NR, state.NTH, state.NZ, 0.0f)};
#pragma omp parallel for collapse(2) if (static_cast<std::size_t>(domain.nx()) * domain.ny() * domain.nz() >= 32768)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                fields.u(i, j, k)   = 0.5f * ((*state.u)(i - 1, j, k) + (*state.u)(i, j, k));
                fields.v(i, j, k)   = 0.5f * ((*state.v)(i, j - 1, k) + (*state.v)(i, j, k));
                const float w_below = k == domain.k_begin ? 0.0f : (*state.w)(i, j, k - 1);
                fields.w(i, j, k)   = 0.5f * (w_below + (*state.w)(i, j, k));
                fields.rho(i, j, k) = (*state.rho)(i, j, k);
            }
    return fields;
}

void validate_resolved(const ResolvedFields& fields, const PhysicalCellDomain& domain)
{
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                if (!std::isfinite(static_cast<double>(fields.u(i, j, k))) ||
                    !std::isfinite(static_cast<double>(fields.v(i, j, k))) ||
                    !std::isfinite(static_cast<double>(fields.w(i, j, k))) ||
                    !std::isfinite(static_cast<double>(fields.rho(i, j, k))) || fields.rho(i, j, k) <= 0.0f)
                {
                    throw std::invalid_argument(
                        "dynamic Smagorinsky requires finite velocity and positive finite density");
                }
            }
}

template <typename Sample>
double test_filter(const PhysicalCellDomain& domain, int i, int j, int k, Sample&& sample)
{
    constexpr std::array<double, 3> weights{0.25, 0.5, 0.25};
    double                          sum = 0.0;
    for (int di = -1; di <= 1; ++di)
        for (int dj = -1; dj <= 1; ++dj)
            for (int dk = -1; dk <= 1; ++dk)
            {
                const int ii = wrap(i + di, domain.i_begin, domain.i_end);
                const int jj = wrap(j + dj, domain.j_begin, domain.j_end);
                const int kk = std::clamp(k + dk, domain.k_begin, domain.k_end - 1);
                sum += weights[static_cast<std::size_t>(di + 1)] * weights[static_cast<std::size_t>(dj + 1)] *
                       weights[static_cast<std::size_t>(dk + 1)] * sample(ii, jj, kk);
            }
    return sum;
}

Tensor strain(const ResolvedFields&      fields,
              const GridMetrics&         grid,
              const std::vector<double>& z_center,
              const PhysicalCellDomain&  domain,
              int                        i,
              int                        j,
              int                        k)
{
    const int    im = wrap(i - 1, domain.i_begin, domain.i_end);
    const int    ip = wrap(i + 1, domain.i_begin, domain.i_end);
    const int    jm = wrap(j - 1, domain.j_begin, domain.j_end);
    const int    jp = wrap(j + 1, domain.j_begin, domain.j_end);
    const double dx = grid_metric::local_dx(grid, i, j, k);
    const double dy = grid_metric::local_dy(grid, i, j, k);
    const double dz =
        k == domain.k_begin     ? z_center[static_cast<std::size_t>(k + 1)] - z_center[static_cast<std::size_t>(k)]
        : k + 1 == domain.k_end ? z_center[static_cast<std::size_t>(k)] - z_center[static_cast<std::size_t>(k - 1)]
                                : z_center[static_cast<std::size_t>(k + 1)] - z_center[static_cast<std::size_t>(k - 1)];
    const int  km           = k == domain.k_begin ? k : k - 1;
    const int  kp           = k + 1 == domain.k_end ? k : k + 1;
    const auto derivative_x = [&](const Field3D& field)
    { return (static_cast<double>(field(ip, j, k)) - static_cast<double>(field(im, j, k))) / (2.0 * dx); };
    const auto derivative_y = [&](const Field3D& field)
    { return (static_cast<double>(field(i, jp, k)) - static_cast<double>(field(i, jm, k))) / (2.0 * dy); };
    const auto derivative_z = [&](const Field3D& field)
    { return (static_cast<double>(field(i, j, kp)) - static_cast<double>(field(i, j, km))) / dz; };
    const double dudx = derivative_x(fields.u);
    const double dudy = derivative_y(fields.u);
    const double dudz = derivative_z(fields.u);
    const double dvdx = derivative_x(fields.v);
    const double dvdy = derivative_y(fields.v);
    const double dvdz = derivative_z(fields.v);
    const double dwdx = derivative_x(fields.w);
    const double dwdy = derivative_y(fields.w);
    const double dwdz = derivative_z(fields.w);
    return {dudx, dvdy, dwdz, 0.5 * (dudy + dvdx), 0.5 * (dudz + dwdx), 0.5 * (dvdz + dwdy)};
}

double contraction(const Tensor& left, const Tensor& right)
{
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2] +
           2.0 * (left[3] * right[3] + left[4] * right[4] + left[5] * right[5]);
}

double magnitude(const Tensor& tensor)
{
    return std::sqrt(2.0 * contraction(tensor, tensor));
}

std::array<Field3D, 6> build_grid_model_tensor(const ResolvedFields&      fields,
                                               const TurbulenceConfig&    config,
                                               const GridMetrics&         grid,
                                               const std::vector<double>& z_center,
                                               const PhysicalCellDomain&  domain)
{
    std::array<Field3D, 6> model;
    for (auto& component : model)
        component.resize(fields.u.size_r(), fields.u.size_th(), fields.u.size_z(), 0.0f);
#pragma omp parallel for collapse(2) if (static_cast<std::size_t>(domain.nx()) * domain.ny() * domain.nz() >= 32768)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const Tensor resolved = strain(fields, grid, z_center, domain, i, j, k);
                const double delta    = eddy_viscosity::compute_filter_width(config, grid, i, j, k);
                const double scale    = delta * delta * magnitude(resolved);
                for (std::size_t component = 0; component < model.size(); ++component)
                    model[component](i, j, k) = static_cast<float>(scale * resolved[component]);
            }
    return model;
}

ResolvedFields favre_test_filter(const ResolvedFields& fields, const PhysicalCellDomain& domain)
{
    ResolvedFields filtered{Field3D(fields.u.size_r(), fields.u.size_th(), fields.u.size_z(), 0.0f),
                            Field3D(fields.u.size_r(), fields.u.size_th(), fields.u.size_z(), 0.0f),
                            Field3D(fields.u.size_r(), fields.u.size_th(), fields.u.size_z(), 0.0f),
                            Field3D(fields.u.size_r(), fields.u.size_th(), fields.u.size_z(), 0.0f)};
#pragma omp parallel for collapse(2) if (static_cast<std::size_t>(domain.nx()) * domain.ny() * domain.nz() >= 32768)
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double density =
                    test_filter(domain,
                                i,
                                j,
                                k,
                                [&](int ii, int jj, int kk) { return static_cast<double>(fields.rho(ii, jj, kk)); });
                filtered.rho(i, j, k)       = static_cast<float>(density);
                const Field3D* velocities[] = {&fields.u, &fields.v, &fields.w};
                Field3D*       outputs[]    = {&filtered.u, &filtered.v, &filtered.w};
                for (std::size_t component = 0; component < 3; ++component)
                {
                    const double momentum =
                        test_filter(domain,
                                    i,
                                    j,
                                    k,
                                    [&](int ii, int jj, int kk)
                                    {
                                        return static_cast<double>(fields.rho(ii, jj, kk)) *
                                               static_cast<double>((*velocities[component])(ii, jj, kk));
                                    });
                    (*outputs[component])(i, j, k) = static_cast<float>(momentum / density);
                }
            }
    return filtered;
}

} // namespace

void compute_coefficient(const TurbulenceConfig&    config,
                         const GridMetrics&         grid,
                         const TurbulenceStateView& state,
                         Field3D&                   effective_cs)
{
    using tmv::init::config::LateralBC;
    const bool periodic_x =
        global_lateral_bc_west == LateralBC::Periodic && global_lateral_bc_east == LateralBC::Periodic;
    const bool periodic_y =
        global_lateral_bc_south == LateralBC::Periodic && global_lateral_bc_north == LateralBC::Periodic;
    if (!state.c_grid || grid.terrain_metrics_active || config.mode != "3d" || config.stability_correction != "none" ||
        !periodic_x || !periodic_y)
    {
        throw std::invalid_argument(
            "dynamic Smagorinsky requires a flat periodic 3-D C-grid with no separate stability correction");
    }
    if (!std::isfinite(config.Cs) || config.Cs <= 0.0)
        throw std::invalid_argument("dynamic Smagorinsky requires a positive finite Cs upper bound");
    if ((config.filter_width != "cubic_root" && config.filter_width != "dx" && config.filter_width != "user") ||
        !std::isfinite(config.user_filter_width_m) ||
        (config.filter_width == "user" ? config.user_filter_width_m <= 0.0 : config.user_filter_width_m != 0.0))
    {
        throw std::invalid_argument("dynamic Smagorinsky requires a valid filter-width selection");
    }
    const auto shape_matches = [&](const Field3D* field)
    {
        return field != nullptr && field->size_r() == state.NR && field->size_th() == state.NTH &&
               field->size_z() == state.NZ;
    };
    if (!shape_matches(state.u) || !shape_matches(state.v) || !shape_matches(state.w) || !shape_matches(state.rho))
        throw std::invalid_argument("dynamic Smagorinsky requires shape-matched velocity and density fields");
    grid_metric::sanitize_positive(grid.dx);
    grid_metric::sanitize_positive(grid.dy);
    grid_metric::validate_vertical_metrics(grid, state.NR, state.NTH, state.NZ);

    const PhysicalCellDomain domain = physical_scalar_domain(state.NR, state.NTH, state.NZ, StaggerType::CGrid);
    if (domain.nx() < 4 || domain.ny() < 4 || domain.nz() < 3)
        throw std::invalid_argument("dynamic Smagorinsky requires at least 4x4x3 physical cells");

    const ResolvedFields resolved = reconstruct_centers(state, domain);
    validate_resolved(resolved, domain);
    const ResolvedFields      filtered   = favre_test_filter(resolved, domain);
    const std::vector<double> z_center   = vertical_centers(grid, state.NZ);
    const auto                grid_model = build_grid_model_tensor(resolved, config, grid, z_center, domain);
    Field3D                   candidate(state.NR, state.NTH, state.NZ, 0.0f);
    const double              maximum_c    = config.Cs * config.Cs;
    const Field3D*            velocities[] = {&resolved.u, &resolved.v, &resolved.w};

    for (int k = domain.k_begin; k < domain.k_end; ++k)
    {
        double numerator   = 0.0;
        double denominator = 0.0;
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
            {
                const double rho_hat        = filtered.rho(i, j, k);
                const double velocity_hat[] = {filtered.u(i, j, k), filtered.v(i, j, k), filtered.w(i, j, k)};
                Tensor       leonard{};
                for (std::size_t component = 0; component < kTensorPairs.size(); ++component)
                {
                    const int    first  = kTensorPairs[component].first;
                    const int    second = kTensorPairs[component].second;
                    const double product =
                        test_filter(domain,
                                    i,
                                    j,
                                    k,
                                    [&](int ii, int jj, int kk)
                                    {
                                        return static_cast<double>(resolved.rho(ii, jj, kk)) *
                                               static_cast<double>((*velocities[first])(ii, jj, kk)) *
                                               static_cast<double>((*velocities[second])(ii, jj, kk));
                                    }) /
                        rho_hat;
                    leonard[component] = product - velocity_hat[first] * velocity_hat[second];
                }
                const double trace = (leonard[0] + leonard[1] + leonard[2]) / 3.0;
                leonard[0] -= trace;
                leonard[1] -= trace;
                leonard[2] -= trace;

                Tensor       model{};
                const double test_delta      = 2.0 * eddy_viscosity::compute_filter_width(config, grid, i, j, k);
                const Tensor filtered_strain = strain(filtered, grid, z_center, domain, i, j, k);
                const double filtered_scale  = test_delta * test_delta * magnitude(filtered_strain);
                for (std::size_t component = 0; component < model.size(); ++component)
                {
                    const double grid_term =
                        test_filter(domain,
                                    i,
                                    j,
                                    k,
                                    [&](int ii, int jj, int kk)
                                    {
                                        return static_cast<double>(resolved.rho(ii, jj, kk)) *
                                               static_cast<double>(grid_model[component](ii, jj, kk));
                                    }) /
                        rho_hat;
                    model[component] = 2.0 * (grid_term - filtered_scale * filtered_strain[component]);
                }
                numerator += contraction(leonard, model);
                denominator += contraction(model, model);
            }

        if (!std::isfinite(numerator) || !std::isfinite(denominator))
            throw std::runtime_error("dynamic Smagorinsky least-squares reduction is non-finite");
        double coefficient = 0.0;
        if (denominator > std::numeric_limits<double>::epsilon())
            coefficient = std::clamp(numerator / denominator, 0.0, maximum_c);
        if (!std::isfinite(coefficient))
            throw std::runtime_error("dynamic Smagorinsky least-squares coefficient is non-finite");
        const float cs = static_cast<float>(std::sqrt(coefficient));
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                candidate(i, j, k) = cs;
    }
    effective_cs = std::move(candidate);
}

} // namespace tmv::dynamic_smagorinsky
