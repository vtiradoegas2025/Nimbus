#include "mynn.hpp"
#include "mynn_column_geometry.hpp"

#include "compute/dispatch/boundary_layer.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tmv
{
namespace
{

struct MynnBatchColumnWork
{
    surface_fluxes::BulkFluxes fluxes;
    double                     pbl_height = 0.0;
};

std::size_t validate_batch_column(const BoundaryLayerConfig& config, const BoundaryLayerColumnStateView& column)
{
    if (!std::isfinite(config.dt_pbl) || config.dt_pbl <= 0.0 || !std::isfinite(config.pbl_max_height) ||
        config.pbl_max_height <= 0.0)
        throw std::invalid_argument("MYNN requires positive finite cadence and maximum PBL height");
    const std::size_t levels = column.theta.size();
    if (levels == 0 || !column.u || !column.v || !column.qv || !column.p || !column.rho || !column.z_int ||
        column.u.size() != levels || column.v.size() != levels || column.qv.size() != levels ||
        column.p.size() != levels || column.rho.size() != levels || column.z_int.size() != levels + 1 ||
        (column.tke && column.tke.size() != levels) || column.workspace == nullptr)
        throw std::invalid_argument("MYNN native batch requires consistently shaped column profiles and workspace");
    for (std::size_t level = 0; level < levels; ++level)
    {
        if (!std::isfinite(column.u[level]) || !std::isfinite(column.v[level]) || !std::isfinite(column.theta[level]) ||
            column.theta[level] <= 0.0 || !std::isfinite(column.qv[level]) || column.qv[level] < 0.0 ||
            !std::isfinite(column.p[level]) || column.p[level] <= 0.0 || !std::isfinite(column.rho[level]) ||
            column.rho[level] <= 0.0 || !std::isfinite(column.z_int[level]) ||
            !std::isfinite(column.z_int[level + 1]) || column.z_int[level + 1] <= column.z_int[level] ||
            (level > 0 && column.p[level] > column.p[level - 1]) ||
            (column.tke && (!std::isfinite(column.tke[level]) || column.tke[level] < 0.0)))
            throw std::invalid_argument("MYNN native batch column contains non-finite or unphysical state");
    }
    return levels;
}

void rethrow_first(const std::vector<std::exception_ptr>& failures)
{
    const auto failure = std::find_if(
        failures.begin(), failures.end(), [](const std::exception_ptr& value) { return value != nullptr; });
    if (failure != failures.end())
        std::rethrow_exception(*failure);
}

} // namespace

bool MYNNScheme::supports_native_column_batch(const BoundaryLayerConfig&,
                                              const BoundaryLayerComputeKernels& kernels) const
{
    return kernels.supports(BoundaryLayerKernelOp::MynnLevel25Stability);
}

bool MYNNScheme::compute_columns_native(const BoundaryLayerConfig&          config,
                                        const SurfaceConfig&                surface,
                                        const BoundaryLayerColumnStateView* columns,
                                        std::size_t                         column_count,
                                        BoundaryLayerTendencies*            tendencies,
                                        BoundaryLayerDiagnostics*           diagnostics,
                                        BoundaryLayerComputeKernels&        kernels)
{
    if (columns == nullptr || tendencies == nullptr || diagnostics == nullptr || column_count == 0 ||
        !supports_native_column_batch(config, kernels))
        return false;

    try
    {
        const std::size_t levels           = validate_batch_column(config, columns[0]);
        const std::size_t faces_per_column = levels - 1;
        if (column_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            (faces_per_column != 0 &&
             column_count > static_cast<std::size_t>(std::numeric_limits<int>::max()) / faces_per_column))
            return false;
        const std::size_t                face_count = column_count * faces_per_column;
        std::vector<MynnBatchColumnWork> work(column_count);
        std::vector<std::exception_ptr>  failures(column_count);
        std::vector<float>               gm(face_count);
        std::vector<float>               gh(face_count);
        std::vector<float>               length(face_count);
        std::vector<float>               q_squared(face_count);

#pragma omp parallel for if (column_count >= 8) schedule(static)
        for (std::size_t column_index = 0; column_index < column_count; ++column_index)
        {
            try
            {
                const auto& column = columns[column_index];
                if (validate_batch_column(config, column) != levels)
                    throw std::invalid_argument("MYNN native batch columns must have equal heights");
                auto& workspace = *column.workspace;
                workspace.tke.assign(levels, 0.1);
                if (column.tke)
                    for (std::size_t level = 0; level < levels; ++level)
                        workspace.tke[level] = column.tke[level];
                if (config.apply_surface_fluxes)
                    work[column_index].fluxes = surface_fluxes::compute_surface_fluxes(config,
                                                                                       surface,
                                                                                       column.u_sfc,
                                                                                       column.v_sfc,
                                                                                       column.theta_sfc,
                                                                                       column.qv_sfc,
                                                                                       column.theta[0],
                                                                                       column.qv[0],
                                                                                       column.p[0],
                                                                                       column.rho[0],
                                                                                       column.z_sfc,
                                                                                       column.surface_slope_x,
                                                                                       column.surface_slope_y);
                work[column_index].pbl_height = diagnose_pbl_height(column, workspace.tke, config);
                workspace.mixing_length.resize(levels);
                compute_mixing_length(column, work[column_index].pbl_height, workspace.mixing_length);
                workspace.K_m.assign(levels + 1, 0.0);
                workspace.K_h.assign(levels + 1, 0.0);
                workspace.K_e.assign(levels + 1, 0.0);

                for (std::size_t face = 1; face < levels; ++face)
                {
                    const std::size_t lower           = face - 1;
                    const std::size_t upper           = face;
                    const auto gradients = mynn::interface_gradients(column, face);
                    const double lower_weight = gradients.lower_weight;
                    const double upper_weight = gradients.upper_weight;
                    const std::size_t index = column_index * faces_per_column + lower;
                    gm[index] = static_cast<float>(gradients.shear_s2);
                    gh[index] = static_cast<float>(gradients.buoyancy_s2);
                    q_squared[index] =
                        static_cast<float>(2.0 * (lower_weight * std::max(workspace.tke[lower], 1.0e-8) +
                                                  upper_weight * std::max(workspace.tke[upper], 1.0e-8)));
                    length[index] = static_cast<float>(std::max(lower_weight * workspace.mixing_length[lower] +
                                                                    upper_weight * workspace.mixing_length[upper],
                                                                1.0));
                }
            }
            catch (...)
            {
                failures[column_index] = std::current_exception();
            }
        }
        rethrow_first(failures);

        std::vector<float> sm(face_count);
        std::vector<float> sh(face_count);
        std::vector<float> gradient_richardson(face_count);
        std::vector<float> flux_richardson(face_count);
        std::vector<float> equilibrium_limiter(face_count);
        if (face_count > 0)
        {
            const MynnLevel25StabilityDispatch dispatch{gm.data(),
                                                        gh.data(),
                                                        length.data(),
                                                        q_squared.data(),
                                                        sm.data(),
                                                        sh.data(),
                                                        gradient_richardson.data(),
                                                        flux_richardson.data(),
                                                        equilibrium_limiter.data(),
                                                        static_cast<int>(face_count)};
            if (!kernels.dispatch_mynn_level25_stability(dispatch))
                return false;
        }

        std::vector<BoundaryLayerTendencies>  candidate_tendencies(column_count);
        std::vector<BoundaryLayerDiagnostics> candidate_diagnostics(column_count);
        std::fill(failures.begin(), failures.end(), std::exception_ptr{});
#pragma omp parallel for if (column_count >= 8) schedule(static)
        for (std::size_t column_index = 0; column_index < column_count; ++column_index)
        {
            try
            {
                const auto& column    = columns[column_index];
                auto&       workspace = *column.workspace;
                for (std::size_t face = 1; face < levels; ++face)
                {
                    const std::size_t index = column_index * faces_per_column + face - 1;
                    const double      length_times_q =
                        static_cast<double>(length[index]) * std::sqrt(static_cast<double>(q_squared[index]));
                    workspace.K_m[face] = std::clamp(length_times_q * static_cast<double>(sm[index]), 0.0, 1000.0);
                    workspace.K_h[face] = std::clamp(length_times_q * static_cast<double>(sh[index]), 0.0, 1000.0);
                    workspace.K_e[face] = workspace.K_h[face];
                }
                auto& candidate = candidate_tendencies[column_index];
                candidate.dtkedt_pbl.assign(levels, 0.0);
                candidate.dqcdt_pbl.assign(levels, 0.0);
                candidate.dqidt_pbl.assign(levels, 0.0);
                update_tke(column,
                           workspace.tke,
                           workspace.mixing_length,
                           work[column_index].fluxes,
                           workspace.K_m,
                           workspace.K_h,
                           workspace.K_e,
                           config.dt_pbl,
                           candidate.dtkedt_pbl);
                apply_diffusion_tendencies(config.dt_pbl, column, workspace.K_m, workspace.K_h, work[column_index].fluxes, candidate);

                auto& diagnostic                        = candidate_diagnostics[column_index];
                diagnostic.include_diffusivity_profiles = diagnostics[column_index].include_diffusivity_profiles;
                diagnostic.pbl_height                   = work[column_index].pbl_height;
                diagnostic.ustar                        = work[column_index].fluxes.ustar;
                diagnostic.tau_u                        = work[column_index].fluxes.tau_u;
                diagnostic.tau_v                        = work[column_index].fluxes.tau_v;
                diagnostic.H                            = work[column_index].fluxes.H;
                diagnostic.E                            = work[column_index].fluxes.E;
                diagnostic.Cd                           = work[column_index].fluxes.Cd;
                diagnostic.Ch                           = work[column_index].fluxes.Ch;
                diagnostic.Ce                           = work[column_index].fluxes.Ce;
                if (diagnostic.include_diffusivity_profiles)
                {
                    diagnostic.K_m = workspace.K_m;
                    diagnostic.K_h = workspace.K_h;
                    diagnostic.K_e = workspace.K_e;
                }
            }
            catch (...)
            {
                failures[column_index] = std::current_exception();
            }
        }
        rethrow_first(failures);
        for (std::size_t column = 0; column < column_count; ++column)
        {
            tendencies[column]  = std::move(candidate_tendencies[column]);
            diagnostics[column] = std::move(candidate_diagnostics[column]);
        }
        return true;
    }
    catch (const std::bad_alloc&)
    {
        return false;
    }
    catch (const std::length_error&)
    {
        return false;
    }
}

} // namespace tmv
