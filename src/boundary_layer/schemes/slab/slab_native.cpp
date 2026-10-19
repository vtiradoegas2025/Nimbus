#include "slab.hpp"
#include "compute/dispatch/boundary_layer.hpp"
#include "compute/slab_column_contract.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <unordered_set>
namespace tmv
{
namespace
{
void pack_slab(double value, float& high, float& low)
{
    high = static_cast<float>(value);
    low  = static_cast<float>(value - high);
    if (!std::isfinite(high) || !std::isfinite(low) || (value != 0 && high == 0))
        throw std::overflow_error("native slab input exceeds compensated storage");
}
void validate_slab_column(const BoundaryLayerColumnStateView& c, std::size_t levels)
{
    if (!c.slab_state || c.theta.size() != levels || c.qv.size() != levels || c.u.size() != levels ||
        c.v.size() != levels || c.p.size() != levels || c.rho.size() != levels || c.z_int.size() != levels + 1)
        throw std::invalid_argument("native slab columns require matching profiles and private slab state");
    for (std::size_t k = 0; k < levels; ++k)
        if (!std::isfinite(c.theta[k]) || c.theta[k] <= 0 || !std::isfinite(c.qv[k]) || c.qv[k] < 0 || c.qv[k] > .1 ||
            !std::isfinite(c.u[k]) || !std::isfinite(c.v[k]) || !std::isfinite(c.p[k]) || c.p[k] <= 0 ||
            !std::isfinite(c.rho[k]) || c.rho[k] <= 0 || !std::isfinite(c.z_int[k]) || !std::isfinite(c.z_int[k + 1]) ||
            c.z_int[k + 1] <= c.z_int[k])
            throw std::invalid_argument("native slab column has invalid physical state");
}
} // namespace
bool SlabScheme::supports_native_column_batch(const BoundaryLayerConfig&         cfg,
                                              const BoundaryLayerComputeKernels& kernels) const
{
    return (cfg.pbl_top_method == "ri" || cfg.pbl_top_method == "theta") &&
           (cfg.surface_layer_id == "bulk" || cfg.surface_layer_id == "monin_obukhov") &&
           kernels.supports(BoundaryLayerKernelOp::SlabColumn);
}
bool SlabScheme::compute_columns_native(const BoundaryLayerConfig&          cfg,
                                        const SurfaceConfig&                sfc,
                                        const BoundaryLayerColumnStateView* columns,
                                        std::size_t                         count,
                                        BoundaryLayerTendencies*            tendencies,
                                        BoundaryLayerDiagnostics*           diagnostics,
                                        BoundaryLayerComputeKernels&        kernels)
{
    if (!columns || !tendencies || count == 0 || !supports_native_column_batch(cfg, kernels))
        return false;
    const auto levels = columns[0].theta.size();
    if (levels == 0 || levels > 1000000 ||
        count > std::size_t(std::numeric_limits<int>::max()) / (46 + 14 * (levels + 1)))
        return false;
    if (!std::isfinite(cfg.min_ustar) || cfg.min_ustar < 0 || !std::isfinite(min_h_) || min_h_ <= 0 ||
        max_h_ < min_h_ || !std::isfinite(max_h_))
        throw std::invalid_argument("native slab height and surface policies are invalid");
    std::unordered_set<SlabColumnState*> owners;
    for (std::size_t c = 0; c < count; ++c)
    {
        validate_slab_column(columns[c], levels);
        if (!owners.insert(columns[c].slab_state).second)
            throw std::invalid_argument("native slab batch aliases column owners");
    }
    SlabColumnDispatch d;
    auto&              p = d.constants;
    p.columns            = int(count);
    p.layers             = int(levels);
    p.top_method         = cfg.pbl_top_method == "theta";
    p.surface_method     = cfg.surface_layer_id == "monin_obukhov";
    p.apply_flux         = cfg.apply_surface_fluxes;
    std::vector<float> input(nimbus::compute::slab_column::input_count(d)),
        output(nimbus::compute::slab_column::output_count(d));
    for (std::size_t c = 0; c < count; ++c)
    {
        const auto&                  col    = columns[c];
        float*                       target = input.data() + c * nimbus::compute::slab_column::input_stride(d);
        const std::array<double, 23> parameters{min_h_,
                                                max_h_,
                                                col.u_sfc,
                                                col.v_sfc,
                                                col.theta_sfc,
                                                col.qv_sfc,
                                                col.z_sfc,
                                                col.surface_slope_x,
                                                col.surface_slope_y,
                                                sfc.z0m,
                                                sfc.z0h,
                                                cfg.min_ustar,
                                                physical_constants::von_karman_constant,
                                                physical_constants::gravity_ms2,
                                                physical_constants::specific_heat_cp_jkgk,
                                                physical_constants::reference_pressure_pa,
                                                physical_constants::gas_constant_dry_air_jkgk /
                                                    physical_constants::specific_heat_cp_jkgk,
                                                .61,
                                                1.0 / 300,
                                                1.5,
                                                .01,
                                                physical_constants::pi,
                                                200};
        for (std::size_t f = 0; f < parameters.size(); ++f)
            pack_slab(parameters[f], target[2 * f], target[2 * f + 1]);
        for (std::size_t k = 0; k <= levels; ++k)
        {
            auto* r = target + 46 + 14 * k;
            pack_slab(col.z_int[k], r[12], r[13]);
            if (k == levels)
                continue;
            const std::array<double, 6> values{col.theta[k], col.qv[k], col.u[k], col.v[k], col.p[k], col.rho[k]};
            for (std::size_t f = 0; f < values.size(); ++f)
                pack_slab(values[f], r[2 * f], r[2 * f + 1]);
        }
    }
    d.input  = input.data();
    d.output = output.data();
    if (!nimbus::compute::slab_column::validate(d))
        throw std::invalid_argument("native slab dispatch input contract failed");
    if (!kernels.dispatch_slab_column(d))
        return false;
    if (!nimbus::compute::slab_column::finite_candidate(output.data(), d))
        throw std::runtime_error("native slab returned invalid candidates");
    std::vector<BoundaryLayerTendencies>  next(count);
    std::vector<BoundaryLayerDiagnostics> diag(count);
    std::vector<SlabColumnState>          slab(count);
    for (std::size_t c = 0; c < count; ++c)
    {
        const auto read = [&](std::size_t offset)
        {
            const auto n = c * nimbus::compute::slab_column::output_stride(d) + offset;
            return double(output[n]) + output[n + 1];
        };
        slab[c] = {read(0), read(2), read(4), true};
        auto& t = next[c];
        for (auto* values :
             {&t.dudt_pbl, &t.dvdt_pbl, &t.dthetadt_pbl, &t.dqvdt_pbl, &t.dqcdt_pbl, &t.dqidt_pbl, &t.dtkedt_pbl})
            values->assign(levels, 0);
        for (std::size_t k = 0; k < levels; ++k)
        {
            t.dudt_pbl[k]     = read(22 + 8 * k);
            t.dvdt_pbl[k]     = read(24 + 8 * k);
            t.dthetadt_pbl[k] = read(26 + 8 * k);
            t.dqvdt_pbl[k]    = read(28 + 8 * k);
        }
        if (diagnostics)
        {
            auto& f      = diag[c];
            f            = diagnostics[c];
            f.pbl_height = slab[c].h_m;
            f.ustar      = read(6);
            f.tau_u      = read(8);
            f.tau_v      = read(10);
            f.H          = read(12);
            f.E          = read(14);
            f.Cd         = read(16);
            f.Ch         = read(18);
            f.Ce         = read(20);
            if (f.include_diffusivity_profiles)
            {
                f.K_m.assign(levels + 1, 0);
                f.K_h.assign(levels + 1, 0);
                f.K_e.assign(levels + 1, 0);
            }
        }
    }
    for (std::size_t c = 0; c < count; ++c)
    {
        tendencies[c]          = std::move(next[c]);
        *columns[c].slab_state = slab[c];
        if (diagnostics)
            diagnostics[c] = std::move(diag[c]);
    }
    return true;
}
} // namespace tmv
