#include "microphysics/prognostic_transport.hpp"
#include "numerics/advection/advection.hpp"
#include "numerics/advection/advection_cartesian.hpp"
#include "numerics/advection/advection_perf.hpp"

#include "compute/compute_backend.hpp"
#include "compute/compute_kernel_template.hpp"
#include "compute/parity_check.hpp"
#include "core/infra/coordinate_system.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "core/runtime/runtime_config.hpp"
#include "core/runtime/process_state.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/field/field_bounds.hpp"
#include "diagnostics/diagnostic_cell_volume.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "microphysics/microphysics_base.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <string>
#include <vector>

namespace
{

// One pass discovers every active hydrometeor species.
void scan_hydrometeor_activity(
    tmv::AdvectionRuntimeState& runtime,
    const std::array<const Field3D*, tmv::AdvectionRuntimeState::hydrometeor_count>& fields)
{
    std::array<bool, tmv::AdvectionRuntimeState::hydrometeor_count> active{};
    std::size_t max_count = 0;
    for (const Field3D* field : fields)
        max_count = std::max(max_count, field != nullptr ? field->size() : 0U);

    for (std::size_t idx = 0; idx < max_count; ++idx)
    {
        bool complete = true;
        for (std::size_t species = 0; species < fields.size(); ++species)
        {
            if (active[species])
                continue;
            complete = false;
            const Field3D* field = fields[species];
            if (field != nullptr && idx < field->size()
                && field->data()[idx] != 0.0f)
            {
                active[species] = true;
            }
        }
        if (complete)
            break;
    }
    runtime.publish_hydrometeor_activity(
        active, runtime.nr, runtime.nth, runtime.nz);
}


bool active_backend_is_cartesian_cgrid(const SimulationStateView& execution_state)
{
    return execution_state.configuration->coordinate_system == CoordinateSystem::Cartesian
        && execution_state.configuration->stagger_type == StaggerType::CGrid;
}

bool advection_disabled(const tmv::AdvectionRuntimeState& runtime)
{
    std::string scheme = runtime.config.scheme_id;
    std::transform(scheme.begin(), scheme.end(), scheme.begin(),
                   [](unsigned char ch)
                   {
                       return static_cast<char>(std::tolower(ch));
                   });
    return scheme == "none";
}

tmv::CartesianFaceReconstruction conservative_reconstruction(
    const tmv::AdvectionRuntimeState& runtime)
{
    std::string scheme = runtime.config.scheme_id;
    std::transform(scheme.begin(), scheme.end(), scheme.begin(),
                   [](unsigned char ch)
                   {
                       return static_cast<char>(std::tolower(ch));
                   });
    if (scheme == "tvd")
        return tmv::CartesianFaceReconstruction::MonotonicCentered;
    if (scheme == "weno5")
        return tmv::CartesianFaceReconstruction::WENO5;
    throw std::invalid_argument(
        "Cartesian conservative advection requires scheme tvd or weno5");
}

void require_transport_field(const SimulationStateView& execution_state, const Field3D& field, const char* label,
                             float lower, float upper)
{
    if (field.size_r() != execution_state.grid.nr || field.size_th() != execution_state.grid.nth ||
        field.size_z() != execution_state.grid.nz)
    {
        throw std::invalid_argument(
            std::string("thermodynamic advection ") + label +
            " shape does not match the grid");
    }
    for (std::size_t index = 0; index < field.size(); ++index)
    {
        const float value = field.data()[index];
        if (!std::isfinite(static_cast<double>(value)) ||
            value < lower || value > upper)
        {
            const int k = static_cast<int>(
                index % static_cast<std::size_t>(execution_state.grid.nz));
            const std::size_t ij =
                index / static_cast<std::size_t>(execution_state.grid.nz);
            const int j = static_cast<int>(
                ij % static_cast<std::size_t>(execution_state.grid.nth));
            const int i = static_cast<int>(
                ij / static_cast<std::size_t>(execution_state.grid.nth));
            std::ostringstream message;
            message << std::setprecision(
                           std::numeric_limits<float>::max_digits10)
                    << "thermodynamic advection requires finite, in-range "
                    << label << " values throughout the transport stencil: "
                    << "value=" << value << " bounds=[" << lower << ','
                    << upper << "] at (" << i << ',' << j << ',' << k << ')';
            throw std::invalid_argument(message.str());
        }
    }
}

void require_finite_transport_field(const SimulationStateView& execution_state, const Field3D& field, const char* label)
{
    require_transport_field(execution_state,
        field, label,
        -std::numeric_limits<float>::max(),
        std::numeric_limits<float>::max());
}

std::uint32_t binary32_bits(float value) noexcept
{
    static_assert(sizeof(float) == sizeof(std::uint32_t),
                  "passive-tracer canonicalization requires binary32 storage");
    static_assert(std::numeric_limits<float>::is_iec559
                      && std::numeric_limits<float>::digits == 24,
                  "passive-tracer canonicalization requires IEEE-754 binary32");
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool is_binary32_subnormal(std::uint32_t bits) noexcept
{
    constexpr std::uint32_t exponent_mask = 0x7f800000U;
    constexpr std::uint32_t fraction_mask = 0x007fffffU;
    return (bits & exponent_mask) == 0U
        && (bits & fraction_mask) != 0U;
}

bool is_negative_binary32_zero(std::uint32_t bits) noexcept
{
    return bits == 0x80000000U;
}

double exact_binary32_subnormal_value(std::uint32_t bits) noexcept
{
    constexpr std::uint32_t sign_mask = 0x80000000U;
    constexpr std::uint32_t fraction_mask = 0x007fffffU;
    const double magnitude = std::ldexp(
        static_cast<double>(bits & fraction_mask), -149);
    return (bits & sign_mask) != 0U ? -magnitude : magnitude;
}

bool valid_tracer_roundoff_report(
    const tmv::PositiveDefiniteScalarRoundoffReport& report) noexcept
{
    if (!std::isfinite(report.physical_inventory_correction_kg)
        || report.physical_inventory_correction_kg < 0.0
        || report.thermodynamic_energy_correction_j != 0.0
        || report.negative_zero_count
            > std::numeric_limits<std::size_t>::max()
                - report.negative_subnormal_count)
    {
        return false;
    }
    const std::size_t canonicalization_count =
        report.negative_subnormal_count + report.negative_zero_count;
    if (report.closure_canonicalization_count > canonicalization_count
        || (report.negative_subnormal_count == 0
            && report.physical_inventory_correction_kg != 0.0))
    {
        return false;
    }
    const std::uint32_t minimum_bits = binary32_bits(
        report.most_negative_repaired_value);
    const bool minimum_is_negative_subnormal =
        is_binary32_subnormal(minimum_bits)
        && (minimum_bits & 0x80000000U) != 0U;
    return report.negative_subnormal_count == 0
        ? minimum_bits == 0U
        : minimum_is_negative_subnormal;
}

bool zero_incomplete_tracer_roundoff_report(
    const tmv::PositiveDefiniteScalarRoundoffReport& report) noexcept
{
    return report.physical_inventory_correction_kg == 0.0
        && report.thermodynamic_energy_correction_j == 0.0
        && binary32_bits(report.most_negative_repaired_value) == 0U
        && report.negative_subnormal_count == 0
        && report.negative_zero_count == 0
        && report.closure_canonicalization_count == 0
        && !report.complete;
}

bool valid_tracer_roundoff_runtime_for_producer(
    const tmv::AdvectionRuntimeState& runtime) noexcept
{
    const auto& last = runtime.tracer_roundoff_report;
    const auto& cumulative = runtime.cumulative_tracer_roundoff_report;
    if (!valid_tracer_roundoff_report(last)
        || !valid_tracer_roundoff_report(cumulative))
    {
        return false;
    }
    if (runtime.tracer_roundoff_report_step_count == 0)
    {
        return zero_incomplete_tracer_roundoff_report(last)
            && zero_incomplete_tracer_roundoff_report(cumulative);
    }
    // reset_for_shape deliberately clears only the last-step report. Accept
    // that idle state while retaining strict cumulative-history validation.
    if ((!last.complete
         && !zero_incomplete_tracer_roundoff_report(last))
        || cumulative.complete
            != runtime.tracer_roundoff_cumulative_history_known
        || cumulative.negative_subnormal_count
            < last.negative_subnormal_count
        || cumulative.negative_zero_count < last.negative_zero_count
        || cumulative.closure_canonicalization_count
            < last.closure_canonicalization_count
        || cumulative.physical_inventory_correction_kg
            < last.physical_inventory_correction_kg)
    {
        return false;
    }
    const std::uint32_t last_fraction = binary32_bits(
        last.most_negative_repaired_value) & 0x007fffffU;
    const std::uint32_t cumulative_fraction = binary32_bits(
        cumulative.most_negative_repaired_value) & 0x007fffffU;
    return cumulative_fraction >= last_fraction;
}

bool is_physical_cell(const PhysicalCellDomain& domain,
                      int i, int j, int k) noexcept
{
    return i >= domain.i_begin && i < domain.i_end
        && j >= domain.j_begin && j < domain.j_end
        && k >= domain.k_begin && k < domain.k_end;
}

tmv::PositiveDefiniteScalarRoundoffReport
canonicalize_passive_tracer_roundoff(
    const SimulationStateView& simulation,
    tmv::AdvectionRuntimeState& runtime)
{
    if (!valid_tracer_roundoff_runtime_for_producer(runtime))
    {
        throw std::runtime_error(
            "passive-tracer roundoff transaction rejected malformed "
            "caller-owned ledger state");
    }
    tmv::PositiveDefiniteScalarRoundoffReport report;
    const Field3D& density = simulation.fields.rho;
    Field3D& marker = simulation.fields.tracer;
    const int nr = simulation.grid.nr;
    const int nth = simulation.grid.nth;
    const int nz = simulation.grid.nz;
    if (marker.size_r() != nr || marker.size_th() != nth
        || marker.size_z() != nz || density.size_r() != nr
        || density.size_th() != nth || density.size_z() != nz)
    {
        throw std::invalid_argument(
            "passive-tracer roundoff transaction shape does not match the grid");
    }
    tmv::physical_grid_metrics::validate(simulation);
    const PhysicalCellDomain physical = physical_scalar_domain(
        nr, nth, nz, simulation.configuration->stagger_type);
    if (physical.empty())
    {
        throw std::invalid_argument(
            "passive-tracer roundoff transaction has no physical cells");
    }

    // 0: physical correction; 1: negative subnormals; 2: negative zero;
    // 3: boundary-closure canonicalizations; 4: material tracer violations;
    // 5: invalid physical accounting state.
    const auto sums = tmv::diagnostics::fixed_slab_reduce<6>(
        0, nr, [&](int i, std::array<double, 6>& partial)
        {
            for (int j = 0; j < nth; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const std::size_t index =
                        (static_cast<std::size_t>(i)
                             * static_cast<std::size_t>(nth)
                         + static_cast<std::size_t>(j))
                            * static_cast<std::size_t>(nz)
                        + static_cast<std::size_t>(k);
                    const std::uint32_t bits = binary32_bits(
                        marker.data()[index]);
                    constexpr std::uint32_t sign_mask = 0x80000000U;
                    constexpr std::uint32_t exponent_mask = 0x7f800000U;
                    constexpr std::uint32_t magnitude_mask = 0x7fffffffU;
                    const bool physical_cell = is_physical_cell(
                        physical, i, j, k);
                    const bool subnormal = is_binary32_subnormal(bits);
                    const bool negative_zero =
                        is_negative_binary32_zero(bits);
                    const bool nonfinite =
                        (bits & exponent_mask) == exponent_mask;
                    const bool negative_normal =
                        (bits & sign_mask) != 0U
                        && (bits & magnitude_mask) != 0U
                        && !subnormal && !nonfinite;
                    const bool above_contract = !nonfinite && !subnormal
                        && (bits & sign_mask) == 0U
                        && marker.data()[index] > passive_tracer_max;
                    if (nonfinite || negative_normal || above_contract)
                    {
                        partial[4] += 1.0;
                        continue;
                    }

                    const bool negative_subnormal = subnormal
                        && (bits & sign_mask) != 0U;
                    if (!negative_subnormal && !negative_zero)
                        continue;
                    if (negative_subnormal)
                        partial[1] += 1.0;
                    else
                        partial[2] += 1.0;
                    if (!physical_cell)
                    {
                        partial[3] += 1.0;
                        continue;
                    }

                    const double rho_value = static_cast<double>(
                        density.data()[index]);
                    const double volume = diagnostic_cell_volume(
                        simulation, i, j, k);
                    if (!std::isfinite(rho_value) || rho_value <= 0.0
                        || !std::isfinite(volume) || volume <= 0.0)
                    {
                        partial[5] += 1.0;
                        continue;
                    }
                    if (negative_subnormal)
                    {
                        partial[0] += -exact_binary32_subnormal_value(bits)
                            * rho_value * volume;
                    }
                }
        });

    if (sums[4] != 0.0 || sums[5] != 0.0)
    {
        // Exceptional only: locate the first violation in storage order so
        // diagnostics do not depend on OpenMP scheduling. No value has been
        // changed yet, which makes this failure path transactional.
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j < nth; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    const float value = marker(i, j, k);
                    const std::uint32_t bits = binary32_bits(value);
                    constexpr std::uint32_t sign_mask = 0x80000000U;
                    constexpr std::uint32_t exponent_mask = 0x7f800000U;
                    constexpr std::uint32_t magnitude_mask = 0x7fffffffU;
                    const bool subnormal = is_binary32_subnormal(bits);
                    const bool nonfinite =
                        (bits & exponent_mask) == exponent_mask;
                    const bool negative_normal =
                        (bits & sign_mask) != 0U
                        && (bits & magnitude_mask) != 0U
                        && !subnormal && !nonfinite;
                    const bool above_contract = !nonfinite && !subnormal
                        && (bits & sign_mask) == 0U
                        && value > passive_tracer_max;
                    if (nonfinite || negative_normal || above_contract)
                    {
                        std::ostringstream message;
                        message << std::setprecision(
                                       std::numeric_limits<float>::max_digits10)
                                << "passive-tracer transport produced a material "
                                   "contract violation: value=" << value
                                << " bounds=[0," << passive_tracer_max
                                << "] at (" << i << ',' << j << ',' << k
                                << ")";
                        throw std::runtime_error(message.str());
                    }
                    if (is_physical_cell(physical, i, j, k))
                    {
                        const double rho_value = density(i, j, k);
                        const double volume = diagnostic_cell_volume(
                            simulation, i, j, k);
                        if (!std::isfinite(rho_value) || rho_value <= 0.0
                            || !std::isfinite(volume) || volume <= 0.0)
                        {
                            throw std::runtime_error(
                                "passive-tracer roundoff ledger rejected "
                                "nonphysical density or cell volume");
                        }
                    }
                }
        throw std::runtime_error(
            "passive-tracer roundoff transaction rejected invalid state");
    }

    const auto exact_count = [](double value, const char* label)
    {
        if (!std::isfinite(value) || value < 0.0
            || value > static_cast<double>(
                std::numeric_limits<std::size_t>::max())
            || value != std::floor(value))
        {
            throw std::overflow_error(
                std::string("passive-tracer roundoff ") + label
                + " is not an exact count");
        }
        return static_cast<std::size_t>(value);
    };
    if (!std::isfinite(sums[0]))
    {
        throw std::overflow_error(
            "passive-tracer physical roundoff correction is non-finite");
    }
    report.physical_inventory_correction_kg = sums[0];
    report.negative_subnormal_count = exact_count(
        sums[1], "negative-subnormal count");
    report.negative_zero_count = exact_count(
        sums[2], "negative-zero count");
    report.closure_canonicalization_count = exact_count(
        sums[3], "boundary-closure count");

    const std::size_t canonicalization_count =
        report.negative_subnormal_count
        + report.negative_zero_count;
    if (canonicalization_count != 0)
    {
        std::uint32_t largest_negative_subnormal_fraction = 0U;
        for (std::size_t index = 0; index < marker.size(); ++index)
        {
            const std::uint32_t bits = binary32_bits(marker.data()[index]);
            if (is_binary32_subnormal(bits)
                && (bits & 0x80000000U) != 0U)
            {
                largest_negative_subnormal_fraction = std::max(
                    largest_negative_subnormal_fraction,
                    bits & 0x007fffffU);
            }
        }
        if (largest_negative_subnormal_fraction != 0U)
        {
            const std::uint32_t bits = 0x80000000U
                | largest_negative_subnormal_fraction;
            std::memcpy(&report.most_negative_repaired_value,
                        &bits, sizeof(bits));
        }
    }
    report.complete = true;

    if (runtime.tracer_roundoff_report_step_count
        == std::numeric_limits<std::uint64_t>::max())
    {
        throw std::overflow_error(
            "passive-tracer cumulative roundoff step counter overflow");
    }
    auto cumulative = runtime.cumulative_tracer_roundoff_report;
    const auto checked_add_count = [](std::size_t left, std::size_t right,
                                      const char* label)
    {
        if (right > std::numeric_limits<std::size_t>::max() - left)
        {
            throw std::overflow_error(
                std::string("passive-tracer cumulative ") + label
                + " overflow");
        }
        return left + right;
    };
    cumulative.physical_inventory_correction_kg +=
        report.physical_inventory_correction_kg;
    if (!std::isfinite(cumulative.physical_inventory_correction_kg))
    {
        throw std::overflow_error(
            "passive-tracer cumulative physical correction is non-finite");
    }
    cumulative.thermodynamic_energy_correction_j = 0.0;
    cumulative.negative_subnormal_count = checked_add_count(
        cumulative.negative_subnormal_count,
        report.negative_subnormal_count, "negative-subnormal count");
    cumulative.negative_zero_count = checked_add_count(
        cumulative.negative_zero_count,
        report.negative_zero_count, "negative-zero count");
    cumulative.closure_canonicalization_count = checked_add_count(
        cumulative.closure_canonicalization_count,
        report.closure_canonicalization_count,
        "boundary-closure count");
    if (binary32_bits(report.most_negative_repaired_value)
            > binary32_bits(cumulative.most_negative_repaired_value))
    {
        cumulative.most_negative_repaired_value =
            report.most_negative_repaired_value;
    }
    cumulative.complete =
        runtime.tracer_roundoff_cumulative_history_known;

    // All classification, physical accounting, and cumulative overflow checks
    // are complete. Only this block mutates the field or publishes reports.
    if (canonicalization_count != 0)
    {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j < nth; ++j)
                for (int k = 0; k < nz; ++k)
                {
                    float& value = marker(i, j, k);
                    const std::uint32_t bits = binary32_bits(value);
                    if ((is_binary32_subnormal(bits)
                         && (bits & 0x80000000U) != 0U)
                        || is_negative_binary32_zero(bits))
                    {
                        value = 0.0f;
                    }
                }
        invalidate_resident_field_backend(tmv::require_level_runtime(simulation.compute_runtime, "compute runtime"), "tracer");
    }
    runtime.tracer_roundoff_report = report;
    runtime.cumulative_tracer_roundoff_report = cumulative;
    ++runtime.tracer_roundoff_report_step_count;
    return report;
}

} // namespace

namespace tmv
{

void advect_thermodynamics_3d(
    double dt_advect, double kappa_theta, double kappa_moisture,
    const CartesianCGridTransportContext* context,
    AdvectionRuntimeState* runtime_state)
{
    return advect_thermodynamics_3d(global_simulation_state(), dt_advect, kappa_theta, kappa_moisture, context, runtime_state);
}

static void advect_thermodynamics_impl(const SimulationStateView&            execution_state,
                                       double                                dt_advect,
                                       double                                kappa_theta,
                                       double                                kappa_moisture,
                                       const CartesianCGridTransportContext* context,
                                       AdvectionRuntimeState*                runtime_state)
{
    const int nr = execution_state.grid.nr;
    const int nth = execution_state.grid.nth;
    const int nz = execution_state.grid.nz;
    if (nr <= 0 || nth <= 0 || nz <= 0
        || !std::isfinite(dt_advect) || dt_advect <= 0.0
        || !std::isfinite(kappa_theta) || kappa_theta != 0.0
        || !std::isfinite(kappa_moisture) || kappa_moisture != 0.0
        || !std::isfinite(execution_state.grid.dr) || execution_state.grid.dr <= 0.0
        || !std::isfinite(execution_state.grid.dtheta) || execution_state.grid.dtheta <= 0.0
        || !std::isfinite(execution_state.grid.dz) || execution_state.grid.dz <= 0.0)
    {
        throw std::invalid_argument(
            "thermodynamic advection requires a non-empty grid, positive "
            "finite timestep/spacing, and kappa_theta=kappa_moisture=0; "
            "configure the live-density diffusion scheme instead");
    }

    // These fields feed face reconstructions, including the boundary closure
    // planes. Validate the complete stencil before activity discovery or the
    // first in-place scalar update. A density floor or a NaN-to-zero repair
    // here would silently change the transported equation.
    require_transport_field(execution_state,
        execution_state.fields.rho, "density", density_min_kgm3, density_max_kgm3);
    require_finite_transport_field(execution_state, execution_state.fields.u, "x velocity");
    require_finite_transport_field(execution_state, execution_state.fields.v, "y velocity");
    require_finite_transport_field(execution_state, execution_state.fields.w, "vertical velocity");
    require_transport_field(execution_state, execution_state.fields.theta, "potential temperature",
                            theta_min_k, theta_max_k);
    require_transport_field(execution_state, execution_state.fields.qv, "water vapor", 0.0f, qv_max_kgkg);
    require_transport_field(execution_state, execution_state.fields.qc, "cloud water", 0.0f, hydrometeor_max_kgkg);
    require_transport_field(execution_state, execution_state.fields.qr, "rain water", 0.0f, hydrometeor_max_kgkg);
    require_transport_field(execution_state, execution_state.fields.qi, "cloud ice", 0.0f, hydrometeor_max_kgkg);
    require_transport_field(execution_state, execution_state.fields.qs, "snow", 0.0f, hydrometeor_max_kgkg);
    require_transport_field(execution_state, execution_state.fields.qh, "hail", 0.0f, hydrometeor_max_kgkg);
    require_transport_field(execution_state, execution_state.fields.qg, "graupel", 0.0f, hydrometeor_max_kgkg);
    if (!execution_state.fields.tracer.empty())
    {
        require_transport_field(execution_state,
            execution_state.fields.tracer, "passive tracer", 0.0f, passive_tracer_max);
    }

    const bool perf_on = advection_detail::perf_enabled();
    const auto t0 = perf_on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    AdvectionRuntimeState& runtime = runtime_state != nullptr
        ? *runtime_state : (*execution_state.advection_runtime);
    runtime.reset_for_shape(nr, nth, nz);
    runtime.thermodynamic_boundary_reports = {};
    runtime.thermodynamic_boundary_reports_complete = false;
    runtime.thermodynamic_moist_energy_reports = {};
    runtime.thermodynamic_moist_energy_reports_complete = false;
    const bool boundary_reports_requested =
        context != nullptr
        && runtime.thermodynamic_boundary_reports_requested;
    const bool energy_reports_requested =
        context != nullptr
        && runtime.thermodynamic_moist_energy_reports_requested;
    if (advection_disabled(runtime))
    {
        if (boundary_reports_requested)
            for (auto& report : runtime.thermodynamic_boundary_reports)
                report.complete = true;
        if (energy_reports_requested)
            for (auto& report : runtime.thermodynamic_moist_energy_reports)
                report.complete = true;
        runtime.thermodynamic_boundary_reports_complete =
            boundary_reports_requested;
        runtime.thermodynamic_moist_energy_reports_complete =
            energy_reports_requested;
        runtime.tracer_advected_in_batch = false;
        runtime.invalidate_hydrometeor_activity();
        return;
    }
    const CartesianFaceReconstruction reconstruction =
        conservative_reconstruction(runtime);
    if (runtime.activity_current)
    {
        ++runtime.metadata_consumptions;
    }
    else
    {
        scan_hydrometeor_activity(
            runtime, {&execution_state.fields.qc, &execution_state.fields.qr, &execution_state.fields.qi, &execution_state.fields.qs, &execution_state.fields.qh, &execution_state.fields.qg});
        ++runtime.fallback_scans;
    }
    const bool qc_active = runtime.hydrometeor_active[0];
    const bool qr_active = runtime.hydrometeor_active[1];
    const bool qi_active = runtime.hydrometeor_active[2];
    const bool qs_active = runtime.hydrometeor_active[3];
    const bool qh_active = runtime.hydrometeor_active[4];
    const bool qg_active = runtime.hydrometeor_active[5];

    // Backend batch (GPU audit Phase 3.2, CPU-015): on the Cartesian
    // C-grid, advect every active scalar through one shared operation. Vulkan
    // uses one device submission; CPU consumes host-native fields and the
    // retained context without packing. Theta/moisture guards remain shared.
    bool gpu_advected = false;
    const bool interface_flux_capture = runtime.theta_interface_flux_capture_enabled || runtime.moist_interface.enabled;
    const bool strict_native =
        (execution_state.configuration &&
         (execution_state.configuration->admission.native_weno5 ||
          execution_state.configuration->admission.native_dry_tvd)) ||
        (interface_flux_capture &&
         active_compute_backend_kind(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime")) !=
             ComputeBackendKind::Cpu);
    const bool mapped_device_batch = strict_native && context && context->terrain_metrics() &&
        execution_state.configuration->admission.two_way_accelerator_reference;
    const bool backend_can_batch = active_compute_backend_supports(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime"),
        AdvectionKernelOp::ConservativeCartesianCgridTransport);
    const bool backend_accepts_context = context == nullptr
        || active_compute_backend_supports_supplied_cartesian_cgrid_mass_flux(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime"));
    // Keep report-bearing CPU transport on the direct host batch: it avoids a
    // registry round trip while retaining the same optimized kernel. Device
    // backends consume the report ABI below and close each ledger before its
    // corresponding host field is committed.
    const bool direct_host_science_reporting =
        (boundary_reports_requested || energy_reports_requested) &&
        active_compute_backend_kind(tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime")) ==
            ComputeBackendKind::Cpu;
    if ((reconstruction == CartesianFaceReconstruction::WENO5 ||
         ((interface_flux_capture ||
           (execution_state.configuration && execution_state.configuration->admission.native_dry_tvd)) &&
          reconstruction == CartesianFaceReconstruction::MonotonicCentered))
        && active_backend_is_cartesian_cgrid(execution_state)
        && !(context && context->terrain_metrics())
        && backend_can_batch
        && backend_accepts_context
        && !direct_host_science_reporting)
    {
        if (context != nullptr)
        {
            context->require_valid(execution_state, nr, nth, nz, dt_advect);
        }
        constexpr int kMaxScalars = 9;
        Field3D* fields[kMaxScalars];
        const char* names[kMaxScalars];
        float dt_kappas[kMaxScalars];
        double kappas[kMaxScalars];
        int n = 0;
        fields[n] = &execution_state.fields.theta; names[n] = "theta";
        kappas[n] = kappa_theta;
        dt_kappas[n] = static_cast<float>(dt_advect * kappa_theta); ++n;
        fields[n] = &execution_state.fields.qv; names[n] = "qv";
        kappas[n] = kappa_moisture;
        dt_kappas[n] = static_cast<float>(dt_advect * kappa_moisture); ++n;
        Field3D* hydro_fields[6] = { &execution_state.fields.qc, &execution_state.fields.qr, &execution_state.fields.qi, &execution_state.fields.qs, &execution_state.fields.qh, &execution_state.fields.qg };
        const char* hydro_names[6] = { "qc", "qr", "qi", "qs", "qh", "qg" };
        const bool hydro_active[6] = { qc_active, qr_active, qi_active,
                                       qs_active, qh_active, qg_active };
        for (int h = 0; h < 6; ++h)
        {
            if (!hydro_active[h] && !runtime.moist_interface.enabled) continue;
            fields[n] = hydro_fields[h];
            names[n] = hydro_names[h];
            kappas[n] = kappa_moisture;
            dt_kappas[n] = static_cast<float>(dt_advect * kappa_moisture);
            ++n;
        }
        // The passive tracer rides the same batch (GPU audit Phase
        // 4.3c): one more scalar here is one less submission and one less
        // mass-flux computation than its standalone dispatch. Scalars are
        // independent (each reads only its own field plus u/v/w/rho), so
        // batching it with the thermodynamics is order-equivalent to the
        // separate advect_tracer call, which skips when this ran (the
        // orchestrator advects thermodynamics first).
        const bool tracer_in_batch = execution_state.fields.tracer.size() == execution_state.fields.theta.size();
        if (tracer_in_batch)
        {
            fields[n] = &execution_state.fields.tracer; names[n] = "tracer";
            kappas[n] = kTracerDefaultKappa;
            dt_kappas[n] = static_cast<float>(dt_advect * kTracerDefaultKappa);
            ++n;
        }

        const float* q_in_ptrs[kMaxScalars];
        float* q_out_ptrs[kMaxScalars];
        int pd_flags[kMaxScalars];
        ScalarBoundaryTransportReport*
            device_boundary_reports[kMaxScalars]{};
        CartesianInterfaceFluxRegister* device_interface_registers[kMaxScalars]{};
        ScalarMoistEnergyKind device_energy_kinds[kMaxScalars]{};
        ScalarMoistEnergyTransportReportV1*
            device_energy_reports[kMaxScalars]{};
        for (int s = 0; s < n; ++s)
        {
            q_in_ptrs[s] = fields[s]->data();
            q_out_ptrs[s] = fields[s]->data();
            // theta is the only scalar in the batch that is not
            // positive definite (SCIENCE.md M5).
            pd_flags[s] = (std::strcmp(names[s], "theta") != 0) ? 1 : 0;

            std::size_t report_index = thermodynamic_boundary_report_count;
            ScalarMoistEnergyKind energy_kind =
                ScalarMoistEnergyKind::None;
            if (fields[s] == &execution_state.fields.theta)
            {
                report_index = theta_boundary_report_index;
                energy_kind = ScalarMoistEnergyKind::PotentialTemperature;
            }
            else if (fields[s] == &execution_state.fields.qv)
            {
                report_index = qv_boundary_report_index;
                energy_kind = ScalarMoistEnergyKind::WaterVapor;
            }
            else
            {
                for (int h = 0; h < 6; ++h)
                {
                    if (fields[s] != hydro_fields[h])
                        continue;
                    static constexpr std::size_t kHydroReportIndices[6] = {
                        qc_boundary_report_index, qr_boundary_report_index,
                        qi_boundary_report_index, qs_boundary_report_index,
                        qh_boundary_report_index, qg_boundary_report_index,
                    };
                    report_index = kHydroReportIndices[h];
                    energy_kind = h < 2
                        ? ScalarMoistEnergyKind::LiquidWater
                        : ScalarMoistEnergyKind::IceWater;
                    break;
                }
            }
            if (report_index != thermodynamic_boundary_report_count)
            {
                if (report_index == theta_boundary_report_index && interface_flux_capture)
                    device_interface_registers[s] = &runtime.theta_interface_flux_register;
                else if (report_index != theta_boundary_report_index && runtime.moist_interface.enabled)
                    device_interface_registers[s] = &runtime.moist_interface.water[report_index - 1];
                if (boundary_reports_requested)
                {
                    device_boundary_reports[s] =
                        &runtime.thermodynamic_boundary_reports[report_index];
                }
                if (energy_reports_requested)
                {
                    device_energy_kinds[s] = energy_kind;
                    device_energy_reports[s] =
                        &runtime.thermodynamic_moist_energy_reports[report_index];
                }
            }
        }
        for (int h = 0; h < 6; ++h)
        {
            if (hydro_active[h])
                continue;
            static constexpr std::size_t kHydroReportIndices[6] = {
                qc_boundary_report_index, qr_boundary_report_index,
                qi_boundary_report_index, qs_boundary_report_index,
                qh_boundary_report_index, qg_boundary_report_index,
            };
            if (boundary_reports_requested)
            {
                runtime.thermodynamic_boundary_reports[
                    kHydroReportIndices[h]].complete = true;
            }
            if (energy_reports_requested)
            {
                runtime.thermodynamic_moist_energy_reports[
                    kHydroReportIndices[h]].complete = true;
            }
        }

        // Inverse spacings divided in fp64 and rounded once (GPU audit
        // section 2.3). Cartesian dtheta stores the physical y spacing.
        const double dx_d = execution_state.grid.dr;
        const double dy_d = execution_state.grid.dtheta;
        const double dz_d = execution_state.grid.dz;

        const bool parity_sample = nimbus::parity::sample(
            "advect_scalars_conservative_cartesian_cgrid");
        std::vector<Field3D> snapshots;
        if (parity_sample)
        {
            snapshots.reserve(static_cast<std::size_t>(n));
            for (int s = 0; s < n; ++s)
            {
                snapshots.push_back(*fields[s]);
            }
        }

        if ((primary_state_repairs_enabled || strict_native) &&
            dispatch_advect_scalars_conservative_cartesian_cgrid_backend(
                tmv::require_level_runtime(execution_state.compute_runtime, "compute runtime"),
                execution_state.fields.u.data(),
                execution_state.fields.v.data(),
                execution_state.fields.w.data(),
                execution_state.fields.rho.data(),
                q_in_ptrs,
                q_out_ptrs,
                dt_kappas,
                pd_flags,
                n,
                nr,
                nth,
                nz,
                static_cast<float>(1.0 / dx_d),
                static_cast<float>(1.0 / dy_d),
                static_cast<float>(1.0 / dz_d),
                static_cast<float>(1.0 / (dx_d * dx_d)),
                static_cast<float>(1.0 / (dy_d * dy_d)),
                static_cast<float>(1.0 / (dz_d * dz_d)),
                static_cast<float>(dt_advect),
                density_min_kgm3,
                context ? context->density_begin.data() : nullptr,
                context ? context->mass_flux_x.data() : nullptr,
                context ? context->mass_flux_y.data() : nullptr,
                context ? context->mass_flux_z.data() : nullptr,
                context != nullptr,
                fields,
                kappas,
                context,
                dt_advect,
                boundary_reports_requested ? device_boundary_reports : nullptr,
                energy_reports_requested ? device_energy_kinds : nullptr,
                energy_reports_requested ? device_energy_reports : nullptr,
                static_cast<float>((*execution_state.domain_frame).velocity_ms.u_ms),
                static_cast<float>((*execution_state.domain_frame).velocity_ms.v_ms),
                &execution_state,
                interface_flux_capture ? device_interface_registers : nullptr,
                reconstruction == CartesianFaceReconstruction::WENO5))
        {
            if (strict_native)
            {
                for (int scalar = 0; scalar < n; ++scalar)
                {
                    const bool theta  = std::strcmp(names[scalar], "theta") == 0;
                    const bool vapor  = std::strcmp(names[scalar], "qv") == 0;
                    const bool tracer = std::strcmp(names[scalar], "tracer") == 0;
                    require_transport_field(execution_state,
                                            *fields[scalar],
                                            names[scalar],
                                            theta    ? theta_min_k
                                            : tracer ? std::nextafter(-std::numeric_limits<float>::min(), 0.0f)
                                                     : 0.0f,
                                            theta    ? theta_max_k
                                            : vapor  ? qv_max_kgkg
                                            : tracer ? passive_tracer_max
                                                     : hydrometeor_max_kgkg);
                }
            }
            gpu_advected = true;
            runtime.tracer_advected_in_batch = tracer_in_batch;
            if (boundary_reports_requested)
            {
                runtime.thermodynamic_boundary_reports_complete =
                    std::all_of(
                        runtime.thermodynamic_boundary_reports.begin(),
                        runtime.thermodynamic_boundary_reports.end(),
                        [](const ScalarBoundaryTransportReport& report)
                        {
                            return report.complete;
                        });
                if (!runtime.thermodynamic_boundary_reports_complete)
                {
                    throw std::runtime_error(
                        "device thermodynamic transport did not publish every "
                        "boundary ledger");
                }
            }
            if (energy_reports_requested)
            {
                runtime.thermodynamic_moist_energy_reports_complete =
                    std::all_of(
                        runtime.thermodynamic_moist_energy_reports.begin(),
                        runtime.thermodynamic_moist_energy_reports.end(),
                        [](const ScalarMoistEnergyTransportReportV1& report)
                        {
                            return report.complete;
                        });
                if (!runtime.thermodynamic_moist_energy_reports_complete)
                {
                    throw std::runtime_error(
                        "device thermodynamic transport did not publish every "
                        "moist-energy ledger");
                }
            }
            if (parity_sample)
            {
                // Parity replay (GPU audit Phase 1): rerun each scalar
                // through the CPU production kernels from the pre-dispatch
                // snapshot; the GPU result stays production.
                Field3D ref_a;
                for (int s = 0; s < n; ++s)
                {
                    const bool pd_scalar =
                        std::strcmp(names[s], "theta") != 0;
                    advect_scalar_conservative_3d_cartesian(execution_state, snapshots[static_cast<std::size_t>(s)],
                                                            ref_a, dt_advect,
                                                            pd_scalar, context,
                                                            nullptr,
                                                            ScalarMoistEnergyKind::None,
                                                            nullptr,
                                                            reconstruction);
                    nimbus::parity::check_field(
                        "advect_scalars_conservative_cartesian_cgrid",
                        names[s],
                        ref_a.data(), fields[s]->data(),
                        nr, nth, nz, nimbus::parity::Tier::Stencil);
                }
            }
        }
    }

    if (strict_native && !gpu_advected && !mapped_device_batch)
    {
        throw std::runtime_error(
            "strict native conservative transport is unavailable or failed; CPU substitution is disabled");
    }
    bool reported_batch = false;
    if (!gpu_advected && context != nullptr)
    {
        constexpr std::size_t kMaxScalars = 9;
        CartesianScalarTransport scalars[kMaxScalars];
        std::size_t n = 0;
        scalars[n++] = {
            &execution_state.fields.theta, kappa_theta, false,
            boundary_reports_requested
                ? &runtime.thermodynamic_boundary_reports[
                      theta_boundary_report_index]
                : nullptr,
            energy_reports_requested
                ? ScalarMoistEnergyKind::PotentialTemperature
                : ScalarMoistEnergyKind::None,
            energy_reports_requested
                ? &runtime.thermodynamic_moist_energy_reports[
                      theta_boundary_report_index]
                : nullptr,
            interface_flux_capture
                ? &runtime.theta_interface_flux_register : nullptr};
        scalars[n++] = {
            &execution_state.fields.qv,
            kappa_moisture,
            true,
            boundary_reports_requested ? &runtime.thermodynamic_boundary_reports[qv_boundary_report_index] : nullptr,
            energy_reports_requested ? ScalarMoistEnergyKind::WaterVapor : ScalarMoistEnergyKind::None,
            energy_reports_requested ? &runtime.thermodynamic_moist_energy_reports[qv_boundary_report_index] : nullptr,
            runtime.moist_interface.enabled ? &runtime.moist_interface.water[0] : nullptr};

        Field3D*          hydro_fields[6]         = {&execution_state.fields.qc,
                                                     &execution_state.fields.qr,
                                                     &execution_state.fields.qi,
                                                     &execution_state.fields.qs,
                                                     &execution_state.fields.qh,
                                                     &execution_state.fields.qg};
        const std::size_t hydro_report_indices[6] = {qc_boundary_report_index,
                                                     qr_boundary_report_index,
                                                     qi_boundary_report_index,
                                                     qs_boundary_report_index,
                                                     qh_boundary_report_index,
                                                     qg_boundary_report_index};
        const bool        hydro_active[6]         = {qc_active, qr_active, qi_active, qs_active, qh_active, qg_active};
        for (int h = 0; h < 6; ++h)
        {
            if (hydro_active[h] || runtime.moist_interface.enabled)
            {
                scalars[n++] = {
                    hydro_fields[h],
                    kappa_moisture,
                    true,
                    boundary_reports_requested ? &runtime.thermodynamic_boundary_reports[hydro_report_indices[h]]
                                               : nullptr,
                    energy_reports_requested
                        ? (h < 2 ? ScalarMoistEnergyKind::LiquidWater : ScalarMoistEnergyKind::IceWater)
                        : ScalarMoistEnergyKind::None,
                    energy_reports_requested ? &runtime.thermodynamic_moist_energy_reports[hydro_report_indices[h]]
                                             : nullptr,
                    runtime.moist_interface.enabled ? &runtime.moist_interface.water[hydro_report_indices[h] - 1]
                                                    : nullptr};
            }
            else if (boundary_reports_requested)
            {
                // Activity discovery proved the complete validated field is
                // zero, so skipping its transport has the exact all-zero
                // boundary and commit report.
                runtime.thermodynamic_boundary_reports[
                    hydro_report_indices[h]].complete = true;
            }
            if (!hydro_active[h] && energy_reports_requested)
            {
                runtime.thermodynamic_moist_energy_reports[
                    hydro_report_indices[h]].complete = true;
            }
        }

        const bool tracer_in_batch = execution_state.fields.tracer.size() == execution_state.fields.theta.size();
        if (tracer_in_batch)
        {
            scalars[n++] = {&execution_state.fields.tracer, kTracerDefaultKappa, true};
        }

        advect_scalars_conservative_3d_cartesian(execution_state,
            scalars, n, dt_advect, context, reconstruction);
        if (boundary_reports_requested)
        {
            runtime.thermodynamic_boundary_reports_complete =
                std::all_of(
                    runtime.thermodynamic_boundary_reports.begin(),
                    runtime.thermodynamic_boundary_reports.end(),
                    [](const ScalarBoundaryTransportReport& report)
                    {
                        return report.complete;
                    });
            if (!runtime.thermodynamic_boundary_reports_complete)
            {
                throw std::runtime_error(
                    "CPU thermodynamic transport did not publish every boundary ledger");
            }
        }
        if (energy_reports_requested)
        {
            runtime.thermodynamic_moist_energy_reports_complete =
                std::all_of(
                    runtime.thermodynamic_moist_energy_reports.begin(),
                    runtime.thermodynamic_moist_energy_reports.end(),
                    [](const ScalarMoistEnergyTransportReportV1& report)
                    {
                        return report.complete;
                    });
            if (!runtime.thermodynamic_moist_energy_reports_complete)
            {
                throw std::runtime_error(
                    "CPU thermodynamic transport did not publish every "
                    "moist-energy ledger");
            }
        }
        reported_batch = true;
        runtime.tracer_advected_in_batch = tracer_in_batch;
    }

    if (!gpu_advected && !reported_batch)
    {
        advect_scalar_3d(execution_state, execution_state.fields.theta, dt_advect, kappa_theta,
                         /*positive_definite=*/false, context, &runtime);
    }

    const PhysicalCellDomain physical_domain =
        physical_scalar_domain(nr, nth, nz, execution_state.configuration->stagger_type);
    int theta_clamped = 0;
    int theta_nonfinite = 0;
    #pragma omp parallel for collapse(2) reduction(+:theta_clamped)
    for (int i = physical_domain.i_begin; i < physical_domain.i_end; ++i)
    {
        for (int j = physical_domain.j_begin; j < physical_domain.j_end; ++j)
        {
            for (int k = physical_domain.k_begin; k < physical_domain.k_end; ++k)
            {
                const float theta_old = execution_state.fields.theta(i, j, k);
                if (!std::isfinite(static_cast<double>(theta_old)))
                {
                    #pragma omp atomic update
                    ++theta_nonfinite;
                    continue;
                }
                const float theta_new = clamp_theta_k(theta_old);
                if (theta_new != theta_old) { ++theta_clamped; }
                execution_state.fields.theta(i, j, k) = theta_new;
            }
        }
    }

    if (!gpu_advected && !reported_batch)
    {
        advect_scalar_3d(execution_state, execution_state.fields.qv, dt_advect, kappa_moisture,
                         /*positive_definite=*/true, context, &runtime);
        if (qc_active) advect_scalar_3d(execution_state, execution_state.fields.qc, dt_advect, kappa_moisture, true, context, &runtime);
        if (qr_active) advect_scalar_3d(execution_state, execution_state.fields.qr, dt_advect, kappa_moisture, true, context, &runtime);
        if (qi_active) advect_scalar_3d(execution_state, execution_state.fields.qi, dt_advect, kappa_moisture, true, context, &runtime);
        if (qs_active) advect_scalar_3d(execution_state, execution_state.fields.qs, dt_advect, kappa_moisture, true, context, &runtime);
        if (qh_active) advect_scalar_3d(execution_state, execution_state.fields.qh, dt_advect, kappa_moisture, true, context, &runtime);
        if (qg_active) advect_scalar_3d(execution_state, execution_state.fields.qg, dt_advect, kappa_moisture, true, context, &runtime);
    }

    int qv_sanitized = 0;
    int hydrometeor_sanitized = 0;
    int moisture_nonfinite = 0;
    // Signed mass changed by the positive-definiteness clamps
    // (SCIENCE.md M5): negative undershoots zeroed CREATE water,
    // over-cap values destroy it. rho-weighted, coordinate-aware volume.
    double moisture_mass_created_kg = 0.0;
    const float* const rho_data_clamp = execution_state.fields.rho.data();
    float* const qv_data = execution_state.fields.qv.data();
    float* const qc_data = qc_active ? execution_state.fields.qc.data() : nullptr;
    float* const qr_data = qr_active ? execution_state.fields.qr.data() : nullptr;
    float* const qi_data = qi_active ? execution_state.fields.qi.data() : nullptr;
    float* const qs_data = qs_active ? execution_state.fields.qs.data() : nullptr;
    float* const qh_data = qh_active ? execution_state.fields.qh.data() : nullptr;
    float* const qg_data = qg_active ? execution_state.fields.qg.data() : nullptr;
    const SimulationStateView simulation = execution_state;
    tmv::physical_grid_metrics::validate(simulation);

    #pragma omp parallel for reduction(+:qv_sanitized, hydrometeor_sanitized, moisture_mass_created_kg)
    for (int i = physical_domain.i_begin; i < physical_domain.i_end; ++i)
    {
      for (int j = physical_domain.j_begin; j < physical_domain.j_end; ++j)
      {
       for (int k = physical_domain.k_begin; k < physical_domain.k_end; ++k)
       {
        const std::size_t idx =
            (static_cast<std::size_t>(i) * static_cast<std::size_t>(nth) +
             static_cast<std::size_t>(j)) * static_cast<std::size_t>(nz) +
            static_cast<std::size_t>(k);
        const double clamp_rho_dv = static_cast<double>(rho_data_clamp[idx])
            * diagnostic_cell_volume(simulation, i, j, k);
        const float qv_old = qv_data[idx];
        if (!std::isfinite(static_cast<double>(qv_old)))
        {
            #pragma omp atomic update
            ++moisture_nonfinite;
        }
        else
        {
        const float qv_new = clamp_qv_kgkg(qv_old);
        if (qv_new != qv_old)
        {
            ++qv_sanitized;
            moisture_mass_created_kg +=
                (static_cast<double>(qv_new) - static_cast<double>(qv_old))
                * clamp_rho_dv;
        }
        qv_data[idx] = qv_new;
        }

        if (qc_data)
        {
            const float old_val = qc_data[idx];
            if (!std::isfinite(static_cast<double>(old_val)))
            {
                #pragma omp atomic update
                ++moisture_nonfinite;
            }
            else
            {
            const float new_val = clamp_hydrometeor_kgkg(old_val);
            if (new_val != old_val)
            {
                ++hydrometeor_sanitized;
                moisture_mass_created_kg +=
                    (static_cast<double>(new_val) - static_cast<double>(old_val))
                    * clamp_rho_dv;
            }
            qc_data[idx] = new_val;
            }
        }
        if (qr_data)
        {
            const float old_val = qr_data[idx];
            if (!std::isfinite(static_cast<double>(old_val)))
            {
                #pragma omp atomic update
                ++moisture_nonfinite;
            }
            else
            {
            const float new_val = clamp_hydrometeor_kgkg(old_val);
            if (new_val != old_val)
            {
                ++hydrometeor_sanitized;
                moisture_mass_created_kg +=
                    (static_cast<double>(new_val) - static_cast<double>(old_val))
                    * clamp_rho_dv;
            }
            qr_data[idx] = new_val;
            }
        }
        if (qi_data)
        {
            const float old_val = qi_data[idx];
            if (!std::isfinite(static_cast<double>(old_val)))
            {
                #pragma omp atomic update
                ++moisture_nonfinite;
            }
            else
            {
            const float new_val = clamp_hydrometeor_kgkg(old_val);
            if (new_val != old_val)
            {
                ++hydrometeor_sanitized;
                moisture_mass_created_kg +=
                    (static_cast<double>(new_val) - static_cast<double>(old_val))
                    * clamp_rho_dv;
            }
            qi_data[idx] = new_val;
            }
        }
        if (qs_data)
        {
            const float old_val = qs_data[idx];
            if (!std::isfinite(static_cast<double>(old_val)))
            {
                #pragma omp atomic update
                ++moisture_nonfinite;
            }
            else
            {
            const float new_val = clamp_hydrometeor_kgkg(old_val);
            if (new_val != old_val)
            {
                ++hydrometeor_sanitized;
                moisture_mass_created_kg +=
                    (static_cast<double>(new_val) - static_cast<double>(old_val))
                    * clamp_rho_dv;
            }
            qs_data[idx] = new_val;
            }
        }
        if (qh_data)
        {
            const float old_val = qh_data[idx];
            if (!std::isfinite(static_cast<double>(old_val)))
            {
                #pragma omp atomic update
                ++moisture_nonfinite;
            }
            else
            {
            const float new_val = clamp_hydrometeor_kgkg(old_val);
            if (new_val != old_val)
            {
                ++hydrometeor_sanitized;
                moisture_mass_created_kg +=
                    (static_cast<double>(new_val) - static_cast<double>(old_val))
                    * clamp_rho_dv;
            }
            qh_data[idx] = new_val;
            }
        }
        if (qg_data)
        {
            const float old_val = qg_data[idx];
            if (!std::isfinite(static_cast<double>(old_val)))
            {
                #pragma omp atomic update
                ++moisture_nonfinite;
            }
            else
            {
            const float new_val = clamp_hydrometeor_kgkg(old_val);
            if (new_val != old_val)
            {
                ++hydrometeor_sanitized;
                moisture_mass_created_kg +=
                    (static_cast<double>(new_val) - static_cast<double>(old_val))
                    * clamp_rho_dv;
            }
            qg_data[idx] = new_val;
            }
        }
       }
      }
    }

    if (theta_nonfinite > 0 || moisture_nonfinite > 0)
    {
        throw std::runtime_error(
            "thermodynamic advection produced non-finite prognostic state: theta=" +
            std::to_string(theta_nonfinite) + ", moisture=" +
            std::to_string(moisture_nonfinite));
    }

    if (qv_sanitized > 0 || hydrometeor_sanitized > 0 || theta_clamped > 0)
    {
        // Loud by design (SCIENCE.md M5): these clamps modify state
        // (the moisture ones create or destroy water mass), so they are
        // physics events, not debug chatter.
        tmv::log_warn("[ADVECTION GUARD] qv_sanitized=", qv_sanitized,
                      ", hydrometeor_sanitized=", hydrometeor_sanitized,
                      ", theta_clamped=", theta_clamped,
                      ", moisture_mass_created_kg=", moisture_mass_created_kg);
    }

    if (perf_on)
    {
        advection_detail::perf_totals().thermo_total_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        ++advection_detail::perf_totals().thermo_calls;
    }
    runtime.invalidate_hydrometeor_activity();
}

void advect_tracer_3d(double dt_advect, double kappa,
                      const CartesianCGridTransportContext* context,
                      AdvectionRuntimeState* runtime_state)
{
    return advect_tracer_3d(global_simulation_state(), dt_advect, kappa, context, runtime_state);
}

void advect_tracer_3d(const SimulationStateView& execution_state, double dt_advect, double kappa,
                      const CartesianCGridTransportContext* context,
                      AdvectionRuntimeState* runtime_state)
{
    AdvectionRuntimeState& runtime = runtime_state
        ? *runtime_state : (*execution_state.advection_runtime);
    const bool perf_on = advection_detail::perf_enabled();
    const auto t0 = perf_on ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
    // When the thermodynamics C-grid batch already carried the tracer this
    // step (the orchestrator advects
    // thermodynamics first), this call is a consume-and-skip.
    const bool consume_batched_tracer = runtime.tracer_advected_in_batch;
    // The handoff is cleared only after the roundoff transaction commits. A
    // failed preflight therefore preserves ownership and the prior reports.
    if (!consume_batched_tracer)
    {
        require_transport_field(execution_state,
            execution_state.fields.tracer, "passive tracer", 0.0f, passive_tracer_max);
        advect_scalar_3d(execution_state, execution_state.fields.tracer, dt_advect, kappa,
                         /*positive_definite=*/true, context, &runtime);
    }
    canonicalize_passive_tracer_roundoff(
        execution_state, runtime);
    if (consume_batched_tracer)
        runtime.tracer_advected_in_batch = false;
    const auto& roundoff = runtime.tracer_roundoff_report;
    if (roundoff.negative_subnormal_count != 0
        || roundoff.negative_zero_count != 0)
    {
        tmv::log_warn(
            "[ADVECTION TRACER ROUNDOFF] physical_inventory_correction_kg=",
            roundoff.physical_inventory_correction_kg,
            ", negative_subnormal_count=",
            roundoff.negative_subnormal_count,
            ", negative_zero_count=", roundoff.negative_zero_count,
            ", closure_canonicalization_count=",
            roundoff.closure_canonicalization_count);
    }
    if (perf_on)
    {
        advection_detail::perf_totals().tracer_total_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        ++advection_detail::perf_totals().tracer_calls;
    }
}

void advect_thermodynamics_3d(const SimulationStateView&            execution_state,
                              double                                dt_advect,
                              double                                kappa_theta,
                              double                                kappa_moisture,
                              const CartesianCGridTransportContext* context,
                              AdvectionRuntimeState*                runtime_state)
{
    auto&      runtime     = runtime_state ? *runtime_state : *execution_state.advection_runtime;
    const bool native_weno = execution_state.configuration && execution_state.configuration->admission.native_weno5;
    const bool native_dry_tvd = execution_state.configuration && execution_state.configuration->admission.native_dry_tvd;
    const bool transaction = native_weno || native_dry_tvd;
    if (!transaction)
    {
        advect_thermodynamics_impl(execution_state, dt_advect, kappa_theta, kappa_moisture, context, runtime_state);
        return;
    }
    if (conservative_reconstruction(runtime) !=
            (native_weno ? CartesianFaceReconstruction::WENO5 : CartesianFaceReconstruction::MonotonicCentered) ||
        !active_backend_is_cartesian_cgrid(execution_state) || context == nullptr ||
        execution_state.compute_runtime == nullptr ||
        active_compute_backend_kind(*execution_state.compute_runtime) == ComputeBackendKind::Cpu ||
        execution_state.compute_runtime->fallback_active || execution_state.compute_runtime->config.allow_fallback)
    {
        throw std::runtime_error("strict native conservative transport requires its selected backend and retained transport context");
    }
    std::array<Field3D*, 9> fields{&execution_state.fields.theta,
                                   &execution_state.fields.qv,
                                   &execution_state.fields.qc,
                                   &execution_state.fields.qr,
                                   &execution_state.fields.qi,
                                   &execution_state.fields.qs,
                                   &execution_state.fields.qg,
                                   &execution_state.fields.qh,
                                   &execution_state.fields.tracer};
    std::array<Field3D, 9>  saved;
    for (std::size_t i = 0; i < fields.size(); ++i)
        saved[i] = *fields[i];
    const auto boundaries        = runtime.thermodynamic_boundary_reports;
    const auto energies          = runtime.thermodynamic_moist_energy_reports;
    const auto boundary_complete = runtime.thermodynamic_boundary_reports_complete;
    const auto energy_complete   = runtime.thermodynamic_moist_energy_reports_complete;
    const auto tracer            = runtime.tracer_advected_in_batch;
    const auto tracer_report     = runtime.tracer_roundoff_report;
    const auto cumulative_tracer = runtime.cumulative_tracer_roundoff_report;
    const auto tracer_steps      = runtime.tracer_roundoff_report_step_count;
    const auto activity          = runtime.hydrometeor_active;
    const auto activity_current  = runtime.activity_current;
    const auto dimensions        = std::array<int, 3>{runtime.nr, runtime.nth, runtime.nz};
    const auto theta_interface   = runtime.theta_interface_flux_register;
    const auto dry_interface     = runtime.dry_mass_interface_flux_register;
    const bool theta_capture     = runtime.theta_interface_flux_capture_enabled;
    const bool dry_capture       = runtime.dry_mass_interface_flux_capture_enabled;
    const auto consumed          = runtime.metadata_consumptions;
    const auto scanned           = runtime.fallback_scans;

    try
    {
        advect_thermodynamics_impl(execution_state, dt_advect, kappa_theta, kappa_moisture, context, runtime_state);
    }
    catch (...)
    {
        for (std::size_t i = 0; i < fields.size(); ++i)
            *fields[i] = std::move(saved[i]);
        runtime.thermodynamic_boundary_reports              = boundaries;
        runtime.thermodynamic_moist_energy_reports          = energies;
        runtime.thermodynamic_boundary_reports_complete     = boundary_complete;
        runtime.thermodynamic_moist_energy_reports_complete = energy_complete;
        runtime.tracer_advected_in_batch                    = tracer;
        runtime.tracer_roundoff_report                      = tracer_report;
        runtime.cumulative_tracer_roundoff_report           = cumulative_tracer;
        runtime.tracer_roundoff_report_step_count           = tracer_steps;
        runtime.hydrometeor_active                          = activity;
        runtime.activity_current                            = activity_current;
        runtime.nr                                          = dimensions[0];
        runtime.nth                                         = dimensions[1];
        runtime.nz                                          = dimensions[2];
        runtime.theta_interface_flux_register               = theta_interface;
        runtime.dry_mass_interface_flux_register            = dry_interface;
        runtime.theta_interface_flux_capture_enabled        = theta_capture;
        runtime.dry_mass_interface_flux_capture_enabled     = dry_capture;
        runtime.metadata_consumptions                       = consumed;
        runtime.fallback_scans                              = scanned;
        throw;
    }
}

void advect_microphysics_prognostics_3d(
    MicrophysicsScheme& scheme,
    double dt_advect,
    const CartesianCGridTransportContext* context,
    AdvectionRuntimeState* runtime_state)
{
    return advect_microphysics_prognostics_3d(global_simulation_state(), scheme, dt_advect, context, runtime_state);
}

void advect_microphysics_prognostics_3d(const SimulationStateView& execution_state,
    MicrophysicsScheme& scheme,
    double dt_advect,
    const CartesianCGridTransportContext* context,
    AdvectionRuntimeState* runtime_state)
{
    AdvectionRuntimeState& runtime = runtime_state != nullptr
        ? *runtime_state : (*execution_state.advection_runtime);
    MicrophysicsPrognosticScalarList moments =
        scheme.prognostic_scalars_for_transport();
    auto capture = runtime.private_interface;
    if (!capture.fields.empty())
    {
        capture.require_layout(moments);
        if (!context || advection_disabled(runtime))
            throw std::runtime_error("private interface capture requires conservative transport");
        for (auto& field : capture.fields)
            field.boundary = {};
    }
    if (moments.empty())
    {
        return;
    }

    // Validate the complete collection before advancing its first member.
    // This prevents a malformed scheme descriptor from partially updating
    // private prognostic state.
    for (const auto& moment : moments)
    {
        if (moment.name == nullptr || moment.field == nullptr)
        {
            throw std::invalid_argument(
                "microphysics prognostic transport contains a null descriptor");
        }
        if (moment.field->size_r() != execution_state.grid.nr || moment.field->size_th() != execution_state.grid.nth
            || moment.field->size_z() != execution_state.grid.nz)
        {
            throw std::invalid_argument(
                std::string("microphysics prognostic transport shape mismatch: ")
                + moment.name);
        }
        require_finite_transport_field(execution_state, *moment.field, moment.name);
    }
    const bool native_weno   = execution_state.configuration && execution_state.configuration->admission.native_weno5;
    const bool native_nested = execution_state.configuration &&
                               execution_state.configuration->admission.two_way_accelerator_reference &&
                               execution_state.compute_runtime &&
                               active_compute_backend_kind(*execution_state.compute_runtime) != ComputeBackendKind::Cpu;
    const bool native        = native_weno || native_nested;
    if (native &&
        (context == nullptr ||
         (runtime.config.scheme_id != "weno5" && (!native_nested || runtime.config.scheme_id != "tvd")) ||
         execution_state.compute_runtime == nullptr ||
         active_compute_backend_kind(*execution_state.compute_runtime) == ComputeBackendKind::Cpu ||
         execution_state.compute_runtime->fallback_active || execution_state.compute_runtime->config.allow_fallback))
        throw std::runtime_error("native private-moment transport requires retained conservative execution ownership");
    if (advection_disabled(runtime))
        return;
    CartesianFaceReconstruction reconstruction = conservative_reconstruction(runtime);
    const bool                  coupled_population =
        std::any_of(moments.begin(),
                    moments.end(),
                    [](const auto& moment) { return moment.mass_species != MicrophysicsMassSpecies::None; });
    if (coupled_population)
        reconstruction = uses_weno(reconstruction) ? CartesianFaceReconstruction::WENO5PositiveSupport
                                                   : CartesianFaceReconstruction::MonotonicCenteredPositiveSupport;

    const bool has_volume_concentration = std::any_of(
        moments.begin(), moments.end(), [](const auto& moment)
        {
            return moment.storage
                == MicrophysicsScalarStorage::VolumeConcentration;
        });
    if (has_volume_concentration && context == nullptr)
    {
        throw std::runtime_error(
            "volume-concentration microphysics moments require retained "
            "begin-step density and mass fluxes");
    }

    if (context != nullptr)
    {
        context->require_valid(execution_state, execution_state.grid.nr, execution_state.grid.nth, execution_state.grid.nz, dt_advect);
        auto& candidates = runtime.private_transport.get();
        candidates.resize(moments.size());
        std::array<CartesianScalarTransport, MicrophysicsPrognosticScalarList::capacity> batch{};
        for (std::size_t index = 0; index < moments.size(); ++index)
        {
            microphysics_scalar_to_mixing_ratio(
                moments[index], context->density_begin, candidates[index]);
            batch[index] = {&candidates[index], 0.0, moments[index].positive_definite};
            if (!capture.fields.empty())
            {
                batch[index].boundary_report         = &capture.fields[index].boundary;
                batch[index].interface_flux_register = &capture.fields[index].flux;
            }
        }
        if (native && context->terrain_metrics() == nullptr)
        {
            require_transport_field(execution_state,
                                    execution_state.fields.rho,
                                    "private transport density",
                                    density_min_kgm3,
                                    density_max_kgm3);
            require_transport_field(execution_state,
                                    context->density_begin,
                                    "private transport beginning density",
                                    density_min_kgm3,
                                    density_max_kgm3);
            require_finite_transport_field(execution_state, execution_state.fields.u, "private transport x velocity");
            require_finite_transport_field(execution_state, execution_state.fields.v, "private transport y velocity");
            require_finite_transport_field(
                execution_state, execution_state.fields.w, "private transport vertical velocity");
            for (const double spacing : {execution_state.grid.dr, execution_state.grid.dtheta, execution_state.grid.dz})
                if (!std::isfinite(spacing) || spacing <= 0.0)
                    throw std::invalid_argument("native private-moment transport requires positive finite spacing");
            std::array<const float*, MicrophysicsPrognosticScalarList::capacity> input{};
            std::array<float*, MicrophysicsPrognosticScalarList::capacity>       output{};
            std::array<float, MicrophysicsPrognosticScalarList::capacity>        diffusion{};
            std::array<int, MicrophysicsPrognosticScalarList::capacity>          positive{};
            for (std::size_t field = 0; field < moments.size(); ++field)
            {
                input[field]    = candidates[field].data();
                output[field]   = candidates[field].data();
                positive[field] = moments[field].positive_definite;
            }
            std::array<ScalarBoundaryTransportReport*, MicrophysicsPrognosticScalarList::capacity>  boundaries{};
            std::array<CartesianInterfaceFluxRegister*, MicrophysicsPrognosticScalarList::capacity> interfaces{};
            if (!capture.fields.empty())
                for (std::size_t field = 0; field < moments.size(); ++field)
                {
                    boundaries[field] = &capture.fields[field].boundary;
                    interfaces[field] = &capture.fields[field].flux;
                }
            const auto& grid = execution_state.grid;
            if (!dispatch_advect_scalars_conservative_cartesian_cgrid_backend(
                    *execution_state.compute_runtime,
                    execution_state.fields.u.data(),
                    execution_state.fields.v.data(),
                    execution_state.fields.w.data(),
                    execution_state.fields.rho.data(),
                    input.data(),
                    output.data(),
                    diffusion.data(),
                    positive.data(),
                    static_cast<int>(moments.size()),
                    grid.nr,
                    grid.nth,
                    grid.nz,
                    static_cast<float>(1 / grid.dr),
                    static_cast<float>(1 / grid.dtheta),
                    static_cast<float>(1 / grid.dz),
                    static_cast<float>(1 / (grid.dr * grid.dr)),
                    static_cast<float>(1 / (grid.dtheta * grid.dtheta)),
                    static_cast<float>(1 / (grid.dz * grid.dz)),
                    static_cast<float>(dt_advect),
                    density_min_kgm3,
                    context->density_begin.data(),
                    context->mass_flux_x.data(),
                    context->mass_flux_y.data(),
                    context->mass_flux_z.data(),
                    true,
                    nullptr,
                    nullptr,
                    context,
                    dt_advect,
                    capture.fields.empty() ? nullptr : boundaries.data(),
                    nullptr,
                    nullptr,
                    0,
                    0,
                    &execution_state,
                    capture.fields.empty() ? nullptr : interfaces.data(),
                    uses_weno(reconstruction),
                    preserves_positive_support(reconstruction)))
                throw std::runtime_error("native private-moment transport failed; CPU substitution is disabled");
            for (std::size_t field = 0; field < moments.size(); ++field)
                for (std::size_t index = 0; index < candidates[field].size(); ++index)
                    if (!std::isfinite(candidates[field].data()[index]) ||
                        (moments[field].positive_definite && candidates[field].data()[index] < 0))
                        throw std::runtime_error("native private-moment WENO5 transport produced invalid state");
        }
        else
        {
            advect_scalars_conservative_3d_cartesian(
                execution_state, batch.data(), moments.size(), dt_advect, context, reconstruction);
        }
        for (std::size_t field = 0; field < moments.size(); ++field)
        {
            if (moments[field].storage != MicrophysicsScalarStorage::VolumeConcentration) continue;
            for (std::size_t index = 0; index < execution_state.fields.rho.size(); ++index)
            {
                const float value = static_cast<float>(
                    static_cast<double>(candidates[field].data()[index]) * execution_state.fields.rho.data()[index]);
                if (!std::isfinite(value) || (moments[field].positive_definite && value < 0.0f))
                    throw std::runtime_error("microphysics moment recovery produced invalid state");
                candidates[field].data()[index] = value;
            }
        }
        for (const auto& field : capture.fields)
            if (!field.boundary.complete || !std::isfinite(field.boundary.closure_residual))
                throw std::runtime_error("private conservative transport omitted its boundary ledger");
        for (std::size_t field = 0; field < moments.size(); ++field)
            std::swap(*moments[field].field, candidates[field]);
        runtime.private_interface = std::move(capture);
        return;
    }

    for (const auto& moment : moments)
    {
        advect_scalar_3d(execution_state, *moment.field, dt_advect, 0.0,
                         moment.positive_definite, context, runtime_state);
    }
}


} // namespace tmv
