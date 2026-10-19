#include "core/field/field_inventory.hpp"
#include "diagnostics/deterministic_reduction.hpp"
#include "core/runtime/simulation_checkpoint.hpp"

#include "boundary_layer/factory.hpp"
#include "chaos/factory.hpp"
#include "core/field/field_bounds.hpp"
#include "core/infra/nested_execution.hpp"
#include "core/runtime/boundary_layer_checkpoint.hpp"
#include "core/runtime/checkpoint_format.hpp"
#include "microphysics/factory.hpp"
#include "numerics/advection/factory.hpp"
#include "numerics/diffusion/factory.hpp"
#include "numerics/time_stepping/factory.hpp"
#include "radar/factory.hpp"
#include "radiation/factory.hpp"
#include "radiation/rrtmg_config_limits.hpp"
#include "terrain/factory.hpp"
#include "terrain/conservative_transport.hpp"
#include "terrain/base/topography.hpp"
#include "turbulence/factory.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace tmv::checkpoint
{
namespace
{

using wire::Decoder;
using wire::Encoder;
using wire::Section;

constexpr std::array<std::uint8_t, 8> k_magic                  = {'N', 'I', 'M', 'B', 'S', 'I', 'M', '1'};
constexpr std::uint32_t               k_metadata_section       = 1;
constexpr std::uint32_t               k_configuration_section  = 2;
constexpr std::uint32_t               k_grid_section           = 3;
constexpr std::uint32_t               k_fields_section         = 4;
constexpr std::uint32_t               k_auxiliary_section      = 5;
constexpr std::uint32_t               k_runtime_section        = 6;
constexpr std::uint32_t               k_nesting_section        = 7;
constexpr std::uint32_t               k_trajectory_section     = 8;
constexpr std::uint64_t               k_max_trajectory_parcels = tmv::kMaximumTrajectoryParcels;
constexpr std::uint64_t               k_max_trajectory_samples = tmv::kMaximumTrajectorySamples;

bool valid_trajectory_configuration(const tmv::TrajectoryConfig& config) noexcept
{
    const auto finite_or_nan = [](double value) { return !std::isinf(value); };
    if (!config.enabled || config.nx <= 0 || config.ny <= 0 || config.nz <= 0 || !std::isfinite(config.seed_time_s) ||
        config.seed_time_s < 0.0 || !std::isfinite(config.sample_interval_s) || config.sample_interval_s < 0.0 ||
        config.max_samples_per_parcel <= 0 || !std::isfinite(config.half_x_m) || config.half_x_m < 0.0 ||
        !std::isfinite(config.half_y_m) || config.half_y_m < 0.0 || !std::isfinite(config.half_z_m) ||
        config.half_z_m < 0.0 || !finite_or_nan(config.center_x_m) || !finite_or_nan(config.center_y_m) ||
        !finite_or_nan(config.center_z_m))
        return false;
    const std::uint64_t nx = static_cast<std::uint64_t>(config.nx);
    const std::uint64_t ny = static_cast<std::uint64_t>(config.ny);
    const std::uint64_t nz = static_cast<std::uint64_t>(config.nz);
    return nx <= k_max_trajectory_parcels / ny && nx * ny <= k_max_trajectory_parcels / nz &&
           static_cast<std::uint64_t>(config.max_samples_per_parcel) <= k_max_trajectory_samples / (nx * ny * nz);
}

void encode_trajectory_configuration(Encoder& out, const tmv::TrajectoryConfig& config)
{
    out.boolean(config.enabled);
    out.integer(config.nx);
    out.integer(config.ny);
    out.integer(config.nz);
    out.floating(config.seed_time_s);
    out.floating(config.sample_interval_s);
    out.integer(config.max_samples_per_parcel);
    out.floating(config.half_x_m);
    out.floating(config.half_y_m);
    out.floating(config.half_z_m);
    out.floating(config.center_x_m);
    out.floating(config.center_y_m);
    out.floating(config.center_z_m);
}

bool decode_trajectory_configuration(Decoder& in, tmv::TrajectoryConfig& config, std::uint32_t section_version)
{
    std::int32_t nx = 0, ny = 0, nz = 0, max_samples = 0;
    if (!in.boolean(config.enabled) || !in.integer(nx) || !in.integer(ny) || !in.integer(nz) ||
        !in.floating(config.seed_time_s))
        return false;
    if (section_version >= 2 && (!in.floating(config.sample_interval_s) || !in.integer(max_samples)))
        return false;
    if (section_version >= 2)
        config.max_samples_per_parcel = max_samples;
    if (!in.floating(config.half_x_m) || !in.floating(config.half_y_m) || !in.floating(config.half_z_m) ||
        !in.floating(config.center_x_m) || !in.floating(config.center_y_m) || !in.floating(config.center_z_m))
        return false;
    config.nx = nx;
    config.ny = ny;
    config.nz = nz;
    return valid_trajectory_configuration(config);
}

bool encode_trajectories(Encoder&                  out,
                         const tmv::TrajectorySet& trajectories,
                         double                    checkpoint_time_s,
                         std::string&              error)
{
    const auto& config  = trajectories.configuration();
    const auto& parcels = trajectories.parcels();
    if (!valid_trajectory_configuration(config) || parcels.size() > k_max_trajectory_parcels ||
        (!trajectories.seeded() && !parcels.empty()) ||
        (!trajectories.seeded() && !std::isnan(trajectories.next_sample_time_s())) ||
        (trajectories.seeded() && config.sample_interval_s == 0.0 && !std::isnan(trajectories.next_sample_time_s())) ||
        (trajectories.seeded() && config.sample_interval_s > 0.0 &&
         (!std::isfinite(trajectories.next_sample_time_s()) || trajectories.next_sample_time_s() <= checkpoint_time_s)))
    {
        error = "cannot checkpoint invalid trajectory configuration or parcel ownership";
        return false;
    }
    std::uint64_t total_samples = 0;
    for (std::size_t index = 0; index < parcels.size(); ++index)
    {
        const auto& parcel = parcels[index];
        if (parcel.id != static_cast<int>(index) || !std::isfinite(parcel.x) || !std::isfinite(parcel.y) ||
            !std::isfinite(parcel.z) || (trajectories.seeded() && parcel.path.empty()) ||
            parcel.path.size() > static_cast<std::size_t>(config.max_samples_per_parcel) ||
            parcel.path.size() > k_max_trajectory_samples - total_samples)
        {
            error = "cannot checkpoint invalid trajectory parcel state";
            return false;
        }
        float previous_time = -std::numeric_limits<float>::infinity();
        for (const auto& sample : parcel.path)
        {
            if (!std::isfinite(sample.t) || !std::isfinite(sample.x) || !std::isfinite(sample.y) ||
                !std::isfinite(sample.z) || sample.t < previous_time ||
                static_cast<double>(sample.t) > checkpoint_time_s + 1.0e-3)
            {
                error = "cannot checkpoint invalid or future trajectory path sample";
                return false;
            }
            previous_time = sample.t;
        }
        total_samples += parcel.path.size();
    }
    encode_trajectory_configuration(out, config);
    out.boolean(trajectories.seeded());
    out.floating(trajectories.next_sample_time_s());
    out.unsigned_value<std::uint64_t>(parcels.size());
    for (const auto& parcel : parcels)
    {
        out.integer(parcel.id);
        out.boolean(parcel.active);
        out.floating(parcel.x);
        out.floating(parcel.y);
        out.floating(parcel.z);
        out.unsigned_value<std::uint64_t>(parcel.path.size());
        for (const auto& sample : parcel.path)
        {
            out.floating(sample.t);
            out.floating(sample.x);
            out.floating(sample.y);
            out.floating(sample.z);
            out.floating(sample.w);
            out.floating(sample.zeta);
            out.floating(sample.tilting);
            out.floating(sample.stretching);
            out.floating(sample.baroclinic);
        }
    }
    return true;
}

bool decode_trajectories(Decoder&                        in,
                         tmv::TrajectoryCheckpointState& state,
                         double                          checkpoint_time_s,
                         std::uint32_t                   section_version,
                         std::string&                    error)
{
    std::uint64_t parcel_count = 0;
    if (!decode_trajectory_configuration(in, state.configuration, section_version) || !in.boolean(state.seeded))
    {
        error = "invalid trajectory checkpoint header";
        return false;
    }
    if (section_version >= 2 && !in.floating(state.next_sample_time_s))
    {
        error = "invalid trajectory checkpoint cadence state";
        return false;
    }
    if (!in.unsigned_value(parcel_count) || parcel_count > k_max_trajectory_parcels ||
        (!state.seeded && parcel_count != 0))
    {
        error = "invalid trajectory checkpoint header";
        return false;
    }
    const std::uint64_t requested_count = static_cast<std::uint64_t>(state.configuration.nx) *
                                          static_cast<std::uint64_t>(state.configuration.ny) *
                                          static_cast<std::uint64_t>(state.configuration.nz);
    if (parcel_count > requested_count)
    {
        error = "trajectory checkpoint parcel count exceeds its configured seed lattice";
        return false;
    }
    constexpr std::size_t minimum_parcel_bytes =
        sizeof(std::uint32_t) + sizeof(std::uint8_t) + 3 * sizeof(double) + sizeof(std::uint64_t);
    if (parcel_count > in.remaining() / minimum_parcel_bytes)
    {
        error = "trajectory checkpoint parcel count exceeds its encoded payload";
        return false;
    }
    state.parcels.clear();
    state.parcels.reserve(static_cast<std::size_t>(parcel_count));
    std::uint64_t total_samples = 0;
    for (std::uint64_t index = 0; index < parcel_count; ++index)
    {
        tmv::Parcel   parcel;
        std::int32_t  id           = 0;
        std::uint64_t sample_count = 0;
        if (!in.integer(id) || !in.boolean(parcel.active) || !in.floating(parcel.x) || !in.floating(parcel.y) ||
            !in.floating(parcel.z) || !in.unsigned_value(sample_count) || id != static_cast<std::int32_t>(index) ||
            !std::isfinite(parcel.x) || !std::isfinite(parcel.y) || !std::isfinite(parcel.z) ||
            (state.seeded && sample_count == 0) ||
            sample_count > static_cast<std::uint64_t>(state.configuration.max_samples_per_parcel) ||
            sample_count > k_max_trajectory_samples - total_samples ||
            sample_count > in.remaining() / (sizeof(float) * tmv::kTrajectoryVarCount))
        {
            error = "invalid trajectory checkpoint parcel";
            return false;
        }
        parcel.id = id;
        parcel.path.resize(static_cast<std::size_t>(sample_count));
        float previous_time = -std::numeric_limits<float>::infinity();
        for (auto& sample : parcel.path)
        {
            if (!in.floating(sample.t) || !in.floating(sample.x) || !in.floating(sample.y) || !in.floating(sample.z) ||
                !in.floating(sample.w) || !in.floating(sample.zeta) || !in.floating(sample.tilting) ||
                !in.floating(sample.stretching) || !in.floating(sample.baroclinic) || !std::isfinite(sample.t) ||
                !std::isfinite(sample.x) || !std::isfinite(sample.y) || !std::isfinite(sample.z) ||
                sample.t < previous_time || static_cast<double>(sample.t) > checkpoint_time_s + 1.0e-3)
            {
                error = "invalid trajectory checkpoint path sample";
                return false;
            }
            previous_time = sample.t;
        }
        total_samples += sample_count;
        state.parcels.push_back(std::move(parcel));
    }
    return true;
}

bool valid_storm_tracking_config(const tmv::init::config::DomainMotionSection::Tracking& tracking) noexcept
{
    const bool valid_branch_preference =
        tracking.branch_preference == tmv::init::config::StormBranchPreference::Continuity ||
        tracking.branch_preference == tmv::init::config::StormBranchPreference::BunkersRight ||
        tracking.branch_preference == tmv::init::config::StormBranchPreference::BunkersLeft;
    return std::isfinite(tracking.interval_s) && tracking.interval_s > 0.0 && std::isfinite(tracking.inner_margin_m) &&
           tracking.inner_margin_m >= 0.0 && tracking.maximum_shift_cells >= 0 && std::isfinite(tracking.cooldown_s) &&
           tracking.cooldown_s >= 0.0 && std::isfinite(tracking.minimum_confidence) &&
           tracking.minimum_confidence >= 0.0 && tracking.minimum_confidence <= 1.0 &&
           std::isfinite(tracking.minimum_updraft_ms) && tracking.minimum_updraft_ms > 0.0 &&
           std::isfinite(tracking.minimum_condensate_path_kgm2) && tracking.minimum_condensate_path_kgm2 > 0.0 &&
           std::isfinite(tracking.updraft_helicity_bottom_m_agl) && tracking.updraft_helicity_bottom_m_agl >= 0.0 &&
           std::isfinite(tracking.updraft_helicity_top_m_agl) &&
           tracking.updraft_helicity_top_m_agl > tracking.updraft_helicity_bottom_m_agl &&
           std::isfinite(tracking.minimum_absolute_updraft_helicity_m2s2) &&
           tracking.minimum_absolute_updraft_helicity_m2s2 > 0.0 && tracking.minimum_component_cells > 0 &&
           std::isfinite(tracking.minimum_core_area_m2) && tracking.minimum_core_area_m2 > 0.0 &&
           std::isfinite(tracking.maximum_motion_speed_ms) && tracking.maximum_motion_speed_ms > 0.0 &&
           std::isfinite(tracking.maximum_motion_acceleration_ms2) && tracking.maximum_motion_acceleration_ms2 >= 0.0 &&
           tracking.minimum_continuity_samples >= 2 &&
           tracking.minimum_continuity_samples <=
               static_cast<int>(tmv::diagnostics::storm_continuity_history_capacity) &&
           tracking.lost_after_samples > 0 && tracking.topology_confirmation_samples > 0 && valid_branch_preference &&
           std::isfinite(tracking.branch_velocity_uncertainty_ms) && tracking.branch_velocity_uncertainty_ms > 0.0 &&
           (!tracking.enabled || (tracking.inner_margin_m > 0.0 && tracking.minimum_confidence > 0.0)) &&
           (!tracking.recycle_domain || (tracking.enabled && tracking.maximum_shift_cells > 0));
}

bool valid_domain_motion_config(const RuntimeConfigurationState& configuration) noexcept
{
    using tmv::init::config::DomainMotionMode;
    using tmv::init::config::ExecutionMode;
    const bool  valid_mode = configuration.domain_motion_mode == DomainMotionMode::Stationary ||
                             configuration.domain_motion_mode == DomainMotionMode::ConstantVelocity ||
                             configuration.domain_motion_mode == DomainMotionMode::TrackedWindow ||
                             configuration.domain_motion_mode == DomainMotionMode::MovingNest;
    const auto& origin     = configuration.domain_frame.earth_origin_at_t0_m;
    const auto& velocity   = configuration.domain_frame.velocity_ms;
    const auto& nesting    = configuration.nesting;
    const bool  valid_nesting =
        nesting.parent_ratio_x >= 2 && nesting.parent_ratio_y >= 2 &&
        nesting.parent_time_step_ratio >= std::max(nesting.parent_ratio_x, nesting.parent_ratio_y) &&
        (nesting.feedback == tmv::init::config::NestingFeedback::OneWay ||
         nesting.feedback == tmv::init::config::NestingFeedback::TwoWay) &&
        std::isfinite(nesting.relaxation_e_folding_time_s) && nesting.relaxation_e_folding_time_s > 0.0 &&
        (!nesting.enabled ||
         (nesting.child_origin_parent_cell_i >= 0 && nesting.child_origin_parent_cell_j >= 0 &&
          nesting.child_cells_x > 0 && nesting.child_cells_y > 0 &&
          nesting.child_cells_x % nesting.parent_ratio_x == 0 && nesting.child_cells_y % nesting.parent_ratio_y == 0 &&
          nesting.relaxation_width_child_cells > 0 &&
          2LL * nesting.relaxation_width_child_cells < std::min(nesting.child_cells_x, nesting.child_cells_y) &&
          nesting.boundary_stencil_parent_cells > 0));
    const bool valid_execution = configuration.execution_mode == ExecutionMode::Exploratory ||
                                 configuration.execution_mode == ExecutionMode::Qualified;
    return valid_mode && valid_execution && valid_nesting && std::isfinite(origin.x_m) && std::isfinite(origin.y_m) &&
           std::isfinite(velocity.u_ms) && std::isfinite(velocity.v_ms) &&
           (configuration.domain_motion_mode != DomainMotionMode::Stationary ||
            (velocity.u_ms == 0.0 && velocity.v_ms == 0.0));
}

bool valid_storm_selected_partition(const SimulationRuntimeState& runtime) noexcept
{
    if (!runtime.storm_selected_partition_available)
        return true;
    if (runtime.storm_selected_partition_nx == 0 || runtime.storm_selected_partition_ny == 0 ||
        runtime.storm_selected_partition_nx >
            std::numeric_limits<std::size_t>::max() / runtime.storm_selected_partition_ny)
        return false;
    const std::size_t expected = runtime.storm_selected_partition_nx * runtime.storm_selected_partition_ny;
    if (runtime.storm_selected_partition.size() != expected || runtime.storm_tracking.sample_count == 0)
        return false;
    std::size_t selected_cells = 0;
    for (std::uint8_t value : runtime.storm_selected_partition)
    {
        if (value > 1U)
            return false;
        selected_cells += value;
    }
    return selected_cells > 0;
}

bool valid_storm_partition_authority(const SimulationStorageState& state) noexcept
{
    const auto& runtime = state.runtime;
    if (!runtime.storm_selected_partition_available)
        return true;
    if (state.configuration.domain_motion_mode != tmv::init::config::DomainMotionMode::MovingNest ||
        !state.configuration.nesting.enabled)
        return true;
    return runtime.storm_selected_partition_nx == static_cast<std::size_t>(state.configuration.nesting.child_cells_x) &&
           runtime.storm_selected_partition_ny == static_cast<std::size_t>(state.configuration.nesting.child_cells_y);
}

std::uint32_t tracer_roundoff_float_bits(float value) noexcept
{
    static_assert(sizeof(float) == sizeof(std::uint32_t));
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

bool valid_tracer_roundoff_report(const PositiveDefiniteScalarRoundoffReport& report) noexcept
{
    if (!std::isfinite(report.physical_inventory_correction_kg) || report.physical_inventory_correction_kg < 0.0 ||
        report.thermodynamic_energy_correction_j != 0.0 ||
        report.negative_zero_count > std::numeric_limits<std::size_t>::max() - report.negative_subnormal_count)
    {
        return false;
    }
    const std::size_t canonicalization_count = report.negative_subnormal_count + report.negative_zero_count;
    if (report.closure_canonicalization_count > canonicalization_count ||
        (report.negative_subnormal_count == 0 && report.physical_inventory_correction_kg != 0.0))
    {
        return false;
    }
    const std::uint32_t minimum_bits = tracer_roundoff_float_bits(report.most_negative_repaired_value);
    const bool          minimum_is_negative_subnormal =
        (minimum_bits & 0x80000000U) != 0U && (minimum_bits & 0x7f800000U) == 0U && (minimum_bits & 0x007fffffU) != 0U;
    return report.negative_subnormal_count == 0 ? minimum_bits == 0U : minimum_is_negative_subnormal;
}

bool zero_tracer_roundoff_report(const PositiveDefiniteScalarRoundoffReport& report) noexcept
{
    return report.physical_inventory_correction_kg == 0.0 && report.thermodynamic_energy_correction_j == 0.0 &&
           tracer_roundoff_float_bits(report.most_negative_repaired_value) == 0U &&
           report.negative_subnormal_count == 0 && report.negative_zero_count == 0 &&
           report.closure_canonicalization_count == 0 && !report.complete;
}

bool valid_tracer_roundoff_checkpoint_runtime(const AdvectionRuntimeState& runtime) noexcept
{
    const auto& last       = runtime.tracer_roundoff_report;
    const auto& cumulative = runtime.cumulative_tracer_roundoff_report;
    if (!valid_tracer_roundoff_report(last) || !valid_tracer_roundoff_report(cumulative))
    {
        return false;
    }
    if (runtime.tracer_roundoff_report_step_count == 0)
    {
        return zero_tracer_roundoff_report(last) && zero_tracer_roundoff_report(cumulative);
    }
    if (!last.complete || cumulative.complete != runtime.tracer_roundoff_cumulative_history_known ||
        cumulative.negative_subnormal_count < last.negative_subnormal_count ||
        cumulative.negative_zero_count < last.negative_zero_count ||
        cumulative.closure_canonicalization_count < last.closure_canonicalization_count ||
        cumulative.physical_inventory_correction_kg < last.physical_inventory_correction_kg)
    {
        return false;
    }
    const std::uint32_t last_fraction = tracer_roundoff_float_bits(last.most_negative_repaired_value) & 0x007fffffU;
    const std::uint32_t cumulative_fraction =
        tracer_roundoff_float_bits(cumulative.most_negative_repaired_value) & 0x007fffffU;
    return cumulative_fraction >= last_fraction;
}

template <typename Enum>
void encode_enum(Encoder& out, Enum value)
{
    out.integer(static_cast<std::int32_t>(value));
}

template <typename Enum>
bool decode_enum(Decoder& in, Enum& value)
{
    std::int32_t encoded = 0;
    if (!in.integer(encoded))
        return false;
    value = static_cast<Enum>(encoded);
    return true;
}

template <std::size_t Capacity>
void encode_provenance_text(Encoder& out, const tmv::domain::DomainRecyclingProvenanceText<Capacity>& value)
{
    out.string(std::string(value.view()));
}

template <std::size_t Capacity>
bool decode_provenance_text(Decoder& in, tmv::domain::DomainRecyclingProvenanceText<Capacity>& value)
{
    std::string text;
    if (!in.string(text) || text.empty() || text.size() > Capacity)
        return false;
    value = {};
    std::copy(text.begin(), text.end(), value.bytes.begin());
    value.size = static_cast<std::uint16_t>(text.size());
    return true;
}

bool private_recycling_owners_match(const MicrophysicsRuntimeState&                          runtime,
                                    const tmv::domain::DomainRecyclingTransactionProvenance& provenance)
{
    if (!provenance.private_inventory_complete)
        return true;
    const auto fields =
        runtime.scheme ? runtime.scheme->prognostic_scalars_for_transport() : MicrophysicsPrognosticScalarList{};
    if (fields.size() != provenance.retiring_private_inventory.count)
        return false;
    for (std::size_t f = 0; f < fields.size(); ++f)
        if (!fields[f].name || provenance.retiring_private_inventory.entries[f].name.view() != fields[f].name)
            return false;
    return true;
}

void encode_domain_recycling_provenance(Encoder& out, const tmv::domain::DomainRecyclingTransactionProvenance& value)
{
    out.boolean(value.available);
    if (!value.available)
        return;
    out.unsigned_value(value.sequence);
    encode_provenance_text(out, value.transaction_id);
    encode_provenance_text(out, value.publication_generation);
    encode_provenance_text(out, value.storage_format);
    out.unsigned_value(value.event_manifest_bytes);
    out.unsigned_value(value.event_manifest_fnv1a64);
    out.unsigned_value(value.previous_chain_fnv1a64);
    out.unsigned_value(value.committed_chain_fnv1a64);
    out.floating(value.simulation_time_s);
    out.floating(value.pre_shift_origin_t0_m.x_m);
    out.floating(value.pre_shift_origin_t0_m.y_m);
    out.floating(value.post_shift_origin_t0_m.x_m);
    out.floating(value.post_shift_origin_t0_m.y_m);
    out.floating(value.frame_velocity_ms.u_ms);
    out.floating(value.frame_velocity_ms.v_ms);
    out.floating(value.dx_m);
    out.floating(value.dy_m);
    out.integer(value.nr);
    out.integer(value.nth);
    out.integer(value.nz);
    out.integer(value.shift_i_cells);
    out.integer(value.shift_j_cells);
    out.integer(value.physical_i_begin);
    out.integer(value.physical_i_end);
    out.integer(value.physical_j_begin);
    out.integer(value.physical_j_end);
    out.integer(value.source_i_begin);
    out.integer(value.source_i_end);
    out.integer(value.source_j_begin);
    out.integer(value.source_j_end);
    const auto& c = value.inventories;
    out.floating(c.total_mass);
    out.floating(c.theta_volume);
    out.floating(c.q_volume_total);
    out.floating(c.dry_mass);
    out.floating(c.total_water);
    for (double mass : c.water_species_mass)
        out.floating(mass);
    out.floating(c.moist_total_mass);
    out.floating(c.thermal_energy);
    const auto& m = value.momentum;
    out.floating(m.momentum_x);
    out.floating(m.momentum_y);
    out.floating(m.momentum_z);
    out.floating(m.angular_momentum_z);
    out.floating(m.horizontal_magnitude);
    out.floating(m.total_magnitude);
    out.floating(m.total_mass);
    out.floating(m.reference_linear);
    out.floating(m.reference_angular);
    const auto& e = value.moist_mixture_energy;
    out.floating(e.dry_internal);
    out.floating(e.water_sensible);
    out.floating(e.phase_reference);
    out.floating(e.resolved_kinetic);
    out.floating(e.gravitational);
    out.floating(e.subgrid_tke);
    out.floating(e.total_water_mass);
    out.floating(e.total);
    out.floating(value.theta_storage_residual_energy_j);
    out.unsigned_value(value.physical_columns);
    out.unsigned_value(value.physical_cells);
    encode_provenance_text(out, value.shift_algorithm);
    encode_provenance_text(out, value.frame_algorithm);
    encode_provenance_text(out, value.swath_algorithm);
    encode_provenance_text(out, value.scalar_ledger_basis);
    encode_provenance_text(out, value.momentum_ledger_basis);
    encode_provenance_text(out, value.energy_ledger_basis);
    encode_provenance_text(out, value.theta_residual_ledger_basis);
    out.boolean(value.control_volume_exchange_complete);
    if (value.control_volume_exchange_complete)
    {
        const auto& ac = value.admitted_inventories;
        out.floating(ac.total_mass);
        out.floating(ac.theta_volume);
        out.floating(ac.q_volume_total);
        out.floating(ac.dry_mass);
        out.floating(ac.total_water);
        for (double mass : ac.water_species_mass)
            out.floating(mass);
        out.floating(ac.moist_total_mass);
        out.floating(ac.thermal_energy);
        const auto& am = value.admitted_momentum;
        out.floating(am.momentum_x);
        out.floating(am.momentum_y);
        out.floating(am.momentum_z);
        out.floating(am.angular_momentum_z);
        out.floating(am.horizontal_magnitude);
        out.floating(am.total_magnitude);
        out.floating(am.total_mass);
        out.floating(am.reference_linear);
        out.floating(am.reference_angular);
        const auto& ae = value.admitted_moist_mixture_energy;
        out.floating(ae.dry_internal);
        out.floating(ae.water_sensible);
        out.floating(ae.phase_reference);
        out.floating(ae.resolved_kinetic);
        out.floating(ae.gravitational);
        out.floating(ae.subgrid_tke);
        out.floating(ae.total_water_mass);
        out.floating(ae.total);
        out.floating(value.admitted_theta_storage_residual_energy_j);
        out.floating(value.retiring_fixed_earth_angular_momentum_z);
        out.floating(value.admitted_fixed_earth_angular_momentum_z);
        encode_provenance_text(out, value.control_volume_exchange_basis);
    }


}

bool decode_domain_recycling_provenance(Decoder&                                           in,
                                        tmv::domain::DomainRecyclingTransactionProvenance& value,
                                        bool                                               has_control_volume_exchange)
{
    value = {};
    if (!in.boolean(value.available))
        return false;
    if (!value.available)
        return true;
    auto& c = value.inventories;
    auto& m = value.momentum;
    auto& e = value.moist_mixture_energy;
    if (!in.unsigned_value(value.sequence) || !decode_provenance_text(in, value.transaction_id) ||
        !decode_provenance_text(in, value.publication_generation) ||
        !decode_provenance_text(in, value.storage_format) || !in.unsigned_value(value.event_manifest_bytes) ||
        !in.unsigned_value(value.event_manifest_fnv1a64) || !in.unsigned_value(value.previous_chain_fnv1a64) ||
        !in.unsigned_value(value.committed_chain_fnv1a64) || !in.floating(value.simulation_time_s) ||
        !in.floating(value.pre_shift_origin_t0_m.x_m) || !in.floating(value.pre_shift_origin_t0_m.y_m) ||
        !in.floating(value.post_shift_origin_t0_m.x_m) || !in.floating(value.post_shift_origin_t0_m.y_m) ||
        !in.floating(value.frame_velocity_ms.u_ms) || !in.floating(value.frame_velocity_ms.v_ms) ||
        !in.floating(value.dx_m) || !in.floating(value.dy_m) || !in.integer(value.nr) || !in.integer(value.nth) ||
        !in.integer(value.nz) || !in.integer(value.shift_i_cells) || !in.integer(value.shift_j_cells) ||
        !in.integer(value.physical_i_begin) || !in.integer(value.physical_i_end) ||
        !in.integer(value.physical_j_begin) || !in.integer(value.physical_j_end) || !in.integer(value.source_i_begin) ||
        !in.integer(value.source_i_end) || !in.integer(value.source_j_begin) || !in.integer(value.source_j_end) ||
        !in.floating(c.total_mass) || !in.floating(c.theta_volume) || !in.floating(c.q_volume_total) ||
        !in.floating(c.dry_mass) || !in.floating(c.total_water))
        return false;
    for (double& mass : c.water_species_mass)
        if (!in.floating(mass))
            return false;
    if (!(in.floating(c.moist_total_mass) && in.floating(c.thermal_energy) && in.floating(m.momentum_x) &&
          in.floating(m.momentum_y) && in.floating(m.momentum_z) && in.floating(m.angular_momentum_z) &&
          in.floating(m.horizontal_magnitude) && in.floating(m.total_magnitude) && in.floating(m.total_mass) &&
          in.floating(m.reference_linear) && in.floating(m.reference_angular) && in.floating(e.dry_internal) &&
          in.floating(e.water_sensible) && in.floating(e.phase_reference) && in.floating(e.resolved_kinetic) &&
          in.floating(e.gravitational) && in.floating(e.subgrid_tke) && in.floating(e.total_water_mass) &&
          in.floating(e.total) && in.floating(value.theta_storage_residual_energy_j) &&
          in.unsigned_value(value.physical_columns) && in.unsigned_value(value.physical_cells) &&
          decode_provenance_text(in, value.shift_algorithm) && decode_provenance_text(in, value.frame_algorithm) &&
          decode_provenance_text(in, value.swath_algorithm) && decode_provenance_text(in, value.scalar_ledger_basis) &&
          decode_provenance_text(in, value.momentum_ledger_basis) &&
          decode_provenance_text(in, value.energy_ledger_basis) &&
          decode_provenance_text(in, value.theta_residual_ledger_basis)))
        return false;
    if (has_control_volume_exchange)
    {
        if (!in.boolean(value.control_volume_exchange_complete))
            return false;
        if (value.control_volume_exchange_complete)
        {
            auto& ac = value.admitted_inventories;
            auto& am = value.admitted_momentum;
            auto& ae = value.admitted_moist_mixture_energy;
            if (!in.floating(ac.total_mass) || !in.floating(ac.theta_volume) || !in.floating(ac.q_volume_total) ||
                !in.floating(ac.dry_mass) || !in.floating(ac.total_water))
                return false;
            for (double& mass : ac.water_species_mass)
                if (!in.floating(mass))
                    return false;
            if (!in.floating(ac.moist_total_mass) || !in.floating(ac.thermal_energy) || !in.floating(am.momentum_x) ||
                !in.floating(am.momentum_y) || !in.floating(am.momentum_z) || !in.floating(am.angular_momentum_z) ||
                !in.floating(am.horizontal_magnitude) || !in.floating(am.total_magnitude) ||
                !in.floating(am.total_mass) || !in.floating(am.reference_linear) ||
                !in.floating(am.reference_angular) || !in.floating(ae.dry_internal) ||
                !in.floating(ae.water_sensible) || !in.floating(ae.phase_reference) ||
                !in.floating(ae.resolved_kinetic) || !in.floating(ae.gravitational) || !in.floating(ae.subgrid_tke) ||
                !in.floating(ae.total_water_mass) || !in.floating(ae.total) ||
                !in.floating(value.admitted_theta_storage_residual_energy_j) ||
                !in.floating(value.retiring_fixed_earth_angular_momentum_z) ||
                !in.floating(value.admitted_fixed_earth_angular_momentum_z) ||
                !decode_provenance_text(in, value.control_volume_exchange_basis))
                return false;
        }
    }
    return tmv::domain::valid_domain_recycling_transaction_provenance(value);
}

void encode_double_vector(Encoder& out, const std::vector<double>& values)
{
    out.unsigned_value<std::uint64_t>(values.size());
    for (double value : values)
        out.floating(value);
}

bool decode_double_vector(Decoder& in, std::vector<double>& values)
{
    std::uint64_t size = 0;
    if (!in.unsigned_value(size) || size > wire::max_elements || size > in.remaining() / sizeof(double))
        return false;
    values.resize(static_cast<std::size_t>(size));
    for (double& value : values)
        if (!in.floating(value))
            return false;
    return true;
}

void encode_float_vector(Encoder& out, const std::vector<float>& values)
{
    out.unsigned_value<std::uint64_t>(values.size());
    for (float value : values)
        out.floating(value);
}

bool decode_float_vector(Decoder& in, std::vector<float>& values)
{
    std::uint64_t size = 0;
    if (!in.unsigned_value(size) || size > wire::max_elements || size > in.remaining() / sizeof(float))
        return false;
    values.resize(static_cast<std::size_t>(size));
    for (float& value : values)
        if (!in.floating(value))
            return false;
    return true;
}

void encode_string_vector(Encoder& out, const std::vector<std::string>& values)
{
    out.unsigned_value<std::uint64_t>(values.size());
    for (const auto& value : values)
        out.string(value);
}

bool decode_string_vector(Decoder& in, std::vector<std::string>& values)
{
    std::uint64_t size = 0;
    if (!in.unsigned_value(size) || size > wire::max_elements || size > in.remaining() / sizeof(std::uint32_t))
        return false;
    values.resize(static_cast<std::size_t>(size));
    for (auto& value : values)
        if (!in.string(value))
            return false;
    return true;
}

void encode_string_double_map(Encoder& out, const std::unordered_map<std::string, double>& values)
{
    std::map<std::string, double> ordered(values.begin(), values.end());
    out.unsigned_value<std::uint64_t>(ordered.size());
    for (const auto& entry : ordered)
    {
        out.string(entry.first);
        out.floating(entry.second);
    }
}

bool decode_string_double_map(Decoder& in, std::unordered_map<std::string, double>& values)
{
    std::uint64_t size = 0;
    if (!in.unsigned_value(size) || size > wire::max_elements ||
        size > in.remaining() / (sizeof(std::uint32_t) + sizeof(double)))
        return false;
    values.clear();
    for (std::uint64_t index = 0; index < size; ++index)
    {
        std::string key;
        double      value = 0.0;
        if (!in.string(key) || !in.floating(value) || !values.emplace(std::move(key), value).second)
            return false;
    }
    return true;
}

void encode_field(Encoder& out, const Field3D& field)
{
    out.integer(field.size_r());
    out.integer(field.size_th());
    out.integer(field.size_z());
    out.unsigned_value<std::uint64_t>(field.size());
    out.floating_array(field.data(), field.size());
}

bool decode_field(Decoder& in, Field3D& field)
{
    std::int32_t  nr = 0, nth = 0, nz = 0;
    std::uint64_t count = 0;
    if (!in.integer(nr) || !in.integer(nth) || !in.integer(nz) || !in.unsigned_value(count) || nr < 0 || nth < 0 ||
        nz < 0)
        return false;
    const std::uint64_t expected =
        static_cast<std::uint64_t>(nr) * static_cast<std::uint64_t>(nth) * static_cast<std::uint64_t>(nz);
    if (count != expected || count > wire::max_elements || count > in.remaining() / sizeof(float))
        return false;
    Field3D restored(nr, nth, nz);
    if (!in.floating_array(restored.data(), static_cast<std::size_t>(count)))
        return false;
    field = std::move(restored);
    return true;
}

void encode_matrix(Encoder& out, const std::vector<std::vector<double>>& matrix)
{
    out.unsigned_value<std::uint64_t>(matrix.size());
    for (const auto& row : matrix)
        encode_double_vector(out, row);
}

bool decode_matrix(Decoder& in, std::vector<std::vector<double>>& matrix)
{
    std::uint64_t rows = 0;
    if (!in.unsigned_value(rows) || rows > wire::max_elements || rows > in.remaining() / sizeof(std::uint64_t))
        return false;
    matrix.resize(static_cast<std::size_t>(rows));
    for (auto& row : matrix)
        if (!decode_double_vector(in, row))
            return false;
    if (!matrix.empty())
    {
        const std::size_t columns = matrix.front().size();
        if (!std::all_of(matrix.begin(), matrix.end(), [&](const auto& row) { return row.size() == columns; }))
            return false;
    }
    return true;
}

void encode_portable_state(Encoder& out, const PortableSchemeState& state)
{
    out.unsigned_value(state.version);
    out.unsigned_value<std::uint64_t>(state.fields.size());
    for (const auto& field : state.fields)
        encode_field(out, field);
    out.unsigned_value<std::uint64_t>(state.integers.size());
    for (auto value : state.integers)
        out.unsigned_value(value);
    encode_double_vector(out, state.reals);
}

bool decode_portable_state(Decoder& in, PortableSchemeState& state)
{
    std::uint64_t field_count = 0, integer_count = 0;
    if (!in.unsigned_value(state.version) || !in.unsigned_value(field_count) || field_count > 256)
        return false;
    state.fields.resize(static_cast<std::size_t>(field_count));
    for (auto& field : state.fields)
        if (!decode_field(in, field))
            return false;
    if (!in.unsigned_value(integer_count) || integer_count > 256)
        return false;
    state.integers.resize(static_cast<std::size_t>(integer_count));
    for (auto& value : state.integers)
        if (!in.unsigned_value(value))
            return false;
    return decode_double_vector(in, state.reals) && state.reals.size() <= 1024;
}

void encode_field_bounds(Encoder& out, const tmv::FieldBounds& value)
{
    out.boolean(value.has_min);
    out.boolean(value.has_max);
    out.floating(value.min_value);
    out.floating(value.max_value);
}

bool decode_field_bounds(Decoder& in, tmv::FieldBounds& value)
{
    return in.boolean(value.has_min) && in.boolean(value.has_max) && in.floating(value.min_value) &&
           in.floating(value.max_value);
}

void encode_configuration(Encoder& out, const RuntimeConfigurationState& value)
{
    const double wind[] = {value.wind_profile.u_sfc,
                           value.wind_profile.v_sfc,
                           value.wind_profile.u_1km,
                           value.wind_profile.v_1km,
                           value.wind_profile.u_6km,
                           value.wind_profile.v_6km};
    for (double component : wind)
        out.floating(component);
    const double environment[] = {value.cape_target,
                                  value.sfc_theta_k,
                                  value.sfc_qv_kgkg,
                                  value.tropopause_z_m,
                                  value.bubble_center_x_m,
                                  value.bubble_center_y_m,
                                  value.bubble_center_z_m,
                                  value.bubble_radius_m,
                                  value.bubble_dtheta_k};
    for (double scalar : environment)
        out.floating(scalar);
    out.boolean(value.sounding_enabled);
    out.boolean(value.sounding_allow_placeholder_profiles);
    const auto& sounding = value.sounding_config;
    out.string(sounding.scheme_id);
    out.string(sounding.file_path);
    out.boolean(sounding.use_fallback_profiles);
    out.floating(sounding.interpolation_method);
    out.boolean(sounding.extrapolate_below_ground);
    out.boolean(sounding.extrapolate_above_top);
    out.floating(sounding.min_pressure_hpa);
    out.floating(sounding.max_pressure_hpa);
    out.floating(sounding.min_temperature_k);
    out.floating(sounding.max_temperature_k);

    const auto& source = value.sounding_source;
    encode_enum(out, source.type);
    out.floating(source.parametric.cape_target_jkg);
    out.floating(source.parametric.surface_theta_k);
    out.floating(source.parametric.surface_qv_kgkg);
    out.floating(source.parametric.tropopause_z_m);
    out.string(source.file.path);
    out.string(source.file.scheme_id);
    out.boolean(source.file.require_winds);
    out.boolean(source.file.use_fallback_profiles);
    const auto& targets = source.targets;
    out.floating(targets.target_cape_jkg);
    out.floating(targets.target_cin_jkg);
    out.floating(targets.target_lcl_m);
    out.floating(targets.target_lfc_m);
    out.floating(targets.target_el_m);
    out.floating(targets.surface_theta_k);
    out.floating(targets.surface_pressure_pa);
    out.floating(targets.moisture_qv_kgkg_override);

    encode_enum(out, value.hodograph_source.type);
    const auto&  anchors         = value.hodograph_source.wk_anchors;
    const double anchor_values[] = {
        anchors.u_sfc_ms, anchors.v_sfc_ms, anchors.u_1km_ms, anchors.v_1km_ms, anchors.u_6km_ms, anchors.v_6km_ms};
    for (double scalar : anchor_values)
        out.floating(scalar);
    encode_enum(out, value.trigger_source.type);
    const auto& vortex = value.trigger_source.vortex_seed;
    out.floating(vortex.r_max_m);
    out.floating(vortex.v_max_ms);
    out.floating(vortex.z_top_m);
    out.floating(vortex.center_x_m);
    out.floating(vortex.center_y_m);
    out.floating(vortex.center_z_m);

    out.string(value.microphysics_scheme);
    const auto& microphysics = value.microphysics_config;
    out.floating(microphysics.shared.cloud_water_threshold_kgkg);
    out.floating(microphysics.shared.autoconversion_rate_s_inv);
    out.floating(microphysics.kessler.accretion_coefficient);
    out.floating(microphysics.kessler.evaporation_efficiency);
    out.floating(microphysics.lin.accretion_coefficient);
    out.floating(microphysics.lin.evaporation_rate_s_inv);
    out.floating(microphysics.thompson.evaporation_rate_s_inv);
    out.floating(microphysics.milbrandt.evaporation_rate_s_inv);
    out.boolean(microphysics.milbrandt.triple_moment);
    out.boolean(microphysics.milbrandt.hail_processes);
    out.string(value.dynamics_scheme_name);
    encode_enum(out, value.coordinate_system);
    encode_enum(out, value.stagger_type);
    encode_enum(out, value.lateral_bc_west);
    encode_enum(out, value.lateral_bc_east);
    encode_enum(out, value.lateral_bc_south);
    encode_enum(out, value.lateral_bc_north);
    encode_enum(out, value.bottom_bc);
    encode_enum(out, value.top_bc);
    out.boolean(value.damping_enabled);
    out.floating(value.damping_z_d);
    out.floating(value.damping_alpha);
    out.integer(value.lateral_sponge_width);
    out.floating(value.lateral_sponge_alpha);
    encode_enum(out, value.damping_mode);
    out.boolean(value.w_only_top_sponge);
    out.integer(value.w_only_depth_cells);
    out.floating(value.w_only_alpha_max);
    out.boolean(value.lateral_sponge_enabled);
    out.boolean(value.slow_rk2);
    out.floating(value.lid_bc_gamma);
    out.boolean(value.axis_u_m1_filter);
    out.boolean(value.gpu_apply_microphysics);
    out.boolean(value.gpu_apply_slow);
    out.boolean(value.gpu_apply_sgs);
    out.boolean(value.coriolis_enabled);
    encode_enum(out, value.coriolis_mode);
    out.floating(value.coriolis_latitude_deg);
    encode_enum(out, value.log_profile);
    out.boolean(value.perf_timing_enabled);
    out.integer(value.perf_report_every_steps);
    out.integer(value.perf_warmup_steps);
    out.integer(value.scientific_diagnostics_every_steps);
    encode_enum(out, value.validation_policy.mode);
    encode_enum(out, value.validation_policy.fail_on);
    encode_enum(out, value.validation_policy.strict_scope);
    std::map<std::string, tmv::FieldBounds> bounds(value.validation_policy.field_overrides.begin(),
                                                   value.validation_policy.field_overrides.end());
    out.unsigned_value<std::uint64_t>(bounds.size());
    for (const auto& entry : bounds)
    {
        out.string(entry.first);
        encode_field_bounds(out, entry.second);
    }
    out.string(value.validation_report_path);
    const auto& tracking = value.storm_tracking;
    out.boolean(tracking.enabled);
    out.floating(tracking.interval_s);
    out.floating(tracking.inner_margin_m);
    out.integer(tracking.maximum_shift_cells);
    out.floating(tracking.cooldown_s);
    out.floating(tracking.minimum_confidence);
    out.floating(tracking.minimum_updraft_ms);
    out.floating(tracking.minimum_condensate_path_kgm2);
    out.integer(tracking.minimum_component_cells);
    out.floating(value.configured_dt_large_s);
    out.floating(tracking.maximum_motion_speed_ms);
    out.floating(tracking.maximum_motion_acceleration_ms2);
    out.integer(tracking.minimum_continuity_samples);
    out.integer(tracking.lost_after_samples);
    out.floating(tracking.updraft_helicity_bottom_m_agl);
    out.floating(tracking.updraft_helicity_top_m_agl);
    out.floating(tracking.minimum_absolute_updraft_helicity_m2s2);
    out.integer(tracking.topology_confirmation_samples);
    encode_enum(out, tracking.branch_preference);
    out.floating(tracking.branch_velocity_uncertainty_ms);
    encode_enum(out, value.domain_motion_mode);
    out.floating(value.domain_frame.earth_origin_at_t0_m.x_m);
    out.floating(value.domain_frame.earth_origin_at_t0_m.y_m);
    out.floating(value.domain_frame.velocity_ms.u_ms);
    out.floating(value.domain_frame.velocity_ms.v_ms);
    out.floating(tracking.minimum_core_area_m2);
    out.boolean(tracking.recycle_domain);
    out.integer(value.nesting.parent_ratio_x);
    out.integer(value.nesting.parent_ratio_y);
    out.integer(value.nesting.parent_time_step_ratio);
    encode_enum(out, value.nesting.feedback);
    out.boolean(value.nesting.enabled);
    out.integer(value.nesting.child_origin_parent_cell_i);
    out.integer(value.nesting.child_origin_parent_cell_j);
    out.integer(value.nesting.child_cells_x);
    out.integer(value.nesting.child_cells_y);
    out.integer(value.nesting.relaxation_width_child_cells);
    out.integer(value.nesting.boundary_stencil_parent_cells);
    out.floating(value.nesting.relaxation_e_folding_time_s);
    encode_enum(out, value.execution_mode);
    out.floating(value.microphysics_config.thompson.water_friendly_aerosol_surface_m3);
    out.floating(value.microphysics_config.thompson.ice_friendly_aerosol_surface_m3);
    out.floating(value.bubble_cloud_water_kgkg);
}

bool decode_configuration(Decoder& in, RuntimeConfigurationState& value, std::uint32_t section_version)
{
    double* wind[] = {&value.wind_profile.u_sfc,
                      &value.wind_profile.v_sfc,
                      &value.wind_profile.u_1km,
                      &value.wind_profile.v_1km,
                      &value.wind_profile.u_6km,
                      &value.wind_profile.v_6km};
    for (double* component : wind)
        if (!in.floating(*component))
            return false;
    double* environment[] = {&value.cape_target,
                             &value.sfc_theta_k,
                             &value.sfc_qv_kgkg,
                             &value.tropopause_z_m,
                             &value.bubble_center_x_m,
                             &value.bubble_center_y_m,
                             &value.bubble_center_z_m,
                             &value.bubble_radius_m,
                             &value.bubble_dtheta_k};
    for (double* scalar : environment)
        if (!in.floating(*scalar))
            return false;
    auto& sounding = value.sounding_config;
    if (!in.boolean(value.sounding_enabled) || !in.boolean(value.sounding_allow_placeholder_profiles) ||
        !in.string(sounding.scheme_id) || !in.string(sounding.file_path) ||
        !in.boolean(sounding.use_fallback_profiles) || !in.floating(sounding.interpolation_method) ||
        !in.boolean(sounding.extrapolate_below_ground) || !in.boolean(sounding.extrapolate_above_top) ||
        !in.floating(sounding.min_pressure_hpa) || !in.floating(sounding.max_pressure_hpa) ||
        !in.floating(sounding.min_temperature_k) || !in.floating(sounding.max_temperature_k))
        return false;
    auto& source = value.sounding_source;
    if (!decode_enum(in, source.type) || !in.floating(source.parametric.cape_target_jkg) ||
        !in.floating(source.parametric.surface_theta_k) || !in.floating(source.parametric.surface_qv_kgkg) ||
        !in.floating(source.parametric.tropopause_z_m) || !in.string(source.file.path) ||
        !in.string(source.file.scheme_id) || !in.boolean(source.file.require_winds) ||
        !in.boolean(source.file.use_fallback_profiles))
        return false;
    auto&   targets         = source.targets;
    double* target_values[] = {&targets.target_cape_jkg,
                               &targets.target_cin_jkg,
                               &targets.target_lcl_m,
                               &targets.target_lfc_m,
                               &targets.target_el_m,
                               &targets.surface_theta_k,
                               &targets.surface_pressure_pa,
                               &targets.moisture_qv_kgkg_override};
    for (double* scalar : target_values)
        if (!in.floating(*scalar))
            return false;
    if (!decode_enum(in, value.hodograph_source.type))
        return false;
    auto&   anchors         = value.hodograph_source.wk_anchors;
    double* anchor_values[] = {&anchors.u_sfc_ms,
                               &anchors.v_sfc_ms,
                               &anchors.u_1km_ms,
                               &anchors.v_1km_ms,
                               &anchors.u_6km_ms,
                               &anchors.v_6km_ms};
    for (double* scalar : anchor_values)
        if (!in.floating(*scalar))
            return false;
    auto& trigger = value.trigger_source;
    if (!decode_enum(in, trigger.type))
        return false;
    double* vortex[] = {&trigger.vortex_seed.r_max_m,
                        &trigger.vortex_seed.v_max_ms,
                        &trigger.vortex_seed.z_top_m,
                        &trigger.vortex_seed.center_x_m,
                        &trigger.vortex_seed.center_y_m,
                        &trigger.vortex_seed.center_z_m};
    for (double* scalar : vortex)
        if (!in.floating(*scalar))
            return false;
    if (!in.string(value.microphysics_scheme))
        return false;
    if (section_version >= 2)
    {
        auto& microphysics = value.microphysics_config;
        if (!in.floating(microphysics.shared.cloud_water_threshold_kgkg) ||
            !in.floating(microphysics.shared.autoconversion_rate_s_inv) ||
            !in.floating(microphysics.kessler.accretion_coefficient) ||
            !in.floating(microphysics.kessler.evaporation_efficiency) ||
            !in.floating(microphysics.lin.accretion_coefficient) ||
            !in.floating(microphysics.lin.evaporation_rate_s_inv))
            return false;
        if (section_version >= 3 && (!in.floating(microphysics.thompson.evaporation_rate_s_inv) ||
                                     !in.floating(microphysics.milbrandt.evaporation_rate_s_inv)))
            return false;
        if (!in.boolean(microphysics.milbrandt.triple_moment) || !in.boolean(microphysics.milbrandt.hail_processes))
            return false;
    }
    if (!in.string(value.dynamics_scheme_name) || !decode_enum(in, value.coordinate_system) ||
        !decode_enum(in, value.stagger_type) || !decode_enum(in, value.lateral_bc_west) ||
        !decode_enum(in, value.lateral_bc_east) || !decode_enum(in, value.lateral_bc_south) ||
        !decode_enum(in, value.lateral_bc_north) || !decode_enum(in, value.bottom_bc) ||
        !decode_enum(in, value.top_bc) || !in.boolean(value.damping_enabled) || !in.floating(value.damping_z_d) ||
        !in.floating(value.damping_alpha) || !in.integer(value.lateral_sponge_width) ||
        !in.floating(value.lateral_sponge_alpha) || !decode_enum(in, value.damping_mode) ||
        !in.boolean(value.w_only_top_sponge) || !in.integer(value.w_only_depth_cells) ||
        !in.floating(value.w_only_alpha_max) || !in.boolean(value.lateral_sponge_enabled) ||
        !in.boolean(value.slow_rk2) || !in.floating(value.lid_bc_gamma) || !in.boolean(value.axis_u_m1_filter) ||
        !in.boolean(value.gpu_apply_microphysics) || !in.boolean(value.gpu_apply_slow) ||
        !in.boolean(value.gpu_apply_sgs) || !in.boolean(value.coriolis_enabled) ||
        !decode_enum(in, value.coriolis_mode) || !in.floating(value.coriolis_latitude_deg) ||
        !decode_enum(in, value.log_profile) || !in.boolean(value.perf_timing_enabled) ||
        !in.integer(value.perf_report_every_steps) || !in.integer(value.perf_warmup_steps) ||
        !in.integer(value.scientific_diagnostics_every_steps) || !decode_enum(in, value.validation_policy.mode) ||
        !decode_enum(in, value.validation_policy.fail_on) || !decode_enum(in, value.validation_policy.strict_scope))
        return false;
    std::uint64_t bounds_count = 0;
    if (!in.unsigned_value(bounds_count) || bounds_count > wire::max_elements)
        return false;
    value.validation_policy.field_overrides.clear();
    for (std::uint64_t index = 0; index < bounds_count; ++index)
    {
        std::string      key;
        tmv::FieldBounds bounds;
        if (!in.string(key) || !decode_field_bounds(in, bounds) ||
            !value.validation_policy.field_overrides.emplace(std::move(key), bounds).second)
            return false;
    }
    if (!in.string(value.validation_report_path))
        return false;
    if (section_version >= 4)
    {
        auto& tracking = value.storm_tracking;
        if (!in.boolean(tracking.enabled) || !in.floating(tracking.interval_s) ||
            !in.floating(tracking.inner_margin_m) || !in.integer(tracking.maximum_shift_cells) ||
            !in.floating(tracking.cooldown_s) || !in.floating(tracking.minimum_confidence) ||
            !in.floating(tracking.minimum_updraft_ms) || !in.floating(tracking.minimum_condensate_path_kgm2) ||
            !in.integer(tracking.minimum_component_cells))
        {
            return false;
        }
    }
    if (section_version >= 5 && !in.floating(value.configured_dt_large_s))
        return false;
    if (section_version >= 6)
    {
        auto& tracking = value.storm_tracking;
        if (!in.floating(tracking.maximum_motion_speed_ms) || !in.floating(tracking.maximum_motion_acceleration_ms2) ||
            !in.integer(tracking.minimum_continuity_samples) || !in.integer(tracking.lost_after_samples))
        {
            return false;
        }
    }
    if (section_version >= 7)
    {
        auto& tracking = value.storm_tracking;
        if (!in.floating(tracking.updraft_helicity_bottom_m_agl) || !in.floating(tracking.updraft_helicity_top_m_agl) ||
            !in.floating(tracking.minimum_absolute_updraft_helicity_m2s2))
        {
            return false;
        }
    }
    if (section_version >= 8)
    {
        auto& tracking = value.storm_tracking;
        if (!in.integer(tracking.topology_confirmation_samples) || !decode_enum(in, tracking.branch_preference) ||
            !in.floating(tracking.branch_velocity_uncertainty_ms))
        {
            return false;
        }
    }
    if (section_version >= 9)
    {
        if (!decode_enum(in, value.domain_motion_mode) || !in.floating(value.domain_frame.earth_origin_at_t0_m.x_m) ||
            !in.floating(value.domain_frame.earth_origin_at_t0_m.y_m) ||
            !in.floating(value.domain_frame.velocity_ms.u_ms) || !in.floating(value.domain_frame.velocity_ms.v_ms))
        {
            return false;
        }
    }
    if (section_version >= 10 && !in.floating(value.storm_tracking.minimum_core_area_m2))
        return false;
    if (section_version >= 11 && !in.boolean(value.storm_tracking.recycle_domain))
        return false;
    if (section_version >= 12 &&
        (!in.integer(value.nesting.parent_ratio_x) || !in.integer(value.nesting.parent_ratio_y) ||
         !in.integer(value.nesting.parent_time_step_ratio) || !decode_enum(in, value.nesting.feedback)))
    {
        return false;
    }
    if (section_version >= 13 &&
        (!in.boolean(value.nesting.enabled) || !in.integer(value.nesting.child_origin_parent_cell_i) ||
         !in.integer(value.nesting.child_origin_parent_cell_j) || !in.integer(value.nesting.child_cells_x) ||
         !in.integer(value.nesting.child_cells_y) || !in.integer(value.nesting.relaxation_width_child_cells) ||
         !in.integer(value.nesting.boundary_stencil_parent_cells) ||
         !in.floating(value.nesting.relaxation_e_folding_time_s)))
    {
        return false;
    }
    if (section_version >= 14 && !decode_enum(in, value.execution_mode))
        return false;
    if (section_version >= 15 && (!in.floating(value.microphysics_config.thompson.water_friendly_aerosol_surface_m3) ||
                                  !in.floating(value.microphysics_config.thompson.ice_friendly_aerosol_surface_m3)))
    {
        return false;
    }
    if (section_version >= 16)
    {
        if (!in.floating(value.bubble_cloud_water_kgkg) || !std::isfinite(value.bubble_cloud_water_kgkg) ||
            value.bubble_cloud_water_kgkg < 0.0)
        {
            return false;
        }
    }
    else
    {
        value.bubble_cloud_water_kgkg = 0.0;
    }
    if (!std::isfinite(value.microphysics_config.thompson.water_friendly_aerosol_surface_m3) ||
        value.microphysics_config.thompson.water_friendly_aerosol_surface_m3 <= 50.0e6 ||
        !std::isfinite(value.microphysics_config.thompson.ice_friendly_aerosol_surface_m3) ||
        value.microphysics_config.thompson.ice_friendly_aerosol_surface_m3 <= 0.5e6)
    {
        return false;
    }
    return valid_storm_tracking_config(value.storm_tracking) && valid_domain_motion_config(value);
}

void encode_grid(Encoder& out, const GridStorageState& value)
{
    out.integer(value.nr);
    out.integer(value.nth);
    out.integer(value.nz);
    out.floating(value.dx);
    out.floating(value.dz);
    out.floating(value.dt);
    out.floating(value.dy);
    const auto& geometry = value.geometry;
    encode_double_vector(out, geometry.r);
    encode_double_vector(out, geometry.theta);
    encode_double_vector(out, geometry.z);
    encode_double_vector(out, geometry.r_faces);
    encode_double_vector(out, geometry.theta_faces);
    encode_double_vector(out, geometry.z_faces);
    out.boolean(geometry.staggered);
    out.floating(geometry.inv_dr);
    out.floating(geometry.inv_dz);
    out.floating(geometry.inv_dtheta);
    out.floating(geometry.inv_2dr);
    out.floating(geometry.inv_2dz);
    out.floating(geometry.inv_2dtheta);
    out.floating(geometry.inv_dr2);
    out.floating(geometry.inv_dz2);
    encode_enum(out, geometry.coord);
}

bool decode_grid(Decoder& in, GridStorageState& value)
{
    std::int32_t nr = 0, nth = 0, nz = 0;
    auto&        geometry = value.geometry;
    if (!in.integer(nr) || !in.integer(nth) || !in.integer(nz) || nr < 0 || nth < 0 || nz < 0 ||
        !in.floating(value.dx) || !in.floating(value.dz) || !in.floating(value.dt) || !in.floating(value.dy) ||
        !decode_double_vector(in, geometry.r) || !decode_double_vector(in, geometry.theta) ||
        !decode_double_vector(in, geometry.z) || !decode_double_vector(in, geometry.r_faces) ||
        !decode_double_vector(in, geometry.theta_faces) || !decode_double_vector(in, geometry.z_faces) ||
        !in.boolean(geometry.staggered) || !in.floating(geometry.inv_dr) || !in.floating(geometry.inv_dz) ||
        !in.floating(geometry.inv_dtheta) || !in.floating(geometry.inv_2dr) || !in.floating(geometry.inv_2dz) ||
        !in.floating(geometry.inv_2dtheta) || !in.floating(geometry.inv_dr2) || !in.floating(geometry.inv_dz2) ||
        !decode_enum(in, geometry.coord))
        return false;
    value.nr  = nr;
    value.nth = nth;
    value.nz  = nz;
    return true;
}

template <std::size_t N>
void encode_fields(Encoder& out, const std::array<const Field3D*, N>& fields)
{
    for (const auto* field : fields)
        encode_field(out, *field);
}

template <std::size_t N>
bool decode_fields(Decoder& in, const std::array<Field3D*, N>& fields)
{
    for (auto* field : fields)
        if (!decode_field(in, *field))
            return false;
    return true;
}

std::array<const Field3D*, 15> prognostic_fields(const PrognosticFieldStorage& f)
{
    return tmv::fields::pointers(f, tmv::fields::prognostic);
}

std::array<Field3D*, 15> prognostic_fields(PrognosticFieldStorage& f)
{
    return tmv::fields::pointers(f, tmv::fields::prognostic);
}

std::array<Field3D*, 6> tendency_fields_v1(PhysicsTendencyFieldStorage& f)
{
    return {&f.dtheta_dt_rad, &f.dtheta_dt_pbl, &f.dqv_dt_pbl, &f.du_dt_pbl, &f.dv_dt_pbl, &f.dtke_dt_pbl};
}

std::array<const Field3D*, 8> tendency_fields(const PhysicsTendencyFieldStorage& f)
{
    return tmv::fields::pointers(f, tmv::fields::physics_tendencies);
}

std::array<Field3D*, 8> tendency_fields(PhysicsTendencyFieldStorage& f)
{
    return tmv::fields::pointers(f, tmv::fields::physics_tendencies);
}

std::array<Field3D*, 12> nested_fields_v1(NestedGridStorage& f)
{
    return {&f.rho, &f.p, &f.u, &f.w, &f.v, &f.theta, &f.qv, &f.qc, &f.qr, &f.qh, &f.qg, &f.tracer};
}

std::array<const Field3D*, 14> nested_fields(const NestedGridStorage& f)
{
    return tmv::fields::pointers(f, tmv::fields::nested);
}

std::array<Field3D*, 14> nested_fields(NestedGridStorage& f)
{
    return tmv::fields::pointers(f, tmv::fields::nested);
}

std::array<const Field3D*, 12> diagnostic_fields(const DiagnosticFieldStorage& f)
{
    return tmv::fields::pointers(f, tmv::fields::diagnostics);
}

std::array<Field3D*, 12> diagnostic_fields(DiagnosticFieldStorage& f)
{
    return tmv::fields::pointers(f, tmv::fields::diagnostics);
}

void encode_terrain_field(Encoder& out, const TerrainField3D& field)
{
    out.integer(field.nr);
    out.integer(field.nth);
    out.integer(field.nz);
    encode_double_vector(out, field.values);
}

bool decode_terrain_field(Decoder& in, TerrainField3D& field)
{
    std::int32_t nr = 0, nth = 0, nz = 0;
    if (!in.integer(nr) || !in.integer(nth) || !in.integer(nz) || nr < 0 || nth < 0 || nz < 0 ||
        !decode_double_vector(in, field.values))
        return false;
    const std::uint64_t expected =
        static_cast<std::uint64_t>(nr) * static_cast<std::uint64_t>(nth) * static_cast<std::uint64_t>(nz);
    if (field.values.size() != expected)
        return false;
    field.nr  = nr;
    field.nth = nth;
    field.nz  = nz;
    return true;
}

// Runtime encoding follows below; only state that can influence a later
// completed timestep is persisted. Derived diagnostics and call-local scratch
// are deliberately reconstructed.

void encode_radiation_config(Encoder& out, const RadiationConfig& c, bool include_terrain_shortwave_controls = true)
{
    out.string(c.scheme_id);
    out.boolean(c.do_lw);
    out.boolean(c.do_sw);
    out.floating(c.dt_radiation);
    out.boolean(c.clear_sky_only);
    out.boolean(c.use_subcycling);
    out.boolean(c.enable_diagnostics);
    out.string(c.solar_forcing);
    out.floating(c.latitude_deg);
    out.floating(c.declination_deg);
    out.floating(c.local_solar_time_at_start_h);
    out.floating(c.tau_lw_ref);
    out.floating(c.tau_sw_ref);
    out.floating(c.n_lw);
    out.floating(c.n_sw);
    if (include_terrain_shortwave_controls)
    {
        out.unsigned_value<std::uint64_t>(c.terrain_sw_cosine_nodes);
        out.unsigned_value<std::uint64_t>(c.terrain_sw_azimuth_nodes);
        out.unsigned_value<std::uint64_t>(c.terrain_sw_maximum_bounces);
        out.floating(c.terrain_sw_relative_residual_tolerance);
        out.floating(c.terrain_sw_absolute_residual_tolerance_w);
    }
}

bool decode_radiation_config(Decoder& in, RadiationConfig& c, bool has_terrain_shortwave_controls)
{
    if (!(in.string(c.scheme_id) && in.boolean(c.do_lw) && in.boolean(c.do_sw) && in.floating(c.dt_radiation) &&
          in.boolean(c.clear_sky_only) && in.boolean(c.use_subcycling) && in.boolean(c.enable_diagnostics) &&
          in.string(c.solar_forcing) && in.floating(c.latitude_deg) && in.floating(c.declination_deg) &&
          in.floating(c.local_solar_time_at_start_h) && in.floating(c.tau_lw_ref) && in.floating(c.tau_sw_ref) &&
          in.floating(c.n_lw) && in.floating(c.n_sw)))
        return false;
    if (!has_terrain_shortwave_controls)
        return true;
    std::uint64_t cosine_nodes    = 0;
    std::uint64_t azimuth_nodes   = 0;
    std::uint64_t maximum_bounces = 0;
    if (!in.unsigned_value(cosine_nodes) || !in.unsigned_value(azimuth_nodes) || !in.unsigned_value(maximum_bounces) ||
        cosine_nodes > std::numeric_limits<std::size_t>::max() ||
        azimuth_nodes > std::numeric_limits<std::size_t>::max() ||
        maximum_bounces > std::numeric_limits<std::size_t>::max() ||
        !in.floating(c.terrain_sw_relative_residual_tolerance) ||
        !in.floating(c.terrain_sw_absolute_residual_tolerance_w))
        return false;
    c.terrain_sw_cosine_nodes    = static_cast<std::size_t>(cosine_nodes);
    c.terrain_sw_azimuth_nodes   = static_cast<std::size_t>(azimuth_nodes);
    c.terrain_sw_maximum_bounces = static_cast<std::size_t>(maximum_bounces);
    return true;
}

void encode_rrtmg_config(Encoder& out, const RadiationConfig& c)
{
    out.floating(c.rrtmg_ozone_vmr);
    out.floating(c.rrtmg_co2_vmr);
    out.floating(c.rrtmg_ch4_vmr);
    out.floating(c.rrtmg_n2o_vmr);
    out.floating(c.rrtmg_co_vmr);
    out.floating(c.rrtmg_o2_vmr);
    out.floating(c.rrtmg_n2_vmr);
    out.floating(c.rrtmg_cfc11_vmr);
    out.floating(c.rrtmg_cfc12_vmr);
    out.floating(c.rrtmg_cfc22_vmr);
    out.floating(c.rrtmg_ccl4_vmr);
    out.floating(c.rrtmg_liquid_effective_radius_um);
    out.floating(c.rrtmg_ice_effective_radius_um);
}

bool valid_rrtmg_config(const RadiationConfig& c)
{
    using namespace tmv::rrtmg_config_limits;
    return valid_mole_fraction(c.rrtmg_ozone_vmr, ozone_vmr_max) && valid_mole_fraction(c.rrtmg_co2_vmr, co2_vmr_max) &&
           valid_mole_fraction(c.rrtmg_ch4_vmr, ch4_vmr_max) && valid_mole_fraction(c.rrtmg_n2o_vmr, n2o_vmr_max) &&
           valid_mole_fraction(c.rrtmg_co_vmr, co_vmr_max) && valid_mole_fraction(c.rrtmg_o2_vmr, o2_vmr_max) &&
           valid_mole_fraction(c.rrtmg_n2_vmr, n2_vmr_max) &&
           valid_mole_fraction(c.rrtmg_cfc11_vmr, halocarbon_vmr_max) &&
           valid_mole_fraction(c.rrtmg_cfc12_vmr, halocarbon_vmr_max) &&
           valid_mole_fraction(c.rrtmg_cfc22_vmr, halocarbon_vmr_max) &&
           valid_mole_fraction(c.rrtmg_ccl4_vmr, halocarbon_vmr_max) &&
           valid_liquid_effective_radius(c.rrtmg_liquid_effective_radius_um) &&
           valid_ice_effective_radius(c.rrtmg_ice_effective_radius_um);
}

bool decode_rrtmg_config(Decoder& in, RadiationConfig& c)
{
    return in.floating(c.rrtmg_ozone_vmr) && in.floating(c.rrtmg_co2_vmr) && in.floating(c.rrtmg_ch4_vmr) &&
           in.floating(c.rrtmg_n2o_vmr) && in.floating(c.rrtmg_co_vmr) && in.floating(c.rrtmg_o2_vmr) &&
           in.floating(c.rrtmg_n2_vmr) && in.floating(c.rrtmg_cfc11_vmr) && in.floating(c.rrtmg_cfc12_vmr) &&
           in.floating(c.rrtmg_cfc22_vmr) && in.floating(c.rrtmg_ccl4_vmr) &&
           in.floating(c.rrtmg_liquid_effective_radius_um) && in.floating(c.rrtmg_ice_effective_radius_um) &&
           valid_rrtmg_config(c);
}

void encode_simple_grey_config(Encoder& out, const RadiationConfig& c)
{
    out.unsigned_value<std::uint64_t>(c.subcycle_count);
    out.string(c.simple_grey_optical_model);
    out.floating(c.simple_grey_lw_vapor_absorption_m2kg);
    out.floating(c.simple_grey_lw_liquid_absorption_m2kg);
    out.floating(c.simple_grey_lw_ice_absorption_m2kg);
    out.floating(c.simple_grey_sw_vapor_absorption_m2kg);
    out.floating(c.simple_grey_sw_liquid_absorption_m2kg);
    out.floating(c.simple_grey_sw_ice_absorption_m2kg);
}

bool valid_simple_grey_checkpoint_config(const RadiationConfig& c)
{
    const bool                  clear_model     = c.simple_grey_optical_model == "pressure_clear_sky_v1";
    const bool                  absorbing_model = c.simple_grey_optical_model == "absorbing_water_cloud_path_v1";
    const std::array<double, 6> coefficients{c.simple_grey_lw_vapor_absorption_m2kg,
                                             c.simple_grey_lw_liquid_absorption_m2kg,
                                             c.simple_grey_lw_ice_absorption_m2kg,
                                             c.simple_grey_sw_vapor_absorption_m2kg,
                                             c.simple_grey_sw_liquid_absorption_m2kg,
                                             c.simple_grey_sw_ice_absorption_m2kg};
    return (clear_model || absorbing_model) && (c.scheme_id != "simple_grey" || c.clear_sky_only == clear_model) &&
           ((!c.use_subcycling && c.subcycle_count == 1) ||
            (c.use_subcycling && c.subcycle_count >= 2 && c.subcycle_count <= 64)) &&
           std::all_of(coefficients.begin(),
                       coefficients.end(),
                       [](double value) { return std::isfinite(value) && value >= 0.0; });
}

bool valid_radiation_flux_diagnostics(const RadiationFluxDiagnostics& diagnostics)
{
    const bool no_shape  = diagnostics.columns_x == 0 && diagnostics.columns_y == 0 && diagnostics.levels == 0;
    const bool no_values = diagnostics.upward_longwave_wm2.empty() && diagnostics.downward_longwave_wm2.empty() &&
                           diagnostics.upward_shortwave_wm2.empty() && diagnostics.downward_shortwave_wm2.empty();
    if (no_shape || no_values)
        return no_shape && no_values;
    if (diagnostics.columns_x > wire::max_elements || diagnostics.columns_y > wire::max_elements ||
        diagnostics.levels >= wire::max_elements)
        return false;
    const std::uint64_t x          = diagnostics.columns_x;
    const std::uint64_t y          = diagnostics.columns_y;
    const std::uint64_t interfaces = diagnostics.levels + 1;
    if (x == 0 || y == 0 || interfaces == 0 || x > wire::max_elements / y || x * y > wire::max_elements / interfaces)
        return false;
    const std::size_t expected     = static_cast<std::size_t>(x * y * interfaces);
    const auto        valid_values = [expected](const std::vector<double>& values)
    {
        return values.size() == expected &&
               std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); });
    };
    return valid_values(diagnostics.upward_longwave_wm2) && valid_values(diagnostics.downward_longwave_wm2) &&
           valid_values(diagnostics.upward_shortwave_wm2) && valid_values(diagnostics.downward_shortwave_wm2);
}

bool valid_terrain_shortwave_runtime_contract(const RadiationRuntimeState& radiation)
{
    const auto&       config                = radiation.config;
    const std::size_t minimum_cosine_nodes  = config.scheme_id == "rrtmg" ? 1 : 4;
    const std::size_t minimum_azimuth_nodes = config.scheme_id == "rrtmg" ? 4 : 8;
    if (config.terrain_sw_cosine_nodes < minimum_cosine_nodes || config.terrain_sw_cosine_nodes > 32 ||
        config.terrain_sw_azimuth_nodes < minimum_azimuth_nodes || config.terrain_sw_azimuth_nodes > 128 ||
        config.terrain_sw_azimuth_nodes % 4 != 0 || config.terrain_sw_maximum_bounces == 0 ||
        config.terrain_sw_maximum_bounces > 256 || !std::isfinite(config.terrain_sw_relative_residual_tolerance) ||
        config.terrain_sw_relative_residual_tolerance < 0.0 || config.terrain_sw_relative_residual_tolerance > 1.0e-6 ||
        !std::isfinite(config.terrain_sw_absolute_residual_tolerance_w) ||
        config.terrain_sw_absolute_residual_tolerance_w < 0.0 || !radiation.held_reflection_converged)
        return false;
    const std::array<double, 6> ledger{radiation.held_shortwave_incoming_power_w,
                                       radiation.held_shortwave_atmospheric_absorbed_power_w,
                                       radiation.held_surface_absorbed_power_w,
                                       radiation.held_top_escape_power_w,
                                       radiation.held_lateral_escape_power_w,
                                       radiation.held_unresolved_reflected_power_w};
    for (double value : ledger)
        if (!std::isfinite(value) || value < 0.0)
            return false;
    const long double accounted = static_cast<long double>(ledger[1]) + ledger[2] + ledger[3] + ledger[4] + ledger[5];
    const long double scale     = std::max({1.0L, std::abs(static_cast<long double>(ledger[0])), std::abs(accounted)});
    return std::abs(static_cast<long double>(ledger[0]) - accounted) <= 1.0e-8L * scale;
}

void encode_turbulence_config(Encoder& out, const TurbulenceConfig& c)
{
    out.string(c.scheme_id);
    out.floating(c.dt_sgs);
    out.string(c.mode);
    out.string(c.filter_width);
    out.floating(c.Cs);
    out.floating(c.Pr_t);
    out.floating(c.Sc_t);
    out.boolean(c.dynamic_Cs);
    out.string(c.stability_correction);
    out.boolean(c.enable_moist_buoyancy);
    out.boolean(c.enable_diagnostics);
    out.floating(c.nu_t_max);
    out.floating(c.K_max);
}

bool decode_turbulence_config(Decoder& in, TurbulenceConfig& c)
{
    return in.string(c.scheme_id) && in.floating(c.dt_sgs) && in.string(c.mode) && in.string(c.filter_width) &&
           in.floating(c.Cs) && in.floating(c.Pr_t) && in.floating(c.Sc_t) && in.boolean(c.dynamic_Cs) &&
           in.string(c.stability_correction) && in.boolean(c.enable_moist_buoyancy) &&
           in.boolean(c.enable_diagnostics) && in.floating(c.nu_t_max) && in.floating(c.K_max);
}

void encode_chaos_config(Encoder& out, const chaos::ChaosConfig& c)
{
    out.string(c.scheme_id);
    out.unsigned_value(c.seed);
    out.integer(c.member_id);
    out.floating(c.dt_chaos);
    out.floating(c.tau_t);
    out.floating(c.Lx);
    out.floating(c.Ly);
    out.string(c.filter_id);
    out.floating(c.xi_max);
    out.string(c.taper_id);
    out.floating(c.taper_z1);
    out.floating(c.taper_z2);
    encode_string_vector(out, c.apply_to_ic);
    encode_string_vector(out, c.apply_to_tendencies);
    encode_string_double_map(out, c.sigma_ic);
    encode_string_double_map(out, c.alpha_tend);
}

bool decode_chaos_config(Decoder& in, chaos::ChaosConfig& c)
{
    return in.string(c.scheme_id) && in.unsigned_value(c.seed) && in.integer(c.member_id) && in.floating(c.dt_chaos) &&
           in.floating(c.tau_t) && in.floating(c.Lx) && in.floating(c.Ly) && in.string(c.filter_id) &&
           in.floating(c.xi_max) && in.string(c.taper_id) && in.floating(c.taper_z1) && in.floating(c.taper_z2) &&
           decode_string_vector(in, c.apply_to_ic) && decode_string_vector(in, c.apply_to_tendencies) &&
           decode_string_double_map(in, c.sigma_ic) && decode_string_double_map(in, c.alpha_tend);
}

void encode_terrain_config(Encoder& out, const TerrainConfig& c)
{
    out.string(c.scheme_id);
    out.string(c.coord_id);
    out.floating(c.ztop);
    out.boolean(c.compute_derivatives);
    out.boolean(c.compute_metrics);
    out.floating(c.bell.h0);
    out.floating(c.bell.a);
    out.boolean(c.bell.axisymmetric);
    out.floating(c.schar.H);
    out.floating(c.schar.a);
    out.floating(c.schar.ell);
    out.floating(c.smoothing_decay);
}

bool decode_terrain_config(Decoder& in, TerrainConfig& c)
{
    return in.string(c.scheme_id) && in.string(c.coord_id) && in.floating(c.ztop) &&
           in.boolean(c.compute_derivatives) && in.boolean(c.compute_metrics) && in.floating(c.bell.h0) &&
           in.floating(c.bell.a) && in.boolean(c.bell.axisymmetric) && in.floating(c.schar.H) &&
           in.floating(c.schar.a) && in.floating(c.schar.ell) && in.floating(c.smoothing_decay);
}

void encode_terrain_placement_config(Encoder& out, const TerrainConfig& c)
{
    out.string(c.horizontal_placement_id);
    out.floating(c.feature_center_earth_x_m);
    out.floating(c.feature_center_earth_y_m);
}

void encode_terrain_placement(Encoder& out, const EarthFixedTerrainPlacement& placement)
{
    out.boolean(placement.active);
    out.integer(placement.nr);
    out.integer(placement.nth);
    out.integer(placement.nz);
    out.floating(placement.dx_m);
    out.floating(placement.dy_m);
    out.floating(placement.grid_origin_earth_m.x_m);
    out.floating(placement.grid_origin_earth_m.y_m);
    out.floating(placement.feature_center_earth_m.x_m);
    out.floating(placement.feature_center_earth_m.y_m);
    out.unsigned_value(placement.generation);
}

bool decode_terrain_placement(Decoder& in, TerrainConfig& config, EarthFixedTerrainPlacement& placement)
{
    std::int32_t nr = 0, nth = 0, nz = 0;
    if (!in.string(config.horizontal_placement_id) || !in.floating(config.feature_center_earth_x_m) ||
        !in.floating(config.feature_center_earth_y_m))
        return false;
    if (!in.boolean(placement.active) || !in.integer(nr) || !in.integer(nth) || !in.integer(nz) ||
        !in.floating(placement.dx_m) || !in.floating(placement.dy_m) ||
        !in.floating(placement.grid_origin_earth_m.x_m) || !in.floating(placement.grid_origin_earth_m.y_m) ||
        !in.floating(placement.feature_center_earth_m.x_m) || !in.floating(placement.feature_center_earth_m.y_m) ||
        !in.unsigned_value(placement.generation))
        return false;
    placement.nr  = nr;
    placement.nth = nth;
    placement.nz  = nz;
    return true;
}

bool valid_terrain_placement_checkpoint(const SimulationStorageState& state) noexcept
{
    const auto& config    = state.runtime.terrain.config;
    const auto& placement = state.runtime.terrain.earth_fixed_placement;
    if (placement.sampling != EarthFixedTerrainSampling::Point
        && placement.sampling != EarthFixedTerrainSampling::CellAverageBilinear)
        return false;
    if (!placement.active && placement.sampling != EarthFixedTerrainSampling::Point)
        return false;
    if (!std::isfinite(config.feature_center_earth_x_m) || !std::isfinite(config.feature_center_earth_y_m))
        return false;
    if (config.horizontal_placement_id == "grid_attached")
        return config.feature_center_earth_x_m == 0.0 && config.feature_center_earth_y_m == 0.0 && !placement.active;
    if (config.horizontal_placement_id != "earth_fixed" ||
        (config.scheme_id != "bell" && config.scheme_id != "schar") ||
        (config.coord_id != "btf" && config.coord_id != "smoothed") || !config.compute_derivatives ||
        !config.compute_metrics || !placement.active || placement.nr != state.grid.nr ||
        placement.nth != state.grid.nth || placement.nz != state.grid.nz || placement.dx_m != state.grid.dx ||
        placement.dy_m != state.grid.dy || !std::isfinite(placement.grid_origin_earth_m.x_m) ||
        !std::isfinite(placement.grid_origin_earth_m.y_m) ||
        placement.feature_center_earth_m.x_m != config.feature_center_earth_x_m ||
        placement.feature_center_earth_m.y_m != config.feature_center_earth_y_m || placement.generation == 0)
        return false;
    try
    {
        const auto expected_origin = state.configuration.domain_frame.earth_origin_at(state.time_s);
        return placement.grid_origin_earth_m.x_m == expected_origin.x_m &&
               placement.grid_origin_earth_m.y_m == expected_origin.y_m;
    }
    catch (...)
    {
        return false;
    }
}

void encode_radar_config(Encoder& out, const RadarConfig& c)
{
    out.string(c.scheme_id);
    out.string(c.operator_tier);
    out.floating(c.radar_x);
    out.floating(c.radar_y);
    out.floating(c.radar_z);
    out.string(c.sampling);
    out.boolean(c.apply_scatterer_correction);
    const bool flags[] = {
        c.has_qr, c.has_qs, c.has_qg, c.has_qh, c.has_qi, c.has_Nr, c.has_Ns, c.has_Ng, c.has_Nh, c.has_Ni};
    for (bool flag : flags)
        out.boolean(flag);
}

bool decode_radar_config(Decoder& in, RadarConfig& c)
{
    if (!in.string(c.scheme_id) || !in.string(c.operator_tier) || !in.floating(c.radar_x) || !in.floating(c.radar_y) ||
        !in.floating(c.radar_z) || !in.string(c.sampling) || !in.boolean(c.apply_scatterer_correction))
        return false;
    bool* flags[] = {
        &c.has_qr, &c.has_qs, &c.has_qg, &c.has_qh, &c.has_qi, &c.has_Nr, &c.has_Ns, &c.has_Ng, &c.has_Nh, &c.has_Ni};
    for (bool* flag : flags)
        if (!in.boolean(*flag))
            return false;
    return true;
}

void encode_radar_product_selection(Encoder& out, const RadarProductSelection& selection)
{
    out.boolean(selection.reflectivity);
    out.boolean(selection.radial_velocity);
    out.boolean(selection.zdr);
    out.boolean(selection.kdp);
    out.boolean(selection.rhohv);
    out.string(selection.reflectivity_tier);
    out.string(selection.velocity_sampling);
    out.string(selection.zdr_tier);
    out.floating(selection.radar_x);
    out.floating(selection.radar_y);
    out.floating(selection.radar_z);
    out.boolean(selection.apply_scatterer_correction);
}

bool decode_radar_product_selection(Decoder& in, RadarProductSelection& selection, bool has_complete_polarimetry)
{
    if (!(in.boolean(selection.reflectivity) && in.boolean(selection.radial_velocity) && in.boolean(selection.zdr)))
        return false;
    if (has_complete_polarimetry && !(in.boolean(selection.kdp) && in.boolean(selection.rhohv)))
        return false;
    return in.string(selection.reflectivity_tier) && in.string(selection.velocity_sampling) &&
           in.string(selection.zdr_tier) && in.floating(selection.radar_x) && in.floating(selection.radar_y) &&
           in.floating(selection.radar_z) && in.boolean(selection.apply_scatterer_correction);
}

bool valid_radar_product_selection(const RadarProductSelection& selection) noexcept
{
    return (selection.reflectivity_tier == "fast_da" || selection.reflectivity_tier == "psd_moment") &&
           (selection.velocity_sampling == "point" || selection.velocity_sampling == "beam_volume") &&
           (selection.zdr_tier == "simple" || selection.zdr_tier == "polarimetric_fo" ||
            selection.zdr_tier == "zhang2021_rain_mp") &&
           std::isfinite(selection.radar_x) && std::isfinite(selection.radar_y) && std::isfinite(selection.radar_z);
}

void encode_advection_config(Encoder& out, const AdvectionConfig& c)
{
    out.string(c.scheme_id);
    out.string(c.formulation);
    out.string(c.split_mode);
    out.string(c.limiter_id);
    out.boolean(c.positivity);
    out.floating(c.positivity_dt);
    out.floating(c.cfl_target);
    out.string(c.reconstruct_on);
    out.integer(c.weno_order);
}

bool decode_advection_config(Decoder& in, AdvectionConfig& c)
{
    return in.string(c.scheme_id) && in.string(c.formulation) && in.string(c.split_mode) && in.string(c.limiter_id) &&
           in.boolean(c.positivity) && in.floating(c.positivity_dt) && in.floating(c.cfl_target) &&
           in.string(c.reconstruct_on) && in.integer(c.weno_order);
}

void encode_diffusion_config(Encoder& out, const DiffusionConfig& c)
{
    out.string(c.scheme_id);
    out.integer(c.operator_order);
    out.string(c.operator_type);
    out.string(c.apply_to);
    out.floating(c.K_h);
    out.floating(c.K_v);
    out.string(c.implicit_dim);
    out.floating(c.dt_diffusion);
    out.boolean(c.use_variable_K);
}

void encode_diffusion_profile_config(Encoder& out, const DiffusionConfig& c)
{
    out.string(c.coefficient_source);
    out.floating(c.K_v_surface);
    out.floating(c.K_v_top);
}

bool decode_diffusion_config(Decoder& in, DiffusionConfig& c)
{
    return in.string(c.scheme_id) && in.integer(c.operator_order) && in.string(c.operator_type) &&
           in.string(c.apply_to) && in.floating(c.K_h) && in.floating(c.K_v) && in.string(c.implicit_dim) &&
           in.floating(c.dt_diffusion) && in.boolean(c.use_variable_K);
}

void encode_time_config(Encoder& out, const TimeSteppingConfig& c)
{
    out.string(c.scheme_id);
    out.boolean(c.split_acoustic);
    out.integer(c.n_acoustic_substeps);
    out.floating(c.max_acoustic_speed_ms);
    out.floating(c.min_acoustic_spacing_m);
    out.string(c.physics_splitting);
    out.floating(c.cfl_safety);
    out.boolean(c.adaptive_dt);
    out.floating(c.dt_min);
    out.floating(c.dt_max);
}

bool decode_time_config(Decoder& in, TimeSteppingConfig& c, std::uint32_t section_version)
{
    if (!in.string(c.scheme_id) || !in.boolean(c.split_acoustic) || !in.integer(c.n_acoustic_substeps) ||
        !in.floating(c.max_acoustic_speed_ms))
        return false;
    if (section_version >= 12 && !in.floating(c.min_acoustic_spacing_m))
        return false;
    return in.string(c.physics_splitting) && in.floating(c.cfl_safety) && in.boolean(c.adaptive_dt) &&
           in.floating(c.dt_min) && in.floating(c.dt_max);
}

void encode_runtime(Encoder& out, const SimulationStorageState& storage, std::string& error)
{
    const auto& runtime = storage.runtime;
    if (!valid_terrain_placement_checkpoint(storage))
    {
        error = "cannot checkpoint invalid Earth-fixed terrain identity";
        return;
    }
    out.string(runtime.compute.config.backend);
    out.integer(runtime.compute.config.device_index);
    out.boolean(runtime.compute.config.allow_fallback);
    out.boolean(runtime.compute.config.validate_parity);
    encode_enum(out, runtime.compute.requested_kind);
    {
        std::lock_guard<std::mutex> lock(runtime.parity.mutex);
        out.unsigned_value<std::uint64_t>(runtime.parity.stats.size());
        for (const auto& entry : runtime.parity.stats)
        {
            out.string(std::get<0>(entry.first));
            out.string(std::get<1>(entry.first));
            encode_enum(out, std::get<2>(entry.first));
            const auto& s = entry.second;
            encode_enum(out, s.tier);
            encode_enum(out, s.domain);
            out.unsigned_value(s.samples);
            out.unsigned_value(s.failed_samples);
            out.unsigned_value(s.empty_samples);
            out.unsigned_value(s.cells_compared);
            out.unsigned_value(s.breach_cells);
            out.unsigned_value(s.nonfinite_cells);
            out.floating(s.worst_abs);
            out.floating(s.worst_rel);
            out.integer(s.worst_i);
            out.integer(s.worst_j);
            out.integer(s.worst_k);
            out.floating(s.worst_cpu);
            out.floating(s.worst_gpu);
        }
        out.unsigned_value<std::uint64_t>(runtime.parity.call_counters.size());
        for (const auto& entry : runtime.parity.call_counters)
        {
            out.string(entry.first);
            out.unsigned_value(entry.second);
        }
        out.unsigned_value(runtime.parity.total_samples);
        out.boolean(runtime.parity.any_breach);
        out.integer(runtime.parity.env_enabled);
        out.integer(runtime.parity.env_every);
    }

    encode_radiation_config(out, runtime.radiation.config, false);
    out.floating(runtime.radiation.last_update_time_s);
    out.floating(runtime.radiation.next_update_time_s);
    out.floating(runtime.radiation.held_atmospheric_power_w);
    encode_field(out, runtime.radiation.held_cell_power_w);
    out.floating(runtime.radiation.last_expected_atmospheric_energy_j);
    out.floating(runtime.radiation.last_applied_radiation_heat_j);
    out.floating(runtime.radiation.accumulated_expected_atmospheric_energy_j);
    out.floating(runtime.radiation.accumulated_applied_radiation_heat_j);
    out.unsigned_value(runtime.radiation.applied_step_count);
    out.boolean(runtime.radiation.scheme != nullptr);

    encode_turbulence_config(out, runtime.turbulence.config);
    out.floating(runtime.turbulence.last_update_time_s);
    out.floating(runtime.turbulence.next_update_time_s);
    encode_field(out, runtime.turbulence.held_tendencies.dudt_sgs);
    encode_field(out, runtime.turbulence.held_tendencies.dvdt_sgs);
    encode_field(out, runtime.turbulence.held_tendencies.dwdt_sgs);
    encode_field(out, runtime.turbulence.held_tendencies.dthetadt_sgs);
    encode_field(out, runtime.turbulence.held_tendencies.dqvdt_sgs);
    encode_field(out, runtime.turbulence.held_tendencies.dtkedt_sgs);
    out.floating(runtime.turbulence.last_momentum_work.requested_linear_work_j);
    out.floating(runtime.turbulence.last_momentum_work.finite_step_correction_j);
    out.floating(runtime.turbulence.last_momentum_work.storage_rounding_j);
    out.floating(runtime.turbulence.last_momentum_work.committed_work_j);
    out.floating(runtime.turbulence.cumulative_requested_momentum_work_j);
    out.floating(runtime.turbulence.cumulative_finite_step_momentum_correction_j);
    out.floating(runtime.turbulence.cumulative_momentum_storage_rounding_j);
    out.floating(runtime.turbulence.cumulative_committed_momentum_work_j);
    out.unsigned_value(runtime.turbulence.momentum_work_step_count);
    out.floating(runtime.turbulence.last_explicit_stability_limit_s);
    out.unsigned_value(runtime.turbulence.last_explicit_substep_count);
    out.boolean(runtime.turbulence.scheme != nullptr);

    std::vector<std::uint8_t> boundary_bytes;
    if (!serialize_boundary_layer_runtime(runtime.boundary_layer, boundary_bytes, error))
        return;
    out.boolean(runtime.boundary_layer.scheme != nullptr);
    out.unsigned_value<std::uint64_t>(boundary_bytes.size());
    out.raw(boundary_bytes);

    encode_chaos_config(out, runtime.chaos.config);
    const auto& chaos_diag = runtime.chaos.diagnostics;
    out.floating(chaos_diag.mean_perturbation);
    out.floating(chaos_diag.variance_perturbation);
    out.floating(chaos_diag.correlation_length_x);
    out.floating(chaos_diag.correlation_length_y);
    out.boolean(chaos_diag.negative_moisture_detected);
    out.integer(chaos_diag.negative_moisture_count);
    out.floating(chaos_diag.min_qv_after_perturbation);
    encode_string_vector(out, chaos_diag.warnings);
    encode_string_vector(out, chaos_diag.errors);
    // Wall-clock profiling is observer state, not scientific continuation
    // state. Preserve the v37 byte layout but canonicalize these slots so a
    // restart cannot make an otherwise bitwise-identical checkpoint depend
    // on host scheduling or branch duration.
    out.floating(0.0);
    out.floating(0.0);
    out.floating(0.0);
    out.floating(runtime.chaos.noise_elapsed_s);
    out.boolean(runtime.chaos.pbl_base_tendencies_valid);
    encode_field(out, runtime.chaos.base_du_dt_pbl);
    encode_field(out, runtime.chaos.base_dv_dt_pbl);
    encode_field(out, runtime.chaos.base_dtheta_dt_pbl);
    encode_field(out, runtime.chaos.base_dqv_dt_pbl);
    out.boolean(runtime.chaos.scheme != nullptr);
    encode_portable_state(out, runtime.chaos.scheme ? runtime.chaos.scheme->portable_state() : PortableSchemeState{});

    out.boolean(runtime.rayleigh.upper_initialized);
    out.boolean(runtime.rayleigh.lateral_initialized);
    encode_double_vector(out, runtime.rayleigh.alpha);
    encode_double_vector(out, runtime.rayleigh.u_ref);
    encode_double_vector(out, runtime.rayleigh.v_ref);
    encode_double_vector(out, runtime.rayleigh.w_only_alpha);
    encode_double_vector(out, runtime.rayleigh.lateral_alpha);

    const std::string micro_name =
        runtime.microphysics.scheme ? runtime.microphysics.scheme->get_scheme_name() : std::string{};
    out.string(micro_name);
    encode_portable_state(
        out, runtime.microphysics.scheme ? runtime.microphysics.scheme->portable_state() : PortableSchemeState{});
    for (const auto& repair : runtime.microphysics.last_water_repair_report.species)
    {
        out.floating(repair.lower_bound_correction_kg);
        out.floating(repair.upper_bound_correction_kg);
        out.floating(repair.nonfinite_correction_kg);
        out.unsigned_value<std::uint64_t>(repair.lower_bound_count);
        out.unsigned_value<std::uint64_t>(repair.upper_bound_count);
        out.unsigned_value<std::uint64_t>(repair.nonfinite_count);
        out.floating(repair.lower_bound_roundoff_correction_kg);
        out.unsigned_value<std::uint64_t>(repair.lower_bound_roundoff_count);
        out.floating(repair.storage_rounding_kg);
    }
    out.unsigned_value<std::uint64_t>(runtime.microphysics.last_water_repair_report.nonfinite_tendency_count);
    const auto&  thermo          = runtime.microphysics.last_thermodynamic_repair_report;
    const double thermo_values[] = {thermo.microphysics_heat_j,
                                    thermo.radiation_heat_j,
                                    thermo.boundary_layer_heat_j,
                                    thermo.theta_tendency_limit_correction_j,
                                    thermo.theta_nonfinite_correction_j,
                                    thermo.theta_storage_rounding_j,
                                    thermo.theta_storage_residual_j,
                                    thermo.theta_storage_residual_release_j,
                                    thermo.theta_bound_correction_j,
                                    thermo.microphysics_water_tendency_kg,
                                    thermo.boundary_layer_vapor_tendency_kg};
    for (double value : thermo_values)
        out.floating(value);
    out.unsigned_value<std::uint64_t>(thermo.theta_tendency_limit_count);
    out.unsigned_value<std::uint64_t>(thermo.theta_nonfinite_count);
    out.unsigned_value<std::uint64_t>(thermo.theta_lower_bound_count);
    out.unsigned_value<std::uint64_t>(thermo.theta_upper_bound_count);
    encode_field(out, runtime.microphysics.theta_storage_residual);
    const auto& micro_conservation = runtime.microphysics.last_conservation_report;
    out.boolean(micro_conservation.active);
    out.floating(micro_conservation.tendency_imbalance_kg);
    out.floating(micro_conservation.baseline_q_kg);
    out.floating(micro_conservation.surface_precipitation_kg);
    out.floating(micro_conservation.relative_imbalance);
    for (double species_tendency : thermo.microphysics_water_tendency_kg_by_species)
    {
        out.floating(species_tendency);
    }

    if (runtime.terrain.config.scheme_id != "none")
    {
        try
        {
            terrain_transport::validate_geometry(runtime.terrain.metrics);
        }
        catch (const std::exception& exception)
        {
            error = std::string("cannot checkpoint invalid active terrain geometry: ") + exception.what();
            return;
        }
    }
    encode_terrain_config(out, runtime.terrain.config);
    out.boolean(runtime.terrain.scheme != nullptr);
    encode_matrix(out, runtime.terrain.topography.h);
    encode_matrix(out, runtime.terrain.topography.hx);
    encode_matrix(out, runtime.terrain.topography.hy);
    encode_terrain_field(out, runtime.terrain.metrics.z);
    encode_terrain_field(out, runtime.terrain.metrics.J);
    encode_terrain_field(out, runtime.terrain.metrics.mx);
    encode_terrain_field(out, runtime.terrain.metrics.my);
    encode_terrain_field(out, runtime.terrain.metrics.zeta);
    encode_terrain_field(out, runtime.terrain.metrics.z_face);
    encode_terrain_field(out, runtime.terrain.metrics.Jx_face);
    encode_terrain_field(out, runtime.terrain.metrics.Jy_face);
    encode_terrain_field(out, runtime.terrain.metrics.Gx_face);
    encode_terrain_field(out, runtime.terrain.metrics.Gy_face);
    out.floating(runtime.terrain.metrics.dx);
    out.floating(runtime.terrain.metrics.dy);
    out.floating(runtime.terrain.metrics.dzeta);

    encode_radar_config(out, runtime.radar.selected_config);
    out.boolean(runtime.radar.selected_scheme != nullptr);
    out.integer(runtime.radar.nr);
    out.integer(runtime.radar.nth);
    out.integer(runtime.radar.nz);
    out.floating(runtime.radar.radar_x);
    out.floating(runtime.radar.radar_y);
    out.floating(runtime.radar.radar_z);

    encode_advection_config(out, runtime.advection.config);
    out.boolean(runtime.advection.scheme != nullptr);
    out.boolean(runtime.advection.tracer_advected_in_batch);
    for (bool active : runtime.advection.hydrometeor_active)
        out.boolean(active);
    out.integer(runtime.advection.nr);
    out.integer(runtime.advection.nth);
    out.integer(runtime.advection.nz);
    out.boolean(runtime.advection.activity_current);
    out.unsigned_value(runtime.advection.metadata_consumptions);
    out.unsigned_value(runtime.advection.fallback_scans);
    for (const auto& report : runtime.advection.thermodynamic_boundary_reports)
    {
        out.floating(report.outward_boundary_inventory);
        out.floating(report.flux_implied_state_change);
        out.floating(report.committed_state_change);
        out.floating(report.closure_residual);
        out.boolean(report.complete);
    }
    for (const auto& report : runtime.advection.cumulative_thermodynamic_boundary_reports)
    {
        out.floating(report.outward_boundary_inventory);
        out.floating(report.flux_implied_state_change);
        out.floating(report.committed_state_change);
        out.floating(report.closure_residual);
        out.boolean(report.complete);
    }
    out.unsigned_value(runtime.advection.thermodynamic_boundary_report_step_count);
    const auto encode_scalar_energy_reports = [&out](const ThermodynamicMoistEnergyTransportReports& reports)
    {
        for (const auto& report : reports)
        {
            out.floating(report.dry_internal_change_j);
            out.floating(report.water_sensible_change_j);
            out.floating(report.phase_reference_change_j);
            out.floating(report.carrier_kinetic_change_j);
            out.floating(report.gravitational_change_j);
            out.floating(report.subgrid_tke_change_j);
            out.floating(report.density_history_remap_j);
            out.floating(report.outward_boundary_energy_j);
            out.floating(report.interior_conversion_j);
            out.floating(report.limited_face_state_change_j);
            out.floating(report.requested_state_change_j);
            out.floating(report.storage_rounding_j);
            out.floating(report.committed_state_change_j);
            out.floating(report.closure_residual_j);
            out.boolean(report.complete);
        }
    };
    encode_scalar_energy_reports(runtime.advection.thermodynamic_moist_energy_reports);
    encode_scalar_energy_reports(runtime.advection.cumulative_thermodynamic_moist_energy_reports);
    out.unsigned_value(runtime.advection.thermodynamic_moist_energy_report_step_count);

    encode_diffusion_config(out, runtime.diffusion.config);
    out.boolean(runtime.diffusion.scheme != nullptr);
    out.floating(runtime.diffusion.diagnostics.max_diffusion_number);
    out.floating(runtime.diffusion.last_momentum_work.requested_linear_work_j);
    out.floating(runtime.diffusion.last_momentum_work.finite_step_correction_j);
    out.floating(runtime.diffusion.last_momentum_work.storage_rounding_j);
    out.floating(runtime.diffusion.last_momentum_work.committed_work_j);
    out.floating(runtime.diffusion.cumulative_requested_momentum_work_j);
    out.floating(runtime.diffusion.cumulative_finite_step_momentum_correction_j);
    out.floating(runtime.diffusion.cumulative_momentum_storage_rounding_j);
    out.floating(runtime.diffusion.cumulative_committed_momentum_work_j);
    out.unsigned_value(runtime.diffusion.momentum_work_step_count);

    encode_time_config(out, runtime.time_stepping.config);
    out.boolean(runtime.time_stepping.scheme != nullptr);
    out.integer(runtime.time_stepping.diagnostics.n_steps);
    out.floating(runtime.time_stepping.diagnostics.max_cfl);
    out.floating(runtime.time_stepping.diagnostics.cpu_time_per_step);
    encode_double_vector(out, runtime.time_stepping.diagnostics.dt_history);
    out.boolean(runtime.time_stepping.diagnostics.converged);
    const auto& boundary_mass = runtime.time_stepping.last_boundary_mass;
    out.floating(boundary_mass.outward_boundary_mass_kg);
    out.floating(boundary_mass.flux_implied_state_change_kg);
    out.floating(boundary_mass.committed_state_change_kg);
    out.floating(boundary_mass.closure_residual_kg);
    out.floating(runtime.time_stepping.cumulative_outward_boundary_mass_kg);
    out.floating(runtime.time_stepping.cumulative_flux_implied_state_change_kg);
    out.floating(runtime.time_stepping.cumulative_committed_state_change_kg);
    out.floating(runtime.time_stepping.cumulative_boundary_mass_closure_residual_kg);
    out.unsigned_value(runtime.time_stepping.boundary_mass_step_count);
    const auto encode_acoustic_energy = [&out](const CartesianCGridAcousticEnergyReportV1& report)
    {
        out.floating(report.continuity_eos_dry_internal_change_j);
        out.floating(report.continuity_eos_water_sensible_change_j);
        out.floating(report.continuity_eos_phase_reference_change_j);
        out.floating(report.continuity_carrier_kinetic_change_j);
        out.floating(report.continuity_gravitational_change_j);
        out.floating(report.continuity_subgrid_tke_change_j);
        out.floating(report.pressure_gradient_requested_linear_work_j);
        out.floating(report.pressure_gradient_finite_step_correction_j);
        out.floating(report.pressure_gradient_storage_rounding_j);
        out.floating(report.pressure_gradient_committed_work_j);
        out.floating(report.boundary_committed_work_j);
        out.floating(report.boundary_radiative_requested_linear_work_j);
        out.floating(report.boundary_radiative_finite_step_correction_j);
        out.floating(report.boundary_radiative_storage_rounding_j);
        out.floating(report.boundary_radiative_committed_work_j);
        out.floating(report.boundary_constraint_history_committed_work_j);
        out.unsigned_value(report.boundary_application_count);
        out.boolean(report.boundary_complete);
        out.floating(report.committed_inventory_change_j);
        out.unsigned_value(report.substep_count);
        out.boolean(report.complete);
        out.floating(report.continuity_carrier_requested_change_j);
        out.floating(report.continuity_carrier_outward_boundary_transport_j);
        out.floating(report.continuity_carrier_interior_conversion_j);
        out.floating(report.continuity_carrier_storage_rounding_j);
        out.floating(report.slow_momentum_advection_requested_linear_work_j);
        out.floating(report.kinetic_transport_interior_conversion_j);
        out.floating(report.kinetic_transport_closure_residual_j);
        out.boolean(report.kinetic_transport_complete);
    };
    encode_acoustic_energy(runtime.time_stepping.last_acoustic_energy);
    encode_acoustic_energy(runtime.time_stepping.cumulative_acoustic_energy);
    out.unsigned_value(runtime.time_stepping.acoustic_energy_step_count);
    const auto encode_slow_momentum_energy = [&out](const CartesianSlowMomentumEnergyReportV1& report)
    {
        out.floating(report.advection_requested_linear_work_j);
        out.floating(report.nonadvection_requested_linear_work_j);
        out.floating(report.advection_finite_step_correction_j);
        out.floating(report.nonadvection_finite_step_correction_j);
        out.floating(report.coupled_finite_step_cross_work_j);
        out.floating(report.storage_rounding_j);
        out.floating(report.committed_kinetic_change_j);
        out.floating(report.closure_residual_j);
        out.floating(report.pressure_dry_internal_change_j);
        out.floating(report.pressure_water_sensible_change_j);
        out.floating(report.committed_pressure_energy_change_j);
        out.boolean(report.complete);
    };
    encode_slow_momentum_energy(runtime.time_stepping.last_slow_momentum_energy);
    encode_slow_momentum_energy(runtime.time_stepping.cumulative_slow_momentum_energy);
    out.unsigned_value(runtime.time_stepping.slow_momentum_energy_step_count);
    const auto& moist_commit = runtime.microphysics.last_thermodynamic_repair_report;
    out.floating(moist_commit.moist_dry_internal_change_j);
    out.floating(moist_commit.moist_water_sensible_change_j);
    out.floating(moist_commit.moist_phase_reference_change_j);
    out.floating(moist_commit.moist_carrier_kinetic_change_j);
    out.floating(moist_commit.moist_gravitational_change_j);
    out.floating(moist_commit.moist_subgrid_tke_change_j);
    out.floating(moist_commit.moist_committed_energy_change_j);
    out.floating(moist_commit.moist_energy_closure_residual_j);
    out.boolean(moist_commit.moist_energy_complete);
    const auto encode_sgs_moist_energy = [&out](const TurbulenceMomentumWorkReport& report)
    {
        out.floating(report.moist_dry_internal_change_j);
        out.floating(report.moist_water_sensible_change_j);
        out.floating(report.moist_phase_reference_change_j);
        out.floating(report.moist_carrier_kinetic_change_j);
        out.floating(report.moist_gravitational_change_j);
        out.floating(report.moist_subgrid_tke_change_j);
        out.floating(report.moist_committed_energy_change_j);
        out.floating(report.moist_energy_closure_residual_j);
        out.boolean(report.moist_energy_complete);
    };
    encode_sgs_moist_energy(runtime.turbulence.last_momentum_work);
    encode_sgs_moist_energy(runtime.turbulence.cumulative_moist_energy_work);
    const auto encode_diffusion_moist_energy = [&out](const DiffusionMomentumWorkReport& report)
    {
        out.floating(report.moist_dry_internal_change_j);
        out.floating(report.moist_water_sensible_change_j);
        out.floating(report.moist_phase_reference_change_j);
        out.floating(report.moist_carrier_kinetic_change_j);
        out.floating(report.moist_gravitational_change_j);
        out.floating(report.moist_subgrid_tke_change_j);
        out.floating(report.moist_committed_energy_change_j);
        out.floating(report.moist_energy_closure_residual_j);
        out.boolean(report.moist_energy_complete);
    };
    encode_diffusion_moist_energy(runtime.diffusion.last_momentum_work);
    encode_diffusion_moist_energy(runtime.diffusion.cumulative_moist_energy_work);
    out.unsigned_value(runtime.diffusion.moist_energy_step_count);
    const auto encode_eos_pressure_energy = [&out](const tmv::dynamics_eos::PressureProjectionEnergyReportV1& report)
    {
        out.floating(report.dry_internal_change_j);
        out.floating(report.water_sensible_change_j);
        out.floating(report.phase_reference_change_j);
        out.floating(report.carrier_kinetic_change_j);
        out.floating(report.gravitational_change_j);
        out.floating(report.subgrid_tke_change_j);
        out.floating(report.committed_energy_change_j);
        out.floating(report.closure_residual_j);
        out.boolean(report.complete);
    };
    encode_eos_pressure_energy(runtime.time_stepping.last_eos_pressure_energy);
    encode_eos_pressure_energy(runtime.time_stepping.cumulative_eos_pressure_energy);
    out.unsigned_value(runtime.time_stepping.eos_pressure_energy_step_count);
    const auto encode_sponge_pass = [&out](const tmv::SpongePassEnergyReportV1& pass)
    {
        out.floating(pass.momentum_requested_linear_work_j);
        out.floating(pass.momentum_finite_step_correction_j);
        out.floating(pass.momentum_storage_rounding_j);
        out.floating(pass.dry_internal_change_j);
        out.floating(pass.water_sensible_change_j);
        out.floating(pass.phase_reference_change_j);
        out.floating(pass.carrier_kinetic_change_j);
        out.floating(pass.gravitational_change_j);
        out.floating(pass.subgrid_tke_change_j);
        out.floating(pass.committed_energy_change_j);
        out.floating(pass.closure_residual_j);
        out.boolean(pass.applied);
        out.boolean(pass.complete);
    };
    const auto encode_sponge_energy = [&encode_sponge_pass](const tmv::SpongeEnergyReportV1& report)
    {
        encode_sponge_pass(report.upper);
        encode_sponge_pass(report.w_only);
        encode_sponge_pass(report.lateral);
        // All pass reports are retained independently; this aggregate bit
        // states whether the completed operator transaction is usable.
        // It is not inferred during decoding so corrupt combinations fail.
        // NOLINTNEXTLINE(bugprone-use-after-move)
        return report.complete;
    };
    out.boolean(encode_sponge_energy(runtime.time_stepping.last_sponge_energy));
    out.boolean(encode_sponge_energy(runtime.time_stepping.cumulative_sponge_energy));
    out.unsigned_value(runtime.time_stepping.sponge_energy_step_count);
    const auto encode_final_boundary_energy = [&out](const tmv::FinalBoundaryEnergyReportV1& report)
    {
        out.floating(report.constraint_committed_work_j);
        out.floating(report.dry_internal_change_j);
        out.floating(report.water_sensible_change_j);
        out.floating(report.phase_reference_change_j);
        out.floating(report.carrier_kinetic_change_j);
        out.floating(report.gravitational_change_j);
        out.floating(report.subgrid_tke_change_j);
        out.floating(report.committed_energy_change_j);
        out.floating(report.closure_residual_j);
        out.boolean(report.applied);
        out.boolean(report.complete);
    };
    encode_final_boundary_energy(runtime.time_stepping.last_final_boundary_energy);
    encode_final_boundary_energy(runtime.time_stepping.cumulative_final_boundary_energy);
    out.unsigned_value(runtime.time_stepping.final_boundary_energy_step_count);
    const auto encode_completed_step_energy = [&out](const tmv::CompletedStepEnergyReportV1& report)
    {
        out.floating(report.acoustic_committed_change_j);
        out.floating(report.slow_dynamics_committed_change_j);
        out.floating(report.scalar_transport_committed_change_j);
        out.floating(report.thermodynamic_sources_committed_change_j);
        out.floating(report.sgs_committed_change_j);
        out.floating(report.diffusion_committed_change_j);
        out.floating(report.eos_projection_committed_change_j);
        out.floating(report.sponge_committed_change_j);
        out.floating(report.final_boundary_committed_change_j);
        out.floating(report.committed_owner_sum_j);
        out.floating(report.dry_internal_change_j);
        out.floating(report.water_sensible_change_j);
        out.floating(report.phase_reference_change_j);
        out.floating(report.carrier_kinetic_change_j);
        out.floating(report.gravitational_change_j);
        out.floating(report.subgrid_tke_change_j);
        out.floating(report.closure_residual_j);
        out.boolean(report.complete);
    };
    encode_completed_step_energy(runtime.time_stepping.last_completed_step_energy);
    encode_completed_step_energy(runtime.time_stepping.cumulative_completed_step_energy);
    out.unsigned_value(runtime.time_stepping.completed_step_energy_count);
    const auto& storm = runtime.storm_tracking;
    encode_enum(out, storm.status);
    encode_enum(out, storm.reason);
    out.unsigned_value<std::uint64_t>(storm.sample_count);
    for (std::size_t index = 0; index < storm.sample_count; ++index)
    {
        out.floating(storm.samples[index].time_s);
        out.floating(storm.samples[index].x_m);
        out.floating(storm.samples[index].y_m);
        out.floating(storm.samples[index].confidence);
    }
    out.unsigned_value<std::uint64_t>(storm.missed_samples);
    out.unsigned_value(storm.branch_id);
    out.boolean(storm.motion_available);
    out.floating(storm.velocity_x_ms);
    out.floating(storm.velocity_y_ms);
    out.floating(storm.speed_ms);
    out.floating(storm.velocity_uncertainty_ms);
    out.floating(storm.continuity_confidence);
    out.floating(storm.innovation_distance_m);
    out.floating(storm.association_gate_radius_m);
    encode_enum(out, storm.topology);
    encode_enum(out, storm.pending_topology);
    encode_enum(out, storm.topology_event);
    out.unsigned_value<std::uint64_t>(storm.pending_topology_samples);
    out.unsigned_value(storm.topology_event_id);
    out.boolean(storm.selected_candidate_available);
    out.unsigned_value<std::uint64_t>(storm.selected_candidate_rank);
    out.unsigned_value<std::uint64_t>(storm.selected_component_label);
    encode_enum(out, storm.selected_motion_class);
    out.floating(storm.selected_bunkers_right_mismatch_ms);
    out.floating(storm.selected_bunkers_left_mismatch_ms);
    out.boolean(storm.association_envelope_available);
    out.floating(storm.association_envelope_west_m);
    out.floating(storm.association_envelope_east_m);
    out.floating(storm.association_envelope_south_m);
    out.floating(storm.association_envelope_north_m);
    out.boolean(storm.topology_reorganization);
    out.floating(storm.reorganization_overlap_fraction);
    out.boolean(runtime.storm_selected_partition_available);
    if (runtime.storm_selected_partition_available)
    {
        out.unsigned_value<std::uint64_t>(runtime.storm_selected_partition_nx);
        out.unsigned_value<std::uint64_t>(runtime.storm_selected_partition_ny);
        out.unsigned_value<std::uint64_t>(runtime.storm_selected_partition.size());
        out.raw(runtime.storm_selected_partition);
    }
    const auto& movement = runtime.storm_window_movement;
    out.boolean(movement.armed);
    out.integer(movement.pending_i_direction);
    out.integer(movement.pending_j_direction);
    out.unsigned_value<std::uint64_t>(movement.consecutive_compatible_samples);
    out.boolean(movement.sample_time_available);
    out.floating(movement.last_sample_time_s);
    out.boolean(movement.committed_move_available);
    out.floating(movement.last_committed_time_s);
    out.unsigned_value<std::uint64_t>(static_cast<std::uint64_t>(movement.cumulative_i_shift_cells));
    out.unsigned_value<std::uint64_t>(static_cast<std::uint64_t>(movement.cumulative_j_shift_cells));
    encode_domain_recycling_provenance(out, runtime.domain_recycling_provenance);
    // Runtime v37 appends the conservative stochastic-PBL cache so every
    // earlier runtime section remains a strict prefix and can still decode.
    encode_field(out, runtime.chaos.base_dtke_dt_pbl);
    encode_float_vector(out, runtime.chaos.base_sfc_ustar);
    encode_float_vector(out, runtime.chaos.base_sfc_tau_u);
    encode_float_vector(out, runtime.chaos.base_sfc_tau_v);
    encode_float_vector(out, runtime.chaos.base_sfc_sensible_heat_flux);
    encode_float_vector(out, runtime.chaos.base_sfc_latent_heat_flux);
    encode_float_vector(out, runtime.chaos.base_sfc_moisture_flux);
    // Runtime v38 appends realized open-stress stability provenance. Keeping
    // it at the section tail preserves every older runtime layout as a strict
    // prefix.
    out.floating(runtime.turbulence.last_open_boundary_work_j);
    out.floating(runtime.turbulence.last_irreversible_dissipation_j);
    out.floating(runtime.turbulence.last_stress_closure_residual_j);
    out.floating(runtime.turbulence.cumulative_open_boundary_work_j);
    out.floating(runtime.turbulence.cumulative_irreversible_dissipation_j);
    out.floating(runtime.turbulence.cumulative_stress_closure_residual_j);
    out.unsigned_value(runtime.turbulence.open_boundary_stability_step_count);
    // Runtime v39 appends the mapped-terrain shortwave controls and complete
    // held power ledger. Older runtime layouts remain strict prefixes.
    const auto& radiation = runtime.radiation;
    out.unsigned_value<std::uint64_t>(radiation.config.terrain_sw_cosine_nodes);
    out.unsigned_value<std::uint64_t>(radiation.config.terrain_sw_azimuth_nodes);
    out.unsigned_value<std::uint64_t>(radiation.config.terrain_sw_maximum_bounces);
    out.floating(radiation.config.terrain_sw_relative_residual_tolerance);
    out.floating(radiation.config.terrain_sw_absolute_residual_tolerance_w);
    out.floating(radiation.held_shortwave_incoming_power_w);
    out.floating(radiation.held_shortwave_atmospheric_absorbed_power_w);
    out.floating(radiation.held_surface_absorbed_power_w);
    out.floating(radiation.held_top_escape_power_w);
    out.floating(radiation.held_lateral_escape_power_w);
    out.floating(radiation.held_unresolved_reflected_power_w);
    out.boolean(radiation.held_reflection_converged);
    // Runtime v40 appends the passive-tracer producer-boundary roundoff
    // ledger. Legacy sections decode with incomplete/unknown provenance.
    const auto encode_tracer_roundoff = [&out](const PositiveDefiniteScalarRoundoffReport& report)
    {
        out.unsigned_value<std::uint32_t>(PositiveDefiniteScalarRoundoffReport::report_schema_version);
        out.unsigned_value<std::uint32_t>(PositiveDefiniteScalarRoundoffReport::classification_policy_version);
        out.floating(report.physical_inventory_correction_kg);
        out.floating(report.thermodynamic_energy_correction_j);
        out.floating(report.most_negative_repaired_value);
        out.unsigned_value<std::uint64_t>(report.negative_subnormal_count);
        out.unsigned_value<std::uint64_t>(report.negative_zero_count);
        out.unsigned_value<std::uint64_t>(report.closure_canonicalization_count);
        out.boolean(report.complete);
    };
    encode_tracer_roundoff(runtime.advection.tracer_roundoff_report);
    encode_tracer_roundoff(runtime.advection.cumulative_tracer_roundoff_report);
    out.unsigned_value(runtime.advection.tracer_roundoff_report_step_count);
    out.boolean(runtime.advection.tracer_roundoff_cumulative_history_known);
    // Runtime v41 appends the complete native RRTMG scalar science contract.
    // The tail placement preserves every older runtime layout as a strict
    // prefix while preventing resumed runs from reverting to gas defaults.
    if (!valid_rrtmg_config(radiation.config))
    {
        error = "invalid RRTMG checkpoint configuration";
        return;
    }
    encode_rrtmg_config(out, radiation.config);
    encode_diffusion_profile_config(out, runtime.diffusion.config);
    if (!valid_radar_product_selection(runtime.radar.product_selection))
    {
        error = "invalid radar product checkpoint configuration";
        return;
    }
    encode_radar_product_selection(out, runtime.radar.product_selection);
    // Runtime v45 appends the versioned simple-grey contract and committed
    // interface fluxes; earlier runtime layouts remain strict prefixes.
    if (!valid_simple_grey_checkpoint_config(radiation.config) ||
        !valid_radiation_flux_diagnostics(radiation.held_flux_diagnostics))
    {
        error = "invalid simple-grey checkpoint state";
        return;
    }
    encode_simple_grey_config(out, radiation.config);
    const auto& fluxes = radiation.held_flux_diagnostics;
    out.unsigned_value<std::uint64_t>(fluxes.columns_x);
    out.unsigned_value<std::uint64_t>(fluxes.columns_y);
    out.unsigned_value<std::uint64_t>(fluxes.levels);
    encode_double_vector(out, fluxes.upward_longwave_wm2);
    encode_double_vector(out, fluxes.downward_longwave_wm2);
    encode_double_vector(out, fluxes.upward_shortwave_wm2);
    encode_double_vector(out, fluxes.downward_shortwave_wm2);
    const auto& turbulence = runtime.turbulence.config;
    if (!known_turbulence_stability_model(turbulence.stability_correction))
    {
        error = "unknown turbulence stability model in checkpoint";
        return;
    }
    if (!std::isfinite(turbulence.user_filter_width_m) ||
        (turbulence.filter_width == "user" ? turbulence.user_filter_width_m <= 0.0
                                           : turbulence.user_filter_width_m != 0.0))
    {
        error = "invalid user filter-width checkpoint configuration";
        return;
    }
    out.floating(turbulence.user_filter_width_m);
    out.floating(runtime.microphysics.last_thermodynamic_repair_report.boundary_layer_cloud_liquid_tendency_kg);
    out.floating(runtime.microphysics.last_thermodynamic_repair_report.boundary_layer_cloud_ice_tendency_kg);
    // Runtime v49 appends horizontal terrain placement and its committed
    // Earth-frame generation. The legacy terrain prefix remains unchanged so
    // v48 and older checkpoints restore the grid-attached zero identity.
    encode_terrain_placement_config(out, runtime.terrain.config);
    encode_terrain_placement(out, runtime.terrain.earth_fixed_placement);
    out.floating(runtime.time_stepping.last_completed_step_energy.unsplit_rk_committed_change_j);
    out.floating(runtime.time_stepping.cumulative_completed_step_energy.unsplit_rk_committed_change_j);
    if (runtime.compute.vertical_flux_template_id != "tvd_vertical_flux_v1" &&
        runtime.compute.vertical_flux_template_id != "tvd_vertical_flux_v2")
    {
        error = "invalid vertical-flux execution selection";
        return;
    }
    out.string(runtime.compute.vertical_flux_template_id);
    out.boolean(runtime.chaos.config.native_generation_required);
    out.unsigned_value<std::uint8_t>(static_cast<std::uint8_t>(runtime.terrain.earth_fixed_placement.sampling));
    out.floating(runtime.time_stepping.last_completed_step_energy.coordinate_remap_committed_change_j);
    out.floating(runtime.time_stepping.cumulative_completed_step_energy.coordinate_remap_committed_change_j);
    {
        const auto& value = runtime.domain_recycling_provenance;
        out.boolean(value.private_inventory_complete);
        if (value.private_inventory_complete)
        {
            out.unsigned_value(value.retiring_private_inventory.count);
            for (std::size_t f = 0; f < value.retiring_private_inventory.count; ++f)
            {
                encode_provenance_text(out, value.retiring_private_inventory.entries[f].name);
                out.floating(value.retiring_private_inventory.entries[f].inventory);
                out.floating(value.admitted_private_inventory.entries[f].inventory);
            }
        }
    }
}

GridMetrics checkpoint_grid_metrics(const SimulationStorageState& storage)
{
    GridMetrics grid;
    grid.dx = storage.grid.dx;
    grid.dy = storage.grid.dy;
    grid.dz.assign(static_cast<std::size_t>(std::max(0, storage.grid.nz)), storage.grid.dz);
    grid.z_int = storage.grid.geometry.z_faces;
    return grid;
}

bool decode_runtime(Decoder& in, SimulationStorageState& storage, std::uint32_t section_version, std::string& error)
{
    auto&              runtime           = storage.runtime;
    std::int32_t       device_index      = -1;
    ComputeBackendKind requested_kind    = ComputeBackendKind::Cpu;
    bool               radiation_present = false, turbulence_present = false;
    if (!in.string(runtime.compute.config.backend) || !in.integer(device_index) ||
        !in.boolean(runtime.compute.config.allow_fallback) || !in.boolean(runtime.compute.config.validate_parity) ||
        !decode_enum(in, requested_kind))
        return false;
    runtime.compute.config.device_index = device_index;
    runtime.compute.reset();
    runtime.compute.requested_kind = requested_kind;
    std::uint64_t parity_stats     = 0;
    if (!in.unsigned_value(parity_stats) || parity_stats > wire::max_elements)
        return false;
    runtime.parity.stats.clear();
    for (std::uint64_t index = 0; index < parity_stats; ++index)
    {
        std::string                      kernel, field;
        nimbus::parity::ComparisonDomain key_domain;
        nimbus::parity::ParityFieldStats stats;
        if (!in.string(kernel) || !in.string(field) || !decode_enum(in, key_domain) || !decode_enum(in, stats.tier) ||
            !decode_enum(in, stats.domain) || !in.unsigned_value(stats.samples) ||
            !in.unsigned_value(stats.failed_samples) || !in.unsigned_value(stats.empty_samples) ||
            !in.unsigned_value(stats.cells_compared) || !in.unsigned_value(stats.breach_cells) ||
            !in.unsigned_value(stats.nonfinite_cells) || !in.floating(stats.worst_abs) ||
            !in.floating(stats.worst_rel) || !in.integer(stats.worst_i) || !in.integer(stats.worst_j) ||
            !in.integer(stats.worst_k) || !in.floating(stats.worst_cpu) || !in.floating(stats.worst_gpu) ||
            !runtime.parity.stats
                 .emplace(nimbus::parity::ParityRuntimeState::Key{std::move(kernel), std::move(field), key_domain},
                          stats)
                 .second)
            return false;
    }
    std::uint64_t parity_counters = 0;
    if (!in.unsigned_value(parity_counters) || parity_counters > wire::max_elements)
        return false;
    runtime.parity.call_counters.clear();
    for (std::uint64_t index = 0; index < parity_counters; ++index)
    {
        std::string   key;
        std::uint64_t value = 0;
        if (!in.string(key) || !in.unsigned_value(value) ||
            !runtime.parity.call_counters.emplace(std::move(key), value).second)
            return false;
    }
    if (!in.unsigned_value(runtime.parity.total_samples) || !in.boolean(runtime.parity.any_breach) ||
        !in.integer(runtime.parity.env_enabled) || !in.integer(runtime.parity.env_every))
        return false;
    if (!decode_radiation_config(in, runtime.radiation.config, false) ||
        !in.floating(runtime.radiation.last_update_time_s))
        return false;
    if (section_version >= 2 && (!in.floating(runtime.radiation.next_update_time_s) ||
                                 !in.floating(runtime.radiation.held_atmospheric_power_w)))
        return false;
    if (section_version >= 6 && !decode_field(in, runtime.radiation.held_cell_power_w))
        return false;
    if (section_version >= 2 && (!in.floating(runtime.radiation.last_expected_atmospheric_energy_j) ||
                                 !in.floating(runtime.radiation.last_applied_radiation_heat_j) ||
                                 !in.floating(runtime.radiation.accumulated_expected_atmospheric_energy_j) ||
                                 !in.floating(runtime.radiation.accumulated_applied_radiation_heat_j) ||
                                 !in.unsigned_value(runtime.radiation.applied_step_count)))
        return false;
    if (section_version == 1)
    {
        runtime.radiation.next_update_time_s =
            runtime.radiation.last_update_time_s + runtime.radiation.config.dt_radiation;
    }
    if (section_version < 6 && runtime.radiation.config.scheme_id != "none")
    {
        // Older checkpoints retained only a theta tendency. Its energy changes
        // as density/Exner evolve, so force an immediate radiation refresh
        // before the resumed thermodynamic step instead of inventing a cell-
        // power distribution that was never checkpointed.
        runtime.radiation.held_cell_power_w.release_storage();
        runtime.radiation.next_update_time_s = runtime.radiation.last_update_time_s;
    }
    if (!in.boolean(radiation_present) || !decode_turbulence_config(in, runtime.turbulence.config) ||
        !in.floating(runtime.turbulence.last_update_time_s))
        return false;
    if (section_version >= 4)
    {
        auto& held = runtime.turbulence.held_tendencies;
        if (!in.floating(runtime.turbulence.next_update_time_s) || !decode_field(in, held.dudt_sgs) ||
            !decode_field(in, held.dvdt_sgs) || !decode_field(in, held.dwdt_sgs) ||
            !decode_field(in, held.dthetadt_sgs) || !decode_field(in, held.dqvdt_sgs) ||
            !decode_field(in, held.dtkedt_sgs))
            return false;
        // Device residency never survives a process checkpoint even though
        // the held host tendency does.
        held.resident_source_unchanged = false;
    }
    else
    {
        // Older checkpoints did not persist the held SGS source. Force an
        // immediate scientifically safe refresh on the first resumed step.
        runtime.turbulence.next_update_time_s = runtime.turbulence.last_update_time_s;
    }
    if (section_version >= 9)
    {
        auto& work = runtime.turbulence.last_momentum_work;
        if (!in.floating(work.requested_linear_work_j) || !in.floating(work.finite_step_correction_j) ||
            !in.floating(work.storage_rounding_j) || !in.floating(work.committed_work_j) ||
            !in.floating(runtime.turbulence.cumulative_requested_momentum_work_j) ||
            !in.floating(runtime.turbulence.cumulative_finite_step_momentum_correction_j) ||
            !in.floating(runtime.turbulence.cumulative_momentum_storage_rounding_j) ||
            !in.floating(runtime.turbulence.cumulative_committed_momentum_work_j) ||
            !in.unsigned_value(runtime.turbulence.momentum_work_step_count))
            return false;
    }
    if (section_version >= 14)
    {
        if (!in.floating(runtime.turbulence.last_explicit_stability_limit_s) ||
            !in.unsigned_value(runtime.turbulence.last_explicit_substep_count))
            return false;
    }
    else
    {
        // The first resumed SGS step recomputes its live bound.  Infinity
        // merely withholds the predictive outer-step cap until then.
        runtime.turbulence.last_explicit_stability_limit_s = std::numeric_limits<double>::infinity();
        runtime.turbulence.last_explicit_substep_count     = 0;
    }
    if (!in.boolean(turbulence_present))
        return false;
    bool                      boundary_present = false;
    std::uint64_t             boundary_size    = 0;
    std::vector<std::uint8_t> boundary_bytes;
    if (!in.boolean(boundary_present) || !in.unsigned_value(boundary_size) || boundary_size > wire::max_file_bytes ||
        !in.raw(static_cast<std::size_t>(boundary_size), boundary_bytes) ||
        !deserialize_boundary_layer_runtime(boundary_bytes, runtime.boundary_layer, error))
        return false;
    if (!boundary_present)
        runtime.boundary_layer.scheme.reset();

    bool                chaos_present = false;
    PortableSchemeState chaos_state;
    if (!decode_chaos_config(in, runtime.chaos.config) || !in.floating(runtime.chaos.diagnostics.mean_perturbation) ||
        !in.floating(runtime.chaos.diagnostics.variance_perturbation) ||
        !in.floating(runtime.chaos.diagnostics.correlation_length_x) ||
        !in.floating(runtime.chaos.diagnostics.correlation_length_y) ||
        !in.boolean(runtime.chaos.diagnostics.negative_moisture_detected) ||
        !in.integer(runtime.chaos.diagnostics.negative_moisture_count) ||
        !in.floating(runtime.chaos.diagnostics.min_qv_after_perturbation) ||
        !decode_string_vector(in, runtime.chaos.diagnostics.warnings) ||
        !decode_string_vector(in, runtime.chaos.diagnostics.errors) ||
        !in.floating(runtime.chaos.diagnostics.time_noise_update) ||
        !in.floating(runtime.chaos.diagnostics.time_apply_ic) ||
        !in.floating(runtime.chaos.diagnostics.time_apply_tend) || !in.floating(runtime.chaos.noise_elapsed_s) ||
        !in.boolean(runtime.chaos.pbl_base_tendencies_valid) || !decode_field(in, runtime.chaos.base_du_dt_pbl) ||
        !decode_field(in, runtime.chaos.base_dv_dt_pbl) || !decode_field(in, runtime.chaos.base_dtheta_dt_pbl) ||
        !decode_field(in, runtime.chaos.base_dqv_dt_pbl))
        return false;
    if (section_version < 37)
    {
        // Runtime v1-v36 cached already density-divided PBL tendencies. They
        // cannot be reinterpreted as the v37 conserved force/inventory source
        // currency, so retain restart compatibility but force a fresh cache.
        runtime.chaos.base_du_dt_pbl     = Field3D();
        runtime.chaos.base_dv_dt_pbl     = Field3D();
        runtime.chaos.base_dtheta_dt_pbl = Field3D();
        runtime.chaos.base_dqv_dt_pbl    = Field3D();
        runtime.chaos.base_dtke_dt_pbl   = Field3D();
        runtime.chaos.base_sfc_ustar.clear();
        runtime.chaos.base_sfc_tau_u.clear();
        runtime.chaos.base_sfc_tau_v.clear();
        runtime.chaos.base_sfc_sensible_heat_flux.clear();
        runtime.chaos.base_sfc_latent_heat_flux.clear();
        runtime.chaos.base_sfc_moisture_flux.clear();
        runtime.chaos.pbl_base_tendencies_valid = false;
    }
    if (!in.boolean(chaos_present) || !decode_portable_state(in, chaos_state))
        return false;
    if (chaos_present)
    {
        runtime.chaos.scheme = chaos::create_chaos_scheme(runtime.chaos.config.scheme_id);
        runtime.chaos.scheme->initialize(runtime.chaos.config, checkpoint_grid_metrics(storage));
        if (!runtime.chaos.scheme->restore_portable_state(chaos_state, error))
            return false;
    }
    else if (!empty_portable_scheme_state(chaos_state))
        return false;

    if (!in.boolean(runtime.rayleigh.upper_initialized) || !in.boolean(runtime.rayleigh.lateral_initialized) ||
        !decode_double_vector(in, runtime.rayleigh.alpha) || !decode_double_vector(in, runtime.rayleigh.u_ref) ||
        !decode_double_vector(in, runtime.rayleigh.v_ref) || !decode_double_vector(in, runtime.rayleigh.w_only_alpha) ||
        !decode_double_vector(in, runtime.rayleigh.lateral_alpha))
        return false;

    std::string         micro_name;
    PortableSchemeState micro_state;
    if (!in.string(micro_name) || !decode_portable_state(in, micro_state))
        return false;
    if (!micro_name.empty())
    {
        runtime.microphysics.scheme = create_microphysics_scheme(micro_name);
        runtime.microphysics.scheme->configure(storage.configuration.microphysics_config);
        runtime.microphysics.scheme->set_vertical_spacing_m(storage.grid.dz);
        if (!runtime.microphysics.scheme->restore_portable_state(micro_state, error))
            return false;
    }
    else if (!empty_portable_scheme_state(micro_state))
        return false;
    for (auto& repair : runtime.microphysics.last_water_repair_report.species)
    {
        std::uint64_t lower = 0, upper = 0, nonfinite = 0;
        if (!in.floating(repair.lower_bound_correction_kg) || !in.floating(repair.upper_bound_correction_kg) ||
            !in.floating(repair.nonfinite_correction_kg) || !in.unsigned_value(lower) || !in.unsigned_value(upper) ||
            !in.unsigned_value(nonfinite) || lower > std::numeric_limits<std::size_t>::max() ||
            upper > std::numeric_limits<std::size_t>::max() || nonfinite > std::numeric_limits<std::size_t>::max())
            return false;
        repair.lower_bound_count = static_cast<std::size_t>(lower);
        repair.upper_bound_count = static_cast<std::size_t>(upper);
        repair.nonfinite_count   = static_cast<std::size_t>(nonfinite);
        if (section_version >= 5)
        {
            std::uint64_t roundoff_lower = 0;
            if (!in.floating(repair.lower_bound_roundoff_correction_kg) || !in.unsigned_value(roundoff_lower) ||
                roundoff_lower > std::numeric_limits<std::size_t>::max())
                return false;
            repair.lower_bound_roundoff_count = static_cast<std::size_t>(roundoff_lower);
        }
        if (section_version >= 7 && !in.floating(repair.storage_rounding_kg))
            return false;
    }
    std::uint64_t nonfinite_tendencies = 0;
    if (!in.unsigned_value(nonfinite_tendencies) || nonfinite_tendencies > std::numeric_limits<std::size_t>::max())
        return false;
    runtime.microphysics.last_water_repair_report.nonfinite_tendency_count =
        static_cast<std::size_t>(nonfinite_tendencies);
    auto&   thermo          = runtime.microphysics.last_thermodynamic_repair_report;
    double* thermo_prefix[] = {&thermo.microphysics_heat_j,
                               &thermo.radiation_heat_j,
                               &thermo.boundary_layer_heat_j,
                               &thermo.theta_tendency_limit_correction_j,
                               &thermo.theta_nonfinite_correction_j,
                               &thermo.theta_storage_rounding_j};
    for (double* value : thermo_prefix)
        if (!in.floating(*value))
            return false;
    if (section_version >= 3 && !in.floating(thermo.theta_storage_residual_j))
        return false;
    if (section_version >= 7 && !in.floating(thermo.theta_storage_residual_release_j))
        return false;
    double* thermo_suffix[] = {&thermo.theta_bound_correction_j,
                               &thermo.microphysics_water_tendency_kg,
                               &thermo.boundary_layer_vapor_tendency_kg};
    for (double* value : thermo_suffix)
        if (!in.floating(*value))
            return false;
    std::uint64_t thermo_counts[4]{};
    for (auto& count : thermo_counts)
        if (!in.unsigned_value(count) || count > std::numeric_limits<std::size_t>::max())
            return false;
    thermo.theta_tendency_limit_count = static_cast<std::size_t>(thermo_counts[0]);
    thermo.theta_nonfinite_count      = static_cast<std::size_t>(thermo_counts[1]);
    thermo.theta_lower_bound_count    = static_cast<std::size_t>(thermo_counts[2]);
    thermo.theta_upper_bound_count    = static_cast<std::size_t>(thermo_counts[3]);
    if (section_version >= 3 && !decode_field(in, runtime.microphysics.theta_storage_residual))
        return false;
    if (section_version >= 7)
    {
        auto& micro_conservation = runtime.microphysics.last_conservation_report;
        if (!in.boolean(micro_conservation.active) || !in.floating(micro_conservation.tendency_imbalance_kg) ||
            !in.floating(micro_conservation.baseline_q_kg) ||
            !in.floating(micro_conservation.surface_precipitation_kg) ||
            !in.floating(micro_conservation.relative_imbalance))
            return false;
    }
    if (section_version >= 8)
    {
        for (double& species_tendency : thermo.microphysics_water_tendency_kg_by_species)
        {
            if (!in.floating(species_tendency))
                return false;
        }
    }

    bool terrain_present = false;
    if (!decode_terrain_config(in, runtime.terrain.config) || !in.boolean(terrain_present) ||
        !decode_matrix(in, runtime.terrain.topography.h) || !decode_matrix(in, runtime.terrain.topography.hx) ||
        !decode_matrix(in, runtime.terrain.topography.hy) || !decode_terrain_field(in, runtime.terrain.metrics.z) ||
        !decode_terrain_field(in, runtime.terrain.metrics.J) || !decode_terrain_field(in, runtime.terrain.metrics.mx) ||
        !decode_terrain_field(in, runtime.terrain.metrics.my) ||
        !decode_terrain_field(in, runtime.terrain.metrics.zeta))
        return false;
    if (section_version >= 13)
    {
        if (!decode_terrain_field(in, runtime.terrain.metrics.z_face) ||
            !decode_terrain_field(in, runtime.terrain.metrics.Jx_face) ||
            !decode_terrain_field(in, runtime.terrain.metrics.Jy_face) ||
            !decode_terrain_field(in, runtime.terrain.metrics.Gx_face) ||
            !decode_terrain_field(in, runtime.terrain.metrics.Gy_face) || !in.floating(runtime.terrain.metrics.dx) ||
            !in.floating(runtime.terrain.metrics.dy) || !in.floating(runtime.terrain.metrics.dzeta))
            return false;
    }
    else if (terrain_present && runtime.terrain.config.scheme_id != "none")
    {
        // Runtime sections through v12 predate conservative face geometry.
        // Reconstruct it from the checkpointed scientific configuration and
        // topography instead of publishing an active, incomplete metric set.
        topography::build_conservative_face_metrics(
            runtime.terrain.config,
            runtime.terrain.topography,
            topography::build_zeta_interfaces(storage.grid.nz, runtime.terrain.config.ztop),
            storage.grid.dx,
            storage.grid.dy,
            runtime.terrain.metrics);
    }
    if (terrain_present && runtime.terrain.config.scheme_id != "none")
        terrain_transport::validate_geometry(runtime.terrain.metrics);
    if (terrain_present)
    {
        runtime.terrain.scheme = create_terrain_scheme(runtime.terrain.config.scheme_id);
        runtime.terrain.scheme->initialize(runtime.terrain.config);
    }

    bool         radar_present = false;
    std::int32_t radar_nr = -1, radar_nth = -1, radar_nz = -1;
    if (!decode_radar_config(in, runtime.radar.selected_config) || !in.boolean(radar_present) ||
        !in.integer(radar_nr) || !in.integer(radar_nth) || !in.integer(radar_nz) ||
        !in.floating(runtime.radar.radar_x) || !in.floating(runtime.radar.radar_y) ||
        !in.floating(runtime.radar.radar_z))
        return false;
    runtime.radar.nr  = radar_nr;
    runtime.radar.nth = radar_nth;
    runtime.radar.nz  = radar_nz;
    if (radar_present)
    {
        runtime.radar.selected_scheme = create_radar_scheme(runtime.radar.selected_config.scheme_id);
        runtime.radar.selected_scheme->initialize(
            runtime.radar.selected_config, storage.grid.nr, storage.grid.nth, storage.grid.nz);
        runtime.radar.selected_output.initialize(storage.grid.nr, storage.grid.nth, storage.grid.nz);
    }

    bool advection_present = false;
    if (!decode_advection_config(in, runtime.advection.config) || !in.boolean(advection_present) ||
        !in.boolean(runtime.advection.tracer_advected_in_batch))
        return false;
    for (auto&& active : runtime.advection.hydrometeor_active)
    {
        bool value = false;
        if (!in.boolean(value))
            return false;
        active = value;
    }
    if (!in.integer(runtime.advection.nr) || !in.integer(runtime.advection.nth) || !in.integer(runtime.advection.nz) ||
        !in.boolean(runtime.advection.activity_current) ||
        !in.unsigned_value(runtime.advection.metadata_consumptions) ||
        !in.unsigned_value(runtime.advection.fallback_scans))
        return false;
    if (advection_present)
    {
        runtime.advection.scheme = create_advection_scheme(runtime.advection.config.scheme_id);
        runtime.advection.scheme->initialize(runtime.advection.config);
    }
    if (section_version >= 15)
    {
        const auto decode_reports = [&in](ThermodynamicBoundaryTransportReports& reports)
        {
            for (auto& report : reports)
            {
                if (!in.floating(report.outward_boundary_inventory) || !in.floating(report.flux_implied_state_change) ||
                    !in.floating(report.committed_state_change) || !in.floating(report.closure_residual) ||
                    !in.boolean(report.complete))
                {
                    return false;
                }
                if (!std::isfinite(report.outward_boundary_inventory) ||
                    !std::isfinite(report.flux_implied_state_change) || !std::isfinite(report.committed_state_change) ||
                    !std::isfinite(report.closure_residual))
                {
                    return false;
                }
            }
            return true;
        };
        if (!decode_reports(runtime.advection.thermodynamic_boundary_reports) ||
            !decode_reports(runtime.advection.cumulative_thermodynamic_boundary_reports) ||
            !in.unsigned_value(runtime.advection.thermodynamic_boundary_report_step_count))
        {
            return false;
        }
        runtime.advection.thermodynamic_boundary_reports_complete =
            std::all_of(runtime.advection.thermodynamic_boundary_reports.begin(),
                        runtime.advection.thermodynamic_boundary_reports.end(),
                        [](const ScalarBoundaryTransportReport& report) { return report.complete; });
    }
    if (section_version >= 19)
    {
        const auto decode_reports = [&in](ThermodynamicMoistEnergyTransportReports& reports)
        {
            for (auto& report : reports)
            {
                if (!in.floating(report.dry_internal_change_j) || !in.floating(report.water_sensible_change_j) ||
                    !in.floating(report.phase_reference_change_j) || !in.floating(report.carrier_kinetic_change_j) ||
                    !in.floating(report.gravitational_change_j) || !in.floating(report.subgrid_tke_change_j) ||
                    !in.floating(report.density_history_remap_j) || !in.floating(report.outward_boundary_energy_j) ||
                    !in.floating(report.interior_conversion_j) || !in.floating(report.limited_face_state_change_j) ||
                    !in.floating(report.requested_state_change_j) || !in.floating(report.storage_rounding_j) ||
                    !in.floating(report.committed_state_change_j) || !in.floating(report.closure_residual_j) ||
                    !in.boolean(report.complete))
                {
                    return false;
                }
                const double values[] = {
                    report.dry_internal_change_j,
                    report.water_sensible_change_j,
                    report.phase_reference_change_j,
                    report.carrier_kinetic_change_j,
                    report.gravitational_change_j,
                    report.subgrid_tke_change_j,
                    report.density_history_remap_j,
                    report.outward_boundary_energy_j,
                    report.interior_conversion_j,
                    report.limited_face_state_change_j,
                    report.requested_state_change_j,
                    report.storage_rounding_j,
                    report.committed_state_change_j,
                    report.closure_residual_j,
                };
                if (std::any_of(
                        std::begin(values), std::end(values), [](double value) { return !std::isfinite(value); }))
                {
                    return false;
                }
            }
            return true;
        };
        if (!decode_reports(runtime.advection.thermodynamic_moist_energy_reports) ||
            !decode_reports(runtime.advection.cumulative_thermodynamic_moist_energy_reports) ||
            !in.unsigned_value(runtime.advection.thermodynamic_moist_energy_report_step_count))
        {
            return false;
        }
        runtime.advection.thermodynamic_moist_energy_reports_complete =
            std::all_of(runtime.advection.thermodynamic_moist_energy_reports.begin(),
                        runtime.advection.thermodynamic_moist_energy_reports.end(),
                        [](const ScalarMoistEnergyTransportReportV1& report) { return report.complete; });
    }

    bool diffusion_present = false;
    if (!decode_diffusion_config(in, runtime.diffusion.config) || !in.boolean(diffusion_present) ||
        !in.floating(runtime.diffusion.diagnostics.max_diffusion_number))
        return false;
    if (section_version >= 11)
    {
        auto& work = runtime.diffusion.last_momentum_work;
        if (!in.floating(work.requested_linear_work_j) || !in.floating(work.finite_step_correction_j) ||
            !in.floating(work.storage_rounding_j) || !in.floating(work.committed_work_j) ||
            !in.floating(runtime.diffusion.cumulative_requested_momentum_work_j) ||
            !in.floating(runtime.diffusion.cumulative_finite_step_momentum_correction_j) ||
            !in.floating(runtime.diffusion.cumulative_momentum_storage_rounding_j) ||
            !in.floating(runtime.diffusion.cumulative_committed_momentum_work_j) ||
            !in.unsigned_value(runtime.diffusion.momentum_work_step_count))
            return false;
    }
    if (diffusion_present)
    {
        runtime.diffusion.scheme = create_diffusion_scheme(runtime.diffusion.config.scheme_id);
        runtime.diffusion.scheme->initialize(runtime.diffusion.config);
    }

    bool time_present = false;
    if (!decode_time_config(in, runtime.time_stepping.config, section_version) || !in.boolean(time_present) ||
        !in.integer(runtime.time_stepping.diagnostics.n_steps) ||
        !in.floating(runtime.time_stepping.diagnostics.max_cfl) ||
        !in.floating(runtime.time_stepping.diagnostics.cpu_time_per_step) ||
        !decode_double_vector(in, runtime.time_stepping.diagnostics.dt_history) ||
        !in.boolean(runtime.time_stepping.diagnostics.converged))
        return false;
    if (section_version >= 10)
    {
        auto& boundary_mass = runtime.time_stepping.last_boundary_mass;
        if (!in.floating(boundary_mass.outward_boundary_mass_kg) ||
            !in.floating(boundary_mass.flux_implied_state_change_kg) ||
            !in.floating(boundary_mass.committed_state_change_kg) || !in.floating(boundary_mass.closure_residual_kg) ||
            !in.floating(runtime.time_stepping.cumulative_outward_boundary_mass_kg) ||
            !in.floating(runtime.time_stepping.cumulative_flux_implied_state_change_kg) ||
            !in.floating(runtime.time_stepping.cumulative_committed_state_change_kg) ||
            !in.floating(runtime.time_stepping.cumulative_boundary_mass_closure_residual_kg) ||
            !in.unsigned_value(runtime.time_stepping.boundary_mass_step_count))
            return false;
    }
    if (section_version >= 16)
    {
        const auto decode_acoustic_energy = [&in, section_version](CartesianCGridAcousticEnergyReportV1& report)
        {
            if (!in.floating(report.continuity_eos_dry_internal_change_j) ||
                !in.floating(report.continuity_eos_water_sensible_change_j) ||
                !in.floating(report.continuity_eos_phase_reference_change_j) ||
                !in.floating(report.continuity_carrier_kinetic_change_j) ||
                !in.floating(report.continuity_gravitational_change_j) ||
                !in.floating(report.continuity_subgrid_tke_change_j) ||
                !in.floating(report.pressure_gradient_requested_linear_work_j) ||
                !in.floating(report.pressure_gradient_finite_step_correction_j) ||
                !in.floating(report.pressure_gradient_storage_rounding_j) ||
                !in.floating(report.pressure_gradient_committed_work_j) ||
                (section_version >= 17 &&
                 (!in.floating(report.boundary_committed_work_j) ||
                  (section_version >= 18 && (!in.floating(report.boundary_radiative_requested_linear_work_j) ||
                                             !in.floating(report.boundary_radiative_finite_step_correction_j) ||
                                             !in.floating(report.boundary_radiative_storage_rounding_j) ||
                                             !in.floating(report.boundary_radiative_committed_work_j) ||
                                             !in.floating(report.boundary_constraint_history_committed_work_j))) ||
                  !in.unsigned_value(report.boundary_application_count) || !in.boolean(report.boundary_complete))) ||
                !in.floating(report.committed_inventory_change_j) || !in.unsigned_value(report.substep_count) ||
                !in.boolean(report.complete) ||
                (section_version >= 21 && (!in.floating(report.continuity_carrier_requested_change_j) ||
                                           !in.floating(report.continuity_carrier_outward_boundary_transport_j) ||
                                           !in.floating(report.continuity_carrier_interior_conversion_j) ||
                                           !in.floating(report.continuity_carrier_storage_rounding_j) ||
                                           !in.floating(report.slow_momentum_advection_requested_linear_work_j) ||
                                           !in.floating(report.kinetic_transport_interior_conversion_j) ||
                                           !in.floating(report.kinetic_transport_closure_residual_j) ||
                                           !in.boolean(report.kinetic_transport_complete))))
            {
                return false;
            }
            const double values[] = {
                report.continuity_eos_dry_internal_change_j,
                report.continuity_eos_water_sensible_change_j,
                report.continuity_eos_phase_reference_change_j,
                report.continuity_carrier_kinetic_change_j,
                report.continuity_gravitational_change_j,
                report.continuity_subgrid_tke_change_j,
                report.continuity_carrier_requested_change_j,
                report.continuity_carrier_outward_boundary_transport_j,
                report.continuity_carrier_interior_conversion_j,
                report.continuity_carrier_storage_rounding_j,
                report.slow_momentum_advection_requested_linear_work_j,
                report.kinetic_transport_interior_conversion_j,
                report.kinetic_transport_closure_residual_j,
                report.pressure_gradient_requested_linear_work_j,
                report.pressure_gradient_finite_step_correction_j,
                report.pressure_gradient_storage_rounding_j,
                report.pressure_gradient_committed_work_j,
                report.boundary_committed_work_j,
                report.boundary_radiative_requested_linear_work_j,
                report.boundary_radiative_finite_step_correction_j,
                report.boundary_radiative_storage_rounding_j,
                report.boundary_radiative_committed_work_j,
                report.boundary_constraint_history_committed_work_j,
                report.committed_inventory_change_j,
            };
            return std::all_of(
                       std::begin(values), std::end(values), [](double value) { return std::isfinite(value); }) &&
                   report.pressure_gradient_finite_step_correction_j >= 0.0 &&
                   report.boundary_radiative_finite_step_correction_j >= 0.0 &&
                   (report.complete == (report.substep_count > 0)) &&
                   (report.boundary_complete == (report.boundary_application_count > 0)) &&
                   (section_version < 21 || report.kinetic_transport_complete == report.complete);
        };
        if (!decode_acoustic_energy(runtime.time_stepping.last_acoustic_energy) ||
            !decode_acoustic_energy(runtime.time_stepping.cumulative_acoustic_energy) ||
            !in.unsigned_value(runtime.time_stepping.acoustic_energy_step_count))
        {
            return false;
        }
        if ((runtime.time_stepping.acoustic_energy_step_count > 0) !=
                runtime.time_stepping.cumulative_acoustic_energy.complete ||
            (section_version >= 17 && (runtime.time_stepping.acoustic_energy_step_count > 0) !=
                                          runtime.time_stepping.cumulative_acoustic_energy.boundary_complete))
        {
            return false;
        }
    }
    if (section_version >= 20)
    {
        const auto decode_slow_momentum_energy = [&in, section_version](CartesianSlowMomentumEnergyReportV1& report)
        {
            if (!in.floating(report.advection_requested_linear_work_j) ||
                !in.floating(report.nonadvection_requested_linear_work_j) ||
                !in.floating(report.advection_finite_step_correction_j) ||
                !in.floating(report.nonadvection_finite_step_correction_j) ||
                !in.floating(report.coupled_finite_step_cross_work_j) || !in.floating(report.storage_rounding_j) ||
                !in.floating(report.committed_kinetic_change_j) || !in.floating(report.closure_residual_j) ||
                (section_version >= 29 && (!in.floating(report.pressure_dry_internal_change_j) ||
                                           !in.floating(report.pressure_water_sensible_change_j) ||
                                           !in.floating(report.committed_pressure_energy_change_j))) ||
                !in.boolean(report.complete))
            {
                return false;
            }
            const double values[] = {
                report.advection_requested_linear_work_j,
                report.nonadvection_requested_linear_work_j,
                report.advection_finite_step_correction_j,
                report.nonadvection_finite_step_correction_j,
                report.coupled_finite_step_cross_work_j,
                report.storage_rounding_j,
                report.committed_kinetic_change_j,
                report.closure_residual_j,
                report.pressure_dry_internal_change_j,
                report.pressure_water_sensible_change_j,
                report.committed_pressure_energy_change_j,
            };
            return std::all_of(
                       std::begin(values), std::end(values), [](double value) { return std::isfinite(value); }) &&
                   report.advection_finite_step_correction_j >= 0.0 &&
                   report.nonadvection_finite_step_correction_j >= 0.0 &&
                   report.committed_pressure_energy_change_j ==
                       report.pressure_dry_internal_change_j + report.pressure_water_sensible_change_j;
        };
        if (!decode_slow_momentum_energy(runtime.time_stepping.last_slow_momentum_energy) ||
            !decode_slow_momentum_energy(runtime.time_stepping.cumulative_slow_momentum_energy) ||
            !in.unsigned_value(runtime.time_stepping.slow_momentum_energy_step_count) ||
            ((runtime.time_stepping.slow_momentum_energy_step_count > 0) !=
             runtime.time_stepping.cumulative_slow_momentum_energy.complete))
        {
            return false;
        }
    }
    if (section_version >= 22)
    {
        auto& report = runtime.microphysics.last_thermodynamic_repair_report;
        if (!in.floating(report.moist_dry_internal_change_j) || !in.floating(report.moist_water_sensible_change_j) ||
            !in.floating(report.moist_phase_reference_change_j) ||
            !in.floating(report.moist_carrier_kinetic_change_j) || !in.floating(report.moist_gravitational_change_j) ||
            !in.floating(report.moist_subgrid_tke_change_j) || !in.floating(report.moist_committed_energy_change_j) ||
            !in.floating(report.moist_energy_closure_residual_j) || !in.boolean(report.moist_energy_complete))
        {
            return false;
        }
        const double values[] = {
            report.moist_dry_internal_change_j,
            report.moist_water_sensible_change_j,
            report.moist_phase_reference_change_j,
            report.moist_carrier_kinetic_change_j,
            report.moist_gravitational_change_j,
            report.moist_subgrid_tke_change_j,
            report.moist_committed_energy_change_j,
            report.moist_energy_closure_residual_j,
        };
        const double reservoir_sum = report.moist_dry_internal_change_j + report.moist_water_sensible_change_j +
                                     report.moist_phase_reference_change_j + report.moist_carrier_kinetic_change_j +
                                     report.moist_gravitational_change_j + report.moist_subgrid_tke_change_j;
        const bool   all_zero =
            std::all_of(std::begin(values), std::end(values), [](double value) { return value == 0.0; });
        if (!std::all_of(std::begin(values), std::end(values), [](double value) { return std::isfinite(value); }) ||
            report.moist_energy_closure_residual_j != report.moist_committed_energy_change_j - reservoir_sum ||
            (!report.moist_energy_complete && !all_zero))
        {
            return false;
        }
    }
    if (section_version >= 23)
    {
        auto decode_sgs_moist_energy = [&in](TurbulenceMomentumWorkReport& report)
        {
            if (!in.floating(report.moist_dry_internal_change_j) ||
                !in.floating(report.moist_water_sensible_change_j) ||
                !in.floating(report.moist_phase_reference_change_j) ||
                !in.floating(report.moist_carrier_kinetic_change_j) ||
                !in.floating(report.moist_gravitational_change_j) || !in.floating(report.moist_subgrid_tke_change_j) ||
                !in.floating(report.moist_committed_energy_change_j) ||
                !in.floating(report.moist_energy_closure_residual_j) || !in.boolean(report.moist_energy_complete))
            {
                return false;
            }
            const double values[] = {
                report.moist_dry_internal_change_j,
                report.moist_water_sensible_change_j,
                report.moist_phase_reference_change_j,
                report.moist_carrier_kinetic_change_j,
                report.moist_gravitational_change_j,
                report.moist_subgrid_tke_change_j,
                report.moist_committed_energy_change_j,
                report.moist_energy_closure_residual_j,
            };
            const double reservoir_sum = report.committed_work_j + report.moist_dry_internal_change_j +
                                         report.moist_water_sensible_change_j + report.moist_phase_reference_change_j +
                                         report.moist_carrier_kinetic_change_j + report.moist_gravitational_change_j +
                                         report.moist_subgrid_tke_change_j;
            const bool   all_zero =
                std::all_of(std::begin(values), std::end(values), [](double value) { return value == 0.0; });
            return std::all_of(
                       std::begin(values), std::end(values), [](double value) { return std::isfinite(value); }) &&
                   (!report.moist_energy_complete ? all_zero
                                                  : report.moist_energy_closure_residual_j ==
                                                        report.moist_committed_energy_change_j - reservoir_sum);
        };
        auto& cumulative                    = runtime.turbulence.cumulative_moist_energy_work;
        cumulative.requested_linear_work_j  = runtime.turbulence.cumulative_requested_momentum_work_j;
        cumulative.finite_step_correction_j = runtime.turbulence.cumulative_finite_step_momentum_correction_j;
        cumulative.storage_rounding_j       = runtime.turbulence.cumulative_momentum_storage_rounding_j;
        cumulative.committed_work_j         = runtime.turbulence.cumulative_committed_momentum_work_j;
        if (!decode_sgs_moist_energy(runtime.turbulence.last_momentum_work) || !decode_sgs_moist_energy(cumulative))
        {
            return false;
        }
    }
    if (section_version >= 24)
    {
        auto decode_diffusion_moist_energy = [&in](DiffusionMomentumWorkReport& report)
        {
            if (!in.floating(report.moist_dry_internal_change_j) ||
                !in.floating(report.moist_water_sensible_change_j) ||
                !in.floating(report.moist_phase_reference_change_j) ||
                !in.floating(report.moist_carrier_kinetic_change_j) ||
                !in.floating(report.moist_gravitational_change_j) || !in.floating(report.moist_subgrid_tke_change_j) ||
                !in.floating(report.moist_committed_energy_change_j) ||
                !in.floating(report.moist_energy_closure_residual_j) || !in.boolean(report.moist_energy_complete))
                return false;
            const double values[] = {
                report.moist_dry_internal_change_j,
                report.moist_water_sensible_change_j,
                report.moist_phase_reference_change_j,
                report.moist_carrier_kinetic_change_j,
                report.moist_gravitational_change_j,
                report.moist_subgrid_tke_change_j,
                report.moist_committed_energy_change_j,
                report.moist_energy_closure_residual_j,
            };
            const double reservoir_sum = report.committed_work_j + report.moist_dry_internal_change_j +
                                         report.moist_water_sensible_change_j + report.moist_phase_reference_change_j +
                                         report.moist_carrier_kinetic_change_j + report.moist_gravitational_change_j +
                                         report.moist_subgrid_tke_change_j;
            const bool   all_zero =
                std::all_of(std::begin(values), std::end(values), [](double value) { return value == 0.0; });
            return std::all_of(
                       std::begin(values), std::end(values), [](double value) { return std::isfinite(value); }) &&
                   (!report.moist_energy_complete ? all_zero
                                                  : report.moist_energy_closure_residual_j ==
                                                        report.moist_committed_energy_change_j - reservoir_sum);
        };
        auto& cumulative                    = runtime.diffusion.cumulative_moist_energy_work;
        cumulative.requested_linear_work_j  = runtime.diffusion.cumulative_requested_momentum_work_j;
        cumulative.finite_step_correction_j = runtime.diffusion.cumulative_finite_step_momentum_correction_j;
        cumulative.storage_rounding_j       = runtime.diffusion.cumulative_momentum_storage_rounding_j;
        cumulative.committed_work_j         = runtime.diffusion.cumulative_committed_momentum_work_j;
        if (!decode_diffusion_moist_energy(runtime.diffusion.last_momentum_work) ||
            !decode_diffusion_moist_energy(cumulative) || !in.unsigned_value(runtime.diffusion.moist_energy_step_count))
            return false;
    }
    if (section_version >= 25)
    {
        const auto decode_eos_pressure_energy = [&in](tmv::dynamics_eos::PressureProjectionEnergyReportV1& report)
        {
            if (!in.floating(report.dry_internal_change_j) || !in.floating(report.water_sensible_change_j) ||
                !in.floating(report.phase_reference_change_j) || !in.floating(report.carrier_kinetic_change_j) ||
                !in.floating(report.gravitational_change_j) || !in.floating(report.subgrid_tke_change_j) ||
                !in.floating(report.committed_energy_change_j) || !in.floating(report.closure_residual_j) ||
                !in.boolean(report.complete))
            {
                return false;
            }
            const double values[] = {
                report.dry_internal_change_j,
                report.water_sensible_change_j,
                report.phase_reference_change_j,
                report.carrier_kinetic_change_j,
                report.gravitational_change_j,
                report.subgrid_tke_change_j,
                report.committed_energy_change_j,
                report.closure_residual_j,
            };
            const double reservoir_sum = report.dry_internal_change_j + report.water_sensible_change_j +
                                         report.phase_reference_change_j + report.carrier_kinetic_change_j +
                                         report.gravitational_change_j + report.subgrid_tke_change_j;
            const bool   all_zero =
                std::all_of(std::begin(values), std::end(values), [](double value) { return value == 0.0; });
            return std::all_of(
                       std::begin(values), std::end(values), [](double value) { return std::isfinite(value); }) &&
                   (!report.complete ? all_zero
                                     : report.closure_residual_j == report.committed_energy_change_j - reservoir_sum);
        };
        if (!decode_eos_pressure_energy(runtime.time_stepping.last_eos_pressure_energy) ||
            !decode_eos_pressure_energy(runtime.time_stepping.cumulative_eos_pressure_energy) ||
            !in.unsigned_value(runtime.time_stepping.eos_pressure_energy_step_count) ||
            ((runtime.time_stepping.eos_pressure_energy_step_count > 0) !=
             runtime.time_stepping.cumulative_eos_pressure_energy.complete))
        {
            return false;
        }
    }
    if (section_version >= 26)
    {
        const auto decode_sponge_pass = [&in](tmv::SpongePassEnergyReportV1& pass)
        {
            if (!in.floating(pass.momentum_requested_linear_work_j) ||
                !in.floating(pass.momentum_finite_step_correction_j) ||
                !in.floating(pass.momentum_storage_rounding_j) || !in.floating(pass.dry_internal_change_j) ||
                !in.floating(pass.water_sensible_change_j) || !in.floating(pass.phase_reference_change_j) ||
                !in.floating(pass.carrier_kinetic_change_j) || !in.floating(pass.gravitational_change_j) ||
                !in.floating(pass.subgrid_tke_change_j) || !in.floating(pass.committed_energy_change_j) ||
                !in.floating(pass.closure_residual_j) || !in.boolean(pass.applied) || !in.boolean(pass.complete))
                return false;
            const double values[] = {
                pass.momentum_requested_linear_work_j,
                pass.momentum_finite_step_correction_j,
                pass.momentum_storage_rounding_j,
                pass.dry_internal_change_j,
                pass.water_sensible_change_j,
                pass.phase_reference_change_j,
                pass.carrier_kinetic_change_j,
                pass.gravitational_change_j,
                pass.subgrid_tke_change_j,
                pass.committed_energy_change_j,
                pass.closure_residual_j,
            };
            const double reservoir_sum = pass.dry_internal_change_j + pass.water_sensible_change_j +
                                         pass.phase_reference_change_j + pass.carrier_kinetic_change_j +
                                         pass.gravitational_change_j + pass.subgrid_tke_change_j;
            const bool   all_zero =
                std::all_of(std::begin(values), std::end(values), [](double value) { return value == 0.0; });
            return std::all_of(
                       std::begin(values), std::end(values), [](double value) { return std::isfinite(value); }) &&
                   (!pass.complete ? all_zero && !pass.applied
                                   : pass.momentum_storage_rounding_j == pass.carrier_kinetic_change_j -
                                                                             pass.momentum_requested_linear_work_j -
                                                                             pass.momentum_finite_step_correction_j &&
                                         pass.closure_residual_j == pass.committed_energy_change_j - reservoir_sum);
        };
        const auto decode_sponge_energy = [&in, &decode_sponge_pass](tmv::SpongeEnergyReportV1& report)
        {
            return decode_sponge_pass(report.upper) && decode_sponge_pass(report.w_only) &&
                   decode_sponge_pass(report.lateral) && in.boolean(report.complete) &&
                   report.complete == (report.upper.complete && report.w_only.complete && report.lateral.complete);
        };
        if (!decode_sponge_energy(runtime.time_stepping.last_sponge_energy) ||
            !decode_sponge_energy(runtime.time_stepping.cumulative_sponge_energy) ||
            !in.unsigned_value(runtime.time_stepping.sponge_energy_step_count) ||
            ((runtime.time_stepping.sponge_energy_step_count > 0) !=
             runtime.time_stepping.cumulative_sponge_energy.complete))
        {
            return false;
        }
    }
    if (section_version >= 27)
    {
        const auto decode_final_boundary_energy = [&in](tmv::FinalBoundaryEnergyReportV1& report)
        {
            if (!in.floating(report.constraint_committed_work_j) || !in.floating(report.dry_internal_change_j) ||
                !in.floating(report.water_sensible_change_j) || !in.floating(report.phase_reference_change_j) ||
                !in.floating(report.carrier_kinetic_change_j) || !in.floating(report.gravitational_change_j) ||
                !in.floating(report.subgrid_tke_change_j) || !in.floating(report.committed_energy_change_j) ||
                !in.floating(report.closure_residual_j) || !in.boolean(report.applied) || !in.boolean(report.complete))
            {
                return false;
            }
            const double values[] = {
                report.constraint_committed_work_j,
                report.dry_internal_change_j,
                report.water_sensible_change_j,
                report.phase_reference_change_j,
                report.carrier_kinetic_change_j,
                report.gravitational_change_j,
                report.subgrid_tke_change_j,
                report.committed_energy_change_j,
                report.closure_residual_j,
            };
            const double reservoir_sum = report.dry_internal_change_j + report.water_sensible_change_j +
                                         report.phase_reference_change_j + report.carrier_kinetic_change_j +
                                         report.gravitational_change_j + report.subgrid_tke_change_j;
            const bool   all_zero =
                std::all_of(std::begin(values), std::end(values), [](double value) { return value == 0.0; });
            return std::all_of(
                       std::begin(values), std::end(values), [](double value) { return std::isfinite(value); }) &&
                   (!report.complete
                        ? all_zero && !report.applied
                        : report.applied && report.constraint_committed_work_j == report.carrier_kinetic_change_j &&
                              report.closure_residual_j == report.committed_energy_change_j - reservoir_sum);
        };
        if (!decode_final_boundary_energy(runtime.time_stepping.last_final_boundary_energy) ||
            !decode_final_boundary_energy(runtime.time_stepping.cumulative_final_boundary_energy) ||
            !in.unsigned_value(runtime.time_stepping.final_boundary_energy_step_count) ||
            ((runtime.time_stepping.final_boundary_energy_step_count > 0) !=
             runtime.time_stepping.cumulative_final_boundary_energy.complete))
        {
            return false;
        }
    }
    if (section_version >= 28)
    {
        const auto decode_completed_step_energy = [&in, section_version](tmv::CompletedStepEnergyReportV1& report)
        {
            const bool prefix =
                in.floating(report.acoustic_committed_change_j) &&
                in.floating(report.slow_dynamics_committed_change_j) &&
                in.floating(report.scalar_transport_committed_change_j) &&
                in.floating(report.thermodynamic_sources_committed_change_j) &&
                in.floating(report.sgs_committed_change_j) && in.floating(report.diffusion_committed_change_j) &&
                in.floating(report.eos_projection_committed_change_j) &&
                in.floating(report.sponge_committed_change_j) && in.floating(report.final_boundary_committed_change_j);
            if (!prefix)
                return false;
            report.unsplit_rk_committed_change_j = 0.0;
            report.coordinate_remap_committed_change_j = 0.0;
            return in.floating(report.committed_owner_sum_j) && in.floating(report.dry_internal_change_j) &&
                   in.floating(report.water_sensible_change_j) && in.floating(report.phase_reference_change_j) &&
                   in.floating(report.carrier_kinetic_change_j) && in.floating(report.gravitational_change_j) &&
                   in.floating(report.subgrid_tke_change_j) && in.floating(report.closure_residual_j) &&
                   in.boolean(report.complete) &&
                   (section_version >= 50 || tmv::structurally_valid_completed_step_energy_report(report));
        };
        if (!decode_completed_step_energy(runtime.time_stepping.last_completed_step_energy) ||
            !decode_completed_step_energy(runtime.time_stepping.cumulative_completed_step_energy) ||
            !in.unsigned_value(runtime.time_stepping.completed_step_energy_count) ||
            ((runtime.time_stepping.completed_step_energy_count > 0) !=
             runtime.time_stepping.cumulative_completed_step_energy.complete))
        {
            return false;
        }
    }
    if (section_version >= 30)
    {
        auto&         storm          = runtime.storm_tracking;
        std::uint64_t sample_count   = 0;
        std::uint64_t missed_samples = 0;
        if (!decode_enum(in, storm.status) || !decode_enum(in, storm.reason) || !in.unsigned_value(sample_count) ||
            sample_count > tmv::diagnostics::storm_continuity_history_capacity)
        {
            return false;
        }
        storm.sample_count = static_cast<std::size_t>(sample_count);
        for (std::size_t index = 0; index < storm.sample_count; ++index)
        {
            if (!in.floating(storm.samples[index].time_s) || !in.floating(storm.samples[index].x_m) ||
                !in.floating(storm.samples[index].y_m) || !in.floating(storm.samples[index].confidence))
            {
                return false;
            }
        }
        if (!in.unsigned_value(missed_samples) ||
            missed_samples > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
            !in.unsigned_value(storm.branch_id) || !in.boolean(storm.motion_available) ||
            !in.floating(storm.velocity_x_ms) || !in.floating(storm.velocity_y_ms) || !in.floating(storm.speed_ms) ||
            !in.floating(storm.velocity_uncertainty_ms) || !in.floating(storm.continuity_confidence) ||
            !in.floating(storm.innovation_distance_m) || !in.floating(storm.association_gate_radius_m))
        {
            return false;
        }
        storm.missed_samples = static_cast<std::size_t>(missed_samples);
        if (section_version >= 31)
        {
            std::uint64_t pending_topology_samples = 0;
            std::uint64_t selected_candidate_rank  = 0;
            std::uint64_t selected_component_label = 0;
            if (!decode_enum(in, storm.topology) || !decode_enum(in, storm.pending_topology) ||
                !decode_enum(in, storm.topology_event) || !in.unsigned_value(pending_topology_samples) ||
                pending_topology_samples > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
                !in.unsigned_value(storm.topology_event_id) || !in.boolean(storm.selected_candidate_available) ||
                !in.unsigned_value(selected_candidate_rank) ||
                selected_candidate_rank > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
                !in.unsigned_value(selected_component_label) ||
                selected_component_label > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
                !decode_enum(in, storm.selected_motion_class) ||
                !in.floating(storm.selected_bunkers_right_mismatch_ms) ||
                !in.floating(storm.selected_bunkers_left_mismatch_ms))
            {
                return false;
            }
            storm.pending_topology_samples = static_cast<std::size_t>(pending_topology_samples);
            storm.selected_candidate_rank  = static_cast<std::size_t>(selected_candidate_rank);
            storm.selected_component_label = static_cast<std::size_t>(selected_component_label);
        }
        if (section_version >= 32)
        {
            if (!in.boolean(storm.association_envelope_available) || !in.floating(storm.association_envelope_west_m) ||
                !in.floating(storm.association_envelope_east_m) || !in.floating(storm.association_envelope_south_m) ||
                !in.floating(storm.association_envelope_north_m) || !in.boolean(storm.topology_reorganization) ||
                !in.floating(storm.reorganization_overlap_fraction))
            {
                return false;
            }
        }
        if (section_version >= 33)
        {
            if (!in.boolean(runtime.storm_selected_partition_available))
                return false;
            if (runtime.storm_selected_partition_available)
            {
                std::uint64_t nx = 0, ny = 0, size = 0;
                if (!in.unsigned_value(nx) || !in.unsigned_value(ny) || !in.unsigned_value(size) ||
                    nx > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
                    ny > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
                    size > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) || nx == 0 || ny == 0 ||
                    nx > std::numeric_limits<std::uint64_t>::max() / ny || size != nx * ny ||
                    !in.raw(static_cast<std::size_t>(size), runtime.storm_selected_partition))
                    return false;
                runtime.storm_selected_partition_nx = static_cast<std::size_t>(nx);
                runtime.storm_selected_partition_ny = static_cast<std::size_t>(ny);
            }
        }
        if (section_version >= 34)
        {
            auto&         movement             = runtime.storm_window_movement;
            std::uint64_t confirmation_samples = 0;
            std::uint64_t cumulative_i_bits    = 0;
            std::uint64_t cumulative_j_bits    = 0;
            if (!in.boolean(movement.armed) || !in.integer(movement.pending_i_direction) ||
                !in.integer(movement.pending_j_direction) || !in.unsigned_value(confirmation_samples) ||
                confirmation_samples > static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
                !in.boolean(movement.sample_time_available) || !in.floating(movement.last_sample_time_s) ||
                !in.boolean(movement.committed_move_available) || !in.floating(movement.last_committed_time_s) ||
                !in.unsigned_value(cumulative_i_bits) || !in.unsigned_value(cumulative_j_bits))
            {
                return false;
            }
            movement.consecutive_compatible_samples = static_cast<std::size_t>(confirmation_samples);
            const auto decode_signed_bits           = [](std::uint64_t bits)
            {
                if (bits <= static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
                    return static_cast<std::int64_t>(bits);
                return static_cast<std::int64_t>(-1) - static_cast<std::int64_t>(~bits);
            };
            movement.cumulative_i_shift_cells = decode_signed_bits(cumulative_i_bits);
            movement.cumulative_j_shift_cells = decode_signed_bits(cumulative_j_bits);
        }
        if (!valid_storm_selected_partition(runtime))
        {
            error = "invalid checkpoint exact storm partition state";
            return false;
        }
        if (!tmv::diagnostics::valid_storm_continuity_state(storm))
        {
            error = "invalid checkpoint storm continuity state";
            return false;
        }
        if (!tmv::domain::valid_storm_window_movement_state(runtime.storm_window_movement))
        {
            error = "invalid checkpoint storm-window movement authority state";
            return false;
        }
        const auto& tracking = storage.configuration.storm_tracking;
        if (storm.motion_available &&
            (storm.sample_count < static_cast<std::size_t>(tracking.minimum_continuity_samples) ||
             storm.speed_ms > tracking.maximum_motion_speed_ms))
        {
            error = "checkpoint storm continuity state conflicts with its configured thresholds";
            return false;
        }
    }
    if (section_version >= 35)
    {
        if (!decode_domain_recycling_provenance(in, runtime.domain_recycling_provenance, section_version >= 36))
        {
            error = "invalid checkpoint D5 transaction provenance";
            return false;
        }
    }
    if (section_version >= 37)
    {
        if (!decode_field(in, runtime.chaos.base_dtke_dt_pbl) ||
            !decode_float_vector(in, runtime.chaos.base_sfc_ustar) ||
            !decode_float_vector(in, runtime.chaos.base_sfc_tau_u) ||
            !decode_float_vector(in, runtime.chaos.base_sfc_tau_v) ||
            !decode_float_vector(in, runtime.chaos.base_sfc_sensible_heat_flux) ||
            !decode_float_vector(in, runtime.chaos.base_sfc_latent_heat_flux) ||
            !decode_float_vector(in, runtime.chaos.base_sfc_moisture_flux))
            return false;
    }
    if (section_version >= 38)
    {
        if (!in.floating(runtime.turbulence.last_open_boundary_work_j) ||
            !in.floating(runtime.turbulence.last_irreversible_dissipation_j) ||
            !in.floating(runtime.turbulence.last_stress_closure_residual_j) ||
            !in.floating(runtime.turbulence.cumulative_open_boundary_work_j) ||
            !in.floating(runtime.turbulence.cumulative_irreversible_dissipation_j) ||
            !in.floating(runtime.turbulence.cumulative_stress_closure_residual_j) ||
            !in.unsigned_value(runtime.turbulence.open_boundary_stability_step_count))
            return false;
    }
    if (section_version >= 39)
    {
        std::uint64_t cosine_nodes    = 0;
        std::uint64_t azimuth_nodes   = 0;
        std::uint64_t maximum_bounces = 0;
        auto&         radiation       = runtime.radiation;
        if (!in.unsigned_value(cosine_nodes) || !in.unsigned_value(azimuth_nodes) ||
            !in.unsigned_value(maximum_bounces) || cosine_nodes > std::numeric_limits<std::size_t>::max() ||
            azimuth_nodes > std::numeric_limits<std::size_t>::max() ||
            maximum_bounces > std::numeric_limits<std::size_t>::max() ||
            !in.floating(radiation.config.terrain_sw_relative_residual_tolerance) ||
            !in.floating(radiation.config.terrain_sw_absolute_residual_tolerance_w) ||
            !in.floating(radiation.held_shortwave_incoming_power_w) ||
            !in.floating(radiation.held_shortwave_atmospheric_absorbed_power_w) ||
            !in.floating(radiation.held_surface_absorbed_power_w) || !in.floating(radiation.held_top_escape_power_w) ||
            !in.floating(radiation.held_lateral_escape_power_w) ||
            !in.floating(radiation.held_unresolved_reflected_power_w) ||
            !in.boolean(radiation.held_reflection_converged))
            return false;
        radiation.config.terrain_sw_cosine_nodes    = static_cast<std::size_t>(cosine_nodes);
        radiation.config.terrain_sw_azimuth_nodes   = static_cast<std::size_t>(azimuth_nodes);
        radiation.config.terrain_sw_maximum_bounces = static_cast<std::size_t>(maximum_bounces);
        if (!valid_terrain_shortwave_runtime_contract(radiation))
            return false;
    }
    if (section_version >= 40)
    {
        const auto decode_tracer_roundoff = [&in](PositiveDefiniteScalarRoundoffReport& report)
        {
            std::uint32_t schema                   = 0;
            std::uint32_t policy                   = 0;
            std::uint64_t negative_subnormal_count = 0;
            std::uint64_t negative_zero_count      = 0;
            std::uint64_t closure_count            = 0;
            if (!in.unsigned_value(schema) || !in.unsigned_value(policy) ||
                schema != PositiveDefiniteScalarRoundoffReport::report_schema_version ||
                policy != PositiveDefiniteScalarRoundoffReport::classification_policy_version ||
                !in.floating(report.physical_inventory_correction_kg) ||
                !in.floating(report.thermodynamic_energy_correction_j) ||
                !in.floating(report.most_negative_repaired_value) || !in.unsigned_value(negative_subnormal_count) ||
                !in.unsigned_value(negative_zero_count) || !in.unsigned_value(closure_count) ||
                !in.boolean(report.complete) || negative_subnormal_count > std::numeric_limits<std::size_t>::max() ||
                negative_zero_count > std::numeric_limits<std::size_t>::max() ||
                closure_count > std::numeric_limits<std::size_t>::max())
            {
                return false;
            }
            report.negative_subnormal_count       = static_cast<std::size_t>(negative_subnormal_count);
            report.negative_zero_count            = static_cast<std::size_t>(negative_zero_count);
            report.closure_canonicalization_count = static_cast<std::size_t>(closure_count);
            return valid_tracer_roundoff_report(report);
        };
        auto&      last       = runtime.advection.tracer_roundoff_report;
        auto&      cumulative = runtime.advection.cumulative_tracer_roundoff_report;
        const bool decoded    = decode_tracer_roundoff(last) && decode_tracer_roundoff(cumulative) &&
                                in.unsigned_value(runtime.advection.tracer_roundoff_report_step_count) &&
                                in.boolean(runtime.advection.tracer_roundoff_cumulative_history_known);
        if (!decoded || !valid_tracer_roundoff_checkpoint_runtime(runtime.advection))
        {
            error = "invalid passive-tracer roundoff checkpoint ledger";
            return false;
        }
    }
    else
    {
        runtime.advection.tracer_roundoff_report                   = {};
        runtime.advection.cumulative_tracer_roundoff_report        = {};
        runtime.advection.tracer_roundoff_report_step_count        = 0;
        runtime.advection.tracer_roundoff_cumulative_history_known = false;
    }
    if (section_version >= 41)
    {
        if (!decode_rrtmg_config(in, runtime.radiation.config))
        {
            error = "invalid RRTMG checkpoint configuration";
            return false;
        }
    }
    else if (runtime.radiation.config.scheme_id == "rrtmg")
    {
        error = "legacy checkpoint does not contain the required RRTMG composition";
        return false;
    }
    if (section_version >= 42 &&
        (!in.string(runtime.diffusion.config.coefficient_source) ||
         !in.floating(runtime.diffusion.config.K_v_surface) || !in.floating(runtime.diffusion.config.K_v_top)))
    {
        error = "invalid variable-diffusion checkpoint configuration";
        return false;
    }
    if (section_version >= 42)
    {
        const auto& config = runtime.diffusion.config;
        const bool  linear = config.coefficient_source == "linear_height";
        if ((config.coefficient_source != "constant" && !linear) || config.use_variable_K != linear ||
            !std::isfinite(config.K_v_surface) || config.K_v_surface < 0.0 || !std::isfinite(config.K_v_top) ||
            config.K_v_top < 0.0 || (linear && (config.scheme_id != "explicit" || config.K_v != 0.0)))
        {
            error = "invalid variable-diffusion checkpoint configuration";
            return false;
        }
    }
    if (section_version >= 43 &&
        (!decode_radar_product_selection(in, runtime.radar.product_selection, section_version >= 44) ||
         !valid_radar_product_selection(runtime.radar.product_selection)))
    {
        error = "invalid radar product checkpoint configuration";
        return false;
    }
    if (section_version >= 45)
    {
        auto&         radiation      = runtime.radiation;
        std::uint64_t subcycle_count = 0;
        std::uint64_t columns_x      = 0;
        std::uint64_t columns_y      = 0;
        std::uint64_t levels         = 0;
        auto&         fluxes         = radiation.held_flux_diagnostics;
        if (!in.unsigned_value(subcycle_count) || subcycle_count > std::numeric_limits<std::size_t>::max() ||
            !in.string(radiation.config.simple_grey_optical_model) ||
            !in.floating(radiation.config.simple_grey_lw_vapor_absorption_m2kg) ||
            !in.floating(radiation.config.simple_grey_lw_liquid_absorption_m2kg) ||
            !in.floating(radiation.config.simple_grey_lw_ice_absorption_m2kg) ||
            !in.floating(radiation.config.simple_grey_sw_vapor_absorption_m2kg) ||
            !in.floating(radiation.config.simple_grey_sw_liquid_absorption_m2kg) ||
            !in.floating(radiation.config.simple_grey_sw_ice_absorption_m2kg) || !in.unsigned_value(columns_x) ||
            !in.unsigned_value(columns_y) || !in.unsigned_value(levels) ||
            columns_x > std::numeric_limits<std::size_t>::max() ||
            columns_y > std::numeric_limits<std::size_t>::max() || levels > std::numeric_limits<std::size_t>::max() ||
            !decode_double_vector(in, fluxes.upward_longwave_wm2) ||
            !decode_double_vector(in, fluxes.downward_longwave_wm2) ||
            !decode_double_vector(in, fluxes.upward_shortwave_wm2) ||
            !decode_double_vector(in, fluxes.downward_shortwave_wm2))
        {
            error = "invalid simple-grey checkpoint state";
            return false;
        }
        radiation.config.subcycle_count = static_cast<std::size_t>(subcycle_count);
        fluxes.columns_x                = static_cast<std::size_t>(columns_x);
        fluxes.columns_y                = static_cast<std::size_t>(columns_y);
        fluxes.levels                   = static_cast<std::size_t>(levels);
        if (!valid_simple_grey_checkpoint_config(radiation.config) || !valid_radiation_flux_diagnostics(fluxes))
        {
            error = "invalid simple-grey checkpoint state";
            return false;
        }
    }
    else if (runtime.radiation.config.scheme_id == "simple_grey")
    {
        // Pre-v45 simple-grey checkpoints had no versioned cloud-optics or
        // subcycle contract; their clear_sky_only=false flag still executed
        // the pressure-only branch. Restore that historical behavior rather
        // than silently applying the new water/cloud-path model.
        runtime.radiation.config.simple_grey_optical_model = "pressure_clear_sky_v1";
        runtime.radiation.config.clear_sky_only            = true;
        runtime.radiation.config.use_subcycling            = false;
        runtime.radiation.config.subcycle_count            = 1;
    }
    if (section_version >= 46)
    {
        if (!in.floating(runtime.turbulence.config.user_filter_width_m))
        {
            error = "invalid user filter-width checkpoint configuration";
            return false;
        }
    }
    else
    {
        runtime.turbulence.config.user_filter_width_m = 0.0;
    }
    if (!std::isfinite(runtime.turbulence.config.user_filter_width_m) ||
        (runtime.turbulence.config.filter_width == "user" ? runtime.turbulence.config.user_filter_width_m <= 0.0
                                                          : runtime.turbulence.config.user_filter_width_m != 0.0))
    {
        error = "invalid user filter-width checkpoint configuration";
        return false;
    }
    auto& cloud_thermo = runtime.microphysics.last_thermodynamic_repair_report;
    if (section_version >= 47)
    {
        if (!in.floating(cloud_thermo.boundary_layer_cloud_liquid_tendency_kg) ||
            !in.floating(cloud_thermo.boundary_layer_cloud_ice_tendency_kg))
        {
            error = "invalid boundary-layer cloud checkpoint ledger";
            return false;
        }
    }
    else
    {
        cloud_thermo.boundary_layer_cloud_liquid_tendency_kg = 0.0;
        cloud_thermo.boundary_layer_cloud_ice_tendency_kg    = 0.0;
    }
    if (!std::isfinite(cloud_thermo.boundary_layer_cloud_liquid_tendency_kg) ||
        !std::isfinite(cloud_thermo.boundary_layer_cloud_ice_tendency_kg))
    {
        error = "invalid boundary-layer cloud checkpoint ledger";
        return false;
    }
    if (section_version >= 49)
    {
        if (!decode_terrain_placement(in, runtime.terrain.config, runtime.terrain.earth_fixed_placement) ||
            !valid_terrain_placement_checkpoint(storage))
        {
            error = "invalid Earth-fixed terrain checkpoint identity";
            return false;
        }
    }
    if (section_version >= 50)
    {
        if (!in.floating(runtime.time_stepping.last_completed_step_energy.unsplit_rk_committed_change_j) ||
            !in.floating(runtime.time_stepping.cumulative_completed_step_energy.unsplit_rk_committed_change_j) ||
            (section_version < 54 &&
             (!tmv::structurally_valid_completed_step_energy_report(runtime.time_stepping.last_completed_step_energy) ||
              !tmv::structurally_valid_completed_step_energy_report(
                  runtime.time_stepping.cumulative_completed_step_energy))))
        {
            error = "invalid unsplit-RK completed-step checkpoint owner";
            return false;
        }
    }
    if (section_version < 49)
    {
        runtime.terrain.config.horizontal_placement_id  = "grid_attached";
        runtime.terrain.config.feature_center_earth_x_m = 0.0;
        runtime.terrain.config.feature_center_earth_y_m = 0.0;
        runtime.terrain.earth_fixed_placement           = {};
    }
    if (section_version >= 51 &&
        (!in.string(runtime.compute.vertical_flux_template_id) ||
         (runtime.compute.vertical_flux_template_id != "tvd_vertical_flux_v1" &&
          runtime.compute.vertical_flux_template_id != "tvd_vertical_flux_v2")))
    {
        error = "invalid checkpoint vertical-flux execution selection";
        return false;
    }
    if (section_version >= 52 && !in.boolean(runtime.chaos.config.native_generation_required))
        return false;
    runtime.terrain.earth_fixed_placement.sampling = EarthFixedTerrainSampling::Point;
    if (section_version >= 53)
    {
        std::uint8_t sampling = 0;
        if (!in.unsigned_value(sampling) || sampling > 1)
        {
            error = "invalid Earth-fixed geometry sampling";
            return false;
        }
        runtime.terrain.earth_fixed_placement.sampling = static_cast<EarthFixedTerrainSampling>(sampling);
        if (!valid_terrain_placement_checkpoint(storage))
        {
            error = "invalid Earth-fixed geometry sampling ownership";
            return false;
        }
    }
    if (section_version >= 54)
    {
        if (!in.floating(runtime.time_stepping.last_completed_step_energy.coordinate_remap_committed_change_j)
            || !in.floating(runtime.time_stepping.cumulative_completed_step_energy.coordinate_remap_committed_change_j)
            || !tmv::structurally_valid_completed_step_energy_report(runtime.time_stepping.last_completed_step_energy)
            || !tmv::structurally_valid_completed_step_energy_report(runtime.time_stepping.cumulative_completed_step_energy))
        {
            error = "invalid coordinate-remapping completed-step owner";
            return false;
        }
    }
    if (section_version >= 55)
    {
        auto& value = runtime.domain_recycling_provenance;
        if (!in.boolean(value.private_inventory_complete))
            return false;
        if (value.private_inventory_complete)
        {
            auto& retiring = value.retiring_private_inventory;
            auto& admitted = value.admitted_private_inventory;
            if (!in.unsigned_value(retiring.count) || retiring.count > retiring.entries.size())
                return false;
            admitted.count = retiring.count;
            for (std::size_t f = 0; f < retiring.count; ++f)
            {
                if (!decode_provenance_text(in, retiring.entries[f].name) ||
                    !in.floating(retiring.entries[f].inventory) || !in.floating(admitted.entries[f].inventory))
                    return false;
                admitted.entries[f].name = retiring.entries[f].name;
            }
        }
        if (!tmv::domain::valid_domain_recycling_transaction_provenance(value) ||
            !private_recycling_owners_match(runtime.microphysics, value))
        {
            error = "invalid private recycling inventory provenance";
            return false;
        }
    }
    if (runtime.chaos.scheme)
        runtime.chaos.scheme->configure_compute_backend(nullptr, runtime.chaos.config.native_generation_required);
    if (section_version < 48 && runtime.turbulence.config.scheme_id == "tke" &&
        runtime.turbulence.config.stability_correction == "none")
    {
        runtime.turbulence.config.stability_correction = "tke_based";
    }
    // Initialize schemes only after every versioned configuration tail has
    // been decoded and validated.
    if (radiation_present)
    {
        runtime.radiation.scheme = create_radiation_scheme(runtime.radiation.config.scheme_id);
        runtime.radiation.scheme->initialize(runtime.radiation.config);
    }
    if (!known_turbulence_stability_model(runtime.turbulence.config.stability_correction))
    {
        error = "unknown turbulence stability model in checkpoint";
        return false;
    }
    if (turbulence_present)
    {
        runtime.turbulence.scheme = create_turbulence_scheme(runtime.turbulence.config.scheme_id);
        runtime.turbulence.scheme->initialize(runtime.turbulence.config);
    }
    if (time_present)
    {
        runtime.time_stepping.scheme = create_time_stepping_scheme(runtime.time_stepping.config.scheme_id);
        runtime.time_stepping.scheme->initialize(runtime.time_stepping.config, nullptr);
    }
    return true;
}

void encode_nest_relationship(Encoder& out, const tmv::domain::OneWayNestRelationshipV1& value)
{
    out.unsigned_value(value.version);
    out.integer(value.parent_cells_x);
    out.integer(value.parent_cells_y);
    out.integer(value.levels_z);
    out.integer(value.parent_ratio_x);
    out.integer(value.parent_ratio_y);
    out.integer(value.parent_time_step_ratio);
    out.integer(value.child_origin_parent_cell_i);
    out.integer(value.child_origin_parent_cell_j);
    out.integer(value.child_cells_x);
    out.integer(value.child_cells_y);
    out.integer(value.child_parent_span_x);
    out.integer(value.child_parent_span_y);
    out.integer(value.relaxation_width_child_cells);
    out.integer(value.boundary_stencil_parent_cells);
    out.floating(value.parent_dx_m);
    out.floating(value.parent_dy_m);
    out.floating(value.parent_dt_s);
    out.floating(value.child_dx_m);
    out.floating(value.child_dy_m);
    out.floating(value.child_dt_s);
}

bool decode_nest_relationship(Decoder& in, tmv::domain::OneWayNestRelationshipV1& value)
{
    return in.unsigned_value(value.version) && in.integer(value.parent_cells_x) && in.integer(value.parent_cells_y) &&
           in.integer(value.levels_z) && in.integer(value.parent_ratio_x) && in.integer(value.parent_ratio_y) &&
           in.integer(value.parent_time_step_ratio) && in.integer(value.child_origin_parent_cell_i) &&
           in.integer(value.child_origin_parent_cell_j) && in.integer(value.child_cells_x) &&
           in.integer(value.child_cells_y) && in.integer(value.child_parent_span_x) &&
           in.integer(value.child_parent_span_y) && in.integer(value.relaxation_width_child_cells) &&
           in.integer(value.boundary_stencil_parent_cells) && in.floating(value.parent_dx_m) &&
           in.floating(value.parent_dy_m) && in.floating(value.parent_dt_s) && in.floating(value.child_dx_m) &&
           in.floating(value.child_dy_m) && in.floating(value.child_dt_s);
}

bool same_nest_relationship(const tmv::domain::OneWayNestRelationshipV1& left,
                            const tmv::domain::OneWayNestRelationshipV1& right)
{
    Encoder left_bytes;
    Encoder right_bytes;
    encode_nest_relationship(left_bytes, left);
    encode_nest_relationship(right_bytes, right);
    return left_bytes.bytes == right_bytes.bytes;
}

bool same_terrain_config(const TerrainConfig& left, const TerrainConfig& right)
{
    Encoder left_bytes;
    Encoder right_bytes;
    encode_terrain_config(left_bytes, left);
    encode_terrain_placement_config(left_bytes, left);
    encode_terrain_config(right_bytes, right);
    encode_terrain_placement_config(right_bytes, right);
    return left_bytes.good() && right_bytes.good() && left_bytes.bytes == right_bytes.bytes;
}

bool exact_terrain_field(const TerrainField3D& left, const TerrainField3D& right) noexcept
{
    return left.nr == right.nr && left.nth == right.nth && left.nz == right.nz && left.values == right.values;
}

bool valid_terrain_field(const TerrainField3D& field, int nr, int nth, int nz) noexcept
{
    if (field.nr != nr || field.nth != nth || field.nz != nz)
        return false;
    return std::all_of(field.values.begin(), field.values.end(), [](double value) { return std::isfinite(value); });
}

bool empty_earth_fixed_geometry(const tmv::EarthFixedTerrainGeometry& geometry) noexcept
{
    const auto  empty_field = [](const TerrainField3D& field) { return field.values.empty(); };
    const auto& metrics     = geometry.metrics;
    return geometry.topography.h.empty() && geometry.topography.hx.empty() && geometry.topography.hy.empty() &&
           empty_field(metrics.z) && empty_field(metrics.J) && empty_field(metrics.mx) && empty_field(metrics.my) &&
           empty_field(metrics.zeta) && empty_field(metrics.z_face) && empty_field(metrics.Jx_face) &&
           empty_field(metrics.Jy_face) && empty_field(metrics.Gx_face) && empty_field(metrics.Gy_face);
}

bool valid_active_terrain_checkpoint(const tmv::domain::NestedActiveTerrainStateV1& active,
                                     const tmv::domain::OneWayNestRelationshipV1&   relationship,
                                     const TerrainConfig&                           runtime_config,
                                     tmv::domain::HorizontalPoint                   expected_parent_origin_earth_m,
                                     double                                         parent_time_s,
                                     double                                         child_time_s,
                                     std::string&                                   error)
{
    if (!active.ready)
    {
        const bool empty_fields =
            active.child_state.scalar.values.empty() && active.child_state.x_flux.values.empty() &&
            active.child_state.y_flux.values.empty() && active.child_state.zeta_flux.values.empty() &&
            active.boundary.relaxation_weight.values.empty() && active.boundary.ground_u_ms.values.empty() &&
            active.boundary.ground_v_ms.values.empty() && active.boundary.ground_w_ms.values.empty();
        const auto& ledger = active.ledger;
        const bool  empty_ledger =
            ledger.parent_time_s == 0.0 && ledger.child_time_s == 0.0 && ledger.parent_scalar_inventory == 0.0 &&
            ledger.child_scalar_inventory == 0.0 && ledger.scalar_inventory_residual == 0.0 &&
            ledger.normalized_scalar_inventory_residual == 0.0 && ledger.maximum_x_flux_residual == 0.0 &&
            ledger.maximum_normalized_x_flux_residual == 0.0 && ledger.maximum_y_flux_residual == 0.0 &&
            ledger.maximum_normalized_y_flux_residual == 0.0 && ledger.maximum_zeta_flux_residual == 0.0 &&
            ledger.maximum_normalized_zeta_flux_residual == 0.0 && ledger.maximum_ground_tangency_residual_ms == 0.0 &&
            ledger.relaxation_owned_values == 0;
        if (active.generation != 0 || active.publication_count != 0 || active.geometry.generation != 0 ||
            active.parent_time_s != 0.0 || active.child_time_s != 0.0 || !empty_fields || !empty_ledger ||
            !empty_earth_fixed_geometry(active.geometry.parent) ||
            !empty_earth_fixed_geometry(active.geometry.child_storage))
        {
            error = "inactive active-terrain checkpoint contains durable state";
            return false;
        }
        return true;
    }

    const auto& geometry = active.geometry;
    const auto& ledger   = active.ledger;
    if (!same_nest_relationship(geometry.relationship, relationship) ||
        !same_terrain_config(geometry.config, runtime_config) ||
        geometry.parent_origin_earth_m.x_m != expected_parent_origin_earth_m.x_m ||
        geometry.parent_origin_earth_m.y_m != expected_parent_origin_earth_m.y_m)
    {
        error = "active-terrain checkpoint identity does not match live nesting and terrain configuration";
        return false;
    }
    if (!std::isfinite(active.parent_time_s) || active.parent_time_s < 0.0 ||
        active.parent_time_s != active.child_time_s || active.parent_time_s != parent_time_s ||
        active.child_time_s != child_time_s || ledger.parent_time_s != active.parent_time_s ||
        ledger.child_time_s != active.child_time_s)
    {
        error = "active-terrain checkpoint clocks are not synchronized";
        return false;
    }
    if (active.generation == 0 || active.publication_count == 0 || active.generation != active.publication_count ||
        geometry.generation != active.generation)
    {
        error = "active-terrain checkpoint publication identity is inconsistent";
        return false;
    }

    tmv::domain::NestedMappedTerrainGeometryV1 rebuilt;
    try
    {
        rebuilt = tmv::domain::build_stationary_nested_terrain_geometry(geometry.config,
                                                                        geometry.relationship,
                                                                        geometry.parent_origin_earth_m,
                                                                        geometry.generation,
                                                                        geometry.sampling_version);
    }
    catch (const std::exception& exception)
    {
        error = std::string("invalid active-terrain checkpoint geometry: ") + exception.what();
        return false;
    }
    if (rebuilt.child_origin_earth_m.x_m != geometry.child_origin_earth_m.x_m ||
        rebuilt.child_origin_earth_m.y_m != geometry.child_origin_earth_m.y_m ||
        rebuilt.feature_center_earth_m.x_m != geometry.feature_center_earth_m.x_m ||
        rebuilt.feature_center_earth_m.y_m != geometry.feature_center_earth_m.y_m)
    {
        error = "active-terrain checkpoint Earth-frame identity does not reconstruct exactly";
        return false;
    }

    const int nx = relationship.child_cells_x;
    const int ny = relationship.child_cells_y;
    const int nz = relationship.levels_z;
    if (!valid_terrain_field(active.child_state.scalar, nx, ny, nz) ||
        !valid_terrain_field(active.child_state.x_flux, nx + 1, ny, nz) ||
        !valid_terrain_field(active.child_state.y_flux, nx, ny + 1, nz) ||
        !valid_terrain_field(active.child_state.zeta_flux, nx, ny, nz + 1) ||
        !valid_terrain_field(active.boundary.relaxation_weight, nx, ny, nz) ||
        !valid_terrain_field(active.boundary.ground_u_ms, nx, ny, 1) ||
        !valid_terrain_field(active.boundary.ground_v_ms, nx, ny, 1) ||
        !valid_terrain_field(active.boundary.ground_w_ms, nx, ny, 1))
    {
        error = "active-terrain checkpoint contains malformed or nonfinite fields";
        return false;
    }

    const double                                ground_u_ms = active.boundary.ground_u_ms.values.front();
    const double                                ground_v_ms = active.boundary.ground_v_ms.values.front();
    tmv::domain::NestedTerrainTransferLedgerV1  boundary_ledger;
    tmv::domain::NestedTerrainTangentBoundaryV1 rebuilt_boundary;
    try
    {
        rebuilt_boundary =
            tmv::domain::build_nested_terrain_tangent_boundary(rebuilt, ground_u_ms, ground_v_ms, &boundary_ledger);
    }
    catch (const std::exception& exception)
    {
        error = std::string("invalid active-terrain checkpoint boundary: ") + exception.what();
        return false;
    }
    if (!exact_terrain_field(active.boundary.relaxation_weight, rebuilt_boundary.relaxation_weight) ||
        !exact_terrain_field(active.boundary.ground_u_ms, rebuilt_boundary.ground_u_ms) ||
        !exact_terrain_field(active.boundary.ground_v_ms, rebuilt_boundary.ground_v_ms) ||
        !exact_terrain_field(active.boundary.ground_w_ms, rebuilt_boundary.ground_w_ms))
    {
        error = "active-terrain checkpoint boundary does not match reconstructed terrain";
        return false;
    }

    const double ledger_values[]              = {ledger.parent_scalar_inventory,
                                                 ledger.child_scalar_inventory,
                                                 ledger.scalar_inventory_residual,
                                                 ledger.normalized_scalar_inventory_residual,
                                                 ledger.maximum_x_flux_residual,
                                                 ledger.maximum_normalized_x_flux_residual,
                                                 ledger.maximum_y_flux_residual,
                                                 ledger.maximum_normalized_y_flux_residual,
                                                 ledger.maximum_zeta_flux_residual,
                                                 ledger.maximum_normalized_zeta_flux_residual,
                                                 ledger.maximum_ground_tangency_residual_ms};
    const double encoded_inventory_difference = ledger.child_scalar_inventory - ledger.parent_scalar_inventory;
    const double inventory_roundoff_tolerance =
        std::numeric_limits<double>::epsilon() *
        std::max({1.0, std::abs(ledger.parent_scalar_inventory), std::abs(ledger.child_scalar_inventory)});
    if (!std::all_of(
            std::begin(ledger_values), std::end(ledger_values), [](double value) { return std::isfinite(value); }) ||
        ledger.normalized_scalar_inventory_residual < 0.0 || ledger.maximum_x_flux_residual < 0.0 ||
        ledger.maximum_normalized_x_flux_residual < 0.0 || ledger.maximum_y_flux_residual < 0.0 ||
        ledger.maximum_normalized_y_flux_residual < 0.0 || ledger.maximum_zeta_flux_residual < 0.0 ||
        ledger.maximum_normalized_zeta_flux_residual < 0.0 ||
        ledger.maximum_ground_tangency_residual_ms != boundary_ledger.maximum_ground_tangency_residual_ms ||
        ledger.relaxation_owned_values != boundary_ledger.relaxation_owned_values ||
        std::abs(ledger.scalar_inventory_residual - encoded_inventory_difference) > inventory_roundoff_tolerance)
    {
        error = "active-terrain checkpoint transfer ledger is inconsistent";
        return false;
    }
    return true;
}

void encode_active_terrain_checkpoint(Encoder& out, const tmv::domain::NestedActiveTerrainStateV1& active)
{
    out.boolean(active.ready);
    if (!active.ready)
        return;
    out.unsigned_value<std::uint32_t>(tmv::domain::NestedActiveTerrainStateV1::schema_version);
    encode_terrain_config(out, active.geometry.config);
    encode_terrain_placement_config(out, active.geometry.config);
    encode_nest_relationship(out, active.geometry.relationship);
    out.floating(active.geometry.parent_origin_earth_m.x_m);
    out.floating(active.geometry.parent_origin_earth_m.y_m);
    out.floating(active.geometry.child_origin_earth_m.x_m);
    out.floating(active.geometry.child_origin_earth_m.y_m);
    out.floating(active.geometry.feature_center_earth_m.x_m);
    out.floating(active.geometry.feature_center_earth_m.y_m);
    out.unsigned_value(active.geometry.generation);
    out.unsigned_value(active.geometry.sampling_version);
    encode_terrain_field(out, active.child_state.scalar);
    encode_terrain_field(out, active.child_state.x_flux);
    encode_terrain_field(out, active.child_state.y_flux);
    encode_terrain_field(out, active.child_state.zeta_flux);
    encode_terrain_field(out, active.boundary.relaxation_weight);
    encode_terrain_field(out, active.boundary.ground_u_ms);
    encode_terrain_field(out, active.boundary.ground_v_ms);
    encode_terrain_field(out, active.boundary.ground_w_ms);
    const auto& ledger = active.ledger;
    out.floating(ledger.parent_time_s);
    out.floating(ledger.child_time_s);
    out.floating(ledger.parent_scalar_inventory);
    out.floating(ledger.child_scalar_inventory);
    out.floating(ledger.scalar_inventory_residual);
    out.floating(ledger.normalized_scalar_inventory_residual);
    out.floating(ledger.maximum_x_flux_residual);
    out.floating(ledger.maximum_normalized_x_flux_residual);
    out.floating(ledger.maximum_y_flux_residual);
    out.floating(ledger.maximum_normalized_y_flux_residual);
    out.floating(ledger.maximum_zeta_flux_residual);
    out.floating(ledger.maximum_normalized_zeta_flux_residual);
    out.floating(ledger.maximum_ground_tangency_residual_ms);
    out.unsigned_value(ledger.relaxation_owned_values);
    out.floating(active.parent_time_s);
    out.floating(active.child_time_s);
    out.unsigned_value(active.generation);
    out.unsigned_value(active.publication_count);
}

bool decode_active_terrain_checkpoint(Decoder&                                     in,
                                      tmv::domain::NestedActiveTerrainStateV1&     active,
                                      const tmv::domain::OneWayNestRelationshipV1& relationship,
                                      const TerrainConfig&                         runtime_config,
                                      tmv::domain::HorizontalPoint                 expected_parent_origin_earth_m,
                                      double                                       parent_time_s,
                                      double                                       child_time_s,
                                      std::string&                                 error)
{
    if (!in.boolean(active.ready))
        return false;
    if (!active.ready)
        return true;
    std::uint32_t schema_version = 0;
    auto&         geometry       = active.geometry;
    auto&         ledger         = active.ledger;
    geometry.sampling_version    = 1;
    if (!in.unsigned_value(schema_version) ||
        (schema_version != 1 && schema_version != tmv::domain::NestedActiveTerrainStateV1::schema_version) ||
        !decode_terrain_config(in, geometry.config) || !in.string(geometry.config.horizontal_placement_id) ||
        !in.floating(geometry.config.feature_center_earth_x_m) ||
        !in.floating(geometry.config.feature_center_earth_y_m) ||
        !decode_nest_relationship(in, geometry.relationship) || !in.floating(geometry.parent_origin_earth_m.x_m) ||
        !in.floating(geometry.parent_origin_earth_m.y_m) || !in.floating(geometry.child_origin_earth_m.x_m) ||
        !in.floating(geometry.child_origin_earth_m.y_m) || !in.floating(geometry.feature_center_earth_m.x_m) ||
        !in.floating(geometry.feature_center_earth_m.y_m) || !in.unsigned_value(geometry.generation) ||
        (schema_version >= 2 && !in.unsigned_value(geometry.sampling_version)) ||
        !decode_terrain_field(in, active.child_state.scalar) || !decode_terrain_field(in, active.child_state.x_flux) ||
        !decode_terrain_field(in, active.child_state.y_flux) ||
        !decode_terrain_field(in, active.child_state.zeta_flux) ||
        !decode_terrain_field(in, active.boundary.relaxation_weight) ||
        !decode_terrain_field(in, active.boundary.ground_u_ms) ||
        !decode_terrain_field(in, active.boundary.ground_v_ms) ||
        !decode_terrain_field(in, active.boundary.ground_w_ms) || !in.floating(ledger.parent_time_s) ||
        !in.floating(ledger.child_time_s) || !in.floating(ledger.parent_scalar_inventory) ||
        !in.floating(ledger.child_scalar_inventory) || !in.floating(ledger.scalar_inventory_residual) ||
        !in.floating(ledger.normalized_scalar_inventory_residual) || !in.floating(ledger.maximum_x_flux_residual) ||
        !in.floating(ledger.maximum_normalized_x_flux_residual) || !in.floating(ledger.maximum_y_flux_residual) ||
        !in.floating(ledger.maximum_normalized_y_flux_residual) || !in.floating(ledger.maximum_zeta_flux_residual) ||
        !in.floating(ledger.maximum_normalized_zeta_flux_residual) ||
        !in.floating(ledger.maximum_ground_tangency_residual_ms) ||
        !in.unsigned_value(ledger.relaxation_owned_values) || !in.floating(active.parent_time_s) ||
        !in.floating(active.child_time_s) || !in.unsigned_value(active.generation) ||
        !in.unsigned_value(active.publication_count))
    {
        if (error.empty() && (schema_version != 1 && schema_version != tmv::domain::NestedActiveTerrainStateV1::schema_version))
            error = "unsupported active-terrain checkpoint schema version";
        return false;
    }
    try
    {
        auto rebuilt = tmv::domain::build_stationary_nested_terrain_geometry(geometry.config,
                                                                             geometry.relationship,
                                                                             geometry.parent_origin_earth_m,
                                                                             geometry.generation,
                                                                             geometry.sampling_version);
        if (rebuilt.child_origin_earth_m.x_m != geometry.child_origin_earth_m.x_m ||
            rebuilt.child_origin_earth_m.y_m != geometry.child_origin_earth_m.y_m ||
            rebuilt.feature_center_earth_m.x_m != geometry.feature_center_earth_m.x_m ||
            rebuilt.feature_center_earth_m.y_m != geometry.feature_center_earth_m.y_m)
        {
            error = "active-terrain checkpoint Earth-frame identity does not reconstruct exactly";
            return false;
        }
        geometry = std::move(rebuilt);
    }
    catch (const std::exception& exception)
    {
        error = std::string("cannot reconstruct active-terrain checkpoint: ") + exception.what();
        return false;
    }
    return valid_active_terrain_checkpoint(
        active, relationship, runtime_config, expected_parent_origin_earth_m, parent_time_s, child_time_s, error);
}

void encode_surface_state(Encoder& out, const SurfaceDiagnosticFieldStorage& value)
{
    encode_float_vector(out, value.ustar);
    encode_float_vector(out, value.tau_u);
    encode_float_vector(out, value.tau_v);
    encode_float_vector(out, value.sensible_heat_flux);
    encode_float_vector(out, value.latent_heat_flux);
    encode_float_vector(out, value.moisture_flux);
    encode_float_vector(out, value.pbl_height);
    encode_float_vector(out, value.precipitation_rate_mm_h);
    encode_float_vector(out, value.accumulated_rainfall_mm);
    encode_float_vector(out, value.accumulated_rainfall_residual_mm);
    out.floating(value.accumulated_rainfall_last_update_s);
}

bool decode_surface_state(Decoder& in, SurfaceDiagnosticFieldStorage& value)
{
    return decode_float_vector(in, value.ustar) && decode_float_vector(in, value.tau_u) &&
           decode_float_vector(in, value.tau_v) && decode_float_vector(in, value.sensible_heat_flux) &&
           decode_float_vector(in, value.latent_heat_flux) && decode_float_vector(in, value.moisture_flux) &&
           decode_float_vector(in, value.pbl_height) && decode_float_vector(in, value.precipitation_rate_mm_h) &&
           decode_float_vector(in, value.accumulated_rainfall_mm) &&
           decode_float_vector(in, value.accumulated_rainfall_residual_mm) &&
           in.floating(value.accumulated_rainfall_last_update_s);
}

void encode_moist_energy(Encoder& out, const MoistMixtureEnergyBudgetV2& value)
{
    out.floating(value.dry_internal);
    out.floating(value.water_sensible);
    out.floating(value.phase_reference);
    out.floating(value.resolved_kinetic);
    out.floating(value.gravitational);
    out.floating(value.subgrid_tke);
    out.floating(value.total_water_mass);
    out.floating(value.total);
}

bool decode_moist_energy(Decoder& in, MoistMixtureEnergyBudgetV2& value)
{
    return in.floating(value.dry_internal) && in.floating(value.water_sensible) && in.floating(value.phase_reference) &&
           in.floating(value.resolved_kinetic) && in.floating(value.gravitational) && in.floating(value.subgrid_tke) &&
           in.floating(value.total_water_mass) && in.floating(value.total);
}

void encode_child_relaxation_report(Encoder& out, const NestedChildRelaxationReport& value)
{
    const tmv::domain::ConservedScalarRelaxationReport* scalars[] = {&value.theta,
                                                                     &value.qv,
                                                                     &value.qc,
                                                                     &value.qr,
                                                                     &value.qi,
                                                                     &value.qs,
                                                                     &value.qg,
                                                                     &value.qh,
                                                                     &value.tracer,
                                                                     &value.tke};
    for (const auto* scalar : scalars)
    {
        out.floating(scalar->target_dry_mass_change_kg);
        out.floating(scalar->committed_dry_mass_change_kg);
        out.floating(scalar->dry_storage_residual_kg);
        out.floating(scalar->target_carrier_change);
        out.floating(scalar->committed_carrier_change);
        out.floating(scalar->carrier_storage_residual);
        out.unsigned_value<std::uint64_t>(scalar->forced_cells);
    }
    const auto& momentum = value.momentum;
    out.floating(momentum.target_x_change_kgms);
    out.floating(momentum.committed_x_change_kgms);
    out.floating(momentum.x_storage_residual_kgms);
    out.floating(momentum.target_y_change_kgms);
    out.floating(momentum.committed_y_change_kgms);
    out.floating(momentum.y_storage_residual_kgms);
    out.floating(momentum.target_z_change_kgms);
    out.floating(momentum.committed_z_change_kgms);
    out.floating(momentum.z_storage_residual_kgms);
    out.floating(value.pressure.target_pressure_volume_integral_pa_m3);
    out.floating(value.pressure.committed_pressure_volume_integral_pa_m3);
    out.floating(value.pressure.storage_residual_pa_m3);
    out.unsigned_value<std::uint64_t>(value.pressure.diagnosed_cells);
    encode_moist_energy(out, value.energy.before);
    encode_moist_energy(out, value.energy.after);
    encode_moist_energy(out, value.energy.committed_change);
    out.floating(value.energy.reservoir_sum_closure_residual_j);
    const auto& private_report = value.microphysics_private;
    out.unsigned_value<std::uint64_t>(private_report.scalar_count);
    for (std::size_t index = 0; index < private_report.scalar_count; ++index)
    {
        const auto& name = private_report.name[index];
        const auto  end  = std::find(name.begin(), name.end(), '\0');
        out.string(std::string(name.begin(), end));
        out.unsigned_value(private_report.storage_semantic[index]);
        out.boolean(private_report.positive_definite[index]);
        out.floating(private_report.target_carrier_change[index]);
        out.floating(private_report.committed_carrier_change[index]);
        out.floating(private_report.carrier_storage_residual[index]);
        out.unsigned_value<std::uint64_t>(private_report.forced_cells[index]);
    }
}

bool decode_child_relaxation_report(Decoder& in, NestedChildRelaxationReport& value, bool decode_private)
{
    tmv::domain::ConservedScalarRelaxationReport* scalars[] = {&value.theta,
                                                               &value.qv,
                                                               &value.qc,
                                                               &value.qr,
                                                               &value.qi,
                                                               &value.qs,
                                                               &value.qg,
                                                               &value.qh,
                                                               &value.tracer,
                                                               &value.tke};
    for (auto* scalar : scalars)
    {
        std::uint64_t forced_cells = 0;
        if (!in.floating(scalar->target_dry_mass_change_kg) || !in.floating(scalar->committed_dry_mass_change_kg) ||
            !in.floating(scalar->dry_storage_residual_kg) || !in.floating(scalar->target_carrier_change) ||
            !in.floating(scalar->committed_carrier_change) || !in.floating(scalar->carrier_storage_residual) ||
            !in.unsigned_value(forced_cells) || forced_cells > std::numeric_limits<std::size_t>::max())
            return false;
        scalar->forced_cells = static_cast<std::size_t>(forced_cells);
    }
    auto&         momentum        = value.momentum;
    std::uint64_t diagnosed_cells = 0;
    if (!in.floating(momentum.target_x_change_kgms) || !in.floating(momentum.committed_x_change_kgms) ||
        !in.floating(momentum.x_storage_residual_kgms) || !in.floating(momentum.target_y_change_kgms) ||
        !in.floating(momentum.committed_y_change_kgms) || !in.floating(momentum.y_storage_residual_kgms) ||
        !in.floating(momentum.target_z_change_kgms) || !in.floating(momentum.committed_z_change_kgms) ||
        !in.floating(momentum.z_storage_residual_kgms) ||
        !in.floating(value.pressure.target_pressure_volume_integral_pa_m3) ||
        !in.floating(value.pressure.committed_pressure_volume_integral_pa_m3) ||
        !in.floating(value.pressure.storage_residual_pa_m3) || !in.unsigned_value(diagnosed_cells) ||
        diagnosed_cells > std::numeric_limits<std::size_t>::max() || !decode_moist_energy(in, value.energy.before) ||
        !decode_moist_energy(in, value.energy.after) || !decode_moist_energy(in, value.energy.committed_change) ||
        !in.floating(value.energy.reservoir_sum_closure_residual_j))
        return false;
    value.pressure.diagnosed_cells = static_cast<std::size_t>(diagnosed_cells);
    if (decode_private)
    {
        std::uint64_t scalar_count = 0;
        if (!in.unsigned_value(scalar_count) || scalar_count > NestedPrivateScalarRelaxationReportV1::maximum_scalars)
            return false;
        auto& private_report        = value.microphysics_private;
        private_report.scalar_count = static_cast<std::size_t>(scalar_count);
        for (std::size_t index = 0; index < private_report.scalar_count; ++index)
        {
            std::string   name;
            std::uint64_t forced_cells = 0;
            if (!in.string(name) || name.empty() ||
                name.size() >= NestedPrivateScalarRelaxationReportV1::maximum_name_bytes ||
                !in.unsigned_value(private_report.storage_semantic[index]) ||
                !in.boolean(private_report.positive_definite[index]) ||
                !in.floating(private_report.target_carrier_change[index]) ||
                !in.floating(private_report.committed_carrier_change[index]) ||
                !in.floating(private_report.carrier_storage_residual[index]) || !in.unsigned_value(forced_cells) ||
                forced_cells > std::numeric_limits<std::size_t>::max())
                return false;
            std::copy(name.begin(), name.end(), private_report.name[index].begin());
            private_report.forced_cells[index] = static_cast<std::size_t>(forced_cells);
        }
    }
    return true;
}

bool valid_child_relaxation_report(const NestedChildRelaxationReport& value) noexcept
{
    const tmv::domain::ConservedScalarRelaxationReport* scalars[] = {&value.theta,
                                                                     &value.qv,
                                                                     &value.qc,
                                                                     &value.qr,
                                                                     &value.qi,
                                                                     &value.qs,
                                                                     &value.qg,
                                                                     &value.qh,
                                                                     &value.tracer,
                                                                     &value.tke};
    for (const auto* scalar : scalars)
    {
        const double values[] = {scalar->target_dry_mass_change_kg,
                                 scalar->committed_dry_mass_change_kg,
                                 scalar->dry_storage_residual_kg,
                                 scalar->target_carrier_change,
                                 scalar->committed_carrier_change,
                                 scalar->carrier_storage_residual};
        if (!std::all_of(std::begin(values), std::end(values), [](double item) { return std::isfinite(item); }) ||
            scalar->dry_storage_residual_kg !=
                scalar->target_dry_mass_change_kg - scalar->committed_dry_mass_change_kg ||
            scalar->carrier_storage_residual != scalar->target_carrier_change - scalar->committed_carrier_change)
            return false;
    }
    const auto&  momentum          = value.momentum;
    const double momentum_values[] = {momentum.target_x_change_kgms,
                                      momentum.committed_x_change_kgms,
                                      momentum.x_storage_residual_kgms,
                                      momentum.target_y_change_kgms,
                                      momentum.committed_y_change_kgms,
                                      momentum.y_storage_residual_kgms,
                                      momentum.target_z_change_kgms,
                                      momentum.committed_z_change_kgms,
                                      momentum.z_storage_residual_kgms};
    if (!std::all_of(
            std::begin(momentum_values), std::end(momentum_values), [](double item) { return std::isfinite(item); }) ||
        momentum.x_storage_residual_kgms != momentum.target_x_change_kgms - momentum.committed_x_change_kgms ||
        momentum.y_storage_residual_kgms != momentum.target_y_change_kgms - momentum.committed_y_change_kgms ||
        momentum.z_storage_residual_kgms != momentum.target_z_change_kgms - momentum.committed_z_change_kgms)
        return false;
    const auto valid_energy = [](const MoistMixtureEnergyBudgetV2& energy)
    {
        const double values[] = {energy.dry_internal,
                                 energy.water_sensible,
                                 energy.phase_reference,
                                 energy.resolved_kinetic,
                                 energy.gravitational,
                                 energy.subgrid_tke,
                                 energy.total_water_mass,
                                 energy.total};
        return std::all_of(std::begin(values), std::end(values), [](double item) { return std::isfinite(item); });
    };
    const auto&  change            = value.energy.committed_change;
    const double pressure_values[] = {value.pressure.target_pressure_volume_integral_pa_m3,
                                      value.pressure.committed_pressure_volume_integral_pa_m3,
                                      value.pressure.storage_residual_pa_m3};
    const auto&  private_report    = value.microphysics_private;
    if (private_report.scalar_count > NestedPrivateScalarRelaxationReportV1::maximum_scalars)
        return false;
    for (std::size_t index = 0; index < private_report.scalar_count; ++index)
    {
        const auto&  name      = private_report.name[index];
        const auto   end       = std::find(name.begin(), name.end(), '\0');
        const double target    = private_report.target_carrier_change[index];
        const double committed = private_report.committed_carrier_change[index];
        const double residual  = private_report.carrier_storage_residual[index];
        if (end == name.begin() ||
            private_report.storage_semantic[index] >
                static_cast<std::uint8_t>(tmv::MicrophysicsScalarStorage::VolumeConcentration) ||
            !std::isfinite(target) || !std::isfinite(committed) || !std::isfinite(residual) ||
            residual != target - committed)
            return false;
        for (std::size_t prior = 0; prior < index; ++prior)
            if (private_report.name[prior] == name)
                return false;
    }
    return std::all_of(std::begin(pressure_values),
                       std::end(pressure_values),
                       [](double item) { return std::isfinite(item); }) &&
           value.pressure.storage_residual_pa_m3 == value.pressure.target_pressure_volume_integral_pa_m3 -
                                                        value.pressure.committed_pressure_volume_integral_pa_m3 &&
           valid_energy(value.energy.before) && valid_energy(value.energy.after) && valid_energy(change) &&
           std::isfinite(value.energy.reservoir_sum_closure_residual_j) &&
           value.energy.reservoir_sum_closure_residual_j ==
               change.total - (change.dry_internal + change.water_sensible + change.phase_reference +
                               change.resolved_kinetic + change.gravitational + change.subgrid_tke);
}

void encode_interface_source_ledger(Encoder& out, const NestedInterfaceSourceLedgerV2& value)
{
    out.unsigned_value(NestedInterfaceSourceLedgerV2::schema_version);
    out.boolean(value.available);
    out.unsigned_value(value.application_count);
    encode_child_relaxation_report(out, value.last);
    encode_child_relaxation_report(out, value.cumulative);
}

bool decode_interface_source_ledger(Decoder& in, NestedInterfaceSourceLedgerV2& value, bool decode_private)
{
    std::uint32_t schema_version = 0;
    return in.unsigned_value(schema_version) &&
           schema_version == (decode_private ? NestedInterfaceSourceLedgerV2::schema_version : 1U) &&
           in.boolean(value.available) && in.unsigned_value(value.application_count) &&
           decode_child_relaxation_report(in, value.last, decode_private) &&
           decode_child_relaxation_report(in, value.cumulative, decode_private);
}

void encode_nested_carrier_feedback(Encoder& out, const NestedWaterFeedbackReport& item)
{
    const auto& p = item.parent;
    for (double v : {p.initial_inventory,
                     p.final_inventory,
                     p.child_inventory,
                     p.restriction_change,
                     p.interface_mismatch,
                     p.applied_reflux,
                     p.storage_rounding,
                     p.closure_residual})
        out.floating(v);
    out.unsigned_value<std::uint64_t>(p.restricted_cells);
    out.unsigned_value<std::uint64_t>(p.refluxed_cells);
    out.boolean(p.complete);
    out.floating(p.redistributed_inventory);
    out.unsigned_value<std::uint64_t>(p.redistributed_cells);
    out.floating(p.global_redistributed_inventory);
    out.unsigned_value<std::uint64_t>(p.global_redistributed_cells);
    out.floating(p.child_synchronized_inventory);
    out.floating(p.fine_requested_transfer);
    out.floating(p.fine_storage_rounding);
    out.floating(p.phase_exchange_inventory);
    out.unsigned_value<std::uint64_t>(p.constrained_cells);
    for (double v : {item.child_begin,
                     item.child_transport,
                     item.interface_source,
                     item.physics_source,
                     item.transport_storage,
                     item.child_change,
                     item.closure_residual,
                     item.tolerance})
        out.floating(v);
}

void encode_two_way_feedback_report(Encoder& out, const NestedTwoWayFeedbackReportV1& value)
{
    out.unsigned_value(NestedTwoWayFeedbackReportV1::schema_version);
    out.boolean(value.complete);
    out.unsigned_value<std::uint64_t>(value.restricted_scalar_cells);
    out.unsigned_value<std::uint64_t>(value.restricted_x_faces);
    out.unsigned_value<std::uint64_t>(value.restricted_y_faces);
    out.unsigned_value<std::uint64_t>(value.restricted_z_faces);
    out.unsigned_value<std::uint64_t>(value.refluxed_scalar_cells);
    for (const auto* values : {&value.interface_mismatch,
                               &value.applied_reflux,
                               &value.storage_residual,
                               &value.child_interface_source,
                               &value.child_dynamics_source,
                               &value.child_dynamics_storage,
                               &value.child_interface_transport,
                               &value.child_interval_change,
                               &value.child_composite_residual,
                               &value.child_composite_tolerance})
        for (double item : *values)
            out.floating(item);
    out.boolean(value.moist);
    if (value.moist)
        for (const auto& item : value.water)
            encode_nested_carrier_feedback(out, item);
    const auto& movement = value.relocation;
    out.boolean(movement.complete);
    if (movement.complete)
    {
        out.unsigned_value(movement.movement_sequence);
        out.unsigned_value(movement.feedback_application_count);
        for (const auto* values : {&movement.before,
                                   &movement.retired,
                                   &movement.admitted,
                                   &movement.boundary_source,
                                   &movement.after,
                                   &movement.arithmetic_scale})
            for (double item : *values)
                out.floating(item);
        out.unsigned_value<std::uint64_t>(movement.private_scalars.size());
        for (const auto& scalar : movement.private_scalars)
        {
            out.string(scalar.name);
            for (double item : {scalar.before,
                                scalar.retired,
                                scalar.admitted,
                                scalar.boundary_source,
                                scalar.after,
                                scalar.arithmetic_scale})
                out.floating(item);
        }
    }
    out.unsigned_value<std::uint64_t>(value.phase_projection_cells);
    out.floating(value.phase_theta_inventory_change);
    out.floating(value.phase_thermal_roundoff_j);
    out.floating(value.phase_thermal_scale_j);
    out.unsigned_value<std::uint64_t>(value.private_scalars.size());
    for (const auto& scalar : value.private_scalars)
    {
        out.string(scalar.name);
        encode_nested_carrier_feedback(out, scalar.carrier);
    }
}

bool decode_nested_carrier_feedback(Decoder& in, NestedWaterFeedbackReport& item, std::uint32_t schema)
{
    auto& p = item.parent;
    for (auto* v : {&p.initial_inventory,
                    &p.final_inventory,
                    &p.child_inventory,
                    &p.restriction_change,
                    &p.interface_mismatch,
                    &p.applied_reflux,
                    &p.storage_rounding,
                    &p.closure_residual})
        if (!in.floating(*v))
            return false;
    std::uint64_t restricted = 0, refluxed = 0;
    if (!in.unsigned_value(restricted) || !in.unsigned_value(refluxed) ||
        restricted > std::numeric_limits<std::size_t>::max() || refluxed > std::numeric_limits<std::size_t>::max() ||
        !in.boolean(p.complete))
        return false;
    p.restricted_cells = restricted;
    p.refluxed_cells   = refluxed;
    if (schema >= 5)
    {
        std::uint64_t redistributed = 0;
        if (!in.floating(p.redistributed_inventory) || !in.unsigned_value(redistributed) ||
            redistributed > std::numeric_limits<std::size_t>::max())
            return false;
        p.redistributed_cells = redistributed;
    }
    if (schema >= 6)
    {
        std::uint64_t count = 0;
        if (!in.floating(p.global_redistributed_inventory) || !in.unsigned_value(count) ||
            count > std::numeric_limits<std::size_t>::max())
            return false;
        p.global_redistributed_cells = count;
        if (!in.floating(p.child_synchronized_inventory) || !in.floating(p.fine_requested_transfer) ||
            !in.floating(p.fine_storage_rounding))
            return false;
    }
    else
        p.child_synchronized_inventory = p.child_inventory;
    if (schema >= 8 && !in.floating(p.phase_exchange_inventory))
        return false;
    if (schema >= 10)
    {
        std::uint64_t constrained = 0;
        if (!in.unsigned_value(constrained) || constrained > std::numeric_limits<std::size_t>::max())
            return false;
        p.constrained_cells = constrained;
    }

    for (auto* v : {&item.child_begin,
                    &item.child_transport,
                    &item.interface_source,
                    &item.physics_source,
                    &item.transport_storage,
                    &item.child_change,
                    &item.closure_residual,
                    &item.tolerance})
        if (!in.floating(*v))
            return false;
    return true;
}

bool decode_two_way_feedback_report(Decoder& in, NestedTwoWayFeedbackReportV1& value)
{
    std::uint32_t                schema = 0;
    std::array<std::uint64_t, 5> counts{};
    if (!in.unsigned_value(schema) ||
        (schema < 2 || schema > NestedTwoWayFeedbackReportV1::schema_version) ||
        !in.boolean(value.complete))
        return false;
    for (auto& count : counts)
        if (!in.unsigned_value(count) || count > std::numeric_limits<std::size_t>::max())
            return false;
    value.restricted_scalar_cells = counts[0];
    value.restricted_x_faces      = counts[1];
    value.restricted_y_faces      = counts[2];
    value.restricted_z_faces      = counts[3];
    value.refluxed_scalar_cells   = counts[4];
    for (auto* values : {&value.interface_mismatch,
                         &value.applied_reflux,
                         &value.storage_residual,
                         &value.child_interface_source,
                         &value.child_dynamics_source})
        for (double& item : *values)
            if (!in.floating(item))
                return false;
    if (schema >= 3)
        for (double& item : value.child_dynamics_storage)
            if (!in.floating(item))
                return false;
    for (auto* values : {&value.child_interface_transport,
                         &value.child_interval_change,
                         &value.child_composite_residual,
                         &value.child_composite_tolerance})
        for (double& item : *values)
            if (!in.floating(item))
                return false;
    value.moist = false;
    value.water = {};
    if (schema >= 4)
    {
        if (!in.boolean(value.moist))
            return false;
        if (value.moist)
            for (auto& item : value.water)
                if (!decode_nested_carrier_feedback(in, item, schema))
                    return false;
    }
    value.relocation = {};
    if (schema >= 7)
    {
        auto& movement = value.relocation;
        if (!in.boolean(movement.complete))
            return false;
        if (movement.complete)
        {
            if (!in.unsigned_value(movement.movement_sequence) ||
                !in.unsigned_value(movement.feedback_application_count))
                return false;
            for (auto* values : {&movement.before,
                                 &movement.retired,
                                 &movement.admitted,
                                 &movement.boundary_source,
                                 &movement.after,
                                 &movement.arithmetic_scale})
                for (double& item : *values)
                    if (!in.floating(item))
                        return false;
            if (schema >= 11)
            {
                std::uint64_t count = 0;
                if (!in.unsigned_value(count) || count > 16)
                    return false;
                movement.private_scalars.resize(count);
                for (auto& scalar : movement.private_scalars)
                {
                    if (!in.string(scalar.name))
                        return false;
                    for (double* item : {&scalar.before,
                                         &scalar.retired,
                                         &scalar.admitted,
                                         &scalar.boundary_source,
                                         &scalar.after,
                                         &scalar.arithmetic_scale})
                        if (!in.floating(*item))
                            return false;
                }
            }
            if (!valid_nested_two_way_relocation_report(movement))
                return false;
        }
    }
    value.phase_projection_cells       = 0;
    value.phase_theta_inventory_change = 0;
    value.phase_thermal_roundoff_j     = 0;
    value.phase_thermal_scale_j        = 0;
    if (schema >= 8)
    {
        std::uint64_t count = 0;
        if (!in.unsigned_value(count) || count > std::numeric_limits<std::size_t>::max() ||
            !in.floating(value.phase_theta_inventory_change) || !in.floating(value.phase_thermal_roundoff_j) ||
            !in.floating(value.phase_thermal_scale_j))
            return false;
        value.phase_projection_cells = count;
    }
    value.private_scalars.clear();
    if (schema >= 9)
    {
        std::uint64_t count = 0;
        if (!in.unsigned_value(count) || count > 16)
            return false;
        value.private_scalars.resize(count);
        for (auto& scalar : value.private_scalars)
            if (!in.string(scalar.name) || !decode_nested_carrier_feedback(in, scalar.carrier, schema) ||
                (schema < 12 && scalar.carrier.parent.phase_exchange_inventory != 0))
                return false;
    }
    return valid_nested_phase_feedback_report(value) && valid_nested_private_feedback_reports(value.private_scalars);
}

bool decode_five_doubles(Decoder& in, std::array<double, 5>& values)
{
    for (double& value : values)
        if (!in.floating(value))
            return false;
    return true;
}

bool valid_interface_source_ledger(const NestedInterfaceSourceLedgerV2& value) noexcept
{
    return value.available && value.application_count > 0 && valid_child_relaxation_report(value.last) &&
           valid_child_relaxation_report(value.cumulative);
}

bool valid_synchronized_nested_checkpoint(const SimulationStorageState& state, std::string& error)
{
    const auto& nested = state.nested_execution;
    if (!nested.relationship_ready || !nested.reference_state_ready || !nested.physical_primary_state_ready ||
        !nested.parent_boundary_state_ready)
    {
        error = "active nesting checkpoint requires complete child and boundary state";
        return false;
    }
    const auto& selection    = state.configuration.nesting;
    const auto& relationship = nested.relationship;
    if (!selection.enabled || selection.parent_ratio_x != relationship.parent_ratio_x ||
        selection.parent_ratio_y != relationship.parent_ratio_y ||
        selection.parent_time_step_ratio != relationship.parent_time_step_ratio ||
        selection.child_origin_parent_cell_i != nested.initial_parent_origin_i ||
        selection.child_origin_parent_cell_j != nested.initial_parent_origin_j ||
        selection.child_cells_x != relationship.child_cells_x ||
        selection.child_cells_y != relationship.child_cells_y ||
        selection.relaxation_width_child_cells != relationship.relaxation_width_child_cells ||
        selection.boundary_stencil_parent_cells != relationship.boundary_stencil_parent_cells)
    {
        error = "active nesting checkpoint does not match its configured selection";
        return false;
    }
    if (!nested.parent_time_bracket.available ||
        nested.parent_time_bracket.completed_child_substeps != nested.relationship.parent_time_step_ratio ||
        nested.child.time_s != state.time_s)
    {
        error = "active nesting checkpoint requires a synchronized parent/child endpoint";
        return false;
    }
    const auto parent_physical =
        physical_scalar_domain(state.grid.nr, state.grid.nth, state.grid.nz, StaggerType::CGrid);
    const auto parent_grid_origin = state.configuration.domain_frame.earth_origin_at(state.time_s);
    const tmv::domain::HorizontalPoint parent_physical_origin{
        parent_grid_origin.x_m + parent_physical.i_begin * state.grid.dx,
        parent_grid_origin.y_m + parent_physical.j_begin * state.grid.dy};
    if (!valid_active_terrain_checkpoint(nested.active_terrain,
                                         nested.relationship,
                                         state.runtime.terrain.config,
                                         parent_physical_origin,
                                         state.time_s,
                                         nested.child.time_s,
                                         error))
        return false;
    if (nested.active_terrain.ready &&
        state.configuration.domain_motion_mode == tmv::init::config::DomainMotionMode::MovingNest &&
        nested.active_terrain.geometry.sampling_version != 2)
    {
        error = "moving mapped nesting requires parent-lattice terrain sampling";
        return false;
    }
    if (!valid_interface_source_ledger(nested.interface_source))
    {
        error = "active nesting checkpoint requires valid interface source provenance";
        return false;
    }
    if (selection.feedback == tmv::init::config::NestingFeedback::TwoWay)
    {
        const auto& feedback = nested.last_two_way_feedback;
        const auto  finite   = [](const auto& values)
        { return std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); }); };
        if (nested.two_way_feedback_application_count == 0 || !feedback.complete ||
            !valid_child_relaxation_report(nested.two_way_interval_source_begin) ||
            !finite(feedback.interface_mismatch) || !finite(feedback.applied_reflux) ||
            !finite(feedback.storage_residual) || !finite(feedback.child_interface_source) ||
            !finite(feedback.child_dynamics_source) || !finite(feedback.child_dynamics_storage) ||
            !finite(feedback.child_interface_transport) || !finite(feedback.child_interval_change) ||
            !finite(feedback.child_composite_residual) || !finite(feedback.child_composite_tolerance) ||
            !finite(nested.two_way_child_interval_begin_inventory) ||
            !std::isfinite(nested.two_way_dry_mass_storage_begin) || !std::isfinite(nested.two_way_theta_storage_begin))
        {
            error = "active two-way nesting checkpoint lacks complete feedback provenance";
            return false;
        }
        for (std::size_t currency = 0; currency < 5; ++currency)
        {
            if (feedback.child_composite_tolerance[currency] < 0.0 ||
                std::abs(feedback.child_composite_residual[currency]) > feedback.child_composite_tolerance[currency])
            {
                error = "active two-way nesting checkpoint violates child composite closure";
                return false;
            }
        }
        const auto& relocation = feedback.relocation;
        if ((!relocation.complete && !relocation.private_scalars.empty()) ||
            (relocation.complete &&
             (!valid_nested_two_way_relocation_report(relocation) ||
              relocation.movement_sequence != nested.movement_sequence ||
              relocation.feedback_application_count != nested.two_way_feedback_application_count)))
        {
            error = "invalid two-way child relocation checkpoint ledger";
            return false;
        }
        if (!valid_nested_phase_feedback_report(feedback) || feedback.phase_projection_cells > state.fields.rho.size())
        {
            error = "invalid phase-coupled two-way feedback checkpoint ledger";
            return false;
        }
        if (feedback.moist)
            for (const auto& item : feedback.water)
                if (!valid_nested_water_feedback_report(item))
                {
                    error = "invalid moist two-way feedback checkpoint ledger";
                    return false;
                }
    }
    if (!valid_tracer_roundoff_checkpoint_runtime(nested.child.runtime.advection))
    {
        error = "active nesting checkpoint has invalid child passive-tracer "
                "roundoff provenance";
        return false;
    }
    try
    {
        require_valid_nested_execution_relationship(state);
    }
    catch (const std::exception& exception)
    {
        error = std::string("invalid active nesting checkpoint state: ") + exception.what();
        return false;
    }

    const auto& fields      = nested.child.fields;
    const auto  exact_shape = [&](const Field3D& field)
    {
        return field.size_r() == nested.child.grid.nr && field.size_th() == nested.child.grid.nth &&
               field.size_z() == nested.child.grid.nz;
    };
    const auto finite_field = [&](const Field3D& field)
    {
        return exact_shape(field) &&
               std::all_of(field.data(), field.data() + field.size(), [](float value) { return std::isfinite(value); });
    };
    for (const Field3D* field : prognostic_fields(fields))
    {
        if (!finite_field(*field))
        {
            error = "active nesting checkpoint contains incomplete or nonfinite child fields";
            return false;
        }
    }
    if (selection.feedback == tmv::init::config::NestingFeedback::TwoWay)
    {
        const auto zero_field = [](const Field3D& field)
        { return std::all_of(field.data(), field.data() + field.size(), [](float value) { return value == 0.0f; }); };
        const Field3D* child_unowned[]  = {&fields.qv,
                                           &fields.qc,
                                           &fields.qr,
                                           &fields.qi,
                                           &fields.qs,
                                           &fields.qg,
                                           &fields.qh,
                                           &fields.tracer,
                                           &fields.tke};
        const auto&    parent_fields    = state.fields;
        const Field3D* parent_unowned[] = {&parent_fields.qv,
                                           &parent_fields.qc,
                                           &parent_fields.qr,
                                           &parent_fields.qi,
                                           &parent_fields.qs,
                                           &parent_fields.qg,
                                           &parent_fields.qh,
                                           &parent_fields.tracer,
                                           &parent_fields.tke};
        for(std::size_t f=0;f<std::size(child_unowned);++f)
            if ((!nested.last_two_way_feedback.moist || f >= 7) &&
                (!zero_field(*child_unowned[f]) || !zero_field(*parent_unowned[f])))
            {
                error = "active two-way nesting checkpoint contains an unowned carrier";
                return false;
            }
    }
    for (std::size_t index = 0; index < fields.rho.size(); ++index)
    {
        const float rho            = fields.rho.data()[index];
        const float pressure       = fields.p.data()[index];
        const float theta          = fields.theta.data()[index];
        const float qv             = fields.qv.data()[index];
        const float tracer         = fields.tracer.data()[index];
        const float hydrometeors[] = {fields.qc.data()[index],
                                      fields.qr.data()[index],
                                      fields.qi.data()[index],
                                      fields.qs.data()[index],
                                      fields.qg.data()[index],
                                      fields.qh.data()[index]};
        if (rho < density_min_kgm3 || rho > density_max_kgm3 || pressure < pressure_min_pa ||
            pressure > pressure_max_pa || theta < theta_min_k || theta > theta_max_k || qv < 0.0f || qv > qv_max_kgkg ||
            tracer < 0.0f || tracer > passive_tracer_max || std::signbit(tracer) ||
            !std::all_of(std::begin(hydrometeors),
                         std::end(hydrometeors),
                         [](float value) { return value >= 0.0f && value <= hydrometeor_max_kgkg; }) ||
            fields.tke.data()[index] < 0.0f || std::abs(fields.u.data()[index]) > wind_horizontal_abs_max_ms ||
            std::abs(fields.v.data()[index]) > wind_horizontal_abs_max_ms ||
            std::abs(fields.w.data()[index]) > wind_vertical_abs_max_ms)
        {
            error = "active nesting checkpoint child state exceeds strict physical bounds";
            return false;
        }
    }
    if (selection.feedback == tmv::init::config::NestingFeedback::TwoWay && nested.last_two_way_feedback.moist)
    {
        const std::array<const Field3D*, 7> water{
            &fields.qv, &fields.qc, &fields.qr, &fields.qi, &fields.qs, &fields.qg, &fields.qh};
        std::array<tmv::diagnostics::CompensatedSum, 7> totals;
        const auto&                                     r = nested.relationship;
        for (int i = 1; i <= r.child_cells_x; ++i)
            for (int j = 1; j <= r.child_cells_y; ++j)
                for (int k = 0; k < r.levels_z; ++k)
                {
                    const auto&  metrics = nested.active_terrain.geometry.child_storage.metrics;
                    const double volume =
                        nested.active_terrain.ready
                            ? metrics.J(i, j, k) * metrics.dx * metrics.dy * metrics.dzeta
                            : nested.child.grid.dx * nested.child.grid.dy * nested.child.process_grid_metrics.dz[k];
                    for (std::size_t f = 0; f < water.size(); ++f)
                        totals[f].add(static_cast<double>(fields.rho(i, j, k)) * (*water[f])(i, j, k) * volume);
                }
        for (std::size_t f = 0; f < water.size(); ++f)
        {
            const auto& relocation = nested.last_two_way_feedback.relocation;
            const double synchronized = nested.last_two_way_feedback.water[f].parent.child_synchronized_inventory;
            const double expected = relocation.complete ? relocation.after[f + 2] : synchronized;
            if (relocation.complete && std::abs(relocation.before[f + 2] - synchronized) >
                128 * std::numeric_limits<double>::epsilon() * std::max(1.0, synchronized))
            {
                error = "two-way relocation differs from its preceding feedback inventory";
                return false;
            }
            const double actual    = totals[f].value();
            const double tolerance = 128 * std::numeric_limits<double>::epsilon() * std::max({1.0, expected, actual});
            if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance)
            {
                error = "two-way checkpoint synchronized water inventory differs from its child state";
                return false;
            }
        }
    }
    if (selection.feedback == tmv::init::config::NestingFeedback::TwoWay &&
        nested.last_two_way_feedback.relocation.complete)
    {
        const auto actual = nested_primary_inventory(
            nested.relationship,
            nested.child.process_grid_metrics.dz,
            fields,
            0,
            0,
            false,
            false,
            nested.active_terrain.ready ? &nested.active_terrain.geometry.child_storage.metrics : nullptr);
        const auto& expected = nested.last_two_way_feedback.relocation.after;
        for (std::size_t f = 0; f < actual.size(); ++f)
            if (std::abs(actual[f] - expected[f]) > 128 * std::numeric_limits<double>::epsilon() *
                                                        std::max({1.0, std::abs(actual[f]), std::abs(expected[f])}))
            {
                error = "two-way relocation checkpoint inventory differs from child state";
                return false;
            }
    }
    if (!state.runtime.microphysics.scheme || !nested.child.runtime.microphysics.scheme ||
        state.runtime.microphysics.scheme->get_scheme_name() !=
            nested.child.runtime.microphysics.scheme->get_scheme_name())
    {
        error = "active nesting checkpoint microphysics selection differs between levels";
        return false;
    }
    auto        parent_private     = state.runtime.microphysics.scheme->prognostic_scalars_for_transport();
    auto        child_private      = nested.child.runtime.microphysics.scheme->prognostic_scalars_for_transport();
    const auto& last_private       = nested.interface_source.last.microphysics_private;
    const auto& cumulative_private = nested.interface_source.cumulative.microphysics_private;
    if (parent_private.size() != child_private.size() || last_private.scalar_count != parent_private.size() ||
        cumulative_private.scalar_count != parent_private.size())
    {
        error = "active nesting checkpoint private microphysics layout count differs";
        return false;
    }
    for (std::size_t slot = 0; slot < parent_private.size(); ++slot)
    {
        const auto&       parent       = parent_private[slot];
        const auto&       child        = child_private[slot];
        const auto&       durable_name = last_private.name[slot];
        const auto        name_end     = std::find(durable_name.begin(), durable_name.end(), '\0');
        const std::string name(durable_name.begin(), name_end);
        if (!parent.name || !child.name || !parent.field || !child.field || name != parent.name || name != child.name ||
            cumulative_private.name[slot] != durable_name ||
            last_private.storage_semantic[slot] != static_cast<std::uint8_t>(parent.storage) ||
            cumulative_private.storage_semantic[slot] != last_private.storage_semantic[slot] ||
            child.storage != parent.storage || last_private.positive_definite[slot] != parent.positive_definite ||
            cumulative_private.positive_definite[slot] != last_private.positive_definite[slot] ||
            child.positive_definite != parent.positive_definite || !finite_field(*child.field) ||
            child.field->size_r() != nested.child.grid.nr || child.field->size_th() != nested.child.grid.nth ||
            child.field->size_z() != nested.child.grid.nz || parent.field->size_r() != state.grid.nr ||
            parent.field->size_th() != state.grid.nth || parent.field->size_z() != state.grid.nz)
        {
            error = "active nesting checkpoint private microphysics identity or shape differs";
            return false;
        }
        for (std::size_t index = 0; index < child.field->size(); ++index)
            if (!std::isfinite(child.field->data()[index]) ||
                (child.positive_definite && child.field->data()[index] < 0.0f))
            {
                error = "active nesting checkpoint private microphysics field is invalid";
                return false;
            }
    }
    if (selection.feedback == tmv::init::config::NestingFeedback::TwoWay)
    {
        const auto& reports = nested.last_two_way_feedback.private_scalars;
        if (reports.size() != parent_private.size() || !valid_nested_private_feedback_reports(reports) ||
            !state.runtime.advection.private_interface.interval_empty() ||
            !nested.child.runtime.advection.private_interface.interval_empty())
        {
            error = "two-way checkpoint lacks closed private carrier provenance";
            return false;
        }
        const auto& relocated = nested.last_two_way_feedback.relocation;
        if (relocated.complete && relocated.private_scalars.size() != reports.size())
        {
            error = "two-way relocation checkpoint omits private carrier provenance";
            return false;
        }
        for (std::size_t f = 0; f < reports.size(); ++f)
        {
            tmv::diagnostics::CompensatedSum total;
            const auto&                      r = nested.relationship;
            for (int i = 1; i <= r.child_cells_x; ++i)
                for (int j = 1; j <= r.child_cells_y; ++j)
                    for (int k = 0; k < r.levels_z; ++k)
                    {
                        const auto&  metrics = nested.active_terrain.geometry.child_storage.metrics;
                        const double volume =
                            nested.active_terrain.ready
                                ? metrics.J(i, j, k) * metrics.dx * metrics.dy * metrics.dzeta
                                : nested.child.grid.dx * nested.child.grid.dy * nested.child.process_grid_metrics.dz[k];
                        total.add(static_cast<double>(fields.rho(i, j, k)) * (*child_private[f].field)(i, j, k) *
                                  volume);
                    }
            const auto   uncovered_cells = (std::size_t(r.parent_cells_x) * r.parent_cells_y -
                                            std::size_t(r.child_parent_span_x) * r.child_parent_span_y) *
                                           r.levels_z;
            const double synchronized    = reports[f].carrier.parent.child_synchronized_inventory;
            const double inventory       = relocated.complete ? relocated.private_scalars[f].after : synchronized;
            if (relocated.complete &&
                (relocated.private_scalars[f].name != reports[f].name ||
                 std::abs(relocated.private_scalars[f].before - synchronized) >
                     128 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(synchronized))))
            {
                error = "two-way private relocation differs from its preceding feedback inventory";
                return false;
            }
            if (reports[f].carrier.parent.constrained_cells > uncovered_cells || !parent_private[f].name ||
                reports[f].name != parent_private[f].name ||
                parent_private[f].storage != tmv::MicrophysicsScalarStorage::DryAirMixingRatio ||
                !parent_private[f].positive_definite || !std::isfinite(total.value()) ||
                std::abs(total.value() - inventory) >
                    128 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(inventory)))
            {
                error = "two-way checkpoint private carrier endpoint differs from feedback";
                return false;
            }
        }
    }
    return true;
}

bool encode_nested_execution(Encoder& out, const SimulationStorageState& state, std::string& error)
{
    if (!valid_synchronized_nested_checkpoint(state, error))
        return false;
    const auto& nested = state.nested_execution;
    out.boolean(nested.relationship_ready);
    out.boolean(nested.reference_state_ready);
    out.boolean(nested.physical_primary_state_ready);
    out.boolean(nested.parent_boundary_state_ready);
    encode_nest_relationship(out, nested.relationship);
    out.unsigned_value(nested.parent_time_bracket.version);
    out.boolean(nested.parent_time_bracket.available);
    out.floating(nested.parent_time_bracket.previous_parent_time_s);
    out.floating(nested.parent_time_bracket.current_parent_time_s);
    out.integer(nested.parent_time_bracket.completed_child_substeps);
    out.unsigned_value(nested.movement_sequence);
    out.integer(nested.initial_parent_origin_i);
    out.integer(nested.initial_parent_origin_j);
    out.integer(nested.cumulative_parent_shift_i);
    out.integer(nested.cumulative_parent_shift_j);

    const auto& child = nested.child;
    encode_grid(out, child.grid);
    out.floating(child.time_s);
    out.floating(child.coriolis_f);
    out.floating(child.domain_frame.earth_origin_at_t0_m.x_m);
    out.floating(child.domain_frame.earth_origin_at_t0_m.y_m);
    out.floating(child.domain_frame.velocity_ms.u_ms);
    out.floating(child.domain_frame.velocity_ms.v_ms);
    encode_double_vector(out, child.base.rho0);
    encode_double_vector(out, child.base.p0);
    encode_double_vector(out, child.base.qv0);
    encode_double_vector(out, child.base.theta0);
    encode_double_vector(out, child.base.u0);
    encode_double_vector(out, child.base.v0);
    encode_fields(out, prognostic_fields(child.fields));
    encode_fields(out, tendency_fields(child.tendencies));
    encode_surface_state(out, child.surface);
    encode_fields(out, diagnostic_fields(child.diagnostics));
    encode_interface_source_ledger(out, nested.interface_source);
    encode_child_relaxation_report(out, nested.two_way_interval_source_begin);
    for (double value : nested.two_way_child_interval_begin_inventory)
        out.floating(value);
    out.floating(nested.two_way_dry_mass_storage_begin);
    out.floating(nested.two_way_theta_storage_begin);
    encode_two_way_feedback_report(out, nested.last_two_way_feedback);
    out.unsigned_value(nested.two_way_feedback_application_count);

    SimulationStorageState runtime_shell(child.grid);
    runtime_shell.configuration                                       = state.configuration;
    runtime_shell.configuration.domain_frame                          = child.domain_frame;
    runtime_shell.time_s                                              = child.time_s;
    static_cast<AtmosphericLevelRuntimeState&>(runtime_shell.runtime) = child.runtime;
    out.unsigned_value<std::uint32_t>(55);
    encode_runtime(out, runtime_shell, error);
    out.integer(nested.cumulative_parent_recycling_shift_i);
    out.integer(nested.cumulative_parent_recycling_shift_j);
    Encoder active_terrain_size([](const std::uint8_t*, std::size_t) { return true; });
    encode_active_terrain_checkpoint(active_terrain_size, nested.active_terrain);
    if (!active_terrain_size.good())
    {
        error = active_terrain_size.error();
        return false;
    }
    out.unsigned_value<std::uint64_t>(active_terrain_size.size());
    encode_active_terrain_checkpoint(out, nested.active_terrain);
    out.unsigned_value<std::uint64_t>(active_terrain_size.size());
    return error.empty() && out.good();
}

bool decode_nested_execution(Decoder&                in,
                             SimulationStorageState& state,
                             std::uint32_t           section_version,
                             bool                    synthesize_legacy_selection,
                             std::string&            error)
{
    if (section_version != 1 && section_version != 2 && section_version != 3 && section_version != 4 &&
        section_version != 5 && section_version != 6 && section_version != 7 && section_version != 8)
    {
        error = "unsupported active nesting checkpoint section version";
        return false;
    }
    NestedExecutionStorage nested;
    if (!in.boolean(nested.relationship_ready) || !in.boolean(nested.reference_state_ready) ||
        !in.boolean(nested.physical_primary_state_ready) || !in.boolean(nested.parent_boundary_state_ready) ||
        !decode_nest_relationship(in, nested.relationship) || !in.unsigned_value(nested.parent_time_bracket.version) ||
        !in.boolean(nested.parent_time_bracket.available) ||
        !in.floating(nested.parent_time_bracket.previous_parent_time_s) ||
        !in.floating(nested.parent_time_bracket.current_parent_time_s) ||
        !in.integer(nested.parent_time_bracket.completed_child_substeps) ||
        !in.unsigned_value(nested.movement_sequence) || !in.integer(nested.initial_parent_origin_i) ||
        !in.integer(nested.initial_parent_origin_j) || !in.integer(nested.cumulative_parent_shift_i) ||
        !in.integer(nested.cumulative_parent_shift_j) || !decode_grid(in, nested.child.grid) ||
        !in.floating(nested.child.time_s) || !in.floating(nested.child.coriolis_f) ||
        !in.floating(nested.child.domain_frame.earth_origin_at_t0_m.x_m) ||
        !in.floating(nested.child.domain_frame.earth_origin_at_t0_m.y_m) ||
        !in.floating(nested.child.domain_frame.velocity_ms.u_ms) ||
        !in.floating(nested.child.domain_frame.velocity_ms.v_ms) || !decode_double_vector(in, nested.child.base.rho0) ||
        !decode_double_vector(in, nested.child.base.p0) || !decode_double_vector(in, nested.child.base.qv0) ||
        !decode_double_vector(in, nested.child.base.theta0) || !decode_double_vector(in, nested.child.base.u0) ||
        !decode_double_vector(in, nested.child.base.v0) || !decode_fields(in, prognostic_fields(nested.child.fields)) ||
        !(section_version >= 3 ? decode_fields(in, tendency_fields(nested.child.tendencies))
                               : decode_fields(in, tendency_fields_v1(nested.child.tendencies))) ||
        !decode_surface_state(in, nested.child.surface) ||
        !decode_fields(in, diagnostic_fields(nested.child.diagnostics)) ||
        !decode_interface_source_ledger(in, nested.interface_source, section_version >= 2) ||
        (section_version >= 4 && (!decode_child_relaxation_report(in, nested.two_way_interval_source_begin, true) ||
                                  !decode_five_doubles(in, nested.two_way_child_interval_begin_inventory) ||
                                  (section_version >= 6 && (!in.floating(nested.two_way_dry_mass_storage_begin) ||
                                                            !in.floating(nested.two_way_theta_storage_begin))) ||
                                  !decode_two_way_feedback_report(in, nested.last_two_way_feedback) ||
                                  !in.unsigned_value(nested.two_way_feedback_application_count))))
    {
        return false;
    }
    std::uint32_t runtime_version = 0;
    if (!in.unsigned_value(runtime_version) ||
        (runtime_version != 39 && runtime_version != 40 && runtime_version != 41 && runtime_version != 42 &&
         runtime_version != 43 && runtime_version != 44 && runtime_version != 45 && runtime_version != 46 &&
         runtime_version != 47 && runtime_version != 48 && runtime_version != 49 && runtime_version != 50 && runtime_version != 51 && runtime_version != 52 && runtime_version != 53 && runtime_version != 54 && runtime_version != 55))
    {
        error = "unsupported child level-runtime checkpoint version";
        return false;
    }
    SimulationStorageState runtime_shell(nested.child.grid);
    runtime_shell.configuration              = state.configuration;
    runtime_shell.configuration.domain_frame = nested.child.domain_frame;
    runtime_shell.time_s                     = nested.child.time_s;
    if (!decode_runtime(in, runtime_shell, runtime_version, error))
        return false;
    nested.child.runtime = std::move(static_cast<AtmosphericLevelRuntimeState&>(runtime_shell.runtime));
    if (section_version >= 7 && (!in.integer(nested.cumulative_parent_recycling_shift_i) ||
                                 !in.integer(nested.cumulative_parent_recycling_shift_j)))
        return false;
    if (section_version >= 8)
    {
        std::uint64_t             active_terrain_size          = 0;
        std::uint64_t             trailing_active_terrain_size = 0;
        std::vector<std::uint8_t> active_terrain_bytes;
        if (!in.unsigned_value(active_terrain_size) || active_terrain_size > wire::max_file_bytes ||
            active_terrain_size > in.remaining() ||
            in.remaining() - static_cast<std::size_t>(active_terrain_size) < sizeof(trailing_active_terrain_size) ||
            !in.raw(static_cast<std::size_t>(active_terrain_size), active_terrain_bytes) ||
            !in.unsigned_value(trailing_active_terrain_size) || trailing_active_terrain_size != active_terrain_size)
        {
            error = "invalid active-terrain checkpoint payload extent";
            return false;
        }
        Decoder    active_in(active_terrain_bytes.data(), active_terrain_bytes.size());
        const auto parent_physical =
            physical_scalar_domain(state.grid.nr, state.grid.nth, state.grid.nz, StaggerType::CGrid);
        const auto parent_grid_origin = state.configuration.domain_frame.earth_origin_at(state.time_s);
        const tmv::domain::HorizontalPoint parent_physical_origin{
            parent_grid_origin.x_m + parent_physical.i_begin * state.grid.dx,
            parent_grid_origin.y_m + parent_physical.j_begin * state.grid.dy};
        if (!decode_active_terrain_checkpoint(active_in,
                                              nested.active_terrain,
                                              nested.relationship,
                                              state.runtime.terrain.config,
                                              parent_physical_origin,
                                              state.time_s,
                                              nested.child.time_s,
                                              error) ||
            !active_in.finished())
        {
            if (error.empty())
                error = active_in.error().empty() ? "invalid active-terrain checkpoint payload" : active_in.error();
            return false;
        }
    }
    if (nested.active_terrain.ready)
    {
        if (!same_terrain_config(nested.active_terrain.geometry.config, nested.child.runtime.terrain.config))
        {
            error = "active-terrain checkpoint differs from child terrain ownership";
            return false;
        }
        nested.child.runtime.terrain.topography = nested.active_terrain.geometry.child_storage.topography;
        nested.child.runtime.terrain.metrics    = nested.active_terrain.geometry.child_storage.metrics;
    }
    if (section_version < 6)
    {
        nested.two_way_dry_mass_storage_begin =
            nested.child.runtime.time_stepping.cumulative_boundary_mass_closure_residual_kg;
        nested.two_way_theta_storage_begin =
            nested.child.runtime.advection.cumulative_thermodynamic_boundary_reports[tmv::theta_boundary_report_index]
                .closure_residual;
    }
    if (section_version == 1)
    {
        const bool parent_private = state.runtime.microphysics.scheme &&
                                    !state.runtime.microphysics.scheme->prognostic_scalars_for_transport().empty();
        const bool child_private =
            nested.child.runtime.microphysics.scheme &&
            !nested.child.runtime.microphysics.scheme->prognostic_scalars_for_transport().empty();
        if (parent_private || child_private)
        {
            error = "legacy active nesting checkpoint lacks private microphysics interface provenance";
            return false;
        }
    }
    prepare_nested_child_process_grid_metrics(nested);
    state.nested_execution = std::move(nested);
    prepare_nested_child_process_grid_metrics(state.nested_execution);
    if (synthesize_legacy_selection)
    {
        const auto& relationship                = state.nested_execution.relationship;
        auto&       selection                   = state.configuration.nesting;
        selection.enabled                       = true;
        selection.parent_ratio_x                = relationship.parent_ratio_x;
        selection.parent_ratio_y                = relationship.parent_ratio_y;
        selection.parent_time_step_ratio        = relationship.parent_time_step_ratio;
        selection.child_origin_parent_cell_i    = state.nested_execution.initial_parent_origin_i;
        selection.child_origin_parent_cell_j    = state.nested_execution.initial_parent_origin_j;
        selection.child_cells_x                 = relationship.child_cells_x;
        selection.child_cells_y                 = relationship.child_cells_y;
        selection.relaxation_width_child_cells  = relationship.relaxation_width_child_cells;
        selection.boundary_stencil_parent_cells = relationship.boundary_stencil_parent_cells;
    }
    return valid_synchronized_nested_checkpoint(state, error);
}

} // namespace

void encode_auxiliary_section(Encoder& out, const SimulationStorageState& state)
{
    encode_double_vector(out, state.base.rho0);
    encode_double_vector(out, state.base.p0);
    encode_double_vector(out, state.base.qv0);
    encode_double_vector(out, state.base.theta0);
    encode_double_vector(out, state.base.u0);
    encode_double_vector(out, state.base.v0);
    encode_float_vector(out, state.surface.ustar);
    encode_float_vector(out, state.surface.tau_u);
    encode_float_vector(out, state.surface.tau_v);
    encode_float_vector(out, state.surface.sensible_heat_flux);
    encode_float_vector(out, state.surface.latent_heat_flux);
    encode_float_vector(out, state.surface.moisture_flux);
    encode_float_vector(out, state.surface.pbl_height);
    encode_float_vector(out, state.surface.precipitation_rate_mm_h);
    encode_float_vector(out, state.surface.accumulated_rainfall_mm);
    encode_float_vector(out, state.surface.accumulated_rainfall_residual_mm);
    out.floating(state.surface.accumulated_rainfall_last_update_s);
    out.boolean(state.nested.config.enabled);
    out.integer(state.nested.config.center_i);
    out.integer(state.nested.config.center_j);
    out.integer(state.nested.config.center_k);
    out.integer(state.nested.config.nest_size_r);
    out.integer(state.nested.config.nest_size_th);
    out.integer(state.nested.config.nest_size_z);
    out.floating(state.nested.config.refinement);
    encode_fields(out, nested_fields(state.nested));
    encode_fields(out, diagnostic_fields(state.diagnostics));
}

bool build_simulation_sections(const SimulationStorageState& state,
                               const tmv::TrajectorySet*     trajectories,
                               std::vector<Section>&         sections,
                               std::string&                  error,
                               bool                          validation_only = false)
{
    error.clear();
    sections.clear();
    if (state.nested_execution.child_execution_active)
    {
        error = "cannot checkpoint while the nested child is process-bound";
        return false;
    }
    const bool nested_state_present =
        state.nested_execution.relationship_ready || state.nested_execution.reference_state_ready ||
        state.nested_execution.physical_primary_state_ready || state.nested_execution.parent_boundary_state_ready;
    if (nested_state_present && !valid_synchronized_nested_checkpoint(state, error))
        return false;
    if (!valid_tracer_roundoff_checkpoint_runtime(state.runtime.advection))
    {
        error = "cannot checkpoint invalid passive-tracer roundoff provenance";
        return false;
    }
    if (!valid_terrain_shortwave_runtime_contract(state.runtime.radiation))
    {
        const auto&  radiation = state.runtime.radiation;
        const double accounted = radiation.held_shortwave_atmospheric_absorbed_power_w +
                                 radiation.held_surface_absorbed_power_w + radiation.held_top_escape_power_w +
                                 radiation.held_lateral_escape_power_w + radiation.held_unresolved_reflected_power_w;
        std::ostringstream detail;
        detail << "cannot checkpoint invalid mapped-terrain shortwave contract"
               << " (scheme=" << radiation.config.scheme_id
               << ", quadrature=" << radiation.config.terrain_sw_cosine_nodes << 'x'
               << radiation.config.terrain_sw_azimuth_nodes
               << ", bounces=" << radiation.config.terrain_sw_maximum_bounces
               << ", converged=" << (radiation.held_reflection_converged ? 1 : 0)
               << ", incoming_w=" << radiation.held_shortwave_incoming_power_w << ", accounted_w=" << accounted
               << ", unresolved_w=" << radiation.held_unresolved_reflected_power_w << ')';
        error = detail.str();
        return false;
    }
    if (!valid_terrain_placement_checkpoint(state))
    {
        error = "cannot checkpoint invalid Earth-fixed terrain identity";
        return false;
    }
    if (!valid_storm_tracking_config(state.configuration.storm_tracking))
    {
        error = "cannot checkpoint invalid storm tracking configuration";
        return false;
    }
    if (!valid_domain_motion_config(state.configuration))
    {
        error = "cannot checkpoint invalid domain-motion frame configuration";
        return false;
    }
    if (state.configuration.microphysics_scheme == "milbrandt" &&
        state.configuration.microphysics_config.milbrandt.triple_moment)
    {
        error = "cannot checkpoint active Milbrandt-Yau with the obsolete "
                "triple_moment control enabled";
        return false;
    }
    const auto& thompson = state.configuration.microphysics_config.thompson;
    if (!std::isfinite(thompson.water_friendly_aerosol_surface_m3) ||
        thompson.water_friendly_aerosol_surface_m3 <= 50.0e6 ||
        !std::isfinite(thompson.ice_friendly_aerosol_surface_m3) || thompson.ice_friendly_aerosol_surface_m3 <= 0.5e6)
    {
        error = "cannot checkpoint invalid Thompson aerosol profile";
        return false;
    }
    if (!tmv::diagnostics::valid_storm_continuity_state(state.runtime.storm_tracking))
    {
        error = "cannot checkpoint invalid storm continuity state";
        return false;
    }
    if (!valid_storm_selected_partition(state.runtime) || !valid_storm_partition_authority(state))
    {
        error = "cannot checkpoint invalid exact storm partition authority";
        return false;
    }
    if (!tmv::domain::valid_storm_window_movement_state(state.runtime.storm_window_movement))
    {
        error = "cannot checkpoint invalid storm-window movement authority state";
        return false;
    }
    const auto& provenance = state.runtime.domain_recycling_provenance;
    if (!tmv::domain::valid_domain_recycling_transaction_provenance(provenance) ||
        !private_recycling_owners_match(state.runtime.microphysics, provenance))
    {
        error = "cannot checkpoint invalid D5 transaction provenance";
        return false;
    }
    if (provenance.available &&
        (provenance.nr != state.grid.nr || provenance.nth != state.grid.nth || provenance.nz != state.grid.nz ||
         provenance.simulation_time_s > state.time_s ||
         provenance.post_shift_origin_t0_m.x_m != state.configuration.domain_frame.earth_origin_at_t0_m.x_m ||
         provenance.post_shift_origin_t0_m.y_m != state.configuration.domain_frame.earth_origin_at_t0_m.y_m ||
         provenance.frame_velocity_ms.u_ms != state.configuration.domain_frame.velocity_ms.u_ms ||
         provenance.frame_velocity_ms.v_ms != state.configuration.domain_frame.velocity_ms.v_ms))
    {
        error = "cannot checkpoint D5 provenance inconsistent with live grid/frame/time";
        return false;
    }
    if (state.runtime.storm_tracking.motion_available &&
        (state.runtime.storm_tracking.sample_count <
             static_cast<std::size_t>(state.configuration.storm_tracking.minimum_continuity_samples) ||
         state.runtime.storm_tracking.speed_ms > state.configuration.storm_tracking.maximum_motion_speed_ms))
    {
        error = "cannot checkpoint storm continuity state that conflicts with its configured thresholds";
        return false;
    }
    if (validation_only)
        return true;
    Encoder metadata;
    metadata.string("completed_timestep");
    metadata.floating(state.time_s);
    metadata.floating(state.coriolis_f);
    sections.push_back({k_metadata_section, 1, wire::section_required, std::move(metadata.bytes)});

    Encoder configuration;
    encode_configuration(configuration, state.configuration);
    sections.push_back({k_configuration_section, 16, wire::section_required, std::move(configuration.bytes)});

    Encoder grid;
    encode_grid(grid, state.grid);
    sections.push_back({k_grid_section, 1, wire::section_required, std::move(grid.bytes)});

    Encoder fields;
    encode_fields(fields, prognostic_fields(state.fields));
    encode_fields(fields, tendency_fields(state.tendencies));
    sections.push_back({k_fields_section, 2, wire::section_required, std::move(fields.bytes)});

    Encoder auxiliary;
    encode_auxiliary_section(auxiliary, state);
    sections.push_back({k_auxiliary_section, 5, wire::section_required, std::move(auxiliary.bytes)});

    Encoder runtime;
    encode_runtime(runtime, state, error);
    if (!error.empty())
        return false;
    sections.push_back({k_runtime_section, 55, wire::section_required, std::move(runtime.bytes)});

    if (trajectories != nullptr)
    {
        Encoder trajectory;
        if (!encode_trajectories(trajectory, *trajectories, state.time_s, error))
            return false;
        if (!trajectory.good())
        {
            error = trajectory.error();
            return false;
        }
        // Optional by presence, required when present: readers must never
        // silently discard parcel provenance from a trajectory-enabled run.
        sections.push_back({k_trajectory_section, 2, wire::section_required, std::move(trajectory.bytes)});
    }

    Encoder nesting;
    if (nested_state_present)
    {
        if (!encode_nested_execution(nesting, state, error))
            return false;
        sections.push_back({k_nesting_section, 8, wire::section_required, std::move(nesting.bytes)});
    }

    for (const auto* encoder : {&metadata, &configuration, &grid, &fields, &auxiliary, &runtime})
    {
        if (!encoder->good())
        {
            error = encoder->error();
            return false;
        }
    }
    return true;
}

bool build_streaming_simulation_sections(const SimulationStorageState&        state,
                                         const tmv::TrajectorySet*            trajectories,
                                         std::vector<wire::StreamingSection>& sections,
                                         std::string&                         error)
{
    std::vector<Section> validation;
    if (!build_simulation_sections(state, trajectories, validation, error, true))
        return false;
    sections.clear();
    sections.reserve(trajectories == nullptr ? 7U : 8U);
    const auto add = [&](std::uint32_t id, std::uint32_t version, wire::StreamingSection::encode_type encoder)
    { sections.push_back({id, version, wire::section_required, std::move(encoder)}); };
    add(k_metadata_section,
        1,
        [&state](Encoder& out, std::string&)
        {
            out.string("completed_timestep");
            out.floating(state.time_s);
            out.floating(state.coriolis_f);
            return out.good();
        });
    add(k_configuration_section,
        16,
        [&state](Encoder& out, std::string&)
        {
            encode_configuration(out, state.configuration);
            return out.good();
        });
    add(k_grid_section,
        1,
        [&state](Encoder& out, std::string&)
        {
            encode_grid(out, state.grid);
            return out.good();
        });
    add(k_fields_section,
        2,
        [&state](Encoder& out, std::string&)
        {
            encode_fields(out, prognostic_fields(state.fields));
            encode_fields(out, tendency_fields(state.tendencies));
            return out.good();
        });
    add(k_auxiliary_section,
        5,
        [&state](Encoder& out, std::string&)
        {
            encode_auxiliary_section(out, state);
            return out.good();
        });
    add(k_runtime_section,
        55,
        [&state](Encoder& out, std::string& section_error)
        {
            encode_runtime(out, state, section_error);
            return section_error.empty() && out.good();
        });
    if (trajectories != nullptr)
    {
        add(k_trajectory_section,
            2,
            [&state, trajectories](Encoder& out, std::string& section_error)
            { return encode_trajectories(out, *trajectories, state.time_s, section_error) && out.good(); });
    }
    const bool nested_state_present =
        state.nested_execution.relationship_ready || state.nested_execution.reference_state_ready ||
        state.nested_execution.physical_primary_state_ready || state.nested_execution.parent_boundary_state_ready;
    if (nested_state_present)
    {
        add(k_nesting_section,
            8,
            [&state](Encoder& out, std::string& section_error)
            { return encode_nested_execution(out, state, section_error) && out.good(); });
    }
    return true;
}

bool serialize_simulation_storage(const SimulationStorageState& state,
                                  std::vector<std::uint8_t>&    bytes,
                                  std::string&                  error)
{
    return serialize_simulation_storage(state, nullptr, bytes, error);
}

bool serialize_simulation_storage(const SimulationStorageState& state,
                                  const tmv::TrajectorySet*     trajectories,
                                  std::vector<std::uint8_t>&    bytes,
                                  std::string&                  error)
{
    std::vector<Section> sections;
    if (!build_simulation_sections(state, trajectories, sections, error))
        return false;
    return wire::wrap_sections(k_magic, simulation_checkpoint_version, sections, bytes, error);
}

bool simulation_checkpoint_compatible(const SimulationStorageState& requested,
                                      const SimulationStorageState& checkpoint,
                                      std::string&                  error)
{
    if (requested.runtime.compute.vertical_flux_template_id != checkpoint.runtime.compute.vertical_flux_template_id)
    {
        error = "checkpoint vertical-flux execution selection differs from the requested run";
        return false;
    }

    error.clear();
    RuntimeConfigurationState checkpoint_configuration_for_match = checkpoint.configuration;
    // Compute state-commit ownership is selected by the target branch just
    // like its backend and memory topology.  These switches choose where an
    // otherwise identical tendency is committed; they do not select a
    // different equation, scheme, or physical parameter.  Normalize only
    // this backend-local tuple while retaining the exact scientific match for
    // every other serialized configuration byte.
    checkpoint_configuration_for_match.gpu_apply_microphysics = requested.configuration.gpu_apply_microphysics;
    checkpoint_configuration_for_match.gpu_apply_slow         = requested.configuration.gpu_apply_slow;
    checkpoint_configuration_for_match.gpu_apply_sgs          = requested.configuration.gpu_apply_sgs;
    // Execution mode controls admission, not the serialized equations. Permit
    // only promotion of an otherwise exact exploratory checkpoint into a
    // currently admitted qualified request; the qualified request has already
    // passed the ordinary public science gate before compatibility is checked.
    adopt_checkpoint_branch_configuration(
        requested.configuration, checkpoint_configuration_for_match);
    const auto& provenance                                    = checkpoint.runtime.domain_recycling_provenance;
    if (provenance.available)
    {
        if (!tmv::domain::valid_domain_recycling_transaction_provenance(provenance) ||
            !private_recycling_owners_match(checkpoint.runtime.microphysics, provenance) ||
            provenance.post_shift_origin_t0_m.x_m != checkpoint.configuration.domain_frame.earth_origin_at_t0_m.x_m ||
            provenance.post_shift_origin_t0_m.y_m != checkpoint.configuration.domain_frame.earth_origin_at_t0_m.y_m)
        {
            error = "checkpoint has invalid D5 frame provenance";
            return false;
        }
        // Earth origin is mutable recycling state, not a different scientific YAML
        // selection. It may differ only when the checkpoint carries the exact
        // committed displacement that produced it. Velocity and every other
        // configuration byte remain part of the strict match below.
        checkpoint_configuration_for_match.domain_frame.earth_origin_at_t0_m =
            requested.configuration.domain_frame.earth_origin_at_t0_m;
    }
    Encoder requested_configuration;
    Encoder checkpoint_configuration;
    encode_configuration(requested_configuration, requested.configuration);
    encode_configuration(checkpoint_configuration, checkpoint_configuration_for_match);
    if (!requested_configuration.good() || !checkpoint_configuration.good())
    {
        error = "could not encode checkpoint compatibility configuration";
        return false;
    }
    if (requested_configuration.bytes != checkpoint_configuration.bytes)
    {
        const std::size_t shared = std::min(
            requested_configuration.bytes.size(),
            checkpoint_configuration.bytes.size());
        std::size_t first_difference = 0;
        while (first_difference < shared &&
               requested_configuration.bytes[first_difference] ==
                   checkpoint_configuration.bytes[first_difference])
        {
            ++first_difference;
        }
        error = "checkpoint scientific configuration does not exactly match "
                "the requested config (first encoded difference at byte " +
            std::to_string(first_difference) + ")";
        return false;
    }

    const auto encode_microphysics_configuration = [](Encoder& out, const SimulationStorageState& state)
    {
        const auto& micro = state.configuration.microphysics_config;
        out.floating(micro.shared.cloud_water_threshold_kgkg);
        out.floating(micro.shared.autoconversion_rate_s_inv);
        out.floating(micro.kessler.accretion_coefficient);
        out.floating(micro.kessler.evaporation_efficiency);
        out.floating(micro.lin.accretion_coefficient);
        out.floating(micro.lin.evaporation_rate_s_inv);
        out.floating(micro.thompson.evaporation_rate_s_inv);
        out.floating(micro.thompson.water_friendly_aerosol_surface_m3);
        out.floating(micro.thompson.ice_friendly_aerosol_surface_m3);
        out.floating(micro.milbrandt.evaporation_rate_s_inv);
        out.boolean(micro.milbrandt.triple_moment);
        out.boolean(micro.milbrandt.hail_processes);
    };
    const auto encode_boundary_configuration = [](Encoder& out, const SimulationStorageState& state)
    {
        const auto& pbl = state.runtime.boundary_layer.config;
        out.string(pbl.scheme_id);
        out.floating(pbl.dt_pbl);
        out.boolean(pbl.apply_surface_fluxes);
        out.string(pbl.surface_layer_id);
        out.boolean(pbl.clear_sky_only);
        out.boolean(pbl.enable_tke);
        out.string(pbl.pbl_top_method);
        out.floating(pbl.pbl_max_height);
        out.floating(pbl.min_ustar);
        const auto& surface = state.runtime.boundary_layer.surface_config;
        out.floating(surface.z0m);
        out.floating(surface.z0h);
        out.floating(surface.albedo);
        out.floating(surface.emissivity);
        out.floating(surface.Tsfc);
        out.floating(surface.qsfc);
    };
    const auto same_module_configuration = [&](const char* label, const auto& encode)
    {
        Encoder requested_module;
        Encoder checkpoint_module;
        encode(requested_module, requested);
        encode(checkpoint_module, checkpoint);
        if (!requested_module.good() || !checkpoint_module.good() || requested_module.bytes != checkpoint_module.bytes)
        {
            error = std::string("checkpoint ") + label + " configuration does not exactly match the requested config";
            return false;
        }
        return true;
    };
    const auto same_radar_product_selection = [&]()
    {
        const auto& expected = requested.runtime.radar.product_selection;
        const auto& actual   = checkpoint.runtime.radar.product_selection;
        const auto  inactive = [](const RadarProductSelection& selection)
        {
            return !selection.reflectivity && !selection.radial_velocity && !selection.zdr && !selection.kdp &&
                   !selection.rhohv;
        };
        if (inactive(expected) && inactive(actual))
            return true;
        const auto mismatch = [&](bool equal, const char* field)
        {
            if (equal)
                return false;
            error = std::string("checkpoint radar configuration field ") + field +
                    " does not exactly match the requested config";
            return true;
        };
        return !(mismatch(expected.reflectivity == actual.reflectivity, "reflectivity") ||
                 mismatch(expected.radial_velocity == actual.radial_velocity, "radial_velocity") ||
                 mismatch(expected.zdr == actual.zdr, "zdr") || mismatch(expected.kdp == actual.kdp, "kdp") ||
                 mismatch(expected.rhohv == actual.rhohv, "rhohv") ||
                 mismatch(expected.reflectivity_tier == actual.reflectivity_tier, "reflectivity_tier") ||
                 mismatch(expected.velocity_sampling == actual.velocity_sampling, "velocity_sampling") ||
                 mismatch(expected.zdr_tier == actual.zdr_tier, "zdr_tier") ||
                 mismatch(expected.radar_x == actual.radar_x, "radar_x") ||
                 mismatch(expected.radar_y == actual.radar_y, "radar_y") ||
                 mismatch(expected.radar_z == actual.radar_z, "radar_z") ||
                 mismatch(expected.apply_scatterer_correction == actual.apply_scatterer_correction,
                          "apply_scatterer_correction"));
    };
    if (!same_module_configuration("microphysics", encode_microphysics_configuration) ||
        !same_module_configuration("radiation",
                                   [](Encoder& out, const SimulationStorageState& state)
                                   {
                                       encode_radiation_config(out, state.runtime.radiation.config);
                                       encode_rrtmg_config(out, state.runtime.radiation.config);
                                       encode_simple_grey_config(out, state.runtime.radiation.config);
                                   }) ||
        !same_module_configuration("turbulence",
                                   [](Encoder& out, const SimulationStorageState& state)
                                   {
                                       encode_turbulence_config(out, state.runtime.turbulence.config);
                                       out.floating(state.runtime.turbulence.config.user_filter_width_m);
                                   }) ||
        !same_module_configuration("boundary-layer", encode_boundary_configuration) ||
        !same_module_configuration("chaos",
                                   [](Encoder& out, const SimulationStorageState& state)
                                   { encode_chaos_config(out, state.runtime.chaos.config); }) ||
        !same_module_configuration("terrain",
                                   [](Encoder& out, const SimulationStorageState& state)
                                   {
                                       encode_terrain_config(out, state.runtime.terrain.config);
                                       encode_terrain_placement_config(out, state.runtime.terrain.config);
                                       out.unsigned_value<std::uint8_t>(static_cast<std::uint8_t>(state.runtime.terrain.earth_fixed_placement.sampling));
                                   }) ||
        !same_radar_product_selection() ||
        !same_module_configuration("advection",
                                   [](Encoder& out, const SimulationStorageState& state)
                                   { encode_advection_config(out, state.runtime.advection.config); }) ||
        !same_module_configuration("diffusion",
                                   [](Encoder& out, const SimulationStorageState& state)
                                   {
                                       encode_diffusion_config(out, state.runtime.diffusion.config);
                                       encode_diffusion_profile_config(out, state.runtime.diffusion.config);
                                   }) ||
        !same_module_configuration("time-stepping",
                                   [](Encoder& out, const SimulationStorageState& state)
                                   {
                                       const auto& time = state.runtime.time_stepping.config;
                                       // Acoustic speed and minimum physical spacing are diagnosed
                                       // continuation state updated every step, not user compatibility
                                       // choices.
                                       out.string(time.scheme_id);
                                       out.boolean(time.split_acoustic);
                                       out.integer(time.n_acoustic_substeps);
                                       out.string(time.physics_splitting);
                                       out.floating(time.cfl_safety);
                                       out.boolean(time.adaptive_dt);
                                       out.floating(time.dt_min);
                                       out.floating(time.dt_max);
                                   }))
        return false;

    Encoder requested_grid;
    Encoder checkpoint_grid;
    // grid.dt is live continuation state. It can differ from the configured
    // large timestep after adaptive CFL selection or an exact endpoint clamp.
    // The immutable requested value is compared in RuntimeConfigurationState.
    GridStorageState requested_grid_contract  = requested.grid;
    GridStorageState checkpoint_grid_contract = checkpoint.grid;
    requested_grid_contract.dt                = 0.0;
    checkpoint_grid_contract.dt               = 0.0;
    encode_grid(requested_grid, requested_grid_contract);
    encode_grid(checkpoint_grid, checkpoint_grid_contract);
    if (!requested_grid.good() || !checkpoint_grid.good())
    {
        error = "could not encode checkpoint compatibility grid";
        return false;
    }
    if (requested_grid.bytes != checkpoint_grid.bytes)
    {
        error = "checkpoint grid/timestep contract does not exactly match the requested config";
        return false;
    }
    return true;
}

void adopt_checkpoint_branch_configuration(
    const RuntimeConfigurationState& requested,
    RuntimeConfigurationState& restored)
{
    restored.admission = requested.admission;
    if (restored.execution_mode ==
            tmv::init::config::ExecutionMode::Exploratory &&
        requested.execution_mode ==
            tmv::init::config::ExecutionMode::Qualified)
    {
        restored.execution_mode = tmv::init::config::ExecutionMode::Qualified;
    }
    if (restored.sounding_source.type ==
            tmv::init::SoundingSourceConfig::Type::ParametricCAPE &&
        requested.sounding_source.type ==
            tmv::init::SoundingSourceConfig::Type::ParametricCAPE)
    {
        restored.sounding_config = requested.sounding_config;
        restored.sounding_source.file = requested.sounding_source.file;
        restored.sounding_source.targets = requested.sounding_source.targets;
    }
}

static bool deserialize_validated_sections(const std::vector<wire::SectionView>& sections,
                                           SimulationStorageState& state,
                                           SimulationCheckpointAttachments& attachments,
                                           std::string& error)
{
    error.clear();
    SimulationCheckpointAttachments restored_attachments;
    const std::array<std::uint32_t, 6> required = {k_metadata_section,
                                                   k_configuration_section,
                                                   k_grid_section,
                                                   k_fields_section,
                                                   k_auxiliary_section,
                                                   k_runtime_section};
    for (std::uint32_t id : required)
    {
        if (std::none_of(sections.begin(), sections.end(), [&](const wire::SectionView& section) { return section.id == id; }))
        {
            error = "missing required simulation checkpoint section";
            return false;
        }
    }
    for (const auto& section : sections)
    {
        const bool known_optional = section.id == k_nesting_section || section.id == k_trajectory_section;
        if (std::find(required.begin(), required.end(), section.id) == required.end() && !known_optional &&
            (section.flags & wire::section_required) != 0)
        {
            error = "unknown required simulation checkpoint section";
            return false;
        }
    }

    SimulationStorageState restored;
    try
    {
        // Decode in dependency order even when a valid writer reorders sections.
        for (std::uint32_t required_id : required)
        {
            const auto found = std::find_if(
                sections.begin(), sections.end(), [&](const wire::SectionView& section) { return section.id == required_id; });
            const wire::SectionView& section = *found;
            Decoder        in(section.payload.data(), section.payload.size());
            const bool     supported_section =
                section.id == k_auxiliary_section
                    ? section.version == 1 || section.version == 3 || section.version == 4 || section.version == 5
                : section.id == k_configuration_section
                    ? section.version == 1 || section.version == 2 || section.version == 3 || section.version == 4 ||
                          section.version == 5 || section.version == 6 || section.version == 7 ||
                          section.version == 8 || section.version == 9 || section.version == 10 ||
                          section.version == 11 || section.version == 12 || section.version == 13 ||
                          section.version == 14 || section.version == 15 || section.version == 16
                : section.id == k_fields_section ? section.version == 1 || section.version == 2
                : section.id == k_runtime_section
                    ? section.version == 1 || section.version == 2 || section.version == 3 || section.version == 4 ||
                          section.version == 5 || section.version == 6 || section.version == 7 ||
                          section.version == 8 || section.version == 9 || section.version == 10 ||
                          section.version == 11 || section.version == 12 || section.version == 13 ||
                          section.version == 14 || section.version == 15 || section.version == 16 ||
                          section.version == 17 || section.version == 18 || section.version == 19 ||
                          section.version == 20 || section.version == 21 || section.version == 22 ||
                          section.version == 23 || section.version == 24 || section.version == 25 ||
                          section.version == 26 || section.version == 27 || section.version == 28 ||
                          section.version == 29 || section.version == 30 || section.version == 31 ||
                          section.version == 32 || section.version == 33 || section.version == 34 ||
                          section.version == 35 || section.version == 36 || section.version == 37 ||
                          section.version == 38 || section.version == 39 || section.version == 40 ||
                          section.version == 41 || section.version == 42 || section.version == 43 ||
                          section.version == 44 || section.version == 45 || section.version == 46 ||
                          section.version == 47 || section.version == 48 || section.version == 49 ||
                          section.version == 50 || section.version == 51 || section.version == 52 || section.version == 53 || section.version == 54 || section.version == 55
                : section.id == k_trajectory_section ? section.version == 1 || section.version == 2
                                                     : section.version == 1;
            if (!supported_section)
            {
                error = "unsupported simulation checkpoint section version";
                return false;
            }
            bool ok = true;
            switch (section.id)
            {
            case k_metadata_section:
            {
                std::string boundary;
                ok = in.string(boundary) && boundary == "completed_timestep" && in.floating(restored.time_s) &&
                     in.floating(restored.coriolis_f);
                break;
            }
            case k_configuration_section:
                ok = decode_configuration(in, restored.configuration, section.version);
                break;
            case k_grid_section:
                ok                     = decode_grid(in, restored.grid);
                restored.nested.config = NestedGridConfig(restored.grid.nr, restored.grid.nth, restored.grid.nz);
                break;
            case k_fields_section:
                ok = decode_fields(in, prognostic_fields(restored.fields)) &&
                     (section.version >= 2 ? decode_fields(in, tendency_fields(restored.tendencies))
                                           : decode_fields(in, tendency_fields_v1(restored.tendencies)));
                break;
            case k_auxiliary_section:
            {
                auto& n = restored.nested.config;
                ok = decode_double_vector(in, restored.base.rho0) && decode_double_vector(in, restored.base.p0) &&
                     decode_double_vector(in, restored.base.qv0) && decode_double_vector(in, restored.base.theta0) &&
                     decode_double_vector(in, restored.base.u0) && decode_double_vector(in, restored.base.v0) &&
                     decode_float_vector(in, restored.surface.ustar) &&
                     decode_float_vector(in, restored.surface.tau_u) &&
                     decode_float_vector(in, restored.surface.tau_v) &&
                     decode_float_vector(in, restored.surface.sensible_heat_flux) &&
                     decode_float_vector(in, restored.surface.latent_heat_flux) &&
                     decode_float_vector(in, restored.surface.moisture_flux) &&
                     decode_float_vector(in, restored.surface.pbl_height);
                if (ok && section.version >= 3)
                {
                    if (section.version >= 4)
                    {
                        ok = decode_float_vector(in, restored.surface.precipitation_rate_mm_h);
                    }
                    ok = ok && decode_float_vector(in, restored.surface.accumulated_rainfall_mm) &&
                         (section.version < 5 ||
                          decode_float_vector(in, restored.surface.accumulated_rainfall_residual_mm)) &&
                         in.floating(restored.surface.accumulated_rainfall_last_update_s);
                    if (ok && section.version == 3 &&
                        restored.surface.accumulated_rainfall_mm.size() ==
                            static_cast<std::size_t>(restored.grid.nr) * static_cast<std::size_t>(restored.grid.nth))
                    {
                        // Auxiliary v3 accumulated output tranches in
                        // `(j * nr + i)` order. Runtime/recycling surface ownership
                        // is canonical `(i * nth + j)` beginning in v4.
                        const std::vector<float> legacy = restored.surface.accumulated_rainfall_mm;
                        for (int i = 0; i < restored.grid.nr; ++i)
                            for (int j = 0; j < restored.grid.nth; ++j)
                                restored.surface
                                    .accumulated_rainfall_mm[static_cast<std::size_t>(i) *
                                                                 static_cast<std::size_t>(restored.grid.nth) +
                                                             static_cast<std::size_t>(j)] =
                                    legacy[static_cast<std::size_t>(j) * static_cast<std::size_t>(restored.grid.nr) +
                                           static_cast<std::size_t>(i)];
                    }
                }
                ok = ok && in.boolean(n.enabled) && in.integer(n.center_i) && in.integer(n.center_j) &&
                     in.integer(n.center_k) && in.integer(n.nest_size_r) && in.integer(n.nest_size_th) &&
                     in.integer(n.nest_size_z) && in.floating(n.refinement);
                if (ok)
                {
                    ok = section.version == 1 ? decode_fields(in, nested_fields_v1(restored.nested))
                                              : decode_fields(in, nested_fields(restored.nested));
                }
                if (ok)
                    ok = decode_fields(in, diagnostic_fields(restored.diagnostics));
                break;
            }
            case k_runtime_section:
                ok = decode_runtime(in, restored, section.version, error);
                break;
            default:
                break;
            }
            if (!ok || !in.finished())
            {
                if (error.empty())
                    error = in.error().empty() ? "invalid simulation checkpoint section" : in.error();
                return false;
            }
        }

        const auto nesting = std::find_if(
            sections.begin(), sections.end(), [](const wire::SectionView& section) { return section.id == k_nesting_section; });
        if (nesting != sections.end())
        {
            if ((nesting->flags & wire::section_required) == 0)
            {
                error = "active nesting checkpoint section must be required";
                return false;
            }
            Decoder    in(nesting->payload.data(), nesting->payload.size());
            const auto configuration =
                std::find_if(sections.begin(),
                             sections.end(),
                             [](const wire::SectionView& section) { return section.id == k_configuration_section; });
            if (!decode_nested_execution(in, restored, nesting->version, configuration->version < 13, error) ||
                !in.finished())
            {
                if (error.empty())
                    error = in.error().empty() ? "invalid active nesting checkpoint section" : in.error();
                return false;
            }
        }

        const auto trajectory = std::find_if(sections.begin(),
                                             sections.end(),
                                             [](const wire::SectionView& section) { return section.id == k_trajectory_section; });
        if (trajectory != sections.end())
        {
            if ((trajectory->flags & wire::section_required) == 0 ||
                (trajectory->version != 1 && trajectory->version != 2))
            {
                error = "trajectory checkpoint section must be required version 1 or 2";
                return false;
            }
            Decoder                        in(trajectory->payload.data(), trajectory->payload.size());
            tmv::TrajectoryCheckpointState restored_trajectories;
            if (!decode_trajectories(in, restored_trajectories, restored.time_s, trajectory->version, error) ||
                !in.finished())
            {
                if (error.empty())
                    error = in.error().empty() ? "invalid trajectory checkpoint section" : in.error();
                return false;
            }
            restored_attachments.trajectories = std::move(restored_trajectories);
        }
    }
    catch (const std::exception& exception)
    {
        error = std::string("cannot reconstruct simulation checkpoint: ") + exception.what();
        return false;
    }
    if (!valid_storm_partition_authority(restored))
    {
        error = "restored exact storm partition does not match its tracking-grid authority";
        return false;
    }
    static_assert(std::is_nothrow_move_assignable_v<SimulationCheckpointAttachments>);
    state = std::move(restored);
    attachments = std::move(restored_attachments);
    return true;
}

bool deserialize_simulation_storage(const std::vector<std::uint8_t>& bytes,
                                    SimulationStorageState& state,
                                    SimulationCheckpointAttachments& attachments,
                                    std::string& error)
{
    std::vector<wire::SectionView> sections;
    if (!wire::unwrap_section_views(bytes, k_magic, simulation_checkpoint_version, sections, error, "simulation"))
        return false;
    return deserialize_validated_sections(sections, state, attachments, error);
}

bool deserialize_simulation_storage(const std::vector<std::uint8_t>& bytes,
                                    SimulationStorageState&          state,
                                    std::string&                     error)
{
    SimulationCheckpointAttachments attachments;
    return deserialize_simulation_storage(bytes, state, attachments, error);
}

bool write_simulation_checkpoint(const std::filesystem::path&  path,
                                 const SimulationStorageState& state,
                                 std::string&                  error)
{
    return write_simulation_checkpoint(path, state, nullptr, error);
}

bool write_simulation_checkpoint(const std::filesystem::path&  path,
                                 const SimulationStorageState& state,
                                 const tmv::TrajectorySet*     trajectories,
                                 std::string&                  error)
{
    SimulationCheckpointIntegrity ignored;
    return write_simulation_checkpoint(path, state, trajectories, ignored, error);
}

bool write_simulation_checkpoint(const std::filesystem::path&   path,
                                 const SimulationStorageState&  state,
                                 const tmv::TrajectorySet*      trajectories,
                                 SimulationCheckpointIntegrity& integrity,
                                 std::string&                   error,
                                 std::uint64_t                  minimum_free_bytes_after_publication)
{
    std::vector<wire::StreamingSection> sections;
    if (!build_streaming_simulation_sections(state, trajectories, sections, error))
        return false;
    wire::PublishedFileIntegrity published;
    if (!wire::write_streaming_sections_atomic(path,
                                               k_magic,
                                               simulation_checkpoint_version,
                                               sections,
                                               error,
                                               "simulation",
                                               &published,
                                               minimum_free_bytes_after_publication))
        return false;
    std::ostringstream fnv;
    fnv << std::hex << std::setfill('0') << std::setw(16) << published.fnv1a64;
    integrity.bytes   = published.bytes;
    integrity.fnv1a64 = fnv.str();
    integrity.sha256  = std::move(published.sha256);
    return true;
}

bool read_simulation_checkpoint(const std::filesystem::path& path, SimulationStorageState& state, std::string& error)
{
    SimulationCheckpointAttachments attachments;
    return read_simulation_checkpoint(path, state, attachments, error);
}

bool read_simulation_checkpoint(const std::filesystem::path&     path,
                                SimulationStorageState&          state,
                                SimulationCheckpointAttachments& attachments,
                                std::string&                     error)
{
    SimulationCheckpointIntegrity ignored;
    return read_simulation_checkpoint(path, state, attachments, ignored, error);
}

bool read_simulation_checkpoint(const std::filesystem::path&     path,
                                SimulationStorageState&          state,
                                SimulationCheckpointAttachments& attachments,
                                SimulationCheckpointIntegrity&   integrity,
                                std::string&                     error)
{
    std::vector<std::uint8_t> bytes;
    if (!wire::read_file(path, bytes, error, "simulation"))
        return false;
    std::vector<wire::SectionView> sections;
    wire::PublishedFileIntegrity published;
    if (!wire::unwrap_section_views(bytes, k_magic, simulation_checkpoint_version, sections, error, "simulation", &published))
        return false;
    std::ostringstream fnv;
    fnv << std::hex << std::setfill('0') << std::setw(16) << published.fnv1a64;
    SimulationCheckpointIntegrity candidate{published.bytes, fnv.str(), std::move(published.sha256)};
    if (!deserialize_validated_sections(sections, state, attachments, error))
        return false;
    static_assert(std::is_nothrow_move_assignable_v<SimulationCheckpointIntegrity>);
    integrity = std::move(candidate);
    return true;
}

} // namespace tmv::checkpoint
