#pragma once

#include "microphysics/config.hpp"
#include "microphysics/microphysics_base.hpp"

#include <string>

struct SimulationStateView;

namespace tmv
{
extern MicrophysicsRuntimeState& global_microphysics_runtime_state;

struct MicrophysicsPerfSnapshot
{
    std::uint64_t sedimentation_substeps = 0;
    double pbl_reprojection_s = 0.0;
    double input_validation_s = 0.0;
    double sedimentation_limit_s = 0.0;
    double transaction_snapshot_s = 0.0;
    double scheme_s = 0.0;
    double chaos_s = 0.0;
    double column_admission_s = 0.0;
    double tendency_contract_s = 0.0;
    double validation_s = 0.0;
    double private_state_s = 0.0;
    double precipitation_s = 0.0;
    double projection_s = 0.0;
    double apply_s = 0.0;
    double finalize_s = 0.0;
};

void reset_microphysics_perf_stats() noexcept;
MicrophysicsPerfSnapshot microphysics_perf_snapshot() noexcept;

// Compatibility binding into the process-owned runtime.
extern std::unique_ptr<MicrophysicsScheme>& microphysics_scheme;

void initialize_microphysics(const std::string& scheme_name = "kessler");
void initialize_microphysics(const std::string& scheme_name,
                             const MicrophysicsConfig& config);
void initialize_microphysics(
    MicrophysicsRuntimeState& runtime,
    const std::string& scheme_name = "kessler");
void initialize_microphysics(MicrophysicsRuntimeState& runtime,
                             const std::string& scheme_name,
                             const MicrophysicsConfig& config);
void step_microphysics(const SimulationStateView& state, double dt_micro);
void step_microphysics(double dt_micro);
Field3D completed_cloud_evaporation_return(const SimulationStateView& state, double interval);
void invalidate_microphysics_reaction_snapshot(MicrophysicsRuntimeState& runtime) noexcept;
void open_thermodynamic_source_budget_csv(const std::string& path);
void close_thermodynamic_source_budget_csv();
void write_thermodynamic_source_budget_row(
    const SimulationStateView& state, double current_time_s, double dt_s);

const char* surface_precipitation_accumulation_algorithm_id() noexcept;

void invalidate_microphysics_transients_after_domain_recycling(
    const SimulationStateView& state);
const char* microphysics_recycling_invalidation_algorithm_id() noexcept;

const MicrophysicsWaterRepairReport& last_microphysics_water_repair_report();
const MicrophysicsWaterRepairReport& last_microphysics_water_repair_report(
    const SimulationStateView& state);
const MicrophysicsThermodynamicRepairReport&
last_microphysics_thermodynamic_repair_report();
const MicrophysicsThermodynamicRepairReport&
last_microphysics_thermodynamic_repair_report(const SimulationStateView& state);
}

using tmv::global_microphysics_runtime_state;
using tmv::microphysics_scheme;
using tmv::MicrophysicsPerfSnapshot;
using tmv::reset_microphysics_perf_stats;
using tmv::microphysics_perf_snapshot;
using tmv::initialize_microphysics;
using tmv::last_microphysics_water_repair_report;
using tmv::last_microphysics_thermodynamic_repair_report;
using tmv::step_microphysics;
using tmv::open_thermodynamic_source_budget_csv;
using tmv::close_thermodynamic_source_budget_csv;
using tmv::write_thermodynamic_source_budget_row;
