#pragma once

#include "core/infra/nested_boundary_forcing.hpp"
#include "diagnostics/energy_budget.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

struct NestedPrivateScalarRelaxationReportV1
{
    static constexpr std::uint32_t schema_version = 1;
    static constexpr std::size_t maximum_scalars = 16;
    static constexpr std::size_t maximum_name_bytes = 96;

    std::size_t scalar_count = 0;
    std::array<std::array<char, maximum_name_bytes>, maximum_scalars> name{};
    std::array<std::uint8_t, maximum_scalars> storage_semantic{};
    std::array<bool, maximum_scalars> positive_definite{};
    std::array<double, maximum_scalars> target_carrier_change{};
    std::array<double, maximum_scalars> committed_carrier_change{};
    std::array<double, maximum_scalars> carrier_storage_residual{};
    std::array<std::size_t, maximum_scalars> forced_cells{};
};

struct NestedChildRelaxationReport
{
    tmv::domain::ConservedScalarRelaxationReport theta;
    tmv::domain::ConservedScalarRelaxationReport qv;
    tmv::domain::ConservedScalarRelaxationReport qc;
    tmv::domain::ConservedScalarRelaxationReport qr;
    tmv::domain::ConservedScalarRelaxationReport qi;
    tmv::domain::ConservedScalarRelaxationReport qs;
    tmv::domain::ConservedScalarRelaxationReport qg;
    tmv::domain::ConservedScalarRelaxationReport qh;
    tmv::domain::ConservedScalarRelaxationReport tracer;
    tmv::domain::ConservedScalarRelaxationReport tke;
    tmv::domain::ConservedMomentumRelaxationReport momentum;
    tmv::domain::PressureDiagnosisReport pressure;
    NestedPrivateScalarRelaxationReportV1 microphysics_private;
    struct InterfaceEnergyReportV1
    {
        static constexpr std::uint32_t schema_version = 1;
        static constexpr std::uint32_t energy_basis_version =
            MoistMixtureEnergyBudgetV2::basis_version;

        MoistMixtureEnergyBudgetV2 before;
        MoistMixtureEnergyBudgetV2 after;
        MoistMixtureEnergyBudgetV2 committed_change;
        double reservoir_sum_closure_residual_j = 0.0;
    } energy;
};

struct NestedInterfaceSourceLedgerV2
{
    static constexpr std::uint32_t schema_version = 2;

    bool available = false;
    std::uint64_t application_count = 0;
    NestedChildRelaxationReport last;
    NestedChildRelaxationReport cumulative;
};
