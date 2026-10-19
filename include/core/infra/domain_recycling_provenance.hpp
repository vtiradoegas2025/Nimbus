#pragma once

#include "core/infra/domain_motion.hpp"
#include "diagnostics/conservation_budget.hpp"
#include "diagnostics/energy_budget.hpp"
#include "diagnostics/momentum_budget.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace tmv::domain
{

template <std::size_t Capacity>
struct DomainRecyclingProvenanceText
{
    std::array<char, Capacity + 1> bytes{};
    std::uint16_t size = 0;

    [[nodiscard]] std::string_view view() const noexcept
    {
        return {bytes.data(), size};
    }
};

struct DomainRecyclingPrivateInventory
{
    struct Entry
    {
        DomainRecyclingProvenanceText<63> name;
        double inventory = 0;
    };
    std::array<Entry, 16> entries{};
    std::uint16_t count = 0;
};

bool valid_domain_recycling_private_inventory(const DomainRecyclingPrivateInventory& inventory) noexcept;

/**
 * Last committed recycling transaction, stored without heap ownership so its commit
 * can participate in the allocation-free/no-throw atmospheric transaction.
 * The swath manifest hash binds this state record to the exact durable
 * captured-precommit event; tile manifests bind that event to payload bytes.
 */
struct DomainRecyclingTransactionProvenance
{
    bool available = false;
    std::uint64_t sequence = 0;
    DomainRecyclingProvenanceText<128> transaction_id;
    DomainRecyclingProvenanceText<128> publication_generation;
    DomainRecyclingProvenanceText<16> storage_format;
    std::uint64_t event_manifest_bytes = 0;
    std::uint64_t event_manifest_fnv1a64 = 0;
    std::uint64_t previous_chain_fnv1a64 = 0;
    std::uint64_t committed_chain_fnv1a64 = 0;

    double simulation_time_s = 0.0;
    HorizontalPoint pre_shift_origin_t0_m;
    HorizontalPoint post_shift_origin_t0_m;
    HorizontalVelocity frame_velocity_ms;
    double dx_m = 0.0;
    double dy_m = 0.0;

    int nr = 0;
    int nth = 0;
    int nz = 0;
    int shift_i_cells = 0;
    int shift_j_cells = 0;
    int physical_i_begin = 0;
    int physical_i_end = 0;
    int physical_j_begin = 0;
    int physical_j_end = 0;
    int source_i_begin = 0;
    int source_i_end = 0;
    int source_j_begin = 0;
    int source_j_end = 0;

    ConservationBudget inventories;
    MomentumBudget momentum;
    MoistMixtureEnergyBudgetV2 moist_mixture_energy;
    double theta_storage_residual_energy_j = 0.0;
    // Runtime sections through v35 contain the retiring side only. New recycling
    // transactions bind both sides of the instantaneous control-volume
    // exchange; the flag keeps those legacy checkpoints readable without
    // pretending their historical record was complete.
    bool control_volume_exchange_complete = false;
    ConservationBudget admitted_inventories;
    MomentumBudget admitted_momentum;
    MoistMixtureEnergyBudgetV2 admitted_moist_mixture_energy;
    double admitted_theta_storage_residual_energy_j = 0.0;
    double retiring_fixed_earth_angular_momentum_z = 0.0;
    double admitted_fixed_earth_angular_momentum_z = 0.0;
    bool private_inventory_complete = false;
    DomainRecyclingPrivateInventory retiring_private_inventory;
    DomainRecyclingPrivateInventory admitted_private_inventory;
    std::uint64_t physical_columns = 0;
    std::uint64_t physical_cells = 0;

    DomainRecyclingProvenanceText<96> shift_algorithm;
    DomainRecyclingProvenanceText<96> frame_algorithm;
    DomainRecyclingProvenanceText<96> swath_algorithm;
    DomainRecyclingProvenanceText<96> scalar_ledger_basis;
    DomainRecyclingProvenanceText<96> momentum_ledger_basis;
    DomainRecyclingProvenanceText<96> energy_ledger_basis;
    DomainRecyclingProvenanceText<96> theta_residual_ledger_basis;
    DomainRecyclingProvenanceText<96> control_volume_exchange_basis;
};

bool valid_domain_recycling_transaction_provenance(
    const DomainRecyclingTransactionProvenance& provenance) noexcept;

std::uint64_t compute_domain_recycling_transaction_chain_fnv1a64(
    std::uint64_t previous_chain_fnv1a64,
    std::uint64_t sequence,
    std::uint64_t event_manifest_bytes,
    std::uint64_t event_manifest_fnv1a64,
    std::string_view transaction_id,
    std::string_view publication_generation) noexcept;

const char* domain_recycling_transaction_provenance_algorithm_id() noexcept;

} // namespace tmv::domain
