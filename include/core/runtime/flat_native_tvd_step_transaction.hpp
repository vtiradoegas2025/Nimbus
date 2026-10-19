#pragma once

#include "core/runtime/atmospheric_level.hpp"

#include <exception>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

namespace tmv::runtime
{
class FlatCgridStepTransaction
{
public:
    static_assert(std::is_nothrow_swappable_v<nimbus::parity::ParityRuntimeState>);
    static_assert(std::is_nothrow_swappable_v<tmv::diagnostics::StormContinuityState>);
    static_assert(std::is_nothrow_swappable_v<tmv::domain::StormWindowMovementState>);
    static_assert(std::is_nothrow_swappable_v<tmv::domain::DomainRecyclingTransactionProvenance>);

    explicit FlatCgridStepTransaction(SimulationStorageState& owner)
        : owner_(owner), prior_parity_(owner.runtime.parity),
          prior_storm_tracking_(owner.runtime.storm_tracking),
          prior_storm_window_movement_(owner.runtime.storm_window_movement),
          prior_storm_selected_partition_available_(owner.runtime.storm_selected_partition_available),
          prior_storm_selected_partition_nx_(owner.runtime.storm_selected_partition_nx),
          prior_storm_selected_partition_ny_(owner.runtime.storm_selected_partition_ny),
          prior_storm_selected_partition_(owner.runtime.storm_selected_partition),
          prior_domain_recycling_provenance_(owner.runtime.domain_recycling_provenance),
          parent_(active_),
          owns_snapshot_(!parent_ || &parent_->owner_ != &owner)
    {
        if (owns_snapshot_)
        {
            state_ownership::capture_level(prior_, owner_);
        }
        active_ = this;
    }

    FlatCgridStepTransaction(const FlatCgridStepTransaction&) = delete;
    FlatCgridStepTransaction& operator=(const FlatCgridStepTransaction&) = delete;

    ~FlatCgridStepTransaction() noexcept
    {
        if (!committed_ && owns_snapshot_)
        {
            auto& boundary = owner_.runtime.boundary_conditions;
            auto operator_owner = std::move(boundary.execution_operator);
            state_ownership::exchange_levels(owner_, prior_);
            boundary.execution_operator = std::move(operator_owner);
            using std::swap;
            swap(owner_.runtime.parity, prior_parity_);
            swap(owner_.runtime.storm_tracking, prior_storm_tracking_);
            swap(owner_.runtime.storm_window_movement, prior_storm_window_movement_);
            swap(owner_.runtime.storm_selected_partition_available, prior_storm_selected_partition_available_);
            swap(owner_.runtime.storm_selected_partition_nx, prior_storm_selected_partition_nx_);
            swap(owner_.runtime.storm_selected_partition_ny, prior_storm_selected_partition_ny_);
            swap(owner_.runtime.storm_selected_partition, prior_storm_selected_partition_);
            swap(owner_.runtime.domain_recycling_provenance, prior_domain_recycling_provenance_);
            try
            {
                make_simulation_state_view(owner_);
            }
            catch (...)
            {
                std::terminate();
            }
            if (owner_.runtime.compute.backend)
            {
                owner_.runtime.compute.backend->invalidate_all_resident_fields();
            }
        }
        active_ = parent_;
    }

    void commit() noexcept
    {
        committed_ = true;
    }

    static bool active_for(const SimulationStorageState& owner) noexcept
    {
        return active_ && &active_->owner_ == &owner;
    }

private:
    SimulationStorageState& owner_;
    AtmosphericLevelStorage prior_;
    nimbus::parity::ParityRuntimeState prior_parity_;
    tmv::diagnostics::StormContinuityState prior_storm_tracking_;
    tmv::domain::StormWindowMovementState prior_storm_window_movement_;
    bool prior_storm_selected_partition_available_ = false;
    std::size_t prior_storm_selected_partition_nx_ = 0;
    std::size_t prior_storm_selected_partition_ny_ = 0;
    std::vector<std::uint8_t> prior_storm_selected_partition_;
    tmv::domain::DomainRecyclingTransactionProvenance prior_domain_recycling_provenance_;
    FlatCgridStepTransaction* parent_ = nullptr;
    bool owns_snapshot_ = false;
    bool committed_ = false;
    inline static thread_local FlatCgridStepTransaction* active_ = nullptr;
};
}
