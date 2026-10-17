#pragma once

#include "core/runtime/simulation_state.hpp"

inline bool nested_two_way_feedback_interval_closed(const SimulationStorageState& storage) noexcept
{
    if (storage.configuration.nesting.feedback != tmv::init::config::NestingFeedback::TwoWay)
        return true;
    const auto& nested = storage.nested_execution;
    if (!nested.last_two_way_feedback.complete || nested.two_way_feedback_application_count == 0 ||
        !valid_nested_phase_feedback_report(nested.last_two_way_feedback) ||
        !valid_nested_private_feedback_reports(nested.last_two_way_feedback.private_scalars))
        return false;
    if (nested.last_two_way_feedback.moist)
        for (const auto& report : nested.last_two_way_feedback.water)
            if (!valid_nested_water_feedback_report(report))
                return false;
    const auto& relocation = nested.last_two_way_feedback.relocation;
    if ((!relocation.complete && !relocation.private_scalars.empty()) ||
        (relocation.complete && (!valid_nested_two_way_relocation_report(relocation) ||
                                 relocation.movement_sequence != nested.movement_sequence ||
                                 relocation.feedback_application_count != nested.two_way_feedback_application_count)))
        return false;
    if (relocation.complete)
    {
        const auto& previous = nested.last_two_way_feedback.private_scalars;
        if (relocation.private_scalars.size() != previous.size())
            return false;
        for (std::size_t f = 0; f < previous.size(); ++f)
        {
            const double inventory = previous[f].carrier.parent.child_synchronized_inventory;
            if (relocation.private_scalars[f].name != previous[f].name ||
                std::abs(relocation.private_scalars[f].before - inventory) >
                    128 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(inventory)))
                return false;
        }
    }
    const auto closed = [](const auto& runtime)
    {
        const auto& advection = runtime.advection;
        if (!advection.theta_interface_flux_register.interval_empty() ||
            !advection.dry_mass_interface_flux_register.interval_empty() ||
            !runtime.dynamics.momentum_interface_flux_register.interval_empty())
            return false;
        if (!advection.private_interface.interval_empty())
            return false;
        for (double source : advection.moist_interface.committed_physics)
            if (source != 0)
                return false;
        return std::all_of(advection.moist_interface.water.begin(),
                           advection.moist_interface.water.end(),
                           [](const auto& field) { return field.interval_empty(); });
    };
    return closed(storage.runtime) && closed(nested.child.runtime);
}

template <class Runtime>
void invalidate_nested_feedback_registers(Runtime& runtime) noexcept
{
    runtime.advection.theta_interface_flux_register           = {};
    runtime.advection.theta_interface_flux_capture_enabled    = false;
    runtime.advection.dry_mass_interface_flux_register        = {};
    runtime.advection.dry_mass_interface_flux_capture_enabled = false;
    runtime.advection.moist_interface                         = {};
    runtime.advection.private_interface                       = {};
    runtime.dynamics.momentum_interface_flux_register         = {};
    runtime.dynamics.momentum_interface_flux_capture_enabled  = false;
}
