#include "core/infra/domain_recycling_provenance.hpp"

#include "core/infra/domain_recycling.hpp"

#include <algorithm>
#include <cmath>
#include <initializer_list>

namespace tmv::domain
{
namespace
{

template <std::size_t Capacity>
bool valid_text(const DomainRecyclingProvenanceText<Capacity>& text,
                bool identifier_only) noexcept
{
    if (text.size == 0 || text.size > Capacity || text.bytes[text.size] != '\0')
        return false;
    for (std::size_t index = 0; index < text.size; ++index)
    {
        const unsigned char value =
            static_cast<unsigned char>(text.bytes[index]);
        if (identifier_only)
        {
            if (!((value >= 'a' && value <= 'z')
                  || (value >= 'A' && value <= 'Z')
                  || (value >= '0' && value <= '9') || value == '-'
                  || value == '_'))
                return false;
        }
        else if (value < 0x21 || value > 0x7e)
            return false;
    }
    return true;
}

bool all_finite(std::initializer_list<double> values) noexcept
{
    return std::all_of(values.begin(), values.end(),
                       [](double value) { return std::isfinite(value); });
}

bool close_identity(double lhs, double rhs) noexcept
{
    const double scale = std::max({1.0, std::abs(lhs), std::abs(rhs)});
    return std::isfinite(lhs) && std::isfinite(rhs)
        && std::abs(lhs - rhs) <= 4.0e-13 * scale;
}

bool valid_inventory_identities(
    const ConservationBudget& c,
    const MomentumBudget& m,
    const MoistMixtureEnergyBudgetV2& e) noexcept
{
    double species_total = 0.0;
    for (double mass : c.water_species_mass)
        species_total += mass;
    const double energy_total = e.dry_internal + e.water_sensible
        + e.phase_reference + e.resolved_kinetic + e.gravitational
        + e.subgrid_tke;
    return close_identity(c.total_mass, c.dry_mass)
        && close_identity(c.q_volume_total, c.total_water)
        && close_identity(c.total_water, species_total)
        && close_identity(c.moist_total_mass, c.dry_mass + c.total_water)
        && close_identity(m.total_mass, c.dry_mass)
        && close_identity(m.horizontal_magnitude,
                          std::hypot(m.momentum_x, m.momentum_y))
        && close_identity(m.total_magnitude,
                          std::hypot(m.horizontal_magnitude, m.momentum_z))
        && close_identity(e.total_water_mass, c.total_water)
        && close_identity(e.total, energy_total);
}

} // namespace

std::uint64_t compute_domain_recycling_transaction_chain_fnv1a64(
    std::uint64_t previous_chain_fnv1a64,
    std::uint64_t sequence,
    std::uint64_t event_manifest_bytes,
    std::uint64_t event_manifest_fnv1a64,
    std::string_view transaction_id,
    std::string_view publication_generation) noexcept
{
    std::uint64_t hash = UINT64_C(14695981039346656037);
    const auto mix_u64 = [&hash](std::uint64_t value) {
        for (unsigned byte = 0; byte < 8; ++byte)
        {
            hash ^= static_cast<std::uint8_t>(value >> (byte * 8U));
            hash *= UINT64_C(1099511628211);
        }
    };
    const auto mix_text = [&hash](std::string_view text) {
        for (const unsigned char byte : text)
        {
            hash ^= byte;
            hash *= UINT64_C(1099511628211);
        }
        hash ^= UINT8_C(0xff);
        hash *= UINT64_C(1099511628211);
    };
    mix_u64(previous_chain_fnv1a64);
    mix_u64(sequence);
    mix_u64(event_manifest_bytes);
    mix_u64(event_manifest_fnv1a64);
    mix_text(transaction_id);
    mix_text(publication_generation);
    return hash;
}

bool valid_domain_recycling_private_inventory(const DomainRecyclingPrivateInventory& value) noexcept
{
    if (value.count > value.entries.size())
        return false;
    for (std::size_t i = 0; i < value.entries.size(); ++i)
    {
        const auto& entry = value.entries[i];
        if (i >= value.count)
        {
            if (entry.name.size != 0 || entry.inventory != 0 ||
                std::any_of(entry.name.bytes.begin(), entry.name.bytes.end(), [](char c) { return c != 0; }))
                return false;
            continue;
        }
        if (!valid_text(entry.name, false) || !std::isfinite(entry.inventory) || entry.inventory < 0)
            return false;
        for (std::size_t j = 0; j < i; ++j)
            if (entry.name.view() == value.entries[j].name.view())
                return false;
    }
    return true;
}

bool valid_domain_recycling_transaction_provenance(const DomainRecyclingTransactionProvenance& value) noexcept
{
    if (!valid_domain_recycling_private_inventory(value.retiring_private_inventory) ||
        !valid_domain_recycling_private_inventory(value.admitted_private_inventory) ||
        value.retiring_private_inventory.count != value.admitted_private_inventory.count)
        return false;
    if ((!value.available || !value.private_inventory_complete) &&
        (value.retiring_private_inventory.count != 0 || value.admitted_private_inventory.count != 0))
        return false;
    if (value.private_inventory_complete && (!value.available || !value.control_volume_exchange_complete))
        return false;
    for (std::size_t i = 0; i < value.retiring_private_inventory.count; ++i)
        if (value.retiring_private_inventory.entries[i].name.view() !=
            value.admitted_private_inventory.entries[i].name.view())
            return false;
    if (!value.available)
        return value.sequence == 0;
    if (value.sequence == 0 || !valid_text(value.transaction_id, true)
        || !valid_text(value.publication_generation, true)
        || !valid_text(value.storage_format, true)
        || (value.storage_format.view() != "npy_3d"
            && value.storage_format.view() != "netcdf"
            && value.storage_format.view() != "zfp")
        || value.event_manifest_bytes == 0
        || value.committed_chain_fnv1a64
               != compute_domain_recycling_transaction_chain_fnv1a64(
                      value.previous_chain_fnv1a64, value.sequence,
                      value.event_manifest_bytes,
                      value.event_manifest_fnv1a64,
                      value.transaction_id.view(),
                      value.publication_generation.view())
        || !valid_text(value.shift_algorithm, false)
        || !valid_text(value.frame_algorithm, false)
        || !valid_text(value.swath_algorithm, false)
        || !valid_text(value.scalar_ledger_basis, false)
        || !valid_text(value.momentum_ledger_basis, false)
        || !valid_text(value.energy_ledger_basis, false)
        || !valid_text(value.theta_residual_ledger_basis, false)
        || !all_finite({value.simulation_time_s,
                        value.pre_shift_origin_t0_m.x_m,
                        value.pre_shift_origin_t0_m.y_m,
                        value.post_shift_origin_t0_m.x_m,
                        value.post_shift_origin_t0_m.y_m,
                        value.frame_velocity_ms.u_ms,
                        value.frame_velocity_ms.v_ms, value.dx_m,
                        value.dy_m})
        || value.simulation_time_s < 0.0 || value.dx_m <= 0.0
        || value.dy_m <= 0.0)
        return false;

    WholeCellShiftPlan canonical;
    try
    {
        canonical = plan_whole_cell_shift_for_physical_domain(
            value.nr, value.nth, value.nz, value.shift_i_cells,
            value.shift_j_cells, value.physical_i_begin,
            value.physical_i_end, value.physical_j_begin,
            value.physical_j_end);
    }
    catch (...)
    {
        return false;
    }
    if (canonical.source_i_begin != value.source_i_begin
        || canonical.source_i_end != value.source_i_end
        || canonical.source_j_begin != value.source_j_begin
        || canonical.source_j_end != value.source_j_end
        || value.post_shift_origin_t0_m.x_m
               != value.pre_shift_origin_t0_m.x_m
                    + static_cast<double>(value.shift_i_cells) * value.dx_m
        || value.post_shift_origin_t0_m.y_m
               != value.pre_shift_origin_t0_m.y_m
                    + static_cast<double>(value.shift_j_cells) * value.dy_m)
        return false;
    const std::uint64_t columns = static_cast<std::uint64_t>(
        canonical.trailing_cells / static_cast<std::size_t>(canonical.nz));
    if (canonical.trailing_cells % static_cast<std::size_t>(canonical.nz) != 0
        || value.physical_columns != columns
        || value.physical_cells
               != static_cast<std::uint64_t>(canonical.trailing_cells))
        return false;

    const auto& c = value.inventories;
    const auto& m = value.momentum;
    const auto& e = value.moist_mixture_energy;
    if (!all_finite({c.total_mass, c.theta_volume, c.q_volume_total,
                     c.dry_mass, c.total_water, c.moist_total_mass,
                     c.thermal_energy, m.momentum_x, m.momentum_y,
                     m.momentum_z, m.angular_momentum_z,
                     m.horizontal_magnitude, m.total_magnitude,
                     m.total_mass, m.reference_linear,
                     m.reference_angular, e.dry_internal,
                     e.water_sensible, e.phase_reference,
                     e.resolved_kinetic, e.gravitational,
                     e.subgrid_tke, e.total_water_mass, e.total,
                     value.theta_storage_residual_energy_j}))
        return false;
    for (double mass : c.water_species_mass)
        if (!std::isfinite(mass) || mass < 0.0)
            return false;
    if (!(c.total_mass >= 0.0 && c.dry_mass >= 0.0
        && c.total_water >= 0.0 && c.moist_total_mass >= 0.0
        && m.total_mass >= 0.0 && m.reference_linear >= 0.0
        && m.reference_angular >= 0.0 && e.total_water_mass >= 0.0)
        || !valid_inventory_identities(c, m, e))
        return false;

    if (!value.control_volume_exchange_complete)
    {
        const auto& ac = value.admitted_inventories;
        const auto& am = value.admitted_momentum;
        const auto& ae = value.admitted_moist_mixture_energy;
        return value.control_volume_exchange_basis.size == 0
            && ac.total_mass == 0.0 && ac.theta_volume == 0.0
            && ac.q_volume_total == 0.0 && ac.dry_mass == 0.0
            && ac.total_water == 0.0 && ac.moist_total_mass == 0.0
            && ac.thermal_energy == 0.0
            && std::all_of(
                ac.water_species_mass.begin(), ac.water_species_mass.end(),
                [](double mass) { return mass == 0.0; })
            && am.momentum_x == 0.0 && am.momentum_y == 0.0
            && am.momentum_z == 0.0 && am.angular_momentum_z == 0.0
            && am.horizontal_magnitude == 0.0
            && am.total_magnitude == 0.0 && am.total_mass == 0.0
            && am.reference_linear == 0.0 && am.reference_angular == 0.0
            && ae.dry_internal == 0.0 && ae.water_sensible == 0.0
            && ae.phase_reference == 0.0 && ae.resolved_kinetic == 0.0
            && ae.gravitational == 0.0 && ae.subgrid_tke == 0.0
            && ae.total_water_mass == 0.0 && ae.total == 0.0
            && value.admitted_theta_storage_residual_energy_j == 0.0
            && value.retiring_fixed_earth_angular_momentum_z == 0.0
            && value.admitted_fixed_earth_angular_momentum_z == 0.0;
    }

    const auto& ac = value.admitted_inventories;
    const auto& am = value.admitted_momentum;
    const auto& ae = value.admitted_moist_mixture_energy;
    if (!valid_text(value.control_volume_exchange_basis, false)
        || value.control_volume_exchange_basis.view()
               != "instantaneous_fixed_earth_control_volume_exchange_v1"
        || !all_finite({
            ac.total_mass, ac.theta_volume, ac.q_volume_total, ac.dry_mass,
            ac.total_water, ac.moist_total_mass, ac.thermal_energy,
            am.momentum_x, am.momentum_y, am.momentum_z,
            am.angular_momentum_z, am.horizontal_magnitude,
            am.total_magnitude, am.total_mass, am.reference_linear,
            am.reference_angular, ae.dry_internal, ae.water_sensible,
            ae.phase_reference, ae.resolved_kinetic, ae.gravitational,
            ae.subgrid_tke, ae.total_water_mass, ae.total,
            value.admitted_theta_storage_residual_energy_j,
            value.retiring_fixed_earth_angular_momentum_z,
            value.admitted_fixed_earth_angular_momentum_z})
        || ac.total_mass < 0.0 || ac.dry_mass < 0.0
        || ac.total_water < 0.0 || ac.moist_total_mass < 0.0
        || am.total_mass < 0.0 || am.reference_linear < 0.0
        || am.reference_angular < 0.0 || ae.total_water_mass < 0.0
        || value.admitted_theta_storage_residual_energy_j != 0.0)
        return false;
    for (double mass : ac.water_species_mass)
        if (!std::isfinite(mass) || mass < 0.0)
            return false;
    if (!valid_inventory_identities(ac, am, ae))
        return false;

    const double local_center_x = 0.5
        * static_cast<double>(value.physical_i_begin + value.physical_i_end)
        * value.dx_m;
    const double local_center_y = 0.5
        * static_cast<double>(value.physical_j_begin + value.physical_j_end)
        * value.dy_m;
    const double pre_center_x = value.pre_shift_origin_t0_m.x_m
        + value.frame_velocity_ms.u_ms * value.simulation_time_s
        + local_center_x;
    const double pre_center_y = value.pre_shift_origin_t0_m.y_m
        + value.frame_velocity_ms.v_ms * value.simulation_time_s
        + local_center_y;
    const double post_center_x = value.post_shift_origin_t0_m.x_m
        + value.frame_velocity_ms.u_ms * value.simulation_time_s
        + local_center_x;
    const double post_center_y = value.post_shift_origin_t0_m.y_m
        + value.frame_velocity_ms.v_ms * value.simulation_time_s
        + local_center_y;
    return close_identity(
               value.retiring_fixed_earth_angular_momentum_z,
               m.angular_momentum_z + pre_center_x * m.momentum_y
                   - pre_center_y * m.momentum_x)
        && close_identity(
               value.admitted_fixed_earth_angular_momentum_z,
               am.angular_momentum_z + post_center_x * am.momentum_y
                   - post_center_y * am.momentum_x);
}

const char* domain_recycling_transaction_provenance_algorithm_id() noexcept
{
    return "checkpoint_bound_d5_transaction_provenance_v1";
}

} // namespace tmv::domain
