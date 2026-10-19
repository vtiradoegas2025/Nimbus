#include "compute/native_conservative_reduction.hpp"
#include "terrain/conservative_transport.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/infra/nested_scalar_feedback.hpp"
#include "diagnostics/deterministic_reduction.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <iomanip>

namespace
{
using Relationship = tmv::domain::OneWayNestRelationshipV1;
void validate_relationship(const Relationship& r)
{
    tmv::domain::validate_one_way_nest_relationship(r);
    if ((r.parent_ratio_x != 2 && r.parent_ratio_x != 3) || r.parent_ratio_x != r.parent_ratio_y)
        throw std::invalid_argument("nested scalar feedback requires isotropic 2:1 or 3:1 refinement");
}
void shape(const Field3D& f, int nx, int ny, int nz)
{
    if (f.size_r() != nx + 2 || f.size_th() != ny + 2 || f.size_z() != nz)
        throw std::invalid_argument("nested scalar feedback field shape mismatch");
}
void validate_flux(const tmv::CartesianThetaInterfaceFluxes& f, const Relationship& r)
{
    const auto x = std::size_t(r.child_parent_span_y) * r.levels_z, y = std::size_t(r.child_parent_span_x) * r.levels_z;
    if (f.west.size() != x || f.east.size() != x || f.south.size() != y || f.north.size() != y ||
        !std::isfinite(f.accumulated_dt_s) || f.accumulated_dt_s <= 0 || !f.contribution_count)
        throw std::invalid_argument("nested scalar feedback needs complete integrated interface fluxes");
    for (auto* side : {&f.west, &f.east, &f.south, &f.north})
        for (double value : *side)
            if (!std::isfinite(value))
                throw std::invalid_argument("nested scalar interface flux is not finite");
}
double inventory(float density, float scalar, double volume, bool concentration, bool positive)
{
    if (!std::isfinite(density) || density <= 0 || !std::isfinite(scalar) || (positive && scalar < 0) ||
        !std::isfinite(volume) || volume <= 0)
        throw std::invalid_argument("nested scalar feedback received an invalid physical reservoir");
    const double value = volume * (concentration ? 1.0 : double(density)) * scalar;
    if (!std::isfinite(value))
        throw std::overflow_error("nested scalar inventory overflowed");
    return value;
}
template <class ParentVolume, class ChildVolume>
PreparedNestedScalarFeedback prepare(const Relationship&                       r,
                                     const ParentVolume&                       parent_volume,
                                     const ChildVolume&                        child_volume,
                                     const Field3D&                            rho_before,
                                     const Field3D&                            rho_after,
                                     const Field3D&                            rho_child,
                                     const Field3D&                            parent,
                                     const Field3D&                            child,
                                     const tmv::CartesianThetaInterfaceFluxes& coarse,
                                     const tmv::CartesianThetaInterfaceFluxes& fine,
                                     bool                                      concentration,
                                     bool                                      positive,
                                     bool                                      allow_fine_redistribution,
                                     ComputeBackendRuntimeState*               compute,
                                     NestedScalarFeedbackBounds                bounds)
{
    validate_relationship(r);
    for (auto* field : {&rho_before, &rho_after, &parent})
        shape(*field, r.parent_cells_x, r.parent_cells_y, r.levels_z);
    for (auto* field : {&rho_child, &child})
        shape(*field, r.child_cells_x, r.child_cells_y, r.levels_z);
    if ((bounds.parent == nullptr) != (bounds.child == nullptr) || (bounds.parent && !positive))
        throw std::invalid_argument("nested scalar lower bounds require a complete positive pair");
    if (bounds.parent)
    {
        shape(*bounds.parent, r.parent_cells_x, r.parent_cells_y, r.levels_z);
        shape(*bounds.child, r.child_cells_x, r.child_cells_y, r.levels_z);
    }
    validate_flux(coarse, r);
    validate_flux(fine, r);
    if (std::abs(coarse.accumulated_dt_s - fine.accumulated_dt_s) >
        64 * std::numeric_limits<double>::epsilon() * std::max(coarse.accumulated_dt_s, fine.accumulated_dt_s))
        throw std::invalid_argument("nested scalar interface flux intervals disagree");
    nimbus::compute::NativeConservativeReduction native(compute);
    PreparedNestedScalarFeedback                 result;
    result.scalar                           = parent;
    auto&                            report = result.report;
    tmv::diagnostics::CompensatedSum initial, final, restriction, reflux, rounding, mismatch, child_inventory;
    for (std::size_t n = 0; n < coarse.west.size(); ++n)
    {
        mismatch.add(coarse.west[n] - fine.west[n]);
        mismatch.add(fine.east[n] - coarse.east[n]);
    }
    for (std::size_t n = 0; n < coarse.south.size(); ++n)
    {
        mismatch.add(coarse.south[n] - fine.south[n]);
        mismatch.add(fine.north[n] - coarse.north[n]);
    }
    const int      oi = r.child_origin_parent_cell_i, oj = r.child_origin_parent_cell_j, sx = r.child_parent_span_x,
                   sy = r.child_parent_span_y, ratio = r.parent_ratio_x;
    TerrainField3D targets;
    targets.resize(r.parent_cells_x + 2, r.parent_cells_y + 2, r.levels_z, 0.0);
    for (int i = 1; i <= r.parent_cells_x; ++i)
        for (int j = 1; j <= r.parent_cells_y; ++j)
            for (int k = 0; k < r.levels_z; ++k)
            {
                const double volume = parent_volume(i, j, k);
                const double before = inventory(rho_before(i, j, k), parent(i, j, k), volume, concentration, positive);
                initial.add(before);
                double target = before;
                if (native.enabled())
                    native.begin_row();
                if (i > oi && i <= oi + sx && j > oj && j <= oj + sy)
                {
                    tmv::diagnostics::CompensatedSum restricted;
                    const int                        ci = (i - oi - 1) * ratio + 1, cj = (j - oj - 1) * ratio + 1;
                    for (int di = 0; di < ratio; ++di)
                        for (int dj = 0; dj < ratio; ++dj)
                            restricted.add(inventory(rho_child(ci + di, cj + dj, k),
                                                     child(ci + di, cj + dj, k),
                                                     child_volume(ci + di, cj + dj, k),
                                                     concentration,
                                                     positive));
                    if (native.enabled())
                        for (int di = 0; di < ratio; ++di)
                            for (int dj = 0; dj < ratio; ++dj)
                                native.term(concentration ? 1 : rho_child(ci + di, cj + dj, k),
                                            child(ci + di, cj + dj, k),
                                            child_volume(ci + di, cj + dj, k));
                    target = restricted.value();
                    child_inventory.add(target);
                    restriction.add(target - before);
                    ++report.restricted_cells;
                }
                else if (native.enabled())
                    native.term(concentration ? 1 : rho_before(i, j, k), parent(i, j, k), volume);
                double delta     = 0;
                bool   interface = false;
                if (j > oj && j <= oj + sy)
                {
                    const auto bin = std::size_t(j - oj - 1) * r.levels_z + k;
                    if (i == oi)
                    {
                        delta     = coarse.west[bin] - fine.west[bin];
                        interface = true;
                    }
                    if (i == oi + sx + 1)
                    {
                        delta     = fine.east[bin] - coarse.east[bin];
                        interface = true;
                    }
                }
                if (i > oi && i <= oi + sx)
                {
                    const auto bin = std::size_t(i - oi - 1) * r.levels_z + k;
                    if (j == oj)
                    {
                        delta     = coarse.south[bin] - fine.south[bin];
                        interface = true;
                    }
                    if (j == oj + sy + 1)
                    {
                        delta     = fine.north[bin] - coarse.north[bin];
                        interface = true;
                    }
                }
                target += delta;
                reflux.add(delta);
                if (interface)
                    ++report.refluxed_cells;
                if (!std::isfinite(target))
                    throw std::overflow_error("nested scalar reflux inventory overflowed");
                targets(i, j, k) = target;
                if (native.enabled())
                {
                    native.term(1, delta, 1);
                    native.end_row();
                }
            }
    if (native.enabled())
    {
        native.dispatch();
        std::size_t row = 0;
        for (int i = 1; i <= r.parent_cells_x; ++i)
            for (int j = 1; j <= r.parent_cells_y; ++j)
                for (int k = 0; k < r.levels_z; ++k)
                {
                    const double target     = native.inventory(row++);
                    const double difference = target - targets(i, j, k);
                    if (i > oi && i <= oi + sx && j > oj && j <= oj + sy)
                    {
                        restriction.add(difference);
                        child_inventory.add(difference);
                    }
                    else
                        rounding.add(difference);
                    targets(i, j, k) = target;
                }
    }
    TerrainField3D                   floors;
    tmv::diagnostics::CompensatedSum fine_available;
    if (bounds.parent)
    {
        floors.resize(r.parent_cells_x + 2, r.parent_cells_y + 2, r.levels_z, 0.0);
        for (int i = 1; i <= r.parent_cells_x; ++i)
            for (int j = 1; j <= r.parent_cells_y; ++j)
                for (int k = 0; k < r.levels_z; ++k)
                {
                    const double floor = inventory(
                        rho_after(i, j, k), (*bounds.parent)(i, j, k), parent_volume(i, j, k), concentration, true);
                    if (i > oi && i <= oi + sx && j > oj && j <= oj + sy)
                        continue;
                    floors(i, j, k) = floor;
                    targets(i, j, k) -= floor;
                    if (floor > 0)
                        ++report.constrained_cells;
                }
        for (int i = 1; i <= r.child_cells_x; ++i)
            for (int j = 1; j <= r.child_cells_y; ++j)
                for (int k = 0; k < r.levels_z; ++k)
                {
                    const double before =
                        inventory(rho_child(i, j, k), child(i, j, k), child_volume(i, j, k), concentration, true);
                    const double floor = inventory(
                        rho_child(i, j, k), (*bounds.child)(i, j, k), child_volume(i, j, k), concentration, true);
                    if (floor > before)
                        throw std::invalid_argument("nested scalar fine lower bound exceeds its reservoir");
                    fine_available.add(before - floor);
                }
    }
    bool needs_global_redistribution = false;
    if (positive)
        for (int i = 1; i <= r.parent_cells_x; ++i)
            for (int j = 1; j <= r.parent_cells_y; ++j)
                for (int k = 0; k < r.levels_z; ++k)
                {
                    const double deficit = -targets(i, j, k);
                    if (deficit <= 0)
                        continue;
                    std::array<double*, 6> donors{};
                    std::size_t            count     = 0;
                    long double            available = 0;
                    for (const auto& offset : std::array<std::array<int, 3>, 6>{
                             {{-1, 0, 0}, {1, 0, 0}, {0, -1, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 1}}})
                    {
                        const int ni = i + offset[0], nj = j + offset[1], nk = k + offset[2];
                        if (ni < 1 || ni > r.parent_cells_x || nj < 1 || nj > r.parent_cells_y || nk < 0 ||
                            nk >= r.levels_z || (ni > oi && ni <= oi + sx && nj > oj && nj <= oj + sy))
                            continue;
                        double& value = targets(ni, nj, nk);
                        if (value > 0)
                        {
                            donors[count++] = &value;
                            available += value;
                        }
                    }
                    if (available < deficit)
                    {
                        needs_global_redistribution = true;
                        continue;
                    }
                    // A lower-bound CAAS correction stays on the uncovered coarse grid;
                    // borrowing from covered cells would disagree with the retained child.
                    const long double fraction = static_cast<long double>(deficit) / available;
                    for (std::size_t n = 0; n < count; ++n)
                        *donors[n] = static_cast<double>(*donors[n] * (1.0L - fraction));
                    targets(i, j, k) = 0;
                    report.redistributed_inventory += deficit;
                    ++report.redistributed_cells;
                }
    if (needs_global_redistribution)
    {
        tmv::diagnostics::CompensatedSum available, deficit;
        for (int i = 1; i <= r.parent_cells_x; ++i)
            for (int j = 1; j <= r.parent_cells_y; ++j)
            {
                if (i > oi && i <= oi + sx && j > oj && j <= oj + sy)
                    continue;
                for (int k = 0; k < r.levels_z; ++k)
                {
                    const double value = targets(i, j, k);
                    if (value < 0)
                    {
                        deficit.add(-value);
                        ++report.global_redistributed_cells;
                    }
                    else
                        available.add(value);
                }
            }
        const bool   use_fine = allow_fine_redistribution && available.value() < deficit.value();
        const double capacity =
            available.value() + (use_fine ? (bounds.child ? fine_available.value() : child_inventory.value()) : 0.0);
        if (!std::isfinite(capacity))
            throw std::overflow_error("nested scalar donor inventory overflowed");
        if (capacity < deficit.value())
        {
            std::ostringstream message;
            message << std::setprecision(17)
                    << "nested scalar reflux exhausts the uncovered coarse inventory: deficit=" << deficit.value()
                    << ", available=" << available.value();
            throw NestedScalarInventoryExhausted(message.str());
        }
        const long double fraction = static_cast<long double>(deficit.value()) / capacity;
        for (int i = 1; i <= r.parent_cells_x; ++i)
            for (int j = 1; j <= r.parent_cells_y; ++j)
            {
                if (i > oi && i <= oi + sx && j > oj && j <= oj + sy)
                    continue;
                for (int k = 0; k < r.levels_z; ++k)
                {
                    double& value = targets(i, j, k);
                    value         = value > 0 ? static_cast<double>(value * (1.0L - fraction)) : 0.0;
                }
            }
        if (use_fine)
        {
            result.child_scalar = child;
            tmv::diagnostics::CompensatedSum requested, storage, synchronized;
            if (native.enabled())
            {
                native.clear();
                for (int i = 1; i <= r.child_cells_x; ++i)
                    for (int j = 1; j <= r.child_cells_y; ++j)
                        for (int k = 0; k < r.levels_z; ++k)
                        {
                            const double volume = child_volume(i, j, k);
                            native.begin_row();
                            const double lower = bounds.child ? (*bounds.child)(i, j, k) : 0.0;
                            native.term(concentration ? 1 : rho_child(i, j, k),
                                        double(child(i, j, k)) - lower,
                                        double(volume * (1.0L - fraction)));
                            if (bounds.child)
                                native.term(concentration ? 1 : rho_child(i, j, k), lower, volume);
                            native.denominator();
                            native.term(concentration ? 1 : rho_child(i, j, k), 1, volume);
                            native.end_row();
                        }
                native.dispatch();
            }
            std::size_t fine_native_row = 0;
            for (int i = 1; i <= r.child_cells_x; ++i)
                for (int j = 1; j <= r.child_cells_y; ++j)
                    for (int k = 0; k < r.levels_z; ++k)
                    {
                        const double volume = child_volume(i, j, k);
                        const double before =
                            inventory(rho_child(i, j, k), child(i, j, k), volume, concentration, true);
                        const double lower   = bounds.child ? (*bounds.child)(i, j, k) : 0.0;
                        const double floor   = inventory(rho_child(i, j, k), float(lower), volume, concentration, true);
                        const double transfer = static_cast<double>((before - floor) * fraction);
                        const double target   = std::max(floor, before - transfer);
                        const double carrier = inventory(rho_child(i, j, k), 1, volume, concentration, true);
                        float stored = native.enabled() ? native.stored(fine_native_row++) : static_cast<float>(target / carrier);
                        if (!std::isfinite(stored) || stored < 0)
                            throw std::overflow_error("nested scalar feedback cannot represent its fine candidate");
                        if (bounds.child)
                            stored = std::max(stored, float(lower));
                        const double actual  = inventory(rho_child(i, j, k), stored, volume, concentration, true);
                        result.child_scalar(i, j, k) = stored;
                        // The transfer can be smaller than one inventory ULP; retain it
                        // independently of the rounded target and stored population.
                        requested.add(transfer);
                        storage.add(actual - before);
                        storage.add(transfer);
                        synchronized.add(actual);
                    }
            for (int i = oi + 1; i <= oi + sx; ++i)
                for (int j = oj + 1; j <= oj + sy; ++j)
                    for (int k = 0; k < r.levels_z; ++k)
                    {
                        tmv::diagnostics::CompensatedSum restricted;
                        for (int di = 0; di < ratio; ++di)
                            for (int dj = 0; dj < ratio; ++dj)
                            {
                                const int ci = (i - oi - 1) * ratio + di + 1;
                                const int cj = (j - oj - 1) * ratio + dj + 1;
                                restricted.add(inventory(rho_child(ci, cj, k),
                                                         result.child_scalar(ci, cj, k),
                                                         child_volume(ci, cj, k),
                                                         concentration,
                                                         true));
                            }
                        targets(i, j, k) = restricted.value();
                    }
            if (native.enabled())
            {
                native.clear();
                for (int i = oi + 1; i <= oi + sx; ++i)
                    for (int j = oj + 1; j <= oj + sy; ++j)
                        for (int k = 0; k < r.levels_z; ++k)
                        {
                            native.begin_row();
                            for (int di = 0; di < ratio; ++di)
                                for (int dj = 0; dj < ratio; ++dj)
                                {
                                    const int ci = (i - oi - 1) * ratio + di + 1, cj = (j - oj - 1) * ratio + dj + 1;
                                    native.term(concentration ? 1 : rho_child(ci, cj, k),
                                                result.child_scalar(ci, cj, k),
                                                child_volume(ci, cj, k));
                                }
                            native.end_row();
                        }
                native.dispatch();
                std::size_t row = 0;
                for (int i = oi + 1; i <= oi + sx; ++i)
                    for (int j = oj + 1; j <= oj + sy; ++j)
                        for (int k = 0; k < r.levels_z; ++k)
                        {
                            const double target = native.inventory(row++);
                            rounding.add(target - targets(i, j, k));
                            targets(i, j, k) = target;
                        }
            }
            report.fine_requested_transfer      = requested.value();
            report.fine_storage_rounding        = storage.value();
            report.child_synchronized_inventory = synchronized.value();
            rounding.add(report.fine_storage_rounding);
        }
        report.global_redistributed_inventory = deficit.value();
        report.redistributed_inventory += report.global_redistributed_inventory;
        report.redistributed_cells += report.global_redistributed_cells;
    }
    if (bounds.parent)
        for (int i = 1; i <= r.parent_cells_x; ++i)
            for (int j = 1; j <= r.parent_cells_y; ++j)
                for (int k = 0; k < r.levels_z; ++k)
                    targets(i, j, k) += floors(i, j, k);
    if (native.enabled())
    {
        native.clear();
        for (int i = 1; i <= r.parent_cells_x; ++i)
            for (int j = 1; j <= r.parent_cells_y; ++j)
                for (int k = 0; k < r.levels_z; ++k)
                {
                    native.begin_row();
                    native.term(1, targets(i, j, k), 1);
                    native.denominator();
                    native.term(concentration ? 1 : rho_after(i, j, k), 1, parent_volume(i, j, k));
                    native.end_row();
                }
        native.dispatch();
    }
    std::size_t native_row = 0;
    for (int i = 1; i <= r.parent_cells_x; ++i)
        for (int j = 1; j <= r.parent_cells_y; ++j)
            for (int k = 0; k < r.levels_z; ++k)
            {
                const double volume      = parent_volume(i, j, k);
                const double target      = targets(i, j, k);
                const double denominator = inventory(rho_after(i, j, k), 1, volume, concentration, true);
                if (denominator <= 0)
                    throw std::domain_error("nested scalar recovery has no carrier");
                float stored = native.enabled() ? native.stored(native_row++) : float(target / denominator);
                if (bounds.parent && !(i > oi && i <= oi + sx && j > oj && j <= oj + sy))
                    stored = std::max(stored, (*bounds.parent)(i, j, k));
                if (!std::isfinite(stored) || (positive && stored < 0))
                    throw std::overflow_error("nested scalar feedback cannot represent its candidate");
                const double actual    = inventory(rho_after(i, j, k), stored, volume, concentration, positive);
                result.scalar(i, j, k) = stored;
                final.add(actual);
                rounding.add(actual - target);
            }
    if (result.child_scalar.empty())
        report.child_synchronized_inventory = child_inventory.value();
    report.child_inventory    = child_inventory.value();
    report.initial_inventory  = initial.value();
    report.final_inventory    = final.value();
    report.restriction_change = restriction.value();
    report.interface_mismatch = mismatch.value();
    report.applied_reflux     = reflux.value();
    report.storage_rounding   = rounding.value();
    report.closure_residual   = report.final_inventory - report.initial_inventory - report.restriction_change -
                                report.applied_reflux - report.storage_rounding;
    const double scale        = std::max({1.0,
                                          std::abs(report.initial_inventory),
                                          std::abs(report.final_inventory),
                                          std::abs(report.restriction_change),
                                          std::abs(report.applied_reflux)});
    if (!std::isfinite(scale) || !std::isfinite(report.closure_residual) ||
        std::abs(report.closure_residual) > 128 * std::numeric_limits<double>::epsilon() * scale ||
        std::abs(report.interface_mismatch - report.applied_reflux) >
            128 * std::numeric_limits<double>::epsilon() * scale)
        throw std::runtime_error("nested scalar feedback inventory does not close");
    report.complete = true;
    return result;
}
} // namespace

PreparedNestedScalarFeedback prepare_nested_scalar_feedback_flat(const Relationship&        r,
                                                                 const std::vector<double>& thickness,
                                                                 const Field3D&             before,
                                                                 const Field3D&             after,
                                                                 const Field3D&             child_density,
                                                                 const Field3D&             parent,
                                                                 const Field3D&             child,
                                                                 const tmv::CartesianThetaInterfaceFluxes& coarse,
                                                                 const tmv::CartesianThetaInterfaceFluxes& fine,
                                                                 bool                        concentration,
                                                                 bool                        positive,
                                                                 bool                        allow_fine_redistribution,
                                                                 ComputeBackendRuntimeState* compute,
                                                                 NestedScalarFeedbackBounds  bounds)
{
    validate_relationship(r);
    if (thickness.size() != std::size_t(r.levels_z))
        throw std::invalid_argument("nested scalar layer geometry is incomplete");
    for (double v : thickness)
        if (!std::isfinite(v) || v <= 0)
            throw std::invalid_argument("nested scalar layer thickness is invalid");
    return prepare(
        r,
        [&](int, int, int k) { return r.parent_dx_m * r.parent_dy_m * thickness[k]; },
        [&](int, int, int k) { return r.child_dx_m * r.child_dy_m * thickness[k]; },
        before,
        after,
        child_density,
        parent,
        child,
        coarse,
        fine,
        concentration,
        positive,
        allow_fine_redistribution,
        compute,
        bounds);
}
PreparedNestedScalarFeedback
prepare_nested_scalar_feedback_mapped(const tmv::domain::NestedMappedTerrainGeometryV1& geometry,
                                      const Field3D&                                    before,
                                      const Field3D&                                    after,
                                      const Field3D&                                    child_density,
                                      const Field3D&                                    parent,
                                      const Field3D&                                    child,
                                      const tmv::CartesianThetaInterfaceFluxes&         coarse,
                                      const tmv::CartesianThetaInterfaceFluxes&         fine,
                                      bool                                              concentration,
                                      bool                                              positive,
                                      bool                                              allow_fine_redistribution,
                                      ComputeBackendRuntimeState*                       compute,
                                      NestedScalarFeedbackBounds                        bounds)
{
    const auto& r = geometry.relationship;
    validate_relationship(r);
    const auto& p = geometry.parent.metrics;
    const auto& c = geometry.child_storage.metrics;
    if (geometry.generation == 0 || p.J.size_r() != r.parent_cells_x || p.J.size_th() != r.parent_cells_y ||
        p.J.size_z() != r.levels_z || c.J.size_r() != r.child_cells_x + 2 || c.J.size_th() != r.child_cells_y + 2 ||
        c.J.size_z() != r.levels_z || p.dx != r.parent_dx_m || p.dy != r.parent_dy_m || c.dx != r.child_dx_m ||
        c.dy != r.child_dy_m)
        throw std::invalid_argument("mapped nested scalar feedback geometry is incomplete");
    return prepare(
        r,
        [&](int i, int j, int k) { return p.dx * p.dy * p.dzeta * p.J(i - 1, j - 1, k); },
        [&](int i, int j, int k) { return c.dx * c.dy * c.dzeta * c.J(i, j, k); },
        before,
        after,
        child_density,
        parent,
        child,
        coarse,
        fine,
        concentration,
        positive,
        allow_fine_redistribution,
        compute,
        bounds);
}

std::vector<double> nested_private_inventory(const SimulationStateView& state)
{
    if (!state.microphysics_runtime || !state.microphysics_runtime->scheme)
        throw std::invalid_argument("private inventory requires its live microphysics owner");
    const auto fields = state.microphysics_runtime->scheme->prognostic_scalars_for_transport();
    if (fields.empty())
        return {};
    tmv::physical_grid_metrics::validate(state);
    const tmv::physical_grid_metrics::ControlVolumeMassView metrics(state);
    const auto                                              domain = physical_scalar_domain(state.grid);
    std::vector<double>                                     result;
    result.reserve(fields.size());
    for (const auto& scalar : fields)
    {
        if (!scalar.field || scalar.storage != tmv::MicrophysicsScalarStorage::DryAirMixingRatio ||
            !scalar.positive_definite || scalar.field->size_r() != state.grid.nr ||
            scalar.field->size_th() != state.grid.nth || scalar.field->size_z() != state.grid.nz)
            throw std::invalid_argument("private inventory requires complete positive dry-specific fields");
        tmv::diagnostics::CompensatedSum sum;
        for (int i = domain.i_begin; i < domain.i_end; ++i)
            for (int j = domain.j_begin; j < domain.j_end; ++j)
                for (int k = domain.k_begin; k < domain.k_end; ++k)
                    sum.add(inventory(state.fields.rho(i, j, k),
                                      (*scalar.field)(i, j, k),
                                      metrics.scalar_volume_m3(i, j, k),
                                      false,
                                      true));
        if (!std::isfinite(sum.value()))
            throw std::overflow_error("private inventory overflowed");
        result.push_back(sum.value());
    }
    return result;
}

double nested_private_carrier_inventory(const tmv::domain::OneWayNestRelationshipV1& r,
                                        const std::vector<double>&                   thickness,
                                        const Field3D&                               density,
                                        const Field3D&                               scalar,
                                        int                                          shifted_i,
                                        int                                          shifted_j,
                                        bool                                         outside_shifted_domain,
                                        const TerrainMetrics3D*                      metrics)
{
    validate_relationship(r);
    shape(density, r.child_cells_x, r.child_cells_y, r.levels_z);
    shape(scalar, r.child_cells_x, r.child_cells_y, r.levels_z);
    if (thickness.size() != std::size_t(r.levels_z))
        throw std::invalid_argument("private relocation inventory requires complete vertical geometry");
    if (metrics)
    {
        tmv::terrain_transport::validate_geometry(*metrics);
        if (metrics->J.size_r() != r.child_cells_x + 2 || metrics->J.size_th() != r.child_cells_y + 2 ||
            metrics->J.size_z() != r.levels_z || metrics->dx != r.child_dx_m || metrics->dy != r.child_dy_m)
            throw std::invalid_argument("private relocation geometry differs from the child");
    }
    tmv::diagnostics::CompensatedSum total;
    for (int i = 1; i <= r.child_cells_x; ++i)
        for (int j = 1; j <= r.child_cells_y; ++j)
        {
            const auto si = std::int64_t(i) + shifted_i, sj = std::int64_t(j) + shifted_j;
            if (outside_shifted_domain && si >= 1 && si <= r.child_cells_x && sj >= 1 && sj <= r.child_cells_y)
                continue;
            for (int k = 0; k < r.levels_z; ++k)
            {
                const double volume = metrics ? metrics->J(i, j, k) * metrics->dx * metrics->dy * metrics->dzeta
                                              : r.child_dx_m * r.child_dy_m * thickness[k];
                total.add(inventory(density(i, j, k), scalar(i, j, k), volume, false, true));
            }
        }
    if (!std::isfinite(total.value()))
        throw std::overflow_error("private relocation inventory overflowed");
    return total.value();
}

std::array<double, 8> nested_thermodynamic_inventory(const SimulationStateView& state)
{
    tmv::physical_grid_metrics::validate(state);
    const tmv::physical_grid_metrics::ControlVolumeMassView metrics(state);
    const auto                                              domain = physical_scalar_domain(state.grid);
    const std::array<const Field3D*, 8>                     fields{&state.fields.theta,
                                                                   &state.fields.qv,
                                                                   &state.fields.qc,
                                                                   &state.fields.qr,
                                                                   &state.fields.qi,
                                                                   &state.fields.qs,
                                                                   &state.fields.qg,
                                                                   &state.fields.qh};
    for (auto* field : fields)
        if (field->size_r() != state.grid.nr || field->size_th() != state.grid.nth || field->size_z() != state.grid.nz)
            throw std::invalid_argument("nested thermodynamic inventory storage is incomplete");
    std::array<tmv::diagnostics::CompensatedSum, 8> sums;
    for (int i = domain.i_begin; i < domain.i_end; ++i)
        for (int j = domain.j_begin; j < domain.j_end; ++j)
            for (int k = domain.k_begin; k < domain.k_end; ++k)
            {
                const double volume = metrics.scalar_volume_m3(i, j, k);
                for (std::size_t f = 0; f < fields.size(); ++f)
                    sums[f].add(inventory(state.fields.rho(i, j, k), (*fields[f])(i, j, k), volume, false, true));
            }
    std::array<double, 8> result{};
    for (std::size_t f = 0; f < result.size(); ++f)
    {
        result[f] = sums[f].value();
        if (!std::isfinite(result[f]))
            throw std::overflow_error("nested thermodynamic inventory overflowed");
    }
    return result;
}

std::array<double, 12> nested_primary_inventory(const tmv::domain::OneWayNestRelationshipV1& r,
                                                const std::vector<double>&                   thickness,
                                                const PrognosticFieldStorage&                fields,
                                                int                                          shifted_i,
                                                int                                          shifted_j,
                                                bool                                         outside_shifted_domain,
                                                bool                                         absolute_contributions,
                                                const TerrainMetrics3D*                      metrics)
{
    validate_relationship(r);
    if (thickness.size() != std::size_t(r.levels_z))
        throw std::invalid_argument("nested primary inventory has incomplete vertical geometry");
    if (metrics)
    {
        tmv::terrain_transport::validate_geometry(*metrics);
        if (metrics->J.size_r() != r.child_cells_x + 2 || metrics->J.size_th() != r.child_cells_y + 2 ||
            metrics->J.size_z() != r.levels_z || metrics->dx != r.child_dx_m || metrics->dy != r.child_dy_m)
            throw std::invalid_argument("nested primary inventory mapped geometry differs from the child");
    }
    const std::array<const Field3D*, 8> scalars{
        &fields.theta, &fields.qv, &fields.qc, &fields.qr, &fields.qi, &fields.qs, &fields.qg, &fields.qh};
    for (const auto* field : {&fields.rho, &fields.u, &fields.v, &fields.w})
        shape(*field, r.child_cells_x, r.child_cells_y, r.levels_z);
    for (const auto* scalar : scalars)
        shape(*scalar, r.child_cells_x, r.child_cells_y, r.levels_z);
    const auto velocity = [absolute_contributions](float value)
    { return absolute_contributions ? std::abs(double(value)) : double(value); };
    std::array<tmv::diagnostics::CompensatedSum, 12> totals;
    for (int i = 1; i <= r.child_cells_x; ++i)
        for (int j = 1; j <= r.child_cells_y; ++j)
        {
            const auto si = std::int64_t(i) + shifted_i, sj = std::int64_t(j) + shifted_j;
            if (outside_shifted_domain && si >= 1 && si <= r.child_cells_x && sj >= 1 && sj <= r.child_cells_y)
                continue;
            for (int k = 0; k < r.levels_z; ++k)
            {
                const double volume = metrics ? metrics->J(i, j, k) * metrics->dx * metrics->dy * metrics->dzeta
                                              : r.child_dx_m * r.child_dy_m * thickness[k];
                const double mass   = inventory(fields.rho(i, j, k), 1, volume, false, true);
                totals[0].add(mass);
                totals[9].add(.5 * mass * (velocity(fields.u(i - 1, j, k)) + velocity(fields.u(i, j, k))));
                totals[10].add(.5 * mass * (velocity(fields.v(i, j - 1, k)) + velocity(fields.v(i, j, k))));
                totals[11].add(.5 * mass *
                               ((k > 0 ? velocity(fields.w(i, j, k - 1)) : 0.0) + velocity(fields.w(i, j, k))));
                if (metrics && k == 0)
                {
                    const double ground = absolute_contributions
                                              ? .5 * ((velocity(fields.u(i - 1, j, 0)) + velocity(fields.u(i, j, 0))) *
                                                          std::abs(metrics->Gx_face(i, j, 0)) +
                                                      (velocity(fields.v(i, j - 1, 0)) + velocity(fields.v(i, j, 0))) *
                                                          std::abs(metrics->Gy_face(i, j, 0)))
                                              : tmv::terrain_transport::cgrid_ground_vertical_velocity(
                                                    *metrics, fields.u, fields.v, i, j);
                    totals[11].add(.5 * mass * ground);
                }
                for (std::size_t f = 0; f < scalars.size(); ++f)
                    totals[f + 1].add(inventory(fields.rho(i, j, k), (*scalars[f])(i, j, k), volume, false, true));
            }
        }
    std::array<double, 12> result{};
    for (std::size_t f = 0; f < result.size(); ++f)
    {
        result[f] = totals[f].value();
        if (!std::isfinite(result[f]))
            throw std::overflow_error("nested primary inventory overflowed");
    }
    return result;
}
