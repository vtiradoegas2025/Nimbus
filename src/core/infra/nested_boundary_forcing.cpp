#include "core/infra/nested_boundary_forcing.hpp"

#include "core/field/field_bounds.hpp"
#include "dynamics/eos_pressure.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <utility>

namespace tmv::domain
{
namespace
{

class FixedOrderSum
{
public:
    void add(double value) noexcept
    {
        const double corrected = value - compensation_;
        const double next = sum_ + corrected;
        compensation_ = (next - sum_) - corrected;
        sum_ = next;
    }

    double value() const noexcept { return sum_; }

private:
    double sum_ = 0.0;
    double compensation_ = 0.0;
};

[[noreturn]] void reject(const char* reason)
{
    throw std::invalid_argument(
        std::string("invalid conservative parent-to-child prolongation: ") + reason);
}

bool shape_is(const Field3D& field, int nx, int ny, int nz) noexcept
{
    return field.size_r() == nx && field.size_th() == ny &&
           field.size_z() == nz;
}

bool aliases(const Field3D& a, const Field3D& b) noexcept
{
    return a.data() != nullptr && a.data() == b.data();
}

struct ConservedValue
{
    double density = 0.0;
    double specific = 0.0;
    double carrier = 0.0;
};

ConservedValue blend(const float& rho_previous,
                     const float& rho_current,
                     const float& specific_previous,
                     const float& specific_current,
                     const ParentTemporalWeights& weights,
                     bool require_nonnegative_specific)
{
    const auto valid_endpoint = [require_nonnegative_specific](double rho,
                                                               double specific) {
        return std::isfinite(rho) && rho > 0.0 && std::isfinite(specific) &&
               (!require_nonnegative_specific || specific >= 0.0);
    };

    ConservedValue result;
    if (weights.current == 0.0)
    {
        const double rp = rho_previous;
        const double sp = specific_previous;
        if (!valid_endpoint(rp, sp))
            reject("source density/specific state is nonfinite or outside its declared bounds");
        result.density = rp;
        result.specific = sp;
        result.carrier = rp * sp;
    }
    else if (weights.previous == 0.0)
    {
        const double rc = rho_current;
        const double sc = specific_current;
        if (!valid_endpoint(rc, sc))
            reject("source density/specific state is nonfinite or outside its declared bounds");
        result.density = rc;
        result.specific = sc;
        result.carrier = rc * sc;
    }
    else
    {
        const double rp = rho_previous;
        const double rc = rho_current;
        const double sp = specific_previous;
        const double sc = specific_current;
        if (!valid_endpoint(rp, sp) || !valid_endpoint(rc, sc))
            reject("source density/specific state is nonfinite or outside its declared bounds");
        result.density = weights.previous * rp + weights.current * rc;
        result.carrier = weights.previous * rp * sp + weights.current * rc * sc;
        result.specific = result.carrier / result.density;
    }
    if (!std::isfinite(result.density) || result.density <= 0.0 ||
        !std::isfinite(result.carrier) || !std::isfinite(result.specific) ||
        result.density > std::numeric_limits<float>::max() ||
        std::abs(result.specific) > std::numeric_limits<float>::max() ||
        (require_nonnegative_specific &&
         (result.carrier < 0.0 || result.specific < 0.0)))
    {
        reject("conserved temporal target is not representable");
    }
    return result;
}

void add_finite(FixedOrderSum& sum, double value)
{
    if (!std::isfinite(value))
        reject("inventory is not finite or representable");
    sum.add(value);
    if (!std::isfinite(sum.value()))
        reject("inventory sum is not finite or representable");
}

double blend_conserved(const float& previous,
                       const float& current,
                       const ParentTemporalWeights& weights)
{
    double result = 0.0;
    if (weights.current == 0.0)
    {
        result = previous;
        if (!std::isfinite(result))
            reject("source conserved face state is nonfinite");
    }
    else if (weights.previous == 0.0)
    {
        result = current;
        if (!std::isfinite(result))
            reject("source conserved face state is nonfinite");
    }
    else
    {
        const double previous_value = previous;
        const double current_value = current;
        if (!std::isfinite(previous_value) || !std::isfinite(current_value))
            reject("source conserved face state is nonfinite");
        result = weights.previous * previous_value +
                 weights.current * current_value;
    }
    if (!std::isfinite(result) ||
        std::abs(result) > std::numeric_limits<float>::max())
    {
        reject("conserved face target is not representable");
    }
    return result;
}

double blend_positive_density(const float& previous,
                              const float& current,
                              const ParentTemporalWeights& weights)
{
    const auto valid = [](double value) {
        return std::isfinite(value) && value > 0.0;
    };
    double result = 0.0;
    if (weights.current == 0.0)
    {
        result = previous;
        if (!valid(result))
            reject("active source density is not finite and positive");
    }
    else if (weights.previous == 0.0)
    {
        result = current;
        if (!valid(result))
            reject("active source density is not finite and positive");
    }
    else
    {
        if (!valid(previous) || !valid(current))
            reject("active source density is not finite and positive");
        result = weights.previous * static_cast<double>(previous) +
                 weights.current * static_cast<double>(current);
    }
    if (!valid(result))
        reject("blended source density is not finite and positive");
    return result;
}

double blend_momentum_density(const float& rho_previous,
                              const float& rho_current,
                              const float& velocity_previous,
                              const float& velocity_current,
                              const ParentTemporalWeights& weights)
{
    const auto valid = [](double rho, double velocity) {
        return std::isfinite(rho) && rho > 0.0 &&
               std::isfinite(velocity);
    };
    double result = 0.0;
    if (weights.current == 0.0)
    {
        if (!valid(rho_previous, velocity_previous))
            reject("active source momentum state is invalid");
        result = static_cast<double>(rho_previous) * velocity_previous;
    }
    else if (weights.previous == 0.0)
    {
        if (!valid(rho_current, velocity_current))
            reject("active source momentum state is invalid");
        result = static_cast<double>(rho_current) * velocity_current;
    }
    else
    {
        if (!valid(rho_previous, velocity_previous) ||
            !valid(rho_current, velocity_current))
        {
            reject("active source momentum state is invalid");
        }
        result = weights.previous * static_cast<double>(rho_previous) *
                     velocity_previous +
                 weights.current * static_cast<double>(rho_current) *
                     velocity_current;
    }
    if (!std::isfinite(result))
        reject("blended source momentum is not finite");
    return result;
}

void require_native_storage_shape(const Field3D& field,
                                  int physical_nx,
                                  int physical_ny,
                                  int nz)
{
    if (!shape_is(field, physical_nx + 2, physical_ny + 2, nz))
        reject("field does not use the native halo-bearing C-grid shape");
}

double committed_child_scalar_mass(const Field3D& child_rho,
                                   int ci,
                                   int cj,
                                   int k,
                                   double cell_area,
                                   double dz)
{
    const double rho = child_rho[ci + 1][cj + 1][k];
    const double mass = rho * cell_area * dz;
    if (!std::isfinite(rho) || rho <= 0.0 ||
        !std::isfinite(mass) || mass <= 0.0)
    {
        reject("committed child density does not define positive finite mass");
    }
    return mass;
}

} // namespace

void prolong_conserved_scalar_flat(
    const OneWayNestRelationshipV1& relationship,
    const ParentTimeBracketV1& bracket,
    const std::vector<double>& layer_thickness_m,
    const Field3D& previous_dry_density,
    const Field3D& current_dry_density,
    const Field3D& previous_specific,
    const Field3D& current_specific,
    bool require_nonnegative_specific,
    Field3D& child_dry_density,
    Field3D& child_specific,
    ConservedScalarProlongationReport& report)
{
    const auto canonical_weights = parent_temporal_weights(
        relationship, bracket);
    const bool parent_shapes =
        shape_is(previous_dry_density, relationship.parent_cells_x,
                 relationship.parent_cells_y, relationship.levels_z) &&
        shape_is(current_dry_density, relationship.parent_cells_x,
                 relationship.parent_cells_y, relationship.levels_z) &&
        shape_is(previous_specific, relationship.parent_cells_x,
                 relationship.parent_cells_y, relationship.levels_z) &&
        shape_is(current_specific, relationship.parent_cells_x,
                 relationship.parent_cells_y, relationship.levels_z);
    const bool child_shapes =
        shape_is(child_dry_density, relationship.child_cells_x,
                 relationship.child_cells_y, relationship.levels_z) &&
        shape_is(child_specific, relationship.child_cells_x,
                 relationship.child_cells_y, relationship.levels_z);
    if (!parent_shapes || !child_shapes ||
        layer_thickness_m.size() !=
            static_cast<std::size_t>(relationship.levels_z))
    {
        reject("field or vertical-metric shape does not match the relationship");
    }
    if (aliases(child_dry_density, child_specific) ||
        aliases(child_dry_density, previous_dry_density) ||
        aliases(child_dry_density, current_dry_density) ||
        aliases(child_dry_density, previous_specific) ||
        aliases(child_dry_density, current_specific) ||
        aliases(child_specific, previous_dry_density) ||
        aliases(child_specific, current_dry_density) ||
        aliases(child_specific, previous_specific) ||
        aliases(child_specific, current_specific))
    {
        reject("source and destination storage must not alias");
    }
    for (double dz : layer_thickness_m)
    {
        if (!std::isfinite(dz) || dz <= 0.0)
            reject("vertical layer thickness must be finite and positive");
    }

    const double parent_area = relationship.parent_dx_m * relationship.parent_dy_m;
    const double child_area = relationship.child_dx_m * relationship.child_dy_m;
    if (!std::isfinite(parent_area) || parent_area <= 0.0 ||
        !std::isfinite(child_area) || child_area <= 0.0)
    {
        reject("horizontal cell area is not finite or representable");
    }

    FixedOrderSum target_dry;
    FixedOrderSum target_carrier;
    for (int pi = relationship.child_origin_parent_cell_i;
         pi < relationship.child_origin_parent_cell_i +
                  relationship.child_parent_span_x;
         ++pi)
    {
        for (int pj = relationship.child_origin_parent_cell_j;
             pj < relationship.child_origin_parent_cell_j +
                      relationship.child_parent_span_y;
             ++pj)
        {
            for (int k = 0; k + 1 < relationship.levels_z; ++k)
            {
                const auto value = blend(previous_dry_density[pi][pj][k],
                                         current_dry_density[pi][pj][k],
                                         previous_specific[pi][pj][k],
                                         current_specific[pi][pj][k],
                                         canonical_weights,
                                         require_nonnegative_specific);
                const double volume =
                    parent_area * layer_thickness_m[static_cast<std::size_t>(k)];
                if (!std::isfinite(volume))
                    reject("parent cell volume is not finite or representable");
                add_finite(target_dry, value.density * volume);
                add_finite(target_carrier, value.carrier * volume);
            }
        }
    }

    // Predict the exact FP32 commit and close its report before mutation.
    FixedOrderSum committed_dry;
    FixedOrderSum committed_carrier;
    for (int ci = 0; ci < relationship.child_cells_x; ++ci)
    {
        const int pi = relationship.child_origin_parent_cell_i +
                       ci / relationship.parent_ratio_x;
        for (int cj = 0; cj < relationship.child_cells_y; ++cj)
        {
            const int pj = relationship.child_origin_parent_cell_j +
                           cj / relationship.parent_ratio_y;
            for (int k = 0; k + 1 < relationship.levels_z; ++k)
            {
                const auto value = blend(previous_dry_density[pi][pj][k],
                                         current_dry_density[pi][pj][k],
                                         previous_specific[pi][pj][k],
                                         current_specific[pi][pj][k],
                                         canonical_weights,
                                         require_nonnegative_specific);
                const double volume =
                    child_area * layer_thickness_m[static_cast<std::size_t>(k)];
                if (!std::isfinite(volume))
                    reject("child cell volume is not finite or representable");
                const float stored_rho = static_cast<float>(value.density);
                const float stored_specific = static_cast<float>(value.specific);
                add_finite(committed_dry,
                           static_cast<double>(stored_rho) * volume);
                add_finite(committed_carrier,
                           static_cast<double>(stored_rho) *
                               static_cast<double>(stored_specific) * volume);
            }
        }
    }

    ConservedScalarProlongationReport candidate;
    candidate.target_dry_mass_kg = target_dry.value();
    candidate.committed_dry_mass_kg = committed_dry.value();
    candidate.dry_storage_residual_kg =
        candidate.target_dry_mass_kg - candidate.committed_dry_mass_kg;
    candidate.target_carrier_inventory = target_carrier.value();
    candidate.committed_carrier_inventory = committed_carrier.value();
    candidate.carrier_storage_residual =
        candidate.target_carrier_inventory - candidate.committed_carrier_inventory;
    if (!std::isfinite(candidate.dry_storage_residual_kg) ||
        !std::isfinite(candidate.carrier_storage_residual))
    {
        reject("storage-rounding residual is not finite or representable");
    }
    candidate.parent_cells =
        static_cast<std::size_t>(relationship.child_parent_span_x) *
        static_cast<std::size_t>(relationship.child_parent_span_y) *
        static_cast<std::size_t>(relationship.levels_z);
    candidate.child_cells =
        static_cast<std::size_t>(relationship.child_cells_x) *
        static_cast<std::size_t>(relationship.child_cells_y) *
        static_cast<std::size_t>(relationship.levels_z);

    // All rejection paths have completed; commit without allocating.
    for (int ci = 0; ci < relationship.child_cells_x; ++ci)
    {
        const int pi = relationship.child_origin_parent_cell_i +
                       ci / relationship.parent_ratio_x;
        for (int cj = 0; cj < relationship.child_cells_y; ++cj)
        {
            const int pj = relationship.child_origin_parent_cell_j +
                           cj / relationship.parent_ratio_y;
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const auto value = blend(previous_dry_density[pi][pj][k],
                                         current_dry_density[pi][pj][k],
                                         previous_specific[pi][pj][k],
                                         current_specific[pi][pj][k],
                                         canonical_weights,
                                         require_nonnegative_specific);
                child_dry_density[ci][cj][k] =
                    static_cast<float>(value.density);
                child_specific[ci][cj][k] =
                    static_cast<float>(value.specific);
            }
        }
    }
    report = candidate;
}

void prolong_conserved_scalar_flat_storage(
    const OneWayNestRelationshipV1& relationship,
    const ParentTimeBracketV1& bracket,
    const std::vector<double>& layer_thickness_m,
    const Field3D& previous_dry_density,
    const Field3D& current_dry_density,
    const Field3D& previous_specific,
    const Field3D& current_specific,
    bool require_nonnegative_specific,
    Field3D& child_dry_density,
    Field3D& child_specific,
    ConservedScalarProlongationReport& report)
{
    const auto weights = parent_temporal_weights(relationship, bracket);
    const Field3D* parent_fields[] = {&previous_dry_density,
                                     &current_dry_density,
                                     &previous_specific,
                                     &current_specific};
    for (const Field3D* field : parent_fields)
        require_native_storage_shape(*field, relationship.parent_cells_x,
                                     relationship.parent_cells_y,
                                     relationship.levels_z);
    require_native_storage_shape(child_dry_density,
                                 relationship.child_cells_x,
                                 relationship.child_cells_y,
                                 relationship.levels_z);
    require_native_storage_shape(child_specific,
                                 relationship.child_cells_x,
                                 relationship.child_cells_y,
                                 relationship.levels_z);
    if (aliases(child_dry_density, child_specific))
        reject("child density and specific destinations must not alias");
    for (const Field3D* parent : parent_fields)
        if (aliases(child_dry_density, *parent) ||
            aliases(child_specific, *parent))
            reject("parent and child scalar storage must not alias");
    if (layer_thickness_m.size() !=
        static_cast<std::size_t>(relationship.levels_z))
        reject("vertical metric does not match the relationship");
    for (double dz : layer_thickness_m)
        if (!std::isfinite(dz) || dz <= 0.0)
            reject("vertical layer thickness must be finite and positive");

    const double parent_area = relationship.parent_dx_m *
                               relationship.parent_dy_m;
    const double child_area = relationship.child_dx_m *
                              relationship.child_dy_m;
    FixedOrderSum target_dry;
    FixedOrderSum target_carrier;
    FixedOrderSum committed_dry;
    FixedOrderSum committed_carrier;
    for (int pi = relationship.child_origin_parent_cell_i;
         pi < relationship.child_origin_parent_cell_i +
                  relationship.child_parent_span_x; ++pi)
        for (int pj = relationship.child_origin_parent_cell_j;
             pj < relationship.child_origin_parent_cell_j +
                      relationship.child_parent_span_y; ++pj)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const auto value = blend(
                    previous_dry_density[pi + 1][pj + 1][k],
                    current_dry_density[pi + 1][pj + 1][k],
                    previous_specific[pi + 1][pj + 1][k],
                    current_specific[pi + 1][pj + 1][k], weights,
                    require_nonnegative_specific);
                const double volume = parent_area *
                    layer_thickness_m[static_cast<std::size_t>(k)];
                add_finite(target_dry, value.density * volume);
                add_finite(target_carrier, value.carrier * volume);
            }
    for (int ci = 0; ci < relationship.child_cells_x; ++ci)
        for (int cj = 0; cj < relationship.child_cells_y; ++cj)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const int pi = relationship.child_origin_parent_cell_i +
                               ci / relationship.parent_ratio_x;
                const int pj = relationship.child_origin_parent_cell_j +
                               cj / relationship.parent_ratio_y;
                const auto value = blend(
                    previous_dry_density[pi + 1][pj + 1][k],
                    current_dry_density[pi + 1][pj + 1][k],
                    previous_specific[pi + 1][pj + 1][k],
                    current_specific[pi + 1][pj + 1][k], weights,
                    require_nonnegative_specific);
                const float stored_rho = static_cast<float>(value.density);
                const float stored_specific = static_cast<float>(value.specific);
                const double volume = child_area *
                    layer_thickness_m[static_cast<std::size_t>(k)];
                add_finite(committed_dry, stored_rho * volume);
                add_finite(committed_carrier,
                           static_cast<double>(stored_rho) *
                               stored_specific * volume);
            }

    ConservedScalarProlongationReport candidate;
    candidate.target_dry_mass_kg = target_dry.value();
    candidate.committed_dry_mass_kg = committed_dry.value();
    candidate.dry_storage_residual_kg = target_dry.value() -
                                        committed_dry.value();
    candidate.target_carrier_inventory = target_carrier.value();
    candidate.committed_carrier_inventory = committed_carrier.value();
    candidate.carrier_storage_residual = target_carrier.value() -
                                         committed_carrier.value();
    candidate.parent_cells = static_cast<std::size_t>(
        relationship.child_parent_span_x) *
        relationship.child_parent_span_y * relationship.levels_z;
    candidate.child_cells = static_cast<std::size_t>(
        relationship.child_cells_x) * relationship.child_cells_y *
        relationship.levels_z;
    if (!std::isfinite(candidate.dry_storage_residual_kg) ||
        !std::isfinite(candidate.carrier_storage_residual))
        reject("scalar storage residual is not finite");

    for (int ci = 0; ci < relationship.child_cells_x; ++ci)
        for (int cj = 0; cj < relationship.child_cells_y; ++cj)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const int pi = relationship.child_origin_parent_cell_i +
                               ci / relationship.parent_ratio_x;
                const int pj = relationship.child_origin_parent_cell_j +
                               cj / relationship.parent_ratio_y;
                const auto value = blend(
                    previous_dry_density[pi + 1][pj + 1][k],
                    current_dry_density[pi + 1][pj + 1][k],
                    previous_specific[pi + 1][pj + 1][k],
                    current_specific[pi + 1][pj + 1][k], weights,
                    require_nonnegative_specific);
                child_dry_density[ci + 1][cj + 1][k] =
                    static_cast<float>(value.density);
                child_specific[ci + 1][cj + 1][k] =
                    static_cast<float>(value.specific);
            }
    report = candidate;
}

void prolong_conserved_horizontal_face_flat(
    const OneWayNestRelationshipV1& relationship,
    const ParentTimeBracketV1& bracket,
    NestedHorizontalLocation owner,
    const std::vector<double>& layer_thickness_m,
    const Field3D& previous_conserved_density,
    const Field3D& current_conserved_density,
    Field3D& child_conserved_density,
    ConservedFaceProlongationReport& report)
{
    const auto canonical_weights = parent_temporal_weights(
        relationship, bracket);
    if (owner != NestedHorizontalLocation::XFace &&
        owner != NestedHorizontalLocation::YFace)
    {
        reject("conserved face owner must be an x or y face");
    }

    const bool x_face = owner == NestedHorizontalLocation::XFace;
    const int parent_nx = relationship.parent_cells_x + (x_face ? 1 : 0);
    const int parent_ny = relationship.parent_cells_y + (x_face ? 0 : 1);
    const int child_nx = relationship.child_cells_x + (x_face ? 1 : 0);
    const int child_ny = relationship.child_cells_y + (x_face ? 0 : 1);
    if (!shape_is(previous_conserved_density, parent_nx, parent_ny,
                  relationship.levels_z) ||
        !shape_is(current_conserved_density, parent_nx, parent_ny,
                  relationship.levels_z) ||
        !shape_is(child_conserved_density, child_nx, child_ny,
                  relationship.levels_z) ||
        layer_thickness_m.size() !=
            static_cast<std::size_t>(relationship.levels_z))
    {
        reject("face field or vertical-metric shape does not match the relationship");
    }
    if (aliases(child_conserved_density, previous_conserved_density) ||
        aliases(child_conserved_density, current_conserved_density))
    {
        reject("source and destination storage must not alias");
    }
    for (double dz : layer_thickness_m)
    {
        if (!std::isfinite(dz) || dz <= 0.0)
            reject("vertical layer thickness must be finite and positive");
    }

    const int axis_origin = x_face
        ? relationship.child_origin_parent_cell_i
        : relationship.child_origin_parent_cell_j;
    const int axis_span = x_face
        ? relationship.child_parent_span_x
        : relationship.child_parent_span_y;
    const int axis_end = axis_origin + axis_span;
    const int ratio = x_face
        ? relationship.parent_ratio_x
        : relationship.parent_ratio_y;
    const double normal_spacing = x_face
        ? relationship.parent_dx_m
        : relationship.parent_dy_m;
    const double parent_transverse_spacing = x_face
        ? relationship.parent_dy_m
        : relationship.parent_dx_m;
    const double child_transverse_spacing = x_face
        ? relationship.child_dy_m
        : relationship.child_dx_m;

    const auto source_value = [&](int face, int transverse, int k) {
        return x_face
            ? blend_conserved(previous_conserved_density[face][transverse][k],
                              current_conserved_density[face][transverse][k],
                              canonical_weights)
            : blend_conserved(previous_conserved_density[transverse][face][k],
                              current_conserved_density[transverse][face][k],
                              canonical_weights);
    };
    const auto source_width = [&](int face) {
        return (face == axis_origin || face == axis_end) ? 0.5 : 1.0;
    };

    FixedOrderSum target;
    const int parent_transverse_origin = x_face
        ? relationship.child_origin_parent_cell_j
        : relationship.child_origin_parent_cell_i;
    const int parent_transverse_span = x_face
        ? relationship.child_parent_span_y
        : relationship.child_parent_span_x;
    for (int face = axis_origin; face <= axis_end; ++face)
    {
        for (int transverse = parent_transverse_origin;
             transverse < parent_transverse_origin + parent_transverse_span;
             ++transverse)
        {
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const double volume = source_width(face) * normal_spacing *
                    parent_transverse_spacing *
                    layer_thickness_m[static_cast<std::size_t>(k)];
                if (!std::isfinite(volume))
                    reject("parent face dual volume is not finite or representable");
                add_finite(target, source_value(face, transverse, k) * volume);
            }
        }
    }

    const auto remapped = [&](int child_face, int child_transverse, int k) {
        const double center = axis_origin +
            static_cast<double>(child_face) / static_cast<double>(ratio);
        const double left = std::max(
            static_cast<double>(axis_origin), center - 0.5 / ratio);
        const double right = std::min(
            static_cast<double>(axis_end), center + 0.5 / ratio);
        const double width = right - left;
        const int first_face = std::max(
            axis_origin, static_cast<int>(std::floor(left + 0.5)));
        const int last_face = std::min(
            axis_end,
            static_cast<int>(std::floor(std::nextafter(
                right, -std::numeric_limits<double>::infinity()) + 0.5)));
        const int parent_transverse = parent_transverse_origin +
            child_transverse /
                (x_face ? relationship.parent_ratio_y
                        : relationship.parent_ratio_x);
        double weighted = 0.0;
        for (int face = first_face; face <= last_face; ++face)
        {
            const double source_left = std::max(
                static_cast<double>(axis_origin), face - 0.5);
            const double source_right = std::min(
                static_cast<double>(axis_end), face + 0.5);
            const double overlap = std::max(
                0.0, std::min(right, source_right) -
                         std::max(left, source_left));
            weighted += overlap * source_value(face, parent_transverse, k);
        }
        const double value = weighted / width;
        if (!std::isfinite(value) ||
            std::abs(value) > std::numeric_limits<float>::max())
        {
            reject("remapped conserved face target is not representable");
        }
        return std::pair<double, double>{value, width};
    };

    const int child_axis_faces = (x_face ? relationship.child_cells_x
                                         : relationship.child_cells_y) + 1;
    const int child_transverse_cells = x_face ? relationship.child_cells_y
                                              : relationship.child_cells_x;
    FixedOrderSum committed;
    for (int face = 0; face < child_axis_faces; ++face)
    {
        for (int transverse = 0; transverse < child_transverse_cells;
             ++transverse)
        {
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const auto [value, width] = remapped(face, transverse, k);
                const double volume = width * normal_spacing *
                    child_transverse_spacing *
                    layer_thickness_m[static_cast<std::size_t>(k)];
                if (!std::isfinite(volume))
                    reject("child face dual volume is not finite or representable");
                add_finite(committed,
                           static_cast<double>(static_cast<float>(value)) * volume);
            }
        }
    }

    ConservedFaceProlongationReport candidate;
    candidate.target_inventory = target.value();
    candidate.committed_inventory = committed.value();
    candidate.storage_residual = candidate.target_inventory -
                                 candidate.committed_inventory;
    if (!std::isfinite(candidate.storage_residual))
        reject("face storage-rounding residual is not finite or representable");
    candidate.parent_owners = static_cast<std::size_t>(axis_span + 1) *
        static_cast<std::size_t>(parent_transverse_span) *
        static_cast<std::size_t>(relationship.levels_z);
    candidate.child_owners = static_cast<std::size_t>(child_axis_faces) *
        static_cast<std::size_t>(child_transverse_cells) *
        static_cast<std::size_t>(relationship.levels_z);

    for (int face = 0; face < child_axis_faces; ++face)
    {
        for (int transverse = 0; transverse < child_transverse_cells;
             ++transverse)
        {
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const float value = static_cast<float>(
                    remapped(face, transverse, k).first);
                if (x_face)
                    child_conserved_density[face][transverse][k] = value;
                else
                    child_conserved_density[transverse][face][k] = value;
            }
        }
    }
    report = candidate;
}

void prolong_cgrid_momentum_flat_storage(
    const OneWayNestRelationshipV1& relationship,
    const ParentTimeBracketV1& bracket,
    const std::vector<double>& layer_thickness_m,
    const Field3D& previous_parent_dry_density,
    const Field3D& current_parent_dry_density,
    const Field3D& previous_parent_u,
    const Field3D& current_parent_u,
    const Field3D& previous_parent_v,
    const Field3D& current_parent_v,
    const Field3D& previous_parent_w,
    const Field3D& current_parent_w,
    const Field3D& committed_child_dry_density,
    Field3D& child_u,
    Field3D& child_v,
    Field3D& child_w,
    ConservedMomentumProlongationReport& report)
{
    const auto canonical_weights = parent_temporal_weights(
        relationship, bracket);
    if (layer_thickness_m.size() !=
        static_cast<std::size_t>(relationship.levels_z))
    {
        reject("vertical metric does not match the relationship");
    }
    for (double dz : layer_thickness_m)
    {
        if (!std::isfinite(dz) || dz <= 0.0)
            reject("vertical layer thickness must be finite and positive");
    }

    const Field3D* parent_fields[] = {
        &previous_parent_dry_density, &current_parent_dry_density,
        &previous_parent_u, &current_parent_u,
        &previous_parent_v, &current_parent_v,
        &previous_parent_w, &current_parent_w};
    for (const Field3D* field : parent_fields)
    {
        require_native_storage_shape(*field, relationship.parent_cells_x,
                                     relationship.parent_cells_y,
                                     relationship.levels_z);
    }
    const Field3D* child_fields[] = {
        &committed_child_dry_density, &child_u, &child_v, &child_w};
    for (const Field3D* field : child_fields)
    {
        require_native_storage_shape(*field, relationship.child_cells_x,
                                     relationship.child_cells_y,
                                     relationship.levels_z);
    }

    for (int ci = 0; ci < relationship.child_cells_x; ++ci)
        for (int cj = 0; cj < relationship.child_cells_y; ++cj)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const int pi = relationship.child_origin_parent_cell_i +
                               ci / relationship.parent_ratio_x;
                const int pj = relationship.child_origin_parent_cell_j +
                               cj / relationship.parent_ratio_y;
                const float canonical_rho = static_cast<float>(
                    blend_positive_density(
                        previous_parent_dry_density[pi + 1][pj + 1][k],
                        current_parent_dry_density[pi + 1][pj + 1][k],
                        canonical_weights));
                if (committed_child_dry_density[ci + 1][cj + 1][k] !=
                    canonical_rho)
                {
                    reject("committed child density is not the canonical parent target");
                }
            }
    if (aliases(child_u, child_v) || aliases(child_u, child_w) ||
        aliases(child_v, child_w) ||
        aliases(child_u, committed_child_dry_density) ||
        aliases(child_v, committed_child_dry_density) ||
        aliases(child_w, committed_child_dry_density))
    {
        reject("child velocity destinations must not alias");
    }
    for (const Field3D* parent : parent_fields)
    {
        if (aliases(child_u, *parent) || aliases(child_v, *parent) ||
            aliases(child_w, *parent))
        {
            reject("parent and child momentum storage must not alias");
        }
    }

    const double child_area = relationship.child_dx_m *
                              relationship.child_dy_m;
    if (!std::isfinite(child_area) || child_area <= 0.0)
        reject("child horizontal area is not finite and positive");

    const auto horizontal_candidate = [&](bool x_face, int face,
                                          int transverse, int k) {
        const int axis_origin = x_face
            ? relationship.child_origin_parent_cell_i
            : relationship.child_origin_parent_cell_j;
        const int axis_span = x_face
            ? relationship.child_parent_span_x
            : relationship.child_parent_span_y;
        const int ratio = x_face ? relationship.parent_ratio_x
                                 : relationship.parent_ratio_y;
        const int transverse_origin = x_face
            ? relationship.child_origin_parent_cell_j
            : relationship.child_origin_parent_cell_i;
        const int transverse_ratio = x_face
            ? relationship.parent_ratio_y
            : relationship.parent_ratio_x;
        const int parent_transverse = transverse_origin +
                                      transverse / transverse_ratio;
        const double center = axis_origin +
            static_cast<double>(face) / ratio;
        const double left = std::max(static_cast<double>(axis_origin),
                                    center - 0.5 / ratio);
        const double right = std::min(
            static_cast<double>(axis_origin + axis_span),
            center + 0.5 / ratio);
        FixedOrderSum mass_line;
        FixedOrderSum momentum_line;
        for (int parent_face = axis_origin;
             parent_face <= axis_origin + axis_span; ++parent_face)
        {
            const int scalar_sides[2] = {parent_face - 1, parent_face};
            const double side_left[2] = {parent_face - 0.5,
                                         static_cast<double>(parent_face)};
            const double side_right[2] = {static_cast<double>(parent_face),
                                          parent_face + 0.5};
            for (int side = 0; side < 2; ++side)
            {
                const int scalar_axis = scalar_sides[side];
                if (scalar_axis < axis_origin ||
                    scalar_axis >= axis_origin + axis_span)
                {
                    continue;
                }
                const double overlap = std::max(
                    0.0, std::min(right, side_right[side]) -
                             std::max(left, side_left[side]));
                if (overlap == 0.0)
                    continue;
                const int rho_i = (x_face ? scalar_axis : parent_transverse) + 1;
                const int rho_j = (x_face ? parent_transverse : scalar_axis) + 1;
                const int velocity_i = x_face ? parent_face
                                              : parent_transverse + 1;
                const int velocity_j = x_face ? parent_transverse + 1
                                              : parent_face;
                const Field3D& previous_velocity = x_face
                    ? previous_parent_u : previous_parent_v;
                const Field3D& current_velocity = x_face
                    ? current_parent_u : current_parent_v;
                add_finite(mass_line, overlap * blend_positive_density(
                    previous_parent_dry_density[rho_i][rho_j][k],
                    current_parent_dry_density[rho_i][rho_j][k],
                    canonical_weights));
                add_finite(momentum_line, overlap * blend_momentum_density(
                    previous_parent_dry_density[rho_i][rho_j][k],
                    current_parent_dry_density[rho_i][rho_j][k],
                    previous_velocity[velocity_i][velocity_j][k],
                    current_velocity[velocity_i][velocity_j][k],
                    canonical_weights));
            }
        }
        const double normal_spacing = x_face ? relationship.parent_dx_m
                                             : relationship.parent_dy_m;
        const double transverse_spacing = x_face ? relationship.child_dy_m
                                                 : relationship.child_dx_m;
        const double target_momentum = momentum_line.value() *
            normal_spacing * transverse_spacing *
            layer_thickness_m[static_cast<std::size_t>(k)];
        const double target_mass = mass_line.value() *
            normal_spacing * transverse_spacing *
            layer_thickness_m[static_cast<std::size_t>(k)];

        const int child_axis_cells = x_face
            ? relationship.child_cells_x : relationship.child_cells_y;
        const int left_cell = face - 1;
        const int right_cell = face;
        double dual_mass = 0.0;
        if (left_cell >= 0)
        {
            const int ci = x_face ? left_cell : transverse;
            const int cj = x_face ? transverse : left_cell;
            dual_mass += 0.5 * committed_child_scalar_mass(
                committed_child_dry_density, ci, cj, k, child_area,
                layer_thickness_m[static_cast<std::size_t>(k)]);
        }
        if (right_cell < child_axis_cells)
        {
            const int ci = x_face ? right_cell : transverse;
            const int cj = x_face ? transverse : right_cell;
            dual_mass += 0.5 * committed_child_scalar_mass(
                committed_child_dry_density, ci, cj, k, child_area,
                layer_thickness_m[static_cast<std::size_t>(k)]);
        }
        const double velocity = target_momentum / dual_mass;
        if (!std::isfinite(target_momentum) || !std::isfinite(velocity) ||
            std::abs(velocity) > std::numeric_limits<float>::max())
        {
            reject("child horizontal momentum target is not representable");
        }
        struct Candidate { double momentum; double target_mass; double mass; };
        return Candidate{target_momentum, target_mass, dual_mass};
    };

    const auto vertical_candidate = [&](int ci, int cj, int k) {
        const int pi = relationship.child_origin_parent_cell_i +
                       ci / relationship.parent_ratio_x;
        const int pj = relationship.child_origin_parent_cell_j +
                       cj / relationship.parent_ratio_y;
        const auto endpoint = [&](const Field3D& rho,
                                  const Field3D& velocity) {
            const double lower_mass = 0.5 * static_cast<double>(
                rho[pi + 1][pj + 1][k]) * child_area *
                layer_thickness_m[static_cast<std::size_t>(k)];
            double mass = lower_mass;
            if (k + 1 < relationship.levels_z)
            {
                mass += 0.5 * static_cast<double>(
                    rho[pi + 1][pj + 1][k + 1]) * child_area *
                    layer_thickness_m[static_cast<std::size_t>(k + 1)];
            }
            const double w = velocity[pi + 1][pj + 1][k];
            if (!std::isfinite(mass) || mass <= 0.0 || !std::isfinite(w))
                reject("active source vertical momentum state is invalid");
            return std::pair<double, double>{mass, mass * w};
        };
        double target_mass = 0.0;
        double target_momentum = 0.0;
        if (canonical_weights.current == 0.0)
        {
            const auto value = endpoint(previous_parent_dry_density,
                                        previous_parent_w);
            target_mass = value.first;
            target_momentum = value.second;
        }
        else if (canonical_weights.previous == 0.0)
        {
            const auto value = endpoint(current_parent_dry_density,
                                        current_parent_w);
            target_mass = value.first;
            target_momentum = value.second;
        }
        else
        {
            const auto previous = endpoint(previous_parent_dry_density,
                                           previous_parent_w);
            const auto current = endpoint(current_parent_dry_density,
                                          current_parent_w);
            target_mass = canonical_weights.previous * previous.first +
                          canonical_weights.current * current.first;
            target_momentum = canonical_weights.previous * previous.second +
                              canonical_weights.current * current.second;
        }
        double dual_mass = 0.5 * committed_child_scalar_mass(
            committed_child_dry_density, ci, cj, k, child_area,
            layer_thickness_m[static_cast<std::size_t>(k)]);
        if (k + 1 < relationship.levels_z)
        {
            dual_mass += 0.5 * committed_child_scalar_mass(
                committed_child_dry_density, ci, cj, k + 1, child_area,
                layer_thickness_m[static_cast<std::size_t>(k + 1)]);
        }
        const double velocity = target_momentum / dual_mass;
        if (!std::isfinite(target_momentum) || !std::isfinite(velocity) ||
            std::abs(velocity) > std::numeric_limits<float>::max())
        {
            reject("child vertical momentum target is not representable");
        }
        struct Candidate { double momentum; double target_mass; double mass; };
        return Candidate{target_momentum, target_mass, dual_mass};
    };

    FixedOrderSum target_mass_x;
    FixedOrderSum committed_mass_x;
    FixedOrderSum target_x;
    FixedOrderSum committed_x;
    FixedOrderSum target_mass_y;
    FixedOrderSum committed_mass_y;
    FixedOrderSum target_y;
    FixedOrderSum committed_y;
    FixedOrderSum target_mass_z;
    FixedOrderSum committed_mass_z;
    FixedOrderSum target_z;
    FixedOrderSum committed_z;
    for (int face = 0; face <= relationship.child_cells_x; ++face)
        for (int cj = 0; cj < relationship.child_cells_y; ++cj)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const auto [momentum, target_mass, mass] =
                    horizontal_candidate(true, face, cj, k);
                const float stored = static_cast<float>(momentum / mass);
                add_finite(target_mass_x, target_mass);
                add_finite(committed_mass_x, mass);
                add_finite(target_x, momentum);
                add_finite(committed_x, mass * stored);
            }
    for (int ci = 0; ci < relationship.child_cells_x; ++ci)
        for (int face = 0; face <= relationship.child_cells_y; ++face)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const auto [momentum, target_mass, mass] =
                    horizontal_candidate(false, face, ci, k);
                const float stored = static_cast<float>(momentum / mass);
                add_finite(target_mass_y, target_mass);
                add_finite(committed_mass_y, mass);
                add_finite(target_y, momentum);
                add_finite(committed_y, mass * stored);
            }
    for (int ci = 0; ci < relationship.child_cells_x; ++ci)
        for (int cj = 0; cj < relationship.child_cells_y; ++cj)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const auto [momentum, target_mass, mass] =
                    vertical_candidate(ci, cj, k);
                const float stored = static_cast<float>(momentum / mass);
                add_finite(target_mass_z, target_mass);
                add_finite(committed_mass_z, mass);
                add_finite(target_z, momentum);
                add_finite(committed_z, mass * stored);
            }

    ConservedMomentumProlongationReport candidate;
    candidate.target_x_dual_dry_mass_kg = target_mass_x.value();
    candidate.committed_x_dual_dry_mass_kg = committed_mass_x.value();
    candidate.x_dry_mass_residual_kg = target_mass_x.value() -
                                       committed_mass_x.value();
    candidate.target_x_momentum_kgms = target_x.value();
    candidate.committed_x_momentum_kgms = committed_x.value();
    candidate.x_storage_residual_kgms = target_x.value() - committed_x.value();
    candidate.target_y_momentum_kgms = target_y.value();
    candidate.target_y_dual_dry_mass_kg = target_mass_y.value();
    candidate.committed_y_dual_dry_mass_kg = committed_mass_y.value();
    candidate.y_dry_mass_residual_kg = target_mass_y.value() -
                                       committed_mass_y.value();
    candidate.committed_y_momentum_kgms = committed_y.value();
    candidate.y_storage_residual_kgms = target_y.value() - committed_y.value();
    candidate.target_z_momentum_kgms = target_z.value();
    candidate.target_z_dual_dry_mass_kg = target_mass_z.value();
    candidate.committed_z_dual_dry_mass_kg = committed_mass_z.value();
    candidate.z_dry_mass_residual_kg = target_mass_z.value() -
                                       committed_mass_z.value();
    candidate.committed_z_momentum_kgms = committed_z.value();
    candidate.z_storage_residual_kgms = target_z.value() - committed_z.value();
    candidate.child_x_faces = static_cast<std::size_t>(
        relationship.child_cells_x + 1) * relationship.child_cells_y *
        relationship.levels_z;
    candidate.child_y_faces = static_cast<std::size_t>(
        relationship.child_cells_y + 1) * relationship.child_cells_x *
        relationship.levels_z;
    candidate.child_z_faces = static_cast<std::size_t>(
        relationship.child_cells_x) * relationship.child_cells_y *
        relationship.levels_z;
    const double residuals[] = {candidate.x_dry_mass_residual_kg,
                                candidate.y_dry_mass_residual_kg,
                                candidate.z_dry_mass_residual_kg,
                                candidate.x_storage_residual_kgms,
                                candidate.y_storage_residual_kgms,
                                candidate.z_storage_residual_kgms};
    if (!std::all_of(std::begin(residuals), std::end(residuals),
                     [](double value) { return std::isfinite(value); }))
    {
        reject("momentum storage residual is not finite");
    }

    for (int face = 0; face <= relationship.child_cells_x; ++face)
        for (int cj = 0; cj < relationship.child_cells_y; ++cj)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const auto [momentum, target_mass, mass] =
                    horizontal_candidate(true, face, cj, k);
                static_cast<void>(target_mass);
                child_u[face][cj + 1][k] =
                    static_cast<float>(momentum / mass);
            }
    for (int ci = 0; ci < relationship.child_cells_x; ++ci)
        for (int face = 0; face <= relationship.child_cells_y; ++face)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const auto [momentum, target_mass, mass] =
                    horizontal_candidate(false, face, ci, k);
                static_cast<void>(target_mass);
                child_v[ci + 1][face][k] =
                    static_cast<float>(momentum / mass);
            }
    for (int ci = 0; ci < relationship.child_cells_x; ++ci)
        for (int cj = 0; cj < relationship.child_cells_y; ++cj)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const auto [momentum, target_mass, mass] =
                    vertical_candidate(ci, cj, k);
                static_cast<void>(target_mass);
                child_w[ci + 1][cj + 1][k] =
                    static_cast<float>(momentum / mass);
            }
    report = candidate;
}

void diagnose_child_pressure_flat_storage(
    const OneWayNestRelationshipV1& relationship,
    const std::vector<double>& layer_thickness_m,
    const Field3D& committed_child_dry_density,
    const Field3D& committed_child_potential_temperature,
    Field3D& child_pressure,
    PressureDiagnosisReport& report)
{
    require_native_storage_shape(committed_child_dry_density,
                                 relationship.child_cells_x,
                                 relationship.child_cells_y,
                                 relationship.levels_z);
    require_native_storage_shape(committed_child_potential_temperature,
                                 relationship.child_cells_x,
                                 relationship.child_cells_y,
                                 relationship.levels_z);
    require_native_storage_shape(child_pressure,
                                 relationship.child_cells_x,
                                 relationship.child_cells_y,
                                 relationship.levels_z);
    if (aliases(child_pressure, committed_child_dry_density) ||
        aliases(child_pressure, committed_child_potential_temperature))
    {
        reject("pressure destination must not alias its EOS sources");
    }
    if (layer_thickness_m.size() !=
        static_cast<std::size_t>(relationship.levels_z))
    {
        reject("vertical metric does not match the relationship");
    }
    const double child_area = relationship.child_dx_m *
                              relationship.child_dy_m;
    FixedOrderSum target;
    FixedOrderSum committed;
    for (int ci = 0; ci < relationship.child_cells_x; ++ci)
        for (int cj = 0; cj < relationship.child_cells_y; ++cj)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const double dz = layer_thickness_m[static_cast<std::size_t>(k)];
                if (!std::isfinite(dz) || dz <= 0.0)
                    reject("vertical layer thickness must be finite and positive");
                const float rho = committed_child_dry_density[ci + 1][cj + 1][k];
                const float theta =
                    committed_child_potential_temperature[ci + 1][cj + 1][k];
                const double diagnosed = tmv::dynamics_eos::pressure_fp64(
                    rho, theta);
                const float stored = static_cast<float>(diagnosed);
                const double volume = child_area * dz;
                if (!std::isfinite(rho) || rho <= 0.0f ||
                    !std::isfinite(theta) || theta <= 0.0f ||
                    rho < density_min_kgm3 || rho > density_max_kgm3 ||
                    theta < theta_min_k || theta > theta_max_k ||
                    !std::isfinite(diagnosed) || diagnosed <= 0.0 ||
                    !std::isfinite(stored) || stored < pressure_min_pa ||
                    stored > pressure_max_pa ||
                    !std::isfinite(volume) || volume <= 0.0)
                {
                    reject("committed child rho/theta does not define pressure");
                }
                add_finite(target, diagnosed * volume);
                add_finite(committed, static_cast<double>(stored) * volume);
            }

    PressureDiagnosisReport candidate;
    candidate.target_pressure_volume_integral_pa_m3 = target.value();
    candidate.committed_pressure_volume_integral_pa_m3 = committed.value();
    candidate.storage_residual_pa_m3 = target.value() - committed.value();
    candidate.diagnosed_cells = static_cast<std::size_t>(
        relationship.child_cells_x) * relationship.child_cells_y *
        relationship.levels_z;
    if (!std::isfinite(candidate.storage_residual_pa_m3))
        reject("pressure storage residual is not finite");

    for (int ci = 0; ci < relationship.child_cells_x; ++ci)
        for (int cj = 0; cj < relationship.child_cells_y; ++cj)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const float rho = committed_child_dry_density[ci + 1][cj + 1][k];
                const float theta =
                    committed_child_potential_temperature[ci + 1][cj + 1][k];
                child_pressure[ci + 1][cj + 1][k] = static_cast<float>(
                    tmv::dynamics_eos::pressure_fp64(rho, theta));
            }
    report = candidate;
}

double nested_scalar_relaxation_fraction(
    const OneWayNestRelationshipV1& relationship,
    const NestedRelaxationConfigV1& config,
    int child_i,
    int child_j,
    double child_step_s)
{
    static_cast<void>(parent_temporal_weights(relationship, 0));
    if (!std::isfinite(config.e_folding_time_s) ||
        config.e_folding_time_s <= 0.0 ||
        !std::isfinite(child_step_s) || child_step_s < 0.0 ||
        child_i < 0 || child_i >= relationship.child_cells_x ||
        child_j < 0 || child_j >= relationship.child_cells_y)
    {
        reject("relaxation configuration, timestep, or child index is invalid");
    }
    const int distance = std::min(
        std::min(child_i, relationship.child_cells_x - 1 - child_i),
        std::min(child_j, relationship.child_cells_y - 1 - child_j));
    if (distance == 0)
        return 1.0;
    if (distance >= relationship.relaxation_width_child_cells)
        return 0.0;
    const double phase = 0.5 * std::acos(-1.0) *
        static_cast<double>(distance) /
        relationship.relaxation_width_child_cells;
    const double taper = std::pow(std::cos(phase), 2.0);
    const double fraction = -std::expm1(
        -child_step_s / config.e_folding_time_s * taper);
    if (!std::isfinite(fraction) || fraction < 0.0 || fraction > 1.0)
        reject("relaxation fraction is not finite and bounded");
    return fraction;
}

void relax_conserved_scalar_flat_storage(
    const OneWayNestRelationshipV1& relationship,
    const ParentTimeBracketV1& bracket,
    const NestedRelaxationConfigV1& config,
    double child_step_s,
    const std::vector<double>& layer_thickness_m,
    const Field3D& previous_parent_dry_density,
    const Field3D& current_parent_dry_density,
    const Field3D& previous_parent_specific,
    const Field3D& current_parent_specific,
    const Field3D& old_child_dry_density,
    const Field3D& old_child_specific,
    bool require_nonnegative_specific,
    Field3D& candidate_child_dry_density,
    Field3D& candidate_child_specific,
    ConservedScalarRelaxationReport& report)
{
    const auto weights = parent_temporal_weights(relationship, bracket);
    const Field3D* parent_fields[] = {&previous_parent_dry_density,
                                     &current_parent_dry_density,
                                     &previous_parent_specific,
                                     &current_parent_specific};
    for (const Field3D* field : parent_fields)
        require_native_storage_shape(*field, relationship.parent_cells_x,
                                     relationship.parent_cells_y,
                                     relationship.levels_z);
    const Field3D* child_fields[] = {&old_child_dry_density,
                                    &old_child_specific,
                                    &candidate_child_dry_density,
                                    &candidate_child_specific};
    for (const Field3D* field : child_fields)
        require_native_storage_shape(*field, relationship.child_cells_x,
                                     relationship.child_cells_y,
                                     relationship.levels_z);
    if (aliases(candidate_child_dry_density, candidate_child_specific))
        reject("relaxation destinations must not alias");
    for (const Field3D* source : parent_fields)
        if (aliases(candidate_child_dry_density, *source) ||
            aliases(candidate_child_specific, *source))
            reject("relaxation parent and destination storage must not alias");
    if (aliases(candidate_child_dry_density, old_child_dry_density) ||
        aliases(candidate_child_dry_density, old_child_specific) ||
        aliases(candidate_child_specific, old_child_dry_density) ||
        aliases(candidate_child_specific, old_child_specific))
        reject("relaxation old and candidate child storage must not alias");
    if (layer_thickness_m.size() !=
        static_cast<std::size_t>(relationship.levels_z))
        reject("vertical metric does not match the relationship");
    for (double dz : layer_thickness_m)
        if (!std::isfinite(dz) || dz <= 0.0)
            reject("vertical layer thickness must be finite and positive");
    const double child_area = relationship.child_dx_m *
                              relationship.child_dy_m;

    struct Candidate
    {
        double rho = 0.0;
        double specific = 0.0;
        double carrier = 0.0;
        double old_rho = 0.0;
        double old_carrier = 0.0;
        double fraction = 0.0;
    };
    const auto cell_candidate = [&](int ci, int cj, int k) {
        Candidate result;
        result.fraction = nested_scalar_relaxation_fraction(
            relationship, config, ci, cj, child_step_s);
        result.old_rho = old_child_dry_density[ci + 1][cj + 1][k];
        const double old_specific = old_child_specific[ci + 1][cj + 1][k];
        if (!std::isfinite(result.old_rho) || result.old_rho <= 0.0 ||
            !std::isfinite(old_specific) ||
            (require_nonnegative_specific && old_specific < 0.0))
            reject("old child scalar state is invalid");
        result.old_carrier = result.old_rho * old_specific;
        if (result.fraction == 0.0)
        {
            result.rho = result.old_rho;
            result.specific = old_specific;
            result.carrier = result.old_carrier;
            return result;
        }
        const int pi = relationship.child_origin_parent_cell_i +
                       ci / relationship.parent_ratio_x;
        const int pj = relationship.child_origin_parent_cell_j +
                       cj / relationship.parent_ratio_y;
        const auto target = blend(
            previous_parent_dry_density[pi + 1][pj + 1][k],
            current_parent_dry_density[pi + 1][pj + 1][k],
            previous_parent_specific[pi + 1][pj + 1][k],
            current_parent_specific[pi + 1][pj + 1][k], weights,
            require_nonnegative_specific);
        result.rho = (1.0 - result.fraction) * result.old_rho +
                     result.fraction * target.density;
        result.carrier = (1.0 - result.fraction) * result.old_carrier +
                         result.fraction * target.carrier;
        result.specific = result.carrier / result.rho;
        if (!std::isfinite(result.rho) || result.rho <= 0.0 ||
            !std::isfinite(result.carrier) ||
            !std::isfinite(result.specific) ||
            result.rho > std::numeric_limits<float>::max() ||
            std::abs(result.specific) > std::numeric_limits<float>::max() ||
            (require_nonnegative_specific && result.specific < 0.0))
            reject("relaxed scalar target is not representable");
        return result;
    };

    FixedOrderSum target_dry_change;
    FixedOrderSum committed_dry_change;
    FixedOrderSum target_carrier_change;
    FixedOrderSum committed_carrier_change;
    std::size_t forced_cells = 0;
    for (int ci = 0; ci < relationship.child_cells_x; ++ci)
        for (int cj = 0; cj < relationship.child_cells_y; ++cj)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const auto value = cell_candidate(ci, cj, k);
                const double volume = child_area *
                    layer_thickness_m[static_cast<std::size_t>(k)];
                const float stored_rho = static_cast<float>(value.rho);
                const float stored_specific = static_cast<float>(value.specific);
                add_finite(target_dry_change,
                           (value.rho - value.old_rho) * volume);
                add_finite(committed_dry_change,
                           (static_cast<double>(stored_rho) - value.old_rho) *
                               volume);
                add_finite(target_carrier_change,
                           (value.carrier - value.old_carrier) * volume);
                add_finite(committed_carrier_change,
                           (static_cast<double>(stored_rho) * stored_specific -
                            value.old_carrier) * volume);
                if (value.fraction > 0.0)
                    ++forced_cells;
            }
    ConservedScalarRelaxationReport candidate_report;
    candidate_report.target_dry_mass_change_kg = target_dry_change.value();
    candidate_report.committed_dry_mass_change_kg =
        committed_dry_change.value();
    candidate_report.dry_storage_residual_kg = target_dry_change.value() -
                                               committed_dry_change.value();
    candidate_report.target_carrier_change = target_carrier_change.value();
    candidate_report.committed_carrier_change =
        committed_carrier_change.value();
    candidate_report.carrier_storage_residual =
        target_carrier_change.value() - committed_carrier_change.value();
    candidate_report.forced_cells = forced_cells;
    if (!std::isfinite(candidate_report.dry_storage_residual_kg) ||
        !std::isfinite(candidate_report.carrier_storage_residual))
        reject("relaxation storage residual is not finite");

    for (int ci = 0; ci < relationship.child_cells_x; ++ci)
        for (int cj = 0; cj < relationship.child_cells_y; ++cj)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const auto value = cell_candidate(ci, cj, k);
                candidate_child_dry_density[ci + 1][cj + 1][k] =
                    static_cast<float>(value.rho);
                candidate_child_specific[ci + 1][cj + 1][k] =
                    static_cast<float>(value.specific);
            }
    report = candidate_report;
}

void relax_cgrid_momentum_flat_storage(
    const OneWayNestRelationshipV1& relationship,
    const NestedRelaxationConfigV1& config,
    double child_step_s,
    const std::vector<double>& layer_thickness_m,
    const Field3D& old_dry_density,
    const Field3D& old_u,
    const Field3D& old_v,
    const Field3D& old_w,
    const Field3D& target_dry_density,
    const Field3D& target_u,
    const Field3D& target_v,
    const Field3D& target_w,
    const Field3D& committed_new_dry_density,
    Field3D& candidate_u,
    Field3D& candidate_v,
    Field3D& candidate_w,
    ConservedMomentumRelaxationReport& report)
{
    const Field3D* fields[] = {
        &old_dry_density, &old_u, &old_v, &old_w,
        &target_dry_density, &target_u, &target_v, &target_w,
        &committed_new_dry_density, &candidate_u, &candidate_v, &candidate_w};
    for (const Field3D* field : fields)
        require_native_storage_shape(*field, relationship.child_cells_x,
                                     relationship.child_cells_y,
                                     relationship.levels_z);
    for (int a = 0; a < 3; ++a)
        for (int b = a + 1; b < 3; ++b)
            if (aliases(*fields[9 + a], *fields[9 + b]))
                reject("momentum relaxation destinations must not alias");
    for (int a = 9; a < 12; ++a)
        for (int b = 0; b < 9; ++b)
            if (aliases(*fields[a], *fields[b]))
                reject("momentum relaxation source and destination must not alias");
    if (layer_thickness_m.size() !=
        static_cast<std::size_t>(relationship.levels_z))
        reject("vertical metric does not match the relationship");
    for (double dz : layer_thickness_m)
        if (!std::isfinite(dz) || dz <= 0.0)
            reject("vertical layer thickness must be finite and positive");
    const double area = relationship.child_dx_m * relationship.child_dy_m;
    const auto mass = [&](const Field3D& rho, int ci, int cj, int k) {
        const double value = static_cast<double>(rho[ci + 1][cj + 1][k]) *
            area * layer_thickness_m[static_cast<std::size_t>(k)];
        if (!std::isfinite(value) || value <= 0.0)
            reject("momentum relaxation density does not define positive mass");
        return value;
    };
    struct FaceCandidate { double desired; double old; double new_mass; };
    const auto horizontal = [&](bool x_face, int face, int transverse, int k) {
        const int count = x_face ? relationship.child_cells_x
                                 : relationship.child_cells_y;
        const Field3D& old_velocity = x_face ? old_u : old_v;
        const Field3D& target_velocity = x_face ? target_u : target_v;
        const int vi = x_face ? face : transverse + 1;
        const int vj = x_face ? transverse + 1 : face;
        const double u_old = old_velocity[vi][vj][k];
        const double u_target = target_velocity[vi][vj][k];
        if (!std::isfinite(u_old) || !std::isfinite(u_target))
            reject("momentum relaxation velocity target is nonfinite");
        double desired = 0.0;
        double old_momentum = 0.0;
        double new_mass = 0.0;
        for (int side = 0; side < 2; ++side)
        {
            const int cell = face - 1 + side;
            if (cell < 0 || cell >= count)
                continue;
            const int ci = x_face ? cell : transverse;
            const int cj = x_face ? transverse : cell;
            const double fraction = nested_scalar_relaxation_fraction(
                relationship, config, ci, cj, child_step_s);
            const double old_half = 0.5 * mass(old_dry_density, ci, cj, k);
            const double target_half = 0.5 * mass(target_dry_density, ci, cj, k);
            new_mass += 0.5 * mass(committed_new_dry_density, ci, cj, k);
            old_momentum += old_half * u_old;
            desired += (1.0 - fraction) * old_half * u_old +
                       fraction * target_half * u_target;
        }
        const double velocity = desired / new_mass;
        if (!std::isfinite(velocity) ||
            std::abs(velocity) > std::numeric_limits<float>::max())
            reject("relaxed horizontal velocity is not representable");
        return FaceCandidate{desired, old_momentum, new_mass};
    };
    const auto vertical = [&](int ci, int cj, int k) {
        const double fraction = nested_scalar_relaxation_fraction(
            relationship, config, ci, cj, child_step_s);
        const double old_velocity = old_w[ci + 1][cj + 1][k];
        const double target_velocity = target_w[ci + 1][cj + 1][k];
        double old_mass = 0.5 * mass(old_dry_density, ci, cj, k);
        double target_mass = 0.5 * mass(target_dry_density, ci, cj, k);
        double new_mass = 0.5 * mass(committed_new_dry_density, ci, cj, k);
        if (k + 1 < relationship.levels_z)
        {
            old_mass += 0.5 * mass(old_dry_density, ci, cj, k + 1);
            target_mass += 0.5 * mass(target_dry_density, ci, cj, k + 1);
            new_mass += 0.5 * mass(committed_new_dry_density, ci, cj, k + 1);
        }
        if (!std::isfinite(old_velocity) || !std::isfinite(target_velocity))
            reject("vertical momentum relaxation velocity is nonfinite");
        const double old_momentum = old_mass * old_velocity;
        const double desired = (1.0 - fraction) * old_momentum +
                               fraction * target_mass * target_velocity;
        const double velocity = desired / new_mass;
        if (!std::isfinite(velocity) ||
            std::abs(velocity) > std::numeric_limits<float>::max())
            reject("relaxed vertical velocity is not representable");
        return FaceCandidate{desired, old_momentum, new_mass};
    };
    const auto horizontal_owned_mass = [&](const Field3D& rho, bool x_face,
                                           int face, int transverse, int k) {
        const int count = x_face ? relationship.child_cells_x
                                 : relationship.child_cells_y;
        const int ri = x_face ? face : transverse + 1;
        const int rj = x_face ? transverse + 1 : face;
        const int ni = x_face ? face + 1 : transverse + 1;
        const int nj = x_face ? transverse + 1 : face + 1;
        const double volume = area *
            layer_thickness_m[static_cast<std::size_t>(k)];
        const double first = rho[ri][rj][k];
        const double second = rho[ni][nj][k];
        double value = 0.0;
        if (std::isfinite(first) && first > 0.0)
            value += 0.5 * volume * first;
        if (std::isfinite(second) && second > 0.0)
            value += 0.5 * volume * second;
        if (face < 0 || face > count || !std::isfinite(value) || value <= 0.0)
            reject("momentum relaxation owned-face density is invalid");
        return value;
    };
    FixedOrderSum tx, cx, ty, cy, tz, cz;
    for (int f = 0; f <= relationship.child_cells_x; ++f)
        for (int j = 0; j < relationship.child_cells_y; ++j)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const auto c = horizontal(true, f, j, k);
                const double old_momentum = horizontal_owned_mass(
                    old_dry_density, true, f, j, k) * old_u[f][j + 1][k];
                const double new_mass = horizontal_owned_mass(
                    committed_new_dry_density, true, f, j, k);
                const double desired_velocity = c.desired / c.new_mass;
                add_finite(tx, new_mass * desired_velocity - old_momentum);
                add_finite(cx, new_mass *
                    static_cast<float>(desired_velocity) - old_momentum);
            }
    for (int i = 0; i < relationship.child_cells_x; ++i)
        for (int f = 0; f <= relationship.child_cells_y; ++f)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const auto c = horizontal(false, f, i, k);
                const double old_momentum = horizontal_owned_mass(
                    old_dry_density, false, f, i, k) * old_v[i + 1][f][k];
                const double new_mass = horizontal_owned_mass(
                    committed_new_dry_density, false, f, i, k);
                const double desired_velocity = c.desired / c.new_mass;
                add_finite(ty, new_mass * desired_velocity - old_momentum);
                add_finite(cy, new_mass *
                    static_cast<float>(desired_velocity) - old_momentum);
            }
    for (int i = 0; i < relationship.child_cells_x; ++i)
        for (int j = 0; j < relationship.child_cells_y; ++j)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const auto c = vertical(i, j, k);
                add_finite(tz, c.desired - c.old);
                add_finite(cz, c.new_mass *
                    static_cast<float>(c.desired / c.new_mass) - c.old);
            }
    ConservedMomentumRelaxationReport candidate;
    candidate.target_x_change_kgms = tx.value();
    candidate.committed_x_change_kgms = cx.value();
    candidate.x_storage_residual_kgms = tx.value() - cx.value();
    candidate.target_y_change_kgms = ty.value();
    candidate.committed_y_change_kgms = cy.value();
    candidate.y_storage_residual_kgms = ty.value() - cy.value();
    candidate.target_z_change_kgms = tz.value();
    candidate.committed_z_change_kgms = cz.value();
    candidate.z_storage_residual_kgms = tz.value() - cz.value();
    const double residuals[] = {candidate.x_storage_residual_kgms,
        candidate.y_storage_residual_kgms, candidate.z_storage_residual_kgms};
    for (double value : residuals)
        if (!std::isfinite(value))
            reject("momentum relaxation residual is nonfinite");
    for (int f = 0; f <= relationship.child_cells_x; ++f)
        for (int j = 0; j < relationship.child_cells_y; ++j)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const auto c = horizontal(true, f, j, k);
                candidate_u[f][j + 1][k] =
                    static_cast<float>(c.desired / c.new_mass);
            }
    for (int i = 0; i < relationship.child_cells_x; ++i)
        for (int f = 0; f <= relationship.child_cells_y; ++f)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const auto c = horizontal(false, f, i, k);
                candidate_v[i + 1][f][k] =
                    static_cast<float>(c.desired / c.new_mass);
            }
    for (int i = 0; i < relationship.child_cells_x; ++i)
        for (int j = 0; j < relationship.child_cells_y; ++j)
            for (int k = 0; k < relationship.levels_z; ++k)
            {
                const auto c = vertical(i, j, k);
                candidate_w[i + 1][j + 1][k] =
                    static_cast<float>(c.desired / c.new_mass);
            }
    report = candidate;
}


} // namespace tmv::domain
