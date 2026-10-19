#include "catch2/catch.hpp"

#include "core/infra/nested_terrain_geometry.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{

tmv::domain::OneWayNestRelationshipV1 relationship(int ratio = 2)
{
    tmv::domain::OneWayNestRelationshipInput input;
    input.parent_cells_x                = 12;
    input.parent_cells_y                = 10;
    input.parent_levels_z               = 4;
    input.parent_dx_m                   = 400.0;
    input.parent_dy_m                   = 300.0;
    input.parent_dt_s                   = 2.0;
    input.parent_ratio_x                = ratio;
    input.parent_ratio_y                = ratio;
    input.parent_time_step_ratio        = ratio;
    input.child_origin_parent_cell_i    = 4;
    input.child_origin_parent_cell_j    = 3;
    input.child_cells_x                 = 3 * ratio;
    input.child_cells_y                 = 3 * ratio;
    input.child_levels_z                = 4;
    input.relaxation_width_child_cells  = 1;
    input.boundary_stencil_parent_cells = 1;
    return tmv::domain::prepare_one_way_nest_relationship(input);
}

tmv::TerrainConfig bell_config(const char* placement = "grid_attached")
{
    tmv::TerrainConfig config;
    config.scheme_id               = "bell";
    config.coord_id                = "btf";
    config.horizontal_placement_id = placement;
    config.ztop                    = 6000.0;
    config.compute_derivatives     = true;
    config.compute_metrics         = true;
    config.bell.h0                 = 450.0;
    config.bell.a                  = 1200.0;
    config.bell.axisymmetric       = true;
    return config;
}

tmv::TerrainConfig schar_config()
{
    auto config      = bell_config();
    config.scheme_id = "schar";
    config.schar.H   = 350.0;
    config.schar.a   = 1400.0;
    config.schar.ell = 900.0;
    return config;
}

tmv::domain::NestedMappedConservativeStateV1 parent_state(const tmv::domain::OneWayNestRelationshipV1& r)
{
    tmv::domain::NestedMappedConservativeStateV1 state;
    state.scalar.resize(r.parent_cells_x, r.parent_cells_y, r.levels_z, 0.0);
    state.x_flux.resize(r.parent_cells_x + 1, r.parent_cells_y, r.levels_z, 0.0);
    state.y_flux.resize(r.parent_cells_x, r.parent_cells_y + 1, r.levels_z, 0.0);
    state.zeta_flux.resize(r.parent_cells_x, r.parent_cells_y, r.levels_z + 1, 0.0);
    for (int i = 0; i < r.parent_cells_x; ++i)
        for (int j = 0; j < r.parent_cells_y; ++j)
        {
            for (int k = 0; k < r.levels_z; ++k)
            {
                state.scalar(i, j, k) = 1.0 + 0.03 * i + 0.05 * j + 0.07 * k;
                state.y_flux(i, j, k) = -2.0 + 0.11 * i - 0.04 * j + 0.02 * k;
            }
            state.y_flux(i, r.parent_cells_y, 0) = 0.5 + 0.01 * i;
            for (int k = 0; k <= r.levels_z; ++k)
                state.zeta_flux(i, j, k) = 0.2 + 0.01 * i - 0.02 * j + 0.03 * k;
        }
    for (int i = 0; i <= r.parent_cells_x; ++i)
        for (int j = 0; j < r.parent_cells_y; ++j)
            for (int k = 0; k < r.levels_z; ++k)
                state.x_flux(i, j, k) = 3.0 + 0.08 * i + 0.06 * j - 0.01 * k;
    for (int i = 0; i < r.parent_cells_x; ++i)
        for (int k = 0; k < r.levels_z; ++k)
            state.y_flux(i, r.parent_cells_y, k) = 0.5 + 0.01 * i + 0.02 * k;
    return state;
}

void check_same_field(const tmv::TerrainField3D& left, const tmv::TerrainField3D& right)
{
    REQUIRE(left.nr == right.nr);
    REQUIRE(left.nth == right.nth);
    REQUIRE(left.nz == right.nz);
    REQUIRE(left.values == right.values);
}

void check_same_candidate(const tmv::domain::NestedActiveTerrainStateV1& left,
                          const tmv::domain::NestedActiveTerrainStateV1& right)
{
    REQUIRE(left.ready == right.ready);
    REQUIRE(left.generation == right.generation);
    REQUIRE(left.publication_count == right.publication_count);
    REQUIRE(left.parent_time_s == right.parent_time_s);
    REQUIRE(left.child_time_s == right.child_time_s);
    check_same_field(left.geometry.parent.metrics.J, right.geometry.parent.metrics.J);
    check_same_field(left.geometry.child_storage.metrics.J, right.geometry.child_storage.metrics.J);
    check_same_field(left.child_state.scalar, right.child_state.scalar);
    check_same_field(left.child_state.x_flux, right.child_state.x_flux);
    check_same_field(left.child_state.y_flux, right.child_state.y_flux);
    check_same_field(left.child_state.zeta_flux, right.child_state.zeta_flux);
    check_same_field(left.boundary.relaxation_weight, right.boundary.relaxation_weight);
    check_same_field(left.boundary.ground_w_ms, right.boundary.ground_w_ms);
    REQUIRE(left.ledger.parent_scalar_inventory == right.ledger.parent_scalar_inventory);
    REQUIRE(left.ledger.child_scalar_inventory == right.ledger.child_scalar_inventory);
    REQUIRE(left.ledger.scalar_inventory_residual == right.ledger.scalar_inventory_residual);
    REQUIRE(left.ledger.normalized_scalar_inventory_residual == right.ledger.normalized_scalar_inventory_residual);
    REQUIRE(left.ledger.maximum_x_flux_residual == right.ledger.maximum_x_flux_residual);
    REQUIRE(left.ledger.maximum_normalized_x_flux_residual == right.ledger.maximum_normalized_x_flux_residual);
    REQUIRE(left.ledger.maximum_y_flux_residual == right.ledger.maximum_y_flux_residual);
    REQUIRE(left.ledger.maximum_normalized_y_flux_residual == right.ledger.maximum_normalized_y_flux_residual);
    REQUIRE(left.ledger.maximum_zeta_flux_residual == right.ledger.maximum_zeta_flux_residual);
    REQUIRE(left.ledger.maximum_normalized_zeta_flux_residual == right.ledger.maximum_normalized_zeta_flux_residual);
    REQUIRE(left.ledger.maximum_ground_tangency_residual_ms == right.ledger.maximum_ground_tangency_residual_ms);
    REQUIRE(left.ledger.relaxation_owned_values == right.ledger.relaxation_owned_values);
}

bool reject_candidate(const tmv::domain::NestedActiveTerrainStateV1& candidate, void* context)
{
    auto* observed_generation = static_cast<std::uint64_t*>(context);
    *observed_generation      = candidate.generation;
    return false;
}

} // namespace

TEST_CASE("stationary nested Bell and Schar geometry share one Earth-frame parent-child identity",
          "[core][nested_grid][active_terrain][geometry]")
{
    const auto                         r      = relationship();
    auto                               config = bell_config();
    const tmv::domain::HorizontalPoint origin{1000.0, -2000.0};
    const auto geometry = tmv::domain::build_stationary_nested_terrain_geometry(config, r, origin, 7);

    REQUIRE(geometry.generation == 7);
    REQUIRE(geometry.parent.metrics.dx == r.parent_dx_m);
    REQUIRE(geometry.parent.metrics.dy == r.parent_dy_m);
    REQUIRE(geometry.child_storage.metrics.dx == r.child_dx_m);
    REQUIRE(geometry.child_storage.metrics.dy == r.child_dy_m);
    REQUIRE(geometry.child_storage.metrics.J.size_r() == r.child_cells_x + 2);
    REQUIRE(geometry.child_storage.metrics.J.size_th() == r.child_cells_y + 2);
    REQUIRE(geometry.child_origin_earth_m.x_m == origin.x_m + r.child_origin_parent_cell_i * r.parent_dx_m);
    REQUIRE(geometry.child_origin_earth_m.y_m == origin.y_m + r.child_origin_parent_cell_j * r.parent_dy_m);
    REQUIRE(geometry.feature_center_earth_m.x_m == origin.x_m + 0.5 * r.parent_cells_x * r.parent_dx_m);
    REQUIRE(geometry.feature_center_earth_m.y_m == origin.y_m + 0.5 * r.parent_cells_y * r.parent_dy_m);
    const double child_center_x =
        geometry.child_origin_earth_m.x_m + 0.5 * r.child_dx_m - geometry.feature_center_earth_m.x_m;
    const double child_center_y =
        geometry.child_origin_earth_m.y_m + 0.5 * r.child_dy_m - geometry.feature_center_earth_m.y_m;
    const double a2          = config.bell.a * config.bell.a;
    const double denominator = a2 + child_center_x * child_center_x + child_center_y * child_center_y;
    const double reference_height =
        config.bell.h0 * config.bell.a * config.bell.a * config.bell.a / std::pow(denominator, 1.5);
    const double reference_factor =
        -3.0 * config.bell.h0 * config.bell.a * config.bell.a * config.bell.a / std::pow(denominator, 2.5);
    REQUIRE(geometry.child_storage.topography.h[1][1] == Approx(reference_height).epsilon(2.0e-15));
    REQUIRE(geometry.child_storage.topography.hx[1][1] == Approx(reference_factor * child_center_x).epsilon(2.0e-15));
    REQUIRE(geometry.child_storage.topography.hy[1][1] == Approx(reference_factor * child_center_y).epsilon(2.0e-15));

    auto earth_fixed                     = config;
    earth_fixed.horizontal_placement_id  = "earth_fixed";
    earth_fixed.feature_center_earth_x_m = 8123.0;
    earth_fixed.feature_center_earth_y_m = -4567.0;
    const auto fixed = tmv::domain::build_stationary_nested_terrain_geometry(earth_fixed, r, origin, 8);
    CHECK(fixed.feature_center_earth_m.x_m == 8123.0);
    CHECK(fixed.feature_center_earth_m.y_m == -4567.0);

    auto ignored_center                     = config;
    ignored_center.feature_center_earth_x_m = 1.0;
    REQUIRE_THROWS(tmv::domain::build_stationary_nested_terrain_geometry(ignored_center, r, origin, 1));
    const auto schar          = schar_config();
    const auto schar_geometry = tmv::domain::build_stationary_nested_terrain_geometry(schar, r, origin, 9);
    REQUIRE(schar_geometry.config.scheme_id == "schar");
    REQUIRE(schar_geometry.generation == 9);
    REQUIRE(schar_geometry.parent.topography.h != geometry.parent.topography.h);
    REQUIRE(schar_geometry.child_storage.metrics.J.values != geometry.child_storage.metrics.J.values);
    auto unsupported      = config;
    unsupported.scheme_id = "none";
    REQUIRE_THROWS(tmv::domain::build_stationary_nested_terrain_geometry(unsupported, r, origin, 1));
    const auto ratio3 = tmv::domain::build_stationary_nested_terrain_geometry(config, relationship(3), origin, 10);
    REQUIRE(ratio3.relationship.parent_ratio_x == 3);
    REQUIRE(ratio3.relationship.parent_ratio_y == 3);
    REQUIRE(ratio3.child_storage.metrics.dx == r.parent_dx_m / 3.0);
    REQUIRE(ratio3.child_storage.metrics.J.values != geometry.child_storage.metrics.J.values);

    auto anisotropic           = relationship(3);
    anisotropic.parent_ratio_y = 2;
    REQUIRE_THROWS(tmv::domain::build_stationary_nested_terrain_geometry(config, anisotropic, origin, 1));
    auto insufficient_time                   = relationship(3);
    insufficient_time.parent_time_step_ratio = 2;
    REQUIRE_THROWS(tmv::domain::build_stationary_nested_terrain_geometry(config, insufficient_time, origin, 1));
}

TEST_CASE("mapped Bell and Schar 3:1 transfers are conservative, distinct, deterministic, and atomic",
          "[core][nested_grid][active_terrain][ratio3][conservation][threading][atomicity]")
{
    const auto r3 = relationship(3);
    const auto r2 = relationship(2);
    for (const auto& config : {bell_config(), schar_config()})
    {
        CAPTURE(config.scheme_id);
        const auto parent3   = parent_state(r3);
        const auto geometry3 = tmv::domain::build_stationary_nested_terrain_geometry(config, r3, {1000.0, -2000.0}, 3);
        const auto geometry2 = tmv::domain::build_stationary_nested_terrain_geometry(config, r2, {1000.0, -2000.0}, 2);
        REQUIRE(geometry3.child_storage.metrics.J.values != geometry2.child_storage.metrics.J.values);

        tmv::domain::NestedTerrainTransferLedgerV1 ledger;
        const auto child = tmv::domain::prolong_nested_mapped_conservative_state(geometry3, parent3, &ledger);
        REQUIRE(ledger.normalized_scalar_inventory_residual <= 1.0e-15);
        REQUIRE(ledger.maximum_normalized_x_flux_residual <= 1.0e-15);
        REQUIRE(ledger.maximum_normalized_y_flux_residual <= 1.0e-15);
        REQUIRE(ledger.maximum_normalized_zeta_flux_residual <= 1.0e-15);
        const auto restricted = tmv::domain::restrict_nested_mapped_conservative_state(geometry3, child);
        for (int pi = 0; pi < r3.child_parent_span_x; ++pi)
            for (int pj = 0; pj < r3.child_parent_span_y; ++pj)
                for (int k = 0; k < r3.levels_z; ++k)
                    REQUIRE(restricted.scalar(pi, pj, k) == Approx(parent3.scalar(r3.child_origin_parent_cell_i + pi,
                                                                                  r3.child_origin_parent_cell_j + pj,
                                                                                  k))
                                                                .epsilon(3.0e-14));

        tmv::domain::NestedActiveTerrainStateV1 live;
#ifdef _OPENMP
        omp_set_dynamic(0);
        omp_set_num_threads(1);
#endif
        auto one = tmv::domain::prepare_nested_active_terrain_commit(
            live, config, r3, {1000.0, -2000.0}, parent3, 14.0, 9.0, -2.0);
#ifdef _OPENMP
        omp_set_num_threads(4);
#endif
        auto four = tmv::domain::prepare_nested_active_terrain_commit(
            live, config, r3, {1000.0, -2000.0}, parent3, 14.0, 9.0, -2.0);
        check_same_candidate(one.candidate, four.candidate);
        REQUIRE(tmv::domain::commit_nested_active_terrain(one, live));
        const auto before              = live;
        auto       invalid_parent      = parent3;
        invalid_parent.scalar(4, 3, 1) = std::numeric_limits<double>::quiet_NaN();
        REQUIRE_THROWS(tmv::domain::prepare_nested_active_terrain_commit(
            live, config, r3, {1000.0, -2000.0}, invalid_parent, 16.0, 9.0, -2.0));
        check_same_candidate(live, before);
    }
}

TEST_CASE("stationary nested Schar candidate is deterministic and commit atomic",
          "[core][nested_grid][active_terrain][schar][atomicity][threading]")
{
    const auto                              r      = relationship();
    const auto                              parent = parent_state(r);
    tmv::domain::NestedActiveTerrainStateV1 one_live;
    tmv::domain::NestedActiveTerrainStateV1 four_live;
#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(1);
#endif
    auto one = tmv::domain::prepare_nested_active_terrain_commit(
        one_live, schar_config(), r, {1000.0, -2000.0}, parent, 14.0, 9.0, -2.0);
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    auto four = tmv::domain::prepare_nested_active_terrain_commit(
        four_live, schar_config(), r, {1000.0, -2000.0}, parent, 14.0, 9.0, -2.0);
    check_same_candidate(one.candidate, four.candidate);
    REQUIRE(one.candidate.geometry.config.scheme_id == "schar");
    REQUIRE(tmv::domain::commit_nested_active_terrain(one, one_live));
    REQUIRE(tmv::domain::commit_nested_active_terrain(four, four_live));
    check_same_candidate(one_live, four_live);
}

TEST_CASE("mapped terrain prolongation closes scalar and staggered flux ledgers",
          "[core][nested_grid][active_terrain][conservation]")
{
    const auto r        = relationship();
    const auto geometry = tmv::domain::build_stationary_nested_terrain_geometry(bell_config(), r, {1000.0, -2000.0}, 1);
    const auto parent   = parent_state(r);
    tmv::domain::NestedTerrainTransferLedgerV1 ledger;
    const auto child = tmv::domain::prolong_nested_mapped_conservative_state(geometry, parent, &ledger);

    REQUIRE(ledger.normalized_scalar_inventory_residual <= 1.0e-15);
    REQUIRE(ledger.maximum_normalized_x_flux_residual <= 1.0e-15);
    REQUIRE(ledger.maximum_normalized_y_flux_residual <= 1.0e-15);
    REQUIRE(ledger.maximum_normalized_zeta_flux_residual <= 1.0e-15);

    const auto restricted = tmv::domain::restrict_nested_mapped_scalar(geometry, child.scalar);
    for (int pi = 0; pi < r.child_parent_span_x; ++pi)
        for (int pj = 0; pj < r.child_parent_span_y; ++pj)
            for (int k = 0; k < r.levels_z; ++k)
                REQUIRE(restricted(pi, pj, k) ==
                        Approx(parent.scalar(r.child_origin_parent_cell_i + pi, r.child_origin_parent_cell_j + pj, k))
                            .epsilon(2.0e-14));

    const auto restricted_state = tmv::domain::restrict_nested_mapped_conservative_state(geometry, child);
    check_same_field(restricted, restricted_state.scalar);
    for (int pi = 0; pi <= r.child_parent_span_x; ++pi)
        for (int pj = 0; pj < r.child_parent_span_y; ++pj)
            for (int k = 0; k < r.levels_z; ++k)
                REQUIRE(restricted_state.x_flux(pi, pj, k) ==
                        Approx(parent.x_flux(r.child_origin_parent_cell_i + pi, r.child_origin_parent_cell_j + pj, k))
                            .epsilon(2.0e-14));
    for (int pi = 0; pi < r.child_parent_span_x; ++pi)
        for (int pj = 0; pj <= r.child_parent_span_y; ++pj)
            for (int k = 0; k < r.levels_z; ++k)
                REQUIRE(restricted_state.y_flux(pi, pj, k) ==
                        Approx(parent.y_flux(r.child_origin_parent_cell_i + pi, r.child_origin_parent_cell_j + pj, k))
                            .epsilon(2.0e-14));
    for (int pi = 0; pi < r.child_parent_span_x; ++pi)
        for (int pj = 0; pj < r.child_parent_span_y; ++pj)
            for (int k = 0; k <= r.levels_z; ++k)
                REQUIRE(
                    restricted_state.zeta_flux(pi, pj, k) ==
                    Approx(parent.zeta_flux(r.child_origin_parent_cell_i + pi, r.child_origin_parent_cell_j + pj, k))
                        .epsilon(2.0e-14));

    const auto boundary = tmv::domain::build_nested_terrain_tangent_boundary(geometry, 12.0, -3.5, &ledger);
    REQUIRE(ledger.maximum_ground_tangency_residual_ms < 1.0e-15);
    REQUIRE(ledger.relaxation_owned_values ==
            static_cast<std::uint64_t>((2 * r.child_cells_x + 2 * (r.child_cells_y - 2)) * r.levels_z));
    REQUIRE(boundary.relaxation_weight(0, 2, 3) == 1.0);
    REQUIRE(boundary.relaxation_weight(1, 1, 3) == 0.0);
    for (int i = 0; i < r.child_cells_x; ++i)
        for (int j = 0; j < r.child_cells_y; ++j)
            REQUIRE(boundary.ground_w_ms(i, j, 0) + geometry.child_storage.metrics.Gx_face(i + 1, j + 1, 0) * 12.0 +
                        geometry.child_storage.metrics.Gy_face(i + 1, j + 1, 0) * -3.5 ==
                    Approx(0.0).margin(1.0e-15));

    auto invalid            = parent;
    invalid.scalar(4, 3, 1) = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS(tmv::domain::prolong_nested_mapped_conservative_state(geometry, invalid));
    auto invalid_child            = child;
    invalid_child.y_flux(1, 1, 1) = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS(tmv::domain::restrict_nested_mapped_conservative_state(geometry, invalid_child));
    invalid_child = child;
    invalid_child.scalar.resize(1, 1, 1, 0.0);
    REQUIRE_THROWS(tmv::domain::restrict_nested_mapped_conservative_state(geometry, invalid_child));
    REQUIRE_THROWS(
        tmv::domain::build_nested_terrain_tangent_boundary(geometry, std::numeric_limits<double>::infinity(), 0.0));
}

TEST_CASE("active-terrain candidate publication is deterministic and failure atomic",
          "[core][nested_grid][active_terrain][atomicity][threading]")
{
    const auto                              r            = relationship();
    const auto                              fixed_config = bell_config();
    const auto                              parent       = parent_state(r);
    tmv::domain::NestedActiveTerrainStateV1 live;

#ifdef _OPENMP
    omp_set_dynamic(0);
    omp_set_num_threads(1);
#endif
    auto one = tmv::domain::prepare_nested_active_terrain_commit(
        live, fixed_config, r, {1000.0, -2000.0}, parent, 14.0, 9.0, -2.0);
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    auto four = tmv::domain::prepare_nested_active_terrain_commit(
        live, fixed_config, r, {1000.0, -2000.0}, parent, 14.0, 9.0, -2.0);
    check_same_candidate(one.candidate, four.candidate);
#ifdef _OPENMP
    omp_set_num_threads(1);
#endif
    const auto restricted_one =
        tmv::domain::restrict_nested_mapped_conservative_state(one.candidate.geometry, one.candidate.child_state);
#ifdef _OPENMP
    omp_set_num_threads(4);
#endif
    const auto restricted_four =
        tmv::domain::restrict_nested_mapped_conservative_state(four.candidate.geometry, four.candidate.child_state);
    check_same_field(restricted_one.scalar, restricted_four.scalar);
    check_same_field(restricted_one.x_flux, restricted_four.x_flux);
    check_same_field(restricted_one.y_flux, restricted_four.y_flux);
    check_same_field(restricted_one.zeta_flux, restricted_four.zeta_flux);
    REQUIRE_FALSE(live.ready);
    REQUIRE(live.generation == 0);
    REQUIRE(tmv::domain::commit_nested_active_terrain(one, live));
    REQUIRE(live.ready);
    REQUIRE(live.generation == 1);
    REQUIRE(live.publication_count == 1);
    REQUIRE(live.child_time_s == 14.0);
    REQUIRE_FALSE(tmv::domain::commit_nested_active_terrain(one, live));
    REQUIRE_FALSE(tmv::domain::commit_nested_active_terrain(four, live));

    const auto    before_veto         = live;
    std::uint64_t observed_generation = 0;
    REQUIRE_THROWS_AS(
        tmv::domain::prepare_nested_active_terrain_commit(
            live, fixed_config, r, {1000.0, -2000.0}, parent, 16.0, 9.0, -2.0, reject_candidate, &observed_generation),
        std::runtime_error);
    REQUIRE(observed_generation == 2);
    check_same_candidate(live, before_veto);

    auto stale = tmv::domain::prepare_nested_active_terrain_commit(
        live, fixed_config, r, {1000.0, -2000.0}, parent, 16.0, 9.0, -2.0);
    ++live.publication_count;
    REQUIRE_FALSE(tmv::domain::commit_nested_active_terrain(stale, live));
    --live.publication_count;

    auto stale_clock = tmv::domain::prepare_nested_active_terrain_commit(
        live, fixed_config, r, {1000.0, -2000.0}, parent, 16.0, 9.0, -2.0);
    live.child_time_s += 0.25;
    REQUIRE_FALSE(tmv::domain::commit_nested_active_terrain(stale_clock, live));
    live.child_time_s -= 0.25;

    auto stale_field = tmv::domain::prepare_nested_active_terrain_commit(
        live, fixed_config, r, {1000.0, -2000.0}, parent, 16.0, 9.0, -2.0);
    live.child_state.scalar(0, 0, 0) += 1.0;
    REQUIRE_FALSE(tmv::domain::commit_nested_active_terrain(stale_field, live));
    live.child_state.scalar(0, 0, 0) -= 1.0;

    auto invalid_parent            = parent;
    invalid_parent.x_flux(4, 3, 0) = std::numeric_limits<double>::quiet_NaN();
    const auto before_invalid      = live;
    REQUIRE_THROWS(tmv::domain::prepare_nested_active_terrain_commit(
        live, fixed_config, r, {1000.0, -2000.0}, invalid_parent, 16.0, 9.0, -2.0));
    check_same_candidate(live, before_invalid);

    auto overflowing_parent = parent;
    overflowing_parent.scalar.values.assign(overflowing_parent.scalar.size(), std::numeric_limits<double>::max());
    REQUIRE_THROWS(tmv::domain::prepare_nested_active_terrain_commit(
        live, fixed_config, r, {1000.0, -2000.0}, overflowing_parent, 16.0, 9.0, -2.0));
    check_same_candidate(live, before_invalid);

    auto exhausted              = live;
    exhausted.generation        = std::numeric_limits<std::uint64_t>::max();
    const auto exhausted_before = exhausted;
    REQUIRE_THROWS(tmv::domain::prepare_nested_active_terrain_commit(
        exhausted, fixed_config, r, {1000.0, -2000.0}, parent, 16.0, 9.0, -2.0));
    check_same_candidate(exhausted, exhausted_before);
    exhausted                               = live;
    exhausted.publication_count             = std::numeric_limits<std::uint64_t>::max();
    const auto publication_exhausted_before = exhausted;
    REQUIRE_THROWS(tmv::domain::prepare_nested_active_terrain_commit(
        exhausted, fixed_config, r, {1000.0, -2000.0}, parent, 16.0, 9.0, -2.0));
    check_same_candidate(exhausted, publication_exhausted_before);
}

TEST_CASE("Parent-lattice terrain preserves moving overlap exactly at both refinement ratios",
          "[core][nested_grid][active_terrain][movement][lattice]")
{
    for (int ratio : {2, 3})
        for (const auto& config : {bell_config(), schar_config()})
            for (const auto shift : {std::pair<int, int>{1, -1}, {-1, 1}, {1, 1}})
            {
                auto before_relationship = relationship(ratio);
                auto after_relationship  = before_relationship;
                after_relationship.child_origin_parent_cell_i += shift.first;
                after_relationship.child_origin_parent_cell_j += shift.second;
                const tmv::domain::HorizontalPoint anchor{123456.125, -654321.625};
                const auto                         before =
                    tmv::domain::build_stationary_nested_terrain_geometry(config, before_relationship, anchor, 1, 2);
                const auto after =
                    tmv::domain::build_stationary_nested_terrain_geometry(config, after_relationship, anchor, 2, 2);
                const auto& a = before.child_storage.metrics;
                const auto& b = after.child_storage.metrics;
                for (int i = 1; i <= after_relationship.child_cells_x; ++i)
                    for (int j = 1; j <= after_relationship.child_cells_y; ++j)
                    {
                        const int si = i + ratio * shift.first, sj = j + ratio * shift.second;
                        if (si < 1 || si > after_relationship.child_cells_x || sj < 1 ||
                            sj > after_relationship.child_cells_y)
                            continue;
                        for (int k = 0; k < after_relationship.levels_z; ++k)
                        {
                            CHECK(a.J(si, sj, k) == b.J(i, j, k));
                            CHECK(a.z(si, sj, k) == b.z(i, j, k));
                            CHECK(a.z_face(si, sj, k) == b.z_face(i, j, k));
                            CHECK(a.z_face(si, sj, k + 1) == b.z_face(i, j, k + 1));
                        }
                    }
                REQUIRE_THROWS(
                    tmv::domain::build_stationary_nested_terrain_geometry(config, before_relationship, anchor, 1, 3));
            }
}

TEST_CASE("Earth-fixed parent rebasing preserves evolved child geometry and state atomically",
          "[core][nested_grid][active_terrain][recycling]")
{
    for (int ratio : {2, 3})
        for (auto config : {bell_config(), schar_config()})
        {
            auto r = relationship(ratio);
            config.horizontal_placement_id = "earth_fixed";
            config.feature_center_earth_x_m = 8123;
            config.feature_center_earth_y_m = -4567;
            tmv::domain::HorizontalPoint origin{1000, -2000};
            auto initial = tmv::domain::prepare_nested_active_terrain_commit(
                {}, config, r, origin, parent_state(r), 0, 5, -2);
            auto live = initial.candidate;
            live.child_state.scalar.values[0] += .25;
            const auto original_child = live.child_state;
            const auto original_geometry = live.geometry.child_storage;
            for (int shift : {1, -1, 2, -2})
            {
                auto next = r;
                next.child_origin_parent_cell_i -= shift;
                const tmv::domain::HorizontalPoint next_origin{origin.x_m + shift * r.parent_dx_m, origin.y_m};
                auto prepared = tmv::domain::prepare_nested_terrain_parent_rebase(live, next, next_origin);
                REQUIRE(tmv::domain::can_commit_nested_active_terrain(prepared, live));
                const auto before = live;
                live.geometry.parent.metrics.J.values[0] += .001;
                CHECK_FALSE(tmv::domain::can_commit_nested_active_terrain(prepared, live));
                CHECK_FALSE(tmv::domain::commit_nested_active_terrain(prepared, live));
                live = before;
                REQUIRE(tmv::domain::commit_nested_active_terrain(prepared, live));
                CHECK_FALSE(tmv::domain::commit_nested_active_terrain(prepared, live));
                CHECK(live.geometry.feature_center_earth_m.x_m == config.feature_center_earth_x_m);
                CHECK(live.geometry.feature_center_earth_m.y_m == config.feature_center_earth_y_m);
                check_same_field(live.child_state.scalar, original_child.scalar);
                check_same_field(live.child_state.x_flux, original_child.x_flux);
                check_same_field(live.child_state.y_flux, original_child.y_flux);
                check_same_field(live.child_state.zeta_flux, original_child.zeta_flux);
                check_same_field(live.geometry.child_storage.metrics.J, original_geometry.metrics.J);
                check_same_field(live.geometry.child_storage.metrics.z_face, original_geometry.metrics.z_face);
                CHECK(live.publication_count == live.generation);
                REQUIRE_THROWS(tmv::domain::prepare_nested_terrain_parent_rebase(live, next,
                    {next_origin.x_m + 1, next_origin.y_m}));
                r = next;
                origin = next_origin;
            }
        }
}
