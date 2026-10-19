#include "catch2/catch.hpp"
#include "compute/backend_runtime.hpp"
#include <cstdlib>
#include <memory>
#include "core/infra/nested_scalar_feedback.hpp"
#include "core/infra/nested_feedback.hpp"
#include <cmath>
#include <cstring>
#include <limits>

namespace
{
ComputeBackendRuntimeState* test_backend()
{
    const char* selected = std::getenv("NIMBUS_SCALAR_FEEDBACK_BACKEND");
    if (!selected)
        return nullptr;
    static std::unique_ptr<ComputeBackendRuntimeState> runtime;
    if (!runtime)
    {
        runtime                        = std::make_unique<ComputeBackendRuntimeState>();
        runtime->config.backend        = selected;
        runtime->config.allow_fallback = false;
        std::string error;
        const bool  initialized = initialize_compute_backend_runtime(*runtime, CoordinateSystem::Cartesian, error);
        INFO(error);
        REQUIRE(initialized);
    }
    return runtime.get();
}

PreparedNestedScalarFeedback prepare_flat(const tmv::domain::OneWayNestRelationshipV1& r,
                                          const std::vector<double>&                   layers,
                                          const Field3D&                               before,
                                          const Field3D&                               after,
                                          const Field3D&                               density,
                                          const Field3D&                               parent,
                                          const Field3D&                               child,
                                          const tmv::CartesianThetaInterfaceFluxes&    coarse,
                                          const tmv::CartesianThetaInterfaceFluxes&    fine,
                                          bool                                         concentration = false,
                                          bool                                         positive      = true,
                                          bool                                         allow_fine    = false)
{
    return prepare_nested_scalar_feedback_flat(r,
                                               layers,
                                               before,
                                               after,
                                               density,
                                               parent,
                                               child,
                                               coarse,
                                               fine,
                                               concentration,
                                               positive,
                                               allow_fine,
                                               test_backend());
}
PreparedNestedScalarFeedback prepare_mapped(const tmv::domain::NestedMappedTerrainGeometryV1& geometry,
                                            const Field3D&                                    before,
                                            const Field3D&                                    after,
                                            const Field3D&                                    density,
                                            const Field3D&                                    parent,
                                            const Field3D&                                    child,
                                            const tmv::CartesianThetaInterfaceFluxes&         coarse,
                                            const tmv::CartesianThetaInterfaceFluxes&         fine,
                                            bool                                              concentration = false,
                                            bool                                              positive      = true,
                                            bool                                              allow_fine    = false)
{
    return prepare_nested_scalar_feedback_mapped(geometry,
                                                 before,
                                                 after,
                                                 density,
                                                 parent,
                                                 child,
                                                 coarse,
                                                 fine,
                                                 concentration,
                                                 positive,
                                                 allow_fine,
                                                 test_backend());
}

tmv::domain::OneWayNestRelationshipV1 relationship(int ratio)
{
    return tmv::domain::prepare_one_way_nest_relationship(
        {12, 10, 4, 400, 300, .2, ratio, ratio, ratio, 4, 3, 3 * ratio, 3 * ratio, 4, 1, 1});
}
struct Fields
{
    Field3D before, after, child_density, parent, child;
    explicit Fields(const tmv::domain::OneWayNestRelationshipV1& r)
        : before(r.parent_cells_x + 2, r.parent_cells_y + 2, r.levels_z, 1), after(before),
          child_density(r.child_cells_x + 2, r.child_cells_y + 2, r.levels_z, 2), parent(before), child(child_density)
    {
        parent.fill(3);
        child.fill(5);
        for (auto* f : {&before, &after, &child_density, &parent, &child})
            for (int i = 0; i < f->size_r(); ++i)
                for (int j = 0; j < f->size_th(); ++j)
                    if (i == 0 || j == 0 || i == f->size_r() - 1 || j == f->size_th() - 1)
                        for (int k = 0; k < f->size_z(); ++k)
                            (*f)(i, j, k) = std::numeric_limits<float>::quiet_NaN();
        for (int i = 1; i < after.size_r() - 1; ++i)
            for (int j = 1; j < after.size_th() - 1; ++j)
                for (int k = 0; k < after.size_z(); ++k)
                    if ((i > r.child_origin_parent_cell_i &&
                         i <= r.child_origin_parent_cell_i + r.child_parent_span_x &&
                         j > r.child_origin_parent_cell_j &&
                         j <= r.child_origin_parent_cell_j + r.child_parent_span_y) ||
                        (i == 1 && j == 1))
                        after(i, j, k) = 2;
    }
};
tmv::CartesianThetaInterfaceFluxes fluxes(const tmv::domain::OneWayNestRelationshipV1& r)
{
    tmv::CartesianThetaInterfaceFluxes f;
    f.west.assign(r.child_parent_span_y * r.levels_z, 0);
    f.east = f.west;
    f.south.assign(r.child_parent_span_x * r.levels_z, 0);
    f.north              = f.south;
    f.accumulated_dt_s   = .2;
    f.contribution_count = 1;
    return f;
}
} // namespace
TEST_CASE("Nested scalar feedback conserves carriers under restriction, reflux and changed density",
          "[core][nested][scalar-feedback]")
{
    for (int ratio : {2, 3})
        for (bool mapped : {false, true})
            for (bool concentration : {false, true})
            {
                CAPTURE(ratio, mapped, concentration);
                const auto         r = relationship(ratio);
                Fields             f(r);
                tmv::TerrainConfig cfg;
                cfg.scheme_id               = "bell";
                cfg.coord_id                = "btf";
                cfg.horizontal_placement_id = "grid_attached";
                cfg.ztop                    = 1600;
                cfg.bell.h0                 = 200;
                cfg.bell.a                  = 1200;
                const auto geometry         = tmv::domain::build_stationary_nested_terrain_geometry(cfg, r, {0, 0}, 1);
                const std::vector<double> layers{50, 75, 125, 150};
                const auto                pv = [&](int i, int j, int k)
                {
                    const auto& g = geometry.parent.metrics;
                    return mapped ? g.dx * g.dy * g.dzeta * g.J(i - 1, j - 1, k)
                                  : r.parent_dx_m * r.parent_dy_m * layers[k];
                };
                const auto cv = [&](int i, int j, int k)
                {
                    const auto& g = geometry.child_storage.metrics;
                    return mapped ? g.dx * g.dy * g.dzeta * g.J(i, j, k) : r.child_dx_m * r.child_dy_m * layers[k];
                };
                auto coarse = fluxes(r), fine = fluxes(r);
                fine.contribution_count = ratio;
                const int oi = r.child_origin_parent_cell_i, oj = r.child_origin_parent_cell_j,
                          sx = r.child_parent_span_x, sy = r.child_parent_span_y;
                double    expected_reflux = 0;
                for (int j = 0; j < sy; ++j)
                    for (int k = 0; k < r.levels_z; ++k)
                    {
                        const auto n   = j * r.levels_z + k;
                        coarse.west[n] = 2 * pv(oi, oj + j + 1, k);
                        fine.east[n]   = 2 * pv(oi + sx + 1, oj + j + 1, k);
                        expected_reflux += coarse.west[n] + fine.east[n];
                    }
                for (int i = 0; i < sx; ++i)
                    for (int k = 0; k < r.levels_z; ++k)
                    {
                        const auto n    = i * r.levels_z + k;
                        fine.south[n]   = pv(oi + i + 1, oj, k);
                        coarse.north[n] = pv(oi + i + 1, oj + sy + 1, k);
                        expected_reflux -= fine.south[n] + coarse.north[n];
                    }
                const auto original = f.parent;
                const auto result   = mapped ? prepare_mapped(geometry,
                                                              f.before,
                                                              f.after,
                                                              f.child_density,
                                                              f.parent,
                                                              f.child,
                                                              coarse,
                                                              fine,
                                                              concentration)
                                             : prepare_flat(r,
                                                            layers,
                                                            f.before,
                                                            f.after,
                                                            f.child_density,
                                                            f.parent,
                                                            f.child,
                                                            coarse,
                                                            fine,
                                                            concentration);
                REQUIRE(result.report.complete);
                CHECK(std::memcmp(f.parent.data(), original.data(), original.size() * sizeof(float)) == 0);
                CHECK(result.report.restricted_cells == std::size_t(sx * sy * r.levels_z));
                CHECK(result.report.applied_reflux == Approx(expected_reflux).epsilon(1e-12));
                CHECK(result.report.interface_mismatch == Approx(expected_reflux).epsilon(1e-12));
                double initial = 0, final = 0, covered_before = 0, covered_after = 0, child_total = 0;
                for (int i = 1; i <= r.parent_cells_x; ++i)
                    for (int j = 1; j <= r.parent_cells_y; ++j)
                        for (int k = 0; k < r.levels_z; ++k)
                        {
                            const double before =
                                pv(i, j, k) * (concentration ? 1 : f.before(i, j, k)) * f.parent(i, j, k);
                            const double after =
                                pv(i, j, k) * (concentration ? 1 : f.after(i, j, k)) * result.scalar(i, j, k);
                            initial += before;
                            final += after;
                            if (i > oi && i <= oi + sx && j > oj && j <= oj + sy)
                            {
                                covered_before += before;
                                covered_after += after;
                            }
                        }
                for (int i = 1; i <= r.child_cells_x; ++i)
                    for (int j = 1; j <= r.child_cells_y; ++j)
                        for (int k = 0; k < r.levels_z; ++k)
                            child_total +=
                                cv(i, j, k) * (concentration ? 1 : f.child_density(i, j, k)) * f.child(i, j, k);
                CHECK(covered_after == Approx(child_total).epsilon(2e-7));
                CHECK(result.report.restriction_change == Approx(child_total - covered_before).epsilon(1e-12));
                CHECK(result.report.initial_inventory == Approx(initial).epsilon(1e-12));
                CHECK(result.report.final_inventory == Approx(final).epsilon(1e-12));
                CHECK(std::abs(final - initial - result.report.restriction_change - expected_reflux -
                               result.report.storage_rounding) < 1e-12 * final);
                CHECK(result.scalar(1, 1, 0) == (concentration ? 3 : 1.5f));
                CHECK(result.scalar(oi, oj + 1, 0) == 5);
                CHECK(result.scalar(oi + sx + 1, oj + 1, 0) == 5);
                CHECK(result.scalar(oi + 1, oj, 0) == 2);
                CHECK(result.scalar(oi + 1, oj + sy + 1, 0) == 2);
                CHECK(std::isnan(result.scalar(0, 0, 0)));
            }
}
TEST_CASE("Nested scalar feedback rejects stale geometry and impossible reflux without touching inputs",
          "[core][nested][scalar-feedback]")
{
    auto                      r = relationship(2);
    Fields                    f(r);
    const auto                original = f.parent;
    auto                      coarse = fluxes(r), fine = fluxes(r);
    const std::vector<double> layers(r.levels_z, 100);
    auto                      run = [&]
    { return prepare_flat(r, layers, f.before, f.after, f.child_density, f.parent, f.child, coarse, fine); };
    fine.west[0] = 1e30;
    CHECK_THROWS(run());
    fine.west[0]          = 0;
    fine.accumulated_dt_s = .1;
    CHECK_THROWS(run());
    fine.accumulated_dt_s = .2;
    r.child_dx_m *= 2;
    CHECK_THROWS(run());
    CHECK(std::memcmp(f.parent.data(), original.data(), original.size() * sizeof(float)) == 0);
}

TEST_CASE("Nested scalar positivity correction preserves local inventory and child ownership",
          "[core][nested][scalar-feedback][positivity]")
{
    for (int ratio : {2, 3})
    {
        const auto r = relationship(ratio);
        Fields     f(r);
        f.before.fill(1);
        f.after.fill(1);
        f.child_density.fill(1);
        f.parent.fill(0);
        f.child.fill(7);
        auto                      coarse = fluxes(r), fine = fluxes(r);
        const std::vector<double> layers(r.levels_z, 1.0 / (r.parent_dx_m * r.parent_dy_m));
        const int                 i = r.child_origin_parent_cell_i, j = r.child_origin_parent_cell_j + 1;
        f.parent(i - 1, j, 0) = 2;
        f.parent(i, j, 1)     = 6;
        fine.west[0]          = 4;
        const auto result =
            prepare_flat(r, layers, f.before, f.after, f.child_density, f.parent, f.child, coarse, fine);
        CHECK(result.scalar(i, j, 0) == 0);
        CHECK(result.scalar(i - 1, j, 0) == Approx(1));
        CHECK(result.scalar(i, j, 1) == Approx(3));
        CHECK(result.scalar(i + 1, j, 0) == Approx(7));
        CHECK(result.report.redistributed_inventory == Approx(4));
        CHECK(result.report.redistributed_cells == 1);
        CHECK(valid_nested_scalar_feedback_report(result.report));
        CHECK(result.report.closure_residual == Approx(0).margin(1e-10));
        CHECK(f.parent(i - 1, j, 0) == 2);
        fine.west[0] = 9;
        CHECK_THROWS(prepare_flat(r, layers, f.before, f.after, f.child_density, f.parent, f.child, coarse, fine));
        CHECK(f.parent(i, j, 1) == 6);
        auto invalid                    = result.report;
        invalid.redistributed_inventory = -1;
        CHECK_FALSE(valid_nested_scalar_feedback_report(invalid));
    }
}

TEST_CASE("Nested scalar positivity uses available uncovered inventory when immediate neighbors are empty",
          "[core][nested][scalar-feedback][positivity]")
{
    for (int ratio : {2, 3})
    {
        const auto r = relationship(ratio);
        Fields     fields(r);
        fields.before.fill(1);
        fields.after.fill(1);
        fields.child_density.fill(1);
        fields.parent.fill(0);
        fields.child.fill(7);
        fields.parent(1, 1, 0) = 8;
        auto coarse = fluxes(r), fine = fluxes(r);
        fine.west[0]          = 4;
        fine.west[r.levels_z] = 2;
        const std::vector<double> layers(r.levels_z, 1.0 / (r.parent_dx_m * r.parent_dy_m));
        const auto                result = prepare_flat(
            r, layers, fields.before, fields.after, fields.child_density, fields.parent, fields.child, coarse, fine);
        CHECK(result.scalar(1, 1, 0) == Approx(2));
        CHECK(result.scalar(r.child_origin_parent_cell_i, r.child_origin_parent_cell_j + 1, 0) == 0);
        CHECK(result.scalar(r.child_origin_parent_cell_i + 1, r.child_origin_parent_cell_j + 1, 0) == Approx(7));
        CHECK(result.report.global_redistributed_inventory == Approx(6));
        CHECK(result.report.global_redistributed_cells == 2);
        CHECK(result.report.redistributed_inventory == Approx(6));
        CHECK(valid_nested_scalar_feedback_report(result.report));
        CHECK(result.report.closure_residual == Approx(0).margin(1e-10));
        CHECK(fields.parent(1, 1, 0) == 8);
        fine.west[0] = 7;
        CHECK_THROWS(prepare_flat(
            r, layers, fields.before, fields.after, fields.child_density, fields.parent, fields.child, coarse, fine));
        auto invalid                           = result.report;
        invalid.global_redistributed_inventory = 7;
        CHECK_FALSE(valid_nested_scalar_feedback_report(invalid));
    }
}

TEST_CASE("Composite positive reflux synchronizes fine donors with covered coarse cells",
          "[core][nested][scalar-feedback][positivity][composite]")
{
    for (int ratio : {2, 3})
    {
        const auto r = relationship(ratio);
        Fields     fields(r);
        fields.before.fill(1);
        fields.after.fill(1);
        fields.child_density.fill(1);
        fields.parent.fill(0);
        fields.child.fill(7);
        fields.parent(1, 1, 0) = 8;
        auto coarse = fluxes(r), fine = fluxes(r);
        fine.west[0]          = 7;
        fine.west[r.levels_z] = 2;
        const std::vector<double> layers(r.levels_z, 1.0 / (r.parent_dx_m * r.parent_dy_m));
        const auto                result = prepare_flat(r,
                                                        layers,
                                                        fields.before,
                                                        fields.after,
                                                        fields.child_density,
                                                        fields.parent,
                                                        fields.child,
                                                        coarse,
                                                        fine,
                                                        false,
                                                        true,
                                                        true);
        REQUIRE_FALSE(result.child_scalar.empty());
        const double factor = 1.0 - 9.0 / 260.0;
        CHECK(result.scalar(1, 1, 0) == Approx(8 * factor));
        CHECK(result.scalar(r.child_origin_parent_cell_i, r.child_origin_parent_cell_j + 1, 0) == 0);
        CHECK(result.child_scalar(1, 1, 0) == Approx(7 * factor));
        CHECK(result.scalar(r.child_origin_parent_cell_i + 1, r.child_origin_parent_cell_j + 1, 0) ==
              result.child_scalar(1, 1, 0));
        CHECK(result.report.child_inventory == Approx(252));
        CHECK(result.report.child_synchronized_inventory == Approx(252 * factor));
        CHECK(result.report.fine_requested_transfer == Approx(252 * (1 - factor)));
        CHECK(valid_nested_scalar_feedback_report(result.report));
        CHECK(result.report.final_inventory == Approx(251).epsilon(2e-7));
        CHECK(fields.child(1, 1, 0) == 7);
        auto invalid = result.report;
        invalid.child_synchronized_inventory += 1;
        CHECK_FALSE(valid_nested_scalar_feedback_report(invalid));
        fine.west[0] = 1000;
        CHECK_THROWS(prepare_flat(r,
                                  layers,
                                  fields.before,
                                  fields.after,
                                  fields.child_density,
                                  fields.parent,
                                  fields.child,
                                  coarse,
                                  fine,
                                  false,
                                  true,
                                  true));
    }
}

TEST_CASE("Mapped composite positivity preserves terrain-volume inventory on both levels",
          "[core][nested][scalar-feedback][positivity][composite][mapped]")
{
    for (int ratio : {2, 3})
    {
        const auto r = relationship(ratio);
        Fields     fields(r);
        fields.before.fill(1);
        fields.after.fill(1);
        fields.child_density.fill(1);
        fields.parent.fill(0);
        fields.child.fill(7);
        fields.parent(1, 1, 0) = 8;
        tmv::TerrainConfig config;
        config.scheme_id               = "schar";
        config.coord_id                = "btf";
        config.horizontal_placement_id = "grid_attached";
        config.ztop                    = 1600;
        config.schar.H                 = 200;
        config.schar.a                 = 1200;
        config.schar.ell               = 800;
        const auto   geometry          = tmv::domain::build_stationary_nested_terrain_geometry(config, r, {0, 0}, 1);
        const auto&  p                 = geometry.parent.metrics;
        const auto&  c                 = geometry.child_storage.metrics;
        const double coarse_inventory  = 8 * p.dx * p.dy * p.dzeta * p.J(0, 0, 0);
        double       fine_inventory    = 0;
        for (int i = 1; i <= r.child_cells_x; ++i)
            for (int j = 1; j <= r.child_cells_y; ++j)
                for (int k = 0; k < r.levels_z; ++k)
                    fine_inventory += 7 * c.dx * c.dy * c.dzeta * c.J(i, j, k);
        const double deficit = coarse_inventory + fine_inventory / 4;
        const double factor  = 1 - deficit / (coarse_inventory + fine_inventory);
        auto         coarse = fluxes(r), fine = fluxes(r);
        fine.west[0]      = deficit;
        const auto result = prepare_mapped(geometry,
                                           fields.before,
                                           fields.after,
                                           fields.child_density,
                                           fields.parent,
                                           fields.child,
                                           coarse,
                                           fine,
                                           false,
                                           true,
                                           true);
        REQUIRE_FALSE(result.child_scalar.empty());
        CHECK(valid_nested_scalar_feedback_report(result.report));
        CHECK(result.scalar(1, 1, 0) == Approx(8 * factor));
        CHECK(result.child_scalar(1, 1, 0) == Approx(7 * factor));
        CHECK(result.report.fine_requested_transfer == Approx(fine_inventory * (1 - factor)).epsilon(1e-12));
        double covered = 0, synchronized = 0, final_inventory = 0;
        for (int i = 1; i <= r.parent_cells_x; ++i)
            for (int j = 1; j <= r.parent_cells_y; ++j)
                for (int k = 0; k < r.levels_z; ++k)
                {
                    const double mass = result.scalar(i, j, k) * p.dx * p.dy * p.dzeta * p.J(i - 1, j - 1, k);
                    final_inventory += mass;
                    if (i > r.child_origin_parent_cell_i && i <= r.child_origin_parent_cell_i + r.child_parent_span_x &&
                        j > r.child_origin_parent_cell_j && j <= r.child_origin_parent_cell_j + r.child_parent_span_y)
                        covered += mass;
                }
        for (int i = 1; i <= r.child_cells_x; ++i)
            for (int j = 1; j <= r.child_cells_y; ++j)
                for (int k = 0; k < r.levels_z; ++k)
                    synchronized += result.child_scalar(i, j, k) * c.dx * c.dy * c.dzeta * c.J(i, j, k);
        CHECK(covered == Approx(synchronized).epsilon(2e-7));
        CHECK(result.report.child_synchronized_inventory == Approx(synchronized).epsilon(1e-12));
        CHECK(final_inventory ==
              Approx(coarse_inventory + fine_inventory - deficit + result.report.storage_rounding).epsilon(1e-12));
        CHECK(fields.child(1, 1, 0) == 7);
    }
}

TEST_CASE("Nested scalar lower bounds preserve local and fine donor populations",
          "[core][nested][scalar-feedback][bounds]")
{
    for (int ratio : {2, 3})
    {
        const auto r = relationship(ratio);
        Fields     f(r);
        f.before.fill(1);
        f.after.fill(1);
        f.child_density.fill(1);
        f.parent.fill(0);
        f.child.fill(7);
        Field3D                   lower(f.parent.size_r(), f.parent.size_th(), f.parent.size_z(), 0);
        Field3D                   fine_lower(f.child.size_r(), f.child.size_th(), f.child.size_z(), 0);
        auto                      coarse = fluxes(r), fine = fluxes(r);
        const std::vector<double> layers(r.levels_z, 1.0 / (r.parent_dx_m * r.parent_dy_m));
        const int                 i = r.child_origin_parent_cell_i, j = r.child_origin_parent_cell_j + 1;
        f.parent(i - 1, j, 0) = 2;
        f.parent(i, j, 1)     = 6;
        lower(i, j, 0)        = 1;
        lower(i - 1, j, 0)    = 1;
        lower(i, j, 1)        = 3;
        fine.west[0]          = 2;
        const auto run        = [&](bool allow_fine)
        {
            return prepare_nested_scalar_feedback_flat(r,
                                                       layers,
                                                       f.before,
                                                       f.after,
                                                       f.child_density,
                                                       f.parent,
                                                       f.child,
                                                       coarse,
                                                       fine,
                                                       false,
                                                       true,
                                                       allow_fine,
                                                       test_backend(),
                                                       {&lower, &fine_lower});
        };
        const auto local = run(false);
        CHECK(local.scalar(i, j, 0) == 1);
        CHECK(local.scalar(i - 1, j, 0) == Approx(1.25));
        CHECK(local.scalar(i, j, 1) == Approx(3.75));
        CHECK(local.report.redistributed_inventory == Approx(3));
        CHECK(local.report.constrained_cells == 3);
        CHECK(valid_nested_scalar_feedback_report(local.report));
        f.parent.fill(0);
        lower.fill(0);
        f.parent(1, 1, 0)  = 8;
        lower(1, 1, 0)     = 4;
        lower(i, j, 0)     = 1;
        lower(i, j + 1, 0) = 1;
        fine_lower.fill(6);
        fine.west[0]          = 7;
        fine.west[r.levels_z] = 2;
        const auto composite  = run(true);
        REQUIRE_FALSE(composite.child_scalar.empty());
        CHECK(composite.scalar(1, 1, 0) == Approx(6.9));
        CHECK(composite.scalar(i, j, 0) == 1);
        CHECK(composite.scalar(i, j + 1, 0) == 1);
        CHECK(composite.child_scalar(1, 1, 0) == Approx(6.725));
        CHECK(composite.scalar(i + 1, j, 0) == composite.child_scalar(1, 1, 0));
        CHECK(composite.report.final_inventory == Approx(251).epsilon(2e-7));
        CHECK(valid_nested_scalar_feedback_report(composite.report));
        const auto retained_parent = f.parent, retained_child = f.child;
        fine_lower.fill(8);
        CHECK_THROWS(run(true));
        CHECK(std::memcmp(f.parent.data(), retained_parent.data(), f.parent.size() * sizeof(float)) == 0);
        CHECK(std::memcmp(f.child.data(), retained_child.data(), f.child.size() * sizeof(float)) == 0);
        fine_lower.fill(7);
        CHECK_THROWS(run(true));
    }
}

TEST_CASE("Composite reflux retains transfers smaller than fine inventory roundoff",
          "[core][nested][scalar-feedback][positivity][composite][bounds]")
{
    for (int ratio : {2, 3})
        for (bool bounded : {false, true})
        {
            const auto r = relationship(ratio);
            Fields     fields(r);
            fields.before.fill(1);
            fields.after.fill(1);
            fields.child_density.fill(1.125f);
            fields.parent.fill(0);
            fields.child.fill(1e15f);
            Field3D lower(fields.parent.size_r(), fields.parent.size_th(), fields.parent.size_z(), 0);
            Field3D fine_lower(fields.child.size_r(), fields.child.size_th(), fields.child.size_z(), 1.2345e14f);
            const std::vector<double> layers(r.levels_z, 1.0 / (r.parent_dx_m * r.parent_dy_m));
            auto                      coarse = fluxes(r), fine = fluxes(r);
            fine.west[0]      = .125;
            const auto result = prepare_nested_scalar_feedback_flat(
                r,
                layers,
                fields.before,
                fields.after,
                fields.child_density,
                fields.parent,
                fields.child,
                coarse,
                fine,
                false,
                true,
                true,
                test_backend(),
                bounded ? NestedScalarFeedbackBounds{&lower, &fine_lower} : NestedScalarFeedbackBounds{});
            REQUIRE_FALSE(result.child_scalar.empty());
            CHECK(valid_nested_scalar_feedback_report(result.report));
            CHECK(result.report.fine_requested_transfer == Approx(.125).epsilon(1e-12));
            CHECK(result.report.fine_storage_rounding == Approx(.125).epsilon(1e-12));
            CHECK(result.report.child_synchronized_inventory == result.report.child_inventory);
            CHECK(std::memcmp(fields.child.data(), result.child_scalar.data(), fields.child.size() * sizeof(float)) ==
                  0);
        }
}

TEST_CASE("Private feedback carries resolved mass with conserved populations atomically",
          "[core][nested][private-feedback][bounds]")
{
    for (int ratio : {2, 3})
    {
        const auto r = relationship(ratio);
        Fields     f(r);
        f.before.fill(1);
        f.after.fill(1);
        f.child_density.fill(1);
        f.parent.fill(8);
        f.child.fill(7);
        Field3D   mass_before(f.parent.size_r(), f.parent.size_th(), f.parent.size_z(), 1);
        Field3D   mass_after = mass_before;
        Field3D   child_mass(f.child.size_r(), f.child.size_th(), f.child.size_z(), 1);
        const int i = r.child_origin_parent_cell_i, j = r.child_origin_parent_cell_j + 1;
        f.parent(i, j, 0)          = 0;
        mass_before(i, j, 0)       = 0;
        mass_after(i, j, 0)        = .25f;
        const auto retained_parent = f.parent, retained_child = f.child;
        auto       coarse = fluxes(r), fine = fluxes(r);
        fine.west[0] = 2;
        const std::vector<double>                     layers(r.levels_z, 1.0 / (r.parent_dx_m * r.parent_dy_m));
        std::vector<NestedPrivateScalarFeedbackInput> inputs{{"thompson.Nc_kg-1",
                                                              &f.parent,
                                                              &f.child,
                                                              &coarse,
                                                              &fine,
                                                              250,
                                                              0,
                                                              0,
                                                              0,
                                                              &mass_before,
                                                              &mass_after,
                                                              &child_mass}};
        const auto                                    run = [&]()
        {
            return prepare_nested_private_feedback(
                r, layers, nullptr, f.before, f.after, f.child_density, inputs, test_backend());
        };
        const auto result = run();
        REQUIRE(valid_nested_private_feedback_reports(result.reports));
        CHECK(result.parent[0](i, j, 0) == Approx(1.75));
        CHECK(result.reports[0].carrier.parent.redistributed_inventory == Approx(3.75));
        CHECK(result.reports[0].carrier.parent.constrained_cells > 0);
        for (int x = 1; x <= r.parent_cells_x; ++x)
            for (int y = 1; y <= r.parent_cells_y; ++y)
                for (int k = 0; k < r.levels_z; ++k)
                    CHECK(result.parent[0](x, y, k) >= 7 * mass_after(x, y, k));
        CHECK(std::memcmp(f.parent.data(), retained_parent.data(), f.parent.size() * sizeof(float)) == 0);
        CHECK(std::memcmp(f.child.data(), retained_child.data(), f.child.size() * sizeof(float)) == 0);
        inputs[0].parent_mass_after = nullptr;
        CHECK_THROWS(run());
        inputs[0].parent_mass_after = &mass_after;
        child_mass(1, 1, 0)         = std::numeric_limits<float>::quiet_NaN();
        CHECK_THROWS(run());
        CHECK(std::memcmp(f.parent.data(), retained_parent.data(), f.parent.size() * sizeof(float)) == 0);
        CHECK(std::memcmp(f.child.data(), retained_child.data(), f.child.size() * sizeof(float)) == 0);
    }
}

TEST_CASE("Private feedback replays certified evaporation before particle synchronization",
          "[core][nested][private-feedback][reaction]")
{
    for (int ratio : {2, 3})
    {
        const auto r = relationship(ratio);
        Fields     f(r);
        f.before.fill(1);
        f.after.fill(2);
        f.child_density.fill(1);
        f.parent.fill(0);
        f.child.fill(0);
        Field3D cloud_mass = f.parent, child_cloud_mass = f.child;
        Field3D aerosol = f.parent, child_aerosol = f.child, returned = f.parent;
        aerosol.fill(100);
        child_aerosol.fill(100);
        returned(1, 1, 0) = 4;
        auto coarse = fluxes(r), fine = coarse, aerosol_flux = coarse;
        fine.west[0] = 2;
        const std::vector<double>                     layers(r.levels_z, 1.0 / (r.parent_dx_m * r.parent_dy_m));
        std::vector<NestedPrivateScalarFeedbackInput> inputs{
            {"thompson.Nc_kg-1",
             &f.parent,
             &f.child,
             &coarse,
             &fine,
             0,
             0,
             -2,
             0,
             &cloud_mass,
             &cloud_mass,
             &child_cloud_mass},
            {"thompson.Nwfa_kg-1",
             &aerosol,
             &child_aerosol,
             &aerosol_flux,
             &aerosol_flux,
             100.0 * r.child_parent_span_x * r.child_parent_span_y * r.levels_z,
             0,
             0,
             0}};
        const auto run = [&](const Field3D* provenance)
        {
            return prepare_nested_private_feedback(
                r, layers, nullptr, f.before, f.after, f.child_density, inputs, test_backend(), provenance);
        };
        CHECK_THROWS_AS(run(nullptr), NestedScalarInventoryExhausted);
        const auto result = run(&returned);
        REQUIRE(valid_nested_private_feedback_reports(result.reports));
        CHECK(result.parent[0](1, 1, 0) == 0);
        CHECK(result.parent[1](1, 1, 0) == Approx(49));
        CHECK(result.reports[0].carrier.parent.initial_inventory == 0);
        CHECK(result.reports[0].carrier.parent.final_inventory == 0);
        CHECK(result.reports[0].carrier.parent.applied_reflux == Approx(-2));
        CHECK(result.reports[0].carrier.parent.phase_exchange_inventory == Approx(2));
        CHECK(result.reports[1].carrier.parent.phase_exchange_inventory == Approx(-2));
        auto  invalid_reports = result.reports;
        auto& unpaired        = invalid_reports[1].carrier.parent;
        unpaired.storage_rounding += unpaired.phase_exchange_inventory;
        unpaired.phase_exchange_inventory = 0;
        REQUIRE(valid_nested_water_feedback_report(invalid_reports[1].carrier));
        CHECK_FALSE(valid_nested_private_feedback_reports(invalid_reports));
        returned(1, 1, 0) = 1;
        CHECK_THROWS_AS(run(&returned), NestedScalarInventoryExhausted);
        returned(1, 1, 0) = -1;
        CHECK_THROWS_AS(run(&returned), std::invalid_argument);
        returned(1, 1, 0)   = 4;
        cloud_mass(1, 1, 0) = .01f;
        CHECK_THROWS_AS(run(&returned), std::invalid_argument);
        CHECK(f.parent(1, 1, 0) == 0);
        CHECK(aerosol(1, 1, 0) == 100);
        CHECK(returned(1, 1, 0) == 4);
    }
}
