// PBL tests pin unambiguous column properties: flux signs, drag direction,
// friction-velocity behavior, finite tendencies, forcing gates, and no-op
// behavior. They avoid pinning unaudited closure choices as scientific truth.
#include "catch2/catch.hpp"
#include "boundary_layer/boundary_layer_base.hpp"
#include "boundary_layer/base/surface_fluxes.hpp"
#include "boundary_layer/base/vertical_diffusion.hpp"
#include "boundary_layer/base/pbl_height.hpp"
#include "boundary_layer/schemes/mynn/mynn_stability.hpp"
#include "boundary_layer/schemes/mynn/mynn.hpp"
#include "compute/dispatch/boundary_layer.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace
{

// A simple, well-posed test column: nz cell-centered levels and nz+1
// interfaces (slab/YSU index z_int[nz], so the interface array must be nz+1).
struct TestColumn
{
    std::vector<double> u, v, theta, qv, p, rho, z_int, tke, qc, qi,
        radiative_theta_tendency;

    explicit TestColumn(int nz, double u0, double theta_air)
    {
        u.assign(nz, u0);
        v.assign(nz, 0.0);
        theta.assign(nz, 0.0);
        qv.assign(nz, 0.008);
        p.assign(nz, 0.0);
        rho.assign(nz, 1.15);
        tke.assign(nz, 0.1);
        qc.assign(nz, 0.0);
        qi.assign(nz, 0.0);
        radiative_theta_tendency.assign(nz, 0.0);
        z_int.assign(nz + 1, 0.0);

        const double dz = 100.0;
        for (int k = 0; k <= nz; ++k) z_int[k] = k * dz;
        for (int k = 0; k < nz; ++k)
        {
            const double z_mid = 0.5 * (z_int[k] + z_int[k + 1]);
            // gently stable aloft so profiles are physical and finite
            theta[k] = theta_air + 0.003 * z_mid;
            p[k] = 100000.0 * std::exp(-z_mid / 8000.0);
        }
    }

    BoundaryLayerColumnStateView view(double u_sfc, double v_sfc,
                                      double theta_sfc, double qv_sfc,
                                      SlabColumnState* slab_state = nullptr,
                                      double elapsed_time_s = 0.0) const
    {
        BoundaryLayerColumnStateView col;
        col.u = &u; col.v = &v; col.theta = &theta; col.qv = &qv;
        col.p = &p; col.rho = &rho; col.z_int = &z_int;
        col.tke = &tke; col.qc = &qc; col.qi = &qi;
        col.radiative_theta_tendency = &radiative_theta_tendency;
        col.u_sfc = u_sfc; col.v_sfc = v_sfc;
        col.theta_sfc = theta_sfc; col.qv_sfc = qv_sfc;
        col.z_sfc = 10.0;
        col.slab_state = slab_state;
        col.elapsed_time_s = elapsed_time_s;
        return col;
    }
};

bool all_finite(const std::vector<double>& v)
{
    return std::all_of(v.begin(), v.end(),
                       [](double x) { return std::isfinite(x); });
}

bool tendencies_finite(const BoundaryLayerTendencies& t)
{
    return all_finite(t.dudt_pbl) && all_finite(t.dvdt_pbl) &&
           all_finite(t.dthetadt_pbl) && all_finite(t.dqvdt_pbl) &&
           all_finite(t.dqcdt_pbl) && all_finite(t.dqidt_pbl) &&
           all_finite(t.dtkedt_pbl);
}

class CpuMynnBatchKernels final : public BoundaryLayerComputeKernels
{
public:
    bool reject = false;
    int dispatches = 0;

    bool supports(BoundaryLayerKernelOp op) const override
    {
        return op == BoundaryLayerKernelOp::MynnLevel25Stability;
    }

    bool dispatch_mynn_level25_stability(
        const MynnLevel25StabilityDispatch& dispatch) override
    {
        ++dispatches;
        if (reject) return false;
        std::array<std::vector<float>, 5> candidates;
        for (auto& values : candidates)
            values.resize(static_cast<std::size_t>(dispatch.count));
        for (int index = 0; index < dispatch.count; ++index)
        {
            const auto result = tmv::mynn::level25_stability_functions({
                dispatch.gm_s2[index], dispatch.gh_s2[index],
                dispatch.mixing_length_m[index],
                dispatch.q_squared_m2s2[index]});
            candidates[0][static_cast<std::size_t>(index)] =
                static_cast<float>(result.sm);
            candidates[1][static_cast<std::size_t>(index)] =
                static_cast<float>(result.sh);
            candidates[2][static_cast<std::size_t>(index)] =
                static_cast<float>(result.gradient_richardson);
            candidates[3][static_cast<std::size_t>(index)] =
                static_cast<float>(result.flux_richardson);
            candidates[4][static_cast<std::size_t>(index)] =
                static_cast<float>(result.equilibrium_limiter);
        }
        const std::size_t bytes = static_cast<std::size_t>(dispatch.count) *
                                  sizeof(float);
        std::memcpy(dispatch.sm, candidates[0].data(), bytes);
        std::memcpy(dispatch.sh, candidates[1].data(), bytes);
        std::memcpy(dispatch.gradient_richardson, candidates[2].data(), bytes);
        std::memcpy(dispatch.flux_richardson, candidates[3].data(), bytes);
        std::memcpy(dispatch.equilibrium_limiter, candidates[4].data(), bytes);
        return true;
    }
};

} // namespace

TEST_CASE("PBL schemes publish the selected theta and TKE top diagnostics",
          "[physics][boundary_layer][pbl-top]")
{
    SECTION("potential-temperature increase")
    {
        TestColumn column(6, 5.0, 300.0);
        column.theta = {300.2, 299.5, 299.8, 300.4, 301.2, 301.8};
        BoundaryLayerConfig config;
        config.apply_surface_fluxes = false;
        config.pbl_top_method = "theta";
        config.pbl_max_height = 600.0;
        const double expected = boundary_layer::pbl::diagnose_pbl_height_theta(
            column.theta, column.z_int,
            boundary_layer::pbl::theta_increase_threshold_k,
            100.0, config.pbl_max_height);
        REQUIRE(expected == Approx(425.0));

        for (const std::string scheme_id : {"slab", "ysu", "mynn"})
        {
            CAPTURE(scheme_id);
            auto scheme = create_boundary_layer_scheme(scheme_id);
            config.enable_tke = scheme_id == "mynn";
            scheme->initialize(config);
            SlabColumnState slab_state;
            BoundaryLayerTendencies tendencies;
            BoundaryLayerDiagnostics diagnostics;
            scheme->compute_column(
                config, SurfaceConfig{},
                column.view(5.0, 0.0, 300.0, 0.008,
                            scheme_id == "slab" ? &slab_state : nullptr),
                tendencies, &diagnostics);
            REQUIRE(diagnostics.pbl_height == Approx(expected).margin(1.0e-12));
        }
    }

    SECTION("prognostic-TKE threshold")
    {
        TestColumn column(6, 5.0, 300.0);
        column.tke = {0.2, 0.15, 0.08, 0.03, 0.005, 0.001};
        BoundaryLayerConfig config;
        config.apply_surface_fluxes = false;
        config.enable_tke = true;
        config.pbl_top_method = "tke";
        config.pbl_max_height = 600.0;
        const double expected = boundary_layer::pbl::diagnose_pbl_height_tke(
            column.tke, column.z_int,
            boundary_layer::pbl::tke_diagnostic_threshold_m2s2(column.tke),
            100.0, config.pbl_max_height);
        REQUIRE(expected == Approx(430.0));

        auto scheme = create_boundary_layer_scheme("mynn");
        scheme->initialize(config);
        BoundaryLayerTendencies tendencies;
        BoundaryLayerDiagnostics diagnostics;
        scheme->compute_column(
            config, SurfaceConfig{},
            column.view(5.0, 0.0, 300.0, 0.008),
            tendencies, &diagnostics);
        REQUIRE(diagnostics.pbl_height == Approx(expected).margin(1.0e-12));
    }
}

TEST_CASE("MYNN native column batch matches CPU closure and rejects atomically",
          "[physics][boundary_layer][mynn][batch][backend]")
{
    BoundaryLayerConfig config;
    config.dt_pbl = 30.0;
    config.pbl_max_height = 1800.0;
    config.apply_surface_fluxes = true;
    SurfaceConfig surface;
    surface.z0m = 0.1;
    surface.z0h = 0.01;

    constexpr std::size_t column_count = 3;
    std::vector<TestColumn> columns;
    columns.reserve(column_count);
    std::vector<BoundaryLayerColumnWorkspace> workspaces(column_count);
    std::vector<BoundaryLayerColumnStateView> views;
    views.reserve(column_count);
    for (std::size_t column = 0; column < column_count; ++column)
    {
        columns.emplace_back(10, 7.0 + column, 299.0 + column);
        for (std::size_t level = 0; level < columns.back().u.size(); ++level)
        {
            columns.back().u[level] += 0.4 * level * (column + 1);
            columns.back().theta[level] +=
                (column == 1 ? -0.004 : 0.005) *
                columns.back().z_int[level];
        }
        auto view = columns.back().view(
            5.0 + column, 0.3 * column, 304.0, 0.012);
        view.workspace = &workspaces[column];
        views.push_back(view);
    }

    MYNNScheme cpu;
    MYNNScheme native;
    cpu.initialize(config);
    native.initialize(config);
    std::vector<BoundaryLayerTendencies> expected(column_count);
    std::vector<BoundaryLayerDiagnostics> expected_diagnostics(column_count);
    for (std::size_t column = 0; column < column_count; ++column)
        cpu.compute_column(
            config, surface, views[column], expected[column],
            &expected_diagnostics[column]);

    std::vector<BoundaryLayerTendencies> actual(column_count);
    std::vector<BoundaryLayerDiagnostics> actual_diagnostics(column_count);
    CpuMynnBatchKernels kernels;
    REQUIRE(native.compute_columns_native(
        config, surface, views.data(), views.size(), actual.data(),
        actual_diagnostics.data(), kernels));
    REQUIRE(kernels.dispatches == 1);
    for (std::size_t column = 0; column < column_count; ++column)
    {
        const std::vector<double>* expected_fields[] = {
            &expected[column].dudt_pbl, &expected[column].dvdt_pbl,
            &expected[column].dthetadt_pbl, &expected[column].dqvdt_pbl,
            &expected[column].dtkedt_pbl};
        const std::vector<double>* actual_fields[] = {
            &actual[column].dudt_pbl, &actual[column].dvdt_pbl,
            &actual[column].dthetadt_pbl, &actual[column].dqvdt_pbl,
            &actual[column].dtkedt_pbl};
        for (std::size_t field = 0; field < 5; ++field)
            for (std::size_t level = 0;
                 level < expected_fields[field]->size(); ++level)
            {
                CAPTURE(column, field, level);
                REQUIRE((*actual_fields[field])[level] ==
                        Approx((*expected_fields[field])[level])
                            .epsilon(3.0e-5).margin(3.0e-7));
            }
        REQUIRE(actual_diagnostics[column].pbl_height ==
                expected_diagnostics[column].pbl_height);
    }

    const auto preserved = actual;
    kernels.reject = true;
    REQUIRE_FALSE(native.compute_columns_native(
        config, surface, views.data(), views.size(), actual.data(),
        actual_diagnostics.data(), kernels));
    for (std::size_t column = 0; column < column_count; ++column)
    {
        REQUIRE(actual[column].dudt_pbl == preserved[column].dudt_pbl);
        REQUIRE(actual[column].dvdt_pbl == preserved[column].dvdt_pbl);
        REQUIRE(actual[column].dthetadt_pbl == preserved[column].dthetadt_pbl);
        REQUIRE(actual[column].dqvdt_pbl == preserved[column].dqvdt_pbl);
        REQUIRE(actual[column].dtkedt_pbl == preserved[column].dtkedt_pbl);
    }
}

// ─────────────────────────────────────────────────────────────
// Factory
// ─────────────────────────────────────────────────────────────

TEST_CASE("BL factory builds every scheme with a matching name",
          "[physics][boundary_layer]")
{
    for (const std::string id : {"slab", "ysu", "mynn", "none"})
    {
        auto s = create_boundary_layer_scheme(id);
        REQUIRE(s != nullptr);
        REQUIRE(s->name() == id);
    }

    const auto ids = get_available_boundary_layer_schemes();
    for (const std::string id : {"slab", "ysu", "mynn", "none"})
    {
        REQUIRE(std::find(ids.begin(), ids.end(), id) != ids.end());
    }
}

TEST_CASE("BL factory resolves aliases to canonical schemes",
          "[physics][boundary_layer]")
{
    REQUIRE(create_boundary_layer_scheme("yonsei")->name() == "ysu");
    REQUIRE(create_boundary_layer_scheme("mixed_layer")->name() == "slab");
    REQUIRE(create_boundary_layer_scheme("off")->name() == "none");
}

TEST_CASE("BL descriptors define executable capabilities",
          "[physics][boundary_layer][descriptor]")
{
    const auto& descriptors = get_boundary_layer_scheme_descriptors();
    REQUIRE(descriptors.size() == 4);
    for (const auto& descriptor : descriptors)
        REQUIRE(has_complete_scheme_capabilities(descriptor));

    const auto* slab = find_scheme_descriptor(descriptors, "mixed-layer");
    REQUIRE(slab != nullptr);
    REQUIRE(slab->id == "slab");
    REQUIRE(slab->maturity == tmv::SchemeMaturity::Qualified);
    REQUIRE(slab->capabilities.solver_high_water_delta_fields == 31);
    REQUIRE(slab->capabilities.storage.persistent_full_volume_fields == 4);
    REQUIRE(slab->capabilities.storage.peak_scratch_full_volume_fields == 9);
    REQUIRE(slab->capabilities.gpu_implementation_coverage
            == tmv::SchemeCoverage::Covered);
    REQUIRE(slab->capabilities.staggerings
            == std::vector<std::string>{"collocated", "c_grid"});

    const auto* mynn = find_scheme_descriptor(descriptors, "mynn2");
    REQUIRE(mynn != nullptr);
    REQUIRE(mynn->maturity == tmv::SchemeMaturity::Qualified);
    REQUIRE(mynn->runtime_selectable);
    REQUIRE(mynn->capabilities.gpu_implementation_coverage
            == tmv::SchemeCoverage::Covered);
    REQUIRE(mynn->capabilities.cpu_gpu_parity_coverage
            == tmv::SchemeCoverage::Covered);
    REQUIRE(mynn->capabilities.staggerings
            == std::vector<std::string>{"collocated", "c_grid"});

    const auto* none = find_scheme_descriptor(descriptors, "disabled");
    REQUIRE(none != nullptr);
    REQUIRE(none->capabilities.solver_high_water_delta_fields == 0);
    REQUIRE(none->capabilities.staggerings
            == std::vector<std::string>{"collocated", "c_grid"});
}

TEST_CASE("MYNN level-2.5 stability matches the author reference equations",
          "[physics][boundary_layer][mynn][stability]")
{
    using tmv::mynn::Level25StabilityInput;
    using tmv::mynn::level25_stability_functions;

    const auto neutral = level25_stability_functions(
        Level25StabilityInput{1.0e-3, 0.0, 10.0, 0.2});
    const auto stable = level25_stability_functions(
        Level25StabilityInput{1.0e-3, -1.0e-4, 10.0, 0.2});
    const auto unstable = level25_stability_functions(
        Level25StabilityInput{1.0e-3, 1.0e-4, 10.0, 0.2});

    // Independent values transcribed from MYNNmodel.f at the published
    // Nakanishi--Niino constants. These cover the level-2 flux-Richardson
    // solution and the level-2.5 equilibrium limiter.
    REQUIRE(neutral.sm == Approx(0.1699707418879588).epsilon(2.0e-13));
    REQUIRE(neutral.sh == Approx(0.2296901917404849).epsilon(2.0e-13));
    REQUIRE(stable.sm == Approx(0.1563027033690401).epsilon(2.0e-13));
    REQUIRE(stable.sh == Approx(0.1912945107353258).epsilon(2.0e-13));
    REQUIRE(unstable.sm == Approx(0.1761834001146163).epsilon(2.0e-13));
    REQUIRE(unstable.sh == Approx(0.2508085020332575).epsilon(2.0e-13));
    REQUIRE(stable.gradient_richardson > 0.0);
    REQUIRE(unstable.gradient_richardson < 0.0);
    REQUIRE(stable.sm < neutral.sm);
    REQUIRE(stable.sh < neutral.sh);
    REQUIRE(unstable.sm > neutral.sm);
    REQUIRE(unstable.sh > neutral.sh);

    REQUIRE_THROWS_AS(
        level25_stability_functions(
            Level25StabilityInput{-1.0, 0.0, 10.0, 0.2}),
        std::invalid_argument);
}

TEST_CASE("MYNN column diffusivity responds to stable and unstable stratification",
          "[physics][boundary_layer][mynn][stability][column]")
{
    auto evaluate = [](double theta_gradient_kpm) {
        auto scheme = create_boundary_layer_scheme("mynn");
        BoundaryLayerConfig config;
        config.apply_surface_fluxes = false;
        config.enable_tke = true;
        scheme->initialize(config);
        TestColumn column(12, 2.0, 300.0);
        for (std::size_t k = 0; k < column.theta.size(); ++k)
        {
            const double z = 0.5 *
                (column.z_int[k] + column.z_int[k + 1]);
            column.u[k] = 2.0 + static_cast<double>(k);
            column.theta[k] = 300.0 + theta_gradient_kpm * z;
            column.qv[k] = 0.008;
            column.tke[k] = 0.1;
        }
        BoundaryLayerTendencies tendencies;
        BoundaryLayerDiagnostics diagnostics;
        scheme->compute_column(
            config, SurfaceConfig{},
            column.view(column.u[0], column.v[0],
                        column.theta[0], column.qv[0]),
            tendencies, &diagnostics);
        return diagnostics;
    };

    const auto stable = evaluate(3.0e-4);
    const auto neutral = evaluate(0.0);
    const auto unstable = evaluate(-3.0e-4);
    REQUIRE(stable.K_m[1] < neutral.K_m[1]);
    REQUIRE(stable.K_h[1] < neutral.K_h[1]);
    REQUIRE(unstable.K_m[1] > neutral.K_m[1]);
    REQUIRE(unstable.K_h[1] > neutral.K_h[1]);
    REQUIRE(stable.K_m.back() == 0.0);
    REQUIRE(stable.K_h.back() == 0.0);
}

// ─────────────────────────────────────────────────────────────
// Surface-flux kernel (bulk)
// ─────────────────────────────────────────────────────────────

TEST_CASE("Surface fluxes: warm, moist, windy surface drives upward H/E and drag",
          "[physics][boundary_layer][surface]")
{
    const BoundaryLayerConfig bl;      // surface_layer_id = "bulk"
    const SurfaceConfig sfc;
    const double u = 5.0, v = 0.0;
    const auto f = surface_fluxes::compute_surface_fluxes(
        bl, sfc, u, v,
        /*theta_sfc=*/302.0, /*qv_sfc=*/0.012,
        /*theta_air=*/300.0, /*qv_air=*/0.008,
        /*p_air=*/1.0e5, /*rho_air=*/1.15, /*z_sfc=*/10.0);

    REQUIRE(std::isfinite(f.ustar));
    REQUIRE(f.ustar > 0.0);
    REQUIRE(f.Cd > 0.0);
    REQUIRE(f.Ch > 0.0);
    REQUIRE(f.H > 0.0);              // warm surface -> upward sensible heat
    REQUIRE(f.E > 0.0);              // moist surface -> upward latent heat
    REQUIRE(f.tau_u < 0.0);         // drag opposes the +u wind
}

TEST_CASE("Surface fluxes: cold surface gives downward sensible heat",
          "[physics][boundary_layer][surface]")
{
    const BoundaryLayerConfig bl;
    const SurfaceConfig sfc;
    const auto f = surface_fluxes::compute_surface_fluxes(
        bl, sfc, 5.0, 0.0,
        /*theta_sfc=*/298.0, /*qv_sfc=*/0.008,
        /*theta_air=*/300.0, /*qv_air=*/0.008,
        1.0e5, 1.15, 10.0);

    REQUIRE(std::isfinite(f.H));
    REQUIRE(f.H < 0.0);             // cold surface -> downward sensible heat
}

TEST_CASE("Surface sensible heat uses temperature rather than theta currency",
          "[physics][boundary_layer][surface][terrain][budget]")
{
    const BoundaryLayerConfig bl;
    const SurfaceConfig sfc;
    const auto sea_level = surface_fluxes::compute_surface_fluxes(
        bl, sfc, 5.0, 0.0, 302.0, 0.012, 300.0, 0.008,
        physical_constants::reference_pressure_pa, 1.15, 10.0);
    constexpr double elevated_pressure_pa = 70000.0;
    const auto elevated = surface_fluxes::compute_surface_fluxes(
        bl, sfc, 5.0, 0.0, 302.0, 0.012, 300.0, 0.008,
        elevated_pressure_pa, 1.15, 10.0);
    const double elevated_exner = std::pow(
        elevated_pressure_pa / physical_constants::reference_pressure_pa,
        physical_constants::gas_constant_dry_air_jkgk /
            physical_constants::specific_heat_cp_jkgk);

    // Identical potential-temperature transfer produces identical theta flux,
    // while true sensible heat is smaller at lower Exner temperature.
    REQUIRE(elevated.H == Approx(sea_level.H * elevated_exner).epsilon(1.0e-13));
    REQUIRE(elevated.H /
                (boundary_layer_constants::cp * elevated_exner) ==
            Approx(sea_level.H / boundary_layer_constants::cp).epsilon(1.0e-13));
}

TEST_CASE("Terrain surface stress owns complete tangent-wind work",
          "[physics][boundary_layer][surface][terrain][traction][work]")
{
    BoundaryLayerConfig config;
    config.surface_layer_id = "bulk";
    const SurfaceConfig surface;
    constexpr double u = 5.0;
    constexpr double v = -2.0;
    constexpr double slope_x = 0.75;
    constexpr double slope_y = -0.25;
    constexpr double rho = 1.15;
    const auto flux = surface_fluxes::compute_surface_fluxes(
        config, surface, u, v,
        300.0, 0.008, 300.0, 0.008, 1.0e5, rho, 10.0,
        slope_x, slope_y);

    const double ground_w = slope_x * u + slope_y * v;
    const double tangent_speed = std::sqrt(
        u * u + v * v + ground_w * ground_w);
    const double area_scale = std::sqrt(
        1.0 + slope_x * slope_x + slope_y * slope_y);
    const double generalized_work = flux.tau_u * u + flux.tau_v * v;
    REQUIRE(generalized_work == Approx(
        -area_scale * rho * flux.Cd
            * tangent_speed * tangent_speed * tangent_speed)
        .epsilon(2.0e-13));
    REQUIRE(generalized_work < 0.0);
    REQUIRE(flux.ustar == Approx(
        std::sqrt(flux.Cd) * tangent_speed).epsilon(2.0e-13));
}

TEST_CASE("Surface fluxes: u* stays finite and positive in calm conditions",
          "[physics][boundary_layer][surface]")
{
    const BoundaryLayerConfig bl;   // min_ustar = 1e-4
    const SurfaceConfig sfc;
    const auto f = surface_fluxes::compute_surface_fluxes(
        bl, sfc, 0.0, 0.0, 300.0, 0.008, 300.0, 0.008, 1.0e5, 1.15, 10.0);

    REQUIRE(std::isfinite(f.ustar));
    REQUIRE(f.ustar > 0.0);
    REQUIRE(f.ustar >= bl.min_ustar);
}

TEST_CASE("Surface fluxes reject invalid state instead of substituting",
          "[physics][boundary_layer][surface][config-truth]")
{
    BoundaryLayerConfig bl;
    SurfaceConfig sfc;

    SECTION("unknown surface layer")
    {
        bl.surface_layer_id = "invented";
        REQUIRE_THROWS_AS(
            surface_fluxes::compute_surface_fluxes(
                bl, sfc, 5.0, 0.0, 302.0, 0.01, 300.0, 0.008,
                1.0e5, 1.15, 10.0),
            std::invalid_argument);
    }
    SECTION("non-finite wind")
    {
        REQUIRE_THROWS_AS(
            surface_fluxes::compute_surface_fluxes(
                bl, sfc, std::numeric_limits<double>::quiet_NaN(), 0.0,
                302.0, 0.01, 300.0, 0.008, 1.0e5, 1.15, 10.0),
            std::invalid_argument);
    }
    SECTION("invalid density")
    {
        REQUIRE_THROWS_AS(
            surface_fluxes::compute_surface_fluxes(
                bl, sfc, 5.0, 0.0, 302.0, 0.01, 300.0, 0.008,
                1.0e5, 0.0, 10.0),
            std::invalid_argument);
    }
}

TEST_CASE("Surface fluxes: u* increases monotonically with wind (neutral)",
          "[physics][boundary_layer][surface]")
{
    const BoundaryLayerConfig bl;
    const SurfaceConfig sfc;
    auto ustar_at = [&](double wind) {
        // neutral: theta_sfc == theta_air, so stability does not confound the trend
        return surface_fluxes::compute_surface_fluxes(
                   bl, sfc, wind, 0.0, 300.0, 0.008, 300.0, 0.008, 1.0e5, 1.15, 10.0)
            .ustar;
    };
    REQUIRE(ustar_at(2.0) < ustar_at(6.0));
    REQUIRE(ustar_at(6.0) < ustar_at(12.0));
}

// ─────────────────────────────────────────────────────────────
// Surface-layer stability: sign convention + Louis boundedness + MO length
// (regressions for the bulk-Ri sign flip, the unbounded stability factor, and
//  the Monin-Obukhov length sign).
// ─────────────────────────────────────────────────────────────

TEST_CASE("bulk Richardson number sign: warm surface < 0 (unstable), cold > 0 (stable)",
          "[physics][boundary_layer][surface]")
{
    // theta_v(surface) > theta_v(air): a warm/unstable surface -> Ri_b < 0.
    REQUIRE(surface_fluxes::bulk_richardson_number(305.0, 300.0, 5.0, 0.0, 10.0) < 0.0);
    // theta_v(surface) < theta_v(air): a cold/stable surface -> Ri_b > 0.
    REQUIRE(surface_fluxes::bulk_richardson_number(295.0, 300.0, 5.0, 0.0, 10.0) > 0.0);
}

TEST_CASE("Surface fluxes: unstable ENHANCES, stable SUPPRESSES the heat coefficient",
          "[physics][boundary_layer][surface]")
{
    const BoundaryLayerConfig bl;   // bulk
    const SurfaceConfig sfc;
    auto Ch_at = [&](double theta_sfc) {
        return surface_fluxes::compute_surface_fluxes(
                   bl, sfc, 5.0, 0.0, theta_sfc, 0.008, 300.0, 0.008, 1.0e5, 1.15, 10.0)
            .Ch;
    };
    const double Ch_neutral  = Ch_at(300.0);   // theta_sfc == theta_air
    REQUIRE(Ch_at(305.0) > Ch_neutral);         // warm/unstable -> convective enhancement
    REQUIRE(Ch_at(295.0) < Ch_neutral);         // cold/stable   -> suppression
}

TEST_CASE("Surface fluxes: a strongly unstable surface stays bounded (Louis, no runaway)",
          "[physics][boundary_layer][surface]")
{
    const BoundaryLayerConfig bl;
    const SurfaceConfig sfc;
    auto f = [&](double theta_sfc, double z_sfc) {
        return surface_fluxes::compute_surface_fluxes(
            bl, sfc, 5.0, 0.0, theta_sfc, 0.008, 300.0, 0.008, 1.0e5, 1.15, z_sfc);
    };
    const double Ch_neutral = f(300.0, 200.0).Ch;
    const auto strong = f(340.0, 200.0);        // a 40 K skin-air difference at 200 m
    REQUIRE(std::isfinite(strong.Ch));
    REQUIRE(std::isfinite(strong.H));
    REQUIRE(strong.H > 0.0);                     // still upward
    // Louis caps the enhancement at the sqrt(|Ri|) rate; the old linear
    // 1 - 10*Ri_b would exceed 100x neutral here.
    REQUIRE(strong.Ch < 10.0 * Ch_neutral);
}

TEST_CASE("Monin-Obukhov length: warm surface L<0 (unstable), cold L>0 (stable)",
          "[physics][boundary_layer][surface]")
{
    const SurfaceConfig sfc;
    const auto warm = surface_fluxes::compute_monin_obukhov_fluxes(
        sfc, 5.0, 0.0, /*theta_sfc=*/305.0, 0.010, /*theta_air=*/300.0, 0.008,
        1.0e5, 1.15, 10.0, /*min_ustar=*/1.0e-4);
    REQUIRE(std::isfinite(warm.L));
    REQUIRE(warm.L < 0.0);                       // unstable
    REQUIRE(warm.zeta < 0.0);
    REQUIRE(warm.H > 0.0);

    const auto cold = surface_fluxes::compute_monin_obukhov_fluxes(
        sfc, 5.0, 0.0, 295.0, 0.008, 300.0, 0.008, 1.0e5, 1.15, 10.0, 1.0e-4);
    REQUIRE(cold.L > 0.0);                        // stable
    REQUIRE(cold.zeta > 0.0);
}

// ─────────────────────────────────────────────────────────────
// Shared conservative vertical-diffusion operator (vdiff::diffuse_column):
// the single surface-flux lower BC + conservative flux divergence now used by
// all PBL schemes (S4/S5).
// ─────────────────────────────────────────────────────────────

TEST_CASE("vdiff: a well-mixed column with no surface flux has zero tendency",
          "[physics][boundary_layer][vdiff]")
{
    const size_t nz = 8;
    std::vector<double> profile(nz, 300.0);              // uniform
    std::vector<double> K(nz + 1, 10.0);
    std::vector<double> z_int(nz + 1);
    for (size_t i = 0; i <= nz; ++i) z_int[i] = 100.0 * static_cast<double>(i);
    std::vector<double> rho(nz, 1.1);
    std::vector<double> tend;
    boundary_layer::vdiff::diffuse_column(profile, K, z_int, rho, /*sfc=*/0.0, tend);
    REQUIRE(tend.size() == nz);
    for (double t : tend) REQUIRE(std::abs(t) < 1.0e-12);
}

TEST_CASE("vdiff: internal mixing conserves the column mass integral",
          "[physics][boundary_layer][vdiff]")
{
    const size_t nz = 10;
    std::vector<double> profile(nz);
    for (size_t k = 0; k < nz; ++k) profile[k] = 300.0 + 0.5 * static_cast<double>(k);
    std::vector<double> K(nz + 1, 8.0);
    std::vector<double> z_int(nz + 1);
    for (size_t i = 0; i <= nz; ++i) z_int[i] = 80.0 * static_cast<double>(i);
    std::vector<double> rho(nz);
    for (size_t k = 0; k < nz; ++k) rho[k] = 1.2 - 0.02 * static_cast<double>(k);
    std::vector<double> tend;
    boundary_layer::vdiff::diffuse_column(profile, K, z_int, rho, /*sfc=*/0.0, tend);

    double integral = 0.0;
    for (size_t k = 0; k < nz; ++k)
        integral += rho[k] * tend[k] * (z_int[k + 1] - z_int[k]);
    REQUIRE(std::abs(integral) < 1.0e-9);               // no net source with zero surface flux
}

TEST_CASE("vdiff: a positive surface flux warms the lowest cell and adds exactly that mass",
          "[physics][boundary_layer][vdiff]")
{
    const size_t nz = 10;
    std::vector<double> profile(nz, 300.0);             // uniform -> only the surface flux acts
    std::vector<double> K(nz + 1, 5.0);
    std::vector<double> z_int(nz + 1);
    for (size_t i = 0; i <= nz; ++i) z_int[i] = 50.0 * static_cast<double>(i);
    std::vector<double> rho(nz, 1.1);
    const double sfc = 0.05;                             // upward surface mass flux of the scalar
    std::vector<double> tend;
    boundary_layer::vdiff::diffuse_column(profile, K, z_int, rho, sfc, tend);

    REQUIRE(tend[0] > 0.0);                              // surface flux warms the lowest cell
    double integral = 0.0;
    for (size_t k = 0; k < nz; ++k)
        integral += rho[k] * tend[k] * (z_int[k + 1] - z_int[k]);
    REQUIRE(integral == Approx(sfc).margin(1.0e-9));     // net mass added == surface flux
}

TEST_CASE("vdiff rejects malformed physical inputs instead of masking them",
          "[physics][boundary_layer][vdiff][integrity]")
{
    std::vector<double> profile(3, 300.0);
    std::vector<double> K(4, 5.0);
    std::vector<double> z_int{0.0, 100.0, 200.0, 300.0};
    std::vector<double> rho(3, 1.1);
    std::vector<double> tend;

    SECTION("negative diffusivity")
    {
        K[2] = -1.0;
        REQUIRE_THROWS_AS(
            boundary_layer::vdiff::diffuse_column(
                profile, K, z_int, rho, 0.0, tend),
            std::invalid_argument);
    }
    SECTION("non-increasing geometry")
    {
        z_int[2] = z_int[1];
        REQUIRE_THROWS_AS(
            boundary_layer::vdiff::diffuse_column(
                profile, K, z_int, rho, 0.0, tend),
            std::invalid_argument);
    }
    SECTION("non-positive density")
    {
        rho[1] = 0.0;
        REQUIRE_THROWS_AS(
            boundary_layer::vdiff::diffuse_column(
                profile, K, z_int, rho, 0.0, tend),
            std::invalid_argument);
    }
    SECTION("non-finite surface flux")
    {
        REQUIRE_THROWS_AS(
            boundary_layer::vdiff::diffuse_column(
                profile, K, z_int, rho,
                std::numeric_limits<double>::quiet_NaN(), tend),
            std::invalid_argument);
    }
}

// ─────────────────────────────────────────────────────────────
// Shared bulk-Richardson PBL-height routine (pbl::diagnose_pbl_height_ri) (S7).
// ─────────────────────────────────────────────────────────────

TEST_CASE("pbl height: a capping inversion sets the PBL top above the mixed layer",
          "[physics][boundary_layer][pblheight]")
{
    const size_t nz = 16;
    std::vector<double> z_int(nz + 1);
    for (size_t i = 0; i <= nz; ++i) z_int[i] = 250.0 * static_cast<double>(i);
    std::vector<double> theta(nz), qv(nz, 0.01), u(nz), v(nz, 0.0);
    for (size_t k = 0; k < nz; ++k)
    {
        const double zc = 250.0 * static_cast<double>(k) + 125.0;
        theta[k] = (zc < 1000.0) ? 300.0 : 300.0 + 0.01 * (zc - 1000.0); // mixed, then inversion
        u[k] = 0.002 * zc;                                               // gentle shear
    }
    const double h = boundary_layer::pbl::diagnose_pbl_height_ri(
        theta, qv, u, v, z_int, /*Ri_crit=*/0.25, /*min_h=*/100.0, /*max_h=*/3000.0);
    REQUIRE(h > 500.0);
    REQUIRE(h < 1600.0);          // near the ~1 km inversion base, not pinned to the cap
}

TEST_CASE("pbl height: a neutral column returns the cap (no Ri crossing)",
          "[physics][boundary_layer][pblheight]")
{
    const size_t nz = 16;
    std::vector<double> z_int(nz + 1);
    for (size_t i = 0; i <= nz; ++i) z_int[i] = 250.0 * static_cast<double>(i);
    std::vector<double> theta(nz, 300.0), qv(nz, 0.01), u(nz), v(nz, 0.0);
    for (size_t k = 0; k < nz; ++k) u[k] = 0.002 * (250.0 * static_cast<double>(k) + 125.0);
    const double h = boundary_layer::pbl::diagnose_pbl_height_ri(
        theta, qv, u, v, z_int, 0.25, 100.0, 3000.0);
    REQUIRE(h == Approx(3000.0));
}

TEST_CASE("potential-temperature PBL height interpolates the 1.5 K crossing",
          "[physics][boundary_layer][pblheight][theta]")
{
    const std::vector<double> z_int{0.0, 200.0, 400.0, 600.0, 800.0};
    const std::vector<double> theta{300.0, 300.0, 301.0, 302.0};
    const double height = boundary_layer::pbl::diagnose_pbl_height_theta(
        theta, z_int, boundary_layer::pbl::theta_increase_threshold_k,
        100.0, 1000.0);
    REQUIRE(height == Approx(600.0));

    const std::vector<double> neutral(theta.size(), 300.0);
    REQUIRE(boundary_layer::pbl::diagnose_pbl_height_theta(
                neutral, z_int,
                boundary_layer::pbl::theta_increase_threshold_k,
                100.0, 700.0) == Approx(700.0));
}

TEST_CASE("TKE PBL height interpolates the turbulent-energy threshold",
          "[physics][boundary_layer][pblheight][tke]")
{
    const std::vector<double> z_int{0.0, 200.0, 400.0, 600.0, 800.0};
    const std::vector<double> tke{0.20, 0.10, 0.02, 0.0};
    const double height = boundary_layer::pbl::diagnose_pbl_height_tke(
        tke, z_int, boundary_layer::pbl::tke_diagnostic_threshold_m2s2(tke),
        100.0, 1000.0);
    REQUIRE(height == Approx(600.0));

    const std::vector<double> calm(tke.size(), 0.0);
    REQUIRE(boundary_layer::pbl::diagnose_pbl_height_tke(
                calm, z_int,
                boundary_layer::pbl::tke_diagnostic_threshold_m2s2(calm),
                100.0, 1000.0) == Approx(100.0));

    const std::vector<double> energetic{1.0, 0.2, 0.04, 0.0};
    REQUIRE(boundary_layer::pbl::tke_diagnostic_threshold_m2s2(energetic)
            == Approx(0.05));
    REQUIRE(boundary_layer::pbl::diagnose_pbl_height_tke(
                energetic, z_int,
                boundary_layer::pbl::tke_diagnostic_threshold_m2s2(energetic),
                100.0, 1000.0) == Approx(487.5));
}

// ─────────────────────────────────────────────────────────────
// YSU eddy-diffusivity profile shape (S6): parabolic K = kappa w_s z (1-z/h)^2,
// not the old constant clamped K with a degenerate (NaN) surface value.
// ─────────────────────────────────────────────────────────────

TEST_CASE("YSU K-profile is parabolic: zero at surface and PBL top, peak inside",
          "[physics][boundary_layer][ysu]")
{
    auto s = create_boundary_layer_scheme("ysu");
    BoundaryLayerConfig cfg;
    cfg.pbl_max_height = 3000.0;
    cfg.apply_surface_fluxes = true;
    SurfaceConfig sfc;
    s->initialize(cfg);

    // Deep well-mixed layer (uniform theta up to ~1 km) capped by an inversion,
    // so the diagnosed PBL top is ~1 km and the profile spans several levels.
    const int nz = 20;
    TestColumn c(nz, /*u0=*/5.0, /*theta_air=*/300.0);
    for (int k = 0; k < nz; ++k)
    {
        const double zc = 0.5 * (c.z_int[k] + c.z_int[k + 1]);
        c.theta[k] = (zc < 1000.0) ? 300.0 : 300.0 + 0.02 * (zc - 1000.0);
    }
    BoundaryLayerTendencies tend;
    BoundaryLayerDiagnostics diag;
    s->compute_column(cfg, sfc, c.view(5.0, 0.0, /*theta_sfc=*/305.0, 0.012), tend, &diag);

    const auto& K = diag.K_m;
    REQUIRE(K.size() == static_cast<size_t>(nz + 1));
    REQUIRE(all_finite(K));                 // no NaN (old K[1] = 0/0 degeneracy)
    REQUIRE(K.front() < 1.0);               // ~0 at the surface interface (z=0)
    const auto it = std::max_element(K.begin(), K.end());
    const size_t kpeak = static_cast<size_t>(it - K.begin());
    REQUIRE(*it > 5.0);                      // substantial mixing (not degenerate)
    REQUIRE(kpeak > 0);                      // peak is interior, not at the surface
    REQUIRE(kpeak < K.size() - 1);           // ... and not at the very top
    REQUIRE(K[kpeak] > K[10]);               // declines toward the ~1 km PBL top
}

// ─────────────────────────────────────────────────────────────
// compute_column per scheme
// ─────────────────────────────────────────────────────────────

TEST_CASE("BL none scheme is a true no-op", "[physics][boundary_layer]")
{
    auto s = create_boundary_layer_scheme("none");
    BoundaryLayerConfig cfg;
    SurfaceConfig sfc;
    s->initialize(cfg);

    const int nz = 12;
    TestColumn c(nz, /*u0=*/5.0, /*theta_air=*/300.0);
    BoundaryLayerTendencies tend;
    BoundaryLayerDiagnostics diag;
    s->compute_column(cfg, sfc, c.view(5.0, 0.0, 305.0, 0.014), tend, &diag);

    REQUIRE(tend.dthetadt_pbl.size() == static_cast<size_t>(nz));
    for (double x : tend.dudt_pbl)     REQUIRE(x == 0.0);
    for (double x : tend.dvdt_pbl)     REQUIRE(x == 0.0);
    for (double x : tend.dthetadt_pbl) REQUIRE(x == 0.0);
    for (double x : tend.dqvdt_pbl)    REQUIRE(x == 0.0);
    REQUIRE(diag.ustar == 0.0);
    REQUIRE(diag.H == 0.0);
    REQUIRE(diag.pbl_height == 0.0);
}

TEST_CASE("BL slab scheme: finite tendencies, sane fluxes, drag opposes wind",
          "[physics][boundary_layer]")
{
    auto s = create_boundary_layer_scheme("slab");
    BoundaryLayerConfig cfg;        // apply_surface_fluxes = true
    SurfaceConfig sfc;
    s->initialize(cfg);

    const int nz = 12;
    TestColumn c(nz, /*u0=*/6.0, /*theta_air=*/300.0);
    SlabColumnState slab_state;
    BoundaryLayerTendencies tend;
    BoundaryLayerDiagnostics diag;
    s->compute_column(cfg, sfc,
                      c.view(6.0, 0.0, /*theta_sfc=*/303.0, 0.014, &slab_state),
                      tend, &diag);

    REQUIRE(tendencies_finite(tend));
    REQUIRE(tend.dudt_pbl.size() == static_cast<size_t>(nz));
    REQUIRE(diag.ustar > 0.0);
    REQUIRE(diag.H > 0.0);                          // warm surface
    REQUIRE(diag.pbl_height >= 100.0);             // clamped to [min_h, max_h]
    REQUIRE(diag.pbl_height <= cfg.pbl_max_height);
    REQUIRE(tend.dudt_pbl[0] < 0.0);               // drag opposes the +u wind
}

TEST_CASE("BL slab scheme: apply_surface_fluxes=false gates the surface forcing",
          "[physics][boundary_layer]")
{
    auto s = create_boundary_layer_scheme("slab");
    BoundaryLayerConfig cfg;
    cfg.apply_surface_fluxes = false;
    SurfaceConfig sfc;
    s->initialize(cfg);

    const int nz = 12;
    TestColumn c(nz, 6.0, 300.0);
    SlabColumnState slab_state;
    BoundaryLayerTendencies tend;
    BoundaryLayerDiagnostics diag;
    s->compute_column(cfg, sfc,
                      c.view(6.0, 0.0, 303.0, 0.014, &slab_state),
                      tend, &diag);

    REQUIRE(tendencies_finite(tend));
    REQUIRE(diag.H == 0.0);         // no surface sensible-heat flux
    REQUIRE(diag.E == 0.0);         // no surface moisture flux
    REQUIRE(diag.ustar == 0.0);     // no surface stress
}

TEST_CASE("BL slab state is initialized from each resolved column",
          "[physics][boundary_layer][slab][state]")
{
    auto s = create_boundary_layer_scheme("slab");
    BoundaryLayerConfig cfg;
    SurfaceConfig sfc;
    s->initialize(cfg);

    TestColumn warm_dry(/*nz=*/12, /*u0=*/5.0, /*theta_air=*/315.0);
    std::fill(warm_dry.qv.begin(), warm_dry.qv.end(), 0.004);
    SlabColumnState state;
    BoundaryLayerTendencies tend;
    s->compute_column(cfg, sfc,
                      warm_dry.view(5.0, 0.0, 320.0, 0.006, &state,
                                    /*elapsed=*/0.0),
                      tend, nullptr);

    REQUIRE(state.initialized);
    REQUIRE(state.theta_m_k > 315.0);
    REQUIRE(state.qv_m_kgkg == Approx(0.004).margin(1.0e-12));
    REQUIRE(state.h_m >= 100.0);
    // Surface forcing lives only in the returned resolved tendencies; elapsed
    // coupling time must never advance a duplicate hidden reservoir.
    const double theta_after_initialization = state.theta_m_k;
    const double qv_after_initialization = state.qv_m_kgkg;
    s->compute_column(cfg, sfc,
                      warm_dry.view(5.0, 0.0, 320.0, 0.006, &state,
                                    /*elapsed=*/60.0),
                      tend, nullptr);
    REQUIRE(state.theta_m_k == Approx(theta_after_initialization));
    REQUIRE(state.qv_m_kgkg == Approx(qv_after_initialization));
}

TEST_CASE("BL slab resolved tendencies close against every surface flux",
          "[physics][boundary_layer][slab][budget][cpu003]")
{
    auto scheme = create_boundary_layer_scheme("slab");
    BoundaryLayerConfig cfg;
    SurfaceConfig sfc;
    scheme->initialize(cfg);

    TestColumn column(/*nz=*/12, /*u0=*/6.0, /*theta_air=*/300.0);
    std::fill(column.theta.begin(), column.theta.end(), 300.0);
    std::fill(column.qv.begin(), column.qv.end(), 0.008);
    SlabColumnState state;
    BoundaryLayerTendencies tendencies;
    BoundaryLayerDiagnostics diagnostics;

    scheme->compute_column(
        cfg,
        sfc,
        column.view(6.0, 0.0, /*theta_sfc=*/304.0, /*qv_sfc=*/0.014,
                    &state, /*elapsed=*/0.0),
        tendencies,
        &diagnostics);

    REQUIRE(diagnostics.H > 0.0);
    REQUIRE(diagnostics.E > 0.0);
    double resolved_u_force = 0.0;
    double resolved_v_force = 0.0;
    double resolved_theta_flux = 0.0;
    double resolved_vapor_flux = 0.0;
    for (std::size_t k = 0; k < column.theta.size(); ++k)
    {
        const double mass = column.rho[k] *
            (column.z_int[k + 1] - column.z_int[k]);
        resolved_u_force += mass * tendencies.dudt_pbl[k];
        resolved_v_force += mass * tendencies.dvdt_pbl[k];
        resolved_theta_flux += mass * tendencies.dthetadt_pbl[k];
        resolved_vapor_flux += mass * tendencies.dqvdt_pbl[k];
    }
    REQUIRE(resolved_u_force == Approx(diagnostics.tau_u).margin(1.0e-14));
    REQUIRE(resolved_v_force == Approx(diagnostics.tau_v).margin(1.0e-14));
    const double surface_exner = std::pow(
        column.p[0] / physical_constants::reference_pressure_pa,
        physical_constants::gas_constant_dry_air_jkgk /
            physical_constants::specific_heat_cp_jkgk);
    REQUIRE(resolved_theta_flux ==
            Approx(diagnostics.H /
                   (boundary_layer_constants::cp * surface_exner)).margin(1.0e-14));
    REQUIRE(resolved_vapor_flux == Approx(diagnostics.E).margin(1.0e-14));
}

TEST_CASE("BL slab internal mixing conserves nonuniform resolved inventories",
          "[physics][boundary_layer][slab][budget]")
{
    auto scheme = create_boundary_layer_scheme("slab");
    BoundaryLayerConfig cfg;
    cfg.apply_surface_fluxes = false;
    scheme->initialize(cfg);
    TestColumn column(/*nz=*/12, /*u0=*/6.0, /*theta_air=*/300.0);
    column.u[0] = 2.0;
    column.theta[0] = 296.0;
    column.qv[0] = 0.014;
    column.rho[0] = 1.3;
    SlabColumnState state;
    BoundaryLayerTendencies tendencies;
    scheme->compute_column(
        cfg, SurfaceConfig{}, column.view(6.0, 0.0, 300.0, 0.008, &state),
        tendencies, nullptr);

    double u_source = 0.0;
    double theta_source = 0.0;
    double qv_source = 0.0;
    for (std::size_t k = 0; k < column.theta.size(); ++k)
    {
        const double mass = column.rho[k] *
            (column.z_int[k + 1] - column.z_int[k]);
        u_source += mass * tendencies.dudt_pbl[k];
        theta_source += mass * tendencies.dthetadt_pbl[k];
        qv_source += mass * tendencies.dqvdt_pbl[k];
    }
    REQUIRE(u_source == Approx(0.0).margin(1.0e-14));
    REQUIRE(theta_source == Approx(0.0).margin(1.0e-14));
    REQUIRE(qv_source == Approx(0.0).margin(1.0e-14));
}

TEST_CASE("BL column schemes close published surface sources in resolved state",
          "[physics][boundary_layer][budget][surface]")
{
    for (const std::string id : {"ysu", "mynn"})
    {
        auto scheme = create_boundary_layer_scheme(id);
        BoundaryLayerConfig cfg;
        cfg.enable_tke = id == "mynn";
        scheme->initialize(cfg);
        TestColumn column(/*nz=*/12, /*u0=*/6.0, /*theta_air=*/300.0);
        std::fill(column.theta.begin(), column.theta.end(), 300.0);
        std::fill(column.qv.begin(), column.qv.end(), 0.008);
        BoundaryLayerTendencies tendencies;
        BoundaryLayerDiagnostics diagnostics;
        scheme->compute_column(
            cfg, SurfaceConfig{},
            column.view(6.0, 0.0, 304.0, 0.014),
            tendencies, &diagnostics);

        double u_source = 0.0;
        double v_source = 0.0;
        double theta_source = 0.0;
        double vapor_source = 0.0;
        for (std::size_t k = 0; k < column.theta.size(); ++k)
        {
            const double mass = column.rho[k] *
                (column.z_int[k + 1] - column.z_int[k]);
            u_source += mass * tendencies.dudt_pbl[k];
            v_source += mass * tendencies.dvdt_pbl[k];
            theta_source += mass * tendencies.dthetadt_pbl[k];
            vapor_source += mass * tendencies.dqvdt_pbl[k];
        }
        const double exner = std::pow(
            column.p[0] / physical_constants::reference_pressure_pa,
            physical_constants::gas_constant_dry_air_jkgk /
                physical_constants::specific_heat_cp_jkgk);
        INFO("scheme = " << id);
        REQUIRE(u_source == Approx(diagnostics.tau_u).margin(1.0e-12));
        REQUIRE(v_source == Approx(diagnostics.tau_v).margin(1.0e-12));
        REQUIRE(theta_source ==
                Approx(diagnostics.H /
                       (boundary_layer_constants::cp * exner)).margin(1.0e-12));
        REQUIRE(vapor_source == Approx(diagnostics.E).margin(1.0e-12));
    }
}

TEST_CASE("YSU convective scale uses the declared positive column density",
          "[physics][boundary_layer][ysu][density][config-truth]")
{
    auto scheme = create_boundary_layer_scheme("ysu");
    BoundaryLayerConfig cfg;
    scheme->initialize(cfg);
    TestColumn column(/*nz=*/12, /*u0=*/3.0, /*theta_air=*/300.0);
    std::fill(column.theta.begin(), column.theta.end(), 300.0);
    std::fill(column.qv.begin(), column.qv.end(), 0.008);
    std::fill(column.rho.begin(), column.rho.end(), 5.0e-7);

    BoundaryLayerTendencies tendencies;
    BoundaryLayerDiagnostics diagnostics;
    diagnostics.include_diffusivity_profiles = true;
    scheme->compute_column(
        cfg, SurfaceConfig{},
        column.view(3.0, 0.0, 304.0, 0.014),
        tendencies, &diagnostics);

    const double exner = std::pow(
        column.p[0] / physical_constants::reference_pressure_pa,
        physical_constants::gas_constant_dry_air_jkgk /
            physical_constants::specific_heat_cp_jkgk);
    const auto initial_flux = surface_fluxes::compute_surface_fluxes(
        cfg, SurfaceConfig{}, 3.0, 0.0, 304.0, 0.014,
        column.theta[0], column.qv[0], column.p[0], column.rho[0], 10.0);
    const double w_theta_v = initial_flux.H /
        (column.rho[0] * boundary_layer_constants::cp * exner)
        + 0.61 * column.theta[0] * initial_flux.E / column.rho[0];
    const double theta_v = surface_fluxes::virtual_potential_temperature(
        column.theta[0], column.qv[0]);
    const double buoyancy_flux = boundary_layer_constants::g /
        theta_v * w_theta_v;
    const double wstar = buoyancy_flux > 0.0
        ? std::cbrt(buoyancy_flux * diagnostics.pbl_height) : 0.0;
    const double z = column.z_int[1];
    const double mixing_speed = std::cbrt(
        diagnostics.ustar * diagnostics.ustar * diagnostics.ustar
        + 8.0 * boundary_layer_constants::kappa
            * wstar * wstar * wstar * z / diagnostics.pbl_height);
    const double expected_k = z <= diagnostics.pbl_height
        ? boundary_layer_constants::kappa * mixing_speed * z
            * std::pow(1.0 - z / diagnostics.pbl_height, 2)
        : 0.1;

    REQUIRE(diagnostics.K_m[1]
            == Approx(expected_k).epsilon(2.0e-13));
}

TEST_CASE("YSU countergradient heat transport reaches a uniform mixed layer",
          "[physics][boundary_layer][ysu][nonlocal]")
{
    auto scheme = create_boundary_layer_scheme("ysu");
    BoundaryLayerConfig cfg;
    scheme->initialize(cfg);
    TestColumn column(12, 5.0, 300.0);
    std::fill(column.theta.begin(), column.theta.end(), 300.0);
    std::fill(column.qv.begin(), column.qv.end(), 0.008);

    BoundaryLayerTendencies tendencies;
    BoundaryLayerDiagnostics diagnostics;
    scheme->compute_column(
        cfg, SurfaceConfig{}, column.view(5.0, 0.0, 305.0, 0.008),
        tendencies, &diagnostics);

    REQUIRE(diagnostics.H > 0.0);
    REQUIRE(std::abs(tendencies.dthetadt_pbl[3]) > 1.0e-12);
    double column_source = 0.0;
    for (std::size_t k = 0; k < column.theta.size(); ++k)
        column_source += column.rho[k]
            * (column.z_int[k + 1] - column.z_int[k])
            * tendencies.dthetadt_pbl[k];
    const double exner = std::pow(
        column.p[0] / physical_constants::reference_pressure_pa,
        physical_constants::gas_constant_dry_air_jkgk
            / physical_constants::specific_heat_cp_jkgk);
    REQUIRE(column_source == Approx(
        diagnostics.H / (boundary_layer_constants::cp * exner)).margin(1.0e-12));
}

TEST_CASE("YSU entrainment exchanges inversion air without leaking inventory",
          "[physics][boundary_layer][ysu][entrainment]")
{
    auto scheme = create_boundary_layer_scheme("ysu");
    BoundaryLayerConfig cfg;
    cfg.pbl_max_height = 1100.0;
    scheme->initialize(cfg);
    TestColumn column(12, 5.0, 300.0);
    for (std::size_t k = 0; k < column.theta.size(); ++k)
    {
        const bool above_inversion = k >= 5;
        column.theta[k] = above_inversion ? 308.0 : 300.0;
        column.qv[k] = above_inversion ? 0.004 : 0.012;
        column.u[k] = above_inversion ? 8.0 : 5.0;
    }

    BoundaryLayerTendencies tendencies;
    BoundaryLayerDiagnostics diagnostics;
    scheme->compute_column(
        cfg, SurfaceConfig{}, column.view(5.0, 0.0, 304.0, 0.014),
        tendencies, &diagnostics);

    const auto first_above = std::find_if(
        column.z_int.begin() + 1, column.z_int.end() - 1,
        [&](double z) { return z >= diagnostics.pbl_height; });
    REQUIRE(first_above != column.z_int.end() - 1);
    const std::size_t top_face = static_cast<std::size_t>(
        first_above - column.z_int.begin());
    REQUIRE(tendencies.dthetadt_pbl[top_face - 1] > 0.0);
    REQUIRE(tendencies.dthetadt_pbl[top_face] < 0.0);

    double theta_source = 0.0;
    double vapor_source = 0.0;
    for (std::size_t k = 0; k < column.theta.size(); ++k)
    {
        const double mass = column.rho[k]
            * (column.z_int[k + 1] - column.z_int[k]);
        theta_source += mass * tendencies.dthetadt_pbl[k];
        vapor_source += mass * tendencies.dqvdt_pbl[k];
    }
    const double exner = std::pow(
        column.p[0] / physical_constants::reference_pressure_pa,
        physical_constants::gas_constant_dry_air_jkgk
            / physical_constants::specific_heat_cp_jkgk);
    REQUIRE(theta_source == Approx(
        diagnostics.H / (boundary_layer_constants::cp * exner)).margin(1.0e-12));
    REQUIRE(vapor_source == Approx(diagnostics.E).margin(1.0e-12));
}

TEST_CASE("YSU cloud-top cooling follows the declared Lock velocity and K profiles",
          "[physics][boundary_layer][ysu][cloud-top][science]")
{
    auto scheme = create_boundary_layer_scheme("ysu");
    BoundaryLayerConfig clear_config;
    clear_config.apply_surface_fluxes = false;
    clear_config.pbl_top_method = "theta";
    clear_config.pbl_max_height = 1000.0;
    scheme->initialize(clear_config);

    TestColumn column(10, 5.0, 300.0);
    for (std::size_t k = 0; k < column.theta.size(); ++k)
    {
        column.theta[k] = 300.0 + 0.02 * static_cast<double>(k);
        column.qv[k] = 0.012 - 0.0005 * static_cast<double>(k);
        column.u[k] = 4.0 + 0.3 * static_cast<double>(k);
    }
    column.qc[5] = 2.0e-4;
    column.qc[6] = 1.0e-4;
    column.qi[4] = 7.5e-5;
    column.radiative_theta_tendency[6] = -2.0e-3;

    BoundaryLayerTendencies clear_tendencies;
    BoundaryLayerDiagnostics clear_diagnostics;
    scheme->compute_column(
        clear_config, SurfaceConfig{},
        column.view(5.0, 0.0, 300.0, 0.012),
        clear_tendencies, &clear_diagnostics);

    BoundaryLayerConfig cloudy_config = clear_config;
    cloudy_config.clear_sky_only = false;
    BoundaryLayerTendencies cloudy_tendencies;
    BoundaryLayerDiagnostics cloudy_diagnostics;
    scheme->compute_column(
        cloudy_config, SurfaceConfig{},
        column.view(5.0, 0.0, 300.0, 0.012),
        cloudy_tendencies, &cloudy_diagnostics);

    const double cloud_top = 700.0;
    const double cooling_flux = 0.2;
    const double velocity = std::cbrt(
        boundary_layer_constants::g * cloud_top * cooling_flux
        / column.theta[0]);
    const std::size_t face = 3;
    const double z = column.z_int[face];
    const double expected_cloud_k = 0.85 * boundary_layer_constants::kappa
        * velocity * z * (z / cloud_top)
        * std::sqrt(1.0 - z / cloud_top);

    REQUIRE(cloudy_diagnostics.cloud_top_radiative_mixing_active);
    REQUIRE(cloudy_diagnostics.cloud_top_height == Approx(cloud_top));
    REQUIRE(cloudy_diagnostics.cloud_top_cooling_flux_kms
            == Approx(cooling_flux));
    REQUIRE(cloudy_diagnostics.cloud_top_velocity_scale
            == Approx(velocity).epsilon(2.0e-14));
    REQUIRE(cloudy_diagnostics.K_m[face] - clear_diagnostics.K_m[face]
            == Approx(expected_cloud_k).epsilon(2.0e-14));
    REQUIRE(cloudy_diagnostics.K_h[face] - clear_diagnostics.K_h[face]
            == Approx(expected_cloud_k).epsilon(2.0e-14));

    double momentum_source = 0.0;
    double theta_l_source = 0.0;
    double total_water_source = 0.0;
    for (std::size_t k = 0; k < column.theta.size(); ++k)
    {
        const double mass = column.rho[k]
            * (column.z_int[k + 1] - column.z_int[k]);
        momentum_source += mass * cloudy_tendencies.dudt_pbl[k];
        const double exner = std::pow(
            column.p[k] / physical_constants::reference_pressure_pa,
            physical_constants::gas_constant_dry_air_jkgk
                / physical_constants::specific_heat_cp_jkgk);
        const double theta_l_tendency =
            cloudy_tendencies.dthetadt_pbl[k]
            - physical_constants::latent_heat_vaporization_jkg
                / (physical_constants::specific_heat_cp_jkgk * exner)
                * cloudy_tendencies.dqcdt_pbl[k]
            - physical_constants::latent_heat_sublimation_jkg
                / (physical_constants::specific_heat_cp_jkgk * exner)
                * cloudy_tendencies.dqidt_pbl[k];
        theta_l_source += mass * theta_l_tendency;
        total_water_source += mass * (
            cloudy_tendencies.dqvdt_pbl[k]
            + cloudy_tendencies.dqcdt_pbl[k]
            + cloudy_tendencies.dqidt_pbl[k]);
    }
    REQUIRE(momentum_source == Approx(0.0).margin(1.0e-12));
    REQUIRE(theta_l_source == Approx(0.0).margin(1.0e-12));
    REQUIRE(total_water_source == Approx(0.0).margin(1.0e-12));
    REQUIRE(cloudy_tendencies.dthetadt_pbl != clear_tendencies.dthetadt_pbl);
    REQUIRE(cloudy_tendencies.dqvdt_pbl != clear_tendencies.dqvdt_pbl);
    REQUIRE(std::any_of(
        cloudy_tendencies.dqcdt_pbl.begin(),
        cloudy_tendencies.dqcdt_pbl.end(),
        [](double value) { return value != 0.0; }));
    REQUIRE(std::any_of(
        cloudy_tendencies.dqidt_pbl.begin(),
        cloudy_tendencies.dqidt_pbl.end(),
        [](double value) { return value != 0.0; }));

    std::fill(column.radiative_theta_tendency.begin(),
              column.radiative_theta_tendency.end(), 0.0);
    BoundaryLayerTendencies uncooled_tendencies;
    BoundaryLayerDiagnostics uncooled_diagnostics;
    scheme->compute_column(
        cloudy_config, SurfaceConfig{},
        column.view(5.0, 0.0, 300.0, 0.012),
        uncooled_tendencies, &uncooled_diagnostics);
    REQUIRE_FALSE(uncooled_diagnostics.cloud_top_radiative_mixing_active);
    REQUIRE(uncooled_tendencies.dthetadt_pbl
            == clear_tendencies.dthetadt_pbl);
    REQUIRE(uncooled_tendencies.dqvdt_pbl == clear_tendencies.dqvdt_pbl);
    REQUIRE(std::all_of(
        uncooled_tendencies.dqcdt_pbl.begin(),
        uncooled_tendencies.dqcdt_pbl.end(),
        [](double value) { return value == 0.0; }));
    REQUIRE(std::all_of(
        uncooled_tendencies.dqidt_pbl.begin(),
        uncooled_tendencies.dqidt_pbl.end(),
        [](double value) { return value == 0.0; }));
}

TEST_CASE("YSU cloud-top input rejection is failure atomic",
          "[physics][boundary_layer][ysu][cloud-top][transaction]")
{
    auto scheme = create_boundary_layer_scheme("ysu");
    BoundaryLayerConfig config;
    config.clear_sky_only = false;
    scheme->initialize(config);
    TestColumn column(10, 5.0, 300.0);
    column.qc[5] = std::numeric_limits<double>::quiet_NaN();

    BoundaryLayerTendencies tendencies;
    tendencies.dudt_pbl = {11.0};
    tendencies.dvdt_pbl = {12.0};
    tendencies.dthetadt_pbl = {13.0};
    tendencies.dqvdt_pbl = {14.0};
    tendencies.dqcdt_pbl = {15.0};
    tendencies.dqidt_pbl = {16.0};
    tendencies.dtkedt_pbl = {17.0};
    const BoundaryLayerTendencies original = tendencies;
    BoundaryLayerDiagnostics diagnostics;
    diagnostics.pbl_height = 16.0;
    diagnostics.cloud_top_velocity_scale = 17.0;

    REQUIRE_THROWS_AS(
        scheme->compute_column(
            config, SurfaceConfig{},
            column.view(5.0, 0.0, 300.0, 0.012),
            tendencies, &diagnostics),
        std::invalid_argument);
    REQUIRE(tendencies.dudt_pbl == original.dudt_pbl);
    REQUIRE(tendencies.dvdt_pbl == original.dvdt_pbl);
    REQUIRE(tendencies.dthetadt_pbl == original.dthetadt_pbl);
    REQUIRE(tendencies.dqvdt_pbl == original.dqvdt_pbl);
    REQUIRE(tendencies.dqcdt_pbl == original.dqcdt_pbl);
    REQUIRE(tendencies.dqidt_pbl == original.dqidt_pbl);
    REQUIRE(tendencies.dtkedt_pbl == original.dtkedt_pbl);
    REQUIRE(diagnostics.pbl_height == 16.0);
    REQUIRE(diagnostics.cloud_top_velocity_scale == 17.0);
}

TEST_CASE("YSU rejects malformed columns without publishing partial results",
          "[physics][boundary_layer][ysu][transaction]")
{
    auto scheme = create_boundary_layer_scheme("ysu");
    BoundaryLayerConfig cfg;
    scheme->initialize(cfg);
    TestColumn column(12, 5.0, 300.0);
    column.p[9] = std::numeric_limits<double>::quiet_NaN();
    BoundaryLayerTendencies tendencies;
    tendencies.dudt_pbl = {11.0, 12.0};
    tendencies.dvdt_pbl = {13.0};
    tendencies.dthetadt_pbl = {14.0};
    tendencies.dqvdt_pbl = {15.0};
    tendencies.dtkedt_pbl = {16.0};
    const BoundaryLayerTendencies original = tendencies;
    BoundaryLayerDiagnostics diagnostics;
    diagnostics.pbl_height = 17.0;
    diagnostics.K_m = {18.0};

    REQUIRE_THROWS_AS(
        scheme->compute_column(
            cfg, SurfaceConfig{}, column.view(5.0, 0.0, 304.0, 0.014),
            tendencies, &diagnostics),
        std::invalid_argument);
    REQUIRE(tendencies.dudt_pbl == original.dudt_pbl);
    REQUIRE(tendencies.dvdt_pbl == original.dvdt_pbl);
    REQUIRE(tendencies.dthetadt_pbl == original.dthetadt_pbl);
    REQUIRE(tendencies.dqvdt_pbl == original.dqvdt_pbl);
    REQUIRE(tendencies.dtkedt_pbl == original.dtkedt_pbl);
    REQUIRE(diagnostics.pbl_height == 17.0);
    REQUIRE(diagnostics.K_m == std::vector<double>{18.0});
}

TEST_CASE("BL YSU momentum work equals surface work minus shear dissipation",
          "[physics][boundary_layer][ysu][budget][stress-work]")
{
    auto scheme = create_boundary_layer_scheme("ysu");
    BoundaryLayerConfig cfg;
    scheme->initialize(cfg);
    TestColumn column(/*nz=*/12, /*u0=*/0.0, /*theta_air=*/300.0);
    for (std::size_t k = 0; k < column.u.size(); ++k)
    {
        column.u[k] = 4.0 + 0.35 * static_cast<double>(k);
        column.v[k] = -1.0 + 0.12 * static_cast<double>(k);
        column.rho[k] = 1.2 - 0.015 * static_cast<double>(k);
    }
    BoundaryLayerTendencies tendencies;
    BoundaryLayerDiagnostics diagnostics;
    scheme->compute_column(
        cfg, SurfaceConfig{},
        column.view(column.u[0], column.v[0],
                    column.theta[0], column.qv[0]),
        tendencies, &diagnostics);

    for (std::size_t k = 0; k < column.u.size(); ++k)
    {
        column.u[k] += cfg.dt_pbl * tendencies.dudt_pbl[k];
        column.v[k] += cfg.dt_pbl * tendencies.dvdt_pbl[k];
    }
    double tendency_work_wm2 = 0.0;
    for (std::size_t k = 0; k < column.u.size(); ++k)
    {
        const double mass = column.rho[k] *
            (column.z_int[k + 1] - column.z_int[k]);
        tendency_work_wm2 += mass *
            (column.u[k] * tendencies.dudt_pbl[k] +
             column.v[k] * tendencies.dvdt_pbl[k]);
    }
    double shear_dissipation_wm2 = 0.0;
    for (std::size_t face = 1; face < column.u.size(); ++face)
    {
        const double rho_face = 0.5 *
            (column.rho[face - 1] + column.rho[face]);
        const double dz_center = 0.5 *
            (column.z_int[face + 1] - column.z_int[face - 1]);
        const double du_dz =
            (column.u[face] - column.u[face - 1]) / dz_center;
        const double dv_dz =
            (column.v[face] - column.v[face - 1]) / dz_center;
        shear_dissipation_wm2 += rho_face * diagnostics.K_m[face] *
            (du_dz * du_dz + dv_dz * dv_dz) * dz_center;
    }
    const double surface_work_wm2 =
        diagnostics.tau_u * column.u[0] +
        diagnostics.tau_v * column.v[0];
    REQUIRE(tendency_work_wm2 ==
            Approx(surface_work_wm2 - shear_dissipation_wm2).margin(1.0e-12));
    REQUIRE(shear_dissipation_wm2 >= 0.0);
    REQUIRE(tendency_work_wm2 < 0.0);
}

TEST_CASE("BL slab homogenization cannot create resolved kinetic energy",
          "[physics][boundary_layer][slab][budget][stress-work]")
{
    auto scheme = create_boundary_layer_scheme("slab");
    BoundaryLayerConfig cfg;
    cfg.apply_surface_fluxes = false;
    scheme->initialize(cfg);
    TestColumn column(/*nz=*/12, /*u0=*/5.0, /*theta_air=*/300.0);
    column.u[0] = 1.0;
    column.v[0] = 3.0;
    column.rho[0] = 1.3;
    SlabColumnState state;
    BoundaryLayerTendencies tendencies;
    scheme->compute_column(
        cfg, SurfaceConfig{},
        column.view(column.u[0], column.v[0],
                    column.theta[0], column.qv[0], &state),
        tendencies, nullptr);

    double kinetic_power_wm2 = 0.0;
    for (std::size_t k = 0; k < column.u.size(); ++k)
    {
        const double mass = column.rho[k] *
            (column.z_int[k + 1] - column.z_int[k]);
        kinetic_power_wm2 += mass *
            (column.u[k] * tendencies.dudt_pbl[k] +
             column.v[k] * tendencies.dvdt_pbl[k]);
    }
    REQUIRE(kinetic_power_wm2 <= 0.0);
    REQUIRE(kinetic_power_wm2 < -1.0e-6);
}

TEST_CASE("BL slab column results are independent of horizontal traversal order",
          "[physics][boundary_layer][slab][state]")
{
    BoundaryLayerConfig cfg;
    SurfaceConfig sfc;
    TestColumn column_a(/*nz=*/12, /*u0=*/4.0, /*theta_air=*/298.0);
    TestColumn column_b(/*nz=*/12, /*u0=*/9.0, /*theta_air=*/308.0);
    std::fill(column_a.qv.begin(), column_a.qv.end(), 0.012);
    std::fill(column_b.qv.begin(), column_b.qv.end(), 0.005);

    auto run = [&](bool reverse) {
        auto scheme = create_boundary_layer_scheme("slab");
        scheme->initialize(cfg);
        std::array<SlabColumnState, 2> states{};
        BoundaryLayerTendencies tend;
        auto advance = [&](int id, double elapsed) {
            const TestColumn& c = id == 0 ? column_a : column_b;
            const double theta_skin = id == 0 ? 302.0 : 312.0;
            const double qv_skin = id == 0 ? 0.014 : 0.007;
            scheme->compute_column(
                cfg, sfc,
                c.view(c.u[0], 0.0, theta_skin, qv_skin,
                       &states[static_cast<std::size_t>(id)], elapsed),
                tend, nullptr);
        };
        for (double elapsed : {0.0, 60.0})
        {
            if (reverse) { advance(1, elapsed); advance(0, elapsed); }
            else         { advance(0, elapsed); advance(1, elapsed); }
        }
        return states;
    };

    const auto forward = run(false);
    const auto reverse = run(true);
    for (std::size_t n = 0; n < forward.size(); ++n)
    {
        CAPTURE(n);
        REQUIRE(forward[n].h_m == Approx(reverse[n].h_m));
        REQUIRE(forward[n].theta_m_k == Approx(reverse[n].theta_m_k));
        REQUIRE(forward[n].qv_m_kgkg == Approx(reverse[n].qv_m_kgkg));
    }
    REQUIRE(forward[0].theta_m_k != Approx(forward[1].theta_m_k));
    REQUIRE(forward[0].qv_m_kgkg != Approx(forward[1].qv_m_kgkg));
}

TEST_CASE("BL YSU and MYNN schemes produce finite tendencies with active surface",
          "[physics][boundary_layer]")
{
    for (const std::string id : {"ysu", "mynn"})
    {
        auto s = create_boundary_layer_scheme(id);
        REQUIRE(s != nullptr);
        BoundaryLayerConfig cfg;
        SurfaceConfig sfc;
        s->initialize(cfg);

        const int nz = 16;
        TestColumn c(nz, /*u0=*/8.0, /*theta_air=*/300.0);
        BoundaryLayerTendencies tend;
        BoundaryLayerDiagnostics diag;
        s->compute_column(cfg, sfc, c.view(8.0, 0.0, /*theta_sfc=*/304.0, 0.015),
                          tend, &diag);

        INFO("scheme = " << id);
        REQUIRE(tendencies_finite(tend));
        REQUIRE(tend.dudt_pbl.size() == static_cast<size_t>(nz));
        REQUIRE(tend.dthetadt_pbl.size() == static_cast<size_t>(nz));
        REQUIRE(std::isfinite(diag.ustar));
        REQUIRE(diag.ustar > 0.0);     // fluxes active -> nonzero friction velocity
    }
}

TEST_CASE("BL MYNN reuses caller-owned column scratch and can omit profiles",
          "[physics][boundary_layer][workspace]")
{
    auto scheme = create_boundary_layer_scheme("mynn");
    BoundaryLayerConfig cfg;
    cfg.enable_tke = true;
    scheme->initialize(cfg);
    TestColumn column(/*nz=*/16, /*u0=*/8.0, /*theta_air=*/300.0);
    BoundaryLayerColumnWorkspace workspace;
    BoundaryLayerTendencies tendencies;
    BoundaryLayerDiagnostics diagnostics;
    diagnostics.include_diffusivity_profiles = false;
    auto view = column.view(8.0, 0.0, 304.0, 0.015);
    view.workspace = &workspace;

    scheme->compute_column(cfg, SurfaceConfig{}, view, tendencies, &diagnostics);
    const double* tke_storage = workspace.tke.data();
    const double* mixing_storage = workspace.mixing_length.data();
    const double* km_storage = workspace.K_m.data();
    const double* kh_storage = workspace.K_h.data();
    const double* ke_storage = workspace.K_e.data();

    scheme->compute_column(cfg, SurfaceConfig{}, view, tendencies, &diagnostics);
    REQUIRE(workspace.tke.data() == tke_storage);
    REQUIRE(workspace.mixing_length.data() == mixing_storage);
    REQUIRE(workspace.K_m.data() == km_storage);
    REQUIRE(workspace.K_h.data() == kh_storage);
    REQUIRE(workspace.K_e.data() == ke_storage);
    REQUIRE(diagnostics.K_m.empty());
    REQUIRE(diagnostics.K_h.empty());
    REQUIRE(diagnostics.K_e.empty());
    REQUIRE(tendencies_finite(tendencies));
}

TEST_CASE("MYNN rejects malformed columns without publishing partial results",
          "[physics][boundary_layer][mynn][transactional][integrity]")
{
    auto scheme = create_boundary_layer_scheme("mynn");
    BoundaryLayerConfig cfg;
    cfg.enable_tke = true;
    cfg.apply_surface_fluxes = false;
    scheme->initialize(cfg);
    TestColumn column(/*nz=*/12, /*u0=*/8.0, /*theta_air=*/300.0);

    BoundaryLayerTendencies sentinel;
    sentinel.dudt_pbl = {1.0, 2.0};
    sentinel.dvdt_pbl = {3.0};
    sentinel.dthetadt_pbl = {4.0, 5.0, 6.0};
    sentinel.dqvdt_pbl = {7.0};
    sentinel.dtkedt_pbl = {8.0, 9.0};
    BoundaryLayerDiagnostics diagnostic_sentinel;
    diagnostic_sentinel.include_diffusivity_profiles = true;
    diagnostic_sentinel.K_m = {10.0};
    diagnostic_sentinel.K_h = {11.0, 12.0};
    diagnostic_sentinel.K_e = {13.0};
    diagnostic_sentinel.pbl_height = 14.0;
    diagnostic_sentinel.ustar = 15.0;
    diagnostic_sentinel.tau_u = 16.0;
    diagnostic_sentinel.tau_v = 17.0;
    diagnostic_sentinel.H = 18.0;
    diagnostic_sentinel.E = 19.0;
    diagnostic_sentinel.Cd = 20.0;
    diagnostic_sentinel.Ch = 21.0;
    diagnostic_sentinel.Ce = 22.0;

    auto require_unchanged = [&](const BoundaryLayerTendencies& tendencies,
                                 const BoundaryLayerDiagnostics& diagnostics) {
        REQUIRE(tendencies.dudt_pbl == sentinel.dudt_pbl);
        REQUIRE(tendencies.dvdt_pbl == sentinel.dvdt_pbl);
        REQUIRE(tendencies.dthetadt_pbl == sentinel.dthetadt_pbl);
        REQUIRE(tendencies.dqvdt_pbl == sentinel.dqvdt_pbl);
        REQUIRE(tendencies.dtkedt_pbl == sentinel.dtkedt_pbl);
        REQUIRE(diagnostics.K_m == diagnostic_sentinel.K_m);
        REQUIRE(diagnostics.K_h == diagnostic_sentinel.K_h);
        REQUIRE(diagnostics.K_e == diagnostic_sentinel.K_e);
        REQUIRE(diagnostics.pbl_height == diagnostic_sentinel.pbl_height);
        REQUIRE(diagnostics.ustar == diagnostic_sentinel.ustar);
        REQUIRE(diagnostics.tau_u == diagnostic_sentinel.tau_u);
        REQUIRE(diagnostics.tau_v == diagnostic_sentinel.tau_v);
        REQUIRE(diagnostics.H == diagnostic_sentinel.H);
        REQUIRE(diagnostics.E == diagnostic_sentinel.E);
        REQUIRE(diagnostics.Cd == diagnostic_sentinel.Cd);
        REQUIRE(diagnostics.Ch == diagnostic_sentinel.Ch);
        REQUIRE(diagnostics.Ce == diagnostic_sentinel.Ce);
    };
    auto reject = [&](const BoundaryLayerConfig& bad_cfg,
                      const BoundaryLayerColumnStateView& bad_view) {
        BoundaryLayerTendencies tendencies = sentinel;
        BoundaryLayerDiagnostics diagnostics = diagnostic_sentinel;
        REQUIRE_THROWS_AS(
            scheme->compute_column(
                bad_cfg, SurfaceConfig{}, bad_view, tendencies, &diagnostics),
            std::invalid_argument);
        require_unchanged(tendencies, diagnostics);
    };

    auto missing_theta = column.view(8.0, 0.0, 300.0, 0.008);
    missing_theta.theta = nullptr;
    reject(cfg, missing_theta);

    const std::vector<double> short_profile(column.u.size() - 1, 0.0);
    auto bad_shape = column.view(8.0, 0.0, 300.0, 0.008);
    bad_shape.v = &short_profile;
    reject(cfg, bad_shape);

    using ColumnCorruption = void (*)(TestColumn&);
    for (const auto corrupt : std::array<ColumnCorruption, 6>{
             [](TestColumn& value) { value.rho[4] = 0.0; },
             [](TestColumn& value) { value.theta[4] =
                 std::numeric_limits<double>::quiet_NaN(); },
             [](TestColumn& value) { value.qv[4] = -1.0e-6; },
             [](TestColumn& value) { value.tke[4] = -1.0e-6; },
             [](TestColumn& value) { value.z_int[5] = value.z_int[4]; },
             [](TestColumn& value) { value.p[5] = value.p[4] + 1.0; },
         })
    {
        TestColumn malformed = column;
        corrupt(malformed);
        reject(cfg, malformed.view(8.0, 0.0, 300.0, 0.008));
    }

    BoundaryLayerConfig bad_cfg = cfg;
    bad_cfg.dt_pbl = 0.0;
    reject(bad_cfg, column.view(8.0, 0.0, 300.0, 0.008));

    BoundaryLayerTendencies tendencies = sentinel;
    BoundaryLayerDiagnostics diagnostics = diagnostic_sentinel;
    REQUIRE_NOTHROW(scheme->compute_column(
        cfg, SurfaceConfig{}, column.view(8.0, 0.0, 300.0, 0.008),
        tendencies, &diagnostics));
    REQUIRE(tendencies.dudt_pbl.size() == column.u.size());
    REQUIRE(diagnostics.K_m.size() == column.u.size() + 1);
}

TEST_CASE("MYNN TKE production respects wind offsets and moist stratification",
          "[physics][boundary_layer][mynn][tke]")
{
    TestColumn column(3, 0.0, 300.0);
    column.theta.assign(3, 300.0);
    column.qv.assign(3, 0.01);
    BoundaryLayerConfig config;
    config.apply_surface_fluxes = false;
    config.enable_tke = true;
    config.dt_pbl = 1.0;
    config.pbl_top_method = "theta";
    MYNNScheme scheme;
    scheme.initialize(config);
    const auto evaluate = [&] {
        BoundaryLayerTendencies result;
        scheme.compute_column(config, SurfaceConfig{},
            column.view(0.0, 0.0, 300.0, 0.01), result);
        return result;
    };
    const auto baseline = evaluate();
    SECTION("uniform wind cannot create shear without surface exchange")
    {
        column.u.assign(3, 10.0);
        column.v.assign(3, -5.0);
        const auto shifted = evaluate();
        REQUIRE(shifted.dtkedt_pbl == baseline.dtkedt_pbl);
        for (double tendency : shifted.dudt_pbl) REQUIRE(tendency == 0.0);
        for (double tendency : shifted.dvdt_pbl) REQUIRE(tendency == 0.0);
    }
    SECTION("moisture gradients produce signed buoyancy even at constant theta")
    {
        column.qv = {0.015, 0.010, 0.005};
        const auto unstable = evaluate();
        column.qv = {0.005, 0.010, 0.015};
        const auto stable = evaluate();
        for (std::size_t k = 0; k < 3; ++k)
        {
            REQUIRE(unstable.dtkedt_pbl[k] > baseline.dtkedt_pbl[k]);
            REQUIRE(stable.dtkedt_pbl[k] < baseline.dtkedt_pbl[k]);
        }
    }
}

TEST_CASE("MYNN TKE transport uses physical mass and center spacing",
          "[physics][boundary_layer][mynn][tke][conservation]")
{
    TestColumn column(2, 0.0, 300.0);
    column.z_int = {0.0, 100.0, 400.0};
    column.theta.assign(2, 300.0);
    column.qv.assign(2, 0.0);
    column.tke = {0.1, 0.2};
    column.rho = {1.2, 0.8};
    BoundaryLayerConfig config;
    config.apply_surface_fluxes = false;
    config.enable_tke = true;
    config.dt_pbl = 1.0;
    config.pbl_top_method = "theta";
    MYNNScheme scheme;
    scheme.initialize(config);
    BoundaryLayerColumnWorkspace workspace;
    const auto evaluate = [&] {
        auto view = column.view(0.0, 0.0, 300.0, 0.0);
        view.workspace = &workspace;
        BoundaryLayerTendencies result;
        scheme.compute_column(config, SurfaceConfig{}, view, result);
        return result;
    };
    const auto baseline = evaluate();
    const double e0 = column.tke[0] + config.dt_pbl * baseline.dtkedt_pbl[0];
    const double e1 = column.tke[1] + config.dt_pbl * baseline.dtkedt_pbl[1];
    const double transfer = workspace.K_e[1] * (e1 - e0) / 200.0;
    std::array<double, 2> transport{};
    for (std::size_t k = 0; k < 2; ++k)
    {
        const double q2 = 2.0 * (column.tke[k] + config.dt_pbl * baseline.dtkedt_pbl[k]);
        const double dissipation = q2 * std::sqrt(q2)
            / (24.0 * std::max(workspace.mixing_length[k], 1.0));
        transport[k] = baseline.dtkedt_pbl[k] + dissipation;
    }
    REQUIRE(transport[0] == Approx(transfer / 120.0).margin(1.0e-14));
    REQUIRE(transport[1] == Approx(-transfer / 240.0).margin(1.0e-14));
    REQUIRE(120.0 * transport[0] + 240.0 * transport[1] == Approx(0.0).margin(1.0e-13));
    for (double& rho : column.rho) rho *= 2.0;
    const auto scaled = evaluate();
    for (std::size_t k = 0; k < 2; ++k)
        REQUIRE(scaled.dtkedt_pbl[k] == Approx(baseline.dtkedt_pbl[k]).margin(1.0e-14));
}

TEST_CASE("MYNN fine-column mixing preserves water and its surface source",
          "[physics][boundary_layer][mynn][conservation]")
{
    TestColumn column(3, 0.0, 300.0);
    column.z_int = {0.0, 1.0, 3.0, 10.0};
    column.rho = {1.2, 1.0, 0.8};
    column.theta.assign(3, 300.0);
    column.qv = {0.02, 0.0, 0.0};
    column.tke.assign(3, 5.0);
    BoundaryLayerConfig config;
    config.enable_tke = true;
    config.apply_surface_fluxes = false;
    config.dt_pbl = 1.0;
    config.pbl_top_method = "theta";
    MYNNScheme scheme;
    scheme.initialize(config);
    BoundaryLayerTendencies result;
    BoundaryLayerDiagnostics diagnostics;
    scheme.compute_column(config, SurfaceConfig{}, column.view(0, 0, 300, 0.02), result, &diagnostics);
    double change = 0.0;
    for (std::size_t k = 0; k < 3; ++k)
    {
        change += column.rho[k] * (column.z_int[k + 1] - column.z_int[k]) * result.dqvdt_pbl[k];
        const double updated = column.qv[k] + config.dt_pbl * result.dqvdt_pbl[k];
        REQUIRE(updated >= -1.0e-14);
        REQUIRE(updated <= 0.02 + 1.0e-14);
    }
    REQUIRE(change == Approx(0.0).margin(1.0e-14));
    REQUIRE(diagnostics.E == 0.0);

    config.apply_surface_fluxes = true;
    config.dt_pbl = 60.0;
    TestColumn forced(6, 5.0, 300.0);
    scheme.initialize(config);
    scheme.compute_column(config, SurfaceConfig{}, forced.view(5, 0, 305, 0.015), result, &diagnostics);
    std::array<double, 4> rates{};
    for (std::size_t k = 0; k < forced.u.size(); ++k)
    {
        const double mass = forced.rho[k] * (forced.z_int[k + 1] - forced.z_int[k]);
        rates[0] += mass * result.dudt_pbl[k];
        rates[1] += mass * result.dvdt_pbl[k];
        rates[2] += mass * result.dthetadt_pbl[k];
        rates[3] += mass * result.dqvdt_pbl[k];
    }
    const double exner = std::pow(forced.p[0] / 100000.0, 287.0 / 1004.0);
    REQUIRE(rates[0] == Approx(diagnostics.tau_u).margin(1.0e-12));
    REQUIRE(rates[1] == Approx(diagnostics.tau_v).margin(1.0e-12));
    REQUIRE(rates[2] == Approx(diagnostics.H / (1004.0 * exner)).margin(1.0e-12));
    REQUIRE(rates[3] == Approx(diagnostics.E).margin(1.0e-12));
}

TEST_CASE("implicit column mixing matches the two-reservoir backward Euler solution",
          "[physics][boundary_layer][conservation]")
{
    const std::vector<double> profile{0.02, 0.0}, K{0, 3, 0}, z{0, 1, 2}, rho{1, 2};
    std::vector<double> tendency;
    boundary_layer::vdiff::diffuse_column_implicit(profile, K, z, rho, 0.0, 1.0, tendency);
    const double difference = 0.02 / (1.0 + 4.5 * 1.5);
    REQUIRE(tendency[0] == Approx(-4.5 * difference).margin(1.0e-14));
    REQUIRE(tendency[1] == Approx(4.5 * difference / 2.0).margin(1.0e-14));
}

TEST_CASE("YSU fine columns conserve water and remain positive over the held cadence",
          "[physics][boundary_layer][ysu][implicit][positivity][budget]")
{
    BoundaryLayerConfig cfg;
    cfg.dt_pbl = 30.0;
    TestColumn column(20, 5.0, 300.0);
    for (std::size_t k = 0; k < column.theta.size(); ++k)
    {
        column.z_int[k] = static_cast<double>(k);
        column.theta[k] = 300.0 + 0.003 * (k + 0.5);
        column.p[k] = 100000.0 * std::exp(-(k + 0.5) / 8000.0);
    }
    column.z_int.back() = 20.0;
    column.qv[3] = 0.016;
    double skin_vapor = 0.008;
    bool inclined = false;
    SECTION("closed diffusion pulse") { cfg.apply_surface_fluxes = false; }
    SECTION("surface vapor sink") { skin_vapor = 0.0; }
    SECTION("inclined surface drag and sink") { skin_vapor = 0.0; inclined = true; }
    SECTION("cloud mixing composes with the surface solve")
    {
        cfg.clear_sky_only = false;
        for (int k = 0; k < 12; ++k)
        {
            column.qc[k] = 0.001;
            column.qi[k] = 0.0001;
        }
        column.radiative_theta_tendency[11] = -0.02;
        skin_vapor = 0.0;
    }
    auto view = column.view(5.0, 0.0, 302.0, skin_vapor);
    view.z_sfc = 0.5;
    view.surface_slope_x = inclined ? 0.3 : 0.0;
    view.surface_slope_y = inclined ? -0.2 : 0.0;
    auto scheme = create_boundary_layer_scheme("ysu");
    scheme->initialize(cfg);
    BoundaryLayerTendencies tendency;
    BoundaryLayerDiagnostics diagnostics;
    scheme->compute_column(cfg, SurfaceConfig{}, view, tendency, &diagnostics);
    double water_source = 0.0;
    double heat_source = 0.0;
    double u_source = 0.0, v_source = 0.0;
    for (std::size_t k = 0; k < column.theta.size(); ++k)
    {
        CAPTURE(k);
        REQUIRE(column.qv[k] + cfg.dt_pbl * tendency.dqvdt_pbl[k] >= -1.0e-14);
        REQUIRE(column.qc[k] + cfg.dt_pbl * tendency.dqcdt_pbl[k] >= -1.0e-14);
        REQUIRE(column.qi[k] + cfg.dt_pbl * tendency.dqidt_pbl[k] >= -1.0e-14);
        const double mass = column.rho[k] * (column.z_int[k + 1] - column.z_int[k]);
        water_source += mass * (tendency.dqvdt_pbl[k]
            + tendency.dqcdt_pbl[k] + tendency.dqidt_pbl[k]);
        heat_source += mass * tendency.dthetadt_pbl[k];
        u_source += mass * tendency.dudt_pbl[k];
        v_source += mass * tendency.dvdt_pbl[k];
    }
    REQUIRE(water_source == Approx(diagnostics.E).margin(1.0e-13));
    REQUIRE(u_source == Approx(diagnostics.tau_u).margin(1.0e-12));
    REQUIRE(v_source == Approx(diagnostics.tau_v).margin(1.0e-12));
    if (cfg.clear_sky_only)
    {
        const double exner = std::pow(column.p[0] / 100000.0, 287.0 / 1004.0);
        REQUIRE(heat_source == Approx(diagnostics.H / (1004.0 * exner)).margin(1.0e-12));
    }
}

TEST_CASE("MYNN implicit TKE remains positive and converges to analytic dissipation",
          "[physics][boundary_layer][mynn][implicit][tke]")
{
    const auto evolve = [](double interval, double initial)
    {
        TestColumn column(2, 0.0, 300.0);
        column.z_int = {0.0, 1.0, 2.0};
        column.theta.assign(2, 300.0);
        column.qv.assign(2, 0.0);
        column.tke.assign(2, initial);
        BoundaryLayerConfig config;
        config.apply_surface_fluxes = false;
        config.dt_pbl = interval;
        MYNNScheme scheme;
        scheme.initialize(config);
        for (int step = 0; step < static_cast<int>(std::lround(30.0 / interval)); ++step)
        {
            BoundaryLayerTendencies result;
            scheme.compute_column(config, SurfaceConfig{}, column.view(0, 0, 300, 0), result);
            for (int k = 0; k < 2; ++k)
            {
                column.tke[k] += interval * result.dtkedt_pbl[k];
                REQUIRE(column.tke[k] >= 0.0);
                REQUIRE(column.tke[k] <= initial);
            }
        }
        return column.tke[0];
    };
    const double exact = 0.1 / std::pow(1.0 + std::sqrt(2.0) / 24.0 * std::sqrt(0.1) * 30.0, 2);
    const double coarse = evolve(30.0, 0.1);
    const double medium = evolve(3.0, 0.1);
    const double fine = evolve(0.3, 0.1);
    REQUIRE(coarse > 0.0);
    REQUIRE(std::abs(fine - exact) < std::abs(medium - exact));
    REQUIRE(std::abs(medium - exact) < std::abs(coarse - exact));
    REQUIRE(fine == Approx(exact).margin(5.0e-4));
    REQUIRE(evolve(30.0, 0.0) == 0.0);
}

TEST_CASE("MYNN held fine-column surface exchange cannot exhaust vapor or TKE",
          "[physics][boundary_layer][mynn][implicit][surface][positivity]")
{
    for (double interval : {1.0, 30.0, 300.0})
    {
        TestColumn column(20, 5.0, 300.0);
        for (int k = 0; k <= 20; ++k) column.z_int[k] = k;
        for (int k = 0; k < 20; ++k)
        {
            column.theta[k] = 300.0 + 0.003 * (k + 0.5);
            column.p[k] = 100000.0 * std::exp(-(k + 0.5) / 8000.0);
        }
        column.qv[3] = 0.016;
        BoundaryLayerConfig config;
        config.dt_pbl = interval;
        MYNNScheme scheme;
        scheme.initialize(config);
        auto view = column.view(5.0, 0.0, 310.0, 0.0);
        view.z_sfc = 0.5;
        view.surface_slope_x = 0.3;
        view.surface_slope_y = -0.2;
        BoundaryLayerTendencies result;
        BoundaryLayerDiagnostics diagnostics;
        scheme.compute_column(config, SurfaceConfig{}, view, result, &diagnostics);
        double water_source = 0.0, u_force = 0.0, v_force = 0.0;
        for (int k = 0; k < 20; ++k)
        {
            CAPTURE(interval, k);
            REQUIRE(column.qv[k] + interval * result.dqvdt_pbl[k] >= 0.0);
            REQUIRE(column.tke[k] + interval * result.dtkedt_pbl[k] >= 0.0);
            water_source += column.rho[k] * result.dqvdt_pbl[k];
            u_force += column.rho[k] * result.dudt_pbl[k];
            v_force += column.rho[k] * result.dvdt_pbl[k];
        }
        REQUIRE(water_source == Approx(diagnostics.E).margin(1.0e-13));
        REQUIRE(u_force == Approx(diagnostics.tau_u).margin(1.0e-12));
        REQUIRE(v_force == Approx(diagnostics.tau_v).margin(1.0e-12));
    }
}

TEST_CASE("Slab mixing integrates fractional cells continuously across centers and interfaces",
          "[physics][boundary_layer][slab][fractional]")
{
    TestColumn column(4, 4, 300);
    column.z_int = {0, 500, 1000, 1500, 2000};
    column.theta = {304, 300, 296, 292};
    column.qv    = {.014, .01, .008, .005};
    column.u     = {4, 10, 12, 14};
    column.rho   = {1.2, .8, .5, .3};
    for (double center : {250.0, 500.0, 750.0, 1000.0})
    {
        std::array<BoundaryLayerTendencies, 2> results;
        for (int side = 0; side < 2; ++side)
        {
            const double        height = center + (side ? .001 : -.001);
            BoundaryLayerConfig cfg;
            cfg.apply_surface_fluxes = false;
            cfg.pbl_top_method       = "theta";
            cfg.pbl_max_height       = height;
            auto scheme              = create_boundary_layer_scheme("slab");
            scheme->initialize(cfg);
            SlabColumnState state;
            scheme->compute_column(cfg, SurfaceConfig{}, column.view(4, 0, 304, .014, &state), results[side]);
            CHECK(state.h_m == Approx(height).margin(1e-10));
            double mass = 0, theta = 0, vapor = 0;
            for (std::size_t k = 0; k < 4; ++k)
            {
                const double overlap = std::max(0.0, std::min(height, column.z_int[k + 1]) - column.z_int[k]);
                const double weight  = column.rho[k] * overlap;
                mass += weight;
                theta += weight * column.theta[k];
                vapor += weight * column.qv[k];
            }
            CHECK(state.theta_m_k == Approx(theta / mass).margin(1e-12));
            CHECK(state.qv_m_kgkg == Approx(vapor / mass).margin(1e-14));
            double theta_change = 0, vapor_change = 0;
            for (std::size_t k = 0; k < 4; ++k)
            {
                const double fraction = std::max(0.0, std::min(height, column.z_int[k + 1]) - column.z_int[k]) / 500;
                CHECK(results[side].dthetadt_pbl[k] ==
                      Approx(fraction * (theta / mass - column.theta[k]) / 300).margin(1e-13));
                CHECK(results[side].dqvdt_pbl[k] ==
                      Approx(fraction * (vapor / mass - column.qv[k]) / 300).margin(1e-15));
                theta_change += column.rho[k] * 500 * results[side].dthetadt_pbl[k];
                vapor_change += column.rho[k] * 500 * results[side].dqvdt_pbl[k];
            }
            CHECK(std::abs(theta_change) < 1e-12);
            CHECK(std::abs(vapor_change) < 1e-14);
        }
        for (std::size_t k = 0; k < 4; ++k)
        {
            CHECK(std::abs(results[0].dthetadt_pbl[k] - results[1].dthetadt_pbl[k]) < 1e-7);
            CHECK(std::abs(results[0].dqvdt_pbl[k] - results[1].dqvdt_pbl[k]) < 1e-10);
        }
    }
}
