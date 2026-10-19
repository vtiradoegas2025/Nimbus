#include "catch2/catch.hpp"
#include "compute/backend_interface.hpp"
#include "compute/acoustic_cgrid_dispatch_contract.hpp"
#include "compute/compute_backend_factory.hpp"
#include "compute/kernel_startup_plan.hpp"
#include "dynamics/acoustic_energy_report.hpp"
#include "core/runtime/simulation_state.hpp"
#include "numerics/advection/advection_cartesian.hpp"
#include <cstdlib>
#include <cstring>
#include <array>
#include <limits>
#include <vector>

namespace
{
struct AcousticFields
{
    static constexpr int n = 6, cells = n * n * n;
    std::vector<float>   u, v, w, rho, p, theta, rho0, p0, gradient;
    AcousticFields()
        : u(cells), v(cells), w(cells), rho(cells), p(cells), theta(cells), rho0(n, 1), p0(n, 100000), gradient(n, 0)
    {
        for (int i = 0; i < cells; ++i)
        {
            u[i]     = .01f * (i % 9);
            v[i]     = -.02f * (i % 7);
            w[i]     = .005f * (i % 5);
            rho[i]   = 1 + .0001f * (i % 13);
            p[i]     = 100000 + .25f * i;
            theta[i] = 300 + .01f * (i % n);
        }
    }
    AcousticPressureCartesianCgridDispatch pressure()
    {
        AcousticPressureCartesianCgridDispatch d;
        d.u      = u.data();
        d.v      = v.data();
        d.w      = w.data();
        d.rho_in = d.rho_out = rho.data();
        d.p_in = d.p_out = p.data();
        d.theta_data     = theta.data();
        d.rho0_base_data = rho0.data();
        d.p0_base_data   = p0.data();
        d.base_len       = n;
        d.nx = d.ny = d.nz = n;
        d.dx = d.dy = d.dz     = 100;
        d.gamma_val            = 1004.0 / 717.0;
        d.dt_small             = .01f;
        d.rho_min              = .05f;
        d.rho_max              = 5;
        d.p_min                = 100;
        d.p_max                = 150000;
        d.p_ref_fallback       = 100000;
        d.repair_primary_state = false;
        return d;
    }
    AcousticMomentumCartesianCgridDispatch momentum()
    {
        AcousticMomentumCartesianCgridDispatch d;
        d.rho_data    = rho.data();
        d.p_data      = p.data();
        d.dp0_dz_data = gradient.data();
        d.base_len    = n;
        d.u_in = d.u_out = u.data();
        d.v_in = d.v_out = v.data();
        d.w_in = d.w_out = w.data();
        d.nx = d.ny = d.nz = n;
        d.dx = d.dy = d.dz = 100;
        d.dt_small         = .01f;
        d.wind_clamp_h = d.wind_clamp_v = 200;
        d.repair_primary_state          = false;
        return d;
    }
    void compare(const AcousticFields& other) const
    {
        CHECK(u == other.u);
        CHECK(v == other.v);
        CHECK(w == other.w);
        CHECK(rho == other.rho);
        CHECK(p == other.p);
    }
};

auto make_backend(bool metal = false, bool full_catalog = false)
{
    using namespace nimbus::compute;
    auto                 backend = metal ? create_metal_compute_backend() : create_vulkan_compute_backend();
    ComputeBackendConfig config;
    config.backend        = metal ? "metal" : "vulkan";
    config.allow_fallback = false;
    config.cuda_assist = full_catalog && std::getenv("NIMBUS_TEST_CUDA_ASSIST") != nullptr;
    std::string error;
    if (!full_catalog) config.startup_plan = KernelStartupPlan::make(
        "acoustic-input-reuse-test",
        {KernelStartupOwner::CartesianExplicitDiffusion},
        {KernelId::Diffusion, KernelId::AcousticPressureCartesianCgrid, KernelId::AcousticMomentumCartesianCgrid},
        kernel_manifest(),
        error);
    if (!full_catalog) REQUIRE(config.startup_plan);
    backend->configure(config);
    backend->set_coordinate_system(CoordinateSystem::Cartesian);
    const bool initialized = backend->initialize(error);
    INFO(error);
    REQUIRE(initialized);
    backend->begin_resident_step();
    return backend;
}
} // namespace

TEST_CASE("parallel acoustic publication checks include invalid tail cells without mutation",
          "[acoustic-publication]")
{
    constexpr std::size_t count = 131071;
    std::array<std::vector<float>, 5> storage;
    std::array<const float*, 5> fields;
    for (std::size_t f = 0; f < storage.size(); ++f)
    {
        storage[f].assign(count, 1.0f);
        fields[f] = storage[f].data();
    }
    REQUIRE(valid_unrepaired_acoustic_state(fields, count));
    for (std::size_t index : {std::size_t(0), std::size_t(65536), count - 1})
    {
        storage[3][index] = -1.0f;
        REQUIRE_FALSE(valid_unrepaired_acoustic_state(fields, count));
        REQUIRE(storage[3][index] == -1.0f);
        storage[3][index] = 1.0f;
        storage[0][index] = std::numeric_limits<float>::quiet_NaN();
        REQUIRE_FALSE(valid_unrepaired_acoustic_state(fields, count));
        REQUIRE(std::isnan(storage[0][index]));
        storage[0][index] = 1.0f;
    }
}

TEST_CASE("external acoustic transport preparation is atomic until publication",
          "[acoustic-transport-commit]")
{
    SimulationStorageState state{{6, 6, 6, 100, 100, .1, 100}};
    state.configuration.stagger_type = StaggerType::CGrid;
    state.grid.geometry.initialize(6, 6, 6, 100, 100, 100, CoordinateSystem::Cartesian, StaggerType::CGrid);
    const auto view = make_simulation_state_view(state);
    tmv::CartesianCGridTransportContext context;
    context.begin(view, 6, 6, 6, .1);
    Field3D density(6, 6, 6, 1.0f);
    {
        auto abandoned = context.prepare_external_acoustic_loop(density, .1, 8);
        REQUIRE_FALSE(context.has_density_begin());
        REQUIRE(context.substep_count() == 0);
        REQUIRE(context.density_begin(2, 2, 2) == 0.0f);
    }
    REQUIRE_FALSE(context.has_density_begin());
    REQUIRE_THROWS(context.prepare_external_acoustic_loop(density, .2, 8));
    REQUIRE_THROWS(context.prepare_external_acoustic_loop(density, .1, 0));
    density(2, 2, 2) = -1.0f;
    REQUIRE_THROWS(context.prepare_external_acoustic_loop(density, .1, 8));
    REQUIRE(context.substep_count() == 0);
    REQUIRE_FALSE(context.has_density_begin());
    density(2, 2, 2) = 1.0f;
    const double executed = static_cast<double>(static_cast<float>(.1 / 8)) * 8;
    auto prepared = context.prepare_external_acoustic_loop(density, executed, 8);
    auto moved = std::move(prepared);
    prepared.commit();
    REQUIRE_FALSE(context.has_density_begin());
    moved.commit();
    moved.commit();
    REQUIRE(context.has_density_begin());
    REQUIRE(context.substep_count() == 8);
    REQUIRE(context.accumulated_dt() == executed);
    REQUIRE(context.density_begin(2, 2, 2) == 1.0f);
    context.complete(view);
    REQUIRE(context.validate(view, 6, 6, 6, .1));
}

TEST_CASE("strict acoustic batch preserves staged native fields wave history and retained flux",
          "[vulkan][strict-acoustic-batch][device]")
{
    using namespace nimbus::compute;
    auto backend = make_backend(false, true);
    auto* kernels = backend->dynamics_kernels();
    AcousticFields expected, actual;
    constexpr int n = AcousticFields::n;
    const int substeps = GENERATE(1, 3, 8);
    const bool periodic = GENERATE(false, true);
    constexpr double dt_large = .1;
    const float dt_small = static_cast<float>(dt_large / substeps);
    std::vector<float> low(n, 2e-8f), wave(4 * n * n), actual_wave(wave.size(), -99.0f);
    for (std::size_t i = 0; i < wave.size(); ++i) wave[i] = 1.25f + .0025f * i;
    auto expected_wave = wave;
    std::array<std::vector<float>, 3> flux;
    for (auto& field : flux) field.resize(AcousticFields::cells, -17.0f);
    auto p = actual.pressure();
    auto m = actual.momentum();
    AcousticSubstepsBatchedCartesianCgridDispatch d;
    d.u = actual.u.data(); d.v = actual.v.data(); d.w = actual.w.data();
    d.rho = actual.rho.data(); d.p = actual.p.data(); d.theta_data = actual.theta.data();
    d.rho0_base_data = actual.rho0.data(); d.p0_base_data = actual.p0.data();
    d.dp0_dz_data = actual.gradient.data(); d.dp0_dz_low_data = low.data();
    d.nx = d.ny = d.nz = d.base_len = n;
    d.dx = p.dx; d.dy = p.dy; d.dz = p.dz; d.dt_small = dt_small; d.n_substeps = substeps;
    d.gamma_val = static_cast<float>(p.gamma_val);
    d.gamma_low = static_cast<float>(p.gamma_val - static_cast<double>(d.gamma_val));
    d.rho_min = p.rho_min; d.rho_max = p.rho_max; d.p_min = p.p_min; d.p_max = p.p_max;
    d.p_ref_fallback = p.p_ref_fallback; d.wind_clamp_h = m.wind_clamp_h; d.wind_clamp_v = m.wind_clamp_v;
    d.bc_west = d.bc_east = d.bc_south = d.bc_north = periodic ? 0 : 1;
    d.bc_bottom = 0; d.bc_top = 2;
    d.repair_primary_state = false; d.accumulate_mass_flux = true;
    d.mass_flux_x = flux[0].data(); d.mass_flux_y = flux[1].data(); d.mass_flux_z = flux[2].data();
    d.mass_flux_weight = static_cast<double>(dt_small) / dt_large;
    d.wave_history = wave.data(); d.wave_history_output = actual_wave.data();
    if (std::getenv("NIMBUS_TEST_REQUIRE_FP64_BATCH"))
        REQUIRE(kernels->supports_unrepaired_acoustic_batched());
    if (!kernels->supports_unrepaired_acoustic_batched())
    {
        REQUIRE_FALSE(kernels->dispatch_acoustic_substeps_batched_cartesian_cgrid(d));
        actual.compare(expected);
        REQUIRE(actual_wave.front() == -99.0f);
        REQUIRE(flux[0].front() == -17.0f);
        return;
    }
    SimulationStorageState state{{n, n, n, 100, 100, .1, 100}};
    state.configuration.stagger_type = StaggerType::CGrid;
    state.grid.geometry.initialize(n, n, n, 100, 100, 100, CoordinateSystem::Cartesian, StaggerType::CGrid);
    auto view = make_simulation_state_view(state);
    tmv::CartesianCGridTransportContext reference_flux;
    reference_flux.begin(view, n, n, n, dt_large);
    std::array<Field3D, 4> inputs;
    for (auto& field : inputs) field.resize(n, n, n, 0.0f);
    for (int step = 0; step < substeps; ++step)
    {
        const std::array<const std::vector<float>*, 4> fields{&expected.rho, &expected.u, &expected.v, &expected.w};
        for (std::size_t field = 0; field < fields.size(); ++field)
            std::copy(fields[field]->begin(), fields[field]->end(), inputs[field].data());
        reference_flux.accumulate(view, inputs[0], inputs[1], inputs[2], inputs[3], static_cast<double>(dt_small),
            nullptr, nullptr, tmv::AcousticMassFluxPrecision::NativeCompensated);
        auto pressure = expected.pressure(); pressure.dt_small = dt_small;
        auto momentum = expected.momentum(); momentum.dt_small = dt_small; momentum.dp0_dz_low_data = low.data();
        REQUIRE(kernels->dispatch_acoustic_pressure_cartesian_cgrid(pressure));
        REQUIRE(kernels->dispatch_acoustic_momentum_cartesian_cgrid(momentum));
        CgridBoundaryDispatch bc;
        bc.fields = {expected.u.data(), expected.v.data(), expected.w.data(), expected.rho.data(), expected.p.data()};
        bc.output = {expected.u.data(), expected.v.data(), expected.w.data(), expected.rho.data(), expected.p.data()};
        bc.wave = expected_wave.data(); bc.wave_output = expected_wave.data();
        bc.constants.nx = bc.constants.ny = bc.constants.nz = n;
        bc.constants.bc_west = bc.constants.bc_east = bc.constants.bc_south = bc.constants.bc_north = periodic ? 0 : 1;
        bc.constants.bc_bottom = d.bc_bottom; bc.constants.bc_top = d.bc_top;
        bc.constants.dt_small = dt_small; bc.constants.inv_dx = 1.0f / d.dx; bc.constants.inv_dy = 1.0f / d.dy;
        REQUIRE(kernels->dispatch_cgrid_boundary(bc));
    }
    backend->begin_resident_step();
    const auto submissions = backend->successful_dispatch_count();
    REQUIRE(kernels->dispatch_acoustic_substeps_batched_cartesian_cgrid(d));
    REQUIRE(backend->successful_dispatch_count() == submissions + 1);
    REQUIRE(backend->successful_kernel_dispatch_count(KernelId::AcousticRetainedFluxFp64) == static_cast<std::uint64_t>(substeps));
    actual.compare(expected);
    REQUIRE(actual_wave == expected_wave);
    const std::array<const Field3D*, 3> expected_flux{&reference_flux.mass_flux_x, &reference_flux.mass_flux_y, &reference_flux.mass_flux_z};
    for (int field = 0; field < 3; ++field)
    {
        CAPTURE(field, substeps, periodic);
        CHECK(std::memcmp(flux[field].data(), expected_flux[field]->data(), flux[field].size() * sizeof(float)) == 0);
    }
    const auto before = actual;
    const auto wave_before = actual_wave;
    const auto flux_before = flux;
    for (int failure = 0; failure < 5; ++failure)
    {
        auto invalid = d;
        switch (failure)
        {
            case 0: invalid.dx = 0; break;
            case 1: invalid.mass_flux_x = invalid.rho; break;
            case 2: invalid.wave_history_output = invalid.u + 1; break;
            case 3: invalid.mass_flux_weight = 0; break;
            case 4: invalid.gamma_low = std::numeric_limits<float>::quiet_NaN(); break;
        }
        REQUIRE_FALSE(kernels->dispatch_acoustic_substeps_batched_cartesian_cgrid(invalid));
        actual.compare(before);
        REQUIRE(actual_wave == wave_before);
        REQUIRE(flux == flux_before);
    }
    if (substeps == 1 && periodic)
    {
        d.dt_small = 1e-6f;
        d.mass_flux_weight = 1.0;
        std::fill(low.begin(), low.end(), 0.0f);
        const std::uint32_t patterns[] = {0u, 0x80000000u, 1u, 0x80000001u, 0x007fffffu, 0x807fffffu,
            0x00800000u, 0x80800000u, 0x3f800001u, 0xbf800001u, 0x7f7fffffu, 0xff7fffffu};
        for (float density : {1.0f, .5f, std::nextafter(1.0f, 0.0f)})
            for (const auto pattern : patterns)
            {
                CAPTURE(pattern, density);
                float value = 0;
                std::memcpy(&value, &pattern, sizeof(value));
                std::fill(actual.u.begin(), actual.u.end(), value);
                std::fill(actual.v.begin(), actual.v.end(), value);
                std::fill(actual.w.begin(), actual.w.end(), 0.0f);
                std::fill(actual.rho.begin(), actual.rho.end(), density);
                std::fill(actual.p.begin(), actual.p.end(), 100000.0f);
                std::fill(actual.theta.begin(), actual.theta.end(), 300.0f);
                REQUIRE(kernels->dispatch_acoustic_substeps_batched_cartesian_cgrid(d));
                const float oracle = static_cast<float>(std::fma(1.0, static_cast<double>(density) * value, 0.0));
                const std::size_t center = (1 * n + 1) * n + 1;
                CHECK(std::memcmp(&flux[0][center], &oracle, sizeof(float)) == 0);
                CHECK(std::memcmp(&flux[1][center], &oracle, sizeof(float)) == 0);
            }
    }
}

TEST_CASE("Scoped acoustic reuse preserves boundaries and changing profiles exactly",
          "[vulkan][acoustic-input-reuse][device]")
{
    const bool retain_pressure = GENERATE(false, true);
    auto  backend = make_backend();
    auto* kernels = backend->dynamics_kernels();
    REQUIRE(kernels);
    const auto run = [&](bool reuse)
    {
        AcousticFields fields;
        for (int loop = 0; loop < 2; ++loop)
        {
            fields.theta[43] += 2;
            fields.gradient[2] += .01f;
            if (reuse)
                kernels->begin_acoustic_loop_residency(true, retain_pressure);
            for (int step = 0; step < 3; ++step)
            {
                REQUIRE(kernels->dispatch_acoustic_pressure_cartesian_cgrid(fields.pressure()));
                const auto before_momentum = backend->resident_reuse_count();
                REQUIRE(kernels->dispatch_acoustic_momentum_cartesian_cgrid(fields.momentum()));
                if (reuse && retain_pressure)
                    CHECK(backend->resident_reuse_count() >= before_momentum + 5);
                fields.u.front() += .75f;
                fields.v[6] -= .2f;
                fields.w.back() += .125f;
            }
            kernels->release_acoustic_loop_residency();
        }
        return fields;
    };
    const auto reference = run(false);
    const auto hits      = backend->resident_reuse_count();
    const auto cached    = run(true);
    cached.compare(reference);
    CHECK(backend->resident_reuse_count() > hits);
    backend->shutdown();
}

TEST_CASE("Failed acoustic momentum releases velocity inputs before retry",
          "[vulkan][acoustic-input-reuse][device][atomicity]")
{
    auto           backend = make_backend();
    auto*          kernels = backend->dynamics_kernels();
    AcousticFields fields;
    kernels->begin_acoustic_loop_residency(true, true);
    REQUIRE(kernels->dispatch_acoustic_pressure_cartesian_cgrid(fields.pressure()));
    const auto before  = fields;
    auto       invalid = fields.momentum();
    invalid.dx         = 0;
    REQUIRE_FALSE(kernels->dispatch_acoustic_momentum_cartesian_cgrid(invalid));
    fields.compare(before);
    fields.u[43] += 4;
    fields.rho[43] += .2f;
    fields.p[43] += 20;
    auto reference = fields;
    REQUIRE(kernels->dispatch_acoustic_momentum_cartesian_cgrid(fields.momentum()));
    kernels->release_acoustic_loop_residency();
    REQUIRE(kernels->dispatch_acoustic_momentum_cartesian_cgrid(reference.momentum()));
    fields.compare(reference);
    backend->shutdown();
}

TEST_CASE("Acoustic reuse does not affect standalone dispatches in another level",
          "[vulkan][acoustic-input-reuse][device][nested]")
{
    auto           backend = make_backend();
    auto*          kernels = backend->dynamics_kernels();
    AcousticFields parent, child;
    kernels->begin_acoustic_loop_residency(true, true);
    REQUIRE(kernels->dispatch_acoustic_pressure_cartesian_cgrid(parent.pressure()));
    backend->select_residency_scope(ComputeResidencyScope::Child);
    backend->begin_resident_step();
    REQUIRE(kernels->dispatch_acoustic_pressure_cartesian_cgrid(child.pressure()));
    child.theta[43] += 20;
    auto       reference = child;
    const auto hits      = backend->resident_reuse_count();
    REQUIRE(kernels->dispatch_acoustic_pressure_cartesian_cgrid(child.pressure()));
    CHECK(backend->resident_reuse_count() == hits);
    backend->select_residency_scope(ComputeResidencyScope::Root);
    kernels->release_acoustic_loop_residency();
    REQUIRE(kernels->dispatch_acoustic_pressure_cartesian_cgrid(reference.pressure()));
    child.compare(reference);
    backend->shutdown();
}

namespace
{
void exercise_unrepaired_fusion(bool metal)
{
    auto reference_backend = make_backend(metal);
    auto fused_backend = make_backend(metal);
    auto* reference = reference_backend->dynamics_kernels();
    auto* fused = fused_backend->dynamics_kernels();
    REQUIRE(fused->supports_unrepaired_acoustic_fused());
    AcousticFields expected, actual;
    std::vector<float> gradient_low(AcousticFields::n, 2.0e-8f);
    const auto dispatch = [&]() {
        const auto p = actual.pressure();
        const auto m = actual.momentum();
        AcousticSubstepFusedCartesianCgridDispatch d;
        d.u = actual.u.data(); d.v = actual.v.data(); d.w = actual.w.data();
        d.rho = actual.rho.data(); d.p = actual.p.data(); d.theta_data = actual.theta.data();
        d.rho0_base_data = actual.rho0.data(); d.p0_base_data = actual.p0.data();
        d.dp0_dz_data = actual.gradient.data(); d.dp0_dz_low_data = gradient_low.data();
        d.nx = d.ny = d.nz = d.base_len = AcousticFields::n;
        d.dx = p.dx; d.dy = p.dy; d.dz = p.dz; d.dt_small = p.dt_small;
        d.gamma_val = static_cast<float>(p.gamma_val);
        d.gamma_low = static_cast<float>(p.gamma_val - static_cast<double>(d.gamma_val));
        d.rho_min = p.rho_min; d.rho_max = p.rho_max; d.p_min = p.p_min; d.p_max = p.p_max;
        d.p_ref_fallback = p.p_ref_fallback;
        d.wind_clamp_h = m.wind_clamp_h; d.wind_clamp_v = m.wind_clamp_v;
        d.repair_primary_state = false;
        return d;
    };
    fused->begin_acoustic_loop_residency();
    for (int step = 0; step < 4; ++step)
    {
        REQUIRE(reference->dispatch_acoustic_pressure_cartesian_cgrid(expected.pressure()));
        auto momentum = expected.momentum(); momentum.dp0_dz_low_data = gradient_low.data();
        REQUIRE(reference->dispatch_acoustic_momentum_cartesian_cgrid(momentum));
        REQUIRE(fused->dispatch_acoustic_substep_fused_cartesian_cgrid(dispatch()));
        actual.compare(expected);
        for (auto* f : {&actual, &expected})
        {
            f->u.front() += .25f; f->v[6] -= .5f; f->w.back() += .125f;
            f->rho[1] += .001f; f->p[1] += .5f;
        }
        for (const char* name : {"u", "v", "w", "rho", "p"})
        {
            reference_backend->invalidate_resident_field(name);
            fused_backend->invalidate_resident_field(name);
        }
    }
    const auto before = actual;
    auto bad = dispatch(); bad.dx = 0;
    CHECK_FALSE(fused->dispatch_acoustic_substep_fused_cartesian_cgrid(bad));
    actual.compare(before);
    CartesianCGridAcousticEnergyReportV1 report;
    bad = dispatch(); bad.energy_report_host = &report;
    CHECK_FALSE(fused->dispatch_acoustic_substep_fused_cartesian_cgrid(bad));
    actual.compare(before);
    bad = dispatch(); bad.gamma_low = std::numeric_limits<float>::quiet_NaN();
    CHECK_FALSE(fused->dispatch_acoustic_substep_fused_cartesian_cgrid(bad));
    actual.compare(before);
    const auto theta_before = actual.theta[43];
    actual.theta[43] = -1.0f;
    fused_backend->invalidate_resident_field("acoustic_theta");
    CHECK_FALSE(fused->dispatch_acoustic_substep_fused_cartesian_cgrid(dispatch()));
    actual.compare(before);
    REQUIRE(actual.theta[43] == -1.0f);
    actual.theta[43] = theta_before;
    fused_backend->invalidate_resident_field("acoustic_theta");
    REQUIRE(reference->dispatch_acoustic_pressure_cartesian_cgrid(expected.pressure()));
    auto momentum = expected.momentum(); momentum.dp0_dz_low_data = gradient_low.data();
    REQUIRE(reference->dispatch_acoustic_momentum_cartesian_cgrid(momentum));
    REQUIRE(fused->dispatch_acoustic_substep_fused_cartesian_cgrid(dispatch()));
    actual.compare(expected);
    fused->release_acoustic_loop_residency();
    reference_backend->shutdown(); fused_backend->shutdown();
}
}

TEST_CASE("Vulkan unrepaired acoustic fusion matches separate precise stages", "[vulkan][acoustic-fusion][device]")
{
    exercise_unrepaired_fusion(false);
}

TEST_CASE("Metal unrepaired acoustic fusion matches separate precise stages", "[metal][acoustic-fusion][device]")
{
    exercise_unrepaired_fusion(true);
}
