#include "catch2/catch.hpp"
#include "compute/backend_types.hpp"
#include "compute/kernel_startup_plan.hpp"
#include "compute/vulkan_startup_profile.hpp"
#include "compute/vulkan_pipeline_cache_file.hpp"
#include "compute/vulkan_pipeline_cache_signature.hpp"

#include <algorithm>
#include <functional>

namespace
{

using namespace nimbus::compute;

tmv::init::config::ConfigV1 dry_profile()
{
    using namespace tmv::init::config;
    ConfigV1 config;
    config.grid.nx = config.grid.ny = config.grid.nz = 18;
    config.grid.dx = config.grid.dy                  = 500.0;
    config.grid.dz                                   = 250.0;
    config.grid.ztop                                 = 4500.0;
    config.time.dt_large                             = 0.1;
    config.dynamics.scheme                           = "klemp_wilhelmson_split_cartesian_cgrid";
    config.advection.scheme                          = AdvectionScheme::TVD;
    config.microphysics.scheme                       = MicrophysicsScheme::None;
    config.turbulence.scheme                         = TurbulenceScheme::None;
    config.boundary_layer.scheme                     = BoundaryLayerScheme::None;
    config.radiation.scheme                          = RadiationScheme::None;
    config.surface.scheme                            = SurfaceScheme::None;
    config.surface.apply_fluxes                      = false;
    config.terrain.scheme                            = TerrainScheme::None;
    config.boundary_conditions.lateral.west          = LateralBC::RigidSlip;
    config.boundary_conditions.lateral.east          = LateralBC::RigidSlip;
    config.boundary_conditions.lateral.south         = LateralBC::RigidSlip;
    config.boundary_conditions.lateral.north         = LateralBC::RigidSlip;
    config.boundary_conditions.bottom                = BottomBC::RigidNoSlip;
    config.boundary_conditions.top                   = TopBC::Open;
    config.damping.enabled                           = false;
    config.coriolis.enabled                          = false;
    config.extensions.compute.backend                = tmv::init::config::ComputeBackend::GPU;
    config.extensions.compute.allow_fallback         = false;
    config.extensions.compute.gpu_apply_microphysics = false;
    config.extensions.compute.gpu_apply_slow         = false;
    config.extensions.compute.gpu_apply_sgs          = false;
    config.extensions.chaos.scheme                   = ChaosScheme::InitialConditions;
    config.extensions.execution.mode                 = ExecutionMode::Qualified;
    config.output.preset                             = OutputPreset::Minimal;
    config.output.format                             = OutputFormat::Npy3D;
    config.output.async_io                           = false;
    return config;
}

struct FakePipelines
{
    std::array<bool, kernel_count()>         ready{};
    std::array<unsigned int, kernel_count()> creates{};
    std::array<unsigned int, kernel_count()> destroys{};
    std::vector<KernelId>                    observed;
    KernelId                                 failed = KernelId::Count;

    bool prepare(const KernelStartupPlan* plan, std::string& error, const KernelManifest& manifest = kernel_manifest())
    {
        return prepare_startup_pipelines(
            plan,
            manifest,
            [&](const KernelManifestEntry& kernel, std::string& pipeline_error)
            {
                const auto index = static_cast<std::size_t>(kernel.id);
                ++creates[index];
                if (kernel.id == failed)
                {
                    pipeline_error = "injected missing pipeline";
                    return false;
                }
                ready[index] = true;
                return true;
            },
            [&](const KernelManifestEntry& kernel)
            {
                const auto index = static_cast<std::size_t>(kernel.id);
                REQUIRE(ready[index]);
                ready[index] = false;
                ++destroys[index];
            },
            [&](const KernelManifestEntry& kernel, bool success, std::size_t ordinal, const std::string&)
            {
                REQUIRE(ordinal == observed.size() + 1);
                REQUIRE(success == (kernel.id != failed));
                observed.push_back(kernel.id);
            },
            error);
    }

    void shutdown()
    {
        for (std::size_t index = 0; index < ready.size(); ++index)
        {
            if (ready[index])
            {
                ready[index] = false;
                ++destroys[index];
            }
        }
        observed.clear();
    }
};

} // namespace

TEST_CASE("Exact dry Vulkan startup prepares every declared owner dependency", "[vulkan][startup-plan]")
{
    const auto plan = vulkan_startup_plan_for_profile(dry_profile());
    REQUIRE(plan);
    REQUIRE(plan->size() == 15);
    REQUIRE(plan->profile() == "flat-dry-cgrid18-v1");
    for (const auto owner : {KernelStartupOwner::FlatCartesianCgridDynamics,
                             KernelStartupOwner::CartesianScalarTransport,
                             KernelStartupOwner::CartesianExplicitDiffusion})
    {
        for (const auto kernel : startup_owner_kernels(owner))
        {
            REQUIRE(plan->contains(kernel));
        }
    }
    REQUIRE_FALSE(plan->contains(KernelId::ThompsonCompletion));
    REQUIRE_FALSE(plan->contains(KernelId::EarthGeometry));
    REQUIRE_FALSE(plan->contains(KernelId::AcousticContinuityMassFluxCartesianAgrid));
    REQUIRE_FALSE(plan->contains(KernelId::Count));

    FakePipelines pipelines;
    std::string   error = "stale error";
    REQUIRE(pipelines.prepare(plan.get(), error));
    REQUIRE(error.empty());
    REQUIRE(pipelines.observed.size() == plan->size());
    for (std::size_t index = 0; index < kernel_count(); ++index)
    {
        REQUIRE(pipelines.ready[index] == plan->contains(static_cast<KernelId>(index)));
        REQUIRE(pipelines.creates[index] == (pipelines.ready[index] ? 1 : 0));
    }
}

TEST_CASE("Startup scope neighbors retain all-kernel compatibility", "[vulkan][startup-plan]")
{
    using namespace tmv::init::config;
    const std::vector<std::function<void(ConfigV1&)>> mutations = {
        [](ConfigV1& c) { c.extensions.compute.backend = tmv::init::config::ComputeBackend::CPU; },
        [](ConfigV1& c) { c.extensions.compute.backend = tmv::init::config::ComputeBackend::Metal; },
        [](ConfigV1& c) { c.extensions.compute.allow_fallback = true; },
        [](ConfigV1& c) { c.extensions.compute.validate_parity = true; },
        [](ConfigV1& c) { c.extensions.compute.gpu_apply_microphysics = true; },
        [](ConfigV1& c) { c.extensions.compute.gpu_apply_slow = true; },
        [](ConfigV1& c) { c.extensions.compute.gpu_apply_sgs = true; },
        [](ConfigV1& c) { c.grid.staggering = StaggerType::Collocated; },
        [](ConfigV1& c) { ++c.grid.nx; },
        [](ConfigV1& c) { c.grid.dx += 1; },
        [](ConfigV1& c) { c.grid.ztop += 1; },
        [](ConfigV1& c) { c.dynamics.scheme = "klemp_wilhelmson_split_cartesian_agrid"; },
        [](ConfigV1& c) { c.dynamics.axisymmetric = true; },
        [](ConfigV1& c) { c.time.integrator = TimeIntegrator::RK4; },
        [](ConfigV1& c) { c.time.dt_large = 0.2; },
        [](ConfigV1& c) { ++c.time.acoustic_substeps; },
        [](ConfigV1& c) { c.time.adaptive_dt = true; },
        [](ConfigV1& c) { c.advection.scheme = AdvectionScheme::WENO5; },
        [](ConfigV1& c) { c.diffusion.scheme = DiffusionScheme::Implicit; },
        [](ConfigV1& c) { c.diffusion.K_h = 1; },
        [](ConfigV1& c) { c.diffusion.K_v = 1; },
        [](ConfigV1& c) { c.diffusion.coefficient_source = DiffusionCoefficientSource::LinearHeight; },
        [](ConfigV1& c) { c.diffusion.apply_to = DiffusionTarget::ScalarsAndMomentum; },
        [](ConfigV1& c) { c.microphysics.scheme = MicrophysicsScheme::Thompson; },
        [](ConfigV1& c) { c.turbulence.scheme = TurbulenceScheme::Smagorinsky; },
        [](ConfigV1& c) { c.boundary_layer.scheme = BoundaryLayerScheme::MYNN; },
        [](ConfigV1& c) { c.radiation.scheme = RadiationScheme::RRTMG; },
        [](ConfigV1& c) { c.surface.scheme = SurfaceScheme::Slab; },
        [](ConfigV1& c) { c.surface.apply_fluxes = true; },
        [](ConfigV1& c) { c.terrain.scheme = TerrainScheme::Bell; },
        [](ConfigV1& c) { c.domain_motion.frame_velocity.u_ms = 1; },
        [](ConfigV1& c) { c.domain_motion.frame_velocity.v_ms = 1; },
        [](ConfigV1& c) { c.domain_motion.lifetime_planning.enabled = true; },
        [](ConfigV1& c) { c.domain_motion.tracking.enabled = true; },
        [](ConfigV1& c) { c.domain_motion.tracking.recycle_domain = true; },
        [](ConfigV1& c) { c.domain_motion.nesting.enabled = true; },
        [](ConfigV1& c) { c.domain_motion.nesting.feedback = NestingFeedback::TwoWay; },
        [](ConfigV1& c) { c.boundary_conditions.lateral.east = LateralBC::Radiative; },
        [](ConfigV1& c) { c.boundary_conditions.bottom = BottomBC::RigidSlip; },
        [](ConfigV1& c) { c.boundary_conditions.top = TopBC::Rigid; },
        [](ConfigV1& c) { c.damping.enabled = true; },
        [](ConfigV1& c) { c.damping.lateral_sponge_enabled = false; },
        [](ConfigV1& c) { --c.damping.lateral_width; },
        [](ConfigV1& c) { c.damping.lateral_alpha = 0; },
        [](ConfigV1& c) { c.damping.w_only_top_sponge = false; },
        [](ConfigV1& c) { --c.damping.w_only_depth_cells; },
        [](ConfigV1& c) { c.damping.w_only_alpha_max = 0; },
        [](ConfigV1& c) { c.coriolis.enabled = true; },
        [](ConfigV1& c) { c.extensions.chaos.scheme = ChaosScheme::FullStochastic; },
        [](ConfigV1& c) { c.extensions.chaos.native_generation_required = true; },
        [](ConfigV1& c) { c.extensions.solver.slow_rk2 = false; },
        [](ConfigV1& c) { c.extensions.solver.lid_bc_gamma = 0; },
        [](ConfigV1& c) { c.extensions.solver.axis_u_m1_filter = true; },
        [](ConfigV1& c) { c.radar.enabled = true; },
        [](ConfigV1& c) { c.output.trajectories = true; },
        [](ConfigV1& c) { c.output.live_shm = true; },
        [](ConfigV1& c) { c.output.format = OutputFormat::Netcdf; },
        [](ConfigV1& c) { c.output.async_io = true; },
        [](ConfigV1& c) { c.extensions.execution.mode = ExecutionMode::Exploratory; }};
    for (std::size_t index = 0; index < mutations.size(); ++index)
    {
        INFO(index);
        auto config = dry_profile();
        mutations[index](config);
        REQUIRE_FALSE(vulkan_startup_plan_for_profile(config));
    }
    auto continued                 = dry_profile();
    continued.time.run_time        = 4;
    continued.time.output_interval = 301;
    const auto original            = vulkan_startup_plan_for_profile(dry_profile());
    const auto plan                = vulkan_startup_plan_for_profile(continued);
    REQUIRE(plan);
    REQUIRE(plan->cache_identity() == original->cache_identity());
}

TEST_CASE("Startup plans reject invalid duplicate unavailable and incomplete requests", "[vulkan][startup-plan]")
{
    const std::vector<KernelStartupOwner> owners  = {KernelStartupOwner::CartesianExplicitDiffusion};
    const std::vector<KernelId>           kernels = {KernelId::Diffusion};
    std::string                           error;
    const auto                            make = [&](const std::vector<KernelStartupOwner>& requested_owners,
                                                     const std::vector<KernelId>&           requested_kernels,
                                                     const KernelManifest&                  manifest)
    { return KernelStartupPlan::make("test", requested_owners, requested_kernels, manifest, error); };
    REQUIRE(make(owners, kernels, kernel_manifest()));
    REQUIRE_FALSE(make({}, kernels, kernel_manifest()));
    REQUIRE_FALSE(make(owners, {}, kernel_manifest()));
    REQUIRE_FALSE(make({KernelStartupOwner::Count}, kernels, kernel_manifest()));
    REQUIRE_FALSE(make({owners.front(), owners.front()}, kernels, kernel_manifest()));
    REQUIRE_FALSE(make(owners, {KernelId::Diffusion, KernelId::Diffusion}, kernel_manifest()));
    REQUIRE_FALSE(make(owners, {KernelId::Count}, kernel_manifest()));
    REQUIRE_FALSE(make(owners, {KernelId::TvdVerticalFlux}, kernel_manifest()));
    REQUIRE(error == "incomplete kernel startup owner dependencies");
    REQUIRE_FALSE(make(owners, {KernelId::Diffusion, KernelId::DiffuseCartesian}, kernel_manifest()));
    REQUIRE_FALSE(KernelStartupPlan::make("", owners, kernels, kernel_manifest(), error));
    REQUIRE_FALSE(KernelStartupPlan::make(std::string(129, 'x'), owners, kernels, kernel_manifest(), error));
    REQUIRE_FALSE(make(owners, std::vector<KernelId>(kernel_count() + 1, KernelId::Diffusion), kernel_manifest()));
    auto unavailable                                                        = kernel_manifest();
    unavailable[static_cast<std::size_t>(KernelId::Diffusion)].availability = KernelAvailability::Deferred;
    REQUIRE_FALSE(make(owners, kernels, unavailable));
    REQUIRE(error == "required startup kernel is unavailable");
    const auto plan = make(owners, kernels, kernel_manifest());
    REQUIRE(plan);
    REQUIRE_FALSE(plan->validate(unavailable, error));
    FakePipelines pipelines;
    REQUIRE_FALSE(pipelines.prepare(plan.get(), error, unavailable));
    REQUIRE(pipelines.observed.empty());
    REQUIRE(std::all_of(pipelines.creates.begin(), pipelines.creates.end(), [](auto count) { return count == 0; }));
}

TEST_CASE("Startup plan identity canonicalizes order and binds owners and extra requirements", "[vulkan][startup-plan]")
{
    std::vector<KernelStartupOwner> owners = {KernelStartupOwner::FlatCartesianCgridDynamics,
                                              KernelStartupOwner::CartesianScalarTransport};
    std::vector<KernelId>           kernels;
    for (const auto owner : owners)
    {
        const auto dependencies = startup_owner_kernels(owner);
        kernels.insert(kernels.end(), dependencies.begin(), dependencies.end());
    }
    std::string error;
    const auto  first = KernelStartupPlan::make("test", owners, kernels, kernel_manifest(), error);
    REQUIRE(first);
    std::reverse(owners.begin(), owners.end());
    std::reverse(kernels.begin(), kernels.end());
    const auto reversed = KernelStartupPlan::make("test", owners, kernels, kernel_manifest(), error);
    REQUIRE(reversed);
    REQUIRE(reversed->cache_identity() == first->cache_identity());
    kernels.push_back(KernelId::Diffusion);
    const auto expanded = KernelStartupPlan::make("test", owners, kernels, kernel_manifest(), error);
    REQUIRE(expanded);
    REQUIRE(expanded->cache_identity() != first->cache_identity());
    const auto renamed = KernelStartupPlan::make("other", owners, kernels, kernel_manifest(), error);
    REQUIRE(renamed);
    REQUIRE(renamed->cache_identity() != expanded->cache_identity());
    owners.push_back(KernelStartupOwner::CartesianExplicitDiffusion);
    const auto changed_owners = KernelStartupPlan::make("test", owners, kernels, kernel_manifest(), error);
    REQUIRE(changed_owners);
    REQUIRE(changed_owners->cache_identity() != expanded->cache_identity());
}

TEST_CASE("Scoped cache signatures isolate the broad cache and preserve default identity", "[vulkan][startup-plan]")
{
    const auto plan = vulkan_startup_plan_for_profile(dry_profile());
    REQUIRE(plan);
    const auto broad  = tmv_vulkan::pipeline_cache_manifest_signature(kernel_manifest());
    const auto scoped = tmv_vulkan::pipeline_cache_manifest_signature(kernel_manifest(), plan.get());
    REQUIRE(broad == "eefb18304f5c2a4b");
    REQUIRE(scoped != broad);
    REQUIRE(scoped.size() == 16);
    auto changed_manifest                                                         = kernel_manifest();
    changed_manifest[static_cast<std::size_t>(KernelId::Diffusion)].artifact_hash = "0000000000000000";
    REQUIRE(tmv_vulkan::pipeline_cache_manifest_signature(changed_manifest, plan.get()) != scoped);
    tmv_vulkan::pipeline_cache_file::Identity identity;
    identity.manifest_signature = broad;
    const auto broad_filename   = tmv_vulkan::pipeline_cache_file::filename(identity);
    identity.manifest_signature = scoped;
    REQUIRE(tmv_vulkan::pipeline_cache_file::filename(identity) != broad_filename);
}

TEST_CASE("Scoped preparation rolls back every earlier pipeline on any required failure", "[vulkan][startup-plan]")
{
    const auto plan = vulkan_startup_plan_for_profile(dry_profile());
    REQUIRE(plan);
    for (const auto& failed : kernel_manifest())
    {
        if (!plan->contains(failed.id))
        {
            continue;
        }
        INFO(failed.operation_id);
        FakePipelines pipelines;
        pipelines.failed = failed.id;
        std::string error;
        REQUIRE_FALSE(pipelines.prepare(plan.get(), error));
        REQUIRE(error == std::string(failed.operation_id) + ": injected missing pipeline");
        REQUIRE(std::none_of(pipelines.ready.begin(), pipelines.ready.end(), [](bool value) { return value; }));
        for (const auto& kernel : kernel_manifest())
        {
            const auto index = static_cast<std::size_t>(kernel.id);
            REQUIRE(pipelines.destroys[index] ==
                    ((plan->contains(kernel.id) && index < static_cast<std::size_t>(failed.id)) ? 1 : 0));
        }
        const auto before_shutdown = pipelines.destroys;
        pipelines.shutdown();
        pipelines.shutdown();
        REQUIRE(pipelines.destroys == before_shutdown);
        pipelines.failed = KernelId::Count;
        REQUIRE(pipelines.prepare(plan.get(), error));
        REQUIRE(pipelines.observed.size() == plan->size());
        pipelines.shutdown();
        REQUIRE(std::none_of(pipelines.ready.begin(), pipelines.ready.end(), [](bool value) { return value; }));
    }
}

TEST_CASE("Unspecified callers retain all active kernels and optional failure policy", "[vulkan][startup-plan]")
{
    ComputeBackendConfig config;
    REQUIRE_FALSE(config.startup_plan);
    FakePipelines pipelines;
    std::string   error;
    REQUIRE(pipelines.prepare(config.startup_plan.get(), error));
    REQUIRE(pipelines.observed.size() == 77);
    pipelines.shutdown();
    pipelines.failed = KernelId::ThompsonCompletion;
    REQUIRE(pipelines.prepare(nullptr, error));
    REQUIRE(pipelines.observed.size() == 77);
    REQUIRE_FALSE(pipelines.ready[static_cast<std::size_t>(KernelId::ThompsonCompletion)]);
    pipelines.shutdown();
    pipelines.failed = KernelId::TvdVerticalFlux;
    REQUIRE_FALSE(pipelines.prepare(nullptr, error));
    REQUIRE(std::none_of(pipelines.ready.begin(), pipelines.ready.end(), [](bool value) { return value; }));
    pipelines.shutdown();
    pipelines.failed    = KernelId::Count;
    config.startup_plan = vulkan_startup_plan_for_profile(dry_profile());
    const auto saved    = config;
    config              = {};
    REQUIRE_FALSE(config.startup_plan);
    REQUIRE(saved.startup_plan);
    REQUIRE(pipelines.prepare(saved.startup_plan.get(), error));
    REQUIRE(pipelines.observed.size() == 15);
    pipelines.shutdown();
    REQUIRE(pipelines.prepare(config.startup_plan.get(), error));
    REQUIRE(pipelines.observed.size() == 77);
    pipelines.shutdown();
}
