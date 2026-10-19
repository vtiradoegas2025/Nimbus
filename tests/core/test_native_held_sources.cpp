#include "catch2/catch.hpp"
#include "core/runtime/native_held_sources.hpp"
#include "core/runtime/simulation_state.hpp"
#include "core/infra/physical_grid_metrics.hpp"
#include "core/infra/physical_cell_domain.hpp"
#include "terrain/earth_fixed_geometry.hpp"
#include "boundary_layer/schemes/slab/slab.hpp"
#include "compute/backend_runtime.hpp"
#include <array>
#include <cmath>
#include <limits>
namespace
{
using namespace tmv;
void exercise(const char* backend)
{
    ComputeBackendRuntimeState runtime;
    runtime.config.backend        = backend;
    runtime.config.allow_fallback = false;
    std::string error;
    REQUIRE(initialize_compute_backend_runtime(runtime, CoordinateSystem::Cartesian, error));
    INFO(error);
    REQUIRE_FALSE(runtime.fallback_active);
    auto* kernels = runtime.backend->terrain_kernels();
    REQUIRE(kernels);
    REQUIRE(kernels->supports_held_sources());
    for (bool staggered : {false, true})
        for (bool terrain : {false, true})
            for (bool periodic : {false, true})
                for (int flags : {1, 2, 3})
                {
                    CAPTURE(backend, staggered, terrain, periodic, flags);
                    constexpr int          nx = 6, ny = 5, nz = 4;
                    SimulationStorageState s{{nx, ny, nz, 100, 50, .1, 100}};
                    s.grid.geometry.initialize(nx,
                                               ny,
                                               nz,
                                               100,
                                               50,
                                               100,
                                               CoordinateSystem::Cartesian,
                                               staggered ? StaggerType::CGrid : StaggerType::Collocated);
                    s.configuration.stagger_type         = staggered ? StaggerType::CGrid : StaggerType::Collocated;
                    using L                              = init::config::LateralBC;
                    s.configuration.lateral_bc_west      = s.configuration.lateral_bc_east =
                        s.configuration.lateral_bc_south = s.configuration.lateral_bc_north =
                            periodic ? L::Periodic : L::Radiative;
                    s.fields.rho = Field3D(nx, ny, nz, 1);
                    s.fields.p   = Field3D(nx, ny, nz, 95000);
                    if (terrain)
                    {
                        auto& t                          = s.runtime.terrain;
                        t.config.scheme_id               = "bell";
                        t.config.coord_id                = "smoothed";
                        t.config.horizontal_placement_id = "earth_fixed";
                        t.config.ztop                    = 200;
                        t.config.bell.h0                 = 20;
                        t.config.bell.a                  = 200;
                        EarthFixedTerrainBuildRequest request;
                        request.nr       = nx;
                        request.nth      = ny;
                        request.nz       = nz;
                        request.dx_m     = 100;
                        request.dy_m     = 100;
                        request.sampling = EarthFixedTerrainSampling::CellAverageBilinear;
                        auto prepared    = prepare_earth_fixed_terrain_commit(t, request);
                        REQUIRE(commit_earth_fixed_terrain(prepared, t));
                    }
                    std::array<Field3D, 7> sources;
                    for (auto& f : sources)
                        f.resize(nx, ny, nz, 0);
                    for (int i = 0; i < nx; ++i)
                        for (int j = 0; j < ny; ++j)
                            for (int k = 0; k < nz; ++k)
                            {
                                s.fields.rho(i, j, k) = .8f + .01f * i + .02f * j - .03f * k;
                                for (int f = 0; f < 7; ++f)
                                    sources[f](i, j, k) =
                                        float((f % 2 ? -1 : 1) * (1000 + 100 * f + 3 * i - 2 * j + k));
                            }
                    auto                                               view   = make_simulation_state_view(s);
                    const auto                                         domain = physical_scalar_domain(view.grid);
                    const physical_grid_metrics::ControlVolumeMassView mass(view);
                    const auto                                         owned = [&](int i, int j, int k)
                    {
                        return i >= domain.i_begin && i < domain.i_end && j >= domain.j_begin && j < domain.j_end &&
                               k >= domain.k_begin && k < domain.k_end;
                    };
                    for (int i = 0; i < nx; ++i)
                        for (int j = 0; j < ny; ++j)
                            for (int k = 0; k < nz; ++k)
                                if (!owned(i, j, k))
                                    for (int f = 0; f < 6; ++f)
                                        sources[f](i, j, k) = std::numeric_limits<float>::quiet_NaN();
                    HeldSourceInputs input;
                    if (flags & 1)
                    {
                        input.u_force     = &sources[0];
                        input.v_force     = &sources[1];
                        input.theta_rate  = &sources[2];
                        input.vapor_rate  = &sources[3];
                        input.liquid_rate = &sources[4];
                        input.ice_rate    = &sources[5];
                    }
                    if (flags & 2)
                        input.radiation_power = &sources[6];
                    for (int f = 0; f < 7; ++f)
                    {
                        sources[f](domain.i_begin, domain.j_begin, domain.k_begin)       = 1e10f;
                        sources[f](domain.i_end - 1, domain.j_end - 1, domain.k_end - 1) = 1e-36f;
                        if (domain.i_end - domain.i_begin > 2)
                            sources[f](domain.i_begin + 1, domain.j_begin, domain.k_begin) = -1e10f;
                    }
                    auto                                result = reconstruct_native_held_sources(view, input, *kernels);
                    const std::array<const Field3D*, 7> values{&result.du,
                                                               &result.dv,
                                                               &result.dtheta_pbl,
                                                               &result.dqv,
                                                               &result.dqc,
                                                               &result.dqi,
                                                               &result.dtheta_rad};
                    for (int i = 0; i < nx; ++i)
                        for (int j = 0; j < ny; ++j)
                            for (int k = 0; k < nz; ++k)
                            {
                                for (int f = 0; f < 7; ++f)
                                {
                                    if ((f < 6 && !(flags & 1)) || (f == 6 && !(flags & 2)))
                                    {
                                        CHECK(values[f]->empty());
                                        continue;
                                    }
                                    double expected = 0;
                                    if (f == 6)
                                        expected = sources[f](i, j, k) /
                                                   (mass.scalar_mass_kg(s.fields.rho, i, j, k) * 1004 *
                                                    std::pow(s.fields.p(i, j, k) / 100000.0, 287.0 / 1004));
                                    else if (owned(i, j, k))
                                    {
                                        if (!staggered || f >= 2)
                                            expected = sources[f](i, j, k) / mass.scalar_mass_kg(s.fields.rho, i, j, k);
                                        else if (f == 0 && i < (periodic ? domain.i_end : domain.i_end - 1))
                                        {
                                            int right = i + 1 < domain.i_end ? i + 1 : domain.i_begin;
                                            expected  = .5 * (double(sources[0](i, j, k)) + sources[0](right, j, k)) /
                                                        mass.u_dual_mass_kg(s.fields.rho, i, j, k, right);
                                        }
                                        else if (f == 1 && j < (periodic ? domain.j_end : domain.j_end - 1))
                                        {
                                            int north = j + 1 < domain.j_end ? j + 1 : domain.j_begin;
                                            expected  = .5 * (double(sources[1](i, j, k)) + sources[1](i, north, k)) /
                                                        mass.v_dual_mass_kg(s.fields.rho, i, j, k, north);
                                        }
                                    }
                                    CHECK(std::abs((*values[f])(i, j, k) - expected) <=
                                          3e-7 * std::abs(expected) + std::numeric_limits<float>::denorm_min());
                                }
                            }
                    s.fields.rho(domain.i_end - 1, domain.j_end - 1, domain.k_end - 1) = 0;
                    REQUIRE_THROWS(reconstruct_native_held_sources(view, input, *kernels));
                }
    CHECK(runtime.backend->successful_kernel_dispatch_count(nimbus::compute::KernelId::HeldSources) > 0);
}
} // namespace
TEST_CASE("Native held sources use current scalar and dual masses on Metal", "[metal][held_sources]")
{
    exercise("metal");
}
TEST_CASE("Native held sources use current scalar and dual masses on Vulkan", "[vulkan][held_sources]")
{
    exercise("vulkan");
}
