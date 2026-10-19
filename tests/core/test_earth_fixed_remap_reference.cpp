#include "catch2/catch.hpp"
#include "core/runtime/earth_fixed_remap.hpp"
#include "core/runtime/simulation_state.hpp"
#include "diagnostics/momentum_budget.hpp"
#include "dynamics/eos_pressure.hpp"
#include "terrain/conservative_transport.hpp"

#include <array>
#include <cmath>

namespace
{
SimulationStorageState reference_state(int cells, double height)
{
    constexpr int nz=8;
    constexpr double width=4800.0, dz=500.0;
    const int n=cells+2;
    const double dx=width/cells;
    SimulationStorageState s{{n,n,nz,dx,dz,0.2,dx}};
    s.grid.geometry.initialize(n,n,nz,dx,dz,dx,CoordinateSystem::Cartesian,StaggerType::CGrid);
    s.configuration.stagger_type=StaggerType::CGrid;
    s.configuration.domain_frame.earth_origin_at_t0_m={-dx,-dx};
    using BC=tmv::init::config::LateralBC;
    s.configuration.lateral_bc_west=s.configuration.lateral_bc_east=BC::Radiative;
    s.configuration.lateral_bc_south=s.configuration.lateral_bc_north=BC::Radiative;
    s.runtime.turbulence.config.scheme_id="none";
    s.runtime.boundary_layer.config.scheme_id="none";
    s.runtime.radiation.config.scheme_id="none";
    auto& terrain=s.runtime.terrain;
    terrain.config.scheme_id="bell";
    terrain.config.coord_id="smoothed";
    terrain.config.horizontal_placement_id="earth_fixed";
    terrain.config.ztop=nz*dz;
    terrain.config.smoothing_decay=0.1;
    terrain.config.bell.h0=height;
    terrain.config.bell.a=900.0;
    terrain.config.bell.axisymmetric=true;
    terrain.config.feature_center_earth_x_m=width/2;
    terrain.config.feature_center_earth_y_m=width/2;
    tmv::EarthFixedTerrainBuildRequest request;
    request.nr=n; request.nth=n; request.nz=nz; request.dx_m=dx; request.dy_m=dx;
    request.grid_origin_earth_m={-dx,-dx}; request.feature_center_earth_m={width/2,width/2};
    request.sampling=tmv::EarthFixedTerrainSampling::CellAverageBilinear;
    auto prepared=tmv::prepare_earth_fixed_terrain_commit(terrain,request);
    REQUIRE(tmv::commit_earth_fixed_terrain(prepared,terrain));
    s.base.theta0.assign(nz,300.0); s.base.qv0.assign(nz,0.0);
    s.base.u0.assign(nz,0.0); s.base.v0.assign(nz,0.0);
    for (int k=0;k<nz;++k)
    {
        const double z=(k+0.5)*dz;
        s.base.p0.push_back(100000.0*std::pow(1.0-physical_constants::gravity_ms2*z/(physical_constants::specific_heat_cp_jkgk*300.0),
            physical_constants::specific_heat_cp_jkgk/physical_constants::gas_constant_dry_air_jkgk));
    }
    tmv::fields::visit(s.fields,tmv::fields::prognostic,[&](const char*,Field3D& field) {
        field.resize(n,n,nz,0.0f);
    });
    for (int i=0;i<n;++i)
        for (int j=0;j<n;++j)
            for (int k=0;k<nz;++k)
            {
                const double z=terrain.metrics.z(i,j,k);
                s.fields.theta(i,j,k)=300.0f;
                s.fields.p(i,j,k)=static_cast<float>(tmv::terrain_transport::reference_profile_value_fp32(s.base.p0,z,dz));
                s.fields.rho(i,j,k)=static_cast<float>(tmv::terrain_transport::reference_density_fp32(s.base.p0,s.base.theta0,z,dz));
            }
    return s;
}
}

TEST_CASE("Earth-fixed reference-column remapping converges under horizontal refinement",
          "[core][earth_fixed][remap][reference][convergence]")
{
    const auto error=[](int cells) {
        auto s=reference_state(cells,300.0);
        s.configuration.domain_frame.velocity_ms={4.0,-2.0};
        const auto report=tmv::remap_earth_fixed_window(s,s.configuration.domain_frame,0.0,1.0);
        REQUIRE(report.energy.complete);
        std::array<double,2> errors{};
        unsigned count=0;
        for (int i=1;i<s.grid.nr-1;++i)
            for (int j=1;j<s.grid.nth-1;++j)
            {
                const double x=(i-0.5)*s.grid.dx, y=(j-0.5)*s.grid.dy;
                if (x<1200.0 || x>3600.0 || y<1200.0 || y>3600.0) continue;
                for (int k=0;k<s.grid.nz;++k)
                {
                    const double z=s.runtime.terrain.metrics.z(i,j,k);
                    const double rho=tmv::terrain_transport::reference_density_fp32(s.base.p0,s.base.theta0,z,s.grid.dz);
                    errors[0]+=std::abs(s.fields.rho(i,j,k)-rho);
                    errors[1]+=std::abs(s.fields.theta(i,j,k)-300.0);
                    ++count;
                }
            }
        for (auto& value:errors) value/=count;
        return errors;
    };
    const auto coarse=error(12), fine=error(24);
    INFO("density error " << coarse[0] << " -> " << fine[0]);
    INFO("theta error " << coarse[1] << " -> " << fine[1]);
    CHECK(fine[0]<0.8*coarse[0]);
    CHECK(fine[1]<0.8*coarse[1]);
}

TEST_CASE("Flat Earth-fixed remapping preserves uniform primitive fields",
          "[core][earth_fixed][remap][reference][gcl]")
{
    auto s=reference_state(12,0.0);
    const float pressure=static_cast<float>(tmv::dynamics_eos::pressure_fp64(1.0f,300.0f));
    s.base.p0.assign(s.grid.nz,pressure);
    s.base.qv0.assign(s.grid.nz,0.005f);
    s.base.u0.assign(s.grid.nz,8.0);
    s.base.v0.assign(s.grid.nz,-4.0);
    s.fields.rho.fill(1.0f); s.fields.p.fill(pressure);
    s.fields.theta.fill(300.0f); s.fields.qv.fill(0.005f);
    s.fields.u.fill(8.0f); s.fields.v.fill(-4.0f);
    s.configuration.domain_frame.velocity_ms={20.0,-10.0};
    const auto report=tmv::remap_earth_fixed_window(s,s.configuration.domain_frame,0.0,1.0);
    for (int i=1;i<s.grid.nr-1;++i)
        for (int j=1;j<s.grid.nth-1;++j)
            for (int k=0;k<s.grid.nz;++k)
            {
                CHECK(s.fields.rho(i,j,k)==1.0f);
                CHECK(s.fields.theta(i,j,k)==300.0f);
                CHECK(s.fields.p(i,j,k)==pressure);
                CHECK(s.fields.qv(i,j,k)==0.005f);
                CHECK(s.fields.u(i,j,k)==8.0f);
                CHECK(s.fields.v(i,j,k)==-4.0f);
                CHECK(s.fields.w(i,j,k)==0.0f);
            }
    for (std::size_t n=0;n<3;++n)
        CHECK(std::abs(report.momentum_storage_rounding[n])
            < 1.0e-6*report.dry_mass.initial_inventory*180.0);
}

TEST_CASE("Physical momentum includes the terrain-ground half volume",
          "[core][earth_fixed][momentum][ground]")
{
    auto s=reference_state(12,300.0);
    for (int i=0;i<s.grid.nr;++i)
        for (int j=0;j<s.grid.nth;++j)
            for (int k=0;k<s.grid.nz;++k)
                s.fields.u(i,j,k)=static_cast<float>(i);
    auto view=make_simulation_state_view(s);
    const auto legacy=compute_momentum_budget(view);
    const auto physical=compute_physical_momentum_budget_v2(view);
    double expected=0.0;
    const auto& g=s.runtime.terrain.metrics;
    for (int i=1;i<s.grid.nr-1;++i)
        for (int j=1;j<s.grid.nth-1;++j)
            expected+=0.5*s.fields.rho(i,j,0)*s.grid.dx*s.grid.dy
                *(g.z_face(i,j,1)-g.z_face(i,j,0))
                *(-(i-0.5)*g.Gx_face(i,j,0));
    REQUIRE(std::abs(expected)>1.0);
    CHECK(physical.momentum_z-legacy.momentum_z==Approx(expected).epsilon(1.0e-12));
}

TEST_CASE("Earth-fixed remap subcycling closes inventories across reference cells",
          "[core][earth_fixed][remap][subcycling]")
{
    auto s=reference_state(12,300.0);
    s.configuration.domain_frame.velocity_ms={180.0,-180.0};
    const auto report=tmv::remap_earth_fixed_window(s,s.configuration.domain_frame,0.0,4.0);
    REQUIRE(report.substeps>1);
    CHECK(s.time_s==4.0);
    CHECK(s.runtime.terrain.earth_fixed_placement.grid_origin_earth_m.x_m==320.0);
    CHECK(s.runtime.terrain.earth_fixed_placement.grid_origin_earth_m.y_m==-1120.0);
    const auto& mass=report.dry_mass;
    CHECK(std::abs(mass.final_inventory-mass.initial_inventory+mass.boundary_outflow
        -mass.storage_rounding)/mass.initial_inventory<1.0e-12);
    CHECK(report.energy.complete);
    for (int i=1;i<s.grid.nr-1;++i)
        for (int j=1;j<s.grid.nth-1;++j)
            for (int k=0;k<s.grid.nz;++k)
            {
                REQUIRE(s.fields.rho(i,j,k)>0.0f);
                REQUIRE(std::isfinite(s.fields.theta(i,j,k)));
                REQUIRE(s.fields.theta(i,j,k)>290.0f);
                REQUIRE(s.fields.theta(i,j,k)<310.0f);
            }
}
