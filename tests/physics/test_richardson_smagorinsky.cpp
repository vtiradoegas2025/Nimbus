#include "catch2/catch.hpp"
#include "turbulence/schemes/smagorinsky/smagorinsky.hpp"

#include <cmath>
#include <limits>

TEST_CASE("Lilly equilibrium satisfies stable neutral unstable and zero-shear limits",
          "[physics][turbulence][richardson][reference]")
{
    using tmv::eddy_viscosity::lilly_equilibrium_strain_rate;
    CHECK(lilly_equilibrium_strain_rate(0.0004,0.0,0.25)==Approx(0.02));
    CHECK(lilly_equilibrium_strain_rate(0.0004,0.000075,0.25)==Approx(0.01));
    CHECK(lilly_equilibrium_strain_rate(0.0004,0.0001,0.25)==0.0);
    CHECK(lilly_equilibrium_strain_rate(0.0004,0.0002,0.25)==0.0);
    CHECK(lilly_equilibrium_strain_rate(0.0004,-0.0003,0.25)==Approx(0.04));
    CHECK(lilly_equilibrium_strain_rate(0.0,-0.0001,0.25)==Approx(0.02));
    CHECK(lilly_equilibrium_strain_rate(0.0,0.0001,0.25)==0.0);
    CHECK(lilly_equilibrium_strain_rate(0.0004,0.0001,0.5)==Approx(std::sqrt(0.0002)));
    CHECK(std::isfinite(lilly_equilibrium_strain_rate(1e300,-1e300,1e-100)));
    CHECK_THROWS_AS(lilly_equilibrium_strain_rate(-1.0,0.0,1.0),std::invalid_argument);
    CHECK_THROWS_AS(lilly_equilibrium_strain_rate(1.0,0.0,0.0),std::invalid_argument);
    CHECK_THROWS_AS(lilly_equilibrium_strain_rate(1.0,std::numeric_limits<double>::quiet_NaN(),1.0),std::invalid_argument);
}

TEST_CASE("Production Lilly coefficients obey analytic shear and close dry inventories",
          "[physics][turbulence][richardson][conservation][atomicity]")
{
    constexpr int n=10;
    tmv::GridMetrics grid;
    grid.dx=grid.dy=100.0;
    grid.dz.assign(n,100.0);
    Field3D u(n,n,n),v(n,n,n,0.0f),w(n,n,n,0.0f),theta(n,n,n),rho(n,n,n,1.0f),qv(n,n,n,0.008f);
    for (int i=0;i<n;++i)
        for (int j=0;j<n;++j)
            for (int k=0;k<n;++k)
            {
                u(i,j,k)=static_cast<float>(2*std::min(k,n-2));
                theta(i,j,k)=static_cast<float>(300.0+0.125*k);
            }
    tmv::TurbulenceConfig cfg;
    cfg.stability_correction=tmv::kLillyRichardsonModel;
    cfg.Cs=0.2; cfg.Pr_t=cfg.Sc_t=0.25;
    tmv::TurbulenceStateView state;
    state.u=&u;state.v=&v;state.w=&w;state.theta=&theta;state.rho=&rho;state.qv=&qv;
    state.NR=state.NTH=state.NZ=n;state.c_grid=true;
    state.periodic_x=state.periodic_y=true;
    state.open_lateral={false,false,false,false};
    std::vector<double> base(n,1.0);state.rho0=&base;
    tmv::SmagorinskyScheme scheme;
    scheme.initialize(cfg);
    tmv::TurbulenceTendencies tendencies;
    tmv::TurbulenceDiagnostics diagnostics;
    scheme.compute(cfg,grid,state,tendencies,&diagnostics);
    const double n2=tmv::turbulence_constants::g/(300.0+0.125*4)*0.125/100.0;
    const double expected=4.0e2*std::sqrt(0.0004-n2/0.25);
    CHECK(diagnostics.nu_t(4,4,4)==Approx(expected).epsilon(2e-7));
    CHECK(diagnostics.K_theta(4,4,4)==Approx(expected/0.25).epsilon(2e-7));
    long double heat=0.0,px=0.0;
    for (int i=1;i<n-1;++i)
        for (int j=1;j<n-1;++j)
            for (int k=0;k<n;++k)
            {
                heat+=tendencies.dthetadt_sgs(i,j,k)*1e6;
                px+=tendencies.dudt_sgs(i,j,k)*1e6;
            }
    CHECK(std::abs(heat)<0.001);
    CHECK(std::abs(px)<0.02);
    const Field3D held=tendencies.dudt_sgs;
    theta(4,4,4)=std::numeric_limits<float>::quiet_NaN();
    REQUIRE_THROWS(scheme.compute(cfg,grid,state,tendencies,&diagnostics));
    CHECK(std::equal(held.data(),held.data()+held.size(),tendencies.dudt_sgs.data()));
    theta.fill(300.0f);
    cfg.mode="horizontal_only";
    REQUIRE_THROWS(scheme.compute(cfg,grid,state,tendencies,&diagnostics));
    cfg.mode="3d";
    u.fill(0.0f);
    for (int i=0;i<n;++i)
        for (int j=0;j<n;++j)
            for (int k=0;k<n;++k) theta(i,j,k)=static_cast<float>(300.0-0.125*k);
    scheme.compute(cfg,grid,state,tendencies,&diagnostics);
    const double unstable_n2=-tmv::turbulence_constants::g/299.5*0.125/100.0;
    CHECK(diagnostics.nu_t(4,4,4)==Approx(400.0*std::sqrt(-unstable_n2/0.25)).epsilon(2e-7));
}
