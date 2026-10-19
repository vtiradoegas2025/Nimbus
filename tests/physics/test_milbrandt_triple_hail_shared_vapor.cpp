#include "catch2/catch.hpp"
#include "support/milbrandt_hail_vapor_reference.hpp"
#include "core/runtime/simulation_state.hpp"

#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
using namespace tmv::milbrandt_detail;
namespace reference=milbrandt_hail_vapor_reference;

HailLiquidSurfaceInput vapor_source(double t=268.15,float qv=.004f,float qc=.03f,bool snow=true)
{
    HailLiquidSurfaceInput in;
    auto& h=in.before.hail;
    h.thermal={static_cast<float>(85000/(287*t)),85000.f,static_cast<float>(t/std::pow(.85,287./1004)),
        {qv,qc,.001f,.0002f,snow?.0005f:0.f,0.f,.001f}};
    in.before.cloud_number=qc>0?1e8f:0.f;
    h.rain_number=4000;h.hail_number=20;
    h.rain_reflectivity=static_cast<float>(5.6L*h.thermal.water[2]*h.thermal.water[2]/(523.598775598L*523.598775598L*h.rain_number));
    h.hail_reflectivity=static_cast<float>(reference::frozen::spectral::population(3,h.thermal.water[6],20,2).reflectivity_per_kg);
    for(std::size_t i=0;i<2;++i)
    {
        const float q=h.thermal.water[i+3];
        in.before.ice_snow_number[i]=q>0?(i==0?5000.f:100.f):0.f;
        in.before.ice_snow_reflectivity[i]=q>0?static_cast<float>(reference::frozen::spectral::population(i,q,in.before.ice_snow_number[i],2).reflectivity_per_kg):0.f;
    }
    in.duration_s=.01;in.generation=1;in.joint_vapor=true;
    return in;
}

HailLiquidSurfaceTransaction vapor_prepared(const HailLiquidSurfaceInput& in)
{
    HailLiquidSurfaceTransaction tx;std::string error;
    const bool ok=tx.prepare(in,error);INFO(error);REQUIRE(ok);return tx;
}

void vapor_near(double value,long double expected,long double floor=1e-23L)
{
    CAPTURE(value,expected);CHECK(std::abs(value-expected)<3e-8L*std::max(std::abs(expected),floor));
}

reference::State vapor_ideal(const HailLiquidSurfaceTransaction& tx)
{
    const auto q=tx.energy_input().accepted_water;const auto r=tx.receipt();
    return {q[0],q[1],r.cloud_number_after,q[2],r.rain_number_after,r.rain_reflectivity_after,q[6],r.hail_number_after,r.hail_reflectivity_after,
        q[3],r.ice_snow_number_after[0],r.ice_snow_reflectivity_after[0],q[4],r.ice_snow_number_after[1],r.ice_snow_reflectivity_after[1]};
}

double identity_reference(const SourceEosEnergyState& before)
{
    SourceEosEnergyInput in;in.before=before;std::copy(before.water.begin(),before.water.end(),in.accepted_water.begin());in.generation=1;in.duration_s=1;
    SourceEosEnergyTransaction tx;std::string error;REQUIRE(tx.prepare(in,error));return tx.receipt().temperature_eos_reference;
}

void check_vapor(const HailLiquidSurfaceInput& in,const HailLiquidSurfaceTransaction& tx,bool quadrature=false)
{
    const auto expected=reference::evaluate(in,quadrature);REQUIRE(expected.equilibrium_valid);
    const auto r=tx.receipt();const auto energy=tx.energy_receipt();const auto ideal=vapor_ideal(tx);
    const auto c=reference::frozen::cell(in.before);
    vapor_near(r.shared_vapor.surface_proposal,expected.rates.liquid.vapor);
    vapor_near(r.shared_vapor.snow_proposal,expected.raw_snow);
    vapor_near(r.shared_vapor.common_factor,expected.factor);
    vapor_near(r.surface_vapor_transfer,expected.vapor);
    vapor_near(r.shared_vapor.snow_transfer,expected.deposition);
    vapor_near(r.melted_hail_mass,expected.melted);
    vapor_near(r.frozen_liquid_mass,expected.frozen);
    vapor_near(r.shed_liquid_mass,expected.shed);
    vapor_near(r.cloud_equilibrium.cloud_transfer,expected.cloud_transfer);
    for(std::size_t i=0;i<ideal.size();++i)vapor_near(static_cast<double>(ideal[i]),expected.endpoint[i]);
    CHECK(std::abs(energy.temperature_eos_target-expected.temperature)<1e-8L);
    const long double tc=c.temperature-273.15L,C=r.surface_vapor_transfer,D=r.shared_vapor.snow_transfer,A=r.cloud_equilibrium.cloud_transfer;
    const long double L=r.cloud_collected_mass+r.rain_collected_mass,F=r.collected_frozen_mass;
    CHECK(std::abs(r.conductive_heat+2500000.L*C+4186.L*tc*L+2106.L*tc*F+
        334000.L*r.frozen_liquid_mass-(334000.L-2106.L*tc)*r.melted_hail_mass)<1e-10L);
    const long double condensed=std::max(0.L,C),evaporated=std::max(0.L,-C);
    const long double remaining_cp=1004+1850.L*(c.state[0]-condensed-D-A)+4186.L*(c.state[1]+c.state[3]-L+A)+
        2106.L*(c.state[6]+c.state[9]+c.state[12]-r.melted_hail_mass-F+D);
    const long double phase_at_entry=D*((2106-1850)*tc-2834000)+A*((4186-1850)*tc-2500000);
    const long double target=energy.temperature_eos_target;
    const long double path=-1850.L*tc*condensed+r.conductive_heat+phase_at_entry+(target-273.15L)*
        (4186.L*(r.shed_liquid_mass+r.melted_hail_mass)+2106.L*(r.frozen_liquid_mass+F)+1850.L*evaporated)+
        remaining_cp*(target-c.temperature);
    const long double gas_work=(287.L+461.5L*ideal[0])*target-(287.L+461.5L*c.state[0])*c.temperature;
    CHECK(std::abs(path-gas_work-energy.reference_projection_work-energy.ideal_phase_energy_change)<2e-7L);
    CHECK(std::abs(reference::frozen::inventory(target,ideal)-reference::frozen::inventory(energy.temperature_eos_reference,c.state))<2e-7L);
    const auto end=reference::frozen::cell(tx.eos_endpoint());
    CHECK(std::abs(reference::frozen::inventory(end.temperature,end.state)-reference::frozen::inventory(c.temperature,c.state)-energy.committed_internal_energy_change)<2e-7L);
    CHECK(std::isfinite(r.stored_cloud_saturation_residual));
    CHECK(std::abs(r.stored_cloud_saturation_residual-(end.state[0]-reference::cloud::saturation(end.temperature,end.pressure)))<1e-15L);
    if(!r.cloud_equilibrium.cloud_exhausted)CHECK(std::abs(ideal[0]-reference::cloud::saturation(target,in.before.hail.thermal.density*287.L*target))<1e-14L);
    long double water=0;for(std::size_t i:{0,1,3,6,9,12})water+=ideal[i]-c.state[i];CHECK(std::abs(water)<3e-17L);
    CHECK(tx.snow_diffusion().source().initial.cloud_mass_per_kg==0);
    CHECK(tx.snow_diffusion().source().initial.frozen[2].mass_per_kg==0);
    CHECK(tx.snow_diffusion().channels()[0].riming_mass_increment_per_kg==0);
    CHECK(tx.cloud_collection().source().initial.cloud_mass_per_kg==in.before.hail.thermal.water[1]);
}
}

TEST_CASE("wet hail shared vapor composes entry fluxes with cold post-EOS cloud equilibrium",
          "[physics][milbrandt][triple_hail_shared_vapor]")
{
    int surface_condensation=0,surface_evaporation=0,cloud_condensation=0,cloud_evaporation=0;
    for(double t:{268.15,253.15})for(float qv:{.003f,.004f,.005f,.006f})for(double velocity:{0.,1.})for(double dt:{.01,.1})
    {
        auto in=vapor_source(t,qv,.03f);in.vertical_velocity_ms=velocity;in.duration_s=dt;
        CAPTURE(t,qv,velocity,dt);
        const auto tx=vapor_prepared(in);check_vapor(in,tx);
        surface_condensation+=tx.receipt().surface_vapor_transfer>0;surface_evaporation+=tx.receipt().surface_vapor_transfer<0;
        cloud_condensation+=tx.receipt().cloud_equilibrium.cloud_transfer>0;cloud_evaporation+=tx.receipt().cloud_equilibrium.cloud_transfer<0;
        CHECK(tx.receipt().shared_vapor.common_factor==1);
    }
    CHECK(surface_condensation>0);CHECK(surface_evaporation>0);CHECK(cloud_condensation>0);CHECK(cloud_evaporation>0);
    const auto in=vapor_source();check_vapor(in,vapor_prepared(in),true);
}

TEST_CASE("shared original vapor admission never spends signed surface returns",
          "[physics][milbrandt][triple_hail_shared_vapor]")
{
    for(double q:{.0001,.003,.01})for(double h:{-.02,0.,.002,.04})for(double snow:{0.,.001,.03})
    {
        HailSharedVaporAdmission a;std::string error;REQUIRE(admit_hail_shared_vapor(q,h,snow,a,error));
        const long double demand=std::max(0.L,static_cast<long double>(h))+snow;
        const long double factor=demand>q?q/demand:1;
        CHECK(std::abs(a.common_factor-factor)<=8*std::numeric_limits<double>::epsilon());
        CHECK(a.snow_transfer<=q-std::max(0.,a.surface_transfer));
        CHECK(a.original_vapor_remaining==q-std::max(0.,a.surface_transfer)-a.snow_transfer);
        if(h<0)CHECK(a.surface_transfer==h);
        if(snow==0)CHECK(a.snow_transfer==0);
        if(h<0&&snow>q){CHECK(a.snow_transfer<=q);CHECK(a.common_factor<1);}
    }
    for(int bad=0;bad<5;++bad)
    {
        HailSharedVaporAdmission a;a.common_factor=42;std::string error;
        CHECK_FALSE(admit_hail_shared_vapor(bad==0?-1:bad==1?std::numeric_limits<double>::quiet_NaN():bad==4?0:.001,
            bad==2?std::numeric_limits<double>::infinity():.002,bad==3?-.001:.001,a,error));
        CHECK(a.common_factor==42);
    }
}

TEST_CASE("cold cloud equilibrium exhausts only original post-collection cloud and rejects returned vapor condensation",
          "[physics][milbrandt][triple_hail_shared_vapor]")
{
    auto in=vapor_source(268.15,.001f,.001f,false);
    auto base=in.before.hail.thermal;const double tref=identity_reference(base);
    std::array<double,7> q;std::copy(base.water.begin(),base.water.end(),q.begin());
    const double original_cloud=q[1];q[1]*=.1;q[2]+=original_cloud-q[1];
    ColdCloudEosEquilibrium result;std::string error;
    REQUIRE(solve_cold_cloud_eos_equilibrium(base,q,tref,q[1],1e7,q[0],0,result,error));
    CHECK(result.cloud_exhausted);CHECK(result.cloud_transfer==-q[1]);CHECK(result.cloud_number==0);CHECK(result.saturation_residual<0);
    CHECK(-result.cloud_transfer<original_cloud);
    auto returned=vapor_source(268.15,.001f,.03f,false).before.hail.thermal;
    const double returned_t=identity_reference(returned);
    std::copy(returned.water.begin(),returned.water.end(),q.begin());q[0]+=.005;q[1]-=.005;
    result.cloud_transfer=42;
    CHECK_FALSE(solve_cold_cloud_eos_equilibrium(returned,q,returned_t,q[1],1e8,q[0]-.005,0,result,error));
    CHECK(error.find("unowned vapor")!=std::string::npos);CHECK(result.cloud_transfer==42);
}

TEST_CASE("cold cloud EOS bracket inverts independent inventory with monotone saturation residual",
          "[physics][milbrandt][triple_hail_shared_vapor]")
{
    const auto in=vapor_source();const auto before=in.before.hail.thermal;const auto c=reference::frozen::cell(in.before);
    const long double energy=reference::frozen::inventory(reference::reference_temperature(before),c.state);
    double previous_t=0,previous_residual=1;
    for(double transfer:{-.0001,-.00005,0.,.00005,.0001})
    {
        auto q=c.state;q[0]-=transfer;q[1]+=transfer;
        const long double expected_t=reference::temperature(energy,q);
        SourceEosEnergyInput e;e.before=before;std::copy(before.water.begin(),before.water.end(),e.accepted_water.begin());e.accepted_water[0]-=transfer;e.accepted_water[1]+=transfer;e.generation=1;e.duration_s=.01;
        SourceEosEnergyTransaction tx;std::string error;REQUIRE(tx.prepare(e,error));
        CHECK(std::abs(tx.receipt().temperature_eos_target-expected_t)<1e-9L);
        const double residual=static_cast<double>(q[0]-reference::cloud::saturation(expected_t,before.density*287.L*expected_t));
        CHECK(tx.receipt().temperature_eos_target>previous_t);CHECK(residual<previous_residual);
        previous_t=tx.receipt().temperature_eos_target;previous_residual=residual;
    }
    const double tref=identity_reference(before);
    for(int mutation=0;mutation<11;++mutation)
    {
        auto b=before;std::array<double,7> q;std::copy(before.water.begin(),before.water.end(),q.begin());double t=tref,n=1e8,v=q[0],w=0;
        if(mutation==0)q[0]=std::numeric_limits<double>::quiet_NaN();
        if(mutation==1)q[2]=-1;
        if(mutation==2)q[3]+=.001;
        if(mutation==3)t=std::numeric_limits<double>::quiet_NaN();
        if(mutation==4)t=std::nextafter(t,400.);
        if(mutation==5)b.density=0;
        if(mutation==6)b.theta=std::numeric_limits<float>::infinity();
        if(mutation==7)n=-1;
        if(mutation==8)v=-1;
        if(mutation==9)w=std::numeric_limits<double>::quiet_NaN();
        if(mutation==10)q[4]=std::numeric_limits<double>::max();
        ColdCloudEosEquilibrium result;result.cloud_transfer=42;std::string error;
        CHECK_FALSE(solve_cold_cloud_eos_equilibrium(b,q,t,q[1],n,v,w,result,error));CHECK(result.cloud_transfer==42);
    }
}

TEST_CASE("joint wet vapor EOS ownership is atomic and the zero snow competitor recovers the source surface flux",
          "[physics][milbrandt][triple_hail_shared_vapor]")
{
    for(int threads:{1,4})
    {
#ifdef _OPENMP
        omp_set_num_threads(threads);
#endif
        auto in=vapor_source();auto tx=vapor_prepared(in);const auto staged=tx.source_endpoint();std::string error;
        auto foreign=staged;foreign.cloud_number=std::nextafter(foreign.cloud_number,1e9f);
        CHECK_FALSE(tx.accept_source(in.before,foreign,in.generation,error));CHECK_FALSE(tx.accept_source(in.before,staged,in.generation+1,error));
        REQUIRE(tx.accept_source(in.before,staged,in.generation,error));
        SimulationStorageState storage{{5,3,4,4.,5.,.5,6.}};storage.grid.geometry.initialize(5,3,4,4.,5.,6.,CoordinateSystem::Cartesian,StaggerType::CGrid);
        auto& f=storage.fields;for(Field3D* field:{&f.u,&f.v,&f.w,&f.rho,&f.p,&f.theta,&f.qv,&f.qc,&f.qr,&f.qi,&f.qs,&f.qg,&f.qh,&f.tke})field->resize(5,3,4,0.f);
        f.rho.fill(staged.hail.thermal.density);f.p.fill(staged.hail.thermal.pressure);f.theta.fill(staged.hail.thermal.theta);
        const std::array<Field3D*,7> water{&f.qv,&f.qc,&f.qr,&f.qi,&f.qs,&f.qg,&f.qh};for(std::size_t i=0;i<7;++i)water[i]->fill(staged.hail.thermal.water[i]);
        REQUIRE(tmv::dynamics_eos::apply_pressure_projection(make_simulation_state_view(storage)).complete);
        auto after=staged;after.hail.thermal.pressure=f.p(1,1,0);
        auto bad_spectrum=after;bad_spectrum.ice_snow_reflectivity[1]=std::nextafter(bad_spectrum.ice_snow_reflectivity[1],1.f);
        CHECK_FALSE(tx.accept_eos(staged,bad_spectrum,in.generation,error));CHECK(tx.stage()==SourceEosEnergyTransaction::Stage::SourceCommitted);
        auto bad_thermal=after;bad_thermal.hail.thermal.theta=std::nextafter(bad_thermal.hail.thermal.theta,400.f);
        CHECK_FALSE(tx.accept_eos(staged,bad_thermal,in.generation,error));CHECK(tx.stage()==SourceEosEnergyTransaction::Stage::SourceCommitted);
        REQUIRE(tx.accept_eos(staged,after,in.generation,error));
        CHECK_FALSE(tx.accept_eos(staged,after,in.generation,error));
    }
    auto no_snow=vapor_source(268.15,.004f,.03f,false);const auto joint=vapor_prepared(no_snow);
    no_snow.joint_vapor=false;const auto surface=vapor_prepared(no_snow);
    CHECK(joint.receipt().shared_vapor.snow_transfer==0);CHECK(joint.receipt().shared_vapor.common_factor==1);
    CHECK(joint.receipt().surface_vapor_transfer==surface.receipt().surface_vapor_transfer);
    CHECK(joint.receipt().melted_hail_mass==surface.receipt().melted_hail_mass);CHECK(joint.receipt().frozen_liquid_mass==surface.receipt().frozen_liquid_mass);
    for(int mutation=0;mutation<4;++mutation)
    {
        auto bad=vapor_source();if(mutation==0)bad=vapor_source(268.15,.01f,.03f);if(mutation==1)bad=vapor_source(268.15,.001f,.03f);
        if(mutation==2)bad.competing_vapor_transfer=1e-8;if(mutation==3)bad.dry_hail_diffusion=1e-8;
        HailLiquidSurfaceTransaction tx;std::string error;CHECK_FALSE(tx.prepare(bad,error));CHECK(tx.stage()==SourceEosEnergyTransaction::Stage::Empty);CHECK(tx.receipt().shared_vapor.snow_transfer==0);
        REQUIRE(tx.prepare(vapor_source(),error));CHECK(tx.stage()==SourceEosEnergyTransaction::Stage::Prepared);CHECK(tx.receipt().shared_vapor.snow_transfer>0);
    }
    std::array<HailLiquidSurfaceTransaction,4> parallel;std::array<std::thread,4> workers;std::array<bool,4> success{};
    for(std::size_t i=0;i<4;++i)workers[i]=std::thread([&,i]{std::string error;success[i]=parallel[i].prepare(vapor_source(),error);});
    for(auto& worker:workers)worker.join();for(std::size_t i=0;i<4;++i){REQUIRE(success[i]);CHECK(parallel[i].receipt().cloud_equilibrium.cloud_transfer==parallel[0].receipt().cloud_equilibrium.cloud_transfer);CHECK(parallel[i].eos_endpoint().hail.thermal.theta==parallel[0].eos_endpoint().hail.thermal.theta);}
}
