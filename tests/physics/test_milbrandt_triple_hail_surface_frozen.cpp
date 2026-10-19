#include "catch2/catch.hpp"
#include "support/milbrandt_hail_frozen_reference.hpp"
#include "core/runtime/simulation_state.hpp"

#include <iomanip>
#include <iostream>
#include <thread>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace
{
using namespace tmv::milbrandt_detail;
namespace reference=milbrandt_hail_frozen_reference;

template<std::size_t... I>
ColdParameters frozen_parameters(std::index_sequence<I...>)
{
    return {default_cold_parameters[I]...,true};
}

HailLiquidSurfaceInput frozen_source(double t=268.15,float vapor=.003f,float cloud=.03f,
                                    long double shape=2,bool ice=true,bool snow=true)
{
    HailLiquidSurfaceInput in;
    auto& h=in.before.hail;
    h.thermal={static_cast<float>(85000/(287*t)),85000.f,static_cast<float>(t/std::pow(.85,287./1004)),
        {vapor,cloud,.001f,ice?.0002f:0.f,snow?.0005f:0.f,0.f,.001f}};
    in.before.cloud_number=cloud>0?1e8f:0;
    h.rain_number=4000;
    h.rain_reflectivity=static_cast<float>(5.6L*h.thermal.water[2]*h.thermal.water[2]/(523.598775598L*523.598775598L*h.rain_number));
    h.hail_number=20;
    h.hail_reflectivity=static_cast<float>(reference::spectral::population(3,h.thermal.water[6],20,shape).reflectivity_per_kg);
    for(std::size_t i=0;i<2;++i)
    {
        const float q=h.thermal.water[3+i];
        in.before.ice_snow_number[i]=q>0?(i==0?5000.f:100.f):0.f;
        in.before.ice_snow_reflectivity[i]=q>0?static_cast<float>(reference::spectral::population(
            i,q,in.before.ice_snow_number[i],shape).reflectivity_per_kg):0.f;
    }
    in.generation=1; in.duration_s=.01;
    return in;
}

HailLiquidSurfaceTransaction frozen_prepared(const HailLiquidSurfaceInput& in)
{
    HailLiquidSurfaceTransaction tx;
    std::string error;
    const bool ok=tx.prepare(in,error);
    INFO(error); REQUIRE(ok);
    return tx;
}

void frozen_near(double actual,long double expected,long double floor=1e-23L)
{
    CAPTURE(actual,expected);
    CHECK(std::abs(actual-expected)<3e-8L*std::max(floor,std::abs(expected)));
}

reference::State frozen_ideal(const HailLiquidSurfaceTransaction& tx)
{
    const auto q=tx.energy_input().accepted_water;
    const auto r=tx.receipt();
    return {q[0],q[1],r.cloud_number_after,q[2],r.rain_number_after,r.rain_reflectivity_after,
        q[6],r.hail_number_after,r.hail_reflectivity_after,q[3],r.ice_snow_number_after[0],r.ice_snow_reflectivity_after[0],
        q[4],r.ice_snow_number_after[1],r.ice_snow_reflectivity_after[1]};
}

void check_frozen(const HailLiquidSurfaceInput& in,const HailLiquidSurfaceTransaction& tx,bool quadrature=false)
{
    const auto c=reference::cell(in.before);
    const auto e=reference::evaluate(c,in.duration_s,quadrature);
    const auto r=tx.receipt();
    frozen_near(r.collected_frozen_mass,e.collected_frozen);
    frozen_near(r.dry_collected_frozen_mass,e.dry_collected_frozen);
    frozen_near(r.collected_frozen_sensible_heat,e.frozen_sensible);
    frozen_near(r.frozen_liquid_mass,e.frozen);
    frozen_near(r.melted_hail_mass,e.melted);
    frozen_near(r.shed_liquid_mass,e.shed);
    frozen_near(r.dry_full_freezing_heat,e.dry_heat);
    for(std::size_t i=0;i<5;++i)
    {
        frozen_near(tx.frozen_collection().channels()[i].admitted_mass_increment_per_kg,e.wet_frozen.channels[i].mass);
        frozen_near(tx.frozen_collection().channels()[i].admitted_number_increment_per_kg,e.wet_frozen.channels[i].number);
    }
    const auto ideal=frozen_ideal(tx);
    for(std::size_t i=0;i<ideal.size();++i) frozen_near(static_cast<double>(ideal[i]),c.state[i]+e.increments[i]);
    const long double tc=c.temperature-273.15L;
    const long double surface=r.conductive_heat+2500000.L*r.surface_vapor_transfer+
        4186.L*tc*(r.cloud_collected_mass+r.rain_collected_mass)+2106.L*tc*r.collected_frozen_mass+
        334000.L*r.frozen_liquid_mass-(334000.L-2106.L*tc)*r.melted_hail_mass;
    CHECK(std::abs(surface)<1e-10L);
    const auto energy=tx.energy_receipt();
    CHECK(std::abs(reference::inventory(energy.temperature_eos_target,ideal)-
        reference::inventory(energy.temperature_eos_reference,c.state))<2e-7L);
    const auto stored=reference::cell(tx.eos_endpoint());
    CHECK(std::abs(reference::inventory(stored.temperature,stored.state)-reference::inventory(c.temperature,c.state)-
        energy.committed_internal_energy_change)<2e-7L);
    const long double condensed=std::max(0.L,static_cast<long double>(r.surface_vapor_transfer));
    const long double evaporated=std::max(0.L,-static_cast<long double>(r.surface_vapor_transfer));
    const long double liquid=r.cloud_collected_mass+r.rain_collected_mass;
    const long double remaining_cp=1004+1850.L*(c.state[0]-condensed)+4186.L*(c.state[1]+c.state[3]-liquid)+
        2106.L*(c.state[6]+c.state[9]+c.state[12]-r.melted_hail_mass-r.collected_frozen_mass);
    const long double target=energy.temperature_eos_target;
    const long double path=-1850.L*tc*condensed+r.conductive_heat+(target-273.15L)*
        (4186.L*(r.shed_liquid_mass+r.melted_hail_mass)+2106.L*(r.frozen_liquid_mass+r.collected_frozen_mass)+1850.L*evaporated)+
        remaining_cp*(target-c.temperature);
    const long double gas_work=(287.L+461.5L*ideal[0])*target-(287.L+461.5L*c.state[0])*c.temperature;
    CHECK(std::abs(path-gas_work-energy.reference_projection_work-energy.ideal_phase_energy_change)<2e-7L);
    long double water=0;
    for(std::size_t i:{0,1,3,6,9,12}) water+=ideal[i]-c.state[i];
    CHECK(std::abs(water)<3e-17L);
}
}

TEST_CASE("wet hail composes full frozen admission with collected-solid sensible heat and original moments",
          "[physics][milbrandt][triple_hail_surface_frozen]")
{
    int melts=0,freezes=0,evaporates=0,rejections=0;
    for(double t:{272.15,268.15,253.15})
    for(float vapor:{.003f,.015f})
    for(float cloud:{.001f,.03f})
    for(long double shape:{2.L,20.L,150.L})
    for(int presence:{1,2,3})
    for(bool hail_option:{false,true})
    {
        auto in=frozen_source(t,vapor,cloud,shape,presence&1,presence&2);
        in.hail_processes=hail_option;
        CAPTURE(t,vapor,cloud,shape,presence,hail_option);
        const auto ref=reference::evaluate(reference::cell(in.before),in.duration_s);
        if(ref.shed<=0 || ref.liquid.dry_cloud+ref.liquid.rain+ref.liquid.vapor<=0 || ref.dry_heat<=0)
        {
            HailLiquidSurfaceTransaction tx; std::string error;
            CHECK_FALSE(tx.prepare(in,error)); CHECK(tx.stage()==SourceEosEnergyTransaction::Stage::Empty);
            ++rejections; continue;
        }
        const auto tx=frozen_prepared(in);
        check_frozen(in,tx);
        CHECK(tx.frozen_collection().wet_hail_collection());
        CHECK_FALSE(tx.dry_frozen_collection().wet_hail_collection());
        CHECK(tx.frozen_collection().source().initial.frozen[3].mass_per_kg==in.before.hail.thermal.water[6]);
        CHECK(tx.frozen_collection().hail_processes()==hail_option);
        melts+=tx.receipt().melted_hail_mass>0;
        freezes+=tx.receipt().frozen_liquid_mass>0;
        evaporates+=tx.receipt().surface_vapor_transfer<0;
    }
    CHECK(melts>0); CHECK(freezes>0); CHECK(evaporates>0); CHECK(rejections>0);
    const auto in=frozen_source();
    check_frozen(in,frozen_prepared(in),true);
}

TEST_CASE("wet hail frozen composition preserves internal snow destinations without financing new collection",
          "[physics][milbrandt][triple_hail_surface_frozen]")
{
    auto in=frozen_source();
    const auto tx=frozen_prepared(in);
    const auto events=tx.frozen_collection().channels();
    const auto r=tx.receipt();
    REQUIRE(events[0].admitted_mass_increment_per_kg>0);
    REQUIRE(events[2].admitted_mass_increment_per_kg>0);
    REQUIRE(events[3].admitted_mass_increment_per_kg>0);
    REQUIRE(events[4].admitted_number_increment_per_kg>0);
    CHECK(events[3].admitted_mass_increment_per_kg<=in.before.hail.thermal.water[4]);
    CHECK(r.frozen_collection.mass_increment_per_kg[1]==events[0].admitted_mass_increment_per_kg-events[3].admitted_mass_increment_per_kg);
    CHECK(r.frozen_collection.number_increment_per_kg[1]==-events[3].admitted_number_increment_per_kg-events[4].admitted_number_increment_per_kg);
    CHECK(r.hail_number_after==in.before.hail.hail_number-r.melted_hail_number);
    CHECK(r.collected_frozen_mass==events[2].admitted_mass_increment_per_kg+events[3].admitted_mass_increment_per_kg);
    auto without=in;
    without.before.hail.thermal.water[3]=without.before.hail.thermal.water[4]=0;
    without.before.ice_snow_number={}; without.before.ice_snow_reflectivity={};
    const auto base=frozen_prepared(without);
    CHECK(r.normalized_ventilation==base.receipt().normalized_ventilation);
    CHECK(r.surface_vapor_transfer==base.receipt().surface_vapor_transfer);
    CHECK(r.frozen_liquid_mass>base.receipt().frozen_liquid_mass);
    auto no_hail=in;
    no_hail.before.hail.thermal.water[6]=no_hail.before.hail.hail_number=no_hail.before.hail.hail_reflectivity=0;
    const auto no_hail_tx=frozen_prepared(no_hail);
    CHECK(no_hail_tx.receipt().collected_frozen_mass==0);
    CHECK(no_hail_tx.receipt().surface_vapor_transfer==0);
    CHECK(no_hail_tx.receipt().frozen_collection.mass_increment_per_kg[1]>0);
    CHECK(no_hail_tx.energy_input().accepted_water[6]==0);
    check_frozen(no_hail,no_hail_tx);
}

TEST_CASE("wet hail rejects a broken gross snow spectrum even when incoming ice repairs its net moments",
          "[physics][milbrandt][triple_hail_surface_frozen]")
{
    auto in=frozen_source();
    in.before.hail.thermal.water[3]=.02f;
    in.before.ice_snow_reflectivity[0]=static_cast<float>(reference::spectral::population(
        0,in.before.hail.thermal.water[3],in.before.ice_snow_number[0],2).reflectivity_per_kg);
    in.duration_s=20;
    const auto c=reference::cell(in.before);
    const auto expected=reference::evaluate(c,in.duration_s);
    const auto events=expected.wet_frozen.channels;
    const long double power=6/2.078L;
    const long double gross_z=c.state[14]*(1-power*events[3].mass/c.state[12]+
        (power-1)*(events[3].number+events[4].number)/c.state[13]);
    REQUIRE(gross_z<0);
    const auto net=expected.wet_frozen.endpoint[1];
    REQUIRE(net[0]>0); REQUIRE(net[1]>0);
    REQUIRE(net[2]>net[1]*std::pow(net[0]/(.1597L*net[1]),power));
    REQUIRE(expected.shed>0); REQUIRE(expected.dry_heat>0);
    HailLiquidSurfaceTransaction tx; std::string error;
    CHECK_FALSE(tx.prepare(in,error));
    CHECK(error.find("gross original snow")!=std::string::npos);
    CHECK(tx.stage()==SourceEosEnergyTransaction::Stage::Empty);
    in.duration_s=.01;
    const auto bounded=frozen_prepared(in);
    const auto donor=bounded.frozen_collection().source();
    FrozenCollectionProducer capped;
    const auto p=frozen_parameters(std::make_index_sequence<25>{});
    REQUIRE(produce_exact_frozen_collection_admission(p,donor,1e5,capped,error,5,nullptr,ColdSpectrumDomain::GraupelAndHail512,true));
    REQUIRE(capped.channels()[0].admitted_mass_increment_per_kg>0);
    REQUIRE(capped.channels()[3].raw_mass_increment_per_kg>donor.initial.frozen[1].mass_per_kg);
    CHECK(capped.channels()[3].admitted_mass_increment_per_kg<=donor.initial.frozen[1].mass_per_kg);
    const auto reference_admission=reference::spectral::evaluate(donor,1e5,false,true);
    frozen_near(capped.channels()[3].admitted_mass_increment_per_kg,reference_admission.channels[3].mass);
    frozen_near(capped.channels()[3].admitted_number_increment_per_kg,reference_admission.channels[3].number);
}

TEST_CASE("wet hail frozen ownership rejects foreign moments and executes the actual EOS",
          "[physics][milbrandt][triple_hail_surface_frozen]")
{
    for(bool melt:{false,true}) for(int threads:{1,4})
    {
#ifdef _OPENMP
        omp_set_num_threads(threads);
#endif
        const auto in=frozen_source(melt?272.15:268.15,melt?.015f:.003f,melt?.001f:.03f);
        auto tx=frozen_prepared(in);
        const auto stage=tx.source_endpoint();
        std::string error;
        CHECK_FALSE(tx.accept_eos(stage,tx.eos_endpoint(),in.generation,error));
        for(std::size_t i=0;i<2;++i)
        {
            auto foreign=stage; foreign.ice_snow_number[i]=std::nextafter(foreign.ice_snow_number[i],1e8f);
            CHECK_FALSE(tx.accept_source(in.before,foreign,in.generation,error));
        }
        CHECK_FALSE(tx.accept_source(in.before,stage,in.generation+1,error));
        REQUIRE(tx.accept_source(in.before,stage,in.generation,error));
        SimulationStorageState storage{{5,3,4,4.,5.,.5,6.}};
        storage.grid.geometry.initialize(5,3,4,4.,5.,6.,CoordinateSystem::Cartesian,StaggerType::CGrid);
        auto& f=storage.fields;
        for(Field3D* field:{&f.u,&f.v,&f.w,&f.rho,&f.p,&f.theta,&f.qv,&f.qc,&f.qr,&f.qi,&f.qs,&f.qg,&f.qh,&f.tke}) field->resize(5,3,4,0.f);
        f.rho.fill(stage.hail.thermal.density);f.p.fill(stage.hail.thermal.pressure);f.theta.fill(stage.hail.thermal.theta);
        const std::array<Field3D*,7> water{&f.qv,&f.qc,&f.qr,&f.qi,&f.qs,&f.qg,&f.qh};
        for(std::size_t i=0;i<7;++i)water[i]->fill(stage.hail.thermal.water[i]);
        REQUIRE(tmv::dynamics_eos::apply_pressure_projection(make_simulation_state_view(storage)).complete);
        auto after=stage; after.hail.thermal.pressure=f.p(1,1,0);
        for(std::size_t i=0;i<2;++i)
        {
            auto foreign=after; foreign.ice_snow_reflectivity[i]=std::nextafter(foreign.ice_snow_reflectivity[i],1.f);
            CHECK_FALSE(tx.accept_eos(stage,foreign,in.generation,error));
            CHECK(tx.stage()==SourceEosEnergyTransaction::Stage::SourceCommitted);
        }
        REQUIRE(tx.accept_eos(stage,after,in.generation,error));
        CHECK_FALSE(tx.accept_eos(stage,after,in.generation,error));
    }
}

TEST_CASE("wet hail cannot use collected frozen mass to satisfy original hail melting demand",
          "[physics][milbrandt][triple_hail_surface_frozen]")
{
    auto in=frozen_source(272.15,.015f,.001f);
    in.before.hail.thermal.water[6]=1e-5f;
    in.before.hail.hail_reflectivity=static_cast<float>(reference::spectral::population(
        3,in.before.hail.thermal.water[6],in.before.hail.hail_number,2).reflectivity_per_kg);
    in.duration_s=30;
    const auto ref=reference::evaluate(reference::cell(in.before),in.duration_s);
    REQUIRE(ref.melted>in.before.hail.thermal.water[6]);
    REQUIRE(ref.melted<in.before.hail.thermal.water[6]+ref.collected_frozen);
    REQUIRE(ref.shed>0); REQUIRE(ref.dry_heat>0);
    HailLiquidSurfaceTransaction tx;
    std::string error;
    CHECK_FALSE(tx.prepare(in,error));
    CHECK(error.find("exhausts original hail")!=std::string::npos);
    CHECK(tx.stage()==SourceEosEnergyTransaction::Stage::Empty);
    CHECK(tx.receipt().melted_hail_mass==0);
}

TEST_CASE("wet hail frozen surface rejects invalid and unowned state atomically",
          "[physics][milbrandt][triple_hail_surface_frozen]")
{
    for(int mutation=0;mutation<10;++mutation)
    {
        auto in=frozen_source();
        if(mutation==0)in.before.ice_snow_number[0]=0;
        if(mutation==1)in.before.ice_snow_reflectivity[1]=0;
        if(mutation==2)in.before.ice_snow_number[1]=std::numeric_limits<float>::quiet_NaN();
        if(mutation==3)in.before.hail.thermal.water[5]=1e-4f;
        if(mutation==4)in.competing_vapor_transfer=1e-8;
        if(mutation==5)in.dry_hail_diffusion=1e-8;
        if(mutation==6)in.duration_s=1e7;
        if(mutation==7)in.duration_s=std::numeric_limits<double>::denorm_min();
        if(mutation==8)in.before.hail.thermal.water[3]=0;
        if(mutation==9)in.before.ice_snow_reflectivity[0]=-1;
        HailLiquidSurfaceTransaction tx; std::string error;
        CHECK_FALSE(tx.prepare(in,error));
        CHECK(tx.stage()==SourceEosEnergyTransaction::Stage::Empty);
        CHECK(tx.receipt().collected_frozen_mass==0);
    }
}

TEST_CASE("wet hail frozen continuation approaches independent coupled integrations in both phase branches",
          "[physics][milbrandt][triple_hail_surface_frozen]")
{
    for(bool melt:{false,true})
    {
        auto initial=frozen_source(melt?272.15:268.15,melt?.015f:.003f,melt?.001f:.03f);
        std::array<HailLiquidSurfaceTransaction,4> parallel;
        std::array<std::thread,4> workers;
        std::array<bool,4> success{};
        for(std::size_t i=0;i<4;++i) workers[i]=std::thread([&,i]{std::string error;success[i]=parallel[i].prepare(initial,error);});
        for(auto& worker:workers)worker.join();
        for(std::size_t i=0;i<4;++i)
        {
            REQUIRE(success[i]);
            CHECK(parallel[i].receipt().collected_frozen_mass==parallel[0].receipt().collected_frozen_mass);
            CHECK(parallel[i].eos_endpoint().ice_snow_reflectivity==parallel[0].eos_endpoint().ice_snow_reflectivity);
        }
        auto context=reference::cell(initial.before);
        const auto initial_tx=frozen_prepared(initial);
        context.temperature=initial_tx.energy_receipt().temperature_eos_reference;
        context.pressure=context.density*287*context.temperature;
        constexpr double duration=20;
        const auto oracle=reference::rk4(context,duration,256),refined=reference::rk4(context,duration,512);
        CHECK(std::abs(oracle[6]-refined[6])<1e-10L);
        std::array<long double,3> errors{};
        for(int level=0;level<3;++level)
        {
            auto next=initial;
            const int steps=10<<level;
            next.duration_s=duration/steps;
            long double accounted=0;
            for(int step=0;step<steps;++step)
            {
                auto tx=frozen_prepared(next); std::string error;
                REQUIRE(tx.accept_source(next.before,tx.source_endpoint(),next.generation,error));
                REQUIRE(tx.accept_eos(tx.source_endpoint(),tx.eos_endpoint(),next.generation,error));
                accounted+=tx.energy_receipt().committed_internal_energy_change;
                next.before=tx.eos_endpoint();++next.generation;
            }
            const auto end=reference::cell(next.before),before=reference::cell(initial.before);
            CHECK(std::abs(reference::inventory(end.temperature,end.state)-reference::inventory(before.temperature,before.state)-accounted)<1e-6L);
            errors[level]=std::abs(end.state[6]-oracle[6]);
        }
        std::cout<<std::setprecision(17)<<"[Hail frozen refinement] melt="<<melt<<" errors="<<errors[0]<<','<<errors[1]<<','<<errors[2]<<std::endl;
        CHECK(errors[1]<.8L*errors[0]);CHECK(errors[2]<.8L*errors[1]);
    }
}
