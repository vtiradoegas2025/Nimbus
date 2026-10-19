#include "catch2/catch.hpp"
#include "support/milbrandt_hail_liquid_reference.hpp"
#include "core/runtime/simulation_state.hpp"

#include <iomanip>
#include <iostream>
#include <thread>

namespace
{
using namespace tmv::milbrandt_detail;
namespace reference = milbrandt_hail_liquid_reference;

HailLiquidSurfaceInput source(double temperature = 268.15, float vapor = .015f, float cloud = .001f,
                             long double shape = 2, bool rain = true)
{
    HailLiquidSurfaceInput input;
    auto& h = input.before.hail;
    h.thermal = {static_cast<float>(85000 / (287 * temperature)), 85000.f,
        static_cast<float>(temperature / std::pow(.85, 287.0 / 1004)), {vapor, cloud, rain ? .001f : 0.f, 0.f, 0.f, 0.f, .001f}};
    input.before.cloud_number = cloud > 0 ? 1e8f : 0.f;
    h.hail_number = 20.f;
    h.hail_reflectivity = static_cast<float>(reference::spectral::population(3, h.thermal.water[6], h.hail_number, shape).reflectivity_per_kg);
    if (rain)
    {
        h.rain_number = 4000.f;
        h.rain_reflectivity = static_cast<float>(5.6L * h.thermal.water[2] * h.thermal.water[2] /
            (523.598775598L * 523.598775598L * h.rain_number));
    }
    input.generation = 1;
    input.duration_s = .01;
    return input;
}

HailLiquidSurfaceTransaction prepared(const HailLiquidSurfaceInput& input)
{
    HailLiquidSurfaceTransaction transaction;
    std::string error;
    const bool ok = transaction.prepare(input, error);
    INFO(error);
    REQUIRE(ok);
    return transaction;
}

void near(double actual, long double expected, long double floor = 1e-23L)
{
    CAPTURE(actual, expected);
    CHECK(std::abs(actual - expected) < 3e-8L * std::max(std::abs(expected), floor));
}

reference::State ideal_state(const HailLiquidSurfaceTransaction& transaction)
{
    const auto q = transaction.energy_input().accepted_water;
    const auto r = transaction.receipt();
    return {q[0],q[1],r.cloud_number_after,q[2],r.rain_number_after,r.rain_reflectivity_after,
        q[6],r.hail_number_after,r.hail_reflectivity_after};
}

void check(const HailLiquidSurfaceInput& input, const HailLiquidSurfaceTransaction& transaction)
{
    const auto c = reference::cell(input.before);
    const auto expected = reference::evaluate(c, input.duration_s);
    const auto r = transaction.receipt();
    near(r.cloud_collected_mass, expected.cloud);
    near(r.cloud_collected_number, expected.cloud_number);
    near(r.dry_liquid_mass, expected.dry_cloud + expected.rain);
    near(r.rain_collected_mass, expected.rain);
    near(r.rain_collected_number, expected.rain_number);
    near(r.rain_reflectivity_removed, expected.rain_z_removed);
    near(r.surface_vapor_transfer, expected.vapor);
    near(r.conductive_heat, expected.conduction);
    near(r.liquid_sensible_heat, expected.sensible);
    near(r.dry_full_freezing_heat, expected.dry_heat);
    near(r.frozen_liquid_mass, expected.frozen);
    near(r.melted_hail_mass, expected.melted);
    near(r.shed_liquid_mass, expected.shed);
    near(r.shed_rain_number, expected.shed_number);
    near(r.shed_rain_reflectivity, expected.shed_z);
    near(r.melt_rain_reflectivity, expected.melt_z);
    const auto ideal = ideal_state(transaction);
    for (std::size_t i = 0; i < ideal.size(); ++i) near(static_cast<double>(ideal[i]), c.state[i] + expected.increments[i]);
    const long double tc = c.temperature - 273.15L;
    const long double surface_residual = r.conductive_heat + 2500000.L * r.surface_vapor_transfer +
        4186.L * tc * (r.cloud_collected_mass + r.rain_collected_mass) + 334000.L * r.frozen_liquid_mass -
        (334000.L - 2106.L * tc) * r.melted_hail_mass;
    CHECK(std::abs(surface_residual) < 1e-10L);
    const auto e = transaction.energy_receipt();
    CHECK(std::abs(reference::inventory(e.temperature_eos_target, ideal) -
        reference::inventory(e.temperature_eos_reference, c.state)) < 2e-7L);
    const auto stored = reference::cell(transaction.eos_endpoint());
    CHECK(std::abs(reference::inventory(stored.temperature, stored.state) - reference::inventory(c.temperature, c.state) -
        e.committed_internal_energy_change) < 2e-7L);
    const long double condensed = std::max(0.L, static_cast<long double>(r.surface_vapor_transfer));
    const long double evaporated = std::max(0.L, -static_cast<long double>(r.surface_vapor_transfer));
    const long double liquid = r.cloud_collected_mass + r.rain_collected_mass;
    const long double remaining_cp = 1004 + 1850.L * (c.state[0] - condensed) +
        4186.L * (c.state[1] + c.state[3] - liquid) + 2106.L * (c.state[6] - r.melted_hail_mass);
    const long double target = e.temperature_eos_target;
    const long double path = -1850.L * tc * condensed + r.conductive_heat +
        (target - 273.15L) * (4186.L * (r.shed_liquid_mass + r.melted_hail_mass) +
                              2106.L * r.frozen_liquid_mass + 1850.L * evaporated) +
        remaining_cp * (target - c.temperature);
    const long double gas_work = (287.L + 461.5L * ideal[0]) * target - (287.L + 461.5L * c.state[0]) * c.temperature;
    CHECK(std::abs(path - gas_work - e.reference_projection_work - e.ideal_phase_energy_change) < 2e-7L);
    const long double water = ideal[0] + ideal[1] + ideal[3] + ideal[6] - c.state[0] - c.state[1] - c.state[3] - c.state[6];
    CHECK(std::abs(water) < 3e-17L);
}
}

TEST_CASE("wet hail liquid collection composes admitted production channels with independent heat and moment references",
          "[physics][milbrandt][triple_hail_surface_liquid]")
{
    int melting = 0, freezing = 0, evaporation = 0, dry_rejections = 0;
    for (double t : {272.15,268.15,253.15})
    for (float vapor : {.003f,.015f})
    for (float cloud : {.001f,.03f})
    for (long double shape : {2.L,20.L,350.L})
    for (bool hail_processes : {false,true})
    {
        auto input = source(t,vapor,cloud,shape);
        input.hail_processes = hail_processes;
        CAPTURE(t,vapor,cloud,shape,hail_processes);
        const auto expected = reference::evaluate(reference::cell(input.before), input.duration_s);
        if (expected.shed <= 0.0 || expected.dry_cloud + expected.rain + expected.vapor <= 0.0 || expected.dry_heat <= 0.0)
        {
            HailLiquidSurfaceTransaction dry;
            std::string error;
            CHECK_FALSE(dry.prepare(input,error));
            CHECK(dry.stage()==SourceEosEnergyTransaction::Stage::Empty);
            ++dry_rejections;
            continue;
        }
        const auto transaction = prepared(input);
        check(input, transaction);
        const auto r = transaction.receipt();
        melting += r.melted_hail_mass > 0;
        freezing += r.frozen_liquid_mass > 0;
        evaporation += r.surface_vapor_transfer < 0;
        CHECK(transaction.cloud_collection().wet_hail_collection());
        CHECK_FALSE(transaction.dry_cloud_collection().wet_hail_collection());
        CHECK(transaction.cloud_collection().source().initial.cloud_number_per_kg == input.before.cloud_number);
        CHECK(transaction.rain_collection().source().rain.mass_per_kg == input.before.hail.thermal.water[2]);
        CHECK(transaction.rain_collection().source().frozen.initial.frozen[3].reflectivity_per_kg == input.before.hail.hail_reflectivity);
        CHECK(transaction.cloud_collection().hail_processes() == hail_processes);
        CHECK(r.shed_liquid_mass > 0);
        CHECK(r.dry_full_freezing_heat > 0);
    }
    CHECK(melting > 0); CHECK(freezing > 0); CHECK(evaporation > 0); CHECK(dry_rejections > 0);
    const auto input = source(268.15,.003f,.03f,20);
    const auto actual = prepared(input);
    const auto quadrature = reference::evaluate(reference::cell(input.before),input.duration_s,true);
    near(actual.receipt().cloud_collected_mass,quadrature.cloud);
    near(actual.receipt().rain_collected_mass,quadrature.rain);
    near(actual.receipt().surface_vapor_transfer,quadrature.vapor);
}

TEST_CASE("wet hail liquid admission uses capped cloud mass and rejects efficiency induced wet onset",
          "[physics][milbrandt][triple_hail_surface_liquid]")
{
    auto capped_input=source(272.15,.0045f,.001f,2,false);
    capped_input.duration_s=1000;
    const auto capped=prepared(capped_input);
    const auto captured=capped.cloud_collection().channels()[2];
    REQUIRE(captured.raw_mass_increment_per_kg > captured.admitted_mass_increment_per_kg);
    CHECK(capped.receipt().cloud_collected_mass==captured.admitted_mass_increment_per_kg);
    CHECK(capped.receipt().cloud_collected_number==captured.admitted_number_increment_per_kg);
    check(capped_input,capped);
    const auto boundary = [&](bool dry)
    {
        double lo=1e-8,hi=.04;
        for(int iteration=0;iteration<60;++iteration)
        {
            const double mid=(lo+hi)/2;
            const auto input=source(268.15,.003f,static_cast<float>(mid),2,false);
            const auto budget=reference::evaluate(reference::cell(input.before),input.duration_s);
            const bool wet=dry ? budget.dry_heat>0 && budget.dry_cloud+budget.vapor>0 : budget.shed>0;
            if(wet) hi=mid; else lo=mid;
        }
        return (lo+hi)/2;
    };
    const double wet_threshold=boundary(false),dry_threshold=boundary(true);
    REQUIRE(dry_threshold>wet_threshold);
    const auto input=source(268.15,.003f,static_cast<float>((wet_threshold+dry_threshold)/2),2,false);
    const auto expected=reference::evaluate(reference::cell(input.before),input.duration_s);
    REQUIRE(expected.shed>0);
    REQUIRE(expected.dry_heat<0);
    HailLiquidSurfaceTransaction rejected;
    std::string error;
    CHECK_FALSE(rejected.prepare(input,error));
    CHECK(rejected.stage()==SourceEosEnergyTransaction::Stage::Empty);
}

TEST_CASE("wet hail returns rain through a separate shed spectrum after gross donor loss",
          "[physics][milbrandt][triple_hail_surface_liquid]")
{
    auto input = source(272.15,.015f,.001f);
    const auto transaction = prepared(input);
    const auto r = transaction.receipt();
    REQUIRE(r.rain_collected_mass > 0);
    REQUIRE(r.melted_hail_mass > 0);
    CHECK(r.rain_number_after == input.before.hail.rain_number - r.rain_collected_number + r.shed_rain_number + r.melted_hail_number);
    CHECK(r.rain_reflectivity_after == input.before.hail.rain_reflectivity - r.rain_reflectivity_removed + r.shed_rain_reflectivity + r.melt_rain_reflectivity);
    const auto q = transaction.energy_input().accepted_water;
    const double wrong_net = input.before.hail.rain_reflectivity * (1 +
        2*(q[2]-input.before.hail.thermal.water[2])/input.before.hail.thermal.water[2] -
        (r.rain_number_after-input.before.hail.rain_number)/input.before.hail.rain_number);
    CHECK(std::abs(wrong_net-r.rain_reflectivity_after) > .01*std::abs(r.rain_reflectivity_after-input.before.hail.rain_reflectivity));
    bool saw_gross_rejection = false;
    input.before.hail.thermal.water[6] = .04f;
    input.before.hail.hail_number = 4000.f;
    input.before.hail.hail_reflectivity = static_cast<float>(reference::spectral::population(3,.04f,4000,2).reflectivity_per_kg);
    input.before.hail.thermal.water[1] = .03f;
    for (double dt : {5.,10.,20.,50.})
    {
        input.duration_s = dt;
        HailLiquidSurfaceTransaction invalid;
        std::string error;
        if (!invalid.prepare(input,error) && error.find("gross original-rain") != std::string::npos)
        {
            saw_gross_rejection = true;
            CHECK(invalid.stage() == SourceEosEnergyTransaction::Stage::Empty);
        }
    }
    CHECK(saw_gross_rejection);
}

TEST_CASE("wet hail liquid collection does not certify a dry endpoint from its wet source",
          "[physics][milbrandt][triple_hail_surface_liquid]")
{
    auto input=source(272.15,.0045f,.001f,2,false);
    input.before.hail.hail_number=2000;
    input.before.hail.hail_reflectivity=static_cast<float>(reference::spectral::population(
        3,input.before.hail.thermal.water[6],2000,2).reflectivity_per_kg);
    input.duration_s=300;
    const auto before=reference::cell(input.before);
    const auto initial=reference::evaluate(before,input.duration_s);
    REQUIRE(initial.shed>0);
    REQUIRE(initial.dry_cloud+initial.vapor>0);
    REQUIRE(initial.dry_heat>0);
    SourceEosEnergyInput energy_input{input.before.hail.thermal,{},input.generation,input.duration_s};
    for(std::size_t i=0;i<7;++i) energy_input.accepted_water[i]=input.before.hail.thermal.water[i];
    energy_input.accepted_water[0]+=static_cast<double>(initial.increments[0]);
    energy_input.accepted_water[1]+=static_cast<double>(initial.increments[1]);
    energy_input.accepted_water[2]+=static_cast<double>(initial.increments[3]);
    energy_input.accepted_water[6]+=static_cast<double>(initial.increments[6]);
    SourceEosEnergyTransaction energy;
    std::string error;
    REQUIRE(energy.prepare(energy_input,error));
    auto end=before;
    for(std::size_t i=0;i<end.state.size();++i) end.state[i]+=initial.increments[i];
    end.temperature=energy.receipt().temperature_eos_target;
    end.pressure=end.density*287*end.temperature;
    const auto final=reference::evaluate(end,input.duration_s);
    CHECK((final.shed<=0 || final.dry_cloud+final.rain+final.vapor<=0 || final.dry_heat<=0));
    HailLiquidSurfaceTransaction transaction;
    CHECK_FALSE(transaction.prepare(input,error));
    CHECK(error.find("endpoint requires")!=std::string::npos);
    CHECK(transaction.stage()==SourceEosEnergyTransaction::Stage::Empty);
    input.duration_s=.01;
    CHECK(prepared(input).receipt().shed_liquid_mass>0);
}

TEST_CASE("wet hail liquid owner rejects unowned sources and dry boundaries without partial publication",
          "[physics][milbrandt][triple_hail_surface_liquid]")
{
    for (int mutation = 0; mutation < 12; ++mutation)
    {
        auto input = source(268.15,.003f,.03f);
        if (mutation==0) input.competing_vapor_transfer=1e-8;
        if (mutation==1) input.dry_hail_diffusion=1e-8;
        if (mutation==2) input.before.hail.thermal.water[3]=1e-6f;
        if (mutation==3) input.before.cloud_number=0;
        if (mutation==4) input.before.cloud_number=std::numeric_limits<float>::quiet_NaN();
        if (mutation==5) input.generation=0;
        if (mutation==6) input.duration_s=0;
        if (mutation==7) input.duration_s=std::numeric_limits<double>::denorm_min();
        if (mutation==8) input.duration_s=1e7;
        if (mutation==9) input=source(280.15);
        if (mutation==10) input=source(220.15);
        if (mutation==11) input=source(268.15,.003f,0,2,false);
        HailLiquidSurfaceTransaction transaction;
        std::string error;
        CHECK_FALSE(transaction.prepare(input,error));
        CHECK(transaction.stage()==SourceEosEnergyTransaction::Stage::Empty);
        CHECK(transaction.receipt().cloud_collected_mass==0);
    }
    auto absent=source();
    absent.before.hail.thermal.water[6]=absent.before.hail.hail_number=absent.before.hail.hail_reflectivity=0;
    const auto no_hail=prepared(absent);
    CHECK(no_hail.source_endpoint().hail.thermal.water==absent.before.hail.thermal.water);
    CHECK(no_hail.source_endpoint().cloud_number==absent.before.cloud_number);
}

TEST_CASE("wet hail liquid owner has the original no-collection condensation limit",
          "[physics][milbrandt][triple_hail_surface_liquid]")
{
    const auto input=source(268.15,.015f,0,2,false);
    const auto liquid=prepared(input);
    HailSurfaceCondensationTransaction original;
    std::string error;
    REQUIRE(original.prepare({input.before.hail,{},0,input.generation,input.duration_s},error));
    near(liquid.receipt().surface_vapor_transfer,original.receipt().condensed_mass);
    near(liquid.receipt().melted_hail_mass,original.receipt().melted_mass);
    CHECK(liquid.source_endpoint().hail.thermal.water==original.source_endpoint().thermal.water);
    CHECK(liquid.source_endpoint().hail.thermal.theta==original.source_endpoint().thermal.theta);
}

TEST_CASE("wet hail liquid moment generation ownership survives actual EOS and rejects foreign commits",
          "[physics][milbrandt][triple_hail_surface_liquid]")
{
    const auto input=source(268.15,.003f,.03f);
    auto transaction=prepared(input);
    auto stage=transaction.source_endpoint();
    std::string error;
    auto bad=stage;
    bad.cloud_number=std::nextafter(bad.cloud_number,1e9f);
    CHECK_FALSE(transaction.accept_source(input.before,bad,input.generation,error));
    CHECK(transaction.stage()==SourceEosEnergyTransaction::Stage::Prepared);
    CHECK_FALSE(transaction.accept_source(input.before,stage,input.generation+1,error));
    REQUIRE(transaction.accept_source(input.before,stage,input.generation,error));
    SimulationStorageState storage{{5,3,4,4.,5.,.5,6.}};
    storage.grid.geometry.initialize(5,3,4,4.,5.,6.,CoordinateSystem::Cartesian,StaggerType::CGrid);
    auto& f=storage.fields;
    for (Field3D* field:{&f.u,&f.v,&f.w,&f.rho,&f.p,&f.theta,&f.qv,&f.qc,&f.qr,&f.qi,&f.qs,&f.qg,&f.qh,&f.tke}) field->resize(5,3,4,0.f);
    f.rho.fill(stage.hail.thermal.density); f.p.fill(stage.hail.thermal.pressure); f.theta.fill(stage.hail.thermal.theta);
    const std::array<Field3D*,7> water{&f.qv,&f.qc,&f.qr,&f.qi,&f.qs,&f.qg,&f.qh};
    for (std::size_t i=0;i<water.size();++i) water[i]->fill(stage.hail.thermal.water[i]);
    REQUIRE(tmv::dynamics_eos::apply_pressure_projection(make_simulation_state_view(storage)).complete);
    auto after=stage; after.hail.thermal.pressure=f.p(1,1,0);
    bad=after; bad.hail.rain_reflectivity=std::nextafter(bad.hail.rain_reflectivity,1.f);
    CHECK_FALSE(transaction.accept_eos(stage,bad,input.generation,error));
    CHECK(transaction.stage()==SourceEosEnergyTransaction::Stage::SourceCommitted);
    REQUIRE(transaction.accept_eos(stage,after,input.generation,error));
    CHECK_FALSE(transaction.accept_eos(stage,after,input.generation,error));
}

TEST_CASE("wet hail liquid evolution is deterministic and approaches independent melt and freezing integrations",
          "[physics][milbrandt][triple_hail_surface_liquid]")
{
    for (bool melt : {false,true})
    {
        auto initial=source(melt?272.15:268.15,melt?.015f:.003f,melt?.001f:.03f);
        std::array<HailLiquidSurfaceTransaction,4> parallel;
        std::array<std::thread,4> workers;
        std::array<bool,4> success{};
        for(std::size_t i=0;i<workers.size();++i) workers[i]=std::thread([&,i]{std::string error;success[i]=parallel[i].prepare(initial,error);});
        for(auto& worker:workers) worker.join();
        for(std::size_t i=0;i<parallel.size();++i)
        {
            REQUIRE(success[i]);
            CHECK(parallel[i].receipt().surface_vapor_transfer==parallel[0].receipt().surface_vapor_transfer);
            CHECK(parallel[i].receipt().rain_reflectivity_after==parallel[0].receipt().rain_reflectivity_after);
        }
        auto context=reference::cell(initial.before);
        const auto prepared_initial=prepared(initial);
        context.temperature=prepared_initial.energy_receipt().temperature_eos_reference;
        context.pressure=context.density*287*context.temperature;
        constexpr double duration=20;
        const auto oracle=reference::rk4(context,duration,1024);
        const auto refined=reference::rk4(context,duration,2048);
        CHECK(std::abs(oracle[6]-refined[6])<1e-11L);
        std::array<long double,3> errors{};
        for(int level=0;level<3;++level)
        {
            auto next=initial;
            const int steps=10<<level;
            next.duration_s=duration/steps;
            long double accounted=0;
            for(int step=0;step<steps;++step)
            {
                auto transaction=prepared(next);
                std::string error;
                REQUIRE(transaction.accept_source(next.before,transaction.source_endpoint(),next.generation,error));
                REQUIRE(transaction.accept_eos(transaction.source_endpoint(),transaction.eos_endpoint(),next.generation,error));
                accounted+=transaction.energy_receipt().committed_internal_energy_change;
                next.before=transaction.eos_endpoint(); ++next.generation;
            }
            const auto endpoint=reference::cell(next.before);
            const auto beginning=reference::cell(initial.before);
            CHECK(std::abs(reference::inventory(endpoint.temperature,endpoint.state)-reference::inventory(beginning.temperature,beginning.state)-accounted)<1e-6L);
            errors[level]=std::abs(endpoint.state[6]-oracle[6]);
        }
        CAPTURE(melt,errors[0],errors[1],errors[2]);
        std::cout<<std::setprecision(17)<<"[Hail liquid refinement] melt="<<melt<<" errors="<<errors[0]<<','<<errors[1]<<','<<errors[2]<<std::endl;
        CHECK(errors[1]<.8L*errors[0]); CHECK(errors[2]<.8L*errors[1]);
    }
}
