#pragma once

#include "milbrandt_hail_frozen_reference.hpp"
#include "milbrandt_cold_cloud_diffusion_reference.hpp"
#include "milbrandt_cloud_adjustment_reference.hpp"

namespace milbrandt_hail_vapor_reference
{
namespace frozen = milbrandt_hail_frozen_reference;
namespace snow = milbrandt_cold_cloud_diffusion_reference;
namespace cloud = cloud_adjustment_reference;
using State = frozen::State;

inline long double reference_temperature(const tmv::milbrandt_detail::SourceEosEnergyState& s)
{
    constexpr long double k=287.L/1004;
    const long double p=std::pow(s.density*287.L*s.theta/std::pow(100000.L,k),1/(1-k));
    return p/(s.density*287.L);
}

inline long double temperature(long double energy,const State& q)
{
    const long double ice=q[6]+q[9]+q[12];
    const long double cv=717+1388.5L*q[0]+4186*(q[1]+q[3])+2106*ice;
    const long double phase=(2500000.L-461.5L*273.15L)*q[0]-334000*ice;
    return 273.15L+(energy-717*273.15L-phase)/cv;
}

struct Result
{
    frozen::Transfers rates;
    long double raw_snow=0,factor=1,vapor=0,deposition=0,original_vapor_remaining=0;
    long double frozen=0,melted=0,shed=0,cloud_transfer=0,temperature=0,pressure=0;
    State endpoint{};
    bool equilibrium_valid=false,cloud_exhausted=false;
};

inline Result evaluate(const tmv::milbrandt_detail::HailLiquidSurfaceInput& input,bool quadrature=false)
{
    Result out;
    const auto c=frozen::cell(input.before);
    const auto& x=c.state;
    const auto rates=frozen::evaluate(c,input.duration_s,quadrature);
    out.rates=rates;
    snow::Cell s{static_cast<double>(c.temperature),static_cast<double>(c.pressure),static_cast<double>(c.density),static_cast<double>(x[0]),{}};
    s.initial.frozen[0]={static_cast<double>(x[12]),static_cast<double>(x[13]),static_cast<double>(x[14])};
    out.raw_snow=snow::evaluate(s,input.duration_s,quadrature).diffusion[0];
    if(out.raw_snow<0)return out;
    const auto& l=rates.liquid;
    const long double demand=std::max(0.L,l.vapor)+out.raw_snow;
    if(demand>x[0])out.factor=x[0]/demand;
    out.vapor=l.vapor>0?l.vapor*out.factor:l.vapor;
    out.deposition=out.raw_snow*out.factor;
    out.original_vapor_remaining=x[0]-std::max(0.L,out.vapor)-out.deposition;
    const long double tc=c.temperature-273.15L;
    const long double heat=l.conduction+2500000*out.vapor+l.sensible+rates.frozen_sensible;
    if(heat!=0)
    {
        long double lo=0,hi=1;
        for(int iteration=0;iteration<120;++iteration)
        {
            const long double m=(lo+hi)/2;
            const long double remaining=heat+(heat>0?2106*tc-334000:334000)*m;
            if((heat>0&&remaining>0)||(heat<0&&remaining<0))lo=m;else hi=m;
        }
        (heat>0?out.melted:out.frozen)=(lo+hi)/2;
    }
    out.shed=l.cloud+l.rain+out.vapor-out.frozen;
    auto& q=out.endpoint;
    q=x;
    const long double fraction=x[6]>0?out.melted/x[6]:0;
    constexpr long double cr=523.598775598L,ch=471.238898038L;
    const long double shed_number=out.shed/(cr*1e-9L);
    q[0]-=out.vapor+out.deposition;q[1]-=l.cloud;q[2]-=l.cloud_number;
    q[3]+=-l.rain+out.shed+out.melted;
    q[4]+=-l.rain_number+shed_number+x[7]*fraction;
    q[5]+=-l.rain_z_removed+5.6e-18L*shed_number+(ch/cr)*(ch/cr)*x[8]*fraction;
    q[6]+=out.frozen-out.melted+rates.collected_frozen;
    q[7]-=x[7]*fraction;
    q[8]+=x[6]>0?x[8]*(2*out.frozen-out.melted)/x[6]:0;
    q[8]+=rates.wet_frozen.reflectivity_increment[3];
    for(std::size_t i=0;i<2;++i)
    {
        q[9+3*i]+=rates.wet_frozen.mass_increment[i];
        q[10+3*i]+=rates.wet_frozen.number_increment[i];
        q[11+3*i]+=rates.wet_frozen.reflectivity_increment[i];
    }
    q[12]+=out.deposition;
    if(x[12]>0)q[14]+=6/2.078L*x[14]*out.deposition/x[12];
    const auto base=q;
    const long double energy=frozen::inventory(reference_temperature(input.before.hail.thermal),x);
    const auto state=[&](long double transfer){auto next=base;next[0]-=transfer;next[1]+=transfer;return next;};
    const auto exhausted=state(-base[1]);
    const long double exhausted_t=temperature(energy,exhausted);
    if(exhausted_t>233.15L&&exhausted_t<273.15L&&exhausted[0]<cloud::saturation(exhausted_t,c.density*287*exhausted_t))
    {
        out.cloud_transfer=-base[1];out.temperature=exhausted_t;out.cloud_exhausted=true;
    }
    else
    {
        const auto energy_residual=[&](long double t)
        {
            const long double transfer=base[0]-cloud::saturation(t,c.density*287*t);
            return frozen::inventory(t,state(transfer))-energy;
        };
        long double lo=233.15L,hi=273.15L;
        if(energy_residual(lo)>0||energy_residual(hi)<0)return out;
        for(int iteration=0;iteration<160;++iteration)
        {
            const long double mid=(lo+hi)/2;
            if(mid==lo||mid==hi)break;
            if(energy_residual(mid)<0)lo=mid;else hi=mid;
        }
        out.temperature=(lo+hi)/2;
        out.cloud_transfer=base[0]-cloud::saturation(out.temperature,c.density*287*out.temperature);
        if(out.cloud_transfer < -base[1] || out.cloud_transfer>out.original_vapor_remaining)return out;
    }
    q=state(out.cloud_transfer);
    out.pressure=c.density*287*out.temperature;
    if(out.cloud_transfer>0)q[2]=std::max(base[2],cloud::activated(input.vertical_velocity_ms,out.temperature,out.pressure)/c.density);
    else if(out.cloud_transfer<0)q[2]=base[2]*q[1]/base[1];
    out.equilibrium_valid=true;
    return out;
}
}
