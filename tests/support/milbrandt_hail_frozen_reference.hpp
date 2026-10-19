#pragma once

#include "milbrandt_hail_liquid_reference.hpp"
#include <stdexcept>

namespace milbrandt_hail_frozen_reference
{
namespace liquid = milbrandt_hail_liquid_reference;
namespace spectral = milbrandt_frozen_collection_reference;
using State = std::array<long double, 15>;
struct Cell { long double temperature, pressure, density; State state; };

inline Cell cell(const tmv::milbrandt_detail::HailLiquidSurfaceState& source)
{
    const auto base = liquid::cell(source);
    Cell c{base.temperature, base.pressure, base.density, {}};
    std::copy(base.state.begin(), base.state.end(), c.state.begin());
    for (std::size_t i = 0; i < 2; ++i)
    {
        c.state[9 + 3*i] = source.hail.thermal.water[3 + i];
        c.state[10 + 3*i] = source.ice_snow_number[i];
        c.state[11 + 3*i] = source.ice_snow_reflectivity[i];
    }
    return c;
}

struct Transfers
{
    liquid::Transfers liquid;
    spectral::Reference wet_frozen, dry_frozen;
    long double collected_frozen = 0, dry_collected_frozen = 0, frozen_sensible = 0;
    long double frozen = 0, melted = 0, shed = 0, dry_heat = 0;
    State increments{};
};

inline Transfers evaluate(const Cell& c, long double dt, bool quadrature = false)
{
    Transfers r;
    liquid::Cell base{c.temperature, c.pressure, c.density, {}};
    std::copy_n(c.state.begin(), 9, base.state.begin());
    r.liquid = liquid::evaluate(base, dt, quadrature);
    spectral::Cell cold{static_cast<double>(c.temperature), static_cast<double>(c.pressure), static_cast<double>(c.density), {}};
    const auto pop = [&](std::size_t offset) {
        return spectral::Population{static_cast<double>(c.state[offset]), static_cast<double>(c.state[offset+1]), static_cast<double>(c.state[offset+2])};
    };
    cold.initial.frozen = {pop(9),pop(12),{},pop(6)};
    r.wet_frozen = spectral::evaluate(cold,dt,quadrature,true);
    r.dry_frozen = spectral::evaluate(cold,dt,quadrature,false);
    r.collected_frozen = r.wet_frozen.channels[2].mass + r.wet_frozen.channels[3].mass;
    r.dry_collected_frozen = r.dry_frozen.channels[2].mass + r.dry_frozen.channels[3].mass;
    const long double tc = c.temperature - 273.15L;
    const auto& l = r.liquid;
    r.frozen_sensible = 2106 * tc * r.collected_frozen;
    const long double heat = l.conduction + 2500000 * l.vapor + l.sensible + r.frozen_sensible;
    r.dry_heat = l.dry_heat + 2106 * tc * r.dry_collected_frozen;
    if (heat > 0)
    {
        long double lo = 0, hi = std::max(1.L, c.state[6]);
        for (int iteration=0;iteration<100;++iteration)
        {
            const long double m = (lo+hi)/2;
            if (heat + m*(2106*tc-334000) > 0) lo=m; else hi=m;
        }
        r.melted = (lo+hi)/2;
    }
    else if (heat < 0)
    {
        long double lo = 0, hi = std::max(1.L, l.cloud+l.rain+l.vapor);
        for (int iteration=0;iteration<100;++iteration)
        {
            const long double m = (lo+hi)/2;
            if (heat+334000*m < 0) lo=m; else hi=m;
        }
        r.frozen = (lo+hi)/2;
    }
    r.shed = l.cloud+l.rain+l.vapor-r.frozen;
    constexpr long double cr=523.598775598L,ch=471.238898038L;
    const long double melt_fraction = c.state[6]>0 ? r.melted/c.state[6] : 0;
    const long double shed_number = r.shed/(cr*1e-9L);
    r.increments[0]=-l.vapor;
    r.increments[1]=-l.cloud;
    r.increments[2]=-l.cloud_number;
    r.increments[3]=-l.rain+r.shed+r.melted;
    r.increments[4]=-l.rain_number+shed_number+c.state[7]*melt_fraction;
    r.increments[5]=-l.rain_z_removed+5.6e-18L*shed_number+(ch/cr)*(ch/cr)*c.state[8]*melt_fraction;
    r.increments[6]=r.frozen-r.melted+r.collected_frozen;
    r.increments[7]=-c.state[7]*melt_fraction;
    r.increments[8]=c.state[6]>0 ? c.state[8]*(2*r.frozen-r.melted)/c.state[6] : 0;
    r.increments[8]+=r.wet_frozen.reflectivity_increment[3];
    for(std::size_t i=0;i<2;++i)
    {
        r.increments[9+3*i]=r.wet_frozen.mass_increment[i];
        r.increments[10+3*i]=r.wet_frozen.number_increment[i];
        r.increments[11+3*i]=r.wet_frozen.reflectivity_increment[i];
    }
    return r;
}

inline long double inventory(long double t, const State& q)
{
    const long double ice=q[6]+q[9]+q[12];
    return 717*t+(t-273.15L)*(1388.5L*q[0]+4186*(q[1]+q[3])+2106*ice)
        +(2500000.L-461.5L*273.15L)*q[0]-334000*ice;
}

inline State rk4(const Cell& source, long double duration, int steps)
{
    const long double energy = inventory(source.temperature,source.state);
    auto state = source.state;
    const auto derivative = [&](const State& q)
    {
        const long double ice=q[6]+q[9]+q[12];
        const long double cv=717+1388.5L*q[0]+4186*(q[1]+q[3])+2106*ice;
        const long double phase=(2500000.L-461.5L*273.15L)*q[0]-334000*ice;
        const long double t=273.15L+(energy-717*273.15L-phase)/cv;
        constexpr long double rate_interval=1e-5L;
        const auto rates=evaluate({t,source.density*287*t,source.density,q},rate_interval);
        for(const auto& channel:rates.wet_frozen.channels)
            if(channel.factor!=1) throw std::runtime_error("RK4 reference reached a frozen donor cap");
        if(rates.liquid.cloud>=q[1] || rates.liquid.rain>=q[3])
            throw std::runtime_error("RK4 reference reached a liquid donor cap");
        auto slope=rates.increments;
        for(auto& value:slope) value/=rate_interval;
        return slope;
    };
    const long double dt=duration/steps;
    for(int step=0;step<steps;++step)
    {
        const auto a=derivative(state);
        const auto stage=[&](const State& slope,long double fraction)
        {
            auto next=state;
            for(std::size_t i=0;i<state.size();++i) next[i]+=fraction*dt*slope[i];
            return derivative(next);
        };
        const auto b=stage(a,.5L),c=stage(b,.5L),d=stage(c,1);
        for(std::size_t i=0;i<state.size();++i) state[i]+=dt/6*(a[i]+2*b[i]+2*c[i]+d[i]);
    }
    return state;
}
}
