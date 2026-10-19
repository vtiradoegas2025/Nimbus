#pragma once

#include "milbrandt_frozen_collection_reference.hpp"
#include "microphysics/schemes/milbrandt/triple_hail_surface_liquid.hpp"

namespace milbrandt_hail_liquid_reference
{
namespace spectral = milbrandt_frozen_collection_reference;
using State = std::array<long double, 9>;

struct Cell
{
    long double temperature, pressure, density;
    State state;
};

inline Cell cell(const tmv::milbrandt_detail::HailLiquidSurfaceState& source)
{
    const auto& h = source.hail;
    return {h.thermal.theta * std::pow(h.thermal.pressure / 100000.L, 287.L / 1004.L), h.thermal.pressure, h.thermal.density,
        {h.thermal.water[0], h.thermal.water[1], source.cloud_number, h.thermal.water[2], h.rain_number,
         h.rain_reflectivity, h.thermal.water[6], h.hail_number, h.hail_reflectivity}};
}

struct Spectrum { long double alpha = 0, lambda = 0; };

inline Spectrum spectrum(long double q, long double n, long double z, long double coefficient)
{
    if (q == 0) return {};
    const long double ratio = z * n * coefficient * coefficient / (q * q);
    long double lo = -1 + 1e-9L, hi = 1e6L;
    for (int iteration = 0; iteration < 160; ++iteration)
    {
        const long double mid = (lo + hi) / 2;
        const long double g = (mid + 4) * (mid + 5) * (mid + 6) / ((mid + 1) * (mid + 2) * (mid + 3));
        if (g > ratio) lo = mid;
        else hi = mid;
    }
    const long double a = (lo + hi) / 2;
    return {a, std::cbrt(coefficient * n * (a + 1) * (a + 2) * (a + 3) / q)};
}

inline long double moment(const Spectrum& s, long double order, bool quadrature)
{
    const long double numerator = quadrature
        ? spectral::scaled_gamma_integral(s.alpha, order) / spectral::scaled_gamma_integral(s.alpha, 0)
        : spectral::gamma_ratio(s.alpha, order);
    return numerator / std::pow(s.lambda, order);
}

struct Transfers
{
    long double cloud = 0, cloud_number = 0, dry_cloud = 0, rain = 0, rain_number = 0;
    long double vapor = 0, conduction = 0, sensible = 0, dry_heat = 0, frozen = 0, melted = 0, shed = 0;
    long double shed_number = 0, shed_z = 0, melt_number = 0, melt_z = 0, rain_z_removed = 0;
    State increments{};
};

inline Transfers evaluate(const Cell& c, long double dt, bool quadrature = false, bool instantaneous = false)
{
    Transfers r;
    const auto& x = c.state;
    if (x[6] == 0) return r;
    constexpr long double cr = 523.598775598L, ch = 471.238898038L;
    const auto hail = spectrum(x[6], x[7], x[8], ch);
    const auto mh = [&](long double k) { return moment(hail, k, quadrature); };
    const long double density_factor = std::sqrt(1.225L / c.density);
    if (x[1] > 1e-14L && x[6] > 1e-14L)
    {
        const long double inverse = std::cbrt(x[1] / (2 * cr * x[2]));
        const auto mc = [&](long double k)
        {
            const long double g = quadrature ? spectral::scaled_gamma_integral(1, k / 3) / spectral::scaled_gamma_integral(1, 0)
                : std::tgamma(2 + k / 3);
            return g * std::pow(inverse, k);
        };
        const long double common = dt * std::acos(-1.L) / 4 * density_factor * 206.89L * c.density * x[2] * x[7];
        const long double q = common * cr * (mh(2.6384L) * mc(3) + 2 * mh(1.6384L) * mc(4) + mh(.6384L) * mc(5));
        const long double n = common * (mh(2.6384L) + 2 * mh(1.6384L) * mc(1) + mh(.6384L) * mc(2));
        const long double dry_efficiency = std::exp(-8.68e-7L * std::pow(std::cbrt(x[1] / (cr * x[2])), -1.6L) * std::cbrt(x[6] / (ch * x[7])));
        const long double factor = instantaneous ? 1 : std::min({1.L, x[1] / q, x[2] / n});
        const long double dry_factor = instantaneous ? 1 : std::min({1.L, x[1] / (q * dry_efficiency), x[2] / (n * dry_efficiency)});
        r.cloud = q * factor;
        r.cloud_number = n * factor;
        r.dry_cloud = q * dry_efficiency * dry_factor;
    }
    if (x[3] > 1e-14L && x[6] > 1e-14L)
    {
        const auto rain = spectrum(x[3], x[4], x[5], cr);
        const auto mr = [&](long double k) { return moment(rain, k, quadrature); };
        const long double vh = density_factor * 206.89L * mh(3.6384L) / mh(3);
        const long double vr = density_factor * 149.1L * mr(3.5L) / mr(3);
        const long double relative = std::sqrt((vh - vr) * (vh - vr) + .04L * vh * vr);
        const long double common = dt * std::acos(-1.L) / 4 * c.density * x[4] * x[7] * relative;
        const long double q = common * cr * (mh(2) * mr(3) + 2 * mh(1) * mr(4) + mr(5));
        const long double n = common * (mh(2) + 2 * mh(1) * mr(1) + mr(2));
        long double factor = instantaneous ? 1 : std::min({1.L, x[3] / q, x[4] / n});
        if (factor < 1) factor *= 1 - std::ldexp(1.L, -40);
        r.rain = q * factor;
        r.rain_number = n * factor;
        r.rain_z_removed = x[5] * (2 * r.rain / x[3] - r.rain_number / x[4]);
    }
    const long double tc = c.temperature - 273.15L;
    const long double diffusivity = (2.2157e-5L + .0155e-5L * tc) * 1e5L / c.pressure;
    const long double conductivity = .023971L + .000078L * tc;
    const long double viscosity = 1.72e-5L * 393 / (c.temperature + 120) * std::pow(c.temperature / 273.15L, 1.5L) / c.density;
    const long double vent = .78L * mh(1) + .30L * std::cbrt(viscosity / diffusivity) *
        std::sqrt(density_factor * 206.89L / viscosity) * mh(1.8192L);
    const long double b = dt * 2 * std::acos(-1.L) * x[7] * vent;
    const long double saturation = std::clamp(.62198L * 611.21L / std::max(c.pressure - 611.21L, 1.L), 0.L, .2L);
    r.vapor = b * c.density * diffusivity * (x[0] - saturation);
    r.conduction = b * conductivity * tc;
    const long double liquid = r.cloud + r.rain;
    r.sensible = 4186 * tc * liquid;
    const long double heat = r.conduction + 2500000 * r.vapor + r.sensible;
    r.dry_heat = r.conduction + 2500000 * r.vapor + 4186 * tc * (r.dry_cloud + r.rain) + 334000 * (r.dry_cloud + r.rain + r.vapor);
    if (heat > 0)
    {
        long double lo = 0, hi = std::max(1.L, x[6]);
        for (int iteration = 0; iteration < 100; ++iteration)
        {
            const long double m = (lo + hi) / 2;
            if (heat + m * (2106 * tc - 334000) > 0) lo = m;
            else hi = m;
        }
        r.melted = (lo + hi) / 2;
    }
    else
    {
        long double lo = 0, hi = std::max(1.L, liquid + r.vapor);
        for (int iteration = 0; iteration < 100; ++iteration)
        {
            const long double m = (lo + hi) / 2;
            if (heat + m * 334000 < 0) lo = m;
            else hi = m;
        }
        r.frozen = (lo + hi) / 2;
    }
    r.shed = liquid + r.vapor - r.frozen;
    r.shed_number = r.shed / (cr * 1e-9L);
    r.shed_z = r.shed_number * 5.6e-18L;
    r.melt_number = x[7] * r.melted / x[6];
    r.melt_z = (ch / cr) * (ch / cr) * x[8] * r.melted / x[6];
    r.increments = {-r.vapor, -r.cloud, -r.cloud_number,
        -r.rain + r.shed + r.melted, -r.rain_number + r.shed_number + r.melt_number,
        -r.rain_z_removed + r.shed_z + r.melt_z, r.frozen - r.melted, -r.melt_number,
        x[8] * (2 * r.frozen - r.melted) / x[6]};
    return r;
}

inline long double inventory(long double t, const State& q)
{
    return 717 * t + (t - 273.15L) * (1388.5L * q[0] + 4186 * (q[1] + q[3]) + 2106 * q[6])
        + (2500000.L - 461.5L * 273.15L) * q[0] - 334000 * q[6];
}

inline State rk4(const Cell& source, long double duration, int steps)
{
    const long double energy = inventory(source.temperature, source.state);
    auto state = source.state;
    const auto derivative = [&](const State& q)
    {
        const long double cv = 717 + 1388.5L * q[0] + 4186 * (q[1] + q[3]) + 2106 * q[6];
        const long double phase = (2500000.L - 461.5L * 273.15L) * q[0] - 334000 * q[6];
        const long double t = 273.15L + (energy - 717 * 273.15L - phase) / cv;
        return evaluate({t, source.density * 287 * t, source.density, q}, 1, false, true).increments;
    };
    const long double dt = duration / steps;
    for (int step = 0; step < steps; ++step)
    {
        const auto a = derivative(state);
        const auto stage = [&](const State& slope, long double fraction)
        {
            auto next = state;
            for (std::size_t i = 0; i < state.size(); ++i) next[i] += fraction * dt * slope[i];
            return derivative(next);
        };
        const auto b = stage(a, .5L), c = stage(b, .5L), d = stage(c, 1);
        for (std::size_t i = 0; i < state.size(); ++i) state[i] += dt / 6 * (a[i] + 2*b[i] + 2*c[i] + d[i]);
    }
    return state;
}
}
