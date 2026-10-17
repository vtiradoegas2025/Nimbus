#pragma once

namespace tmv
{

/** Runtime-consumed microphysics parameters shared by factory and config IO. */
struct MicrophysicsConfig
{
    struct Shared
    {
        double cloud_water_threshold_kgkg = 1.0e-3;
        double autoconversion_rate_s_inv = 1.0e-3;
    } shared;

    struct Kessler
    {
        double accretion_coefficient = 2.2;
        double evaporation_efficiency = 1.0;
    } kessler;

    struct Lin
    {
        double accretion_coefficient = 2.2;
        double evaporation_rate_s_inv = 1.0e-3;
    } lin;

    struct Thompson
    {
        double evaporation_rate_s_inv = 1.0e-3;
        double water_friendly_aerosol_surface_m3 = 350.0e6;
        double ice_friendly_aerosol_surface_m3 = 2.0e6;
    } thompson;

    struct Milbrandt
    {
        double evaporation_rate_s_inv = 1.0e-3;
        bool triple_moment = false;
        bool hail_processes = true;
    } milbrandt;
};

} // namespace tmv
