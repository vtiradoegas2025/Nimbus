#pragma once

struct WindProfile
{
    double u_sfc;
    double v_sfc;
    double u_1km;
    double v_1km;
    double u_6km;
    double v_6km;
};

extern WindProfile& global_wind_profile;
