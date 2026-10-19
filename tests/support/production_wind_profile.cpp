#include "init/hodograph/wind_profile.hpp"

void compute_wind_profile(
    const WindProfile& profile, double z, double& u, double& v)
{
    if (z <= 1000.0)
    {
        const double fraction = z / 1000.0;
        u = profile.u_sfc + fraction * (profile.u_1km - profile.u_sfc);
        v = profile.v_sfc + fraction * (profile.v_1km - profile.v_sfc);
    }
    else if (z <= 6000.0)
    {
        const double fraction = (z - 1000.0) / 5000.0;
        u = profile.u_1km + fraction * (profile.u_6km - profile.u_1km);
        v = profile.v_1km + fraction * (profile.v_6km - profile.v_1km);
    }
    else
    {
        u = profile.u_6km;
        v = profile.v_6km;
    }
}
