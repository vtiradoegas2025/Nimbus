/**
 * @file rayleigh_damping.cpp
 * @brief Rayleigh damping sponge layer near the domain top.
 *
 * Prevents reflection of upward-propagating gravity waves off the rigid
 * lid by relaxing prognostic fields toward the environmental base state
 * in the upper portion of the domain. Standard technique from Klemp &
 * Lilly (1978), used in CM1, WRF, and ARPS.
 *
 * The damping coefficient follows a sine-squared profile:
 *
 *   alpha(z) = alpha_max * sin^2(pi/2 * (z - z_damp) / (z_top - z_damp))
 *
 * for z > z_damp, and alpha = 0 below. This provides a smooth onset that
 * avoids spurious reflections at the sponge base.
 *
 * Fields are relaxed toward:
 *   u, v  -> environmental wind profile (hodograph)
 *   w     -> 0 (no vertical motion in base state)
 *   theta -> theta0 (reference potential temperature)
 *   rho   -> rho0_base[k]
 *   p     -> p0_base[k]
 */

#include "core/simulation.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <cmath>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

// Forward declaration: defined in tornado_sim.cpp
void compute_wind_profile(const WindProfile& profile, double z, double& u, double& v);

namespace
{

struct RayleighConfig
{
    double depth_fraction = 0.3;   // sponge covers top 30% of domain
    double alpha_max      = 0.05;  // maximum damping rate (1/s)
};

RayleighConfig g_rayleigh_config;
bool g_rayleigh_initialized = false;

// Precomputed per-level damping coefficients and reference winds
std::vector<double> g_alpha;       // damping coefficient per level
std::vector<double> g_u_ref;       // reference u wind per level
std::vector<double> g_v_ref;       // reference v wind per level

void ensure_initialized()
{
    if (g_rayleigh_initialized && static_cast<int>(g_alpha.size()) == NZ)
    {
        return;
    }

    g_alpha.assign(static_cast<size_t>(NZ), 0.0);
    g_u_ref.assign(static_cast<size_t>(NZ), 0.0);
    g_v_ref.assign(static_cast<size_t>(NZ), 0.0);

    const double z_top = static_cast<double>(NZ) * dz;
    const double z_damp = z_top * (1.0 - g_rayleigh_config.depth_fraction);
    const double sponge_depth = z_top - z_damp;

    if (sponge_depth <= 0.0)
    {
        g_rayleigh_initialized = true;
        return;
    }

    const double pi_half = 3.14159265358979323846 * 0.5;

    for (int k = 0; k < NZ; ++k)
    {
        const double z = static_cast<double>(k) * dz;

        // Reference wind from the environmental hodograph
        compute_wind_profile(global_wind_profile, z,
                             g_u_ref[static_cast<size_t>(k)],
                             g_v_ref[static_cast<size_t>(k)]);

        if (z > z_damp)
        {
            const double frac = (z - z_damp) / sponge_depth;
            const double s = std::sin(pi_half * frac);
            g_alpha[static_cast<size_t>(k)] = g_rayleigh_config.alpha_max * s * s;
        }
    }

    const int first_damped = static_cast<int>(std::ceil(z_damp / dz));
    tmv::log_info("[RAYLEIGH] Sponge layer active: z_damp=", z_damp,
                  "m (k>=", first_damped, "), alpha_max=",
                  g_rayleigh_config.alpha_max, " 1/s, depth=",
                  g_rayleigh_config.depth_fraction * 100.0, "% of domain");

    g_rayleigh_initialized = true;
}

} // namespace

void apply_rayleigh_damping(double dt_damp)
{
    ensure_initialized();

    if (NR <= 0 || NTH <= 0 || NZ <= 0)
    {
        return;
    }
    if (static_cast<int>(g_alpha.size()) != NZ)
    {
        return;
    }

    const double dt_safe = std::max(dt_damp, 1.0e-12);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < NR; ++i)
    {
        for (int j = 0; j < NTH; ++j)
        {
            for (int k = 0; k < NZ; ++k)
            {
                const double alpha = g_alpha[static_cast<size_t>(k)];
                if (alpha <= 0.0)
                {
                    continue;
                }

                // Relaxation factor: exponential decay toward reference
                // For small alpha*dt, this is approximately 1 - alpha*dt
                const double factor = 1.0 - std::exp(-alpha * dt_safe);

                const double u_ref = g_u_ref[static_cast<size_t>(k)];
                const double v_ref = g_v_ref[static_cast<size_t>(k)];
                const double w_ref = 0.0;
                const double rho_ref = (static_cast<size_t>(k) < rho0_base.size())
                                           ? rho0_base[static_cast<size_t>(k)] : 1.0;
                const double p_ref = (static_cast<size_t>(k) < p0_base.size())
                                         ? p0_base[static_cast<size_t>(k)] : p0;

                u[i][j][k] += static_cast<float>(factor * (u_ref - static_cast<double>(u[i][j][k])));
                v[i][j][k] += static_cast<float>(factor * (v_ref - static_cast<double>(v[i][j][k])));
                w[i][j][k] += static_cast<float>(factor * (w_ref - static_cast<double>(w[i][j][k])));

                // Relax thermodynamic fields toward base state
                theta[i][j][k] += static_cast<float>(
                    factor * (theta0 - static_cast<double>(theta[i][j][k])));
                rho[i][j][k] += static_cast<float>(
                    factor * (rho_ref - static_cast<double>(rho[i][j][k])));
                p[i][j][k] += static_cast<float>(
                    factor * (p_ref - static_cast<double>(p[i][j][k])));
            }
        }
    }
}

void reset_rayleigh_damping()
{
    g_rayleigh_initialized = false;
    g_alpha.clear();
    g_u_ref.clear();
    g_v_ref.clear();
}
