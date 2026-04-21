/**
 * @file nested_grid.cpp
 * @brief Nested grid initialization and parent feedback.
 *
 * Initializes nested-grid storage by interpolating from the parent grid,
 * and feeds nested-grid updates back to the parent.
 *
 * Extracted from src/core/equations.cpp.
 */

#include "core/simulation.hpp"

#include <algorithm>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

void initialize_nested_grid()
{
    if (!nested_config.enabled) return;

    int nr_nest = nested_config.nest_size_r;
    int nth_nest = nested_config.nest_size_th;
    int nz_nest = nested_config.nest_size_z;

    nest_rho.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_p.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_u.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_w.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_v.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_theta.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_qv.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_qc.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_qr.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_qh.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_qg.resize(nr_nest, nth_nest, nz_nest, 0.0f);
    nest_tracer.resize(nr_nest, nth_nest, nz_nest, 0.0f);

    double ref = nested_config.refinement;
    int ci = nested_config.center_i;
    int cj = nested_config.center_j;
    int ck = nested_config.center_k;

    #pragma omp parallel for collapse(2)
    for (int i_nest = 0; i_nest < nr_nest; ++i_nest)
    {
        for (int j_nest = 0; j_nest < nth_nest; ++j_nest)
        {
            for (int k_nest = 0; k_nest < nz_nest; ++k_nest)
            {
                double i_parent = ci + (i_nest - nr_nest/2.0) / ref;
                double j_parent = cj + (j_nest - nth_nest/2.0) / ref;
                double k_parent = ck + (k_nest - nz_nest/2.0) / ref;

                int i0 = std::max(0, std::min(NR-2, (int)std::floor(i_parent)));
                int j0 = std::max(0, std::min(NTH-2, (int)std::floor(j_parent)));
                int k0 = std::max(0, std::min(NZ-2, (int)std::floor(k_parent)));

                double fi = i_parent - i0;
                double fj = j_parent - j0;
                double fk = k_parent - k0;

                auto interpolate = [&](const Field3D& field, int i0, int j0, int k0, double fi, double fj, double fk) {
                    double v000 = static_cast<float>(field[i0][j0][k0]);
                    double v001 = static_cast<float>(field[i0][j0][k0+1]);
                    double v010 = static_cast<float>(field[i0][j0+1][k0]);
                    double v011 = static_cast<float>(field[i0][j0+1][k0+1]);
                    double v100 = static_cast<float>(field[i0+1][j0][k0]);
                    double v101 = static_cast<float>(field[i0+1][j0][k0+1]);
                    double v110 = static_cast<float>(field[i0+1][j0+1][k0]);
                    double v111 = static_cast<float>(field[i0+1][j0+1][k0+1]);

                    return v000 * (1-fi)*(1-fj)*(1-fk) +
                           v001 * (1-fi)*(1-fj)*fk +
                           v010 * (1-fi)*fj*(1-fk) +
                           v011 * (1-fi)*fj*fk +
                           v100 * fi*(1-fj)*(1-fk) +
                           v101 * fi*(1-fj)*fk +
                           v110 * fi*fj*(1-fk) +
                           v111 * fi*fj*fk;
                };

                nest_rho[i_nest][j_nest][k_nest] = interpolate(rho, i0, j0, k0, fi, fj, fk);
                nest_p[i_nest][j_nest][k_nest] = interpolate(p, i0, j0, k0, fi, fj, fk);
                nest_u[i_nest][j_nest][k_nest] = interpolate(u, i0, j0, k0, fi, fj, fk);
                nest_w[i_nest][j_nest][k_nest] = interpolate(w, i0, j0, k0, fi, fj, fk);
                nest_v[i_nest][j_nest][k_nest] = interpolate(v, i0, j0, k0, fi, fj, fk);
                nest_theta[i_nest][j_nest][k_nest] = interpolate(theta, i0, j0, k0, fi, fj, fk);
                nest_qv[i_nest][j_nest][k_nest] = interpolate(qv, i0, j0, k0, fi, fj, fk);
                nest_qc[i_nest][j_nest][k_nest] = interpolate(qc, i0, j0, k0, fi, fj, fk);
                nest_qr[i_nest][j_nest][k_nest] = interpolate(qr, i0, j0, k0, fi, fj, fk);
                nest_qh[i_nest][j_nest][k_nest] = interpolate(qh, i0, j0, k0, fi, fj, fk);
                nest_qg[i_nest][j_nest][k_nest] = interpolate(qg, i0, j0, k0, fi, fj, fk);
                nest_tracer[i_nest][j_nest][k_nest] = interpolate(tracer, i0, j0, k0, fi, fj, fk);
            }
        }
    }
}

void feedback_to_parent()
{
    if (!nested_config.enabled) return;

    double ref = nested_config.refinement;
    int ci = nested_config.center_i;
    int cj = nested_config.center_j;
    int ck = nested_config.center_k;
    int nr_nest = nested_config.nest_size_r;
    int nth_nest = nested_config.nest_size_th;
    int nz_nest = nested_config.nest_size_z;

    #pragma omp parallel for collapse(2)
    for (int i_nest = 0; i_nest < nr_nest; ++i_nest)
    {
        for (int j_nest = 0; j_nest < nth_nest; ++j_nest)
        {
            for (int k_nest = 0; k_nest < nz_nest; ++k_nest)
            {
                int i_parent = ci + (int)std::round((i_nest - nr_nest/2.0) / ref);
                int j_parent = cj + (int)std::round((j_nest - nth_nest/2.0) / ref);
                int k_parent = ck + (int)std::round((k_nest - nz_nest/2.0) / ref);

                if (i_parent >= 0 && i_parent < NR &&
                    j_parent >= 0 && j_parent < NTH &&
                    k_parent >= 0 && k_parent < NZ)
                {
                    rho[i_parent][j_parent][k_parent] = nest_rho[i_nest][j_nest][k_nest];
                    p[i_parent][j_parent][k_parent] = nest_p[i_nest][j_nest][k_nest];
                    u[i_parent][j_parent][k_parent] = nest_u[i_nest][j_nest][k_nest];
                    w[i_parent][j_parent][k_parent] = nest_w[i_nest][j_nest][k_nest];
                    v[i_parent][j_parent][k_parent] = nest_v[i_nest][j_nest][k_nest];
                    theta[i_parent][j_parent][k_parent] = nest_theta[i_nest][j_nest][k_nest];
                    qv[i_parent][j_parent][k_parent] = nest_qv[i_nest][j_nest][k_nest];
                    qc[i_parent][j_parent][k_parent] = nest_qc[i_nest][j_nest][k_nest];
                    qr[i_parent][j_parent][k_parent] = nest_qr[i_nest][j_nest][k_nest];
                    qh[i_parent][j_parent][k_parent] = nest_qh[i_nest][j_nest][k_nest];
                    qg[i_parent][j_parent][k_parent] = nest_qg[i_nest][j_nest][k_nest];
                    tracer[i_parent][j_parent][k_parent] = nest_tracer[i_nest][j_nest][k_nest];
                }
            }
        }
    }
}
