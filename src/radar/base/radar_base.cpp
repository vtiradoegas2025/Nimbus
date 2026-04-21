/**
 * @file radar_base.cpp
 * @brief Implementation for the radar module.
 *
 * Provides executable logic for the radar runtime path,
 * including initialization, stepping, and diagnostics helpers.
 * This file is part of the src/radar subsystem.
 */

#include "radar/radar_base.hpp"
#include <cmath>
#include <algorithm>

/**
 * @brief Shared radar geometry and sampling utilities implementation
 */

namespace
{

/**
 * @brief Wraps azimuth index into periodic theta range.
 */
int wrap_theta_index(int j, int nth)
{
    if (nth <= 0)
    {
        return 0;
    }
    int wrapped = j % nth;
    if (wrapped < 0)
    {
        wrapped += nth;
    }
    return wrapped;
}

/**
 * @brief Ensures a field has single-cell dimensions (1x1x1).
 */
void ensure_single_cell_shape(Field3D& out)
{
    if (out.size_r() != 1 || out.size_th() != 1 || out.size_z() != 1)
    {
        out.resize(1, 1, 1, 0.0f);
    }
}

/**
 * @brief Copies one sampled cell into a single-cell scratch field.
 */
Field3D* sample_to_single_cell(const Field3D* source, int i, int j, int k, Field3D& out)
{
    if (source == nullptr)
    {
        return nullptr;
    }

    ensure_single_cell_shape(out);
    out(0, 0, 0) = static_cast<float>((*source)[i][j][k]);
    return &out;
}

/**
 * @brief Scratch buffers used for point-sampled radar state evaluation.
 */
struct RadarPointScratch
{
    Field3D u;
    Field3D v;
    Field3D w;
    Field3D qr;
    Field3D qs;
    Field3D qg;
    Field3D qh;
    Field3D qi;
    Field3D Nr;
    Field3D Ns;
    Field3D Ng;
    Field3D Nh;
    Field3D Ni;
    Field3D theta;
    Field3D p;
};

/**
 * @brief Initializes all scratch fields for point-sample extraction.
 */
void initialize_scratch_fields(RadarPointScratch& scratch)
{
    ensure_single_cell_shape(scratch.u);
    ensure_single_cell_shape(scratch.v);
    ensure_single_cell_shape(scratch.w);
    ensure_single_cell_shape(scratch.qr);
    ensure_single_cell_shape(scratch.qs);
    ensure_single_cell_shape(scratch.qg);
    ensure_single_cell_shape(scratch.qh);
    ensure_single_cell_shape(scratch.qi);
    ensure_single_cell_shape(scratch.Nr);
    ensure_single_cell_shape(scratch.Ns);
    ensure_single_cell_shape(scratch.Ng);
    ensure_single_cell_shape(scratch.Nh);
    ensure_single_cell_shape(scratch.Ni);
    ensure_single_cell_shape(scratch.theta);
    ensure_single_cell_shape(scratch.p);
}

}

/**
 * @brief Computes radar line-of-sight unit vector and range.
 */
void RadarGeometry::compute_line_of_sight(double radar_x, double radar_y, double radar_z,
                                         double x, double y, double z,
                                         double& e_r_x, double& e_r_y, double& e_r_z,
                                         double& R) 
                                         
{
    double dx = x - radar_x;
    double dy = y - radar_y;
    double dz = z - radar_z;

    R = std::sqrt(dx*dx + dy*dy + dz*dz);

    if (R < 1e-6) {
        e_r_x = 0.0;
        e_r_y = 0.0;
        e_r_z = 1.0;
        return;
    }

    e_r_x = dx / R;
    e_r_y = dy / R;
    e_r_z = dz / R;
}

/**
 * @brief Samples full 3D state into a single-cell radar point view.
 */
void RadarGeometry::sample_state_point(const RadarStateView& state, int i, int j, int k,
                                      RadarStateView& point_state) 
{
    point_state = RadarStateView{};
    if (state.NR <= 0 || state.NTH <= 0 || state.NZ <= 0)
    {
        return;
    }

    const int ic = std::max(0, std::min(i, state.NR - 1));
    const int jc = wrap_theta_index(j, state.NTH);
    const int kc = std::max(0, std::min(k, state.NZ - 1));

    thread_local RadarPointScratch scratch;
    initialize_scratch_fields(scratch);

    point_state.NR = 1;
    point_state.NTH = 1;
    point_state.NZ = 1;
    point_state.u = sample_to_single_cell(state.u, ic, jc, kc, scratch.u);
    point_state.v = sample_to_single_cell(state.v, ic, jc, kc, scratch.v);
    point_state.w = sample_to_single_cell(state.w, ic, jc, kc, scratch.w);
    point_state.qr = sample_to_single_cell(state.qr, ic, jc, kc, scratch.qr);
    point_state.qs = sample_to_single_cell(state.qs, ic, jc, kc, scratch.qs);
    point_state.qg = sample_to_single_cell(state.qg, ic, jc, kc, scratch.qg);
    point_state.qh = sample_to_single_cell(state.qh, ic, jc, kc, scratch.qh);
    point_state.qi = sample_to_single_cell(state.qi, ic, jc, kc, scratch.qi);
    point_state.Nr = sample_to_single_cell(state.Nr, ic, jc, kc, scratch.Nr);
    point_state.Ns = sample_to_single_cell(state.Ns, ic, jc, kc, scratch.Ns);
    point_state.Ng = sample_to_single_cell(state.Ng, ic, jc, kc, scratch.Ng);
    point_state.Nh = sample_to_single_cell(state.Nh, ic, jc, kc, scratch.Nh);
    point_state.Ni = sample_to_single_cell(state.Ni, ic, jc, kc, scratch.Ni);
    point_state.theta = sample_to_single_cell(state.theta, ic, jc, kc, scratch.theta);
    point_state.p = sample_to_single_cell(state.p, ic, jc, kc, scratch.p);
}
