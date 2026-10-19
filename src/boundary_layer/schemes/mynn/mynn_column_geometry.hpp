#pragma once

#include "boundary_layer/boundary_layer_base.hpp"

namespace tmv::mynn
{
struct InterfaceGradients
{
    double lower_weight;
    double upper_weight;
    double shear_s2;
    double buoyancy_s2;
};

inline InterfaceGradients interface_gradients(
    const BoundaryLayerColumnStateView& column, std::size_t face)
{
    const std::size_t lower = face - 1;
    const double dz_lower = column.z_int[face] - column.z_int[lower];
    const double dz_upper = column.z_int[face + 1] - column.z_int[face];
    const double dz_sum = dz_lower + dz_upper;
    const double lower_weight = dz_upper / dz_sum;
    const double upper_weight = dz_lower / dz_sum;
    const double distance = 0.5 * dz_sum;
    const double du = (column.u[face] - column.u[lower]) / distance;
    const double dv = (column.v[face] - column.v[lower]) / distance;
    const double theta_v_lower = column.theta[lower] * (1.0 + 0.61 * column.qv[lower]);
    const double theta_v_upper = column.theta[face] * (1.0 + 0.61 * column.qv[face]);
    const double theta_v = lower_weight * theta_v_lower + upper_weight * theta_v_upper;
    return {lower_weight, upper_weight, du * du + dv * dv,
            -boundary_layer_constants::g / theta_v * (theta_v_upper - theta_v_lower) / distance};
}
}
