#pragma once

#include <vector>

#include "core/field/field3d.hpp"
#include "core/infra/coordinate_system.hpp"
#include "core/output/derived_slice_scratch.hpp"
#include "core/output/field_registry.hpp"
#include "terrain/terrain_base.hpp"

/**
 * @file derived_slice_compute.hpp
 * @brief Context object for computing derived diagnostic slices.
 */

struct DerivedSliceComputationContext
{
    int nr;
    int nth;
    int nz;
    double dr;
    double dtheta;
    double dz;
    double dt;
    CoordinateSystem coordinate_system;
    OutputFieldDependency dependencies;
    int theta_index;
    float reflectivity_dbz_min;
    float reflectivity_dbz_max;
    const std::vector<float>& theta_azimuth_mean;
    const std::vector<float>& storm_u_level_mean;
    const std::vector<float>& storm_v_level_mean;
    const Field3D& rho;
    const Field3D& p;
    const Field3D& u;
    const Field3D& v;
    const Field3D& w;
    const Field3D& theta;
    const Field3D& qv;
    const Field3D& qc;
    const Field3D& qr;
    const Field3D& qi;
    const Field3D& qs;
    const Field3D& qg;
    const Field3D& qh;
    const Field3D& radar_reflectivity;
    const Field3D& radar_radial_velocity;
    const Field3D& radar_zdr;
    const Field3D& radar_kdp;
    const Field3D& radar_rhohv;
    const Field3D& vorticity_r;
    const Field3D& vorticity_theta;
    const Field3D& vorticity_z;
    const Field3D& angular_momentum;
    const std::vector<double>& theta0_base;
    const TerrainMetrics3D* terrain_metrics;
    double coriolis_f;
    const std::vector<double>* rho0_base = nullptr;
    const std::vector<double>* qv0_base = nullptr;
    const std::vector<double>* p0_base = nullptr;
    const std::vector<double>* z_center_m = nullptr;
};

void compute_derived_slices_for_theta(DerivedSliceScratch& derived_slices,
                                      const DerivedSliceComputationContext& context);
