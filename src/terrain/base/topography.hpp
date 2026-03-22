/**
 * @file topography.hpp
 * @brief Declarations for the terrain module.
 *
 * Defines interfaces, data structures, and contracts used by
 * the terrain runtime and scheme implementations.
 * This file is part of the src/terrain subsystem.
 */

#pragma once
#include "physics/terrain_base.hpp"


namespace topography {

/**
 * @brief Terrain height and gradients for a 2D bell mountain profile.
 */
struct BellResult {
    double h;
    double hx;
    double hy;
};

/**
 * @brief Evaluates bell-mountain topography and first derivatives.
 */
BellResult eval_bell(double x, double y, const TerrainConfig::BellParams& params);

/**
 * @brief Terrain height and radial gradient for Schar topography.
 */
struct ScharResult {
    double h;
    double hx;
};

/**
 * @brief Evaluates Schar topography and first radial derivative.
 */
ScharResult eval_schar(double x, const TerrainConfig::ScharParams& params);

/**
 * @brief Terrain-following coordinate transforms and metric evaluation.
 */
class TerrainFollowingCoordinate {
private:
    TerrainConfig config_;
    Topography2D topography_;

public:
    /**
     * @brief Constructs a coordinate transform from config and topography.
     */
    TerrainFollowingCoordinate(const TerrainConfig& cfg, const Topography2D& topo);

    /**
     * @brief Maps terrain-following height coordinate to physical height.
     */
    double zeta_to_z(double zeta, int i, int j) const;

    /**
     * @brief Maps physical height to terrain-following height coordinate.
     */
    double z_to_zeta(double z, int i, int j) const;

    /**
     * @brief Computes physical height and metric terms at one horizontal point.
     */
    void compute_metrics(double zeta, int i, int j,
                        double& z, double& J, double& mx, double& my) const;

    /**
     * @brief Applies configured terrain smoothing to coordinate deformation.
     */
    double apply_coordinate_smoothing(double h, double zeta) const;
};

/**
 * @brief Initializes topography height and slope arrays.
 */
void initialize_topography(Topography2D& topo, int NR, int NTH);

/**
 * @brief Initializes terrain metric tensors and Jacobians.
 */
void initialize_metrics(TerrainMetrics3D& metrics, int NR, int NTH, int NZ);

/**
 * @brief Builds uniformly distributed terrain-following levels up to `ztop`.
 */
std::vector<double> build_zeta_levels(int NZ, double ztop);

/**
 * @brief Validates terrain metrics and populates diagnostic status.
 */
bool check_coordinate_validity(const TerrainMetrics3D& metrics, TerrainDiagnostics& diag);

}
