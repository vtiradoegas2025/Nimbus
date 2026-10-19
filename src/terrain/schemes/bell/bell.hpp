#pragma once
#include "terrain/base/topography.hpp"

namespace tmv
{

// Bell mountain used by idealized terrain-flow tests.
class BellScheme : public TerrainSchemeBase 
{
private:
    TerrainConfig config_;

public:
    BellScheme();
    std::string name() const override { return "bell"; }
    std::unique_ptr<TerrainSchemeBase> clone() const override
    {
        return std::make_unique<BellScheme>(*this);
    }

    /**
     * @brief Initializes scheme parameters from terrain configuration.
     */
    void initialize(const TerrainConfig& cfg) override;

    /**
 * @brief Builds the topography.
 */
    void build_topography(const TerrainConfig& cfg, Topography2D& topo) override;

    /**
 * @brief Builds the metrics.
 */
    void build_metrics(const TerrainConfig& cfg,
                      const Topography2D& topo,
                      TerrainMetrics3D& metrics,
                      TerrainDiagnostics* diag_opt = nullptr) override;

private:
    // Maps grid indices to physical Cartesian coordinates.
    void get_coordinates(int i, int j, double& x, double& y) const;
};

} // namespace tmv

using tmv::BellScheme;
