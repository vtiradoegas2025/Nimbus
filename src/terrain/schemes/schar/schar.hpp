#pragma once
#include "terrain/base/topography.hpp"

namespace tmv
{

 

// Schar mountain used by idealized mountain-wave tests.
class ScharScheme : public TerrainSchemeBase 
{
private:
    TerrainConfig config_;

public:
    ScharScheme();

    std::string name() const override { return "schar"; }
    std::unique_ptr<TerrainSchemeBase> clone() const override
    {
        return std::make_unique<ScharScheme>(*this);
    }

    void initialize(const TerrainConfig& cfg) override;

    /**
 * @brief Builds the topography.
 */
    void build_topography(const TerrainConfig& cfg, Topography2D& topo) override;

    void build_metrics(const TerrainConfig& cfg,
                      const Topography2D& topo,
                      TerrainMetrics3D& metrics,
                      TerrainDiagnostics* diag_opt = nullptr) override;

private:
    // Maps grid indices to physical Cartesian coordinates.
    void get_coordinates(int i, int j, double& x, double& y) const;
};

} // namespace tmv

using tmv::ScharScheme;
