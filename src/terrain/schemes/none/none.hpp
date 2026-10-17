#pragma once
#include "terrain/base/topography.hpp"

namespace tmv
{

// Flat terrain emits zero topography and identity metrics.
class NoneScheme : public TerrainSchemeBase {
public:
    NoneScheme();
    std::string name() const override { return "none"; }
    std::unique_ptr<TerrainSchemeBase> clone() const override
    {
        return std::make_unique<NoneScheme>(*this);
    }

    void initialize(const TerrainConfig& cfg) override;
    void build_topography(const TerrainConfig& cfg, Topography2D& topo) override;
    void build_metrics(const TerrainConfig& cfg,
                      const Topography2D& topo,
                      TerrainMetrics3D& metrics,
                      TerrainDiagnostics* diag_opt = nullptr) override;
};

} // namespace tmv
