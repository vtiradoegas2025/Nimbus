#pragma once

#include <cassert>
#include <memory>
#include <string>
#include <vector>

/**
 * @file terrain_base.hpp
 * @brief Terrain/topography interfaces and terrain-following metric containers.
 *
 * Declares configuration types and data structures for topography and
 * terrain-following coordinates.
 * Provides base scheme contracts and global terrain subsystem entry points.
 */

namespace terrain_constants
{
inline constexpr double epsilon = 1e-12;
} // namespace terrain_constants

struct TerrainConfig
{
    std::string scheme_id = "none";
    std::string coord_id = "btf";
    double ztop = 20000.0;
    bool compute_derivatives = true;
    bool compute_metrics = true;

    struct BellParams
    {
        double h0 = 1000.0;
        double a = 5000.0;
        bool axisymmetric = false;
    } bell;

    struct ScharParams
    {
        double H = 1000.0;
        double a = 5000.0;
        double ell = 4000.0;
    } schar;

    double smoothing_decay = 0.1;
};

struct Topography2D
{
    std::vector<std::vector<double>> h;
    std::vector<std::vector<double>> hx;
    std::vector<std::vector<double>> hy;
};

struct TerrainField3D
{
    int nr = 0;
    int nth = 0;
    int nz = 0;
    std::vector<double> values;

    /**
     * @brief Resizes and initializes the 3D terrain field.
     * @param nr_in Radial dimension.
     * @param nth_in Azimuthal dimension.
     * @param nz_in Vertical dimension.
     * @param init_value Fill value.
     */
    void resize(int nr_in, int nth_in, int nz_in, double init_value)
    {
        nr = nr_in;
        nth = nth_in;
        nz = nz_in;
        values.assign(static_cast<std::size_t>(nr) * static_cast<std::size_t>(nth) *
                          static_cast<std::size_t>(nz),
                      init_value);
    }

    /**
     * @brief Returns radial size.
     */
    int size_r() const { return nr; }

    /**
     * @brief Returns azimuthal size.
     */
    int size_th() const { return nth; }

    /**
     * @brief Returns vertical size.
     */
    int size_z() const { return nz; }

    /**
     * @brief Returns flattened element count.
     */
    std::size_t size() const { return values.size(); }

    /**
     * @brief Returns true when the storage is empty.
     */
    bool empty() const { return values.empty(); }

    /**
     * @brief Mutable element access with bounds assertions.
     */
    double& operator()(int i, int j, int k)
    {
        assert(i >= 0 && i < nr && j >= 0 && j < nth && k >= 0 && k < nz);
        // Row-major: idx = i*nth*nz + j*nz + k.
        return values[static_cast<std::size_t>(i) * static_cast<std::size_t>(nth) *
                          static_cast<std::size_t>(nz) +
                      static_cast<std::size_t>(j) * static_cast<std::size_t>(nz) +
                      static_cast<std::size_t>(k)];
    }

    /**
     * @brief Const element access with bounds assertions.
     */
    const double& operator()(int i, int j, int k) const
    {
        assert(i >= 0 && i < nr && j >= 0 && j < nth && k >= 0 && k < nz);
        return values[static_cast<std::size_t>(i) * static_cast<std::size_t>(nth) *
                          static_cast<std::size_t>(nz) +
                      static_cast<std::size_t>(j) * static_cast<std::size_t>(nz) +
                      static_cast<std::size_t>(k)];
    }
};

struct TerrainMetrics3D
{
    TerrainField3D z;
    TerrainField3D J;
    TerrainField3D mx;
    TerrainField3D my;
    TerrainField3D zeta;
};

struct TerrainDiagnostics
{
    double max_height = 0.0;
    double max_slope_x = 0.0;
    double max_slope_y = 0.0;
    double min_jacobian = 1.0;
    double max_jacobian = 1.0;
    bool coordinate_folding = false;
    std::vector<std::string> warnings;
};

class TerrainSchemeBase
{
public:
    /**
     * @brief Virtual destructor for polymorphic cleanup.
     */
    virtual ~TerrainSchemeBase() = default;

    /**
     * @brief Returns the scheme identifier.
     * @return Scheme name.
     */
    virtual std::string name() const = 0;

    /**
     * @brief Initializes terrain scheme internals.
     * @param cfg Terrain configuration.
     */
    virtual void initialize(const TerrainConfig& cfg) = 0;

    /**
     * @brief Builds 2D topography fields.
     * @param cfg Terrain configuration.
     * @param topo Output topography fields.
     */
    virtual void build_topography(const TerrainConfig& cfg, Topography2D& topo) = 0;

    /**
     * @brief Builds terrain-following metrics from topography.
     * @param cfg Terrain configuration.
     * @param topo Input topography fields.
     * @param metrics Output terrain metrics.
     * @param diag_opt Optional diagnostics output.
     */
    virtual void build_metrics(const TerrainConfig& cfg,
                               const Topography2D& topo,
                               TerrainMetrics3D& metrics,
                               TerrainDiagnostics* diag_opt = nullptr) = 0;
};

using TerrainSchemeFactory = std::unique_ptr<TerrainSchemeBase> (*)(const TerrainConfig&);

extern std::unique_ptr<TerrainSchemeBase> terrain_scheme;
extern TerrainConfig global_terrain_config;
extern Topography2D global_topography;
extern TerrainMetrics3D global_terrain_metrics;

/**
 * @brief Creates a terrain scheme by name.
 * @param scheme_name Scheme identifier.
 * @return Owning pointer to a scheme instance.
 */
std::unique_ptr<TerrainSchemeBase> create_terrain_scheme(const std::string& scheme_name);

/**
 * @brief Lists registered terrain schemes.
 * @return Collection of scheme identifiers.
 */
std::vector<std::string> get_available_terrain_schemes();

/**
 * @brief Initializes the global terrain subsystem.
 * @param scheme_name Scheme identifier.
 * @param cfg Optional terrain configuration.
 */
void initialize_terrain(const std::string& scheme_name, const TerrainConfig& cfg = TerrainConfig{});

/**
 * @brief Builds terrain fields for the current grid.
 */
void build_terrain_fields();
