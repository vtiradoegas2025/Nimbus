#pragma once

#include <cassert>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "core/infra/coordinate_system.hpp"
#include "core/infra/domain_motion.hpp"

// Terrain-scheme contracts, topography, metrics, and persistent runtime state.

namespace tmv
{

namespace terrain_constants
{
inline constexpr double epsilon = 1e-12;
} // namespace terrain_constants

/**
 * @brief Whether the dynamics solver selected by (coordinate, stagger)
 *        implements terrain-following metric corrections.
 *
 * The Cartesian C-grid CPU foundation now has Jacobian-weighted continuity,
 * retained scalar fluxes, contravariant momentum advection, and a
 * well-balanced terrain pressure force. Physical-height initialization and a
 * downstream column-geometry tranche are also present. The declared CPU scope
 * owns physical column geometry, mapped SGS, diffusion, radiation,
 * diagnostics, and output. Exact stationary-Bell and grid-attached
 * translating-Schär C-grid profiles with PBL traction and implicit scalar
 * diffusion are Ready; other cross-products do not inherit that admission.
 * Complete accelerator execution and Earth-fixed terrain remapping/recycling
 * remain separately fail-closed. The Cartesian A-grid owns the corresponding
 * cell-centred mapped gradients, pressure-coupled contravariant mass flux,
 * scalar history, and tangent closure for its exact CPU profile.
 */
inline bool dynamics_supports_terrain(CoordinateSystem coordinate, StaggerType stagger)
{
    return coordinate == CoordinateSystem::Cartesian
        && (stagger == StaggerType::CGrid
            || stagger == StaggerType::Collocated);
}

struct TerrainConfig
{
    std::string scheme_id = "none";
    std::string coord_id = "btf";
    std::string horizontal_placement_id = "grid_attached";
    double ztop = 20000.0;
    bool compute_derivatives = true;
    bool compute_metrics = true;
    double feature_center_earth_x_m = 0.0;
    double feature_center_earth_y_m = 0.0;

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
    // Cell-centred physical height and point metrics.  The metric convention is
    // J = dz/dzeta, mx = dzeta/dx = -(dz/dx)/J, and similarly for my.
    TerrainField3D z;
    TerrainField3D J;
    TerrainField3D mx;
    TerrainField3D my;
    TerrainField3D zeta;

    // Finite-volume metric set used by conservative transport.  z_face,
    // Gx_face, and Gy_face live on zeta interfaces (NZ+1).  Jx_face and
    // Jy_face live on the vertical x/y faces.  Gx_face = J*mx = -dz/dx and
    // Gy_face = J*my = -dz/dy.  These fields are constructed from one shared
    // discrete height map so their metric identities cancel algebraically.
    TerrainField3D z_face;
    TerrainField3D Jx_face;
    TerrainField3D Jy_face;
    TerrainField3D Gx_face;
    TerrainField3D Gy_face;
    double dx = 0.0;
    double dy = 0.0;
    double dzeta = 0.0;
};

/**
 * @brief Physical vector fluxes sampled on the computational cell faces.
 *
 * x is F_x on x faces, y is F_y on y faces, and z_x/z_y/z_vertical are
 * F_x/F_y/F_z collocated on zeta faces.  The latter three are needed because
 * a sloping zeta face has a non-vertical physical normal.
 */
struct TerrainPhysicalFluxes
{
    TerrainField3D x;
    TerrainField3D y;
    TerrainField3D z_x;
    TerrainField3D z_y;
    TerrainField3D z_vertical;
};

/**
 * @brief Jacobian-weighted contravariant fluxes on computational faces.
 */
struct TerrainContravariantFluxes
{
    TerrainField3D x;
    TerrainField3D y;
    TerrainField3D z;
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

enum class EarthFixedTerrainSampling : std::uint8_t
{
    Point = 0,
    CellAverageBilinear = 1,
};

struct EarthFixedTerrainPlacement
{
    bool active = false;
    int nr = 0;
    int nth = 0;
    int nz = 0;
    double dx_m = 0.0;
    double dy_m = 0.0;
    domain::HorizontalPoint grid_origin_earth_m;
    domain::HorizontalPoint feature_center_earth_m;
    std::uint64_t generation = 0;
    EarthFixedTerrainSampling sampling = EarthFixedTerrainSampling::Point;
};

class TerrainSchemeBase
{
public:
    virtual ~TerrainSchemeBase() = default;
    virtual std::unique_ptr<TerrainSchemeBase> clone() const { return nullptr; }

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

struct TerrainRuntimeState
{
    std::unique_ptr<TerrainSchemeBase> scheme;
    TerrainConfig config;
    Topography2D topography;
    TerrainMetrics3D metrics;
    EarthFixedTerrainPlacement earth_fixed_placement;

    TerrainRuntimeState() = default;
    TerrainRuntimeState(const TerrainRuntimeState& other)
        : config(other.config), topography(other.topography), metrics(other.metrics),
          earth_fixed_placement(other.earth_fixed_placement)
    {
        if (other.scheme)
        {
            scheme = other.scheme->clone();
            if (!scheme)
            {
                throw std::runtime_error(
                    "terrain scheme does not support runtime snapshots: "
                    + other.scheme->name());
            }
        }
    }
    TerrainRuntimeState& operator=(const TerrainRuntimeState& other)
    {
        if (this != &other)
        {
            TerrainRuntimeState copy(other);
            *this = std::move(copy);
        }
        return *this;
    }
    TerrainRuntimeState(TerrainRuntimeState&&) noexcept = default;
    TerrainRuntimeState& operator=(TerrainRuntimeState&&) noexcept = default;
};

} // namespace tmv
#include "terrain/factory.hpp"
#include "terrain/runtime.hpp"

namespace terrain_constants = tmv::terrain_constants;
using tmv::dynamics_supports_terrain;
using tmv::TerrainConfig;
using tmv::Topography2D;
using tmv::TerrainField3D;
using tmv::TerrainMetrics3D;
using tmv::TerrainPhysicalFluxes;
using tmv::TerrainContravariantFluxes;
using tmv::TerrainDiagnostics;
using tmv::TerrainSchemeBase;
using tmv::TerrainRuntimeState;
using tmv::TerrainSchemeFactory;
using tmv::create_terrain_scheme;
using tmv::get_available_terrain_schemes;
