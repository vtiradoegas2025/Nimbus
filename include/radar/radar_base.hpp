#pragma once

#include <memory>
#include <string>

#include "core/field/field3d.hpp"
#include "core/infra/coordinate_system.hpp"
#include "core/infra/grid_geometry.hpp"
#include "terrain/terrain_base.hpp"

/**
 * @file radar_base.hpp
 * @brief Base interfaces and state containers for synthetic radar operators.
 *
 * Defines radar configuration, input state views, output fields,
 * and abstract scheme interfaces used by reflectivity/velocity/ZDR modules.
 * Includes geometry helpers for line-of-sight sampling.
 */

namespace tmv
{

namespace RadarConstants
{
constexpr double lambda = 0.1;
constexpr double K_w_squared = 0.93;
constexpr double c = 2.99792458e8;
// Covers the norm of the accepted (180,180,120) m/s wind components plus
// the 40 m/s bulk scatterer fall-speed cap without truncating a valid beam.
constexpr float velocity_abs_max_ms = 325.0f;
} // namespace RadarConstants

struct RadarConfig;
struct RadarStateView;
struct RadarOut;

struct RadarConfig
{
    std::string scheme_id;
    std::string operator_tier;

    double radar_x = 0.0;
    double radar_y = 0.0;
    double radar_z = 0.0;

    std::string sampling = "point";
    bool apply_scatterer_correction = false;

    bool has_qr = true;
    bool has_qs = false;
    bool has_qg = false;
    bool has_qh = false;
    bool has_qi = false;

    bool has_Nr = false;
    bool has_Ns = false;
    bool has_Ng = false;
    bool has_Nh = false;
    bool has_Ni = false;
};

struct RadarProductSelection
{
    bool reflectivity = false;
    bool radial_velocity = false;
    bool zdr = false;
    bool kdp = false;
    bool rhohv = false;

    std::string reflectivity_tier = "fast_da";
    std::string velocity_sampling = "point";
    std::string zdr_tier = "polarimetric_fo";

    double radar_x = 0.0;
    double radar_y = 0.0;
    double radar_z = 0.0;
    bool apply_scatterer_correction = false;
};

struct RadarStateView
{
    int NR = 0;
    int NTH = 0;
    int NZ = 0;

    /**
     * @brief Placement of the velocity fields referenced below.
     *
     * Scalar and hydrometeor fields are cell-centered for both choices. For
     * CGrid, u/v/w are x/y/z-face values and radar velocity reconstructs them
     * to the scalar center before projecting onto the line of sight.
     */
    StaggerType staggering = StaggerType::Collocated;

    /**
     * Physical coordinate ownership for line-of-sight operators. Geometry is
     * mandatory for velocity; terrain_metrics, when present, supplies exact
     * per-column scalar heights. Offsets preserve the parent-grid location of
     * a sampled one-cell view. The horizontal geometry arrays are local to the
     * computational grid; grid_origin_earth_* places that grid in the same
     * Earth-relative coordinate system as RadarConfig::radar_x/radar_y.
     */
    const GridGeometry* geometry = nullptr;
    const TerrainMetrics3D* terrain_metrics = nullptr;
    int geometry_i_offset = 0;
    int geometry_j_offset = 0;
    int geometry_k_offset = 0;
    double grid_origin_earth_x_m = 0.0;
    double grid_origin_earth_y_m = 0.0;

    const Field3D* u = nullptr;
    const Field3D* v = nullptr;
    const Field3D* w = nullptr;

    const Field3D* qr = nullptr;
    const Field3D* qs = nullptr;
    const Field3D* qg = nullptr;
    const Field3D* qh = nullptr;
    const Field3D* qi = nullptr;

    const Field3D* Nr = nullptr;
    const Field3D* Ns = nullptr;
    const Field3D* Ng = nullptr;
    const Field3D* Nh = nullptr;
    const Field3D* Ni = nullptr;

    const Field3D* theta = nullptr;
    const Field3D* p = nullptr;
    // Prognostic dry-air density (kg_dry/m^3). Hydrometeor q fields are
    // dry-air mixing ratios, so species mass concentration is exactly rho*q.
    const Field3D* rho = nullptr;
};

struct RadarOut
{
    int NR = 0;
    int NTH = 0;
    int NZ = 0;

    Field3D Z_dBZ;
    Field3D Vr;
    Field3D ZDR_dB;
    Field3D KDP_deg_km;
    Field3D rhohv;

    Field3D Ze_linear;
    Field3D ZH_dBZ;
    Field3D ZV_dBZ;

    Field3D Ze_rain;
    Field3D Ze_snow;
    Field3D Ze_graupel;
    Field3D Ze_hail;
    Field3D Ze_ice;

    /**
     * @brief Allocates and zero-initializes all radar output fields.
     * @param nr Radial dimension.
     * @param nth Azimuthal dimension.
     * @param nz Vertical dimension.
     */
    void initialize(int nr, int nth, int nz)
    {
        NR = nr;
        NTH = nth;
        NZ = nz;
        Z_dBZ.resize(nr, nth, nz, 0.0f);
        Vr.resize(nr, nth, nz, 0.0f);
        ZDR_dB.resize(nr, nth, nz, 0.0f);
        KDP_deg_km.resize(nr, nth, nz, 0.0f);
        rhohv.resize(nr, nth, nz, 1.0f);
        Ze_linear.resize(nr, nth, nz, 0.0f);
        ZH_dBZ.resize(nr, nth, nz, 0.0f);
        ZV_dBZ.resize(nr, nth, nz, 0.0f);
        Ze_rain.resize(nr, nth, nz, 0.0f);
        Ze_snow.resize(nr, nth, nz, 0.0f);
        Ze_graupel.resize(nr, nth, nz, 0.0f);
        Ze_hail.resize(nr, nth, nz, 0.0f);
        Ze_ice.resize(nr, nth, nz, 0.0f);
    }

    /** Clear state-derived values without releasing high-resolution storage. */
    void invalidate_generation_values() noexcept
    {
        Z_dBZ.fill(0.0f);
        Vr.fill(0.0f);
        ZDR_dB.fill(0.0f);
        KDP_deg_km.fill(0.0f);
        rhohv.fill(1.0f);
        Ze_linear.fill(0.0f);
        ZH_dBZ.fill(0.0f);
        ZV_dBZ.fill(0.0f);
        Ze_rain.fill(0.0f);
        Ze_snow.fill(0.0f);
        Ze_graupel.fill(0.0f);
        Ze_hail.fill(0.0f);
        Ze_ice.fill(0.0f);
    }
};

/**
 * @brief Bulk rain terminal speed used by synthetic radar corrections.
 * @param q_rain Rain-water dry-air mixing ratio (kg_water/kg_dry).
 * @return Non-negative fall speed (m/s), or zero for invalid input.
 */
float radar_bulk_rain_fall_speed_ms(float q_rain);

/**
 * @brief Dimensionally consistent bulk rain-rate estimate.
 *
 * Computes rho_dry * q_rain * Vt * 3600. The water-mass flux in kg/m2/s
 * converts one-for-one to mm/s of liquid-water depth.
 * @param q_rain Rain-water dry-air mixing ratio (kg_water/kg_dry).
 * @param rho_dry Prognostic dry-air density (kg_dry/m3).
 */
float radar_bulk_rain_rate_mmph(float q_rain, float rho_dry);

/** Fail-closed shared input contracts used by every radar operator. */
void validate_radar_dimensions(const RadarStateView& state,
                               int nr, int nth, int nz,
                               const char* operator_name);
void validate_radar_hydrometeor_state(const RadarConfig& config,
                                      const RadarStateView& state,
                                      const char* operator_name,
                                      bool require_density);
void validate_radar_velocity_state(const RadarConfig& config,
                                   const RadarStateView& state);
void validate_radar_physical_geometry(const RadarStateView& state);

/** Validate a complete candidate product before publication. */
void validate_radar_output_field(const Field3D& field,
                                 int nr, int nth, int nz,
                                 float minimum, float maximum,
                                 const char* field_name);

class RadarSchemeBase
{
public:
    /**
     * @brief Virtual destructor for polymorphic cleanup.
     */
    virtual ~RadarSchemeBase() = default;
    virtual std::unique_ptr<RadarSchemeBase> clone() const = 0;

    /**
     * @brief Initializes scheme resources.
     * @param config Radar operator configuration.
     * @param NR Radial grid dimension.
     * @param NTH Azimuthal grid dimension.
     * @param NZ Vertical grid dimension.
     */
    virtual void initialize(const RadarConfig& config, int NR, int NTH, int NZ) = 0;

    /**
     * @brief Computes radar observables for the given state.
     * @param config Radar operator configuration.
     * @param state Radar state view.
     * @param out Output radar fields.
     */
    virtual void compute(const RadarConfig& config, const RadarStateView& state, RadarOut& out) = 0;

    /**
     * @brief Returns the scheme identifier.
     * @return Scheme name.
     */
    virtual std::string name() const = 0;

    /**
     * Third-party operators fail closed for a recycled window until they
     * explicitly clear every state-generation-local candidate or cache.
     */
    virtual bool supports_domain_recycling_invalidation() const noexcept
    {
        return false;
    }
    virtual void invalidate_generation_state_after_domain_recycling() noexcept
    {
    }
};

class RadarGeometry
{
public:
    /**
     * @brief Computes line-of-sight unit vector and range from radar to point.
     */
    static void compute_line_of_sight(double radar_x,
                                      double radar_y,
                                      double radar_z,
                                      double x,
                                      double y,
                                      double z,
                                      double& e_r_x,
                                      double& e_r_y,
                                      double& e_r_z,
                                      double& R);

    /**
     * @brief Extracts a point sample from a full radar state view.
     *
     * Out-of-domain and boundary-closure indices are clamped to the nearest
     * physical scalar cell; lateral indices are not wrapped periodically.
     * C-grid winds are reconstructed to the sampled scalar center, so the
     * returned one-cell view is marked Collocated.
     */
    static void sample_state_point(const RadarStateView& state,
                                   int i,
                                   int j,
                                   int k,
                                   RadarStateView& point_state);
};

} // namespace tmv

namespace RadarConstants = tmv::RadarConstants;
using RadarConfig = tmv::RadarConfig;
using RadarProductSelection = tmv::RadarProductSelection;
using RadarStateView = tmv::RadarStateView;
using RadarOut = tmv::RadarOut;
using RadarSchemeBase = tmv::RadarSchemeBase;
using RadarGeometry = tmv::RadarGeometry;
using tmv::radar_bulk_rain_fall_speed_ms;
using tmv::radar_bulk_rain_rate_mmph;
