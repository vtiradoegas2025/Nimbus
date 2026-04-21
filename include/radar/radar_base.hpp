#pragma once

#include <string>

#include "core/field3d.hpp"

/**
 * @file radar_base.hpp
 * @brief Base interfaces and state containers for synthetic radar operators.
 *
 * Defines radar configuration, input state views, output fields,
 * and abstract scheme interfaces used by reflectivity/velocity/ZDR modules.
 * Includes geometry helpers for line-of-sight sampling.
 */

namespace RadarConstants
{
constexpr double lambda = 0.1;
constexpr double K_w_squared = 0.93;
constexpr double c = 2.99792458e8;
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

struct RadarStateView
{
    int NR = 0;
    int NTH = 0;
    int NZ = 0;

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
};

struct RadarOut
{
    int NR = 0;
    int NTH = 0;
    int NZ = 0;

    Field3D Z_dBZ;
    Field3D Vr;
    Field3D ZDR_dB;

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
        Ze_linear.resize(nr, nth, nz, 0.0f);
        ZH_dBZ.resize(nr, nth, nz, 0.0f);
        ZV_dBZ.resize(nr, nth, nz, 0.0f);
        Ze_rain.resize(nr, nth, nz, 0.0f);
        Ze_snow.resize(nr, nth, nz, 0.0f);
        Ze_graupel.resize(nr, nth, nz, 0.0f);
        Ze_hail.resize(nr, nth, nz, 0.0f);
        Ze_ice.resize(nr, nth, nz, 0.0f);
    }
};

class RadarSchemeBase
{
public:
    /**
     * @brief Virtual destructor for polymorphic cleanup.
     */
    virtual ~RadarSchemeBase() = default;

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
     */
    static void sample_state_point(const RadarStateView& state,
                                   int i,
                                   int j,
                                   int k,
                                   RadarStateView& point_state);
};
