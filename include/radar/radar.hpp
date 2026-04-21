#pragma once

#include "radar/radar_base.hpp"

/**
 * @file radar.hpp
 * @brief High-level radar-observable API used by runtime diagnostics.
 *
 * Provides convenience static methods that compute synthetic radar
 * products from model state fields.
 * Also defines scheme/tier name constants used in configuration.
 */

class RadarSchemeBase;
struct RadarConfig;
struct RadarStateView;
struct RadarOut;
class RadarGeometry;

class RadarFactory;

class RadarSystem
{
public:
    /**
     * @brief Computes all configured radar observables.
     * @param state Radar operator state view.
     * @param output Radar output container.
     * @param radar_x Radar x position in meters.
     * @param radar_y Radar y position in meters.
     * @param radar_z Radar z position in meters.
     */
    static void compute_all_observables(const RadarStateView& state,
                                        RadarOut& output,
                                        double radar_x = 0.0,
                                        double radar_y = 0.0,
                                        double radar_z = 0.0);

    /**
     * @brief Computes reflectivity from hydrometeor fields.
     * @param state Radar operator state view.
     * @param output Radar output container.
     */
    static void compute_reflectivity(const RadarStateView& state, RadarOut& output);

    /**
     * @brief Computes radial velocity relative to radar location.
     * @param state Radar operator state view.
     * @param output Radar output container.
     * @param radar_x Radar x position in meters.
     * @param radar_y Radar y position in meters.
     * @param radar_z Radar z position in meters.
     */
    static void compute_velocity(const RadarStateView& state,
                                 RadarOut& output,
                                 double radar_x = 0.0,
                                 double radar_y = 0.0,
                                 double radar_z = 0.0);

    /**
     * @brief Computes differential reflectivity.
     * @param state Radar operator state view.
     * @param output Radar output container.
     */
    static void compute_zdr(const RadarStateView& state, RadarOut& output);
};

namespace RadarSchemes
{
inline constexpr char REFLECTIVITY[] = "reflectivity";
inline constexpr char VELOCITY[] = "velocity";
inline constexpr char ZDR[] = "zdr";
} // namespace RadarSchemes

namespace RadarTiers
{
namespace Reflectivity
{
inline constexpr char FAST_DA[] = "fast_da";
inline constexpr char PSD_MOMENT[] = "psd_moment";
} // namespace Reflectivity

namespace ZDR
{
inline constexpr char SIMPLE[] = "simple";
inline constexpr char POLARIMETRIC_FO[] = "polarimetric_fo";
} // namespace ZDR

namespace Velocity
{
inline constexpr char POINT[] = "point";
inline constexpr char BEAM_VOLUME[] = "beam_volume";
} // namespace Velocity
} // namespace RadarTiers
