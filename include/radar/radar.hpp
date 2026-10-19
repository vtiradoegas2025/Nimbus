#pragma once

#include "radar/radar_base.hpp"

namespace tmv
{

struct RadarRuntimeState;

class RadarSystem
{
public:
    static void compute_observables(RadarRuntimeState& runtime,
                                    const RadarProductSelection& selection,
                                    const RadarStateView& state,
                                    RadarOut& output);
    static void compute_observables(const RadarProductSelection& selection,
                                    const RadarStateView& state,
                                    RadarOut& output);

    // Explicit-runtime overloads retain initialized operators across frames.
    static void compute_all_observables(RadarRuntimeState& runtime,
                                        const RadarStateView& state,
                                        RadarOut& output,
                                        double radar_x = 0.0,
                                        double radar_y = 0.0,
                                        double radar_z = 0.0);
    static void compute_all_observables(const RadarStateView& state,
                                        RadarOut& output,
                                        double radar_x = 0.0,
                                        double radar_y = 0.0,
                                        double radar_z = 0.0);

    static void compute_reflectivity(RadarRuntimeState& runtime,
                                     const RadarStateView& state,
                                     RadarOut& output);
    static void compute_reflectivity(const RadarStateView& state, RadarOut& output);

    static void compute_velocity(RadarRuntimeState& runtime,
                                 const RadarStateView& state,
                                 RadarOut& output,
                                 double radar_x = 0.0,
                                 double radar_y = 0.0,
                                 double radar_z = 0.0);
    static void compute_velocity(const RadarStateView& state,
                                 RadarOut& output,
                                 double radar_x = 0.0,
                                 double radar_y = 0.0,
                                 double radar_z = 0.0);

    static void compute_zdr(RadarRuntimeState& runtime,
                            const RadarStateView& state,
                            RadarOut& output);
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
inline constexpr char ZHANG2021_RAIN_MP[] = "zhang2021_rain_mp";
} // namespace ZDR

namespace Velocity
{
inline constexpr char POINT[] = "point";
inline constexpr char BEAM_VOLUME[] = "beam_volume";
} // namespace Velocity
} // namespace RadarTiers

} // namespace tmv

using RadarSystem = tmv::RadarSystem;
namespace RadarSchemes = tmv::RadarSchemes;
namespace RadarTiers = tmv::RadarTiers;
