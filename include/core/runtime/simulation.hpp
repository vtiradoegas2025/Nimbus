#pragma once

// Deprecated integration umbrella. Production code includes the owning
// state, configuration, factory, or runtime contract directly.

#include "boundary_conditions/runtime.hpp"
#include "boundary_layer/runtime.hpp"
#include "chaos/runtime.hpp"
#include "core/field/field_bounds.hpp"
#include "core/infra/physical_constants.hpp"
#include "core/runtime/process_state.hpp"
#include "core/runtime/runtime_config.hpp"
#include "dynamics/runtime.hpp"
#include "init/hodograph/wind_profile.hpp"
#include "microphysics/runtime.hpp"
#include "numerics/advection/advection_base.hpp"
#include "numerics/diffusion/diffusion_base.hpp"
#include "numerics/runtime.hpp"
#include "numerics/time_stepping/time_stepping_base.hpp"
#include "radiation/runtime.hpp"
#include "terrain/runtime.hpp"
#include "turbulence/runtime.hpp"
#include "util/log_profile.hpp"

// Legacy unqualified science constants retained for source compatibility.
// New code uses physical_constants or the owning module constants namespace.
inline constexpr double qc0 = 1.0e-3;
inline constexpr double c_auto = 1.0e-3;
inline constexpr double c_accr = 2.2;
inline constexpr double c_evap = 3.0e-3;
inline constexpr double L_v = physical_constants::latent_heat_vaporization_jkg;
inline constexpr double a_term = 65.0;
inline constexpr double b_term = 0.125;
inline constexpr double Vt_max = 20.0;

inline constexpr double T0 = physical_constants::freezing_temperature_k;
inline constexpr double L_f = physical_constants::latent_heat_fusion_jkg;
inline constexpr double L_s = physical_constants::latent_heat_sublimation_jkg;
inline constexpr double c_rime = 1.0;
inline constexpr double c_freeze = 1.0e-3;
inline constexpr double c_melt = 1.0e-3;
inline constexpr double c_subl = 1.0e-3;
inline constexpr double a_hail = 114.0;
inline constexpr double b_hail = 0.5;
inline constexpr double a_grau = 40.0;
inline constexpr double b_grau = 0.37;
inline constexpr double Vt_max_hail = 40.0;
inline constexpr double Vt_max_grau = 15.0;

inline constexpr double sigma_sb = physical_constants::stefan_boltzmann_wm2k4;
inline constexpr double S0_solar = physical_constants::solar_constant_wm2;
