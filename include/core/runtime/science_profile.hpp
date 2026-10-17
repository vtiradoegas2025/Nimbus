#pragma once

#include "util/scheme_descriptor.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <vector>

namespace tmv::runtime
{

struct SelectedScheme
{
    std::string                          component;
    std::string                          id;
    const std::vector<SchemeDescriptor>* descriptors = nullptr;
};

/** Backend-independent scientific constants for SCI-317's one locally
 * qualified simple-grey profile. Configuration preflight and projected
 * runtime admission both construct this view, so the numeric envelope cannot
 * drift between them.
 */
struct QualifiedSimpleGreyLongwaveSelection
{
    std::string scheme_id;
    std::string solar_forcing;
    bool        do_lw                 = false;
    bool        do_sw                 = false;
    bool        clear_sky_only        = false;
    bool        use_subcycling        = false;
    bool        enable_diagnostics    = false;
    double      cadence_s             = 0.0;
    double      tau_lw_ref            = 0.0;
    double      n_lw                  = 0.0;
    double      surface_emissivity    = 0.0;
    double      surface_temperature_k = 0.0;
};

inline bool qualified_profile_number(double value, double reference)
{
    return std::isfinite(value) && std::abs(value - reference) <= 8.0 * std::numeric_limits<double>::epsilon() *
                                                                      std::max(1.0, std::abs(reference));
}

inline bool is_exact_qualified_simple_grey_longwave(const QualifiedSimpleGreyLongwaveSelection& selection)
{
    return selection.scheme_id == "simple_grey" && selection.solar_forcing == "idealized_diurnal" && selection.do_lw &&
           !selection.do_sw && selection.clear_sky_only && !selection.use_subcycling && !selection.enable_diagnostics &&
           qualified_profile_number(selection.cadence_s, 2.0) && qualified_profile_number(selection.tau_lw_ref, 2.5) &&
           qualified_profile_number(selection.n_lw, 1.0) &&
           qualified_profile_number(selection.surface_emissivity, 0.96) &&
           qualified_profile_number(selection.surface_temperature_k, 290.0);
}

struct QualifiedSimpleGreyAbsorbingWaterCloudSelection
{
    std::string scheme_id;
    std::string optical_model;
    std::string solar_forcing;
    bool        do_lw                       = false;
    bool        do_sw                       = false;
    bool        clear_sky_only              = true;
    bool        use_subcycling              = false;
    std::size_t subcycle_count              = 0;
    bool        enable_diagnostics          = false;
    double      cadence_s                   = 0.0;
    double      tau_lw_ref                  = 0.0;
    double      tau_sw_ref                  = 0.0;
    double      n_lw                        = 0.0;
    double      n_sw                        = 0.0;
    double      lw_vapor_absorption_m2kg    = 0.0;
    double      lw_liquid_absorption_m2kg   = 0.0;
    double      lw_ice_absorption_m2kg      = 0.0;
    double      sw_vapor_absorption_m2kg    = 0.0;
    double      sw_liquid_absorption_m2kg   = 0.0;
    double      sw_ice_absorption_m2kg      = 0.0;
    double      latitude_deg                = 0.0;
    double      declination_deg             = 0.0;
    double      local_solar_time_at_start_h = 0.0;
    double      surface_albedo              = 0.0;
    double      surface_emissivity          = 0.0;
    double      surface_temperature_k       = 0.0;
};

inline bool
is_exact_qualified_simple_grey_absorbing_water_cloud(const QualifiedSimpleGreyAbsorbingWaterCloudSelection& selection)
{
    return selection.scheme_id == "simple_grey" && selection.optical_model == "absorbing_water_cloud_path_v1" &&
           selection.solar_forcing == "idealized_diurnal" && selection.do_lw && selection.do_sw &&
           !selection.clear_sky_only && selection.use_subcycling && selection.subcycle_count == 2 &&
           selection.enable_diagnostics && qualified_profile_number(selection.cadence_s, 2.0) &&
           qualified_profile_number(selection.tau_lw_ref, 2.5) &&
           qualified_profile_number(selection.tau_sw_ref, 0.15) && qualified_profile_number(selection.n_lw, 1.0) &&
           qualified_profile_number(selection.n_sw, 1.0) &&
           qualified_profile_number(selection.lw_vapor_absorption_m2kg, 0.02) &&
           qualified_profile_number(selection.lw_liquid_absorption_m2kg, 85.0) &&
           qualified_profile_number(selection.lw_ice_absorption_m2kg, 45.0) &&
           qualified_profile_number(selection.sw_vapor_absorption_m2kg, 0.001) &&
           qualified_profile_number(selection.sw_liquid_absorption_m2kg, 20.0) &&
           qualified_profile_number(selection.sw_ice_absorption_m2kg, 12.0) &&
           qualified_profile_number(selection.latitude_deg, 0.0) &&
           qualified_profile_number(selection.declination_deg, 0.0) &&
           qualified_profile_number(selection.local_solar_time_at_start_h, 12.0) &&
           qualified_profile_number(selection.surface_albedo, 0.2) &&
           qualified_profile_number(selection.surface_emissivity, 0.96) &&
           qualified_profile_number(selection.surface_temperature_k, 290.0);
}

/** Apply the exact public Kessler qualification envelope before allocation. */
inline void append_qualified_kessler_scope_error(std::vector<std::string>& errors,
                                                 bool                      qualified,
                                                 bool                      kessler_selected,
                                                 bool                      nesting_enabled,
                                                 bool                      exact_public_profile)
{
    if (qualified && kessler_selected && !nesting_enabled && !exact_public_profile)
    {
        errors.push_back("microphysics.scheme=kessler is qualified only in an exact public reference profile");
    }
}

inline void append_qualified_lin_scope_error(std::vector<std::string>& errors,
                                             bool                      qualified,
                                             bool                      lin_selected,
                                             bool                      exact_profile)
{
    if (qualified && lin_selected && !exact_profile)
    {
        errors.push_back("microphysics.scheme=lin is qualified only for maintained flat, RK, mapped, Earth-fixed, or stationary two-way nesting profiles "
                         "with declared backend ownership");
    }
}

struct QualifiedTkeSelection
{
    std::string scheme_id;
    std::string backend;
    std::string coordinate_system;
    std::string staggering;
    std::string mode;
    std::string microphysics;
    std::string boundary_layer;
    std::string surface;
    std::string terrain;
    std::string chaos;
    std::string diffusion;
    std::string diffusion_target;
    std::string west;
    std::string east;
    std::string south;
    std::string north;
    std::string bottom;
    std::string top;
    bool        surface_fluxes     = false;
    bool        physical_diffusion = false;
    bool        rayleigh_damping   = false;
    bool        lateral_sponge     = false;
    bool        w_only_top_sponge  = false;
    double      prandtl            = 0.0;
    double      schmidt            = 0.0;
    std::string stability_correction;
};

inline bool is_exact_qualified_tke_profile(const QualifiedTkeSelection& selection)
{
    const bool diffusion = (selection.diffusion == "explicit" && selection.diffusion_target == "scalars" &&
                            !selection.physical_diffusion) ||
                           (selection.diffusion == "implicit" && selection.diffusion_target == "scalars_and_momentum" &&
                            selection.physical_diffusion);
    return selection.scheme_id == "tke" &&
           (selection.backend == "cpu" || selection.backend == "gpu" || selection.backend == "metal") &&
           selection.coordinate_system == "cartesian" && selection.staggering == "c_grid" && selection.mode == "3d" &&
           (selection.microphysics == "none" || selection.microphysics == "thompson") &&
           selection.boundary_layer == "none" && selection.surface == "none" && !selection.surface_fluxes &&
           selection.terrain == "none" && selection.chaos == "none" && diffusion && !selection.rayleigh_damping &&
           !selection.lateral_sponge && !selection.w_only_top_sponge && selection.west == "periodic" &&
           selection.east == "periodic" && selection.south == "periodic" && selection.north == "periodic" &&
           selection.bottom == "rigid_slip" && selection.top == "rigid" &&
           selection.stability_correction == "tke_based" && qualified_profile_number(selection.prandtl, 1.0 / 3.0) &&
           qualified_profile_number(selection.schmidt, 1.0 / 3.0);
}

inline void append_qualified_tke_scope_error(std::vector<std::string>&    errors,
                                             bool                         qualified,
                                             const QualifiedTkeSelection& selection)
{
    if (qualified && selection.scheme_id == "tke" && !is_exact_qualified_tke_profile(selection))
    {
        errors.push_back("turbulence.scheme=tke requires the maintained flat periodic C-grid reference "
                         "with its declared explicit or implicit diffusion ownership");
    }
}

struct QualifiedWeno5Selection
{
    std::string scheme_id;
    std::string backend;
    std::string coordinate_system;
    std::string staggering;
    std::string dynamics;
    std::string microphysics;
    std::string turbulence;
    std::string boundary_layer;
    std::string surface;
    std::string radiation;
    std::string terrain;
    std::string chaos;
    std::string diffusion;
    std::string diffusion_target;
    std::string west;
    std::string east;
    std::string south;
    std::string north;
    std::string bottom;
    std::string top;
    std::string domain_motion;
    bool        surface_fluxes     = false;
    bool        physical_diffusion = false;
    bool        rayleigh_damping   = false;
    bool        lateral_sponge     = false;
    bool        w_only_top_sponge  = false;
    bool        nesting            = false;
};

inline bool is_exact_qualified_weno5_profile(const QualifiedWeno5Selection& selection)
{
    return selection.scheme_id == "weno5" && selection.backend == "cpu" && selection.coordinate_system == "cartesian" &&
           selection.staggering == "c_grid" && selection.dynamics == "klemp_wilhelmson_split_cartesian_cgrid" &&
           selection.microphysics == "none" && selection.turbulence == "none" && selection.boundary_layer == "none" &&
           selection.surface == "none" && !selection.surface_fluxes && selection.radiation == "rrtmg" &&
           selection.terrain == "none" && selection.chaos == "none" && selection.diffusion == "explicit" &&
           selection.diffusion_target == "scalars" && !selection.physical_diffusion && selection.west == "periodic" &&
           selection.east == "periodic" && selection.south == "periodic" && selection.north == "periodic" &&
           selection.bottom == "rigid_slip" && selection.top == "rigid" && selection.domain_motion == "stationary" &&
           !selection.rayleigh_damping && !selection.lateral_sponge && !selection.w_only_top_sponge &&
           !selection.nesting;
}

inline void append_qualified_weno5_scope_error(std::vector<std::string>&      errors,
                                               bool                           qualified,
                                               const QualifiedWeno5Selection& selection)
{
    if (qualified && selection.scheme_id == "weno5" && !is_exact_qualified_weno5_profile(selection))
    {
        errors.push_back("advection.scheme=weno5 is qualified only for the exact flat "
                         "periodic C-grid CPU RRTMG reference profile");
    }
}

struct QualifiedImplicitDiffusionSelection
{
    std::string scheme_id;
    std::string backend;
    std::string coordinate_system;
    std::string staggering;
    std::string dynamics;
    std::string advection;
    std::string microphysics;
    std::string turbulence;
    std::string boundary_layer;
    std::string surface;
    std::string radiation;
    std::string terrain;
    std::string chaos;
    std::string target;
    std::string west;
    std::string east;
    std::string south;
    std::string north;
    std::string bottom;
    std::string top;
    std::string domain_motion;
    bool        surface_fluxes         = false;
    bool        variable_coefficient   = false;
    bool        rayleigh_damping       = false;
    bool        lateral_sponge         = false;
    bool        w_only_top_sponge      = false;
    bool        nesting                = false;
    double      horizontal_coefficient = 0.0;
    double      vertical_coefficient   = 0.0;
};

inline bool is_exact_qualified_implicit_diffusion_profile(const QualifiedImplicitDiffusionSelection& selection)
{
    return selection.scheme_id == "implicit" && selection.backend == "cpu" &&
           selection.coordinate_system == "cartesian" && selection.staggering == "c_grid" &&
           selection.dynamics == "klemp_wilhelmson_split_cartesian_cgrid" && selection.advection == "tvd" &&
           selection.microphysics == "none" && selection.turbulence == "none" && selection.boundary_layer == "none" &&
           selection.surface == "none" && !selection.surface_fluxes && selection.radiation == "rrtmg" &&
           selection.terrain == "none" && selection.chaos == "none" && selection.target == "scalars" &&
           selection.west == "periodic" && selection.east == "periodic" && selection.south == "periodic" &&
           selection.north == "periodic" && selection.bottom == "rigid_slip" && selection.top == "rigid" &&
           selection.domain_motion == "stationary" && !selection.variable_coefficient && !selection.rayleigh_damping &&
           !selection.lateral_sponge && !selection.w_only_top_sponge && !selection.nesting &&
           qualified_profile_number(selection.horizontal_coefficient, 0.0) &&
           qualified_profile_number(selection.vertical_coefficient, 20.0);
}

inline bool is_structurally_ready_implicit_diffusion_profile(
    const QualifiedImplicitDiffusionSelection& selection)
{
    return selection.scheme_id == "implicit" && selection.backend == "cpu" &&
           selection.coordinate_system == "cartesian" &&
           ((selection.staggering == "c_grid" && selection.dynamics ==
                 "klemp_wilhelmson_split_cartesian_cgrid") ||
            (selection.staggering == "collocated" && selection.dynamics ==
                 "klemp_wilhelmson_split_cartesian_agrid")) &&
           selection.terrain == "none" &&
           (selection.target == "scalars" ||
            selection.target == "scalars_and_momentum") &&
           !selection.variable_coefficient && !selection.nesting &&
           std::isfinite(selection.horizontal_coefficient) &&
           selection.horizontal_coefficient >= 0.0 &&
           std::isfinite(selection.vertical_coefficient) &&
           selection.vertical_coefficient >= 0.0;
}

inline void append_qualified_implicit_diffusion_scope_error(std::vector<std::string>&                  errors,
                                                            bool                                       qualified,
                                                            const QualifiedImplicitDiffusionSelection& selection,
                                                            bool exact_mapped_terrain_profile = false,
                                                            bool exact_native_profile = false)
{
    if (qualified && selection.scheme_id == "implicit" &&
        !is_structurally_ready_implicit_diffusion_profile(selection) &&
        !exact_mapped_terrain_profile && !exact_native_profile)
    {
        errors.push_back("diffusion.scheme=implicit requires the ready constant-coefficient "
                         "Cartesian CPU scalar or scalar-and-momentum envelope, or a "
                         "ready mapped-terrain CPU profile");
    }
}

inline void append_qualified_terrain_scope_error(std::vector<std::string>& errors,
                                                 bool                      qualified,
                                                 const std::string&        terrain_id,
                                                 bool                      exact_mapped_terrain_profile)
{
    if (qualified && terrain_id != "none" && !exact_mapped_terrain_profile)
    {
        errors.push_back("qualified mapped terrain is limited to the exact stationary-bell "
                         "and constant-velocity Schar CPU reference profiles or the exact "
                         "dry stationary Bell/Schar fallback-off Vulkan/Metal mapped-core profiles");
    }
}

/** Machine-readable constants for cross-component science ownership.
 *
 * The C++ runtime validator and the exact-binary scheme catalog both consume
 * this object. Studio's native preflight can therefore evaluate the same
 * policy without maintaining an independent list of allowed combinations.
 */
struct ScienceInteractionPolicy
{
    static const std::string& inactive_scheme_id()
    {
        static const std::string value = "none";
        return value;
    }

    static const std::vector<std::string>& horizontal_only_turbulence_modes()
    {
        static const std::vector<std::string> values = {
            "2d",
            "horizontal_only",
        };
        return values;
    }

    static const std::vector<std::string>& known_surface_schemes()
    {
        static const std::vector<std::string> values = {
            "none",
            "slab",
        };
        return values;
    }

    static const std::string& active_surface_scheme_id()
    {
        static const std::string value = "slab";
        return value;
    }

    static const std::vector<std::string>& terrain_radiation_schemes()
    {
        static const std::vector<std::string> values = {
            "none",
            "simple_grey",
            "rrtmg",
        };
        return values;
    }

    static const std::string& terrain_longwave_radiation_scheme_id()
    {
        static const std::string value = "rrtmg";
        return value;
    }

    static const std::vector<std::string>& terrain_boundary_layer_schemes()
    {
        static const std::vector<std::string> values = {
            "none",
            "slab",
            "ysu",
            "mynn",
        };
        return values;
    }

    static const std::vector<std::string>& terrain_turbulence_schemes()
    {
        static const std::vector<std::string> values = {
            "none",
            "smagorinsky",
            "tke",
        };
        return values;
    }

    static const std::vector<std::string>& terrain_chaos_schemes()
    {
        static const std::vector<std::string> values = {
            "none",
            "initial_conditions",
            "boundary_layer",
            "full_stochastic",
        };
        return values;
    }

    static const std::vector<std::string>& terrain_active_diffusion_schemes()
    {
        static const std::vector<std::string> values = {
            "explicit",
            "implicit",
        };
        return values;
    }

    static const std::vector<std::string>& terrain_active_diffusion_targets()
    {
        static const std::vector<std::string> values = {
            "scalars",
            "momentum",
            "scalars_and_momentum",
            "all",
        };
        return values;
    }

    static const std::vector<std::string>& qualified_zfp_terrain_schemes()
    {
        static const std::vector<std::string> values = {
            "none", "bell", "schar"};
        return values;
    }

    static const std::vector<std::string>& lateral_boundary_modes()
    {
        static const std::vector<std::string> values = {
            "periodic",
            "radiative",
            "zero_gradient",
            "open",
            "rigid",
            "rigid_slip",
        };
        return values;
    }

    static const std::vector<std::string>& radiative_lateral_staggerings()
    {
        static const std::vector<std::string> values = {"c_grid"};
        return values;
    }

    static const std::vector<std::string>& bottom_boundary_modes()
    {
        static const std::vector<std::string> values = {
            "rigid_noslip",
            "rigid_slip",
        };
        return values;
    }

    static const std::vector<std::string>& top_boundary_modes()
    {
        static const std::vector<std::string> values = {
            "rigid",
            "rigid_with_sponge",
            "open",
        };
        return values;
    }

    static const std::string& legacy_open_cgrid_resolution()
    {
        static const std::string value = "radiative";
        return value;
    }

    static const std::string& legacy_open_collocated_resolution()
    {
        static const std::string value = "zero_gradient";
        return value;
    }

    /** GPU apply switches select optional state-commit dispatch candidates.
     *
     * They are inert under a CPU backend and do not imply that every process
     * in the selected scheme is device-owned.  These lists describe the
     * schemes/dynamics layouts for which the corresponding candidate exists.
     */
    static const std::vector<std::string>& gpu_apply_microphysics_schemes()
    {
        static const std::vector<std::string> values = {
            "kessler",
            "lin",
            "thompson",
            "milbrandt",
        };
        return values;
    }

    static const std::vector<std::string>& gpu_apply_sgs_schemes()
    {
        static const std::vector<std::string> values = {"smagorinsky", "tke"};
        return values;
    }

    static const std::vector<std::string>& gpu_apply_slow_dynamics_schemes()
    {
        static const std::vector<std::string> values = {
            "klemp_wilhelmson_split_cartesian_agrid",
            "klemp_wilhelmson_split_cartesian_cgrid",
        };
        return values;
    }

    static const std::vector<std::string>&
    qualified_gpu_apply_slow_dynamics_schemes()
    {
        static const std::vector<std::string> values = {
            "klemp_wilhelmson_split_cartesian_agrid",
            "klemp_wilhelmson_split_cartesian_cgrid",
        };
        return values;
    }

    /** Strict-v1 guard modes in which each requested state-commit dispatch
     * actually owns the commit.  Warn/error retain a non-mutating host commit
     * for kernels whose shader ABI always publishes a bounded candidate.
     */
    static const std::vector<std::string>& gpu_apply_mutating_dispatch_guard_modes()
    {
        static const std::vector<std::string> values = {"off"};
        return values;
    }

    static const std::vector<std::string>& gpu_apply_nonmutating_dispatch_guard_modes()
    {
        static const std::vector<std::string> values = {
            "off",
            "warn",
            "error",
        };
        return values;
    }

    static const std::vector<std::string>& gpu_apply_host_commit_guard_modes()
    {
        static const std::vector<std::string> values = {"warn", "error"};
        return values;
    }
};

inline bool policy_contains(const std::vector<std::string>& values, const std::string& requested)
{
    return std::find(values.begin(), values.end(), requested) != values.end();
}

inline bool is_zfp_output_format(const std::string& output_format)
{
    return output_format == "zfp" || output_format == "zfp_low" || output_format == "zfp_medium" ||
           output_format == "zfp_high";
}

/**
 * Validate scheme construction, maturity, geometry, staggering, and backend
 * from the same descriptors exported to CLI/Studio catalog consumers.
 */
inline std::vector<std::string> validate_science_profile(const std::vector<SelectedScheme>& selections,
                                                         bool                               require_qualified,
                                                         const std::string&                 coordinate_system,
                                                         const std::string&                 staggering,
                                                         const std::string&                 backend,
                                                         bool qualified_collocated_mapped_active = false)
{
    std::vector<std::string> errors;
    for (const auto& selection : selections)
    {
        if (selection.descriptors == nullptr)
        {
            errors.push_back(selection.component + " has no scheme descriptor registry");
            continue;
        }
        const SchemeDescriptor* descriptor = find_scheme_descriptor(*selection.descriptors, selection.id);
        if (descriptor == nullptr)
        {
            errors.push_back(selection.component + " selects unknown scheme '" + selection.id + "'");
            continue;
        }
        if (!descriptor->constructible)
        {
            errors.push_back(selection.component + "=" + descriptor->id + " is unavailable in this binary");
            continue;
        }
        if (!descriptor->runtime_selectable)
        {
            errors.push_back(selection.component + "=" + descriptor->id +
                             " is development-only and unavailable for simulation runs");
            continue;
        }
        if (require_qualified && descriptor->maturity != SchemeMaturity::Qualified)
        {
            errors.push_back(selection.component + "=" + descriptor->id +
                             " has maturity=" + scheme_maturity_name(descriptor->maturity) +
                             "; qualified execution requires maturity=qualified");
        }

        const auto& capabilities = descriptor->capabilities;
        const auto  supports     = [](const std::vector<std::string>& values, const std::string& requested)
        {
            return std::find(values.begin(), values.end(), requested) != values.end() ||
                   std::find(values.begin(), values.end(), "any") != values.end() ||
                   std::find(values.begin(), values.end(), "not_applicable") != values.end();
        };
        if (!supports(capabilities.coordinate_systems, coordinate_system))
        {
            errors.push_back(selection.component + "=" + descriptor->id +
                             " does not support coordinate_system=" + coordinate_system);
        }
        if (!supports(capabilities.staggerings, staggering))
        {
            errors.push_back(selection.component + "=" + descriptor->id + " does not support staggering=" + staggering);
        }

        // A true no-op has no backend work. Active schemes must explicitly
        // advertise the selected backend; implicit CPU fallback is not a
        // qualified capability.
        const bool active = !capabilities.required_state_fields.empty() || !capabilities.produced_state_fields.empty();
        const bool host_preflight_input = selection.component == "environment.sounding.file.scheme_id";
        if (active && !host_preflight_input && !supports(capabilities.backends, backend))
        {
            errors.push_back(selection.component + "=" + descriptor->id + " does not support backend=" + backend);
        }
        if (require_qualified && backend == "gpu" &&
            capabilities.gpu_implementation_coverage == SchemeCoverage::Partial)
        {
            errors.push_back(selection.component + "=" + descriptor->id +
                             " has only partial accelerator implementation; "
                             "qualified execution requires covered accelerator ownership");
        }
    }
    if (coordinate_system == "cartesian" && staggering == "collocated" &&
        !qualified_collocated_mapped_active)
    {
        const auto selected_id = [&](const char* component)
        {
            const auto found = std::find_if(
                selections.begin(), selections.end(),
                [&](const SelectedScheme& value)
                { return value.component == component; });
            return found == selections.end() ? std::string{} : found->id;
        };
        const std::string terrain_id = selected_id("terrain.scheme");
        if (!terrain_id.empty() && terrain_id != "none")
        {
            for (const char* component : {
                     "microphysics.scheme", "radiation.scheme",
                     "boundary_layer.scheme", "turbulence.scheme",
                     "chaos.scheme"})
            {
                const std::string id = selected_id(component);
                if (!id.empty() && id != "none")
                {
                    errors.push_back(
                        std::string(component) + "=" + id +
                        " has no mapped Cartesian A-grid active-physics owner");
                }
            }
        }
    }
    return errors;
}

/** Validate cross-component ownership that individual descriptors cannot
 * express. Active PBL closures already own vertical momentum/heat/moisture
 * mixing, so a second vertically active SGS closure would double count the
 * same unresolved transport. Horizontal-only SGS remains a valid partner. */
inline std::vector<std::string> validate_physics_interactions(const std::string& boundary_layer_id,
                                                              const std::string& turbulence_id,
                                                              const std::string& turbulence_mode)
{
    std::vector<std::string> errors;
    const bool active_boundary_layer = boundary_layer_id != ScienceInteractionPolicy::inactive_scheme_id();
    const bool active_turbulence     = turbulence_id != ScienceInteractionPolicy::inactive_scheme_id();
    const bool horizontal_only =
        policy_contains(ScienceInteractionPolicy::horizontal_only_turbulence_modes(), turbulence_mode);
    if (active_boundary_layer && active_turbulence && !horizontal_only)
    {
        errors.push_back("boundary_layer=" + boundary_layer_id + " and turbulence=" + turbulence_id +
                         " mode=" + turbulence_mode +
                         " both own vertical momentum/heat/moisture mixing; select "
                         "turbulence.mode=2d/horizontal_only or disable one closure");
    }
    return errors;
}

/** Validate the surface/PBL ownership represented as separate strict-v1
 * selections. The surface model has no independent runtime operator: its
 * fluxes are consumed by an explicitly selected boundary-layer closure. */
inline std::vector<std::string> validate_surface_interactions(const std::string& surface_id,
                                                              bool               apply_surface_fluxes,
                                                              const std::string& boundary_layer_id)
{
    std::vector<std::string> errors;
    if (!policy_contains(ScienceInteractionPolicy::known_surface_schemes(), surface_id))
    {
        errors.push_back("surface.scheme selects unknown scheme '" + surface_id + "'");
        return errors;
    }
    if (surface_id == ScienceInteractionPolicy::inactive_scheme_id() && apply_surface_fluxes)
    {
        errors.push_back("surface.apply_fluxes cannot be true when surface.scheme=none");
    }
    if (surface_id == ScienceInteractionPolicy::active_surface_scheme_id() &&
        boundary_layer_id == ScienceInteractionPolicy::inactive_scheme_id())
    {
        errors.push_back("boundary_layer.scheme cannot be none when surface.scheme=slab; "
                         "select boundary_layer.scheme=slab explicitly");
    }
    return errors;
}

/** Validate the scoped terrain-following operator family. Individual scheme
 * descriptors express coordinate/staggering/backend support, but cannot state
 * cross-component requirements such as mapped PBL/diffusion ownership. Keep
 * these rules at pre-allocation
 * validation so a combination accepted by validate-only cannot fail later
 * merely because a flat operator reached active terrain. */
inline std::vector<std::string> validate_terrain_interactions(const std::string& terrain_id,
                                                              const std::string& radiation_id,
                                                              bool               radiation_shortwave_enabled,
                                                              const std::string& boundary_layer_id,
                                                              const std::string& turbulence_id,
                                                              const std::string& chaos_id,
                                                              const std::string& diffusion_id,
                                                              const std::string& diffusion_target,
                                                              bool               diffusion_active)
{
    std::vector<std::string> errors;
    if (terrain_id == ScienceInteractionPolicy::inactive_scheme_id())
        return errors;

    if (!policy_contains(ScienceInteractionPolicy::terrain_radiation_schemes(), radiation_id))
    {
        errors.push_back("terrain=" + terrain_id +
                         " supports radiation=none, mapped simple_grey, or mapped RRTMG longwave; other radiation "
                         "schemes lack a terrain-following transport contract");
    }
    (void)radiation_shortwave_enabled;
    if (!policy_contains(ScienceInteractionPolicy::terrain_boundary_layer_schemes(), boundary_layer_id))
    {
        errors.push_back("terrain=" + terrain_id + " does not support boundary_layer=" + boundary_layer_id +
                         "; the selected closure has no mapped surface-traction contract");
    }
    if (!policy_contains(ScienceInteractionPolicy::terrain_turbulence_schemes(), turbulence_id))
    {
        errors.push_back("terrain=" + terrain_id +
                         " supports turbulence=none, "
                         "smagorinsky, or tke; the selected closure has no mapped stress equation");
    }
    if (!policy_contains(ScienceInteractionPolicy::terrain_chaos_schemes(), chaos_id))
    {
        errors.push_back("terrain=" + terrain_id + " does not support chaos=" + chaos_id +
                         "; the selected stochastic scheme has no mapped physical-grid contract");
    }
    const bool terrain_diffusion_supported =
        diffusion_id == "explicit"
            ? policy_contains(ScienceInteractionPolicy::terrain_active_diffusion_targets(), diffusion_target)
            : diffusion_id == "implicit" &&
                policy_contains(
                    ScienceInteractionPolicy::terrain_active_diffusion_targets(),
                    diffusion_target);
    if (diffusion_active && !terrain_diffusion_supported)
    {
        errors.push_back("terrain=" + terrain_id +
                         " supports explicit mapped scalar and momentum diffusion; "
                         "implicit mapped diffusion supports the same scalar and momentum targets");
    }
    return errors;
}

/** Keep qualified compression claims inside the geometry profiles exercised
 * by the ZFP history qualification. */
inline std::vector<std::string>
validate_output_interactions(const std::string& terrain_id, const std::string& output_format, bool require_qualified)
{
    std::vector<std::string> errors;
    if (require_qualified && is_zfp_output_format(output_format) &&
        !policy_contains(ScienceInteractionPolicy::qualified_zfp_terrain_schemes(), terrain_id))
    {
        errors.push_back("qualified output.format=" + output_format +
                         " does not support terrain=" + terrain_id +
                         "; the selected geometry has not passed the qualified "
                         "ZFP error-budget matrix");
    }
    return errors;
}

/** Validate boundary topology and staggering-dependent closure support.
 *
 * `open` is retained only as a legacy compatibility token. Its resolved
 * equation is published by ScienceInteractionPolicy so Studio cannot label it
 * as one fixed closure while the executable selects another by staggering.
 */
inline std::vector<std::string> validate_boundary_interactions(const std::string& staggering,
                                                               const std::string& west,
                                                               const std::string& east,
                                                               const std::string& south,
                                                               const std::string& north,
                                                               const std::string& bottom,
                                                               const std::string& top)
{
    std::vector<std::string> errors;
    const auto               validate_lateral = [&errors, &staggering](const char* face, const std::string& boundary)
    {
        if (!policy_contains(ScienceInteractionPolicy::lateral_boundary_modes(), boundary))
        {
            errors.push_back(std::string("boundary_conditions.lateral.") + face + " selects unknown closure '" +
                             boundary + "'");
            return;
        }
        const bool radiative =
            boundary == "radiative" || (boundary == "open" && staggering == "c_grid" &&
                                        ScienceInteractionPolicy::legacy_open_cgrid_resolution() == "radiative");
        if (radiative && !policy_contains(ScienceInteractionPolicy::radiative_lateral_staggerings(), staggering))
        {
            errors.push_back(std::string("boundary_conditions.lateral.") + face +
                             "=radiative requires staggering=c_grid; use zero_gradient "
                             "for a collocated grid");
        }
    };
    validate_lateral("west", west);
    validate_lateral("east", east);
    validate_lateral("south", south);
    validate_lateral("north", north);

    const auto require_periodic_pair = [&errors](const char* axis, const std::string& first, const std::string& second)
    {
        if ((first == "periodic") != (second == "periodic"))
        {
            errors.push_back(std::string("periodic ") + axis +
                             " lateral boundaries must be configured as an opposing pair");
        }
    };
    require_periodic_pair("west/east", west, east);
    require_periodic_pair("south/north", south, north);

    if (!policy_contains(ScienceInteractionPolicy::bottom_boundary_modes(), bottom))
    {
        errors.push_back("boundary_conditions.bottom selects unknown closure '" + bottom + "'");
    }
    if (!policy_contains(ScienceInteractionPolicy::top_boundary_modes(), top))
    {
        errors.push_back("boundary_conditions.top selects unknown closure '" + top + "'");
    }
    return errors;
}

inline std::vector<std::string> validate_diagnostic_interactions(int scientific_every_steps)
{
    if (scientific_every_steps < -1)
    {
        return {
            "extensions.diagnostics.scientific_every_steps must be omitted "
            "(compatibility cadence) or >= 0",
        };
    }
    return {};
}

/** Complete value-dependent interaction request.
 *
 * Descriptor validation covers each selected module in isolation. This object
 * carries the option values that determine shared operator ownership and the
 * scoped terrain equation. All C++ preflight surfaces must call the aggregate
 * validator below rather than assembling the individual interaction checks.
 */
struct ScienceInteractionSelection
{
    std::string boundary_layer_id;
    std::string turbulence_id;
    std::string turbulence_mode;
    std::string surface_id;
    bool        apply_surface_fluxes = false;
    std::string terrain_id;
    std::string radiation_id;
    bool        radiation_shortwave_enabled = false;
    std::string chaos_id;
    std::string diffusion_id;
    std::string diffusion_target;
    bool        diffusion_active = false;
    std::string output_format;
    bool        require_qualified                  = false;
    std::string staggering                         = "c_grid";
    std::string lateral_west                       = "radiative";
    std::string lateral_east                       = "radiative";
    std::string lateral_south                      = "radiative";
    std::string lateral_north                      = "radiative";
    std::string bottom_boundary                    = "rigid_noslip";
    std::string top_boundary                       = "rigid_with_sponge";
    int         scientific_diagnostics_every_steps = -1;
    std::string backend                            = "cpu";
    std::string dynamics_id                        = "klemp_wilhelmson_split_cartesian_cgrid";
    std::string microphysics_id                    = "none";
    bool        gpu_apply_microphysics             = false;
    bool        gpu_apply_slow                     = false;
    bool        gpu_apply_sgs                      = false;
};

enum class ComputeCommitOwner
{
    Inactive,
    Host,
    Device,
    Unsupported,
};

inline const char* compute_commit_owner_name(ComputeCommitOwner owner) noexcept
{
    switch (owner)
    {
    case ComputeCommitOwner::Inactive:
        return "inactive";
    case ComputeCommitOwner::Host:
        return "host";
    case ComputeCommitOwner::Device:
        return "device";
    case ComputeCommitOwner::Unsupported:
        return "unsupported";
    }
    return "unsupported";
}

inline ComputeCommitOwner resolve_compute_process_owner(const SchemeDescriptor& descriptor,
                                                         const std::string&      backend)
{
    if (descriptor.id == ScienceInteractionPolicy::inactive_scheme_id())
        return ComputeCommitOwner::Inactive;
    if (backend == "cpu")
        return ComputeCommitOwner::Host;
    if (backend != "gpu")
        return ComputeCommitOwner::Unsupported;
    const auto& capabilities = descriptor.capabilities;
    if (!policy_contains(capabilities.backends, "gpu"))
        return ComputeCommitOwner::Unsupported;
    if (capabilities.gpu_implementation_coverage == SchemeCoverage::Covered)
        return ComputeCommitOwner::Device;
    if (capabilities.gpu_implementation_coverage == SchemeCoverage::None ||
        capabilities.gpu_implementation_coverage == SchemeCoverage::NotApplicable)
        return ComputeCommitOwner::Host;
    return ComputeCommitOwner::Unsupported;
}

struct ComputeOwnershipDecision
{
    ComputeCommitOwner        microphysics_state_commit = ComputeCommitOwner::Inactive;
    ComputeCommitOwner        slow_state_commit         = ComputeCommitOwner::Host;
    ComputeCommitOwner        sgs_state_commit          = ComputeCommitOwner::Inactive;
    std::vector<std::string> errors;
};

inline ComputeOwnershipDecision resolve_compute_ownership(const ScienceInteractionSelection& selection)
{
    const auto active = [](const std::string& id)
    {
        return !id.empty() && id != ScienceInteractionPolicy::inactive_scheme_id();
    };
    ComputeOwnershipDecision decision;
    decision.microphysics_state_commit =
        active(selection.microphysics_id) ? ComputeCommitOwner::Host : ComputeCommitOwner::Inactive;
    decision.sgs_state_commit =
        active(selection.turbulence_id) ? ComputeCommitOwner::Host : ComputeCommitOwner::Inactive;
    if (selection.backend == "cpu")
        return decision;
    if (selection.backend != "gpu")
    {
        decision.microphysics_state_commit = active(selection.microphysics_id)
            ? ComputeCommitOwner::Unsupported : ComputeCommitOwner::Inactive;
        decision.slow_state_commit = ComputeCommitOwner::Unsupported;
        decision.sgs_state_commit = active(selection.turbulence_id)
            ? ComputeCommitOwner::Unsupported : ComputeCommitOwner::Inactive;
        decision.errors.push_back("compute backend '" + selection.backend + "' is unknown; expected cpu or gpu");
        return decision;
    }

    if (selection.gpu_apply_microphysics && active(selection.microphysics_id) &&
        !policy_contains(ScienceInteractionPolicy::gpu_apply_microphysics_schemes(), selection.microphysics_id))
    {
        decision.microphysics_state_commit = ComputeCommitOwner::Unsupported;
        decision.errors.push_back("extensions.compute.gpu_apply_microphysics=true has no device "
                                  "state-commit candidate for microphysics.scheme=" +
                                  selection.microphysics_id);
    }
    else if (selection.gpu_apply_microphysics && active(selection.microphysics_id))
    {
        decision.microphysics_state_commit = ComputeCommitOwner::Device;
    }
    if (selection.gpu_apply_sgs && active(selection.turbulence_id) &&
        !policy_contains(ScienceInteractionPolicy::gpu_apply_sgs_schemes(), selection.turbulence_id))
    {
        decision.sgs_state_commit = ComputeCommitOwner::Unsupported;
        decision.errors.push_back("extensions.compute.gpu_apply_sgs=true has no device state-commit "
                                  "candidate for turbulence.scheme=" +
                                  selection.turbulence_id);
    }
    else if (selection.gpu_apply_sgs && active(selection.turbulence_id))
    {
        decision.sgs_state_commit = ComputeCommitOwner::Device;
    }
    if (selection.gpu_apply_slow &&
        !policy_contains(ScienceInteractionPolicy::gpu_apply_slow_dynamics_schemes(), selection.dynamics_id))
    {
        decision.slow_state_commit = ComputeCommitOwner::Unsupported;
        decision.errors.push_back("extensions.compute.gpu_apply_slow=true has no device state-commit "
                                  "candidate for dynamics.scheme=" +
                                  selection.dynamics_id);
    }
    else if (selection.gpu_apply_slow)
    {
        decision.slow_state_commit = ComputeCommitOwner::Device;
    }
    if (selection.require_qualified && selection.gpu_apply_slow &&
        !policy_contains(
            ScienceInteractionPolicy::
                qualified_gpu_apply_slow_dynamics_schemes(),
            selection.dynamics_id))
    {
        decision.slow_state_commit = ComputeCommitOwner::Unsupported;
        decision.errors.push_back("extensions.compute.gpu_apply_slow=true is not admitted in "
                                  "qualified execution; no qualified device slow-commit profile exists");
    }
    return decision;
}

inline std::vector<std::string> validate_compute_interactions(const ScienceInteractionSelection& selection)
{
    return resolve_compute_ownership(selection).errors;
}

inline std::vector<std::string> validate_science_interactions(const ScienceInteractionSelection& selection)
{
    std::vector<std::string> errors =
        validate_physics_interactions(selection.boundary_layer_id, selection.turbulence_id, selection.turbulence_mode);
    const auto surface_errors = validate_surface_interactions(
        selection.surface_id, selection.apply_surface_fluxes, selection.boundary_layer_id);
    errors.insert(errors.end(), surface_errors.begin(), surface_errors.end());
    const auto terrain_errors = validate_terrain_interactions(selection.terrain_id,
                                                              selection.radiation_id,
                                                              selection.radiation_shortwave_enabled,
                                                              selection.boundary_layer_id,
                                                              selection.turbulence_id,
                                                              selection.chaos_id,
                                                              selection.diffusion_id,
                                                              selection.diffusion_target,
                                                              selection.diffusion_active);
    errors.insert(errors.end(), terrain_errors.begin(), terrain_errors.end());
    const auto output_errors =
        validate_output_interactions(selection.terrain_id, selection.output_format, selection.require_qualified);
    errors.insert(errors.end(), output_errors.begin(), output_errors.end());
    const auto boundary_errors = validate_boundary_interactions(selection.staggering,
                                                                selection.lateral_west,
                                                                selection.lateral_east,
                                                                selection.lateral_south,
                                                                selection.lateral_north,
                                                                selection.bottom_boundary,
                                                                selection.top_boundary);
    errors.insert(errors.end(), boundary_errors.begin(), boundary_errors.end());
    const auto diagnostic_errors = validate_diagnostic_interactions(selection.scientific_diagnostics_every_steps);
    errors.insert(errors.end(), diagnostic_errors.begin(), diagnostic_errors.end());
    const auto compute_errors = validate_compute_interactions(selection);
    errors.insert(errors.end(), compute_errors.begin(), compute_errors.end());
    return errors;
}

} // namespace tmv::runtime
