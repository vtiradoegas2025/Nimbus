#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

/**
 * @file field_registry.hpp
 * @brief Export-field metadata shared by output configuration and runtime IO.
 */

enum class OutputFieldKind
{
    Core3D,
    DerivedVolume,
    SurfaceTranche,
    DirectSurface2D,
    ColumnTranche,
    RadarSynthetic,
};

enum class OutputFieldDependency : std::uint8_t
{
    None                       = 0,
    RadarDiagnostic            = 1U << 0U,
    VorticityDiagnostics       = 1U << 1U,
    AngularMomentumDiagnostics = 1U << 2U,
    PressureDiagnostics        = 1U << 3U,
    CenteredU                  = 1U << 4U,
    CenteredV                  = 1U << 5U,
    CenteredW                  = 1U << 6U,
    PressureComponents         = 1U << 7U,
};

constexpr OutputFieldDependency operator|(OutputFieldDependency lhs, OutputFieldDependency rhs) noexcept
{
    return static_cast<OutputFieldDependency>(static_cast<std::uint8_t>(lhs) | static_cast<std::uint8_t>(rhs));
}

constexpr OutputFieldDependency& operator|=(OutputFieldDependency& lhs, OutputFieldDependency rhs) noexcept
{
    lhs = lhs | rhs;
    return lhs;
}

constexpr bool output_field_has_dependency(OutputFieldDependency dependencies,
                                           OutputFieldDependency dependency) noexcept
{
    return (static_cast<std::uint8_t>(dependencies) & static_cast<std::uint8_t>(dependency)) != 0;
}

constexpr OutputFieldDependency centered_uv_dependencies =
    OutputFieldDependency::CenteredU | OutputFieldDependency::CenteredV;
constexpr OutputFieldDependency centered_uvw_dependencies = centered_uv_dependencies | OutputFieldDependency::CenteredW;

struct OutputFieldDescriptor
{
    const char*           field_id;
    const char*           suffix;
    OutputFieldKind       kind;
    OutputFieldDependency dependencies = OutputFieldDependency::None;
};

inline const char* output_field_category_name(OutputFieldKind kind)
{
    switch (kind)
    {
    case OutputFieldKind::Core3D:
        return "core";
    case OutputFieldKind::DerivedVolume:
        return "derived";
    case OutputFieldKind::SurfaceTranche:
        return "surface";
    case OutputFieldKind::DirectSurface2D:
        return "surface";
    case OutputFieldKind::ColumnTranche:
        return "column";
    case OutputFieldKind::RadarSynthetic:
        return "radar_synthetic";
    }
    return "unknown";
}

inline bool output_field_is_derived(const OutputFieldDescriptor& field)
{
    return field.kind != OutputFieldKind::Core3D;
}

inline bool output_field_is_derived_volume(const OutputFieldDescriptor& field)
{
    return field.kind == OutputFieldKind::DerivedVolume;
}

inline bool output_field_is_direct_surface_2d(const OutputFieldDescriptor& field)
{
    return field.kind == OutputFieldKind::DirectSurface2D;
}

inline bool output_field_is_tranche_product(const OutputFieldDescriptor& field)
{
    return field.kind == OutputFieldKind::SurfaceTranche || field.kind == OutputFieldKind::ColumnTranche ||
           field.kind == OutputFieldKind::RadarSynthetic;
}

/**
 * @brief Whether this field requires either lazy resident diagnostic group.
 */
inline bool output_field_requires_lazy_diagnostics(const OutputFieldDescriptor& field)
{
    return output_field_has_dependency(field.dependencies, OutputFieldDependency::RadarDiagnostic) ||
           output_field_has_dependency(field.dependencies, OutputFieldDependency::VorticityDiagnostics) ||
           output_field_has_dependency(field.dependencies, OutputFieldDependency::AngularMomentumDiagnostics) ||
           output_field_has_dependency(field.dependencies, OutputFieldDependency::PressureDiagnostics);
}

inline const std::vector<OutputFieldDescriptor>& output_field_registry()
{
    static const std::vector<OutputFieldDescriptor> fields = {
        {"u", "u", OutputFieldKind::Core3D, OutputFieldDependency::CenteredU},
        {"v", "v", OutputFieldKind::Core3D, OutputFieldDependency::CenteredV},
        {"w", "w", OutputFieldKind::Core3D, OutputFieldDependency::CenteredW},
        {"rho", "rho", OutputFieldKind::Core3D},
        {"p", "p", OutputFieldKind::Core3D},
        {"theta", "theta", OutputFieldKind::Core3D},
        {"qv", "qv", OutputFieldKind::Core3D},
        {"qc", "qc", OutputFieldKind::Core3D},
        {"qr", "qr", OutputFieldKind::Core3D},
        {"qi", "qi", OutputFieldKind::Core3D},
        {"qs", "qs", OutputFieldKind::Core3D},
        {"qh", "qh", OutputFieldKind::Core3D},
        {"qg", "qg", OutputFieldKind::Core3D},
        {"radar", "radar", OutputFieldKind::Core3D, OutputFieldDependency::RadarDiagnostic},
        {"tracer", "tracer", OutputFieldKind::Core3D},
        {"tke", "tke", OutputFieldKind::Core3D},
        {"vorticity_r", "vorticity_r", OutputFieldKind::Core3D, OutputFieldDependency::VorticityDiagnostics},
        {"vorticity_theta", "vorticity_theta", OutputFieldKind::Core3D, OutputFieldDependency::VorticityDiagnostics},
        {"vorticity_z", "vorticity_z", OutputFieldKind::Core3D, OutputFieldDependency::VorticityDiagnostics},
        {"stretching_term", "stretching_term", OutputFieldKind::Core3D, OutputFieldDependency::VorticityDiagnostics},
        {"tilting_term", "tilting_term", OutputFieldKind::Core3D, OutputFieldDependency::VorticityDiagnostics},
        {"baroclinic_term", "baroclinic_term", OutputFieldKind::Core3D, OutputFieldDependency::VorticityDiagnostics},
        {"p_prime", "p_prime", OutputFieldKind::Core3D, OutputFieldDependency::PressureDiagnostics},
        {"dynamic_pressure", "dynamic_pressure", OutputFieldKind::Core3D,
         OutputFieldDependency::PressureDiagnostics | OutputFieldDependency::PressureComponents},
        {"buoyancy_pressure", "buoyancy_pressure", OutputFieldKind::Core3D,
         OutputFieldDependency::PressureDiagnostics | OutputFieldDependency::PressureComponents},
        {"angular_momentum",
         "angular_momentum",
         OutputFieldKind::Core3D,
         OutputFieldDependency::AngularMomentumDiagnostics},
        {"angular_momentum_tendency",
         "angular_momentum_tendency",
         OutputFieldKind::Core3D,
         OutputFieldDependency::AngularMomentumDiagnostics},

        {"temperature", "temperature", OutputFieldKind::DerivedVolume},
        {"theta_prime", "theta_prime", OutputFieldKind::DerivedVolume},
        {"theta_v", "theta_v", OutputFieldKind::DerivedVolume},
        {"theta_e", "theta_e", OutputFieldKind::DerivedVolume},
        {"dewpoint", "dewpoint", OutputFieldKind::DerivedVolume},
        {"relative_humidity", "relative_humidity", OutputFieldKind::DerivedVolume},
        {"saturation_mixing_ratio", "saturation_mixing_ratio", OutputFieldKind::DerivedVolume},
        {"total_condensate", "total_condensate", OutputFieldKind::DerivedVolume},
        {"cloud_condensate", "cloud_condensate", OutputFieldKind::DerivedVolume},
        {"cloud_liquid_condensate", "cloud_liquid_condensate", OutputFieldKind::DerivedVolume},
        {"cloud_frozen_condensate", "cloud_frozen_condensate", OutputFieldKind::DerivedVolume},
        {"reflectivity_dbz",
         "reflectivity_dbz",
         OutputFieldKind::DerivedVolume,
         OutputFieldDependency::RadarDiagnostic},
        {"radial_velocity",
         "radial_velocity",
         OutputFieldKind::DerivedVolume,
         OutputFieldDependency::RadarDiagnostic},
        {"vorticity_magnitude",
         "vorticity_magnitude",
         OutputFieldKind::DerivedVolume,
         OutputFieldDependency::VorticityDiagnostics},
        {"divergence", "divergence", OutputFieldKind::DerivedVolume, centered_uvw_dependencies},
        {"potential_vorticity",
         "potential_vorticity",
         OutputFieldKind::DerivedVolume,
         OutputFieldDependency::VorticityDiagnostics},
        {"buoyancy", "buoyancy", OutputFieldKind::DerivedVolume},
        {"horizontal_vorticity_streamwise",
         "horizontal_vorticity_streamwise",
         OutputFieldKind::DerivedVolume,
         OutputFieldDependency::VorticityDiagnostics | centered_uv_dependencies},
        {"horizontal_vorticity_crosswise",
         "horizontal_vorticity_crosswise",
         OutputFieldKind::DerivedVolume,
         OutputFieldDependency::VorticityDiagnostics | centered_uv_dependencies},
        {"pressure_gradient_force_x", "pressure_gradient_force_x", OutputFieldKind::DerivedVolume},
        {"pressure_gradient_force_y", "pressure_gradient_force_y", OutputFieldKind::DerivedVolume},
        {"pressure_gradient_force_z", "pressure_gradient_force_z", OutputFieldKind::DerivedVolume},
        {"storm_relative_winds", "storm_relative_winds", OutputFieldKind::DerivedVolume, centered_uv_dependencies},
        {"helicity_density",
         "helicity_density",
         OutputFieldKind::DerivedVolume,
         OutputFieldDependency::VorticityDiagnostics | centered_uvw_dependencies},
        {"okubo_weiss",
         "okubo_weiss",
         OutputFieldKind::DerivedVolume,
         OutputFieldDependency::VorticityDiagnostics | centered_uv_dependencies},
        {"theta_w", "theta_w", OutputFieldKind::DerivedVolume},
        {"zdr", "zdr", OutputFieldKind::DerivedVolume, OutputFieldDependency::RadarDiagnostic},
        {"kdp", "kdp", OutputFieldKind::DerivedVolume, OutputFieldDependency::RadarDiagnostic},
        {"rhohv", "rhohv", OutputFieldKind::DerivedVolume, OutputFieldDependency::RadarDiagnostic},
        {"streamlines", "streamlines", OutputFieldKind::DerivedVolume, centered_uvw_dependencies},
        {"trajectory_paths", "trajectory_paths", OutputFieldKind::DerivedVolume, centered_uvw_dependencies},
        {"q_vectors", "q_vectors", OutputFieldKind::DerivedVolume, centered_uv_dependencies},
        {"turbulent_diffusion_term", "turbulent_diffusion_term", OutputFieldKind::DerivedVolume},
        {"cross_section", "cross_section", OutputFieldKind::DerivedVolume},
        {"rhi_slice", "rhi_slice", OutputFieldKind::DerivedVolume, OutputFieldDependency::RadarDiagnostic},
        {"hodograph_aligned_cross_section",
         "hodograph_aligned_cross_section",
         OutputFieldKind::DerivedVolume,
         centered_uv_dependencies},
        {"forward_trajectories", "forward_trajectories", OutputFieldKind::DerivedVolume, centered_uvw_dependencies},
        {"backward_trajectories", "backward_trajectories", OutputFieldKind::DerivedVolume, centered_uvw_dependencies},
        {"parcel_buoyancy_trajectory", "parcel_buoyancy_trajectory", OutputFieldKind::DerivedVolume},
        {"vorticity_trajectory",
         "vorticity_trajectory",
         OutputFieldKind::DerivedVolume,
         OutputFieldDependency::VorticityDiagnostics},
        {"circulation_material_surface",
         "circulation_material_surface",
         OutputFieldKind::DerivedVolume,
         OutputFieldDependency::AngularMomentumDiagnostics},

        {"u10", "u10", OutputFieldKind::SurfaceTranche},
        {"v10", "v10", OutputFieldKind::SurfaceTranche},
        {"t2", "t2", OutputFieldKind::SurfaceTranche},
        {"td2", "td2", OutputFieldKind::SurfaceTranche},
        {"surface_pressure_perturbation", "surface_pressure_perturbation", OutputFieldKind::SurfaceTranche},
        {"surface_sensible_heat_flux", "surface_sensible_heat_flux", OutputFieldKind::DirectSurface2D},
        {"surface_latent_heat_flux", "surface_latent_heat_flux", OutputFieldKind::DirectSurface2D},
        {"surface_moisture_flux", "surface_moisture_flux", OutputFieldKind::DirectSurface2D},
        {"skin_temperature", "skin_temperature", OutputFieldKind::SurfaceTranche},
        {"cold_pool_boundary", "cold_pool_boundary", OutputFieldKind::SurfaceTranche},
        {"precip_rate", "precip_rate", OutputFieldKind::SurfaceTranche},
        {"accumulated_rainfall", "accumulated_rainfall", OutputFieldKind::SurfaceTranche},
        {"surface_friction_velocity", "surface_friction_velocity", OutputFieldKind::DirectSurface2D},
        {"surface_stress_u", "surface_stress_u", OutputFieldKind::DirectSurface2D},
        {"surface_stress_v", "surface_stress_v", OutputFieldKind::DirectSurface2D},
        {"pbl_height", "pbl_height", OutputFieldKind::DirectSurface2D},

        {"composite_reflectivity",
         "composite_reflectivity",
         OutputFieldKind::ColumnTranche,
         OutputFieldDependency::RadarDiagnostic},
        {"column_max_w", "column_max_w", OutputFieldKind::ColumnTranche},
        {"column_max_vorticity",
         "column_max_vorticity",
         OutputFieldKind::ColumnTranche,
         OutputFieldDependency::VorticityDiagnostics},
        {"vil", "vil", OutputFieldKind::ColumnTranche},
        {"precipitable_water", "precipitable_water", OutputFieldKind::ColumnTranche},
        {"liquid_water_path", "liquid_water_path", OutputFieldKind::ColumnTranche},
        {"cloud_water_path", "cloud_water_path", OutputFieldKind::ColumnTranche},
        {"cloud_top_height", "cloud_top_height", OutputFieldKind::ColumnTranche},
        {"cloud_base_height", "cloud_base_height", OutputFieldKind::ColumnTranche},
        {"lcl", "lcl", OutputFieldKind::ColumnTranche},
        {"lfc", "lfc", OutputFieldKind::ColumnTranche},
        {"el", "el", OutputFieldKind::ColumnTranche},
        {"cape", "cape", OutputFieldKind::ColumnTranche},
        {"cin", "cin", OutputFieldKind::ColumnTranche},
        {"lifted_index", "lifted_index", OutputFieldKind::ColumnTranche},
        {"k_index", "k_index", OutputFieldKind::ColumnTranche},
        {"showalter_index", "showalter_index", OutputFieldKind::ColumnTranche},
        {"total_totals", "total_totals", OutputFieldKind::ColumnTranche},
        {"srh_0_1km", "srh_0_1km", OutputFieldKind::ColumnTranche, centered_uv_dependencies},
        {"srh_0_3km", "srh_0_3km", OutputFieldKind::ColumnTranche, centered_uv_dependencies},
        {"ehi", "ehi", OutputFieldKind::ColumnTranche, centered_uv_dependencies},
        {"scp", "scp", OutputFieldKind::ColumnTranche, centered_uv_dependencies},
        {"stp", "stp", OutputFieldKind::ColumnTranche, centered_uv_dependencies},
        {"updraft_helicity",
         "updraft_helicity",
         OutputFieldKind::ColumnTranche,
         OutputFieldDependency::VorticityDiagnostics | centered_uvw_dependencies},

        {"ppi_sweep", "ppi_sweep", OutputFieldKind::RadarSynthetic, OutputFieldDependency::RadarDiagnostic},
        {"rhi_sweep", "rhi_sweep", OutputFieldKind::RadarSynthetic, OutputFieldDependency::RadarDiagnostic},
        {"bwer", "bwer", OutputFieldKind::RadarSynthetic, OutputFieldDependency::RadarDiagnostic},
        {"mesocyclone_diagnostic",
         "mesocyclone_diagnostic",
         OutputFieldKind::RadarSynthetic,
         OutputFieldDependency::VorticityDiagnostics},
        {"vrot", "vrot", OutputFieldKind::RadarSynthetic},
    };
    return fields;
}

inline const std::vector<std::string>& output_field_names()
{
    static const std::vector<std::string> names = []
    {
        std::vector<std::string> out;
        const auto&              fields = output_field_registry();
        out.reserve(fields.size());
        for (const auto& field : fields)
        {
            out.emplace_back(field.field_id);
        }
        return out;
    }();
    return names;
}

inline const OutputFieldDescriptor* find_output_field(std::string_view field_id)
{
    for (const auto& field : output_field_registry())
    {
        if (field.field_id == field_id)
        {
            return &field;
        }
    }
    return nullptr;
}

template <typename FieldNames>
OutputFieldDependency output_dependencies_for_fields(const FieldNames& field_names)
{
    OutputFieldDependency dependencies = OutputFieldDependency::None;
    for (const auto& name : field_names)
    {
        if (const OutputFieldDescriptor* field = find_output_field(name))
        {
            dependencies |= field->dependencies;
        }
    }
    return dependencies;
}
