#include "diagnostics/parcel_thermodynamics.hpp"
#include "diagnostics/column_integrals.hpp"
#include "diagnostics/physical_reference.hpp"
/**
 * @file test_output_config.cpp
 * @brief Unit tests for OutputConfig parsing, field presets, and disk budget.
 */
#include "catch2/catch.hpp"
#include "core/output/core_volume_writer.hpp"
#include "core/output/derived_slice_scratch.hpp"
#include "core/output/derived_volume_writer.hpp"
#include "core/output/direct_surface_writer.hpp"
#include "core/output/field_registry.hpp"
#include "core/output/output_config.hpp"
#include "core/output/step_manifest.hpp"
#include "core/output/tranche_writer.hpp"
#include "diagnostics/derived_field_math.hpp"
#include "diagnostics/derived_slice_compute.hpp"
#include "diagnostics/field_contract.hpp"

#include <cmath>
#include <limits>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

TEST_CASE("OutputConfig default values", "[core][output_config]")
{
    OutputConfig cfg;
    REQUIRE(cfg.format == OutputFormat::npy_2d_slices);
    REQUIRE(cfg.preset == FieldPreset::all);
    REQUIRE(cfg.async_io == false);
    REQUIRE(cfg.live_shm_interval_s == Approx(1.0));
    REQUIRE_FALSE(cfg.live_shm_full_rate);
    REQUIRE_FALSE(cfg.trajectories);
    REQUIRE(cfg.trajectory_seed_time_s == Approx(0.0));
    REQUIRE(cfg.trajectory_count == 5);
    REQUIRE(cfg.trajectory_sample_interval_s == Approx(0.0));
    REQUIRE(cfg.trajectory_max_samples_per_parcel == 16384);
    REQUIRE(cfg.zfp_tolerance == Approx(1.0e-4));
    REQUIRE(cfg.zfp_keyframe_interval == 0);
    REQUIRE(cfg.zfp_field_execution == "serial");
    REQUIRE(cfg.zfp_field_workers == 0);
    REQUIRE(cfg.zfp_field_scratch_mib == 64);
}

TEST_CASE("parse_output_config reads bounded live shared-memory cadence",
          "[core][output_config][shm]")
{
    const std::unordered_map<std::string, std::string> flat = {
        {"output.live_shm_interval_s", "2.5"},
        {"output.live_shm_full_rate", "true"}};
    const OutputConfig config = parse_output_config(flat);
    REQUIRE(config.live_shm_interval_s == Approx(2.5));
    REQUIRE(config.live_shm_full_rate);

    auto invalid = flat;
    invalid["output.live_shm_interval_s"] = "0";
    REQUIRE_THROWS_AS(parse_output_config(invalid), std::invalid_argument);

    std::string error;
    REQUIRE(validate_live_shm_fields(
        {"w", "reflectivity_dbz", "vorticity_z"}, error));
    REQUIRE_FALSE(validate_live_shm_fields({"w", "w"}, error));
    REQUIRE_FALSE(validate_live_shm_fields({"not_a_field"}, error));
    REQUIRE_FALSE(validate_live_shm_fields({}, error));
}

TEST_CASE("ZFP field execution policy is explicit and bounded",
          "[core][output_config][zfp][parallel]")
{
    std::unordered_map<std::string, std::string> values = {
        {"output.format", "zfp_high"},
        {"output.zfp_field_execution", "field_parallel"},
        {"output.zfp_field_workers", "4"},
        {"output.zfp_field_scratch_mib", "128"}};
    const OutputConfig parsed = parse_output_config(values);
    REQUIRE(parsed.zfp_field_execution == "field_parallel");
    REQUIRE(parsed.zfp_field_workers == 4);
    REQUIRE(parsed.zfp_field_scratch_mib == 128);

    values["output.zfp_field_execution"] = "shared_stream";
    REQUIRE_THROWS_AS(parse_output_config(values), std::invalid_argument);
    values["output.zfp_field_execution"] = "field_parallel";
    values["output.zfp_field_workers"] = "-1";
    REQUIRE_THROWS_AS(parse_output_config(values), std::invalid_argument);
    values["output.zfp_field_workers"] = "1";
    values["output.zfp_field_scratch_mib"] = "0";
    REQUIRE_THROWS_AS(parse_output_config(values), std::invalid_argument);

    CHECK(resolve_zfp_field_workers(0, 64, 8, false, 16, 16) == 1);
    CHECK(resolve_zfp_field_workers(0, 64, 8, false, 4, 16) == 4);
    CHECK(resolve_zfp_field_workers(0, 64, 8, true, 16, 16) == 8);
    CHECK(resolve_zfp_field_workers(4, 8, 8, false, 16, 16) == 4);
    CHECK(resolve_zfp_field_workers(5, 8, 8, true, 4, 16) == 0);
}

TEST_CASE("dedicated volume formats use snapshot writer when synchronous", "[core][output_config][dispatch]")
{
    CHECK_FALSE(requires_snapshot_assembly(OutputFormat::npy_3d, false));
    CHECK_FALSE(requires_snapshot_assembly(OutputFormat::csv, false));
    CHECK(requires_snapshot_assembly(OutputFormat::netcdf, false));
    CHECK(requires_snapshot_assembly(OutputFormat::zfp, false));
    CHECK(requires_snapshot_assembly(OutputFormat::npy_3d, true));
}

TEST_CASE("resolve_output_fields with minimal preset", "[core][output_config]")
{
    OutputConfig cfg;
    cfg.preset = FieldPreset::minimal;
    resolve_output_fields(cfg);

    REQUIRE_FALSE(cfg.resolved_fields.empty());
    // Minimal should have core fields like u, w, theta
    REQUIRE(cfg.resolved_fields.count("u") == 1);
    REQUIRE(cfg.resolved_fields.count("w") == 1);
    REQUIRE(cfg.resolved_fields.count("theta") == 1);
}

TEST_CASE("radar_ready preset publishes only the bounded radar pair",
          "[core][output_config][radar]")
{
    OutputConfig cfg;
    cfg.preset = FieldPreset::radar_ready;
    resolve_output_fields(cfg);
    REQUIRE(cfg.resolved_fields.count("reflectivity_dbz") == 1);
    REQUIRE(cfg.resolved_fields.count("radial_velocity") == 1);
    REQUIRE(cfg.resolved_fields.count("zdr") == 0);
    REQUIRE(cfg.resolved_fields.count("kdp") == 0);
    REQUIRE(cfg.resolved_fields.count("rhohv") == 0);
}

TEST_CASE("resolve_output_fields with presentation preset", "[core][output_config]")
{
    OutputConfig cfg;
    cfg.preset = FieldPreset::presentation;
    resolve_output_fields(cfg);

    // Presentation should be larger than minimal
    OutputConfig minimal_cfg;
    minimal_cfg.preset = FieldPreset::minimal;
    resolve_output_fields(minimal_cfg);

    REQUIRE(cfg.resolved_fields.size() > minimal_cfg.resolved_fields.size());
    REQUIRE(cfg.resolved_fields.count("cloud_liquid_condensate") == 1);
    REQUIRE(cfg.resolved_fields.count("cloud_frozen_condensate") == 1);
}

TEST_CASE("science_validation is the minimal moving-reference comparison preset",
          "[core][output_config][science_validation]")
{
    OutputConfig cfg;
    cfg.preset = FieldPreset::science_validation;
    resolve_output_fields(cfg);

    const std::unordered_set<std::string> expected = {
        "u", "v", "w", "theta", "p", "rho", "qv", "qc", "qr",
        "cold_pool_boundary", "precip_rate", "accumulated_rainfall"};
    REQUIRE(cfg.resolved_fields == expected);
    REQUIRE(cfg.resolved_fields.count("theta_prime") == 0);

    const std::unordered_map<std::string, std::string> flat = {
        {"output.fields", "science_validation"}};
    const OutputConfig parsed = parse_output_config(flat);
    REQUIRE(parsed.preset == FieldPreset::science_validation);
    REQUIRE(std::string(field_preset_name(parsed.preset)) ==
            "science_validation");

    cfg.format = OutputFormat::npy_3d;
    estimate_disk_budget(cfg, 512, 512, 160, 0, 60);
    constexpr std::size_t expected_float_payload =
        sizeof(float) * std::size_t{512} * std::size_t{512} *
        (9 * std::size_t{160} + 3);
    REQUIRE(expected_float_payload == std::size_t{1443} * 1024 * 1024);
    REQUIRE(cfg.estimated_bytes_per_export == expected_float_payload);
}

TEST_CASE("moving_reference_v1 diagnostics are exactly two lossless planar fields",
          "[core][output_config][qualification]")
{
    const std::unordered_map<std::string, std::string> flat = {
        {"output.qualification_diagnostics.preset", "moving_reference_v1"},
        {"output.qualification_diagnostics.interval_s", "30"},
    };
    OutputConfig cfg = parse_output_config(flat);
    REQUIRE(cfg.qualification_diagnostics_preset ==
            QualificationDiagnosticsPreset::moving_reference_v1);
    REQUIRE(cfg.qualification_diagnostics_interval_s == 30);
    REQUIRE(std::string(qualification_diagnostics_preset_name(
                cfg.qualification_diagnostics_preset)) ==
            "moving_reference_v1");
    const std::vector<std::string> expected_fields = {
        "cold_pool_boundary", "accumulated_rainfall"};
    REQUIRE(qualification_diagnostics_fields(
                cfg.qualification_diagnostics_preset) == expected_fields);

    cfg.preset = FieldPreset::science_validation;
    cfg.format = OutputFormat::npy_3d;
    resolve_output_fields(cfg);
    estimate_disk_budget(cfg, 512, 512, 160, 4500, 450);
    const std::size_t planar_payload =
        2 * sizeof(float) * std::size_t{512} * std::size_t{512};
    REQUIRE(cfg.qualification_diagnostics_estimated_bytes_per_export ==
            planar_payload);
    REQUIRE(cfg.qualification_diagnostics_estimated_total_bytes ==
            std::size_t{150} * (planar_payload + 64 * 1024));
    REQUIRE(cfg.estimated_total_bytes >
            cfg.qualification_diagnostics_estimated_total_bytes);
}

TEST_CASE("qualification diagnostics disk estimate is independent of ordinary history",
          "[core][output_config][qualification][disk]")
{
    OutputConfig cfg;
    cfg.qualification_diagnostics_preset =
        QualificationDiagnosticsPreset::moving_reference_v1;
    cfg.qualification_diagnostics_interval_s = 30;
    estimate_disk_budget(cfg, 158, 168, 50, 60, 0);
    const std::size_t planar_payload =
        2 * sizeof(float) * std::size_t{158} * std::size_t{168};
    REQUIRE(cfg.estimated_bytes_per_export == 0);
    REQUIRE(cfg.qualification_diagnostics_estimated_bytes_per_export ==
            planar_payload);
    REQUIRE(cfg.qualification_diagnostics_estimated_total_bytes ==
            2 * (planar_payload + 64 * 1024));
    REQUIRE(cfg.estimated_total_bytes ==
            cfg.qualification_diagnostics_estimated_total_bytes);
}

TEST_CASE("qualification diagnostics disk estimate can be rebound to the child grid",
          "[core][output_config][qualification][disk]")
{
    OutputConfig cfg;
    cfg.qualification_diagnostics_preset =
        QualificationDiagnosticsPreset::moving_reference_v1;
    cfg.qualification_diagnostics_interval_s = 30;
    estimate_disk_budget(cfg, 512, 512, 160, 60, 0);
    const std::size_t root_total = cfg.estimated_total_bytes;

    estimate_qualification_diagnostics_disk_budget(cfg, 158, 168, 60);
    const std::size_t child_payload =
        2 * sizeof(float) * std::size_t{158} * std::size_t{168};
    const std::size_t child_total =
        2 * (child_payload + 64 * 1024);
    REQUIRE(cfg.qualification_diagnostics_estimated_bytes_per_export ==
            child_payload);
    REQUIRE(cfg.qualification_diagnostics_estimated_total_bytes == child_total);
    REQUIRE(cfg.estimated_total_bytes == child_total);
    REQUIRE(cfg.estimated_total_bytes < root_total);
}

TEST_CASE("studio_full is the bounded Studio catalog preset", "[core][output_config][studio]")
{
    OutputConfig presentation;
    presentation.preset = FieldPreset::presentation;
    resolve_output_fields(presentation);

    OutputConfig studio;
    studio.preset = FieldPreset::studio_full;
    resolve_output_fields(studio);

    OutputConfig all;
    all.preset = FieldPreset::all;
    resolve_output_fields(all);

    for (const std::string& field : presentation.resolved_fields)
        REQUIRE(studio.resolved_fields.count(field) == 1);

    const std::vector<std::string> studio_additions = {"stretching_term",
                                                       "tilting_term",
                                                       "baroclinic_term",
                                                       "vorticity_r",
                                                       "vorticity_theta",
                                                       "qc",
                                                       "qg",
                                                       "qh",
                                                       "qs",
                                                       "qi",
                                                       "dynamic_pressure",
                                                       "buoyancy_pressure",
                                                       "pressure_gradient_force_z",
                                                       "divergence",
                                                       "column_max_vorticity",
                                                       "column_max_w",
                                                       "accumulated_rainfall",
                                                       "precip_rate",
                                                       "precipitable_water",
                                                       "liquid_water_path",
                                                       "cloud_water_path",
                                                       "potential_vorticity"};
    for (const std::string& field : studio_additions)
        REQUIRE(studio.resolved_fields.count(field) == 1);

    REQUIRE(studio.resolved_fields.size() == presentation.resolved_fields.size() + studio_additions.size());
    REQUIRE(studio.resolved_fields.size() < all.resolved_fields.size());
}

TEST_CASE("resolve_output_fields with all preset", "[core][output_config]")
{
    OutputConfig cfg;
    cfg.preset = FieldPreset::all;
    resolve_output_fields(cfg);

    const auto& all_names = get_all_field_names();
    const std::size_t radar_dependent = static_cast<std::size_t>(std::count_if(
        output_field_registry().begin(), output_field_registry().end(),
        [](const OutputFieldDescriptor& field)
        {
            return field.kind == OutputFieldKind::RadarSynthetic ||
                   output_field_has_dependency(
                       field.dependencies, OutputFieldDependency::RadarDiagnostic);
        }));
    REQUIRE(cfg.resolved_fields.size() + radar_dependent == all_names.size());
    REQUIRE(cfg.resolved_fields.count("reflectivity_dbz") == 0);
    REQUIRE(cfg.resolved_fields.count("radial_velocity") == 0);
    REQUIRE(cfg.resolved_fields.count("zdr") == 0);
    REQUIRE(cfg.resolved_fields.count("kdp") == 0);
    REQUIRE(cfg.resolved_fields.count("rhohv") == 0);
}

TEST_CASE("resolve_output_fields with custom fields", "[core][output_config]")
{
    OutputConfig cfg;
    cfg.preset        = FieldPreset::custom;
    cfg.custom_fields = {"w", "theta", "reflectivity_dbz"};
    resolve_output_fields(cfg);

    REQUIRE(cfg.resolved_fields.size() == 3);
    REQUIRE(cfg.resolved_fields.count("w") == 1);
    REQUIRE(cfg.resolved_fields.count("theta") == 1);
    REQUIRE(cfg.resolved_fields.count("reflectivity_dbz") == 1);
}

TEST_CASE("selected radar products author the publication inventory",
          "[core][output_config][radar][catalog]")
{
    OutputConfig cfg;
    cfg.preset = FieldPreset::minimal;
    resolve_output_fields(cfg);
    merge_selected_radar_output_fields(
        cfg, {"reflectivity_dbz", "zdr", "rhohv"});

    REQUIRE(cfg.resolved_fields.count("reflectivity_dbz") == 1);
    REQUIRE(cfg.resolved_fields.count("zdr") == 1);
    REQUIRE(cfg.resolved_fields.count("rhohv") == 1);
    REQUIRE(cfg.resolved_fields.count("radial_velocity") == 0);
    REQUIRE(cfg.resolved_fields.count("kdp") == 0);

    merge_selected_radar_output_fields(cfg, {"radial_velocity", "kdp"});
    REQUIRE(cfg.resolved_fields.count("reflectivity_dbz") == 0);
    REQUIRE(cfg.resolved_fields.count("zdr") == 0);
    REQUIRE(cfg.resolved_fields.count("rhohv") == 0);
    REQUIRE(cfg.resolved_fields.count("radial_velocity") == 1);
    REQUIRE(cfg.resolved_fields.count("kdp") == 1);
    REQUIRE_THROWS_AS(
        merge_selected_radar_output_fields(cfg, {"not_a_product"}),
        std::invalid_argument);
}

TEST_CASE("custom output field selection rejects substitutions", "[core][output_config][config-truth]")
{
    OutputConfig cfg;
    cfg.preset = FieldPreset::custom;

    SECTION("empty custom selection")
    {
        REQUIRE_THROWS_AS(resolve_output_fields(cfg), std::invalid_argument);
    }

    SECTION("unknown field mixed with a valid field")
    {
        cfg.custom_fields = {"theta", "definitely_not_a_field"};
        REQUIRE_THROWS_AS(resolve_output_fields(cfg), std::invalid_argument);
    }
}

TEST_CASE("output registry identifies lazy diagnostic consumers", "[core][output_config][cpu016][cpu017]")
{
    const OutputFieldDescriptor* w_field      = find_output_field("w");
    const OutputFieldDescriptor* vorticity    = find_output_field("vorticity_z");
    const OutputFieldDescriptor* derived      = find_output_field("vorticity_magnitude");
    const OutputFieldDescriptor* temperature  = find_output_field("temperature");
    const OutputFieldDescriptor* reflectivity = find_output_field("reflectivity_dbz");
    const OutputFieldDescriptor* radial       = find_output_field("radial_velocity");
    const OutputFieldDescriptor* zdr          = find_output_field("zdr");
    const OutputFieldDescriptor* angular      = find_output_field("angular_momentum");
    const OutputFieldDescriptor* pressure     = find_output_field("dynamic_pressure");
    const OutputFieldDescriptor* surface      = find_output_field("surface_sensible_heat_flux");

    REQUIRE(w_field != nullptr);
    REQUIRE(vorticity != nullptr);
    REQUIRE(derived != nullptr);
    REQUIRE(temperature != nullptr);
    REQUIRE(reflectivity != nullptr);
    REQUIRE(radial != nullptr);
    REQUIRE(zdr != nullptr);
    REQUIRE(angular != nullptr);
    REQUIRE(pressure != nullptr);
    REQUIRE(surface != nullptr);
    REQUIRE_FALSE(output_field_requires_lazy_diagnostics(*w_field));
    REQUIRE(output_field_requires_lazy_diagnostics(*vorticity));
    REQUIRE(output_field_requires_lazy_diagnostics(*derived));
    REQUIRE_FALSE(output_field_requires_lazy_diagnostics(*temperature));
    REQUIRE(output_field_has_dependency(reflectivity->dependencies, OutputFieldDependency::RadarDiagnostic));
    REQUIRE(output_field_has_dependency(radial->dependencies, OutputFieldDependency::RadarDiagnostic));
    REQUIRE(output_field_has_dependency(zdr->dependencies, OutputFieldDependency::RadarDiagnostic));
    REQUIRE_FALSE(output_field_has_dependency(reflectivity->dependencies, OutputFieldDependency::VorticityDiagnostics));
    REQUIRE(output_field_has_dependency(vorticity->dependencies, OutputFieldDependency::VorticityDiagnostics));
    REQUIRE_FALSE(output_field_has_dependency(vorticity->dependencies, OutputFieldDependency::RadarDiagnostic));
    REQUIRE(output_field_has_dependency(angular->dependencies, OutputFieldDependency::AngularMomentumDiagnostics));
    REQUIRE_FALSE(output_field_has_dependency(angular->dependencies, OutputFieldDependency::VorticityDiagnostics));
    REQUIRE(output_field_has_dependency(pressure->dependencies, OutputFieldDependency::PressureDiagnostics));
    REQUIRE_FALSE(
        output_field_has_dependency(pressure->dependencies, OutputFieldDependency::AngularMomentumDiagnostics));
    REQUIRE_FALSE(output_field_requires_lazy_diagnostics(*surface));
}

TEST_CASE("thermodynamic derived slices do not require lazy diagnostic storage", "[core][output_config][cpu016]")
{
    constexpr int         nr         = 2;
    constexpr int         nth        = 2;
    constexpr int         nz         = 2;
    constexpr std::size_t slice_size = nr * nz;

    const Field3D                         rho(nr, nth, nz, 1.0f);
    const Field3D                         pressure(nr, nth, nz, 90000.0f);
    const Field3D                         u(nr, nth, nz, 0.0f);
    const Field3D                         v(nr, nth, nz, 0.0f);
    const Field3D                         w(nr, nth, nz, 0.0f);
    const Field3D                         theta(nr, nth, nz, 300.0f);
    Field3D                               qv(nr, nth, nz, 0.005f);
    Field3D                               hydro(nr, nth, nz, 0.001f);
    const Field3D                         radar_zdr(nr, nth, nz, 2.5f);
    const Field3D                         radar_kdp(nr, nth, nz, 0.4f);
    const Field3D                         radar_rhohv(nr, nth, nz, 0.97f);
    const Field3D                         unallocated_lazy_field;
    const std::vector<float>              theta_mean(slice_size, 300.0f);
    const std::vector<float>              storm_mean(nz, 0.0f);
    const std::vector<double>             theta0(nz, 300.0);
    const std::unordered_set<std::string> temperature_only = {"temperature"};
    DerivedSliceScratch                   slices(slice_size, temperature_only);

    const DerivedSliceComputationContext context{
        nr,
        nth,
        nz,
        1000.0,
        1000.0,
        500.0,
        1.0,
        CoordinateSystem::Cartesian,
        OutputFieldDependency::None,
        0,
        -30.0f,
        80.0f,
        theta_mean,
        storm_mean,
        storm_mean,
        rho,
        pressure,
        u,
        v,
        w,
        theta,
        qv,
        hydro,
        hydro,
        hydro,
        hydro,
        hydro,
        hydro,
        unallocated_lazy_field,
        unallocated_lazy_field,
        radar_zdr,
        radar_kdp,
        radar_rhohv,
        unallocated_lazy_field,
        unallocated_lazy_field,
        unallocated_lazy_field,
        unallocated_lazy_field,
        theta0,
        nullptr,
        0.0,
    };

    REQUIRE_NOTHROW(compute_derived_slices_for_theta(slices, context));
    const std::vector<float>* temperature_values = derived_slice_buffer(slices, "temperature");
    REQUIRE(temperature_values != nullptr);
    for (float temperature : *temperature_values)
    {
        REQUIRE(std::isfinite(temperature));
    }

    const std::vector<std::string> independently_selected_products = {
        "relative_humidity",
        "cloud_liquid_condensate",
        "cloud_frozen_condensate",
        "theta_w",
        "zdr",
        "kdp",
        "rhohv",
        "pressure_gradient_force_x",
        "q_vectors",
        "forward_trajectories",
        "vorticity_trajectory",
        "rhi_slice",
    };
    for (const std::string& field_id : independently_selected_products)
    {
        CAPTURE(field_id);
        const std::unordered_set<std::string> selected = {field_id};
        DerivedSliceScratch                   selected_slices(slice_size, selected);
        REQUIRE_NOTHROW(compute_derived_slices_for_theta(selected_slices, context));
        const std::vector<float>* values = derived_slice_buffer(selected_slices, field_id);
        REQUIRE(values != nullptr);
        REQUIRE(values->size() == slice_size);
        for (float value : *values)
        {
            REQUIRE(std::isfinite(value));
            if (field_id == "cloud_liquid_condensate")
            {
                REQUIRE(value == Approx(0.001f));
            }
            if (field_id == "cloud_frozen_condensate")
            {
                REQUIRE(value == Approx(0.002f));
            }
            if (field_id == "zdr")
            {
                REQUIRE(value == Approx(2.5f));
            }
            if (field_id == "kdp")
            {
                REQUIRE(value == Approx(0.4f));
            }
            if (field_id == "rhohv")
            {
                REQUIRE(value == Approx(0.97f));
            }
        }
    }

    qv[0][0][0]                                             = -0.001f;
    hydro[0][0][0]                                          = -0.001f;
    const std::unordered_set<std::string> invalid_sensitive = {"total_condensate", "theta_v"};
    DerivedSliceScratch                   invalid_slices(slice_size, invalid_sensitive);
    REQUIRE_NOTHROW(compute_derived_slices_for_theta(invalid_slices, context));
    REQUIRE(std::isnan(derived_slice_buffer(invalid_slices, "total_condensate")->at(0)));
    REQUIRE(std::isnan(derived_slice_buffer(invalid_slices, "theta_v")->at(0)));
}

TEST_CASE("derived Ertel PV combines physical theta gradient with absolute vorticity",
          "[core][output_config][potential-vorticity][science]")
{
    constexpr int nr = 4;
    constexpr int nth = 3;
    constexpr int nz = 4;
    constexpr double dx = 700.0;
    constexpr double dy = 900.0;
    constexpr double dz_local = 300.0;
    constexpr double grad_x = 0.003;
    constexpr double grad_y = -0.002;
    constexpr double grad_z = 0.006;
    constexpr double omega_x = 0.011;
    constexpr double omega_y = -0.004;
    constexpr double omega_z = 0.018;
    constexpr double coriolis = 8.0e-5;
    constexpr double density = 0.9;
    constexpr int selected_y = 1;

    Field3D rho_field(nr, nth, nz, static_cast<float>(density));
    Field3D pressure(nr, nth, nz, 90000.0f);
    Field3D zero(nr, nth, nz, 0.0f);
    Field3D theta_field(nr, nth, nz, 0.0f);
    Field3D vort_x(nr, nth, nz, static_cast<float>(omega_x));
    Field3D vort_y(nr, nth, nz, static_cast<float>(omega_y));
    Field3D vort_z_field(nr, nth, nz, static_cast<float>(omega_z));
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < nth; ++j)
            for (int k = 0; k < nz; ++k)
                theta_field(i, j, k) = static_cast<float>(
                    290.0 + grad_x * static_cast<double>(i) * dx
                    + grad_y * static_cast<double>(j) * dy
                    + grad_z * (static_cast<double>(k) + 0.5) * dz_local);

    const std::size_t slice_size = static_cast<std::size_t>(nr * nz);
    const std::vector<float> theta_mean(slice_size, 300.0f);
    const std::vector<float> wind_mean(nz, 0.0f);
    const std::vector<double> theta0(nz, 300.0);
    const std::unordered_set<std::string> selected = {"potential_vorticity"};
    DerivedSliceScratch slices(slice_size, selected);
    const DerivedSliceComputationContext context{
        nr, nth, nz, dx, dy, dz_local, 1.0,
        CoordinateSystem::Cartesian,
        OutputFieldDependency::VorticityDiagnostics,
        selected_y, -30.0f, 80.0f,
        theta_mean, wind_mean, wind_mean,
        rho_field, pressure, zero, zero, zero, theta_field,
        zero, zero, zero, zero, zero, zero, zero,
        zero, zero, zero, zero, zero, vort_x, vort_y, vort_z_field, zero, theta0,
        nullptr, coriolis};
    REQUIRE_NOTHROW(compute_derived_slices_for_theta(slices, context));
    const std::vector<float>* pv =
        derived_slice_buffer(slices, "potential_vorticity");
    REQUIRE(pv != nullptr);
    const double expected =
        (omega_x * grad_x + omega_y * grad_y
         + (omega_z + coriolis) * grad_z) / density;
    for (float value : *pv)
        CHECK(value == Approx(expected).margin(2.0e-8));
}

TEST_CASE("parse_output_config reads format", "[core][output_config]")
{
    std::unordered_map<std::string, std::string> kv;
    kv["output.format"] = "npy_3d";
    auto cfg            = parse_output_config(kv);
    REQUIRE(cfg.format == OutputFormat::npy_3d);
}

TEST_CASE("parse_output_config reads zfp settings", "[core][output_config]")
{
    std::unordered_map<std::string, std::string> kv;
    kv["output.format"]                = "zfp";
    kv["output.zfp_tolerance"]         = "1.0e-5";
    kv["output.zfp_mode"]              = "accuracy";
    kv["output.zfp_keyframe_interval"] = "10";

#ifdef HAVE_ZFP
    auto cfg = parse_output_config(kv);
    REQUIRE(cfg.format == OutputFormat::zfp);
    REQUIRE(cfg.zfp_tolerance == Approx(1.0e-5));
    REQUIRE(cfg.zfp_mode == "accuracy");
    REQUIRE(cfg.zfp_precision_bits == 16);
    REQUIRE(cfg.zfp_keyframe_interval == 10);
#else
    REQUIRE_THROWS_AS(parse_output_config(kv), std::runtime_error);
#endif
}

TEST_CASE("ZFP cannot erase SI Ertel PV with a broad global tolerance",
          "[core][output_config][zfp][potential-vorticity][science]")
{
    OutputConfig config;
    config.zfp_tolerance = 1.0e-3;
    REQUIRE(effective_zfp_accuracy_tolerance(config, "theta")
            == Approx(1.0e-3));
    REQUIRE(effective_zfp_accuracy_tolerance(config, "potential_vorticity")
            == Approx(1.0e-8));
    config.zfp_field_tolerances["potential_vorticity"] = 1.0e-10;
    REQUIRE(effective_zfp_accuracy_tolerance(config, "potential_vorticity")
            == Approx(1.0e-10));
}

TEST_CASE("parse_output_config gives precision mode its own bounded bit parameter",
          "[core][output_config][zfp][science]")
{
#ifdef HAVE_ZFP
    std::unordered_map<std::string, std::string> kv{
        {"output.format", "zfp_high"},
        {"output.zfp_mode", "precision"},
        {"output.zfp_precision_bits", "23"},
    };
    REQUIRE(parse_output_config(kv).zfp_precision_bits == 23);
    kv["output.zfp_precision_bits"] = "33";
    REQUIRE_THROWS_AS(parse_output_config(kv), std::runtime_error);
    kv["output.zfp_precision_bits"] = "16";
    kv["output.zfp_mode"]           = "invented";
    REQUIRE_THROWS_AS(parse_output_config(kv), std::runtime_error);
#else
    SUCCEED("binary compiled without ZFP support");
#endif
}

TEST_CASE("step manifest records exact per-field ZFP transform provenance",
          "[core][output_config][manifest][zfp][science]")
{
    OutputStepManifestContext context;
    context.storage_format   = "zfp";
    context.zfp_field_execution = "field_parallel";
    context.zfp_field_workers = 4;
    context.zfp_field_scratch_mib = 128;
    context.published_fields = {"theta", "u10"};
    context.field_brick_hierarchies = true;
    OutputStepManifestCompression compression;
    compression.field_id             = "theta";
    compression.mode                 = "accuracy";
    compression.mode_parameter       = 1.0e-4;
    compression.mode_parameter_units = "absolute_field_units";
    compression.error_contract       = "zfp_absolute_accuracy_before_prefilters";
    compression.frame_role           = "predictive_delta";
    compression.parent_step_indices  = {6, 5};
    compression.sparse_threshold     = 0.0;
    compression.float16_prefilter    = false;
    context.compression.push_back(compression);

    const std::string manifest = build_output_step_manifest_json(context);
    REQUIRE(manifest.find("\"lossy_compression_contract\": \"nimbus.zfp.per_field.v1\"") != std::string::npos);
    REQUIRE(manifest.find("\"field_execution_policy\": \"field_parallel\"") != std::string::npos);
    REQUIRE(manifest.find("\"field_workers\": 4") != std::string::npos);
    REQUIRE(manifest.find("\"field_scratch_budget_mib\": 128") != std::string::npos);
    REQUIRE(manifest.find("\"file_pattern\": \"theta.zfp3d\"") != std::string::npos);
    REQUIRE(manifest.find("\"file_pattern\": \"u10.npy\"") != std::string::npos);
    REQUIRE(manifest.find("\"mode\": \"accuracy\"") != std::string::npos);
    REQUIRE(manifest.find("\"mode_parameter_units\": \"absolute_field_units\"") != std::string::npos);
    REQUIRE(manifest.find("\"frame_role\": \"predictive_delta\"") != std::string::npos);
    REQUIRE(manifest.find("\"parent_step_indices\": [6, 5]") != std::string::npos);
    REQUIRE(manifest.find("\"decoded_dtype\": \"float32\"") != std::string::npos);
    REQUIRE(manifest.find("\"container_version\": 3") != std::string::npos);
    REQUIRE(manifest.find("\"spatial_layout\": \"independently_decodable_bricks\"") != std::string::npos);
    REQUIRE(manifest.find("\"brick_shape\": [64, 64, 32]") != std::string::npos);
    REQUIRE(manifest.find("\"path\": \"theta.bricks.json\"") != std::string::npos);
    REQUIRE(manifest.find("nimbus-pending-field-brick-sha256:theta") != std::string::npos);
    REQUIRE(manifest.find("\"extrema_semantics\": \"decoded_value_bounds\"") != std::string::npos);
    REQUIRE(manifest.find("\"extrema_absolute_padding\": 0.0001") != std::string::npos);
    REQUIRE(manifest.find("\"culling_safe\": true") != std::string::npos);
    REQUIRE(manifest.find("\"path\": \"u10.bricks.json\"") == std::string::npos);
    context.compression[0].float16_prefilter = true;
    const std::string float16_manifest = build_output_step_manifest_json(context);
    REQUIRE(float16_manifest.find("\"extrema_semantics\": \"pre_storage_source\"") != std::string::npos);
    REQUIRE(float16_manifest.find("\"culling_safe\": false") != std::string::npos);
}

TEST_CASE("parse_output_config accepts zfp tier spellings", "[core][output_config]")
{
#ifdef HAVE_ZFP
    // Each v1 tier name selects the single writer zfp format and sets its
    // default accuracy tolerance (zfp_low=1e-2, zfp_medium=1e-3, zfp_high=1e-4).
    const std::pair<const char*, double> tiers[] = {
        {"zfp_low", 1.0e-2},
        {"zfp_medium", 1.0e-3},
        {"zfp_high", 1.0e-4},
    };
    for (const auto& t : tiers)
    {
        std::unordered_map<std::string, std::string> kv;
        kv["output.format"] = t.first;
        auto cfg            = parse_output_config(kv);
        REQUIRE(cfg.format == OutputFormat::zfp);
        REQUIRE(cfg.zfp_requested_format_tier == t.first);
        REQUIRE(cfg.zfp_tolerance == Approx(t.second));
    }

    // An explicit output.zfp_tolerance overrides the tier default.
    std::unordered_map<std::string, std::string> kv;
    kv["output.format"]        = "zfp_low";
    kv["output.zfp_tolerance"] = "5.0e-4";
    auto cfg                   = parse_output_config(kv);
    REQUIRE(cfg.format == OutputFormat::zfp);
    REQUIRE(cfg.zfp_requested_format_tier == "zfp_low");
    REQUIRE(cfg.zfp_tolerance == Approx(5.0e-4));
#else
    SUCCEED("binary compiled without ZFP support");
#endif
}

TEST_CASE("parse_output_config reads async_io", "[core][output_config]")
{
    std::unordered_map<std::string, std::string> kv;
    kv["output.async_io"] = "true";
    auto cfg              = parse_output_config(kv);
    REQUIRE(cfg.async_io == true);
}

TEST_CASE("parse_output_config reports exact-binary netCDF capability without substituting NPY",
          "[core][output_config][netcdf][sci021]")
{
    std::unordered_map<std::string, std::string> kv;
    kv["output.format"] = "netcdf";
#ifdef HAVE_NETCDF
    REQUIRE(parse_output_config(kv).format == OutputFormat::netcdf);
#else
    REQUIRE_THROWS_AS(parse_output_config(kv), std::runtime_error);
#endif

    kv["output.format"] = "not_a_format";
    REQUIRE_THROWS_AS(parse_output_config(kv), std::runtime_error);
}

TEST_CASE("parse_output_config validates netCDF layout policy", "[core][output_config][netcdf]")
{
    std::unordered_map<std::string, std::string> kv;
    kv["output.netcdf_deflate_level"]    = "4";
    kv["output.netcdf_shuffle"]          = "false";
    kv["output.netcdf_target_chunk_kib"] = "256";
    const OutputConfig config            = parse_output_config(kv);
    REQUIRE(config.netcdf_deflate_level == 4);
    REQUIRE_FALSE(config.netcdf_shuffle);
    REQUIRE(config.netcdf_target_chunk_kib == 256);

    kv["output.netcdf_deflate_level"] = "10";
    REQUIRE_THROWS_AS(parse_output_config(kv), std::runtime_error);
}

TEST_CASE("parse_output_config reads csv format", "[core][output_config]")
{
    std::unordered_map<std::string, std::string> kv;
    kv["output.format"] = "csv";
    auto cfg            = parse_output_config(kv);
    REQUIRE(cfg.format == OutputFormat::csv);
}

TEST_CASE("parse_output_config reads trajectory controls", "[core][output_config][trajectories]")
{
    std::unordered_map<std::string, std::string> kv = {
        {"output.trajectories", "true"},
        {"output.trajectory_seed_time_s", "120.5"},
        {"output.trajectory_count", "7"},
        {"output.trajectory_sample_interval_s", "2.5"},
        {"output.trajectory_max_samples_per_parcel", "1200"}};
    const OutputConfig cfg = parse_output_config(kv);
    REQUIRE(cfg.trajectories);
    REQUIRE(cfg.trajectory_seed_time_s == Approx(120.5));
    REQUIRE(cfg.trajectory_count == 7);
    REQUIRE(cfg.trajectory_sample_interval_s == Approx(2.5));
    REQUIRE(cfg.trajectory_max_samples_per_parcel == 1200);

    kv["output.trajectory_seed_time_s"] = "-1";
    REQUIRE_THROWS_AS(parse_output_config(kv), std::invalid_argument);
    kv["output.trajectory_seed_time_s"] = "0";
    kv["output.trajectory_sample_interval_s"] = "-1";
    REQUIRE_THROWS_AS(parse_output_config(kv), std::invalid_argument);
    kv["output.trajectory_sample_interval_s"] = "1";
    kv["output.trajectory_max_samples_per_parcel"] = "0";
    REQUIRE_THROWS_AS(parse_output_config(kv), std::invalid_argument);
}

TEST_CASE("estimate_disk_budget populates estimates", "[core][output_config]")
{
    OutputConfig cfg;
    cfg.format = OutputFormat::npy_3d;
    cfg.preset = FieldPreset::minimal;
    resolve_output_fields(cfg);

    estimate_disk_budget(cfg, 64, 64, 32, 3600, 60);

    REQUIRE(cfg.estimated_bytes_per_export > 0);
    REQUIRE(cfg.estimated_total_bytes > 0);
    REQUIRE(cfg.estimated_total_bytes >= cfg.estimated_bytes_per_export);
}

TEST_CASE("CSV disk budget accounts for formatted dense rows",
          "[core][output_config][budget][csv]")
{
    OutputConfig npy;
    npy.format = OutputFormat::npy_3d;
    npy.preset = FieldPreset::custom;
    npy.custom_fields = {"u"};
    resolve_output_fields(npy);
    estimate_disk_budget(npy, 10, 9, 8, 10, 10);

    OutputConfig csv = npy;
    csv.format = OutputFormat::csv;
    estimate_disk_budget(csv, 10, 9, 8, 10, 10);
    CHECK(csv.estimated_bytes_per_export ==
          npy.estimated_bytes_per_export * 12U);
}

TEST_CASE("disk budget includes initialized frame and manifest sidecars", "[core][output_config][budget]")
{
    OutputConfig cfg;
    cfg.format = OutputFormat::npy_3d;
    cfg.preset = FieldPreset::minimal;
    resolve_output_fields(cfg);

    estimate_disk_budget(cfg, 16, 16, 12, 20, 10);

    constexpr std::size_t export_count        = 3;
    constexpr std::size_t metadata_per_export = 64 * 1024;
    REQUIRE(cfg.estimated_total_bytes == export_count * (cfg.estimated_bytes_per_export + metadata_per_export));
}

TEST_CASE("output_format_name returns valid strings", "[core][output_config]")
{
    REQUIRE(std::string(output_format_name(OutputFormat::npy_3d)) != "");
    REQUIRE(std::string(output_format_name(OutputFormat::zfp)) != "");
    REQUIRE(std::string(output_format_name(OutputFormat::csv)) != "");
}

TEST_CASE("field_preset_name returns valid strings", "[core][output_config]")
{
    REQUIRE(std::string(field_preset_name(FieldPreset::all)) != "");
    REQUIRE(std::string(field_preset_name(FieldPreset::minimal)) != "");
    REQUIRE(std::string(field_preset_name(FieldPreset::presentation)) != "");
    REQUIRE(std::string(field_preset_name(FieldPreset::studio_full)) == "studio_full");
    REQUIRE(std::string(field_preset_name(FieldPreset::science_validation)) ==
            "science_validation");
}

TEST_CASE("get_all_field_names is non-empty", "[core][output_config]")
{
    const auto& names = get_all_field_names();
    REQUIRE(names.size() > 20);
}

TEST_CASE("export field registry covers all configured output fields", "[core][output_config][field_registry]")
{
    const auto&                     names    = get_all_field_names();
    const auto&                     registry = output_field_registry();
    std::unordered_set<std::string> seen;

    REQUIRE(registry.size() == names.size());
    for (const auto& name : names)
    {
        CAPTURE(name);
        REQUIRE(seen.insert(name).second);
        REQUIRE(find_output_field(name) != nullptr);
        REQUIRE(tmv::find_field_contract(name) != nullptr);
    }
}

TEST_CASE("PBL surface outputs are direct 2D fields", "[core][output_config][field_registry]")
{
    const char* direct_surface_fields[] = {
        "surface_sensible_heat_flux",
        "surface_latent_heat_flux",
        "surface_moisture_flux",
        "surface_friction_velocity",
        "surface_stress_u",
        "surface_stress_v",
        "pbl_height",
    };

    for (const char* name : direct_surface_fields)
    {
        CAPTURE(name);
        const auto* field = find_output_field(name);
        REQUIRE(field != nullptr);
        REQUIRE(output_field_is_direct_surface_2d(*field));
        REQUIRE_FALSE(output_field_is_tranche_product(*field));
    }
}

TEST_CASE("step manifest uses the shared output field registry", "[core][output_config][field_registry][manifest]")
{
    OutputStepManifestContext context;
    context.step_index                     = 7;
    context.simulation_time_s              = 12.5;
    context.step_dir                       = "step_000007";
    context.publication_generation         = "0123456789abcdef0123456789abcdef";
    context.grid.nr                        = 4;
    context.grid.nth                       = 3;
    context.grid.nz                        = 2;
    context.grid.coordinate_system         = "cartesian";
    context.grid.staggering                = "c_grid";
    context.grid.dr_m                      = 1000.0;
    context.grid.dy_m                      = 0.5;
    context.grid.dtheta_rad                = 0.5; // legacy alias
    context.grid.dz_m                      = 250.0;
    context.soundings.enabled              = true;
    context.soundings.scheme               = "test";
    context.soundings.file_path            = "profiles/test.json";
    context.soundings.interpolation_method = "linear";
    context.published_fields               = {"theta", "w"};
    context.field_brick_hierarchies        = true;

    const std::string manifest = build_output_step_manifest_json(context);

    REQUIRE(manifest.find("\"schema\": \"nimbus.step\"") != std::string::npos);
    REQUIRE(manifest.find("\"manifest_version\": 7") != std::string::npos);
    REQUIRE(manifest.find("\"publication_state\": \"complete\"") != std::string::npos);
    REQUIRE(manifest.find("\"publication_generation\": \"0123456789abcdef0123456789abcdef\"") != std::string::npos);
    REQUIRE(manifest.find("\"published_fields\": [\"theta\", \"w\"]") != std::string::npos);
    REQUIRE(manifest.find("\"step_index\": 7") != std::string::npos);
    REQUIRE(manifest.find("\"step_dir\": \"step_000007\"") != std::string::npos);
    REQUIRE(manifest.find("\"coordinate_system\": \"cartesian\"") != std::string::npos);
    REQUIRE(manifest.find("\"data_product\": {") != std::string::npos);
    REQUIRE(manifest.find("\"kind\": \"history\"") != std::string::npos);
    REQUIRE(manifest.find("\"restart_complete\": false") != std::string::npos);
    REQUIRE(manifest.find("\"dimensions\": {\"x\": 4, \"y\": 3, \"z\": 2}") != std::string::npos);
    REQUIRE(manifest.find("\"spacing_m\": {\"x\": 1000, \"y\": 0.5, \"z\": 250}") != std::string::npos);
    REQUIRE(manifest.find("\"native_model_staggering\": \"c_grid\"") != std::string::npos);
    REQUIRE(manifest.find("\"publication_staggering\": \"cell_centered\"") != std::string::npos);
    REQUIRE(manifest.find("\"theta_index\"") == std::string::npos);
    REQUIRE(manifest.find("\"scalar_origin_m\": [500, 0.25, 125]") != std::string::npos);
    REQUIRE(manifest.find("\"physical_bounds_m\": {\"lower\": [1000, 0.5, 0], \"upper\": [3000, 1, 500]}") !=
            std::string::npos);
    REQUIRE(manifest.find("\"valid_index_range\": {\"lower\": [1, 1, 0], \"upper\": [2, 1, 1]}") != std::string::npos);
    REQUIRE(manifest.find("\"max\": 2") != std::string::npos);
    REQUIRE(manifest.find("\"interpolation_method\": \"linear\"") != std::string::npos);
    REQUIRE(manifest.find("\"file_pattern\": \"theta.npy\"") != std::string::npos);
    REQUIRE(manifest.find("\"grid_location\": \"cell_center\"") != std::string::npos);
    REQUIRE(manifest.find("\"cm1_aliases\": [\"th\"]") != std::string::npos);
    REQUIRE(manifest.find("\"path\": \"theta.bricks.json\"") != std::string::npos);
    REQUIRE(manifest.find("\"path\": \"w.bricks.json\"") != std::string::npos);
    REQUIRE(manifest.find("\"extrema_semantics\": \"stored_values\"") != std::string::npos);
    REQUIRE(manifest.find("\"culling_safe\": true") != std::string::npos);

    for (const auto& field : output_field_registry())
    {
        CAPTURE(field.field_id);
        const std::string field_id_entry = std::string("\"field_id\": \"") + field.field_id + "\"";
        const std::string suffix_entry   = std::string("\"file_suffix\": \"") + field.suffix + "\"";
        REQUIRE(manifest.find(field_id_entry) != std::string::npos);
        REQUIRE(manifest.find(suffix_entry) != std::string::npos);
    }
}

TEST_CASE("moving-frame manifest publishes Earth coordinates and lossless frame provenance",
          "[core][output_config][manifest][domain-motion][SCI-195]")
{
    OutputStepManifestContext context;
    context.publication_generation        = "moving-frame";
    context.simulation_time_s             = 30.0;
    context.grid.nr                       = 6;
    context.grid.nth                      = 5;
    context.grid.nz                       = 4;
    context.grid.staggering               = "c_grid";
    context.grid.dr_m                     = 100.0;
    context.grid.dy_m                     = 200.0;
    context.grid.dz_m                     = 50.0;
    context.grid.domain_motion_mode       = "constant_velocity";
    context.grid.grid_origin_t0_earth_x_m = 1000.0;
    context.grid.grid_origin_t0_earth_y_m = -2000.0;
    context.grid.grid_origin_earth_x_m    = 1300.0;
    context.grid.grid_origin_earth_y_m    = -2300.0;
    context.grid.frame_velocity_u_ms      = 10.0;
    context.grid.frame_velocity_v_ms      = -10.0;

    const std::string manifest = build_output_step_manifest_json(context);
    REQUIRE(manifest.find("\"motion_mode\": \"constant_velocity\"") != std::string::npos);
    REQUIRE(manifest.find("\"grid_origin_t0_earth_m\": [1000, -2000]") != std::string::npos);
    REQUIRE(manifest.find("\"grid_origin_earth_m\": [1300, -2300]") != std::string::npos);
    REQUIRE(manifest.find("\"frame_velocity_earth_ms\": [10, -10]") != std::string::npos);
    REQUIRE(manifest.find("\"published_horizontal_wind\": \"earth_relative\"") != std::string::npos);
    REQUIRE(manifest.find("\"horizontal_transport_velocity\": \"grid_relative\"") != std::string::npos);
    REQUIRE(manifest.find("\"scalar_origin_m\": [1350, -2200, 25]") != std::string::npos);
    REQUIRE(manifest.find("\"grid_scalar_origin_m\": [50, 100, 25]") != std::string::npos);
    REQUIRE(manifest.find("\"physical_bounds_m\": {\"lower\": [1400, -2100, 0], \"upper\": [1800, -1500, 200]}") !=
            std::string::npos);

    context.grid.grid_origin_earth_x_m = std::numeric_limits<double>::quiet_NaN();
    REQUIRE_THROWS_AS(build_output_step_manifest_json(context), std::invalid_argument);
}

TEST_CASE("step manifest publishes the cold-pool diagnostic algorithm contract",
          "[core][output_config][field_registry][manifest][cold_pool]")
{
    OutputStepManifestContext context;
    context.publication_generation = "cold-pool-contract";
    context.grid.nr                = 4;
    context.grid.nth               = 3;
    context.grid.nz                = 2;
    context.grid.staggering        = "c_grid";
    context.grid.dr_m              = 100.0;
    context.grid.dy_m              = 100.0;
    context.grid.dz_m              = 50.0;
    context.published_fields       = {"cold_pool_boundary"};

    const std::string manifest = build_output_step_manifest_json(context);
    REQUIRE(manifest.find(tmv::derived::cold_pool_boundary_algorithm_id) !=
            std::string::npos);
    REQUIRE(manifest.find("\"grid_location\": \"surface_cell_center\"") !=
            std::string::npos);
    REQUIRE(manifest.find("\"state_semantic\": \"diagnostic\"") !=
            std::string::npos);
}

TEST_CASE("collocated manifest publishes the A-grid physical ownership domain",
          "[core][output_config][manifest][agrid][AG-001]")
{
    OutputStepManifestContext context;
    context.publication_generation = "agrid-domain";
    context.grid.nr                = 6;
    context.grid.nth               = 5;
    context.grid.nz                = 4;
    context.grid.staggering        = "collocated";
    context.grid.dr_m              = 100.0;
    context.grid.dy_m              = 200.0;
    context.grid.dz_m              = 50.0;

    const std::string manifest = build_output_step_manifest_json(context);
    REQUIRE(manifest.find("\"valid_index_range\": {\"lower\": [1, 1, 1], \"upper\": [4, 3, 2]}") != std::string::npos);
    REQUIRE(manifest.find("\"halo_width\": {\"lower\": [1, 1, 1], \"upper\": [1, 1, 1]}") != std::string::npos);
    REQUIRE(manifest.find("\"physical_bounds_m\": {\"lower\": [100, 200, 50], \"upper\": [500, 800, 150]}") !=
            std::string::npos);

    context.grid.staggering = "unknown";
    REQUIRE_THROWS_AS(build_output_step_manifest_json(context), std::invalid_argument);
}

TEST_CASE("terrain manifest declares lossless non-separable vertical geometry",
          "[core][output_config][manifest][terrain][TE-005]")
{
    OutputStepManifestContext context;
    context.publication_generation = "0123456789abcdef0123456789abcdef";
    context.storage_format         = "zfp";
    context.grid.nr                = 6;
    context.grid.nth               = 5;
    context.grid.nz                = 4;
    context.grid.staggering        = "c_grid";
    context.grid.dr_m              = 100.0;
    context.grid.dy_m              = 200.0;
    context.grid.dz_m              = 50.0;
    context.grid.terrain_following = true;
    context.grid.terrain_min_z_m   = 25.0;
    context.grid.terrain_max_z_m   = 240.0;
    context.grid.terrain_geometry_publication_generation = context.publication_generation;
    context.grid.terrain_geometry_sha256 = std::string(64, 'a');

    const std::string manifest = build_output_step_manifest_json(context);
    REQUIRE(manifest.find("\"kind\": \"terrain_following\"") != std::string::npos);
    REQUIRE(manifest.find("\"center_coordinate\": \"physical_z_center_m\"") != std::string::npos);
    REQUIRE(manifest.find("\"interface_coordinate\": \"physical_z_interface_m\"") != std::string::npos);
    REQUIRE(manifest.find("\"dtype\": \"float64\"") != std::string::npos);
    REQUIRE(manifest.find("\"storage\": \"versioned_npy_geometry\"") != std::string::npos);
    REQUIRE(manifest.find("\"contract\": \"nimbus.terrain-geometry.v1\"") != std::string::npos);
    REQUIRE(manifest.find("../geometry/0123456789abcdef0123456789abcdef/") != std::string::npos);
    REQUIRE(manifest.find("\"horizontal_scalar_origin_m\": [50, 100]") != std::string::npos);
    REQUIRE(manifest.find("\"scalar_origin_m\"") == std::string::npos);
    REQUIRE(manifest.find("\"physical_bounds_m\": {\"lower\": [100, 200, 25], \"upper\": [500, 800, 240]}") !=
            std::string::npos);
    REQUIRE(manifest.find("physical_z_center_m.zfp3d") == std::string::npos);

    context.storage_format            = "netcdf";
    const std::string netcdf_manifest = build_output_step_manifest_json(context);
    REQUIRE(netcdf_manifest.find("\"storage\": \"netcdf_variable\"") != std::string::npos);
}

TEST_CASE("run output contracts carry unique publication generations", "[core][output_config][manifest][generation]")
{
    const std::string first  = make_output_publication_generation();
    const std::string second = make_output_publication_generation();
    REQUIRE(first.size() == 32);
    REQUIRE(second.size() == 32);
    REQUIRE(first != second);

    const std::string initializing = build_output_run_contract_json(first, "initializing");
    const std::string ready        = build_output_run_contract_json(first, "ready");
    REQUIRE(initializing.find("\"publication_generation\": \"" + first + "\"") != std::string::npos);
    REQUIRE(initializing.find("\"publication_state\": \"initializing\"") != std::string::npos);
    REQUIRE(ready.find("\"publication_state\": \"ready\"") != std::string::npos);
    REQUIRE(ready.find("\"data_products\": [\"history\"]") != std::string::npos);

    const std::string nested = build_output_run_contract_json(first, "ready", true);
    REQUIRE(nested.find("\"data_products\": [\"history\", \"nested_child_history\"]") != std::string::npos);
    const std::string qualification =
        build_output_run_contract_json(first, "ready", true, true, true);
    REQUIRE(qualification.find(
                "\"data_products\": [\"history\", \"nested_child_history\", "
                "\"nested_child_qualification_diagnostics\"]") !=
            std::string::npos);
    const std::string qualification_only =
        build_output_run_contract_json(first, "ready", false, true, false);
    REQUIRE(qualification_only.find(
                "\"data_products\": [\"qualification_diagnostics\"]") !=
            std::string::npos);
    const std::string nested_qualification_only =
        build_output_run_contract_json(first, "ready", true, true, false);
    REQUIRE(nested_qualification_only.find(
                "\"data_products\": [\"nested_child_qualification_diagnostics\"]") !=
            std::string::npos);
}

TEST_CASE("step manifests distinguish nested child history", "[core][output_config][manifest][nesting]")
{
    OutputStepManifestContext context;
    context.step_index              = 3;
    context.simulation_time_s       = 1.5;
    context.step_dir                = "step_000003";
    context.publication_generation  = "0123456789abcdef0123456789abcdef";
    context.data_product_kind       = "nested_child_history";
    context.grid.nr                 = 10;
    context.grid.nth                = 10;
    context.grid.nz                 = 4;
    context.grid.coordinate_system  = "cartesian";
    context.grid.staggering         = "c_grid";
    context.grid.dr_m               = 125.0;
    context.grid.dy_m               = 125.0;
    context.grid.dz_m               = 100.0;
    context.published_fields        = {"rho", "theta"};

    const std::string manifest = build_output_step_manifest_json(context);
    REQUIRE(manifest.find("\"kind\": \"nested_child_history\"") != std::string::npos);
    REQUIRE(manifest.find("\"dimensions\": {\"x\": 10, \"y\": 10, \"z\": 4}") != std::string::npos);
}

TEST_CASE("step manifests admit both qualification diagnostic product kinds",
          "[core][output_config][manifest][qualification]")
{
    OutputStepManifestContext context;
    context.step_index             = 0;
    context.simulation_time_s      = 30.0;
    context.step_dir               = "qualification_diagnostics/step_000000";
    context.publication_generation = "0123456789abcdef0123456789abcdef";
    context.storage_format         = "npy_3d";
    context.published_fields       = {
        "accumulated_rainfall", "cold_pool_boundary"};
    context.grid.nr                = 10;
    context.grid.nth               = 10;
    context.grid.nz                = 4;
    context.grid.coordinate_system = "cartesian";
    context.grid.staggering        = "c_grid";
    context.grid.dr_m              = 600.0;
    context.grid.dy_m              = 600.0;
    context.grid.dz_m              = 320.0;

    context.data_product_kind = "qualification_diagnostics";
    const std::string root = build_output_step_manifest_json(context);
    REQUIRE(root.find("\"kind\": \"qualification_diagnostics\"") !=
            std::string::npos);

    context.data_product_kind =
        "nested_child_qualification_diagnostics";
    const std::string child = build_output_step_manifest_json(context);
    REQUIRE(child.find(
                "\"kind\": \"nested_child_qualification_diagnostics\"") !=
            std::string::npos);
}

TEST_CASE("derived slice scratch binds every derived volume registry field",
          "[core][output_config][field_registry][derived]")
{
    DerivedSliceScratch              scratch(4);
    std::vector<DerivedSliceBinding> bindings;
    std::string                      error_message;

    REQUIRE(build_derived_slice_bindings(scratch, bindings, error_message));
    REQUIRE(error_message.empty());

    std::unordered_set<std::string> bound_fields;
    for (const auto& binding : bindings)
    {
        REQUIRE(binding.field_id != nullptr);
        REQUIRE(binding.values != nullptr);
        REQUIRE(binding.values->size() == 4);
        REQUIRE(bound_fields.insert(binding.field_id).second);
    }

    for (const auto& field : output_field_registry())
    {
        if (!output_field_is_derived_volume(field))
        {
            continue;
        }
        CAPTURE(field.field_id);
        REQUIRE(bound_fields.count(field.field_id) == 1);
        REQUIRE(derived_slice_buffer(scratch, field.field_id) != nullptr);
    }
}

TEST_CASE("derived slice scratch allocates only the requested dependency closure", "[core][output_config][cpu016]")
{
    SECTION("one independent product owns one result plus the discard sink")
    {
        const std::unordered_set<std::string> fields = {"temperature"};
        DerivedSliceScratch                   scratch(8, fields);
        REQUIRE(scratch.requested_field_count() == 1);
        REQUIRE(scratch.allocated_array_count() == 2);
        REQUIRE(derived_slice_scratch_array_count(fields) == 2);
        REQUIRE(derived_slice_buffer(scratch, "temperature") != nullptr);
        REQUIRE(derived_slice_buffer(scratch, "vorticity_magnitude") == nullptr);
        REQUIRE(scratch.requests_only_simple_pointwise_products());
    }

    SECTION("dependent products retain only their intermediate slices")
    {
        const std::unordered_set<std::string> wet_bulb = {"theta_w"};
        DerivedSliceScratch                   scratch(8, wet_bulb);
        REQUIRE(scratch.requested_field_count() == 1);
        REQUIRE(scratch.allocated_array_count() == 3);
        REQUIRE(derived_slice_buffer(scratch, "theta_w") != nullptr);
        REQUIRE(derived_slice_buffer(scratch, "temperature") != nullptr);
        REQUIRE(derived_slice_buffer(scratch, "relative_humidity") == nullptr);
        REQUIRE(derived_slice_buffer(scratch, "dewpoint") == nullptr);
        REQUIRE_FALSE(scratch.requests_only_simple_pointwise_products());
    }

    SECTION("tranche-only output owns no derived slice scratch")
    {
        const std::unordered_set<std::string> tranche = {"cape"};
        REQUIRE(derived_slice_scratch_array_count(tranche) == 0);
        REQUIRE(active_output_fields_need_tranche_products(tranche));
        REQUIRE_FALSE(active_output_fields_need_derived_volume_slices(tranche));
    }

    SECTION("derived-only output does not need the tranche registry")
    {
        const std::unordered_set<std::string> derived = {"temperature"};
        REQUIRE(active_output_fields_need_derived_volume_slices(derived));
        REQUIRE_FALSE(active_output_fields_need_tranche_products(derived));
    }
}

TEST_CASE("derived output helpers allocate only active derived volumes",
          "[core][output_config][field_registry][derived]")
{
    std::unordered_set<std::string> active_fields = {
        "temperature",
        "cape",
        "u",
    };
    std::unordered_map<std::string, std::vector<float>> buffers;

    REQUIRE(active_output_fields_need_derived_slices(active_fields));

    prepare_derived_volume_output_buffers(active_fields, true, 6, buffers);
    REQUIRE(buffers.size() == 1);
    REQUIRE(buffers.count("temperature") == 1);
    REQUIRE(buffers["temperature"].size() == 6);

    prepare_derived_volume_output_buffers(active_fields, false, 6, buffers);
    REQUIRE(buffers.empty());

    active_fields = {"u", "theta"};
    REQUIRE_FALSE(active_output_fields_need_derived_slices(active_fields));

    // A custom output containing only a derived product still needs its
    // gridded inputs refreshed before the t=0 publication.
    active_fields = {"vorticity_magnitude"};
    REQUIRE(active_output_fields_need_derived_slices(active_fields));
}

TEST_CASE("derived slice scatter writes a theta plane into a 3D volume",
          "[core][output_config][field_registry][derived]")
{
    constexpr int            nr    = 2;
    constexpr int            nth   = 3;
    constexpr int            nz    = 2;
    const std::vector<float> slice = {
        1.0f,
        2.0f,
        3.0f,
        4.0f,
    };
    std::vector<float> volume(
        static_cast<std::size_t>(nr) * static_cast<std::size_t>(nth) * static_cast<std::size_t>(nz), -1.0f);

    scatter_derived_slice_to_volume(slice, volume, 1, nr, nth, nz);

    const auto volume_index = [=](int i, int j, int k)
    {
        return static_cast<std::size_t>(i) * static_cast<std::size_t>(nth) * static_cast<std::size_t>(nz) +
               static_cast<std::size_t>(j) * static_cast<std::size_t>(nz) + static_cast<std::size_t>(k);
    };

    REQUIRE(volume[volume_index(0, 1, 0)] == Approx(1.0f));
    REQUIRE(volume[volume_index(1, 1, 0)] == Approx(2.0f));
    REQUIRE(volume[volume_index(0, 1, 1)] == Approx(3.0f));
    REQUIRE(volume[volume_index(1, 1, 1)] == Approx(4.0f));
    REQUIRE(volume[volume_index(0, 0, 0)] == Approx(-1.0f));
    REQUIRE(volume[volume_index(1, 2, 1)] == Approx(-1.0f));
}

TEST_CASE("derived slice writer validates bindings and scatters active 3D buffers",
          "[core][output_config][field_registry][derived]")
{
    std::vector<float> temperature = {
        1.0f,
        2.0f,
        3.0f,
        4.0f,
    };
    std::vector<float> dewpoint = {
        10.0f,
        20.0f,
        30.0f,
        40.0f,
    };
    const DerivedSliceBinding bindings[] = {
        {"temperature", &temperature},
        {"dewpoint", &dewpoint},
    };
    const std::unordered_set<std::string>               active_fields = {"temperature"};
    std::unordered_map<std::string, std::vector<float>> volume_buffers;
    volume_buffers["temperature"].assign(2 * 3 * 2, -1.0f);
    int validated_count = 0;

    const bool ok = write_or_scatter_derived_slice_fields(
        bindings,
        2,
        active_fields,
        volume_buffers,
        [&](const char*, std::vector<float>& values)
        {
            ++validated_count;
            REQUIRE_FALSE(values.empty());
            return true;
        },
        true,
        1,
        2,
        3,
        2,
        "unused_step");

    const auto volume_index = [](int i, int j, int k)
    { return static_cast<std::size_t>(i) * 3U * 2U + static_cast<std::size_t>(j) * 2U + static_cast<std::size_t>(k); };

    REQUIRE(ok);
    REQUIRE(validated_count == 2);
    REQUIRE(volume_buffers["temperature"][volume_index(0, 1, 0)] == Approx(1.0f));
    REQUIRE(volume_buffers["temperature"][volume_index(1, 1, 0)] == Approx(2.0f));
    REQUIRE(volume_buffers["temperature"][volume_index(0, 1, 1)] == Approx(3.0f));
    REQUIRE(volume_buffers["temperature"][volume_index(1, 1, 1)] == Approx(4.0f));
    REQUIRE(volume_buffers.count("dewpoint") == 0);
}

TEST_CASE("derived volume writer snapshots async 3D buffers", "[core][output_config][field_registry][derived]")
{
    std::unordered_map<std::string, std::vector<float>> buffers;
    buffers["temperature"] = {1.0f, 2.0f, 3.0f, 4.0f};
    ExportSnapshot snapshot;

    write_or_snapshot_derived_volume_fields(buffers, snapshot, true, OutputFormat::npy_3d, 2, 1, 2, "unused_step");

    REQUIRE(snapshot.fields.size() == 1);
    REQUIRE(snapshot.fields[0].name == "temperature");
    REQUIRE(snapshot.fields[0].dim0 == 2);
    REQUIRE(snapshot.fields[0].dim1 == 1);
    REQUIRE(snapshot.fields[0].dim2 == 2);
    REQUIRE(snapshot.fields[0].is_3d);
    REQUIRE(snapshot.fields[0].data.size() == 4);
    REQUIRE(snapshot.fields[0].data[3] == Approx(4.0f));
}

TEST_CASE("direct surface writer snapshots active 2D surface buffers", "[core][output_config][field_registry][surface]")
{
    const std::unordered_set<std::string> active_fields = {
        "surface_sensible_heat_flux",
        "u",
        "cape",
    };
    const std::vector<float> surface_data = {1.0f, 2.0f, 3.0f, 4.0f};
    ExportSnapshot           snapshot;
    std::string              error_message;

    const bool ok = write_or_snapshot_direct_surface_fields(
        active_fields,
        [&](std::string_view field_id) -> const std::vector<float>*
        {
            if (field_id == "surface_sensible_heat_flux")
            {
                return &surface_data;
            }
            return nullptr;
        },
        error_message,
        snapshot,
        true,
        2,
        2,
        "unused_step");

    REQUIRE(ok);
    REQUIRE(error_message.empty());
    REQUIRE(snapshot.fields.size() == 1);
    REQUIRE(snapshot.fields[0].name == "surface_sensible_heat_flux");
    REQUIRE(snapshot.fields[0].dim0 == 2);
    REQUIRE(snapshot.fields[0].dim1 == 2);
    REQUIRE(snapshot.fields[0].dim2 == 0);
    REQUIRE_FALSE(snapshot.fields[0].is_3d);
    REQUIRE(snapshot.fields[0].data == surface_data);
}

TEST_CASE("direct surface writer reports missing active surface buffers",
          "[core][output_config][field_registry][surface]")
{
    const std::unordered_set<std::string> active_fields = {
        "pbl_height",
    };
    ExportSnapshot snapshot;
    std::string    error_message;

    const bool ok = write_or_snapshot_direct_surface_fields(
        active_fields,
        [](std::string_view) -> const std::vector<float>* { return nullptr; },
        error_message,
        snapshot,
        true,
        2,
        2,
        "unused_step");

    REQUIRE_FALSE(ok);
    REQUIRE(error_message.find("pbl_height") != std::string::npos);
    REQUIRE(snapshot.fields.empty());
}

TEST_CASE("core volume writer snapshots only active 3D fields", "[core][output_config][field_registry][core]")
{
    Field3D u_field(2, 1, 2, 0.0f);
    Field3D v_field(2, 1, 2, 0.0f);
    u_field(0, 0, 0) = 1.0f;
    u_field(1, 0, 0) = 2.0f;
    u_field(0, 0, 1) = 3.0f;
    u_field(1, 0, 1) = 4.0f;

    const std::vector<CoreVolumeFieldBinding<const Field3D*>> bindings = {
        {"u", &u_field},
        {"v", &v_field},
    };
    const std::unordered_set<std::string> active_fields = {"u"};
    ExportSnapshot                        snapshot;

    write_or_snapshot_core_volume_fields(
        bindings, active_fields, snapshot, true, OutputFormat::npy_3d, 2, 1, 2, "unused_step");

    REQUIRE(snapshot.fields.size() == 1);
    REQUIRE(snapshot.fields[0].name == "u");
    REQUIRE(snapshot.fields[0].dim0 == 2);
    REQUIRE(snapshot.fields[0].dim1 == 1);
    REQUIRE(snapshot.fields[0].dim2 == 2);
    REQUIRE(snapshot.fields[0].is_3d);
    REQUIRE(snapshot.fields[0].data.size() == 4);
    REQUIRE(snapshot.fields[0].data[0] == Approx(1.0f));
    REQUIRE(snapshot.fields[0].data[3] == Approx(4.0f));
}

TEST_CASE("planar tranche writer assembles active column and surface products into 2D maps",
          "[core][output_config][field_registry][tranche]")
{
    const std::vector<RegisteredDiagnosticProduct> products = {
        {
            "cape",
            [](int theta_index, std::vector<float>& out)
            {
                out = {
                    static_cast<float>(theta_index + 1),
                    static_cast<float>(theta_index + 10),
                };
                return true;
            },
        },
        {
            "u10",
            [](int theta_index, std::vector<float>& out)
            {
                out = {
                    static_cast<float>(theta_index + 20),
                    static_cast<float>(theta_index + 30),
                };
                return true;
            },
        },
    };
    const std::unordered_set<std::string> active_fields = {"cape", "u10"};
    ExportSnapshot                        snapshot;
    std::string                           error_message;
    int                                   validation_count = 0;

    const bool ok = write_or_snapshot_column_tranche_fields(
        products,
        active_fields,
        [&](const char* field_id, std::vector<float>& values, int theta_index)
        {
            REQUIRE((std::string(field_id) == "cape" || std::string(field_id) == "u10"));
            REQUIRE(theta_index == 0);
            REQUIRE(values.size() == 6);
            ++validation_count;
            return true;
        },
        error_message,
        snapshot,
        true,
        2,
        3,
        7,
        "unused_step");

    REQUIRE(ok);
    REQUIRE(error_message.empty());
    REQUIRE(validation_count == 2);
    REQUIRE(snapshot.fields.size() == 2);
    REQUIRE(snapshot.fields[0].name == "cape");
    REQUIRE(snapshot.fields[0].dim0 == 2);
    REQUIRE(snapshot.fields[0].dim1 == 3);
    REQUIRE_FALSE(snapshot.fields[0].is_3d);
    REQUIRE(snapshot.fields[0].data[0] == Approx(1.0f));
    REQUIRE(snapshot.fields[0].data[2] == Approx(3.0f));
    REQUIRE(snapshot.fields[0].data[4] == Approx(11.0f));
    REQUIRE(snapshot.fields[1].name == "u10");
    REQUIRE(snapshot.fields[1].data[0] == Approx(20.0f));
    REQUIRE(snapshot.fields[1].data[2] == Approx(22.0f));
    REQUIRE(snapshot.fields[1].data[4] == Approx(31.0f));
}

TEST_CASE("tranche slice writer computes and validates active products",
          "[core][output_config][field_registry][tranche]")
{
    const std::vector<RegisteredDiagnosticProduct> products = {
        {
            "vrot",
            [](int theta_index, std::vector<float>& out)
            {
                out = {
                    static_cast<float>(theta_index),
                    static_cast<float>(theta_index + 1),
                    static_cast<float>(theta_index + 2),
                    static_cast<float>(theta_index + 3),
                };
                return true;
            },
        },
        {
            "bwer",
            [](int, std::vector<float>&)
            {
                FAIL("inactive tranche should not be computed");
                return false;
            },
        },
    };
    const std::unordered_set<std::string> active_fields = {"vrot"};
    std::vector<float>                    tranche_slice;
    std::string                           error_message;
    int                                   validation_count = 0;

    const bool ok = write_tranche_slice_fields(
        products,
        active_fields,
        [&](const char* field_id, std::vector<float>& values, int theta_index)
        {
            REQUIRE(std::string(field_id) == "vrot");
            REQUIRE(theta_index == 2);
            REQUIRE(values.size() == 4);
            ++validation_count;
            return true;
        },
        error_message,
        true,
        2,
        9,
        2,
        2,
        "unused_step",
        tranche_slice);

    REQUIRE(ok);
    REQUIRE(error_message.empty());
    REQUIRE(validation_count == 1);
    REQUIRE(tranche_slice[0] == Approx(2.0f));
    REQUIRE(tranche_slice[3] == Approx(5.0f));
}

TEST_CASE("3D tranche slice writer does not recompute assembled planar products",
          "[core][output_config][field_registry][tranche]")
{
    const std::vector<RegisteredDiagnosticProduct> products = {
        {
            "cape",
            [](int, std::vector<float>&)
            {
                FAIL("3D column tranche should already be assembled");
                return false;
            },
        },
        {
            "u10",
            [&](int, std::vector<float>&)
            {
                FAIL("3D surface tranche should already be assembled");
                return false;
            },
        },
    };
    const std::unordered_set<std::string> active_fields = {"cape", "u10"};
    std::vector<float>                    tranche_slice;
    std::string                           error_message;
    int                                   validation_count = 0;

    const bool ok = write_tranche_slice_fields(
        products,
        active_fields,
        [&](const char* field_id, std::vector<float>&, int)
        {
            ++validation_count;
            return true;
        },
        error_message,
        true,
        1,
        11,
        2,
        2,
        "unused_step",
        tranche_slice);

    REQUIRE(ok);
    REQUIRE(error_message.empty());
    REQUIRE(validation_count == 0);
}

TEST_CASE("legacy tranche slices expand compact planar diagnostics without recomputation",
          "[core][output_config][field_registry][tranche]")
{
    int compute_count = 0;
    const std::vector<RegisteredDiagnosticProduct> products = {
        {
            "cold_pool_boundary",
            [&](int theta_index, std::vector<float>& out)
            {
                REQUIRE(theta_index == 1);
                ++compute_count;
                out = {3.0f, 4.0f};
                return true;
            },
        },
    };
    const std::unordered_set<std::string> active_fields = {
        "cold_pool_boundary"};
    std::vector<float> tranche_slice;
    std::string        error_message;
    int                validation_count = 0;

    const bool ok = write_tranche_slice_fields(
        products,
        active_fields,
        [&](const char* field_id, std::vector<float>& values, int theta_index)
        {
            REQUIRE(std::string(field_id) == "cold_pool_boundary");
            REQUIRE(theta_index == 1);
            REQUIRE(values == std::vector<float>{3.0f, 4.0f,
                                                 3.0f, 4.0f,
                                                 3.0f, 4.0f});
            ++validation_count;
            // Stop before filesystem output; expansion is the behavior under
            // test, not the NPY writer.
            return false;
        },
        error_message,
        false,
        1,
        12,
        2,
        3,
        "unused_step",
        tranche_slice);

    REQUIRE_FALSE(ok);
    REQUIRE(error_message.empty());
    REQUIRE(compute_count == 1);
    REQUIRE(validation_count == 1);
}

TEST_CASE("cold-pool density-potential-temperature deficit uses the dry mixing-ratio basis",
          "[core][output_config][field_registry][derived][cold_pool]")
{
    using tmv::derived::density_potential_temperature;
    using tmv::derived::density_potential_temperature_deficit;

    constexpr float theta = 300.0f;
    constexpr float qv    = 0.01f;
    constexpr double epsilon =
        physical_constants::gas_constant_dry_air_jkgk /
        physical_constants::gas_constant_water_vapor_jkgk;
    const double expected = static_cast<double>(theta) *
        (1.0 + static_cast<double>(qv) / epsilon) /
        (1.0 + static_cast<double>(qv));

    REQUIRE(std::string(tmv::derived::cold_pool_boundary_algorithm_id) ==
            "surface_density_potential_temperature_deficit_v1");
    REQUIRE(density_potential_temperature(
                theta, qv, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f) ==
            Approx(expected));
    REQUIRE(density_potential_temperature_deficit(
                theta, qv, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                theta, qv) == Approx(0.0f));

    const float cooled = density_potential_temperature_deficit(
        theta - 2.0f, qv, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        theta, qv);
    REQUIRE(cooled > 1.9f);
    REQUIRE(cooled < 2.1f);
    REQUIRE(density_potential_temperature_deficit(
                theta + 2.0f, qv, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                theta, qv) == Approx(0.0f));

    const float condensate_in_cloud = density_potential_temperature(
        theta, qv, 0.006f, 0.005f, 0.004f, 0.003f, 0.002f, 0.001f);
    const float condensate_in_rain = density_potential_temperature(
        theta, qv, 0.0f, 0.021f, 0.0f, 0.0f, 0.0f, 0.0f);
    REQUIRE(condensate_in_cloud == Approx(condensate_in_rain));
    REQUIRE(condensate_in_cloud <
            density_potential_temperature(
                theta, qv, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f));

    REQUIRE(std::isnan(density_potential_temperature(
        theta, qv, -1.0e-5f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f)));
    REQUIRE(std::isnan(density_potential_temperature(
        theta, qv, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        std::numeric_limits<float>::quiet_NaN())));
    REQUIRE(std::isnan(density_potential_temperature_deficit(
        theta, qv, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
        theta, -0.01f)));

    Field3D theta_field(2, 2, 2, theta);
    Field3D qv_field(2, 2, 2, qv);
    Field3D qc_field(2, 2, 2, 0.0f);
    Field3D qr_field(2, 2, 2, 0.0f);
    Field3D qi_field(2, 2, 2, 0.0f);
    Field3D qs_field(2, 2, 2, 0.0f);
    Field3D qg_field(2, 2, 2, 0.0f);
    Field3D qh_field(2, 2, 2, 0.0f);
    theta_field(0, 1, 0) = theta - 2.0f;
    qc_field(1, 1, 0) = 0.01f;
    std::vector<float> surface_slice;
    std::string        error_message;
    REQUIRE(tmv::derived::
                fill_density_potential_temperature_deficit_surface_slice(
                    theta_field, qv_field, qc_field, qr_field,
                    qi_field, qs_field, qg_field, qh_field,
                    2, 2, 2, 1, 0, theta, qv,
                    surface_slice, error_message));
    REQUIRE(error_message.empty());
    REQUIRE(surface_slice.size() == 2);
    REQUIRE(surface_slice[0] == Approx(cooled));
    REQUIRE(surface_slice[1] > 0.0f);

    const std::vector<float> terrain_reference_theta{theta, theta + 3.0f};
    const std::vector<float> terrain_reference_qv{qv, qv};
    REQUIRE(tmv::derived::
                fill_density_potential_temperature_deficit_surface_slice(
                    theta_field, qv_field, qc_field, qr_field,
                    qi_field, qs_field, qg_field, qh_field,
                    2, 2, 2, 1, 0,
                    terrain_reference_theta, terrain_reference_qv,
                    surface_slice, error_message));
    REQUIRE(surface_slice[0] == Approx(cooled));
    REQUIRE(surface_slice[1] > 3.0f);
    REQUIRE_FALSE(tmv::derived::
                      fill_density_potential_temperature_deficit_surface_slice(
                          theta_field, qv_field, qc_field, qr_field,
                          qi_field, qs_field, qg_field, qh_field,
                          2, 2, 2, 1, 0,
                          std::vector<float>{theta}, terrain_reference_qv,
                          surface_slice, error_message));
    REQUIRE(error_message.find("terrain reference shape") !=
            std::string::npos);

    qh_field(1, 1, 0) = std::numeric_limits<float>::quiet_NaN();
    REQUIRE_FALSE(tmv::derived::
                      fill_density_potential_temperature_deficit_surface_slice(
                          theta_field, qv_field, qc_field, qr_field,
                          qi_field, qs_field, qg_field, qh_field,
                          2, 2, 2, 1, 0, theta, qv,
                          surface_slice, error_message));
    REQUIRE(error_message.find("i=1, j=1, k=0") != std::string::npos);

    Field3D wrong_shape(1, 2, 2, 0.0f);
    REQUIRE_FALSE(tmv::derived::
                      fill_density_potential_temperature_deficit_surface_slice(
                          theta_field, qv_field, wrong_shape, qr_field,
                          qi_field, qs_field, qg_field, qh_field,
                          2, 2, 2, 1, 0, theta, qv,
                          surface_slice, error_message));
    REQUIRE(error_message.find("shape") != std::string::npos);
}

TEST_CASE("derived field math helpers cover thermodynamic and radar scalars",
          "[core][output_config][field_registry][derived]")
{
    float dbz_min = 0.0f;
    float dbz_max = 0.0f;
    tmv::derived::reflectivity_dbz_bounds(dbz_min, dbz_max);
    REQUIRE(dbz_min == Approx(-30.0f));
    REQUIRE(dbz_max == Approx(120.0f));
    REQUIRE(tmv::derived::linear_z_to_dbz(1000.0f, dbz_min, dbz_max) == Approx(30.0f));
    REQUIRE(tmv::derived::linear_z_to_dbz(0.0f, dbz_min, dbz_max) == Approx(dbz_min));

    // Solid-body rotation u=-Omega*y, v=Omega*x has zero strain and
    // zeta=2*Omega, hence W=-4*Omega^2 (rotation dominated).
    REQUIRE(tmv::derived::okubo_weiss_parameter(0.0f, 0.0f, 2.0f, -2.0f, 4.0f) == Approx(-16.0f));
    // Pure normal strain u=a*x, v=-a*y has zeta=0 and W=4*a^2.
    REQUIRE(tmv::derived::okubo_weiss_parameter(3.0f, -3.0f, 0.0f, 0.0f, 0.0f) == Approx(36.0f));
    REQUIRE(std::isnan(
        tmv::derived::okubo_weiss_parameter(std::numeric_limits<float>::quiet_NaN(), 0.0f, 0.0f, 0.0f, 0.0f)));

    Field3D linear_reflectivity(2, 1, 2, 0.0f);
    linear_reflectivity(0, 0, 0) = 1000.0f;
    linear_reflectivity(1, 0, 0) = 0.0f;
    linear_reflectivity(0, 0, 1) = 1.0e13f;
    Field3D reflectivity_dbz;
    tmv::derived::fill_reflectivity_dbz_field(linear_reflectivity, reflectivity_dbz, 2, 1, 2, dbz_min, dbz_max);
    REQUIRE(reflectivity_dbz(0, 0, 0) == Approx(30.0f));
    REQUIRE(reflectivity_dbz(1, 0, 0) == Approx(dbz_min));
    REQUIRE(reflectivity_dbz(0, 0, 1) == Approx(dbz_max));

    Field3D column(1, 1, 3, 0.0f);
    column(0, 0, 0) = 10.0f;
    column(0, 0, 1) = 20.0f;
    column(0, 0, 2) = 30.0f;
    REQUIRE(tmv::derived::sample_field_at_height(column, 0, 0, 5.0f, 3, 10.0) == Approx(10.0f));
    REQUIRE(tmv::derived::sample_field_at_height(column, 0, 0, 10.0f, 3, 10.0) == Approx(15.0f));
    REQUIRE(tmv::derived::sample_field_at_height(column, 0, 0, 25.0f, 3, 10.0) == Approx(30.0f));

    REQUIRE(tmv::derived::lcl_height_from_temperature_dewpoint(300.0f, 296.0f, 3, 100.0) == Approx(300.0f));
    REQUIRE(tmv::derived::mean_level_component({1.0f, 3.0f, 5.0f}, 15.0f, 3, 10.0) == Approx(2.0f));
    REQUIRE(tmv::derived::mean_level_component({1.0f, 3.0f, 5.0f}, 10.0f, 3, 10.0) == Approx(1.0f));

    // Collocated k=0 and k=nz-1 are boundary-closure storage. Height samples
    // and layer means must not ingest those values even when they are extreme.
    Field3D collocated_column(1, 1, 4, 0.0f);
    collocated_column(0, 0, 0) = -9999.0f;
    collocated_column(0, 0, 1) = 20.0f;
    collocated_column(0, 0, 2) = 30.0f;
    collocated_column(0, 0, 3) = 9999.0f;
    REQUIRE(tmv::derived::sample_field_at_height(collocated_column, 0, 0, 0.0f, 4, 10.0, 1, 3) == Approx(20.0f));
    REQUIRE(tmv::derived::sample_field_at_height(collocated_column, 0, 0, 1000.0f, 4, 10.0, 1, 3) == Approx(30.0f));
    REQUIRE(tmv::derived::mean_level_component({-9999.0f, 20.0f, 30.0f, 9999.0f}, 1000.0f, 4, 10.0, 1, 3) ==
            Approx(25.0f));

    std::vector<float> radial_slice;
    tmv::derived::fill_slice_from_radius(radial_slice, 2, 3, [](int i) { return static_cast<float>(i + 1); });
    REQUIRE(radial_slice.size() == 6);
    REQUIRE(radial_slice[0] == Approx(1.0f));
    REQUIRE(radial_slice[1] == Approx(2.0f));
    REQUIRE(radial_slice[4] == Approx(1.0f));
    REQUIRE(radial_slice[5] == Approx(2.0f));

    const float temperature = tmv::derived::temperature_from_theta_and_pressure(300.0f, 100000.0f);
    REQUIRE(temperature == Approx(300.0f));

    const float qsat = tmv::derived::saturation_mixing_ratio_from_temp_pressure(300.0f, 100000.0f);
    REQUIRE(std::isfinite(qsat));
    REQUIRE(qsat > 0.0f);

    const float dewpoint = tmv::derived::dewpoint_from_temp_qv_pressure(300.0f, 0.014f, 100000.0f);
    REQUIRE(std::isfinite(dewpoint));
    REQUIRE(dewpoint < 300.0f);

    const auto scalars = tmv::derived::compute_thermodynamic_derived_scalars(300.0f, 100000.0f, 0.014f);
    REQUIRE(scalars.temperature_k == Approx(300.0f));
    REQUIRE(scalars.saturation_mixing_ratio_kgkg == Approx(qsat));
    REQUIRE(std::isfinite(scalars.relative_humidity_pct));
    REQUIRE(scalars.relative_humidity_pct > 0.0f);
    REQUIRE(scalars.dewpoint_k == Approx(dewpoint));
}

// -- Tier 2: Per-field ZFP compression tiers ----------------------------------

TEST_CASE("apply_default_compression_tiers assigns all resolved fields", "[core][output_config][zfp_tiers]")
{
    OutputConfig cfg;
    cfg.preset                   = FieldPreset::all;
    cfg.zfp_per_field_tolerances = true;
    cfg.zfp_tight_tolerance      = 1.0e-4;
    cfg.zfp_moderate_tolerance   = 1.0e-3;
    cfg.zfp_loose_tolerance      = 1.0e-2;
    resolve_output_fields(cfg);
    apply_default_compression_tiers(cfg);

    // Every resolved field must have a tolerance entry
    for (const auto& name : cfg.resolved_fields)
    {
        REQUIRE(cfg.zfp_field_tolerances.count(name) == 1);
        REQUIRE(cfg.zfp_field_tolerances.at(name) > 0.0);
    }
}

TEST_CASE("qualified ZFP core history uses unit-aware per-field bounds",
          "[core][output_config][zfp][qualified][science]")
{
    const auto& budgets = qualified_zfp_field_budgets();
    REQUIRE(budgets.size() > 8);
    REQUIRE(budgets.at("p").units == std::string("Pa"));
    REQUIRE(budgets.at("p").max_configured_absolute_error == Approx(5.0e-2));
    REQUIRE(budgets.at("qv").units == std::string("kg/kg"));
    REQUIRE(budgets.at("qv").max_configured_absolute_error == Approx(1.0e-8));

    OutputConfig cfg;
    cfg.format                   = OutputFormat::zfp;
    cfg.preset                   = FieldPreset::minimal;
    cfg.zfp_mode                 = "accuracy";
    cfg.zfp_per_field_tolerances = true;
    cfg.keep_last_frames = 3;
    cfg.minimum_free_space_mib = 4096;
    cfg.maximum_retained_history_files = 256;
    resolve_output_fields(cfg);
    for (const std::string& field : cfg.resolved_fields)
    {
        cfg.zfp_field_tolerances[field] = budgets.at(field).max_configured_absolute_error;
    }
    REQUIRE(validate_qualified_zfp_history(cfg).empty());

    const auto& preset_fields = qualified_zfp_preset_fields();
    REQUIRE(preset_fields.size() == qualified_zfp_preset_ids().size());
    for (const std::string& preset : qualified_zfp_preset_ids())
    {
        REQUIRE(preset_fields.count(preset) == 1);
        REQUIRE_FALSE(preset_fields.at(preset).empty());
        for (const std::string& field : preset_fields.at(preset))
        {
            REQUIRE(budgets.count(field) == 1);
            REQUIRE_FALSE(budgets.at(field).units.empty());
        }
    }
}

TEST_CASE("qualified ZFP policy rejects unbounded modes, prefilters, and fields",
          "[core][output_config][zfp][qualified][guard]")
{
    const auto make_valid = []
    {
        OutputConfig cfg;
        cfg.format                   = OutputFormat::zfp;
        cfg.preset                   = FieldPreset::minimal;
        cfg.zfp_mode                 = "accuracy";
        cfg.zfp_per_field_tolerances = true;
        cfg.keep_last_frames = 3;
        cfg.minimum_free_space_mib = 4096;
        cfg.maximum_retained_history_files = 256;
        resolve_output_fields(cfg);
        const auto& budgets = qualified_zfp_field_budgets();
        for (const std::string& field : cfg.resolved_fields)
            cfg.zfp_field_tolerances[field] = budgets.at(field).max_configured_absolute_error;
        return cfg;
    };

    OutputConfig cfg = make_valid();
    cfg.zfp_mode     = "rate";
    REQUIRE_FALSE(validate_qualified_zfp_history(cfg).empty());

    cfg                      = make_valid();
    cfg.zfp_sparse_threshold = 1.0e-9f;
    REQUIRE_FALSE(validate_qualified_zfp_history(cfg).empty());

    cfg                       = make_valid();
    cfg.zfp_float16_prefilter = true;
    REQUIRE_FALSE(validate_qualified_zfp_history(cfg).empty());

    for (int interval : {0, 1, 2})
    {
        cfg = make_valid();
        cfg.zfp_predictive_delta = true;
        cfg.zfp_keyframe_interval = interval;
        const auto invalid = validate_qualified_zfp_history(cfg);
        REQUIRE(invalid.size() == 1);
        REQUIRE(invalid.front().find("requires a keyframe interval of at least three") !=
                std::string::npos);
    }
    cfg = make_valid();
    cfg.zfp_predictive_delta = true;
    cfg.zfp_keyframe_interval = 3;
    REQUIRE(validate_qualified_zfp_history(cfg).empty());

    cfg                             = make_valid();
    cfg.zfp_field_tolerances["rho"] = 1.01e-6;
    const auto excessive            = validate_qualified_zfp_history(cfg);
    REQUIRE(excessive.size() == 1);
    REQUIRE(excessive.front().find("rho") != std::string::npos);

    cfg = make_valid();
    cfg.resolved_fields.insert("not_a_qualified_field");
    cfg.zfp_field_tolerances["not_a_qualified_field"] = 1.0e-2;
    const auto unknown                           = validate_qualified_zfp_history(cfg);
    REQUIRE(unknown.size() == 1);
    REQUIRE(unknown.front().find("no qualified") != std::string::npos);

    cfg = make_valid();
    cfg.zfp_field_tolerances.erase("qv");
    const auto missing = validate_qualified_zfp_history(cfg);
    REQUIRE(missing.size() == 1);
    REQUIRE(missing.front().find("missing") != std::string::npos);

    cfg = make_valid();
    cfg.keep_last_frames = 0;
    REQUIRE_FALSE(validate_qualified_zfp_history(cfg).empty());

    cfg = make_valid();
    cfg.minimum_free_space_mib = 0;
    REQUIRE_FALSE(validate_qualified_zfp_history(cfg).empty());

    cfg = make_valid();
    cfg.maximum_retained_history_files = 0;
    REQUIRE_FALSE(validate_qualified_zfp_history(cfg).empty());
}

TEST_CASE("qualified ZFP rejects selected radar products without affecting non-radar output",
          "[core][output_config][zfp][qualified][radar][guard]")
{
    const auto make_qualified = [](FieldPreset preset)
    {
        OutputConfig config;
        config.format = OutputFormat::zfp;
        config.preset = preset;
        config.zfp_mode = "accuracy";
        config.zfp_per_field_tolerances = true;
        config.keep_last_frames = 3;
        config.minimum_free_space_mib = 4096;
        config.maximum_retained_history_files = 256;
        resolve_output_fields(config);
        for (const std::string& field : config.resolved_fields)
        {
            config.zfp_field_tolerances[field] =
                qualified_zfp_field_budgets().at(field).max_configured_absolute_error;
        }
        return config;
    };

    for (const FieldPreset preset : {FieldPreset::minimal, FieldPreset::presentation,
                                     FieldPreset::studio_full, FieldPreset::all,
                                     FieldPreset::science_validation})
    {
        const auto config = make_qualified(preset);
        REQUIRE(validate_qualified_zfp_history(config).empty());
    }

    const auto baseline = make_qualified(FieldPreset::minimal);
    for (const std::string& field : {"reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv"})
    {
        auto config = baseline;
        merge_selected_radar_output_fields(config, {field});
        const auto errors = validate_qualified_zfp_history(config);
        REQUIRE(errors.size() == 1);
        CHECK(errors.front().find("field '" + field + "' is not admitted to qualified ZFP") !=
              std::string::npos);

        config.format = OutputFormat::npy_3d;
        REQUIRE(validate_qualified_zfp_history(config).empty());
    }

    for (const FieldPreset preset : {FieldPreset::radar_ready,
                                     FieldPreset::radar_polarimetric_ready})
    {
        auto config = baseline;
        config.preset = preset;
        REQUIRE_FALSE(validate_qualified_zfp_history(config).empty());
    }
}

TEST_CASE("reversible radar ZFP candidate budgets checked bricks without opening admission",
          "[core][output_config][zfp][radar][v4]")
{
    OutputConfig candidate;
    candidate.format = OutputFormat::zfp;
    candidate.preset = FieldPreset::radar_ready;
    candidate.zfp_reversible_radar_v4 = true;
    candidate.zfp_field_scratch_mib = 64;
    candidate.async_io = false;
    candidate.resolved_fields = {"reflectivity_dbz", "radial_velocity", "theta"};
    auto legacy = candidate;
    legacy.zfp_reversible_radar_v4 = false;
    estimate_disk_budget(candidate, 12, 12, 48, 1, 1);
    estimate_disk_budget(legacy, 12, 12, 48, 1, 1);
    REQUIRE(candidate.estimated_bytes_per_export > legacy.estimated_bytes_per_export);
    REQUIRE(candidate.estimated_total_bytes >= 2 * candidate.estimated_bytes_per_export);
    REQUIRE_FALSE(validate_qualified_zfp_history(candidate).empty());

    candidate.zfp_field_scratch_mib = 1;
    REQUIRE_THROWS_AS(estimate_disk_budget(candidate, 64, 64, 32, 1, 1),
                      std::invalid_argument);
    candidate.zfp_field_scratch_mib = 64;
    REQUIRE_THROWS_AS(estimate_disk_budget(
        candidate, std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max(), std::numeric_limits<int>::max(), 1, 1),
        std::invalid_argument);
}

TEST_CASE("qualified ZFP budgets lossless missing-value NPY fields without compression discount",
          "[core][output_config][zfp][missing][budget]")
{
    OutputConfig compressed;
    compressed.format = OutputFormat::zfp;
    compressed.async_io = true;
    compressed.zfp_keyframe_interval = 4;
    compressed.keep_last_frames = 1;
    compressed.maximum_retained_history_files = 256;
    compressed.resolved_fields = {"u"};
    estimate_disk_budget(compressed, 12, 12, 48, 4, 1);

    auto mixed = compressed;
    mixed.resolved_fields.insert("k_index");
    estimate_disk_budget(mixed, 12, 12, 48, 4, 1);
    const std::size_t lossless_bytes = 12U * 12U * sizeof(float) + 4096U;
    REQUIRE(mixed.estimated_bytes_per_export ==
            compressed.estimated_bytes_per_export + lossless_bytes);
    REQUIRE(mixed.estimated_total_bytes ==
            compressed.estimated_total_bytes + 5U * lossless_bytes);
    REQUIRE(mixed.estimated_retained_history_files >
            compressed.estimated_retained_history_files);
    auto synchronous = mixed;
    synchronous.async_io = false;
    estimate_disk_budget(synchronous, 12, 12, 48, 4, 1);
    REQUIRE(synchronous.estimated_bytes_per_export ==
            mixed.estimated_bytes_per_export);
    REQUIRE(synchronous.estimated_retained_history_files ==
            mixed.estimated_retained_history_files);
    auto too_many_files = mixed;
    too_many_files.maximum_retained_history_files = 1;
    REQUIRE_THROWS_AS(estimate_disk_budget(too_many_files, 12, 12, 48, 4, 1),
                      std::invalid_argument);
    REQUIRE_THROWS_AS(estimate_disk_budget(
        mixed, std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max(),
        std::numeric_limits<int>::max(), 4, 1), std::invalid_argument);
}

TEST_CASE("reversible radar ZFP public scope is exact",
          "[core][output_config][zfp][radar][scope]")
{
    nimbus::output::QualifiedRadarZfpV4Context context;
    context.qualified = true;
    context.cpu = true;
    context.fallback_off = true;
    context.cartesian = true;
    context.c_grid = true;
    context.flat = true;
    context.stationary = true;
    context.no_nesting = true;
    context.no_tracking = true;
    context.no_side_stream = true;
    context.no_tiered_cadence = true;
    context.no_live_shm = true;
    context.no_trajectories = true;
    context.deterministic_reductions = true;
    context.reviewed_source_profile = true;
    context.selected_radar_products = {
        "reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv"};
    context.nx = 12;
    context.ny = 12;
    context.nz = 48;
    context.dx_m = 1000.0;
    context.dy_m = 1000.0;
    context.dz_m = 250.0;
    context.dt_large_s = 0.2;
    context.acoustic_substeps = 6;
    context.write_every_s = 4;
    OutputConfig base;
    base.format = OutputFormat::zfp;
    base.preset = FieldPreset::minimal;
    base.zfp_mode = "accuracy";
    base.zfp_per_field_tolerances = true;
    base.zfp_field_scratch_mib = 64;
    base.keep_last_frames = 3;
    base.minimum_free_space_mib = 4096;
    base.maximum_retained_history_files = 256;
    resolve_output_fields(base);
    merge_selected_radar_output_fields(base,
        {"reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv"});
    for (const auto& field : base.resolved_fields)
    {
        const auto budget = qualified_zfp_field_budgets().find(field);
        if (budget != qualified_zfp_field_budgets().end())
            base.zfp_field_tolerances[field] =
                budget->second.max_configured_absolute_error;
    }
    for (const auto* tier : {"zfp_low", "zfp_medium", "zfp_high"})
    {
        auto candidate = base;
        candidate.zfp_requested_format_tier = tier;
        context.format_tier = tier;
        REQUIRE(qualified_radar_zfp_v4_candidate_errors(candidate, context).empty());
        REQUIRE(validate_qualified_zfp_history(candidate, &context).empty());
    }
    const std::array<const char*, 5> products{
        "reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv"};
    for (int mask = 1; mask < 32; ++mask)
    {
        auto candidate = base;
        candidate.zfp_requested_format_tier = "zfp_high";
        for (const auto* field : products)
        {
            candidate.resolved_fields.erase(field);
        }
        for (int index = 0; index < 5; ++index)
        {
            if ((mask & (1 << index)) != 0)
            {
                candidate.resolved_fields.insert(products[index]);
            }
        }
        context.selected_radar_products.clear();
        for (int index = 0; index < 5; ++index)
        {
            if ((mask & (1 << index)) != 0)
            {
                context.selected_radar_products.push_back(products[index]);
            }
        }
        context.format_tier = "zfp_high";
        REQUIRE(qualified_radar_zfp_v4_candidate_errors(candidate, context).empty());
        REQUIRE(validate_qualified_zfp_history(candidate, &context).empty());
    }
    auto ready_preset = base;
    ready_preset.zfp_requested_format_tier = "zfp_high";
    ready_preset.preset = FieldPreset::radar_ready;
    ready_preset.resolved_fields.erase("zdr");
    ready_preset.resolved_fields.erase("kdp");
    ready_preset.resolved_fields.erase("rhohv");
    context.selected_radar_products = {"reflectivity_dbz", "radial_velocity"};
    REQUIRE(qualified_radar_zfp_v4_candidate_errors(ready_preset, context).empty());
    auto polar_preset = base;
    polar_preset.zfp_requested_format_tier = "zfp_high";
    polar_preset.preset = FieldPreset::radar_polarimetric_ready;
    polar_preset.resolved_fields.erase("reflectivity_dbz");
    polar_preset.resolved_fields.erase("radial_velocity");
    polar_preset.resolved_fields.insert("qc");
    polar_preset.zfp_field_tolerances["qc"] =
        qualified_zfp_field_budgets().at("qc").max_configured_absolute_error;
    context.selected_radar_products = {"zdr", "kdp", "rhohv"};
    REQUIRE(qualified_radar_zfp_v4_candidate_errors(polar_preset, context).empty());
    base.zfp_requested_format_tier = "zfp_high";
    context.format_tier = "zfp_high";
    context.selected_radar_products = {
        "rhohv", "kdp", "zdr", "radial_velocity", "reflectivity_dbz"};
    REQUIRE(qualified_radar_zfp_v4_candidate_errors(base, context).empty());
    context.selected_radar_products.push_back("rhohv");
    REQUIRE_FALSE(qualified_radar_zfp_v4_candidate_errors(base, context).empty());
    context.selected_radar_products = {
        "reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv"};
    auto invalid = base;
    context.cpu = false;
    REQUIRE_FALSE(qualified_radar_zfp_v4_candidate_errors(invalid, context).empty());
    context.cpu = true;
    context.write_every_s = 301;
    REQUIRE_FALSE(qualified_radar_zfp_v4_candidate_errors(invalid, context).empty());
    context.write_every_s = 4;
    invalid.preset = FieldPreset::presentation;
    REQUIRE_FALSE(qualified_radar_zfp_v4_candidate_errors(invalid, context).empty());
    invalid = base;
    invalid.resolved_fields.insert("potential_vorticity");
    REQUIRE_FALSE(qualified_radar_zfp_v4_candidate_errors(invalid, context).empty());
    invalid = base;
    context.selected_radar_products = {"reflectivity_dbz"};
    REQUIRE_FALSE(qualified_radar_zfp_v4_candidate_errors(invalid, context).empty());
    context.selected_radar_products = {
        "reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv"};
    invalid = base;
    invalid.zfp_field_tolerances["kdp"] = 1.0e-3;
    REQUIRE_FALSE(qualified_radar_zfp_v4_candidate_errors(invalid, context).empty());
    invalid = base;
    invalid.zfp_field_tolerances["potential_vorticity"] = 1.0e-8;
    REQUIRE_FALSE(qualified_radar_zfp_v4_candidate_errors(invalid, context).empty());
    invalid = base;
    invalid.zfp_field_tolerances.erase("qv");
    REQUIRE_FALSE(qualified_radar_zfp_v4_candidate_errors(invalid, context).empty());
    invalid = base;
    invalid.zfp_field_tolerances["qv"] = 1.0;
    REQUIRE_FALSE(qualified_radar_zfp_v4_candidate_errors(invalid, context).empty());
    invalid = base;
    invalid.zfp_sparse_threshold = 1.0e-8f;
    REQUIRE_FALSE(qualified_radar_zfp_v4_candidate_errors(invalid, context).empty());
    invalid = base;
    invalid.zfp_field_scratch_mib = 0;
    REQUIRE_FALSE(qualified_radar_zfp_v4_candidate_errors(invalid, context).empty());
    invalid = base;
    invalid.zfp_field_workers = 17;
    REQUIRE_FALSE(qualified_radar_zfp_v4_candidate_errors(invalid, context).empty());

    nimbus::output::QualifiedRadarZfpV4CliOverrides cli;
    REQUIRE(nimbus::output::qualified_radar_zfp_v4_cli_scope_error(cli).empty());
    cli.dt_large_set = true;
    cli.dt_large_s = 0.1;
    REQUIRE_FALSE(nimbus::output::qualified_radar_zfp_v4_cli_scope_error(cli).empty());
    cli.dt_large_s = 0.2;
    cli.acoustic_substeps_set = true;
    cli.acoustic_substeps = 5;
    REQUIRE_FALSE(nimbus::output::qualified_radar_zfp_v4_cli_scope_error(cli).empty());
    cli.acoustic_substeps = 6;
    cli.write_every_set = true;
    cli.write_every_s = 301;
    REQUIRE_FALSE(nimbus::output::qualified_radar_zfp_v4_cli_scope_error(cli).empty());
    cli.write_every_s = 4;
    cli.live_shm_set = true;
    REQUIRE_FALSE(nimbus::output::qualified_radar_zfp_v4_cli_scope_error(cli).empty());
    cli.live_shm_set = false;
    cli.trajectory_set = true;
    REQUIRE_FALSE(nimbus::output::qualified_radar_zfp_v4_cli_scope_error(cli).empty());
    cli.trajectory_set = false;
    cli.guard_scope_weakened = true;
    REQUIRE_FALSE(nimbus::output::qualified_radar_zfp_v4_cli_scope_error(cli).empty());
    cli.guard_scope_weakened = false;
    cli.threads_set = true;
    cli.threads = 17;
    REQUIRE_FALSE(nimbus::output::qualified_radar_zfp_v4_cli_scope_error(cli).empty());
    cli.threads = 1;
    REQUIRE(nimbus::output::qualified_radar_zfp_v4_cli_scope_error(cli).empty());
}

TEST_CASE("apply_default_compression_tiers assigns correct tiers", "[core][output_config][zfp_tiers]")
{
    OutputConfig cfg;
    cfg.preset                   = FieldPreset::all;
    cfg.zfp_per_field_tolerances = true;
    cfg.zfp_tight_tolerance      = 1.0e-4;
    cfg.zfp_moderate_tolerance   = 1.0e-3;
    cfg.zfp_loose_tolerance      = 1.0e-2;
    resolve_output_fields(cfg);
    apply_default_compression_tiers(cfg);

    // Core dynamics → tight
    REQUIRE(cfg.zfp_field_tolerances.at("u") == Approx(1.0e-4));
    REQUIRE(cfg.zfp_field_tolerances.at("w") == Approx(1.0e-4));
    REQUIRE(cfg.zfp_field_tolerances.at("rho") == Approx(1.0e-4));
    REQUIRE(cfg.zfp_field_tolerances.at("vorticity_z") == Approx(1.0e-4));

    // Moisture / thermo → moderate
    REQUIRE(cfg.zfp_field_tolerances.at("qv") == Approx(1.0e-3));
    REQUIRE(cfg.zfp_field_tolerances.at("temperature") == Approx(1.0e-3));
    REQUIRE(cfg.zfp_field_tolerances.at("relative_humidity") == Approx(1.0e-3));

    // Diagnostics → loose
    REQUIRE(cfg.zfp_field_tolerances.at("cape") == Approx(1.0e-2));

    OutputConfig radar;
    radar.preset = FieldPreset::radar_ready;
    radar.zfp_per_field_tolerances = true;
    radar.zfp_loose_tolerance = 1.0e-2;
    resolve_output_fields(radar);
    apply_default_compression_tiers(radar);
    REQUIRE(radar.zfp_field_tolerances.at("reflectivity_dbz") == Approx(1.0e-2));
}

TEST_CASE("apply_default_compression_tiers is no-op when disabled", "[core][output_config][zfp_tiers]")
{
    OutputConfig cfg;
    cfg.preset                   = FieldPreset::minimal;
    cfg.zfp_per_field_tolerances = false;
    resolve_output_fields(cfg);
    apply_default_compression_tiers(cfg);

    REQUIRE(cfg.zfp_field_tolerances.empty());
}

TEST_CASE("per-field YAML overrides take precedence over tier defaults", "[core][output_config][zfp_tiers]")
{
    OutputConfig cfg;
    cfg.preset                   = FieldPreset::minimal;
    cfg.zfp_per_field_tolerances = true;
    cfg.zfp_tight_tolerance      = 1.0e-4;
    // Pre-populate an explicit override for "u" (normally tight = 1e-4)
    cfg.zfp_field_tolerances["u"] = 5.0e-6;
    resolve_output_fields(cfg);
    apply_default_compression_tiers(cfg);

    // Override preserved — not overwritten by tight tier
    REQUIRE(cfg.zfp_field_tolerances.at("u") == Approx(5.0e-6));
}

TEST_CASE("parse_output_config reads per-field tolerance settings", "[core][output_config][zfp_tiers]")
{
    std::unordered_map<std::string, std::string> kv;
    kv["output.format"]                    = "zfp";
    kv["output.zfp_per_field_tolerances"]  = "true";
    kv["output.zfp_tight_tolerance"]       = "2.0e-5";
    kv["output.zfp_moderate_tolerance"]    = "5.0e-4";
    kv["output.zfp_loose_tolerance"]       = "5.0e-3";
    kv["output.zfp_field_tolerance.theta"] = "1.0e-6";

#ifdef HAVE_ZFP
    auto cfg = parse_output_config(kv);
    REQUIRE(cfg.zfp_per_field_tolerances == true);
    REQUIRE(cfg.zfp_tight_tolerance == Approx(2.0e-5));
    REQUIRE(cfg.zfp_moderate_tolerance == Approx(5.0e-4));
    REQUIRE(cfg.zfp_loose_tolerance == Approx(5.0e-3));
    REQUIRE(cfg.zfp_field_tolerances.count("theta") == 1);
    REQUIRE(cfg.zfp_field_tolerances.at("theta") == Approx(1.0e-6));
#else
    REQUIRE_THROWS_AS(parse_output_config(kv), std::runtime_error);
#endif
}

// -- Tier 2b: Tiered write cadence -------------------------------------------

TEST_CASE("get_field_write_cadence_tier returns correct tiers", "[core][output_config][cadence]")
{
    // Core dynamics → fast
    REQUIRE(get_field_write_cadence_tier("u") == WriteCadenceTier::fast);
    REQUIRE(get_field_write_cadence_tier("w") == WriteCadenceTier::fast);
    REQUIRE(get_field_write_cadence_tier("rho") == WriteCadenceTier::fast);
    REQUIRE(get_field_write_cadence_tier("vorticity_z") == WriteCadenceTier::fast);
    REQUIRE(get_field_write_cadence_tier("p_prime") == WriteCadenceTier::fast);

    // Moisture / thermo → medium
    REQUIRE(get_field_write_cadence_tier("qv") == WriteCadenceTier::medium);
    REQUIRE(get_field_write_cadence_tier("temperature") == WriteCadenceTier::medium);
    REQUIRE(get_field_write_cadence_tier("relative_humidity") == WriteCadenceTier::medium);
    REQUIRE(get_field_write_cadence_tier("tracer") == WriteCadenceTier::medium);

    // Diagnostics / viz → slow
    REQUIRE(get_field_write_cadence_tier("reflectivity_dbz") == WriteCadenceTier::slow);
    REQUIRE(get_field_write_cadence_tier("composite_reflectivity") == WriteCadenceTier::slow);
    REQUIRE(get_field_write_cadence_tier("cape") == WriteCadenceTier::slow);
    REQUIRE(get_field_write_cadence_tier("u10") == WriteCadenceTier::slow);
}

TEST_CASE("parse_output_config reads tiered cadence settings", "[core][output_config][cadence]")
{
    std::unordered_map<std::string, std::string> kv;
    kv["output.tiered_write_cadence"]   = "true";
    kv["output.write_cadence_medium_s"] = "20";
    kv["output.write_cadence_slow_s"]   = "45";

    auto cfg = parse_output_config(kv);
    REQUIRE(cfg.tiered_write_cadence == true);
    REQUIRE(cfg.write_cadence_medium_s == 20);
    REQUIRE(cfg.write_cadence_slow_s == 45);
}

TEST_CASE("tiered cadence defaults are sane", "[core][output_config][cadence]")
{
    OutputConfig cfg;
    REQUIRE(cfg.tiered_write_cadence == false);
    REQUIRE(cfg.write_cadence_medium_s == 30);
    REQUIRE(cfg.write_cadence_slow_s == 60);
}

TEST_CASE("parse_output_config reads retention and free-space controls",
          "[core][output_config][retention]")
{
    const std::unordered_map<std::string, std::string> kv = {
        {"output.keep_last_frames", "8"},
        {"output.minimum_free_space_mib", "1536"},
        {"output.maximum_retained_history_files", "512"},
    };
    const auto cfg = parse_output_config(kv);
    REQUIRE(cfg.keep_last_frames == 8);
    REQUIRE(cfg.minimum_free_space_mib == 1536);
    REQUIRE(cfg.maximum_retained_history_files == 512);

    REQUIRE_THROWS_WITH(
        parse_output_config({{"output.keep_last_frames", "-1"}}),
        Catch::Matchers::Contains("non-negative integer"));
}

TEST_CASE("retained history file estimate is bounded and fail closed",
          "[core][output_config][retention][files]")
{
    OutputConfig cfg;
    cfg.format = OutputFormat::zfp;
    cfg.preset = FieldPreset::minimal;
    cfg.keep_last_frames = 3;
    cfg.zfp_keyframe_interval = 4;
    cfg.maximum_retained_history_files = 256;
    resolve_output_fields(cfg);
    estimate_disk_budget(cfg, 16, 16, 12, 3600, 10);
    REQUIRE(cfg.estimated_retained_history_files == 129);

    cfg.maximum_retained_history_files = 128;
    REQUIRE_THROWS_WITH(
        estimate_disk_budget(cfg, 16, 16, 12, 3600, 10),
        Catch::Matchers::Contains("exceeds output.maximum_retained_history_files"));
}

TEST_CASE("disk budget includes durable hierarchy and reduced payload files",
          "[core][output_config][retention][lod]")
{
    OutputConfig base;
    base.format = OutputFormat::npy_3d;
    base.preset = FieldPreset::minimal;
    base.keep_last_frames = 3;
    resolve_output_fields(base);
    base.async_io = false;
    estimate_disk_budget(base, 130, 70, 35, 20, 10);

    OutputConfig multilevel = base;
    multilevel.async_io = true;
    estimate_disk_budget(multilevel, 130, 70, 35, 20, 10);
    const std::size_t lod_cells = static_cast<std::size_t>(65 * 35 * 18 + 33 * 18 * 9);
    REQUIRE(multilevel.estimated_bytes_per_export - base.estimated_bytes_per_export ==
            multilevel.resolved_fields.size() * lod_cells * sizeof(float) +
                lod_cells * 4 * sizeof(double));
    REQUIRE(multilevel.estimated_retained_history_files -
                base.estimated_retained_history_files ==
            3 * (multilevel.resolved_fields.size() * 3 + 8));
}

TEST_CASE("cadence tiers are consistent with compression tiers", "[core][output_config][cadence]")
{
    // Fields in tight compression tier should be fast cadence,
    // moderate → medium, loose → slow. Verify a sample.
    OutputConfig cfg;
    cfg.preset                   = FieldPreset::all;
    cfg.zfp_per_field_tolerances = true;
    cfg.zfp_tight_tolerance      = 1.0e-4;
    cfg.zfp_moderate_tolerance   = 1.0e-3;
    cfg.zfp_loose_tolerance      = 1.0e-2;
    resolve_output_fields(cfg);
    apply_default_compression_tiers(cfg);

    for (const auto& name : cfg.resolved_fields)
    {
        WriteCadenceTier cadence   = get_field_write_cadence_tier(name);
        double           tolerance = cfg.zfp_field_tolerances.at(name);

        if (cadence == WriteCadenceTier::fast)
        {
            REQUIRE(tolerance == Approx(cfg.zfp_tight_tolerance));
        }
        else if (cadence == WriteCadenceTier::medium)
        {
            REQUIRE(tolerance == Approx(cfg.zfp_moderate_tolerance));
        }
        else
        {
            REQUIRE(tolerance == Approx(cfg.zfp_loose_tolerance));
        }
    }

}

// -- Tier 3: Advanced compression features ------------------------------------

TEST_CASE("parse_output_config reads Tier 3 settings", "[core][output_config][tier3]")
{
    std::unordered_map<std::string, std::string> kv;
    kv["output.zfp_predictive_delta"]  = "true";
    kv["output.zfp_sparse_threshold"]  = "1.0e-10";
    kv["output.zfp_float16_prefilter"] = "true";

    auto cfg = parse_output_config(kv);
    REQUIRE(cfg.zfp_predictive_delta == true);
    REQUIRE(cfg.zfp_sparse_threshold == Approx(1.0e-10f));
    REQUIRE(cfg.zfp_float16_prefilter == true);
}

TEST_CASE("output configuration rejects invalid values instead of retaining defaults",
          "[core][output_config][config-truth]")
{
    std::unordered_map<std::string, std::string> kv;

    SECTION("invalid boolean")
    {
        kv["output.async_io"] = "sometimes";
        REQUIRE_THROWS_AS(parse_output_config(kv), std::invalid_argument);
    }

    SECTION("invalid positive number")
    {
        kv["output.zfp_tolerance"] = "-1";
        REQUIRE_THROWS_AS(parse_output_config(kv), std::invalid_argument);
    }

    SECTION("partially parsed integer")
    {
        kv["output.zfp_keyframe_interval"] = "10frames";
        REQUIRE_THROWS_AS(parse_output_config(kv), std::invalid_argument);
    }

    SECTION("unknown per-field override")
    {
        kv["output.zfp_field_tolerance.not_a_field"] = "1e-4";
        REQUIRE_THROWS_AS(parse_output_config(kv), std::invalid_argument);
    }
}

TEST_CASE("Tier 3 defaults are disabled", "[core][output_config][tier3]")
{
    OutputConfig cfg;
    REQUIRE(cfg.zfp_predictive_delta == false);
    REQUIRE(cfg.zfp_sparse_threshold == 0.0f);
    REQUIRE(cfg.zfp_float16_prefilter == false);
}

TEST_CASE("is_sparse_eligible identifies hydrometeors", "[core][output_config][tier3]")
{
    // Sparse fields
    REQUIRE(is_sparse_eligible("qi"));
    REQUIRE(is_sparse_eligible("qs"));
    REQUIRE(is_sparse_eligible("qh"));
    REQUIRE(is_sparse_eligible("qg"));
    REQUIRE(is_sparse_eligible("qc"));
    REQUIRE(is_sparse_eligible("qr"));
    REQUIRE(is_sparse_eligible("radar"));
    REQUIRE(is_sparse_eligible("reflectivity_dbz"));

    // Non-sparse fields
    REQUIRE_FALSE(is_sparse_eligible("u"));
    REQUIRE_FALSE(is_sparse_eligible("theta"));
    REQUIRE_FALSE(is_sparse_eligible("rho"));
    REQUIRE_FALSE(is_sparse_eligible("temperature"));
}

TEST_CASE("is_float16_eligible identifies bounded diagnostics", "[core][output_config][tier3]")
{
    // Float16-eligible fields
    REQUIRE(is_float16_eligible("relative_humidity"));
    REQUIRE(is_float16_eligible("cape"));
    REQUIRE(is_float16_eligible("stp"));
    REQUIRE(is_float16_eligible("u10"));
    REQUIRE(is_float16_eligible("composite_reflectivity"));

    // Not eligible (core dynamics need full precision)
    REQUIRE_FALSE(is_float16_eligible("u"));
    REQUIRE_FALSE(is_float16_eligible("theta"));
    REQUIRE_FALSE(is_float16_eligible("rho"));
    REQUIRE_FALSE(is_float16_eligible("qv"));
}

TEST_CASE("column publication visits shared diagnostics once per slice and owns async maps",
          "[core][output][columns][ownership]")
{
    int cached_theta = -1;
    int refreshes = 0;
    std::vector<RegisteredDiagnosticProduct> products;
    for (const auto& item : std::vector<std::pair<const char*, int>>{{"cape", 0}, {"cin", 1}, {"lcl", 2}})
        products.push_back({item.first, [&, index = item.second](int th, std::vector<float>& out) {
            if (cached_theta != th) { cached_theta = th; ++refreshes; }
            out = {static_cast<float>(index * 100 + th * 10), static_cast<float>(index * 100 + th * 10 + 1)};
            return true;
        }});
    ExportSnapshot snapshot;
    std::string error;
    REQUIRE(write_or_snapshot_column_tranche_fields(products, {"cape", "cin"},
        [](const char*, std::vector<float>&, int) { return true; }, error, snapshot, true, 2, 3, 0, ""));
    REQUIRE(refreshes == 3);
    REQUIRE(snapshot.fields.size() == 2);
    REQUIRE(snapshot.fields[0].data == std::vector<float>{0, 10, 20, 1, 11, 21});
    REQUIRE(snapshot.fields[1].data == std::vector<float>{100, 110, 120, 101, 111, 121});
}

TEST_CASE("storm integrals use signed physical layers and partial cells",
          "[core][output][science][helicity]")
{
    const std::vector<double> height{500, 1500, 3500};
    const auto u = [](int) { return 10.0; };
    const auto v = [](int k) { return k == 0 ? 0.0 : k == 1 ? 10.0 : 30.0; };
    const auto z = [&](int k) { return height[k]; };
    REQUIRE(tmv::diagnostics::integrate_storm_relative_helicity(0, 3, 500, 3500, 0, 0, z, u, v)
        == Approx(-300.0));
    REQUIRE(tmv::diagnostics::integrate_storm_relative_helicity(0, 3, 1000, 2000, 0, 0, z, u, v)
        == Approx(-100.0));
    REQUIRE(tmv::diagnostics::integrate_storm_relative_helicity(0, 3, 1000, 2000, 0, 0, z, u,
        [&](int k) { return -v(k); }) == Approx(100.0));
    const auto result = tmv::diagnostics::integrate_cell_layer(0, 8, 2000.0, 5000.0,
        [](int k) { return 1000.0 * k; }, [](int k) { return (k + 0.5) * 0.01; });
    REQUIRE(result.valid);
    REQUIRE(result.covered_depth_m == 3000.0);
    REQUIRE(result.value == Approx(105.0));
    const auto shifted = tmv::diagnostics::integrate_cell_layer(0, 8, 2600.0, 5600.0,
        [](int k) { return 600.0 + 1000.0 * k; }, [](int k) { return (k + 0.5) * 0.01; });
    REQUIRE(shifted.value == result.value);
    const auto partial = tmv::diagnostics::integrate_cell_layer(0, 2, 750.0, 1250.0,
        [](int k) { return 1000.0 * k; }, [](int) { return 2.0; });
    REQUIRE(partial.value == 1000.0);
}

TEST_CASE("diagnostic exports use physical gradients reference heights and moist buoyancy",
          "[core][output][science][terrain]")
{
    constexpr int nr = 4, ny = 4, nz = 8;
    Field3D pressure(nr, ny, nz, 100000), rho(nr, ny, nz, 1), theta(nr, ny, nz, 300);
    Field3D u(nr, ny, nz, 0), v(nr, ny, nz, 0), w(nr, ny, nz, 0);
    Field3D qv(nr, ny, nz, 0), qc(nr, ny, nz, 0), zero(nr, ny, nz, 0);
    std::vector<float> mean(nr * nz, 300), wind_mean(nz, 0);
    std::vector<double> theta0(nz, 300), rho0(nz, 1), qv0(nz, 0), p0(nz, 100000);
    TerrainMetrics3D terrain;
    terrain.z.resize(nr, ny, nz, 0);
    for (int k = 0; k < nz; ++k)
    {
        theta0[k] = 300 + 0.002 * (k + 0.5) * 1000;
        for (int i = 0; i < nr; ++i)
            for (int j = 0; j < ny; ++j)
            {
                const double z = 1000 * (k + 0.5) + 100 * i;
                terrain.z(i, j, k) = z;
                pressure(i, j, k) = static_cast<float>(100000 - 10 * z);
                theta(i, j, k) = static_cast<float>(300 + 0.002 * z);
                u(i, j, k) = static_cast<float>(0.01 * z);
            }
    }
    DerivedSliceComputationContext context{nr, ny, nz, 1000, 1000, 1000, 1,
        CoordinateSystem::Cartesian, OutputFieldDependency::None, 1, -30, 120,
        mean, wind_mean, wind_mean, rho, pressure, u, v, w, theta, qv,
        qc, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero, zero,
        theta0, &terrain, 0, &rho0, &qv0, &p0};
    DerivedSliceScratch mapped(nr * nz, {"pressure_gradient_force_x", "divergence", "theta_prime"});
    compute_derived_slices_for_theta(mapped, context);
    REQUIRE((*mapped.buffer("pressure_gradient_force_x"))[9] == Approx(0.0).margin(1.0e-10));
    REQUIRE((*mapped.buffer("divergence"))[9] == Approx(0.0).margin(1.0e-10));
    REQUIRE((*mapped.buffer("theta_prime"))[9] == Approx(0.0).margin(4.0e-5));
    DerivedSliceScratch theta_only(nr * nz, {"theta_prime"});
    compute_derived_slices_for_theta(theta_only, context);
    REQUIRE(*theta_only.buffer("theta_prime") == *mapped.buffer("theta_prime"));

    context.terrain_metrics = nullptr;
    theta.fill(300); qc.fill(0.02f);
    DerivedSliceScratch buoyancy(nr * nz, {"buoyancy"});
    compute_derived_slices_for_theta(buoyancy, context);
    REQUIRE((*buoyancy.buffer("buoyancy"))[9] == Approx(-0.1962).epsilon(1.0e-6));
    qc.fill(0); rho.fill(static_cast<float>(300.0 / 305.0)); theta.fill(305);
    compute_derived_slices_for_theta(buoyancy, context);
    REQUIRE((*buoyancy.buffer("buoyancy"))[9] > 0.16f);
    DerivedSliceScratch combined(nr * nz, {"buoyancy", "pressure_gradient_force_x"});
    compute_derived_slices_for_theta(combined, context);
    REQUIRE(*buoyancy.buffer("buoyancy") == *combined.buffer("buoyancy"));

    std::vector<double> flat_z{500, 2000, 5000, 9000, 14000, 20000, 27000, 35000};
    context.z_center_m = &flat_z;
    rho.fill(1); u.fill(0);
    for (int i = 0; i < nr; ++i)
        for (int j = 0; j < ny; ++j)
            for (int k = 0; k < nz; ++k)
            {
                pressure(i, j, k) = static_cast<float>(100000 - flat_z[k]);
                w(i, j, k) = static_cast<float>(0.0001 * flat_z[k]);
            }
    DerivedSliceScratch stretched(nr * nz, {"pressure_gradient_force_z", "divergence"});
    compute_derived_slices_for_theta(stretched, context);
    REQUIRE((*stretched.buffer("pressure_gradient_force_z"))[9] == Approx(1.0));
    REQUIRE((*stretched.buffer("divergence"))[9] == Approx(0.0001).margin(1.0e-10));
    flat_z[2] = flat_z[1];
    REQUIRE_THROWS(compute_derived_slices_for_theta(stretched, context));
}

TEST_CASE("wet bulb potential temperature follows the saturated pressure adiabat",
          "[core][output_config][theta_w][science]")
{
    using tmv::diagnostics::wet_bulb_potential_temperature;
    const double es = 611.2 * std::exp(17.67 * (280.0 - 273.15) / (280.0 - 29.65));
    const double vapor = (287.0 / 461.5) * es / (50000.0 - es);
    REQUIRE(wet_bulb_potential_temperature(280.0, 50000.0, vapor)
        == Approx(302.603572).margin(2.0e-5));
    REQUIRE(wet_bulb_potential_temperature(280.0, 100000.0, vapor)
        == Approx(280.0).margin(1.0e-10));
    const double dry = wet_bulb_potential_temperature(300.0, 100000.0, 0.0);
    const double damp = wet_bulb_potential_temperature(300.0, 100000.0, 0.008);
    REQUIRE(std::isfinite(dry));
    REQUIRE(dry < damp);
    REQUIRE(damp < 300.0);
    REQUIRE(std::isnan(wet_bulb_potential_temperature(300.0, 0.0, 0.01)));
    REQUIRE(std::isnan(wet_bulb_potential_temperature(300.0, 100000.0, -0.01)));
}

TEST_CASE("step manifest preserves fractional nest geometry exactly", "[core][output_config][manifest][nesting]")
{
    OutputStepManifestContext context;
    context.step_dir               = "step_000001";
    context.publication_generation = "0123456789abcdef0123456789abcdef";
    context.simulation_time_s      = 1.0 / 3.0;
    context.grid.nr = context.grid.nth = 32;
    context.grid.nz                    = 24;
    context.grid.coordinate_system     = "cartesian";
    context.grid.staggering            = "c_grid";
    context.grid.dr_m = context.grid.dy_m = 1000.0 / 3.0;
    context.grid.dz_m                     = 500.0;
    context.grid.grid_origin_t0_earth_x_m = 8000.0 - context.grid.dr_m;
    context.grid.grid_origin_earth_x_m    = context.grid.grid_origin_t0_earth_x_m;
    const auto manifest                   = build_output_step_manifest_json(context);
    const auto number_after               = [&](const std::string& prefix)
    {
        const auto position = manifest.find(prefix);
        REQUIRE(position != std::string::npos);
        return std::stod(manifest.substr(position + prefix.size()));
    };
    CHECK(number_after("\"simulation_time_s\": ") == context.simulation_time_s);
    CHECK(number_after("\"spacing_m\": {\"x\": ") == context.grid.dr_m);
    CHECK(number_after("\"grid_origin_t0_earth_m\": [") == context.grid.grid_origin_t0_earth_x_m);
    CHECK(number_after("\"grid_origin_earth_m\": [") == context.grid.grid_origin_earth_x_m);
}

TEST_CASE("pressure components have an explicit diagnostic solve dependency", "[core][output_config][dependencies]")
{
    const auto perturbation = output_dependencies_for_fields(std::unordered_set<std::string>{"p_prime"});
    const auto components =
        output_dependencies_for_fields(std::unordered_set<std::string>{"dynamic_pressure", "buoyancy_pressure"});
    CHECK(output_field_has_dependency(perturbation, OutputFieldDependency::PressureDiagnostics));
    CHECK_FALSE(output_field_has_dependency(perturbation, OutputFieldDependency::PressureComponents));
    CHECK(output_field_has_dependency(components, OutputFieldDependency::PressureDiagnostics));
    CHECK(output_field_has_dependency(components, OutputFieldDependency::PressureComponents));

    OutputConfig presentation;
    presentation.preset = FieldPreset::presentation;
    resolve_output_fields(presentation);
    const auto dependencies = output_dependencies_for_fields(presentation.resolved_fields);
    CHECK(output_field_has_dependency(dependencies, OutputFieldDependency::VorticityDiagnostics));
    CHECK(output_field_has_dependency(dependencies, OutputFieldDependency::PressureDiagnostics));
    CHECK_FALSE(output_field_has_dependency(dependencies, OutputFieldDependency::PressureComponents));
}
