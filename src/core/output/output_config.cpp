/**
 * @file output_config.cpp
 * @brief Output pipeline configuration parsing and field resolution.
 */

#include "core/output/output_config.hpp"
#include "core/output/field_brick_hierarchy.hpp"
#include "core/output/field_registry.hpp"
#include "diagnostics/field_contract.hpp"
#include "util/log.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <limits>
#include <sstream>
#include <stdexcept>

#ifdef HAVE_ZFP
#include <zfp.h>
#endif

namespace
{

std::size_t checked_output_add(std::size_t lhs, std::size_t rhs)
{
    if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
        throw std::invalid_argument("output resource estimate overflows size_t");
    return lhs + rhs;
}

std::size_t checked_output_mul(std::size_t lhs, std::size_t rhs)
{
    if (rhs != 0 && lhs > std::numeric_limits<std::size_t>::max() / rhs)
        throw std::invalid_argument("output resource estimate overflows size_t");
    return lhs * rhs;
}

struct ReversibleZfpBound
{
    std::size_t field_bytes = 0;
    std::size_t scratch_bytes = 0;
};

ReversibleZfpBound reversible_zfp_field_bound(int nr, int nth, int nz)
{
#ifndef HAVE_ZFP
    (void)nr; (void)nth; (void)nz;
    throw std::invalid_argument("reversible radar ZFP requires libzfp");
#else
    if (nr <= 0 || nth <= 0 || nz <= 0)
        throw std::invalid_argument("reversible radar ZFP requires positive dimensions");
    constexpr std::array<std::size_t, 3> brick{64, 64, 32};
    const std::array<std::size_t, 3> shape{
        static_cast<std::size_t>(nr), static_cast<std::size_t>(nth),
        static_cast<std::size_t>(nz)};
    std::size_t count = 1;
    for (std::size_t axis = 0; axis < 3; ++axis)
        count = checked_output_mul(count,
            shape[axis] / brick[axis] + (shape[axis] % brick[axis] != 0));
    const std::size_t directory_bytes = checked_output_mul(count, 40);
    zfp_stream* stream = zfp_stream_open(nullptr);
    zfp_field* field = zfp_field_3d(nullptr, zfp_type_float, 1, 1, 1);
    if (!stream || !field)
    {
        if (stream) zfp_stream_close(stream);
        if (field) zfp_field_free(field);
        throw std::invalid_argument("cannot create reversible ZFP resource estimator");
    }
    zfp_stream_set_reversible(stream);
    std::size_t maximum = 0;
    std::size_t decoded = 0;
    for (int mask = 0; mask < 8; ++mask)
    {
        std::array<std::size_t, 3> extent{};
        for (std::size_t axis = 0; axis < 3; ++axis)
        {
            const std::size_t remainder = shape[axis] % brick[axis];
            extent[axis] = (mask & (1 << axis)) && remainder != 0
                ? remainder : std::min(shape[axis], brick[axis]);
        }
        zfp_field_set_size_3d(field, extent[2], extent[1], extent[0]);
        maximum = std::max(maximum, zfp_stream_maximum_size(stream, field));
        decoded = std::max(decoded, checked_output_mul(
            checked_output_mul(extent[0], extent[1]),
            checked_output_mul(extent[2], sizeof(float))));
    }
    zfp_field_free(field);
    zfp_stream_close(stream);
    if (maximum == 0 || maximum > 4U * 1024U * 1024U)
        throw std::invalid_argument("reversible radar ZFP brick exceeds reader ceiling");
    const std::size_t scratch = checked_output_add(
        directory_bytes, checked_output_add(maximum, decoded));
    const std::size_t payload = checked_output_mul(count, maximum);
    return {checked_output_add(44, checked_output_add(payload, directory_bytes)),
            scratch};
#endif
}

struct FieldLodEstimate
{
    std::size_t cells = 0;
    std::size_t files = 0;
};

FieldLodEstimate field_lod_estimate(int nx, int ny, int nz)
{
    FieldLodEstimate estimate;
    std::array<std::size_t, 3> shape{
        static_cast<std::size_t>(std::max(1, nx)),
        static_cast<std::size_t>(std::max(1, ny)),
        static_cast<std::size_t>(std::max(1, nz)),
    };
    while (shape[0] > static_cast<std::size_t>(nimbus::output::k_field_brick_shape[0]) ||
           shape[1] > static_cast<std::size_t>(nimbus::output::k_field_brick_shape[1]) ||
           shape[2] > static_cast<std::size_t>(nimbus::output::k_field_brick_shape[2]))
    {
        for (auto& extent : shape) extent = (extent + 1) / 2;
        const std::size_t level_cells = checked_output_mul(
            checked_output_mul(shape[0], shape[1]), shape[2]);
        estimate.cells = checked_output_add(estimate.cells, level_cells);
        ++estimate.files;
    }
    return estimate;
}

bool is_qualified_zfp_preset(FieldPreset preset)
{
    return preset == FieldPreset::minimal || preset == FieldPreset::presentation ||
           preset == FieldPreset::studio_full || preset == FieldPreset::all ||
           preset == FieldPreset::science_validation;
}

std::string local_to_lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

bool local_parse_bool(const std::string& value)
{
    const std::string v = local_to_lower(value);
    if (v == "true" || v == "1" || v == "yes" || v == "on")
    {
        return true;
    }
    if (v == "false" || v == "0" || v == "no" || v == "off")
    {
        return false;
    }
    throw std::invalid_argument("Invalid boolean value '" + value + "'");
}

bool local_try_parse_double(const std::string& value, double& out)
{
    try
    {
        std::size_t pos = 0;
        out             = std::stod(value, &pos);
        return pos == value.size() && std::isfinite(out);
    }
    catch (...)
    {
        return false;
    }
}

bool local_try_parse_int(const std::string& value, int& out)
{
    try
    {
        std::size_t pos = 0;
        out             = std::stoi(value, &pos);
        return pos == value.size();
    }
    catch (...)
    {
        return false;
    }
}

std::vector<std::string> local_parse_string_list(const std::string& value)
{
    std::string cleaned = value;
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '['), cleaned.end());
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), ']'), cleaned.end());
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '"'), cleaned.end());
    cleaned.erase(std::remove(cleaned.begin(), cleaned.end(), '\''), cleaned.end());

    std::vector<std::string> out;
    std::stringstream        ss(cleaned);
    std::string              item;
    while (std::getline(ss, item, ','))
    {
        item.erase(item.begin(),
                   std::find_if(item.begin(), item.end(), [](unsigned char ch) { return !std::isspace(ch); }));
        item.erase(std::find_if(item.rbegin(), item.rend(), [](unsigned char ch) { return !std::isspace(ch); }).base(),
                   item.end());
        if (!item.empty())
        {
            out.push_back(item);
        }
    }
    return out;
}

} // anonymous namespace

namespace
{

// -- Preset field lists -------------------------------------------------------

const std::vector<std::string> k_minimal_fields = {"u", "v", "w", "theta", "p", "rho", "qv", "qr"};

const std::vector<std::string> k_radar_ready_fields = {
    "u", "v", "w", "theta", "p", "rho", "qv", "qr",
    "reflectivity_dbz", "radial_velocity"};

const std::vector<std::string> k_radar_zdr_exploratory_fields = {
    "u", "v", "w", "theta", "p", "rho", "qv", "qr", "zdr"};

const std::vector<std::string> k_radar_polarimetric_ready_fields = {
    "u", "v", "w", "theta", "p", "rho", "qv", "qc", "qr",
    "zdr", "kdp", "rhohv"};

const std::unordered_set<std::string> k_radar_product_fields = {
    "reflectivity_dbz", "radial_velocity", "zdr", "kdp", "rhohv"};

// The cold-pool product consumes live theta and dry-basis water-species state
// and does not require theta_prime to be published. For uncompressed volume output,
// this preset's float payload is exactly 4 * nr * nth * (9 * nz + 3) bytes;
// at 512 x 512 x 160 that is 1,443 MiB before file headers and metadata.
const std::vector<std::string> k_science_validation_fields = {
    "u",
    "v",
    "w",
    "theta",
    "p",
    "rho",
    "qv",
    "qc",
    "qr",
    "cold_pool_boundary",
    "precip_rate",
    "accumulated_rainfall",
};

const std::vector<std::string> k_moving_reference_v1_diagnostic_fields = {
    "cold_pool_boundary",
    "accumulated_rainfall",
};

const std::vector<std::string> k_presentation_fields = {
    // minimal
    "u",
    "v",
    "w",
    "theta",
    "p",
    "rho",
    "qv",
    "qr",
    // additional diagnostics for talks/papers
    "vorticity_z",
    "vorticity_magnitude",
    "total_condensate",
    "cloud_condensate",
    "cloud_liquid_condensate",
    "cloud_frozen_condensate",
    "temperature",
    "buoyancy",
    // perturbation fields (theta' = theta - theta0(z), p' = p - p0(z)) for
    // cold-pool / mesolow diagnostics in Studio
    "theta_prime",
    "p_prime",
    "composite_reflectivity",
    "stp",
    "updraft_helicity",
    // convective environment + cloud structure: plan-view CAPE/CIN, the
    // condensation/free-convection/equilibrium levels, storm-relative helicity
    // and cloud-layer heights. Small 2D column fields, paired with the radar /
    // rotation products above for the supercell-environment figure in Studio.
    "cape",
    "cin",
    "lcl",
    "lfc",
    "el",
    "srh_0_1km",
    "srh_0_3km",
    "vil",
    "cloud_base_height",
    "cloud_top_height",
    // surface energetics + boundary-layer structure
    "surface_sensible_heat_flux",
    "surface_latent_heat_flux",
    "pbl_height"};

const std::vector<std::string> k_studio_full_fields = {
    // Preserve the complete presentation contract.
    "u",
    "v",
    "w",
    "theta",
    "p",
    "rho",
    "qv",
    "qr",
    "vorticity_z",
    "vorticity_magnitude",
    "total_condensate",
    "cloud_condensate",
    "cloud_liquid_condensate",
    "cloud_frozen_condensate",
    "temperature",
    "buoyancy",
    "theta_prime",
    "p_prime",
    "composite_reflectivity",
    "stp",
    "updraft_helicity",
    "cape",
    "cin",
    "lcl",
    "lfc",
    "el",
    "srh_0_1km",
    "srh_0_3km",
    "vil",
    "cloud_base_height",
    "cloud_top_height",
    "surface_sensible_heat_flux",
    "surface_latent_heat_flux",
    "pbl_height",
    // Additional native products consumed by Studio's current view catalog.
    "stretching_term",
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
    // Terrain-aware dry-density-basis column water paths.
    "precipitable_water",
    "liquid_water_path",
    "cloud_water_path",
    // Absolute-vorticity Ertel PV in SI units.
    "potential_vorticity"};

const std::vector<std::string> k_core_only_fields = {"u",
                                                     "v",
                                                     "w",
                                                     "rho",
                                                     "p",
                                                     "theta",
                                                     "qv",
                                                     "qc",
                                                     "qr",
                                                     "qi",
                                                     "qs",
                                                     "qh",
                                                     "qg",
                                                     "radar",
                                                     "tracer",
                                                     "tke",
                                                     "vorticity_r",
                                                     "vorticity_theta",
                                                     "vorticity_z",
                                                     "stretching_term",
                                                     "tilting_term",
                                                     "baroclinic_term",
                                                     "p_prime",
                                                     "dynamic_pressure",
                                                     "buoyancy_pressure",
                                                     "angular_momentum",
                                                     "angular_momentum_tendency"};

const std::unordered_set<std::string>& all_fields_set()
{
    static const std::unordered_set<std::string> fields(output_field_names().begin(), output_field_names().end());
    return fields;
}

const std::vector<std::string>& available_output_field_names()
{
    static const std::vector<std::string> fields = []
    {
        std::vector<std::string> available;
        for (const OutputFieldDescriptor& field : output_field_registry())
            if (field.kind != OutputFieldKind::RadarSynthetic &&
                !output_field_has_dependency(
                    field.dependencies, OutputFieldDependency::RadarDiagnostic))
                available.push_back(field.field_id);
        return available;
    }();
    return fields;
}

// -- Per-field ZFP compression tiers (Tier 2) --------------------------------
// Tight: core prognostic state + vorticity budget — highest fidelity needed
// for numerical stability and scientific reproducibility.
const std::unordered_set<std::string> k_zfp_tight_fields = {"u",
                                                            "v",
                                                            "w",
                                                            "rho",
                                                            "p",
                                                            "theta",
                                                            "tke",
                                                            "p_prime",
                                                            "dynamic_pressure",
                                                            "buoyancy_pressure",
                                                            "vorticity_r",
                                                            "vorticity_theta",
                                                            "vorticity_z",
                                                            "stretching_term",
                                                            "tilting_term",
                                                            "baroclinic_term"};

// Moderate: thermodynamics, moisture species, and derived thermo fields —
// important for physical fidelity but tolerant of slightly relaxed precision.
const std::unordered_set<std::string> k_zfp_moderate_fields = {"qv",
                                                               "qc",
                                                               "qr",
                                                               "qi",
                                                               "qs",
                                                               "qh",
                                                               "qg",
                                                               "temperature",
                                                               "theta_prime",
                                                               "theta_v",
                                                               "theta_e",
                                                               "theta_w",
                                                               "dewpoint",
                                                               "relative_humidity",
                                                               "saturation_mixing_ratio",
                                                               "total_condensate",
                                                               "cloud_condensate",
                                                               "cloud_liquid_condensate",
                                                               "cloud_frozen_condensate",
                                                               "buoyancy",
                                                               "angular_momentum",
                                                               "angular_momentum_tendency",
                                                               "tracer"};

// Loose: everything else — diagnostics, visualization, radar, surface/column.
// These are derived or post-processed fields where 1e-2 is visually and
// scientifically indistinguishable from full precision.
// (No explicit list needed — any field not in tight or moderate is loose.)

// -- Sparse-eligible fields (Tier 3a) -----------------------------------------
// Hydrometeor and radar fields that are zero over >90% of the domain.
// Thresholding removes floating-point noise so ZFP can compress
// zero blocks efficiently.
const std::unordered_set<std::string> k_sparse_eligible_fields = {"qi",
                                                                  "qs",
                                                                  "qh",
                                                                  "qg",
                                                                  "qc",
                                                                  "qr",
                                                                  "radar",
                                                                  "total_condensate",
                                                                  "cloud_condensate",
                                                                  "cloud_liquid_condensate",
                                                                  "cloud_frozen_condensate",
                                                                  "reflectivity_dbz",
                                                                  "zdr",
                                                                  "kdp"};

// -- Float16-eligible fields (Tier 3c) ----------------------------------------
// Bounded diagnostic and surface/column fields where 10-bit mantissa precision
// is sufficient for export. These fields have well-defined physical ranges
// and don't participate in numerical integration.
const std::unordered_set<std::string> k_float16_eligible_fields = {
    // Bounded diagnostics (0-100% or similar small ranges)
    "relative_humidity",
    "rhohv",
    "tracer",
    // Radar-derived (bounded, visualization-oriented)
    "reflectivity_dbz",
    "zdr",
    "kdp",
    // Surface fields (near-surface values with limited range)
    "u10",
    "v10",
    "t2",
    "td2",
    "skin_temperature",
    "surface_sensible_heat_flux",
    "surface_latent_heat_flux",
    "surface_moisture_flux",
    "surface_pressure_perturbation",
    "cold_pool_boundary",
    "precip_rate",
    "accumulated_rainfall",
    "surface_friction_velocity",
    "surface_stress_u",
    "surface_stress_v",
    "pbl_height",
    // Column diagnostics (integrated/derived indices)
    "composite_reflectivity",
    "column_max_w",
    "column_max_vorticity",
    "vil",
    "precipitable_water",
    "liquid_water_path",
    "cloud_water_path",
    "cloud_top_height",
    "cloud_base_height",
    "lcl",
    "lfc",
    "el",
    "cape",
    "cin",
    "lifted_index",
    "k_index",
    "showalter_index",
    "total_totals",
    "srh_0_1km",
    "srh_0_3km",
    "ehi",
    "scp",
    "stp",
    "updraft_helicity",
    // Radar synthetic
    "ppi_sweep",
    "rhi_sweep",
    "bwer",
    "mesocyclone_diagnostic",
    "vrot"};

} // anonymous namespace

const std::unordered_map<std::string, QualifiedZfpFieldBudget>& qualified_zfp_field_budgets()
{
    static const std::unordered_map<std::string, QualifiedZfpFieldBudget> budgets = []
    {
        std::unordered_map<std::string, QualifiedZfpFieldBudget> values;
        std::unordered_set<std::string> admitted_fields;
        for (const auto& [preset, fields] : qualified_zfp_preset_fields())
        {
            (void)preset;
            admitted_fields.insert(fields.begin(), fields.end());
        }
        for (const std::string& field : admitted_fields)
        {
            const tmv::FieldContract* contract = tmv::find_field_contract(field);
            if (contract == nullptr)
                throw std::logic_error("qualified ZFP field '" + field + "' has no field contract");

            double maximum = 1.0e-2;
            if (field == "p" || field == "p_prime" ||
                field == "dynamic_pressure" || field == "buoyancy_pressure")
                maximum = 5.0e-2;
            else if (field == "rho")
                maximum = 1.0e-6;
            else if (field == "potential_vorticity")
                maximum = 1.0e-8;
            else if (field == "u" || field == "v" || field == "w" ||
                     field == "radial_velocity" || field == "u10" || field == "v10")
                maximum = 1.0e-4;
            else if (field == "theta" || field == "theta_prime" ||
                     field == "temperature" || field == "theta_v" ||
                     field == "theta_e" || field == "theta_w" ||
                     field == "dewpoint" || field == "t2" || field == "td2" ||
                     field == "skin_temperature")
                maximum = 1.0e-3;
            else if (field == "qv" || field == "qc" || field == "qr" ||
                     field == "qi" || field == "qs" || field == "qh" ||
                     field == "qg" || field == "saturation_mixing_ratio" ||
                     field == "total_condensate" || field == "cloud_condensate" ||
                     field == "cloud_liquid_condensate" ||
                     field == "cloud_frozen_condensate")
                maximum = 1.0e-8;
            else if (field.find("vorticity") != std::string::npos ||
                     field == "divergence")
                maximum = 1.0e-6;
            else if (field.find("_term") != std::string::npos)
                maximum = 1.0e-7;

            values.emplace(field, QualifiedZfpFieldBudget{maximum, contract->units});
        }
        return values;
    }();
    return budgets;
}

const std::unordered_map<std::string, std::vector<std::string>>& qualified_zfp_preset_fields()
{
    static const std::unordered_map<std::string, std::vector<std::string>> values = []
    {
        std::unordered_map<std::string, std::vector<std::string>> result = {
            {"minimal", get_preset_fields(FieldPreset::minimal)},
            {"presentation", get_preset_fields(FieldPreset::presentation)},
            {"studio_full", get_preset_fields(FieldPreset::studio_full)},
            {"dynamics_full", get_preset_fields(FieldPreset::all)},
            {"science_validation", get_preset_fields(FieldPreset::science_validation)},
        };
        for (auto& [preset, fields] : result)
        {
            (void)preset;
            std::sort(fields.begin(), fields.end());
            fields.erase(std::unique(fields.begin(), fields.end()), fields.end());
        }
        return result;
    }();
    return values;
}

const char* qualified_zfp_mode()
{
    return "accuracy";
}

bool qualified_zfp_requires_per_field_tolerances()
{
    return true;
}

double effective_zfp_accuracy_tolerance(const OutputConfig& config,
                                        const std::string& field_name)
{
    const auto tolerance = config.zfp_field_tolerances.find(field_name);
    double effective = tolerance == config.zfp_field_tolerances.end()
        ? config.zfp_tolerance
        : tolerance->second;

    // Ertel PV is stored in SI units, where a meteorologically meaningful
    // signal is commonly O(1e-6). A generic 1e-3/1e-4 visualization tolerance
    // would quantize that signal to zero while still satisfying the codec's
    // byte-level contract. Cap its absolute error at 0.01 PVU (1e-8 SI).
    if (field_name == "potential_vorticity")
        effective = std::min(effective, 1.0e-8);
    return effective;
}

double qualified_zfp_max_sparse_threshold()
{
    return 0.0;
}

bool qualified_zfp_allows_float16_prefilter()
{
    return false;
}

const std::vector<std::string>& qualified_zfp_preset_ids()
{
    static const std::vector<std::string> values = {
        "minimal", "presentation", "studio_full", "dynamics_full", "science_validation"};
    return values;
}

const std::vector<std::string>& qualified_zfp_field_execution_policies()
{
    static const std::vector<std::string> values = {
        "serial", "field_parallel"};
    return values;
}

int qualified_zfp_max_field_workers() { return 16; }
int qualified_zfp_min_field_scratch_mib() { return 1; }
int qualified_zfp_max_field_scratch_mib() { return 4096; }

std::vector<std::string> qualified_radar_zfp_v4_candidate_errors(
    const OutputConfig& config,
    const nimbus::output::QualifiedRadarZfpV4Context& context)
{
    std::vector<std::string> errors;
    if (config.format != OutputFormat::zfp ||
        !nimbus::output::matches_qualified_radar_zfp_v4_context(context) ||
        config.zfp_requested_format_tier != context.format_tier)
    {
        errors.push_back("qualified radar ZFP v4 requires its exact flat CPU canonical scope");
    }
    std::unordered_set<std::string> selected;
    for (const auto& field : config.resolved_fields)
    {
        if (nimbus::output::is_qualified_radar_zfp_v4_field(field))
        {
            selected.insert(field);
        }
    }
    if (selected.empty())
    {
        errors.push_back("qualified radar ZFP v4 requires a nonempty selected product set");
    }
    const std::unordered_set<std::string> source_selection(
        context.selected_radar_products.begin(),
        context.selected_radar_products.end());
    if (source_selection.size() != context.selected_radar_products.size() ||
        source_selection != selected)
    {
        errors.push_back("qualified radar ZFP v4 published fields differ from the selected radar products");
    }
    std::unordered_set<std::string> expected;
    for (const auto field : nimbus::output::kQualifiedRadarZfpV4CoreFields)
    {
        expected.insert(std::string(field));
    }
    expected.insert(selected.begin(), selected.end());
    if (config.preset == FieldPreset::radar_ready)
    {
        if (selected != std::unordered_set<std::string>{"reflectivity_dbz", "radial_velocity"})
        {
            errors.push_back("radar_ready ZFP v4 requires its exact two-product selection");
        }
        if (context.selected_radar_products !=
            std::vector<std::string>{"reflectivity_dbz", "radial_velocity"})
        {
            errors.push_back("radar_ready ZFP v4 requires its canonical product order");
        }
    }
    else if (config.preset == FieldPreset::radar_polarimetric_ready)
    {
        expected.insert("qc");
        if (selected != std::unordered_set<std::string>{"zdr", "kdp", "rhohv"})
        {
            errors.push_back("radar_polarimetric_ready ZFP v4 requires its exact three-product selection");
        }
        if (context.selected_radar_products !=
            std::vector<std::string>{"zdr", "kdp", "rhohv"})
        {
            errors.push_back("radar_polarimetric_ready ZFP v4 requires its canonical product order");
        }
    }
    else if (config.preset != FieldPreset::minimal)
    {
        errors.push_back("qualified radar ZFP v4 requires minimal or a retained ready radar preset");
    }
    if (config.resolved_fields != expected)
    {
        errors.push_back("qualified radar ZFP v4 field set differs from core plus selected products");
    }
    if (config.zfp_mode != "accuracy" || !config.zfp_per_field_tolerances ||
        config.zfp_sparse_threshold != 0.0f || config.zfp_float16_prefilter ||
        (config.zfp_predictive_delta && config.zfp_keyframe_interval < 3))
    {
        errors.push_back("qualified radar ZFP v4 requires unfiltered accuracy-mode non-radar history");
    }
    if (std::find(qualified_zfp_field_execution_policies().begin(),
                  qualified_zfp_field_execution_policies().end(),
                  config.zfp_field_execution) ==
            qualified_zfp_field_execution_policies().end() ||
        config.zfp_field_workers < 0 ||
        config.zfp_field_workers > qualified_zfp_max_field_workers() ||
        config.zfp_field_scratch_mib < qualified_zfp_min_field_scratch_mib() ||
        config.zfp_field_scratch_mib > qualified_zfp_max_field_scratch_mib() ||
        config.keep_last_frames <= 0 || config.minimum_free_space_mib <= 0 ||
        config.maximum_retained_history_files <= 0)
    {
        errors.push_back("qualified radar ZFP v4 worker, scratch, retention or free-space policy is invalid");
    }
    for (const auto& [field, value] : config.zfp_field_tolerances)
    {
        if (nimbus::output::is_qualified_radar_zfp_v4_field(field))
        {
            (void)value;
            errors.push_back("reversible radar field '" + field +
                             "' cannot have a lossy ZFP tolerance");
        }
        else if (config.resolved_fields.count(field) == 0)
        {
            errors.push_back("unselected field '" + field +
                             "' cannot have a qualified ZFP tolerance");
        }
    }
    const auto& budgets = qualified_zfp_field_budgets();
    for (const auto& field : config.resolved_fields)
    {
        if (nimbus::output::is_qualified_radar_zfp_v4_field(field))
        {
            continue;
        }
        const auto budget = budgets.find(field);
        const auto tolerance = config.zfp_field_tolerances.find(field);
        if (budget == budgets.end() || tolerance == config.zfp_field_tolerances.end() ||
            !std::isfinite(tolerance->second) || tolerance->second <= 0.0 ||
            tolerance->second > budget->second.max_configured_absolute_error)
        {
            errors.push_back("non-radar field '" + field +
                             "' lacks an exact qualified ZFP tolerance");
        }
    }
    return errors;
}

std::vector<std::string> validate_qualified_zfp_history(
    const OutputConfig& config,
    const nimbus::output::QualifiedRadarZfpV4Context* radar_context)
{
    std::vector<std::string> errors;
    if (config.format != OutputFormat::zfp)
    {
        return errors;
    }
    const bool radar_selected = std::any_of(
        config.resolved_fields.begin(), config.resolved_fields.end(),
        [](const std::string& field)
        { return nimbus::output::is_qualified_radar_zfp_v4_field(field); });
    if (radar_selected && radar_context != nullptr)
    {
        errors = qualified_radar_zfp_v4_candidate_errors(config, *radar_context);
        if (!nimbus::output::kQualifiedRadarZfpV4PublicAdmissionOpen)
        {
            for (const auto& field : config.resolved_fields)
            {
                if (nimbus::output::is_qualified_radar_zfp_v4_field(field))
                {
                    errors.push_back("field '" + field +
                        "' is not admitted to qualified ZFP: reversible v4 public gate remains closed");
                }
            }
        }
        return errors;
    }
    if (config.zfp_mode != qualified_zfp_mode())
    {
        errors.push_back("qualified ZFP requires accuracy mode; precision/rate are visualization-only");
    }
    if (qualified_zfp_requires_per_field_tolerances() && !config.zfp_per_field_tolerances)
    {
        errors.push_back("qualified ZFP requires explicit per-field tolerances");
    }
    if (config.zfp_sparse_threshold > qualified_zfp_max_sparse_threshold())
    {
        errors.push_back(
            "qualified ZFP forbids sparse thresholding because its extra error is outside the codec bound");
    }
    if (config.zfp_float16_prefilter && !qualified_zfp_allows_float16_prefilter())
    {
        errors.push_back(
            "qualified ZFP forbids the float16 prefilter because its extra error is outside the codec bound");
    }
    if (config.zfp_predictive_delta && config.zfp_keyframe_interval < 3)
    {
        errors.push_back(
            "qualified ZFP predictive delta requires a keyframe interval of at least three frames");
    }
    if (!is_qualified_zfp_preset(config.preset))
        errors.push_back("selected output preset has no qualified ZFP field-set contract");
    if (std::find(qualified_zfp_field_execution_policies().begin(),
                  qualified_zfp_field_execution_policies().end(),
                  config.zfp_field_execution) ==
        qualified_zfp_field_execution_policies().end())
    {
        errors.push_back(
            "qualified ZFP field execution must be serial or field_parallel");
    }
    if (config.zfp_field_workers < 0 ||
        config.zfp_field_workers > qualified_zfp_max_field_workers() ||
        config.zfp_field_scratch_mib <
            qualified_zfp_min_field_scratch_mib() ||
        config.zfp_field_scratch_mib >
            qualified_zfp_max_field_scratch_mib())
    {
        errors.push_back(
            "qualified ZFP field worker/scratch budgets are outside supported bounds");
    }
    if (config.keep_last_frames <= 0)
        errors.push_back("qualified ZFP requires positive finite frame retention");
    if (config.minimum_free_space_mib <= 0)
        errors.push_back("qualified ZFP requires a positive retained free-space reserve");
    if (config.maximum_retained_history_files <= 0)
        errors.push_back("qualified ZFP requires a positive retained history file ceiling");
    if (config.resolved_fields.empty())
    {
        errors.push_back("qualified ZFP requires at least one resolved field");
        return errors;
    }

    const auto& budgets = qualified_zfp_field_budgets();
    for (const std::string& field : config.resolved_fields)
    {
        if (k_radar_product_fields.count(field) != 0)
        {
            errors.push_back("field '" + field +
                             "' is not admitted to qualified ZFP: decoded radar range and signal contracts remain unresolved");
            continue;
        }
        const auto budget = budgets.find(field);
        if (budget == budgets.end())
        {
            errors.push_back("field '" + field + "' has no qualified unit-aware ZFP error budget");
            continue;
        }
        const auto tolerance = config.zfp_field_tolerances.find(field);
        if (tolerance == config.zfp_field_tolerances.end())
        {
            errors.push_back("field '" + field + "' is missing its explicit ZFP tolerance");
            continue;
        }
        if (!std::isfinite(tolerance->second) || tolerance->second <= 0.0 ||
            tolerance->second > budget->second.max_configured_absolute_error)
        {
            std::ostringstream message;
            message << "field '" << field << "' tolerance=" << tolerance->second
                    << " exceeds qualified maximum=" << budget->second.max_configured_absolute_error << " "
                    << budget->second.units;
            errors.push_back(message.str());
        }
    }
    return errors;
}

const std::vector<std::string>& get_preset_fields(FieldPreset preset)
{
    switch (preset)
    {
    case FieldPreset::minimal:
        return k_minimal_fields;
    case FieldPreset::radar_ready:
        return k_radar_ready_fields;
    case FieldPreset::radar_zdr_exploratory:
        return k_radar_zdr_exploratory_fields;
    case FieldPreset::radar_polarimetric_ready:
        return k_radar_polarimetric_ready_fields;
    case FieldPreset::science_validation:
        return k_science_validation_fields;
    case FieldPreset::presentation:
        return k_presentation_fields;
    case FieldPreset::studio_full:
        return k_studio_full_fields;
    case FieldPreset::core_only:
        return k_core_only_fields;
    case FieldPreset::all:
        return available_output_field_names();
    case FieldPreset::custom:
        return output_field_names(); // fallback
    }
    return output_field_names();
}

const std::vector<std::string>& get_all_field_names()
{
    return output_field_names();
}

const std::vector<std::string>& qualification_diagnostics_fields(
    QualificationDiagnosticsPreset preset)
{
    static const std::vector<std::string> empty;
    switch (preset)
    {
    case QualificationDiagnosticsPreset::none:
        return empty;
    case QualificationDiagnosticsPreset::moving_reference_v1:
        return k_moving_reference_v1_diagnostic_fields;
    }
    return empty;
}

const char* qualification_diagnostics_preset_name(
    QualificationDiagnosticsPreset preset)
{
    switch (preset)
    {
    case QualificationDiagnosticsPreset::none:
        return "none";
    case QualificationDiagnosticsPreset::moving_reference_v1:
        return "moving_reference_v1";
    }
    return "none";
}

void resolve_output_fields(OutputConfig& config)
{
    config.resolved_fields.clear();

    if (config.preset == FieldPreset::custom)
    {
        if (config.custom_fields.empty())
        {
            throw std::invalid_argument("output.fields is empty while output.preset is 'custom'");
        }
        for (const auto& name : config.custom_fields)
        {
            if (all_fields_set().count(name))
            {
                config.resolved_fields.insert(name);
            }
            else
            {
                throw std::invalid_argument("Unknown field '" + name + "' in output.fields");
            }
        }
    }
    else
    {
        const auto& preset_fields = get_preset_fields(config.preset);
        config.resolved_fields.insert(preset_fields.begin(), preset_fields.end());
    }
}

void merge_selected_radar_output_fields(
    OutputConfig& config, const std::vector<std::string>& selected_products)
{
    for (const std::string& field : k_radar_product_fields)
        config.resolved_fields.erase(field);
    for (const std::string& field : selected_products)
    {
        if (k_radar_product_fields.count(field) == 0)
            throw std::invalid_argument("unknown radar output product '" + field + "'");
        config.resolved_fields.insert(field);
    }
}

bool is_sparse_eligible(const std::string& field_name)
{
    return k_sparse_eligible_fields.count(field_name) > 0;
}

bool is_float16_eligible(const std::string& field_name)
{
    return k_float16_eligible_fields.count(field_name) > 0;
}

WriteCadenceTier get_field_write_cadence_tier(const std::string& field_name)
{
    if (k_zfp_tight_fields.count(field_name))
    {
        return WriteCadenceTier::fast;
    }
    if (k_zfp_moderate_fields.count(field_name))
    {
        return WriteCadenceTier::medium;
    }
    return WriteCadenceTier::slow;
}

void apply_default_compression_tiers(OutputConfig& config)
{
    if (!config.zfp_per_field_tolerances)
    {
        return;
    }

    // Assign each resolved field to its tier tolerance.
    // Existing entries (from YAML overrides) are preserved.
    std::size_t tight_count    = 0;
    std::size_t moderate_count = 0;
    std::size_t loose_count    = 0;
    std::size_t explicit_count = 0;

    for (const auto& name : config.resolved_fields)
    {
        if (config.zfp_reversible_radar_v4 &&
            nimbus::output::is_qualified_radar_zfp_v4_field(name))
        {
            continue;
        }
        if (config.zfp_field_tolerances.count(name))
        {
            ++explicit_count;
            continue; // Explicit YAML override takes precedence
        }

        const auto qualified_budget = qualified_zfp_field_budgets().find(name);
        if (config.format == OutputFormat::zfp && is_qualified_zfp_preset(config.preset) &&
            qualified_budget != qualified_zfp_field_budgets().end())
        {
            config.zfp_field_tolerances[name] =
                qualified_budget->second.max_configured_absolute_error;
            ++explicit_count;
        }
        else if (k_zfp_tight_fields.count(name))
        {
            config.zfp_field_tolerances[name] = config.zfp_tight_tolerance;
            ++tight_count;
        }
        else if (k_zfp_moderate_fields.count(name))
        {
            config.zfp_field_tolerances[name] = config.zfp_moderate_tolerance;
            ++moderate_count;
        }
        else
        {
            config.zfp_field_tolerances[name] = config.zfp_loose_tolerance;
            ++loose_count;
        }
    }

    tmv::log_info("[OUTPUT] ZFP per-field tolerances: ",
                  explicit_count,
                  " explicit; defaults assigned: ",
                  tight_count,
                  " tight (",
                  config.zfp_tight_tolerance,
                  "), ",
                  moderate_count,
                  " moderate (",
                  config.zfp_moderate_tolerance,
                  "), ",
                  loose_count,
                  " loose (",
                  config.zfp_loose_tolerance,
                  ")");
}

void estimate_qualification_diagnostics_disk_budget(
    OutputConfig& config, int nr, int nth, int duration_s)
{
    if (config.estimated_total_bytes >=
        config.qualification_diagnostics_estimated_total_bytes)
    {
        config.estimated_total_bytes -=
            config.qualification_diagnostics_estimated_total_bytes;
    }
    config.qualification_diagnostics_estimated_bytes_per_export = 0;
    config.qualification_diagnostics_estimated_total_bytes = 0;
    const auto& fields = qualification_diagnostics_fields(
        config.qualification_diagnostics_preset);
    if (fields.empty() || duration_s < 0)
        return;
    if (config.qualification_diagnostics_interval_s <= 0)
        throw std::invalid_argument(
            "qualification diagnostics interval must be positive");
    const auto horizontal_cells =
        static_cast<std::size_t>(nr) * static_cast<std::size_t>(nth);
    constexpr std::size_t metadata_bytes_per_export = 64 * 1024;
    const auto exports = static_cast<std::size_t>(
        duration_s / config.qualification_diagnostics_interval_s);
    config.qualification_diagnostics_estimated_bytes_per_export =
        fields.size() * horizontal_cells * sizeof(float);
    config.qualification_diagnostics_estimated_total_bytes = exports *
        (config.qualification_diagnostics_estimated_bytes_per_export +
         metadata_bytes_per_export);
    config.estimated_total_bytes +=
        config.qualification_diagnostics_estimated_total_bytes;
}

void estimate_disk_budget(OutputConfig& config, int nr, int nth, int nz, int duration_s, int write_every_s)
{
    config.estimated_bytes_per_export = 0;
    config.estimated_total_bytes = 0;
    config.estimated_retained_history_files = 0;
    config.qualification_diagnostics_estimated_bytes_per_export = 0;
    config.qualification_diagnostics_estimated_total_bytes = 0;
    if (duration_s < 0)
        return;

    if (config.zfp_reversible_radar_v4 &&
        (config.format != OutputFormat::zfp || nr <= 0 || nth <= 0 || nz <= 0))
        throw std::invalid_argument("reversible radar ZFP has invalid format or dimensions");
    const bool checked_zfp = config.format == OutputFormat::zfp;
    const auto horizontal_cells = checked_zfp
        ? checked_output_mul(static_cast<std::size_t>(nr),
                             static_cast<std::size_t>(nth))
        : static_cast<std::size_t>(nr) * static_cast<std::size_t>(nth);
    const auto voxels = checked_zfp
        ? checked_output_mul(horizontal_cells, static_cast<std::size_t>(nz))
        : horizontal_cells * static_cast<std::size_t>(nz);
    constexpr std::size_t metadata_bytes_per_export = 64 * 1024;
    std::size_t num_exports = 0;
    if (write_every_s > 0)
    {
        // Ordinary history includes initialized t=0.
        num_exports = static_cast<std::size_t>(duration_s / write_every_s) + 1;
        std::size_t uncompressed_payload_bytes = 0;
        std::size_t lossless_npy_payload_bytes = 0;
        std::size_t reversible_payload_upper_bytes = 0;
        std::size_t hierarchy_volume_fields = 0;
        ReversibleZfpBound reversible_bound;
        if (config.zfp_reversible_radar_v4)
        {
            reversible_bound = reversible_zfp_field_bound(nr, nth, nz);
            const std::size_t scratch_limit = checked_output_mul(
                static_cast<std::size_t>(config.zfp_field_scratch_mib),
                1024U * 1024U);
            const std::size_t worker_limit = config.zfp_field_execution == "field_parallel"
                ? std::min<std::size_t>(scratch_limit, 2U * 1024U * 1024U)
                : scratch_limit;
            if (reversible_bound.scratch_bytes > worker_limit)
                throw std::invalid_argument(
                    "reversible radar ZFP brick/directory scratch exceeds its field budget");
        }
        for (const std::string& field_name : config.resolved_fields)
        {
            const OutputFieldDescriptor* field = find_output_field(field_name);
            const bool planar_volume_output =
                config.format != OutputFormat::npy_2d_slices && field != nullptr &&
                (field->kind == OutputFieldKind::SurfaceTranche ||
                 field->kind == OutputFieldKind::DirectSurface2D ||
                 field->kind == OutputFieldKind::ColumnTranche);
            const std::size_t field_cells =
                planar_volume_output ? horizontal_cells : voxels;
            if (!planar_volume_output) ++hierarchy_volume_fields;
            if (config.zfp_reversible_radar_v4 &&
                k_radar_product_fields.count(field_name) != 0)
            {
                reversible_payload_upper_bytes = checked_output_add(
                    reversible_payload_upper_bytes, reversible_bound.field_bytes);
                continue;
            }
            // A dense CSV row carries three decimal indices, separators, and
            // a formatted value. Budgeting it as one float can understate the
            // next frame by more than an order of magnitude and defeat the
            // runtime free-space guard.
            const std::size_t bytes_per_cell =
                config.format == OutputFormat::csv && !planar_volume_output
                    ? 48U : sizeof(float);
            const std::size_t field_bytes = checked_zfp
                ? checked_output_mul(field_cells, bytes_per_cell)
                : field_cells * bytes_per_cell;
            const auto* contract = tmv::find_field_contract(field_name);
            if (checked_zfp && contract != nullptr &&
                contract->allows_missing_values)
            {
                constexpr std::size_t npy_header_upper_bytes = 4096;
                lossless_npy_payload_bytes = checked_output_add(
                    lossless_npy_payload_bytes,
                    checked_output_add(field_bytes, npy_header_upper_bytes));
                continue;
            }
            uncompressed_payload_bytes = checked_zfp
                ? checked_output_add(uncompressed_payload_bytes, field_bytes)
                : uncompressed_payload_bytes + field_bytes;
        }

        double compression_ratio = 1.0;
        if (config.format == OutputFormat::zfp)
        {
            compression_ratio =
                (config.zfp_keyframe_interval > 0) ? 10.0 : 4.0;
        }
        const bool assembled_history =
            requires_snapshot_assembly(config.format, config.async_io);
        const FieldLodEstimate lod = assembled_history
            ? field_lod_estimate(nr, nth, nz)
            : FieldLodEstimate{};
        const std::size_t lod_payload_bytes = checked_output_add(
            checked_output_mul(
                checked_output_mul(hierarchy_volume_fields, lod.cells), sizeof(float)),
            assembled_history
                ? checked_output_mul(checked_output_mul(lod.cells, 4), sizeof(double))
                : 0);
        const std::size_t lossy_estimate = static_cast<std::size_t>(
            static_cast<double>(uncompressed_payload_bytes) / compression_ratio);
        config.estimated_bytes_per_export = checked_zfp
            ? checked_output_add(
                checked_output_add(
                    checked_output_add(lossy_estimate, lossless_npy_payload_bytes),
                    reversible_payload_upper_bytes),
                lod_payload_bytes)
            : lossy_estimate + lod_payload_bytes;
        config.estimated_total_bytes = checked_zfp
            ? checked_output_mul(num_exports,
                checked_output_add(config.estimated_bytes_per_export,
                                   metadata_bytes_per_export))
            : num_exports *
                (config.estimated_bytes_per_export + metadata_bytes_per_export);

        const std::size_t dependency_allowance =
            config.format == OutputFormat::zfp && config.zfp_keyframe_interval > 0
            ? static_cast<std::size_t>(config.zfp_keyframe_interval - 1)
            : 0;
        const std::size_t retained_frames = config.keep_last_frames > 0
            ? std::min(num_exports,
                       static_cast<std::size_t>(config.keep_last_frames) + dependency_allowance)
            : num_exports;
        const std::size_t payload_files_per_frame =
            config.format == OutputFormat::netcdf ? 1 : config.resolved_fields.size();
        const std::size_t hierarchy_files_per_frame = assembled_history
            ? checked_output_add(
                checked_output_mul(hierarchy_volume_fields,
                                   checked_output_add(1, lod.files)),
                checked_output_mul(4, lod.files))
            : 0;
        constexpr std::size_t per_frame_sidecars = 2; // manifest + validation report
        constexpr std::size_t series_fixed_files = 20; // head, budgets, summaries, geometry and safety margin
        constexpr std::size_t index_batch_records = 256;
        const std::size_t index_segments =
            retained_frames == 0 ? 0 : (retained_frames + index_batch_records - 1) / index_batch_records;
        config.estimated_retained_history_files = checked_output_add(
            checked_output_add(series_fixed_files, index_segments),
            checked_output_mul(retained_frames,
                checked_output_add(
                    checked_output_add(payload_files_per_frame,
                                       hierarchy_files_per_frame),
                    per_frame_sidecars)));
    }

    const auto& qualification_fields = qualification_diagnostics_fields(
        config.qualification_diagnostics_preset);
    estimate_qualification_diagnostics_disk_budget(
        config, nr, nth, duration_s);
    const std::size_t qualification_exports = qualification_fields.empty()
        ? 0
        : static_cast<std::size_t>(
              duration_s / config.qualification_diagnostics_interval_s);
    if (qualification_exports > 0)
    {
        const std::size_t retained = config.keep_last_frames > 0
            ? std::min(qualification_exports,
                       static_cast<std::size_t>(config.keep_last_frames))
            : qualification_exports;
        constexpr std::size_t series_fixed_files = 20;
        constexpr std::size_t index_batch_records = 256;
        const std::size_t qualification_files = series_fixed_files +
            (retained + index_batch_records - 1) / index_batch_records +
            retained * (qualification_fields.size() + 2);
        config.estimated_retained_history_files =
            std::max(config.estimated_retained_history_files, qualification_files);
    }
    if (config.maximum_retained_history_files > 0 &&
        config.estimated_retained_history_files >
            static_cast<std::size_t>(config.maximum_retained_history_files))
    {
        throw std::invalid_argument(
            "estimated retained history file count " +
            std::to_string(config.estimated_retained_history_files) +
            " exceeds output.maximum_retained_history_files=" +
            std::to_string(config.maximum_retained_history_files));
    }

    if (!config.estimate_disk_budget)
    {
        return;
    }

    const double total_gb = static_cast<double>(config.estimated_total_bytes) /
        (1024.0 * 1024.0 * 1024.0);
    const double per_export_mb = static_cast<double>(config.estimated_bytes_per_export) /
        (1024.0 * 1024.0);

    if (write_every_s > 0)
    {
        tmv::log_info("[OUTPUT] Disk budget: ",
                      config.resolved_fields.size(), " fields x ",
                      num_exports, " ordinary exports");
        tmv::log_info("[OUTPUT]   Per ordinary export: ",
                      std::round(per_export_mb * 10.0) / 10.0, " MB");
    }
    if (qualification_exports > 0)
    {
        const double qualification_mb = static_cast<double>(
            config.qualification_diagnostics_estimated_bytes_per_export) /
            (1024.0 * 1024.0);
        tmv::log_info("[OUTPUT]   Qualification diagnostics: ",
                      qualification_fields.size(), " planar fields x ",
                      qualification_exports, " exports; ",
                      std::round(qualification_mb * 10.0) / 10.0, " MB/export");
    }
    tmv::log_info("[OUTPUT]   Total estimated: ",
                  std::round(total_gb * 10.0) / 10.0, " GB");
    if (write_every_s > 0)
        tmv::log_info("[OUTPUT]   Retained metadata/file estimate: ",
                      config.estimated_retained_history_files,
                      " files per series; ceiling=",
                      config.maximum_retained_history_files > 0
                          ? std::to_string(config.maximum_retained_history_files)
                          : std::string("disabled"));

    if (total_gb > 100.0)
    {
        tmv::log_warn("[OUTPUT] estimated output exceeds 100 GB. "
                      "Consider output.fields: minimal or output.format: zfp");
    }
}

namespace
{

using OutputConfigMap = std::unordered_map<std::string, std::string>;

const std::string* find_config_value(const OutputConfigMap& config, const char* key)
{
    const auto it = config.find(key);
    return it == config.end() ? nullptr : &it->second;
}

void read_bool_option(const OutputConfigMap& config, const char* key, bool& target)
{
    if (const std::string* value = find_config_value(config, key))
    {
        target = local_parse_bool(*value);
    }
}

void read_positive_double_option(const OutputConfigMap& config, const char* key, double& target)
{
    if (const std::string* value = find_config_value(config, key))
    {
        double parsed = 0.0;
        if (local_try_parse_double(*value, parsed) && parsed > 0.0)
        {
            target = parsed;
        }
        else
        {
            throw std::invalid_argument(std::string(key) + " must be a positive finite number; got '" + *value + "'");
        }
    }
}

void read_non_negative_double_option(const OutputConfigMap& config, const char* key, double& target)
{
    if (const std::string* value = find_config_value(config, key))
    {
        double parsed = 0.0;
        if (local_try_parse_double(*value, parsed) && parsed >= 0.0)
        {
            target = parsed;
        }
        else
        {
            throw std::invalid_argument(std::string(key) + " must be a non-negative finite number; got '" + *value +
                                        "'");
        }
    }
}

void read_non_negative_float_option(const OutputConfigMap& config, const char* key, float& target)
{
    if (const std::string* value = find_config_value(config, key))
    {
        double parsed = 0.0;
        if (local_try_parse_double(*value, parsed) && parsed >= 0.0)
        {
            target = static_cast<float>(parsed);
        }
        else
        {
            throw std::invalid_argument(std::string(key) + " must be a non-negative finite number; got '" + *value +
                                        "'");
        }
    }
}

void read_positive_int_option(const OutputConfigMap& config, const char* key, int& target)
{
    if (const std::string* value = find_config_value(config, key))
    {
        int parsed = 0;
        if (local_try_parse_int(*value, parsed) && parsed > 0)
        {
            target = parsed;
        }
        else
        {
            throw std::invalid_argument(std::string(key) + " must be a positive integer; got '" + *value + "'");
        }
    }
}

void read_non_negative_int_option(const OutputConfigMap& config, const char* key, int& target)
{
    if (const std::string* value = find_config_value(config, key))
    {
        int parsed = 0;
        if (local_try_parse_int(*value, parsed) && parsed >= 0)
        {
            target = parsed;
        }
        else
        {
            throw std::invalid_argument(std::string(key) + " must be a non-negative integer; got '" + *value + "'");
        }
    }
}

void apply_zfp_format_option(const std::string& value, OutputConfig& out)
{
#ifdef HAVE_ZFP
    out.format = OutputFormat::zfp;
    out.zfp_requested_format_tier = value;
    // The tier name sets the default accuracy tolerance; an explicit
    // output.zfp_tolerance overrides it after format parsing. A bare "zfp"
    // keeps the struct default (1e-4).
    if (value == "zfp_low")
    {
        out.zfp_tolerance = 1.0e-2;
    }
    else if (value == "zfp_medium")
    {
        out.zfp_tolerance = 1.0e-3;
    }
    else if (value == "zfp_high")
    {
        out.zfp_tolerance = 1.0e-4;
    }
#else
    throw std::runtime_error("output.format '" + value +
                             "' requested but binary was compiled without ZFP support. "
                             "Rebuild with ZFP=1 or change output.format.");
#endif
}

void parse_output_format_option(const OutputConfigMap& config, OutputConfig& out)
{
    const std::string* raw_value = find_config_value(config, "output.format");
    if (!raw_value)
    {
        return;
    }

    const std::string value = local_to_lower(*raw_value);
    if (value == "npy_3d" || value == "npy3d")
    {
        out.format = OutputFormat::npy_3d;
    }
    else if (value == "npy_2d_slices" || value == "npy2d" || value == "legacy")
    {
        out.format = OutputFormat::npy_2d_slices;
    }
    else if (value == "netcdf" || value == "netcdf4")
    {
#ifdef HAVE_NETCDF
        out.format = OutputFormat::netcdf;
#else
        throw std::runtime_error("output.format '" + value +
                                 "' requested but this binary was compiled without libnetcdf support");
#endif
    }
    else if (value == "zfp" || value == "zfp_low" || value == "zfp_medium" || value == "zfp_high")
    {
        apply_zfp_format_option(value, out);
    }
    else if (value == "csv")
    {
        out.format = OutputFormat::csv;
    }
    else
    {
        throw std::runtime_error("Invalid output.format '" + value +
                                 "'; expected npy_3d, npy_2d_slices, netcdf, csv, or a ZFP tier");
    }
}

void parse_output_fields_option(const OutputConfigMap& config, OutputConfig& out)
{
    const std::string* raw_value = find_config_value(config, "output.fields");
    if (!raw_value)
    {
        return;
    }

    const std::string value = local_to_lower(*raw_value);
    if (value == "all")
    {
        out.preset = FieldPreset::all;
    }
    else if (value == "minimal")
    {
        out.preset = FieldPreset::minimal;
    }
    else if (value == "radar_ready")
    {
        out.preset = FieldPreset::radar_ready;
    }
    else if (value == "radar_zdr_exploratory")
    {
        out.preset = FieldPreset::radar_zdr_exploratory;
    }
    else if (value == "radar_polarimetric_ready")
    {
        out.preset = FieldPreset::radar_polarimetric_ready;
    }
    else if (value == "science_validation")
    {
        out.preset = FieldPreset::science_validation;
    }
    else if (value == "presentation")
    {
        out.preset = FieldPreset::presentation;
    }
    else if (value == "studio_full")
    {
        out.preset = FieldPreset::studio_full;
    }
    else if (value == "core_only" || value == "core")
    {
        out.preset = FieldPreset::core_only;
    }
    else if (value == "dynamics_full")
    {
        // User-facing name (the v1 config exposes minimal / presentation /
        // dynamics_full) for the comprehensive field set.
        out.preset = FieldPreset::all;
    }
    else
    {
        out.preset        = FieldPreset::custom;
        out.custom_fields = local_parse_string_list(*raw_value);
    }
}

void parse_zfp_mode_option(const OutputConfigMap& config, OutputConfig& out)
{
    const std::string* raw_value = find_config_value(config, "output.zfp_mode");
    if (!raw_value)
    {
        return;
    }

    const std::string value = local_to_lower(*raw_value);
    if (value == "accuracy" || value == "precision" || value == "rate")
    {
        out.zfp_mode = value;
    }
    else
    {
        throw std::runtime_error("Invalid output.zfp_mode '" + value + "'; expected accuracy, precision, or rate");
    }
}

void parse_zfp_field_tolerance_overrides(const OutputConfigMap& config, OutputConfig& out)
{
    const std::string prefix = "output.zfp_field_tolerance.";
    for (const auto& [key, value] : config)
    {
        if (key.size() > prefix.size() && key.compare(0, prefix.size(), prefix) == 0)
        {
            const std::string field_name = key.substr(prefix.size());
            if (!all_fields_set().count(field_name))
            {
                throw std::invalid_argument("Unknown output field in ZFP tolerance override: '" + field_name + "'");
            }
            double parsed = 0.0;
            if (local_try_parse_double(value, parsed) && parsed > 0.0)
            {
                out.zfp_field_tolerances[field_name] = parsed;
            }
            else
            {
                throw std::invalid_argument("output.zfp_field_tolerance." + field_name +
                                            " must be a positive finite number; got '" + value + "'");
            }
        }
    }
}

void parse_zfp_options(const OutputConfigMap& config, OutputConfig& out)
{
    read_positive_double_option(config, "output.zfp_tolerance", out.zfp_tolerance);
    parse_zfp_mode_option(config, out);
    read_positive_int_option(config, "output.zfp_precision_bits", out.zfp_precision_bits);
    read_positive_int_option(config, "output.zfp_rate_bps", out.zfp_rate_bps);
    read_non_negative_int_option(config, "output.zfp_keyframe_interval", out.zfp_keyframe_interval);
    read_non_negative_float_option(config, "output.zfp_sparse_threshold", out.zfp_sparse_threshold);
    read_bool_option(config, "output.zfp_float16_prefilter", out.zfp_float16_prefilter);
    read_bool_option(config, "output.zfp_predictive_delta", out.zfp_predictive_delta);
    if (const auto it = config.find("output.zfp_field_execution");
        it != config.end())
    {
        if (it->second != "serial" && it->second != "field_parallel")
            throw std::invalid_argument(
                "output.zfp_field_execution must be serial or field_parallel");
        out.zfp_field_execution = it->second;
    }
    read_non_negative_int_option(config, "output.zfp_field_workers",
                                 out.zfp_field_workers);
    read_positive_int_option(config, "output.zfp_field_scratch_mib",
                             out.zfp_field_scratch_mib);
    if (out.zfp_field_workers > 16)
        throw std::invalid_argument(
            "output.zfp_field_workers must be in [0, 16]");
    if (out.zfp_field_scratch_mib > 4096)
        throw std::invalid_argument(
            "output.zfp_field_scratch_mib must be in [1, 4096]");
    read_bool_option(config, "output.zfp_per_field_tolerances", out.zfp_per_field_tolerances);
    read_positive_double_option(config, "output.zfp_tight_tolerance", out.zfp_tight_tolerance);
    read_positive_double_option(config, "output.zfp_moderate_tolerance", out.zfp_moderate_tolerance);
    read_positive_double_option(config, "output.zfp_loose_tolerance", out.zfp_loose_tolerance);
    parse_zfp_field_tolerance_overrides(config, out);
    if (out.zfp_precision_bits < 1 || out.zfp_precision_bits > 32)
    {
        throw std::runtime_error("output.zfp_precision_bits must be in [1, 32]");
    }
}

void parse_tiered_cadence_options(const OutputConfigMap& config, OutputConfig& out)
{
    read_bool_option(config, "output.tiered_write_cadence", out.tiered_write_cadence);
    read_positive_int_option(config, "output.write_cadence_medium_s", out.write_cadence_medium_s);
    read_positive_int_option(config, "output.write_cadence_slow_s", out.write_cadence_slow_s);
}

void parse_qualification_diagnostics_options(
    const OutputConfigMap& config, OutputConfig& out)
{
    if (const std::string* raw = find_config_value(
            config, "output.qualification_diagnostics.preset"))
    {
        const std::string value = local_to_lower(*raw);
        if (value == "none")
        {
            out.qualification_diagnostics_preset =
                QualificationDiagnosticsPreset::none;
        }
        else if (value == "moving_reference_v1")
        {
            out.qualification_diagnostics_preset =
                QualificationDiagnosticsPreset::moving_reference_v1;
        }
        else
        {
            throw std::invalid_argument(
                "output.qualification_diagnostics.preset must be none or "
                "moving_reference_v1; got '" + *raw + "'");
        }
    }
    read_positive_int_option(
        config,
        "output.qualification_diagnostics.interval_s",
        out.qualification_diagnostics_interval_s);
}

} // namespace

OutputConfig parse_output_config(const std::unordered_map<std::string, std::string>& config)
{
    OutputConfig out;
    parse_output_format_option(config, out);
    parse_output_fields_option(config, out);
    read_bool_option(config, "output.async_io", out.async_io);
    read_positive_double_option(config, "output.live_shm_interval_s",
                                out.live_shm_interval_s);
    read_bool_option(config, "output.live_shm_full_rate",
                     out.live_shm_full_rate);
    read_bool_option(config, "output.trajectories", out.trajectories);
    read_non_negative_double_option(config, "output.trajectory_seed_time_s", out.trajectory_seed_time_s);
    read_positive_int_option(config, "output.trajectory_count", out.trajectory_count);
    read_non_negative_double_option(config, "output.trajectory_sample_interval_s",
                                    out.trajectory_sample_interval_s);
    read_positive_int_option(config, "output.trajectory_max_samples_per_parcel",
                             out.trajectory_max_samples_per_parcel);
    read_non_negative_int_option(config, "output.netcdf_deflate_level", out.netcdf_deflate_level);
    if (out.netcdf_deflate_level > 9)
    {
        throw std::runtime_error("output.netcdf_deflate_level must be in [0, 9]");
    }
    read_bool_option(config, "output.netcdf_shuffle", out.netcdf_shuffle);
    read_positive_int_option(config, "output.netcdf_target_chunk_kib", out.netcdf_target_chunk_kib);
    parse_zfp_options(config, out);
    parse_tiered_cadence_options(config, out);
    parse_qualification_diagnostics_options(config, out);
    read_non_negative_int_option(config, "output.keep_last_frames", out.keep_last_frames);
    read_non_negative_int_option(config, "output.minimum_free_space_mib", out.minimum_free_space_mib);
    read_non_negative_int_option(config, "output.maximum_retained_history_files",
                                 out.maximum_retained_history_files);
    read_bool_option(config, "output.estimate_disk_budget", out.estimate_disk_budget);
    return out;
}

const char* output_format_name(OutputFormat fmt)
{
    switch (fmt)
    {
    case OutputFormat::npy_3d:
        return "npy_3d";
    case OutputFormat::npy_2d_slices:
        return "npy_2d_slices";
    case OutputFormat::netcdf:
        return "netcdf";
    case OutputFormat::zfp:
        return "zfp";
    case OutputFormat::csv:
        return "csv";
    }
    return "unknown";
}

const char* field_preset_name(FieldPreset preset)
{
    switch (preset)
    {
    case FieldPreset::all:
        return "all";
    case FieldPreset::minimal:
        return "minimal";
    case FieldPreset::radar_ready:
        return "radar_ready";
    case FieldPreset::radar_zdr_exploratory:
        return "radar_zdr_exploratory";
    case FieldPreset::radar_polarimetric_ready:
        return "radar_polarimetric_ready";
    case FieldPreset::presentation:
        return "presentation";
    case FieldPreset::studio_full:
        return "studio_full";
    case FieldPreset::core_only:
        return "core_only";
    case FieldPreset::custom:
        return "custom";
    case FieldPreset::science_validation:
        return "science_validation";
    }
    return "unknown";
}
