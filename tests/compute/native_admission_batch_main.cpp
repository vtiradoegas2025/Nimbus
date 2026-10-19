#define main nimbus_embedded_runtime_main
#include "../../src/core/runtime/nimbus_sim.cpp"
#undef main

#include "core/infra/sha256.hpp"

#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace
{

std::unordered_map<std::string, std::string> parse_tuple(const std::string& line)
{
    std::unordered_map<std::string, std::string> values;
    std::size_t begin = 0;
    while (begin < line.size())
    {
        const std::size_t end = line.find('|', begin);
        const std::string item = line.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
        const std::size_t separator = item.find('=');
        if (separator == std::string::npos || separator == 0 || separator + 1 == item.size())
            throw std::runtime_error("canonical tuple contains a malformed field");
        if (!values.emplace(item.substr(0, separator), item.substr(separator + 1)).second)
            throw std::runtime_error("canonical tuple contains a duplicate field");
        if (end == std::string::npos)
            break;
        begin = end + 1;
    }
    return values;
}

const std::string& required(const std::unordered_map<std::string, std::string>& values, const char* key)
{
    const auto found = values.find(key);
    if (found == values.end())
        throw std::runtime_error(std::string("canonical tuple omitted ") + key);
    return found->second;
}

bool boolean(const std::unordered_map<std::string, std::string>& values, const char* key)
{
    const auto& value = required(values, key);
    if (value == "true")
        return true;
    if (value == "false")
        return false;
    throw std::runtime_error(std::string("canonical tuple has invalid boolean ") + key);
}

template <class Enum, class Parser>
Enum parsed(const std::unordered_map<std::string, std::string>& values, const char* key, Parser parser)
{
    Enum result{};
    if (!parser(required(values, key), result))
        throw std::runtime_error(std::string("canonical tuple has invalid ") + key);
    return result;
}

tmv::init::config::ConfigV1 materialize(const tmv::init::config::ConfigV1& base,
                                        const std::unordered_map<std::string, std::string>& values)
{
    using namespace tmv::init::config;
    ConfigV1 config = base;
    const std::string& grid = required(values, "grid");
    config.grid.staggering = grid == "collocated" ? StaggerType::Collocated : StaggerType::CGrid;
    if (grid != "collocated" && grid != "c_grid")
        throw std::runtime_error("canonical tuple has invalid grid");
    config.dynamics.scheme = required(values, "dynamics");
    config.extensions.compute.backend = parsed<tmv::init::config::ComputeBackend>(
        values, "backend", parse_compute_backend);
    config.extensions.execution.mode = boolean(values, "qualified")
        ? ExecutionMode::Qualified : ExecutionMode::Exploratory;
    config.microphysics.scheme = parsed<tmv::init::config::MicrophysicsScheme>(
        values, "microphysics", parse_microphysics_scheme);
    config.radiation.scheme = parsed<RadiationScheme>(values, "radiation", parse_radiation_scheme);
    if (config.radiation.scheme != RadiationScheme::None && config.radiation.do_lw &&
        config.surface.tsk_k < 150.0)
        config.surface.tsk_k = 300.0;
    config.boundary_layer.scheme =
        parsed<BoundaryLayerScheme>(values, "boundary_layer", parse_boundary_layer_scheme);
    config.turbulence.scheme = parsed<TurbulenceScheme>(values, "turbulence", parse_turbulence_scheme);
    config.extensions.chaos.scheme = parsed<tmv::init::config::ChaosScheme>(
        values, "chaos", parse_chaos_scheme);
    config.terrain.scheme = parsed<TerrainScheme>(values, "terrain", parse_terrain_scheme);
    config.advection.scheme = parsed<AdvectionScheme>(values, "advection", parse_advection_scheme);
    config.diffusion.scheme = parsed<DiffusionScheme>(values, "diffusion", parse_diffusion_scheme);
    config.time.integrator = parsed<TimeIntegrator>(values, "time_stepping", parse_time_integrator);
    if (values.count("turbulence_mode") != 0)
        config.turbulence.mode = parsed<TurbulenceMode>(values, "turbulence_mode", parse_turbulence_mode);
    if (grid == "c_grid")
    {
        config.surface.scheme = parsed<SurfaceScheme>(values, "surface", parse_surface_scheme);
        config.surface.apply_fluxes = boolean(values, "surface_fluxes");
        config.output.format = parsed<tmv::init::config::OutputFormat>(
            values, "output", parse_output_format);
        const auto& sounding = required(values, "sounding");
        if (sounding == "parametric")
        {
            config.initial_conditions.sounding.type = SoundingType::ParametricCAPE;
            config.initial_conditions.sounding.file_path.clear();
        }
        else if (sounding == "auto" || sounding == "cm1" || sounding == "sharpy")
        {
            config.initial_conditions.sounding.type = SoundingType::File;
            config.initial_conditions.sounding.file_path =
                sounding == "cm1"
                    ? "tests/fixtures/configs/qualification/soundings/catalog_cm1_profile.txt"
                    : "tests/fixtures/configs/qualification/soundings/lin_cloudy_profile.txt";
        }
        else
            throw std::runtime_error("canonical tuple has invalid sounding");
    }
    config.extensions.compute.allow_fallback = false;
    config.extensions.compute.gpu_apply_microphysics = false;
    config.extensions.compute.gpu_apply_slow = false;
    config.extensions.compute.gpu_apply_sgs = false;
    config.domain_motion.mode = DomainMotionMode::Stationary;
    config.domain_motion.nesting.enabled = false;
    config.domain_motion.tracking.enabled = false;
    return config;
}

void json_string(const std::string& value)
{
    std::cout << '"';
    for (const char c : value)
    {
        if (c == '"' || c == '\\')
            std::cout << '\\';
        if (c == '\n')
            std::cout << "\\n";
        else
            std::cout << c;
    }
    std::cout << '"';
}

std::string sounding_cache_key(const tmv::init::SoundingSourceConfig& source,
                               const tmv::init::config::ConfigV1& config)
{
    std::ostringstream key;
    key << std::setprecision(std::numeric_limits<double>::max_digits10)
        << static_cast<int>(source.type) << '|' << config.grid.nz << '|' << config.grid.dz << '|';
    switch (source.type)
    {
    case tmv::init::SoundingSourceConfig::Type::ParametricCAPE:
        key << source.parametric.cape_target_jkg << '|' << source.parametric.surface_theta_k << '|'
            << source.parametric.surface_qv_kgkg << '|' << source.parametric.tropopause_z_m;
        break;
    case tmv::init::SoundingSourceConfig::Type::File:
        key << source.file.path << '|' << source.file.scheme_id << '|' << source.file.require_winds << '|'
            << source.file.use_fallback_profiles << '|' << source.file.interpolation_method << '|'
            << source.file.extrapolate_below_ground << '|' << source.file.extrapolate_above_top << '|'
            << source.file.min_pressure_hpa << '|' << source.file.max_pressure_hpa << '|'
            << source.file.min_temperature_k << '|' << source.file.max_temperature_k;
        break;
    case tmv::init::SoundingSourceConfig::Type::ParametricTargets:
        key << source.targets.target_cape_jkg << '|' << source.targets.target_cin_jkg << '|'
            << source.targets.target_lcl_m << '|' << source.targets.target_lfc_m << '|'
            << source.targets.target_el_m << '|' << source.targets.surface_theta_k << '|'
            << source.targets.surface_pressure_pa << '|' << source.targets.moisture_qv_kgkg_override;
        break;
    }
    return key.str();
}

struct FullPreflightState
{
    std::unordered_map<std::string, tmv::init::Sounding> sounding_columns;
    std::map<std::string, std::size_t> warnings;
    std::size_t serialization_checks = 0;
    std::size_t projection_checks = 0;
    std::size_t compatibility_checks = 0;
    std::size_t sounding_checks = 0;
    std::size_t sounding_materializations = 0;
    std::size_t sounding_cache_hits = 0;
    std::size_t lifetime_checks = 0;
};

class ScopedCoutMute
{
public:
    ScopedCoutMute() : previous_(std::cout.rdbuf(sink_.rdbuf())) {}
    ~ScopedCoutMute() { std::cout.rdbuf(previous_); }

    ScopedCoutMute(const ScopedCoutMute&) = delete;
    ScopedCoutMute& operator=(const ScopedCoutMute&) = delete;

private:
    std::ostringstream sink_;
    std::streambuf* previous_;
};

std::vector<std::string> validate_full_preflight(const tmv::init::config::ConfigV1& config,
                                                 FullPreflightState& state,
                                                 std::string* serialized_digest = nullptr)
{
    std::unordered_map<std::string, std::string> projected;
    try
    {
        const std::string serialized = tmv::init::config::serialize_v1_to_yaml(config);
        if (serialized.find("\nschema: nimbus.v1\n") == std::string::npos)
            throw std::runtime_error("serialized document omitted the strict schema stamp");
        if (serialized_digest != nullptr)
        {
            nimbus::crypto::Sha256 digest;
            digest.update(serialized.data(), serialized.size());
            *serialized_digest = digest.hex_digest();
        }
        ++state.serialization_checks;
    }
    catch (const std::exception& error)
    {
        return {std::string("serialization rejected: ") + error.what()};
    }
    try
    {
        projected = tmv::init::config::serialize_v1_to_v0_map(config);
        (void)projected.at("dynamics.scheme");
        (void)projected.at("turbulence.scheme");
        ++state.projection_checks;
    }
    catch (const std::exception& error)
    {
        return {std::string("runtime projection rejected: ") + error.what()};
    }

    const auto sounding = validation_sounding_config(config);
    const auto hodograph = validation_hodograph_config(config);
    const auto trigger = validation_trigger_config(config);
    tmv::init::ValidationInputs inputs;
    inputs.scheme_id = projected.at("dynamics.scheme");
    inputs.turbulence_scheme_id = projected.at("turbulence.scheme");
    inputs.coordinate = config.grid.coordinate_system;
    inputs.stagger = config.grid.staggering;
    inputs.sounding = sounding;
    inputs.hodograph = hodograph;
    inputs.trigger = trigger;
    const auto compatibility = tmv::init::validate_initial_condition_config(inputs);
    ++state.compatibility_checks;
    for (const auto& warning : compatibility.warnings)
        ++state.warnings[warning];
    if (!compatibility.ok)
    {
        if (compatibility.errors.empty())
            return {"initial-condition compatibility rejected without an error"};
        return {compatibility.errors.front()};
    }

    std::vector<double> z_m(static_cast<std::size_t>(config.grid.nz));
    for (int k = 0; k < config.grid.nz; ++k)
        z_m[static_cast<std::size_t>(k)] = tmv::grid_coordinates::scalar_center(k, config.grid.dz);

    try
    {
        const std::string cache_key = sounding_cache_key(sounding, config);
        auto found = state.sounding_columns.find(cache_key);
        if (found == state.sounding_columns.end())
        {
            ScopedCoutMute mute;
            auto source = tmv::init::make_sounding_source(sounding);
            auto column = source->build(z_m, config.grid.dz);
            column.verify_self_consistent();
            found = state.sounding_columns.emplace(cache_key, std::move(column)).first;
            ++state.sounding_materializations;
        }
        else
            ++state.sounding_cache_hits;
        ++state.sounding_checks;
        validate_domain_lifetime_preflight(config, z_m, found->second, hodograph);
        ++state.lifetime_checks;
    }
    catch (const std::exception& error)
    {
        return {std::string("sounding/lifetime preflight rejected: ") + error.what()};
    }
    return {};
}

} // namespace

int main(int argc, char** argv)
{
    try
    {
        enum class Mode
        {
            ScienceSummary,
            ScienceAdmitted,
            FullSummary,
            FullAdmitted,
            FullAdmittedWithSignatures,
            Classify,
            Yaml,
            ConfigYaml,
        };
        Mode mode = Mode::ScienceSummary;
        std::size_t expected_count = 0;
        std::string expected_digest;
        std::string config_path;
        bool count_supplied = false, digest_supplied = false, mode_supplied = false;
        for (int index = 1; index < argc; ++index)
        {
            const std::string argument = argv[index];
            if (argument == "--expected-count" || argument == "--expected-sha256")
            {
                if (++index == argc)
                    throw std::runtime_error("stream identity option requires a value");
                const std::string value = argv[index];
                if (argument == "--expected-count")
                {
                    if (count_supplied || value.empty() || !std::all_of(value.begin(), value.end(),
                        [](char c)
                        {
                            return c >= '0' && c <= '9';
                        }))
                        throw std::runtime_error("invalid or duplicate expected tuple count");
                    const auto parsed_count = std::stoull(value);
                    if (parsed_count == 0 || parsed_count > std::numeric_limits<std::size_t>::max())
                        throw std::runtime_error("expected tuple count is outside the supported range");
                    expected_count = static_cast<std::size_t>(parsed_count);
                    count_supplied = true;
                }
                else
                {
                    if (digest_supplied || value.size() != 64 || !std::all_of(value.begin(), value.end(),
                        [](char c)
                        {
                            return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
                        }))
                        throw std::runtime_error("invalid or duplicate expected tuple SHA-256");
                    expected_digest = value;
                    digest_supplied = true;
                }
                continue;
            }
            if (mode_supplied)
                throw std::runtime_error("native admission accepts only one output mode");
            mode_supplied = true;
            if (argument == "--admitted")
                mode = Mode::ScienceAdmitted;
            else if (argument == "--full-preflight")
                mode = Mode::FullSummary;
            else if (argument == "--full-admitted")
                mode = Mode::FullAdmitted;
            else if (argument == "--full-admitted-with-signatures")
                mode = Mode::FullAdmittedWithSignatures;
            else if (argument == "--classify")
                mode = Mode::Classify;
            else if (argument == "--yaml")
                mode = Mode::Yaml;
            else if (argument == "--canonical-config")
            {
                mode = Mode::ConfigYaml;
                if (++index == argc)
                    throw std::runtime_error("--canonical-config requires a path");
                config_path = argv[index];
            }
            else
                throw std::runtime_error("unknown native admission option: " + argument);
        }
        const bool single_config = mode == Mode::Yaml || mode == Mode::ConfigYaml;
        if (count_supplied != digest_supplied || (single_config && count_supplied) ||
            (!single_config && !count_supplied))
            throw std::runtime_error("stream modes require an explicit count and SHA-256; single-config modes accept neither");
        const bool full = mode == Mode::FullSummary || mode == Mode::FullAdmitted ||
                          mode == Mode::FullAdmittedWithSignatures || mode == Mode::Classify;
        const bool emit = mode == Mode::ScienceAdmitted || mode == Mode::FullAdmitted ||
                          mode == Mode::FullAdmittedWithSignatures;
        const bool signed_input = mode == Mode::FullAdmittedWithSignatures;
        const auto loaded = tmv::init::config::load_v1_config(
            "tests/fixtures/configs/qualification/agrid_flat_cpu_core_ready.yaml");
        if (!loaded.ok())
            throw std::runtime_error("cannot load canonical native-admission base configuration");
        const auto base = *loaded.config;
        if (mode == Mode::ConfigYaml)
        {
            const auto selected = tmv::init::config::load_v1_config(config_path);
            if (!selected.ok())
                throw std::runtime_error("--canonical-config strict loader rejected " + config_path + ": " +
                                         tmv::init::config::format_errors(selected.errors));
            std::cout << tmv::init::config::serialize_v1_to_yaml(*selected.config);
            return 0;
        }
        if (mode == Mode::Yaml)
        {
            std::string tuple;
            std::string extra;
            if (!std::getline(std::cin, tuple) || tuple.empty() || std::getline(std::cin, extra))
                throw std::runtime_error("--yaml requires exactly one nonempty canonical tuple");
            const auto config = materialize(base, parse_tuple(tuple));
            const auto science_errors = validate_v1_selected_science_profile(config);
            if (!science_errors.empty())
                throw std::runtime_error("--yaml tuple failed native science admission: " + science_errors.front());
            FullPreflightState state;
            const auto preflight_errors = validate_full_preflight(config, state);
            if (!preflight_errors.empty())
                throw std::runtime_error("--yaml tuple failed native full preflight: " + preflight_errors.front());
            std::cout << tmv::init::config::serialize_v1_to_yaml(config);
            return 0;
        }
        nimbus::crypto::Sha256 input_digest;
        nimbus::crypto::Sha256 admitted_digest;
        std::map<std::string, std::size_t> partitions;
        std::map<std::string, std::size_t> rejection_reasons;
        FullPreflightState full_state;
        std::size_t input_count = 0;
        std::size_t science_admitted_count = 0;
        std::size_t admitted_count = 0;
        std::string input_line;
        while (std::getline(std::cin, input_line))
        {
            if (input_line.empty())
                throw std::runtime_error("canonical tuple stream contains an empty line");
            std::string signature;
            std::string line = input_line;
            if (signed_input)
            {
                const std::size_t separator = input_line.find('\t');
                if (separator == std::string::npos || separator == 0 || separator + 1 == input_line.size())
                    throw std::runtime_error("signed canonical tuple stream contains a malformed line");
                signature = input_line.substr(0, separator);
                line = input_line.substr(separator + 1);
            }
            const std::string framed = line + '\n';
            input_digest.update(framed.data(), framed.size());
            ++input_count;
            const auto values = parse_tuple(line);
            const auto config = materialize(base, values);
            std::vector<std::string> errors;
            try
            {
                errors = validate_v1_selected_science_profile(config);
            }
            catch (const std::exception& error)
            {
                errors.push_back(std::string("native projection rejected: ") + error.what());
            }
            const std::string partition = required(values, "grid") + "." + required(values, "backend") + "." +
                                          (boolean(values, "qualified") ? "qualified" : "exploratory");
            const bool science_admitted = errors.empty();
            std::string config_digest;
            if (errors.empty())
            {
                ++science_admitted_count;
                if (full)
                    errors = validate_full_preflight(config, full_state,
                        mode == Mode::Classify ? &config_digest : nullptr);
                if (errors.empty())
                {
                    admitted_digest.update(framed.data(), framed.size());
                    ++admitted_count;
                    ++partitions[partition];
                    if (emit)
                    {
                        if (signed_input)
                            std::cout << signature << '\t';
                        std::cout << line << '\n';
                    }
                }
                else
                    ++rejection_reasons[errors.front()];
            }
            else if (!full)
                ++rejection_reasons[errors.front()];
            if (mode == Mode::Classify)
            {
                std::cout << "{\"tuple\":";
                json_string(line);
                std::cout << ",\"partition\":";
                json_string(partition);
                std::cout << ",\"admissionStatus\":";
                json_string(!science_admitted ? "science_rejected" :
                            errors.empty() ? "preflight_admitted" : "preflight_rejected");
                std::cout << ",\"reason\":";
                json_string(errors.empty() ? "" : errors.front());
                std::cout << ",\"configSha256\":";
                json_string(config_digest);
                std::cout << ",\"soundingPath\":";
                json_string(config.initial_conditions.sounding.file_path);
                std::cout << "}\n";
            }
        }
        if (input_count != expected_count || input_digest.hex_digest() != expected_digest)
            throw std::runtime_error("native admission input tuple identity drifted");
        if (emit || mode == Mode::Classify)
            return 0;
        std::cout << "{\"schema\":\""
                  << (full ? "nimbus.native-full-preflight-batch.v1" : "nimbus.native-admission-batch.v1")
                  << "\",\"version\":1,";
        std::cout << "\"summary\":{\"input\":" << input_count;
        if (full)
            std::cout << ",\"scienceAdmitted\":" << science_admitted_count
                      << ",\"scienceRejected\":" << input_count - science_admitted_count
                      << ",\"preflightAdmitted\":" << admitted_count
                      << ",\"preflightRejected\":" << science_admitted_count - admitted_count;
        else
            std::cout << ",\"admitted\":" << admitted_count << ",\"rejected\":" << input_count - admitted_count;
        std::cout << ",\"rejectionReasons\":" << rejection_reasons.size() << "}";
        if (full)
        {
            std::cout << ",\"checks\":{\"serialization\":" << full_state.serialization_checks
                      << ",\"projection\":" << full_state.projection_checks
                      << ",\"compatibility\":" << full_state.compatibility_checks
                      << ",\"sounding\":" << full_state.sounding_checks
                      << ",\"soundingMaterializations\":" << full_state.sounding_materializations
                      << ",\"soundingCacheHits\":" << full_state.sounding_cache_hits
                      << ",\"lifetime\":" << full_state.lifetime_checks << "}";
        }
        std::cout << ",\"inputTupleSha256\":\"" << input_digest.hex_digest()
                  << "\",\"admittedTupleSha256\":\"" << admitted_digest.hex_digest()
                  << "\",\"partitions\":{";
        bool first = true;
        for (const auto& [name, count] : partitions)
        {
            if (!first)
                std::cout << ',';
            first = false;
            json_string(name);
            std::cout << ':' << count;
        }
        std::cout << "},\"rejections\":[";
        first = true;
        for (const auto& [reason, count] : rejection_reasons)
        {
            if (!first)
                std::cout << ',';
            first = false;
            std::cout << "{\"reason\":";
            json_string(reason);
            std::cout << ",\"count\":" << count << '}';
        }
        std::cout << "]";
        if (full)
        {
            std::cout << ",\"warnings\":[";
            first = true;
            for (const auto& [warning, count] : full_state.warnings)
            {
                if (!first)
                    std::cout << ',';
                first = false;
                std::cout << "{\"warning\":";
                json_string(warning);
                std::cout << ",\"count\":" << count << '}';
            }
            std::cout << ']';
        }
        std::cout << "}\n";
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "native admission batch failed: " << error.what() << '\n';
        return 1;
    }
}
