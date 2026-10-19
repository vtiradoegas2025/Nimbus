#pragma once

#include "init/config/config_v1.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <istream>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace tmv::runtime
{

inline constexpr std::size_t kValidateOnlySourceLimitBytes = 8 * 1024 * 1024;
inline constexpr std::size_t kValidateOnlyContextLimitBytes = 64 * 1024;

inline std::string read_validate_only_source_bytes(std::istream& input)
{
    std::string bytes;
    char chunk[8192];
    while (input)
    {
        input.read(chunk, sizeof(chunk));
        const auto count = static_cast<std::size_t>(input.gcount());
        if (count > kValidateOnlySourceLimitBytes - bytes.size())
        {
            throw std::invalid_argument("preflight context source exceeds 8 MiB");
        }
        bytes.append(chunk, count);
    }
    if (input.bad() || !input.eof())
    {
        throw std::runtime_error("preflight context source read failed");
    }
    return bytes;
}

namespace validate_only_detail
{

class JsonObject
{
public:
    explicit JsonObject(std::ostringstream& output) : output_(output)
    {
        space(1);
        output_ << '{';
    }

    ~JsonObject()
    {
        if (output_.good() && output_.tellp() < static_cast<std::streamoff>(kValidateOnlyContextLimitBytes))
        {
            output_ << '}';
        }
        else
        {
            output_.setstate(std::ios::badbit);
        }
    }

    void field(std::string_view key, std::string_view value)
    {
        prefix(key);
        quote(value);
    }

    void field(std::string_view key, const char* value)
    {
        field(key, std::string_view(value));
    }

    void field(std::string_view key, bool value)
    {
        prefix(key);
        space(5);
        output_ << (value ? "true" : "false");
    }

    void field(std::string_view key, int value)
    {
        prefix(key);
        space(32);
        output_ << value;
    }

    void field(std::string_view key, double value)
    {
        if (!std::isfinite(value))
        {
            throw std::invalid_argument("preflight context contains a nonfinite number");
        }
        prefix(key);
        space(64);
        output_ << value;
    }

    template<typename Writer>
    void object(std::string_view key, Writer writer)
    {
        prefix(key);
        JsonObject child(output_);
        writer(child);
    }

    void strings(std::string_view key, const std::vector<std::string>& values)
    {
        if (values.size() > 512)
        {
            throw std::invalid_argument("preflight context string inventory exceeds its bound");
        }
        prefix(key);
        space(1);
        output_ << '[';
        bool first = true;
        for (const auto& value : values)
        {
            if (!first)
            {
                space(1);
                output_ << ',';
            }
            first = false;
            quote(value);
        }
        space(1);
        output_ << ']';
    }

private:
    void space(std::size_t required)
    {
        if (!output_.good() || output_.tellp() < 0)
        {
            throw std::runtime_error("preflight context serialization stream failed");
        }
        const auto current = static_cast<std::size_t>(output_.tellp());
        if (current > kValidateOnlyContextLimitBytes || required > kValidateOnlyContextLimitBytes - current)
        {
            throw std::invalid_argument("preflight context exceeds 64 KiB");
        }
    }

    void prefix(std::string_view key)
    {
        space(1);
        if (!first_)
        {
            output_ << ',';
        }
        first_ = false;
        quote(key);
        space(1);
        output_ << ':';
    }

    void quote(std::string_view text)
    {
        if (text.size() > 8192)
        {
            throw std::invalid_argument("preflight context string exceeds its bound");
        }
        std::size_t escaped = 2;
        for (const unsigned char ch : text)
        {
            const std::size_t added = ch < 0x20 ? 6 : (ch == '"' || ch == '\\' ? 2 : 1);
            if (added > kValidateOnlyContextLimitBytes - escaped)
            {
                throw std::invalid_argument("preflight context exceeds 64 KiB");
            }
            escaped += added;
        }
        space(escaped);
        constexpr char digits[] = "0123456789abcdef";
        output_ << '"';
        for (const unsigned char ch : text)
        {
            if (ch == '"' || ch == '\\')
            {
                output_ << '\\' << static_cast<char>(ch);
            }
            else if (ch < 0x20)
            {
                output_ << "\\u00" << digits[ch >> 4] << digits[ch & 15];
            }
            else
            {
                output_ << static_cast<char>(ch);
            }
        }
        output_ << '"';
    }

    std::ostringstream& output_;
    bool first_ = true;
};

}

inline std::string serialize_validate_only_context_v1(
    const tmv::init::config::ConfigV1& config,
    std::string_view source_sha256,
    std::string_view generated_metal_scope_kind,
    std::vector<std::string> state_fields)
{
    namespace cfg = tmv::init::config;
    if (source_sha256.size() != 64 || !std::all_of(source_sha256.begin(), source_sha256.end(), [](char ch)
        {
            return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
        }) || (generated_metal_scope_kind != "none" && generated_metal_scope_kind != "canonical" &&
               generated_metal_scope_kind != "east_npy" && generated_metal_scope_kind != "mixed_npy"))
    {
        throw std::invalid_argument("preflight context identity or scope kind is invalid");
    }
    std::sort(state_fields.begin(), state_fields.end());
    if (std::adjacent_find(state_fields.begin(), state_fields.end()) != state_fields.end())
    {
        throw std::invalid_argument("preflight context state inventory contains duplicates");
    }
    const auto& compute = config.extensions.compute;
    const auto& output = config.output;
    const auto& lateral = config.boundary_conditions.lateral;
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << std::setprecision(std::numeric_limits<double>::max_digits10);
    {
        validate_only_detail::JsonObject receipt(stream);
        receipt.field("schema", "nimbus.validate-only-context.v1");
        receipt.field("version", 1);
        receipt.field("sourceKind", "exact_document");
        receipt.field("sourceSha256", source_sha256);
        receipt.field("generatedMetalScopeKind", generated_metal_scope_kind);
        receipt.object("context", [&](auto& context)
        {
            context.field("executionMode", cfg::execution_mode_name(config.extensions.execution.mode));
            context.field("maturityLimitsAcknowledged", config.extensions.execution.mode == cfg::ExecutionMode::Qualified ||
                config.extensions.execution.acknowledge_maturity_limits);
            context.field("backend", cfg::compute_backend_name(compute.backend));
            context.field("allowBackendFallback", compute.allow_fallback);
            context.field("validationGuardMode", cfg::validation_guard_mode_name(config.extensions.validation.guard_mode));
            context.field("gpuApplyMicrophysics", compute.gpu_apply_microphysics);
            context.field("gpuApplySlow", compute.gpu_apply_slow);
            context.field("gpuApplySgs", compute.gpu_apply_sgs);
            context.field("outputFormat", cfg::output_format_name(output.format));
            context.field("coordinateSystem", coordinate_system_name(config.grid.coordinate_system));
            context.field("staggering", stagger_type_name(config.grid.staggering));
            context.object("gridMetrics", [&](auto& grid)
            {
                grid.field("nx", config.grid.nx);
                grid.field("ny", config.grid.ny);
                grid.field("nz", config.grid.nz);
                grid.field("dx", config.grid.dx);
                grid.field("dy", config.grid.dy);
                grid.field("dz", config.grid.dz);
                grid.field("ztop", config.grid.ztop);
            });
            context.object("timeControl", [&](auto& time)
            {
                time.field("dtLarge", config.time.dt_large);
                time.field("acousticSubsteps", config.time.acoustic_substeps);
                time.field("outputInterval", config.time.output_interval);
            });
            context.object("output", [&](auto& fields)
            {
                fields.field("preset", cfg::output_preset_name(output.preset));
                fields.field("asyncIo", output.async_io);
                fields.field("zfpMode", output.zfp_mode);
                fields.field("zfpPerFieldTolerances", output.zfp_per_field_tolerances);
                fields.field("zfpSparseThreshold", output.zfp_sparse_threshold);
                fields.field("zfpFloat16Prefilter", output.zfp_float16_prefilter);
                fields.field("zfpFieldExecution", output.zfp_field_execution);
                fields.field("zfpFieldWorkers", output.zfp_field_workers);
                fields.field("zfpFieldScratchMib", output.zfp_field_scratch_mib);
                fields.field("keepLastFrames", output.keep_last_frames);
                fields.field("minimumFreeSpaceMib", output.minimum_free_space_mib);
                fields.field("maximumRetainedHistoryFiles", output.maximum_retained_history_files);
                fields.object("zfpFieldTolerances", [&](auto& tolerances)
                {
                    std::vector<std::pair<std::string, double>> sorted(output.zfp_field_tolerances.begin(),
                                                                     output.zfp_field_tolerances.end());
                    if (sorted.size() > 256)
                    {
                        throw std::invalid_argument("preflight context tolerance inventory exceeds its bound");
                    }
                    std::sort(sorted.begin(), sorted.end());
                    for (const auto& [key, value] : sorted)
                    {
                        tolerances.field(key, value);
                    }
                });
            });
            context.object("boundary", [&](auto& boundary)
            {
                boundary.field("west", cfg::lateral_bc_name(lateral.west));
                boundary.field("east", cfg::lateral_bc_name(lateral.east));
                boundary.field("south", cfg::lateral_bc_name(lateral.south));
                boundary.field("north", cfg::lateral_bc_name(lateral.north));
                boundary.field("bottom", cfg::bottom_bc_name(config.boundary_conditions.bottom));
                boundary.field("top", cfg::top_bc_name(config.boundary_conditions.top));
            });
            context.object("diagnostics", [&](auto& diagnostics)
            {
                diagnostics.field("scientificEverySteps", config.extensions.diagnostics.scientific_every_steps);
                diagnostics.field("deterministicReductions", config.extensions.diagnostics.deterministic_reductions);
            });
            context.object("initialization", [&](auto& initialization)
            {
                initialization.field("soundingSource", cfg::sounding_type_name(config.initial_conditions.sounding.type));
                initialization.field("hodographSource", cfg::hodograph_type_name(config.initial_conditions.hodograph.type));
                initialization.field("triggerSource", cfg::trigger_type_name(config.initial_conditions.trigger.type));
            });
            context.object("interactions", [&](auto& interactions)
            {
                interactions.field("radiationShortwaveEnabled", config.radiation.do_sw);
                interactions.field("turbulenceMode", cfg::turbulence_mode_name(config.turbulence.mode));
                interactions.field("turbulenceDynamicCoefficient", config.turbulence.dynamic_coefficient);
                interactions.field("surfaceScheme", cfg::surface_scheme_name(config.surface.scheme));
                interactions.field("surfaceApplyFluxes", config.surface.apply_fluxes);
                interactions.field("diffusionTarget", cfg::diffusion_target_name(config.diffusion.apply_to));
                interactions.field("diffusionActive", config.diffusion.K_h > 0.0 || config.diffusion.K_v > 0.0);
            });
            context.strings("availableStateFields", state_fields);
        });
        receipt.object("selectedSchemes", [&](auto& selections)
        {
            const auto dynamics = config.dynamics.scheme == "klemp_wilhelmson_split"
                ? (config.grid.staggering == StaggerType::CGrid ? "klemp_wilhelmson_split_cartesian_cgrid"
                                                               : "klemp_wilhelmson_split_cartesian_agrid")
                : config.dynamics.scheme;
            selections.field("dynamics", dynamics);
            selections.field("microphysics", cfg::microphysics_scheme_name(config.microphysics.scheme));
            selections.field("radiation", cfg::radiation_scheme_name(config.radiation.scheme));
            selections.field("boundary_layer", cfg::boundary_layer_scheme_name(config.boundary_layer.scheme));
            selections.field("turbulence", cfg::turbulence_scheme_name(config.turbulence.scheme));
            selections.field("advection", cfg::resolved_advection_scheme_name(config.advection.scheme, config.grid.staggering));
            selections.field("diffusion", cfg::diffusion_scheme_name(config.diffusion.scheme));
            selections.field("time_stepping", cfg::time_integrator_name(config.time.integrator));
            selections.field("chaos", cfg::chaos_scheme_name(config.extensions.chaos.scheme));
            selections.field("terrain", cfg::terrain_scheme_name(config.terrain.scheme));
            if (config.initial_conditions.sounding.type == cfg::SoundingType::File)
            {
                selections.field("soundings", "auto");
            }
        });
        receipt.strings("radarProducts", config.radar.products);
        receipt.object("unserializedControls", [&](auto& controls)
        {
            controls.field("lateralSpongeEnabled", config.damping.lateral_sponge_enabled);
            controls.field("lateralWidth", config.damping.lateral_width);
            controls.field("lateralAlpha", config.damping.lateral_alpha);
            controls.field("wOnlyTopSponge", config.damping.w_only_top_sponge);
            controls.field("wOnlyDepthCells", config.damping.w_only_depth_cells);
            controls.field("wOnlyAlphaMax", config.damping.w_only_alpha_max);
            controls.field("slowRk2", config.extensions.solver.slow_rk2);
            controls.field("lidBcGamma", config.extensions.solver.lid_bc_gamma);
            controls.field("axisUM1Filter", config.extensions.solver.axis_u_m1_filter);
        });
    }
    if (!stream.good())
    {
        throw std::runtime_error("preflight context serialization stream failed");
    }
    const std::string result = stream.str();
    if (result.size() > kValidateOnlyContextLimitBytes)
    {
        throw std::invalid_argument("preflight context exceeds 64 KiB");
    }
    return result;
}

}
