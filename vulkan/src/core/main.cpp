/**
 * @file main.cpp
 * @brief Command-line entrypoint for the Vulkan viewer utilities.
 *
 * Parses runtime options, validates user-provided numeric ranges, and dispatches
 * to either dataset validation or Vulkan execution paths. This file owns argument
 * hygiene so rendering modules can assume sanitized configuration values.
 */

#include "core/app.hpp"
#include "data/export_dataset.hpp"

#include "util/string_utils.hpp"

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace 
{


/**
 * @brief Parse a bounded signed integer argument from CLI text.
 */
bool parse_int_arg(const char* value, long min_value, long max_value, long& out_value) {
    if (value == nullptr) {return false;}

    errno = 0;
    char* end = nullptr;
    const long parsed = std::strtol(value, &end, 10);

    if (end == value || *end != '\0' || errno == ERANGE) {return false;}

    if (parsed < min_value || parsed > max_value) {return false;}

    out_value = parsed;
    return true;
}

/**
 * @brief Parse a bounded finite floating-point argument from CLI text.
 */
bool parse_float_arg(const char* value, float min_value, float max_value, float& out_value) {
    if (value == nullptr) {return false;}

    errno = 0;
    char* end = nullptr;
    const float parsed = std::strtof(value, &end);
    if (end == value || *end != '\0' || errno == ERANGE || !std::isfinite(parsed)) {return false;}

    if (parsed < min_value || parsed > max_value) {return false;}

    out_value = parsed;
    return true;
}

/**
 * @brief Parse a comma-separated 3D vector and reject near-zero vectors.
 */
bool parse_vec3_csv(const std::string& text, float& x, float& y, float& z) 
{
    std::vector<std::string> parts;
    std::size_t start = 0;

    while (start <= text.size()) 
    {
        const std::size_t comma = text.find(',', start);
        if (comma == std::string::npos) 
        {
            parts.push_back(tmv::strutil::trim_copy(text.substr(start)));
            break;
        }

        parts.push_back(tmv::strutil::trim_copy(text.substr(start, comma - start)));
        start = comma + 1;
    }

    if (parts.size() != 3) {return false;}

    if (!parse_float_arg(parts[0].c_str(), -1000.0f, 1000.0f, x) ||
        !parse_float_arg(parts[1].c_str(), -1000.0f, 1000.0f, y) ||
        !parse_float_arg(parts[2].c_str(), -1000.0f, 1000.0f, z)) 
    {
        return false;
    }

    const float magnitude_sq = x * x + y * y + z * z;
    return magnitude_sq > 1e-10f;
}

/**
 * @brief Split a comma-separated field list and drop empty tokens.
 */
std::vector<std::string> split_csv_fields(const std::string& csv) 
{
    std::vector<std::string> out;
    std::size_t start = 0;

    while (start <= csv.size()) 
    {
        const std::size_t comma = csv.find(',', start);
        if (comma == std::string::npos) 
        {
            const std::string token = tmv::strutil::trim_copy(csv.substr(start));
            if (!token.empty()) {out.push_back(token);}
            break;
        }

        const std::string token = tmv::strutil::trim_copy(csv.substr(start, comma - start));
        if (!token.empty()) {out.push_back(token);}
        start = comma + 1;
    }

    return out;
}


/**
 * @brief Normalize and validate supported volume rendering mode tokens.
 */
bool parse_volume_mode_arg(const std::string& value, std::string& normalized_mode) 
{
    const std::string mode = tmv::strutil::trim_and_lower(value);
    if (mode == "supercell" || mode == "composite" || mode == "isolated" || mode == "cycle") 
    {
        normalized_mode = mode;
        return true;
    }
    if (mode == "single" || mode == "isolate") 
    {
        normalized_mode = "isolated";
        return true;
    }
    if (mode == "together" || mode == "blend")
    {
        normalized_mode = "composite";
        return true;
    }
    if (mode == "animate") 
    {
        normalized_mode = "cycle";
        return true;
    }
    return false;
}

/**
 * @brief Normalize and validate supported texture profile tokens.
 */
bool parse_texture_mode_arg(const std::string& value, std::string& normalized_mode) 
{
    const std::string mode = tmv::strutil::trim_and_lower(value);
    if (mode == "natural" || mode == "smooth") 
    {
        normalized_mode = mode;
        return true;
    }
    if (mode == "realistic" || mode == "detailed" || mode == "detail") 
    {
        normalized_mode = "natural";
        return true;
    }
    if (mode == "linear" || mode == "soft") 
    {
        normalized_mode = "smooth";
        return true;
    }
    return false;
}

/**
 * @brief Normalize and validate supported camera rig modes.
 */
bool parse_camera_mode_arg(const std::string& value, std::string& normalized_mode) 
{
    const std::string mode = tmv::strutil::trim_and_lower(value);
    if (mode == "orbit" || mode == "fixed" || mode == "freefly") 
    {
        normalized_mode = mode;
        return true;
    }
    if (mode == "turntable" || mode == "spin") 
    {
        normalized_mode = "orbit";
        return true;
    }
    if (mode == "static") 
    {
        normalized_mode = "fixed";
        return true;
    }
    if (mode == "free" || mode == "fly" || mode == "firstperson" || mode == "first-person") 
    {
        normalized_mode = "freefly";
        return true;
    }
    return false;
}

/**
 * @brief Build alias candidates for known atmospheric field names.
 */
std::vector<std::string> field_alias_candidates(const std::string& requested_field) 
{
    std::vector<std::string> candidates;
    std::unordered_set<std::string> seen;

    const auto push_unique = [&](const std::string& name) 
    {
        if (name.empty()) 
        {
            return;
        }
        if (seen.insert(name).second) 
        {
            candidates.push_back(name);
        }
    };

    push_unique(requested_field);

    if (requested_field == "qr") 
    {
        push_unique("qrain");
        push_unique("rain");
    } 
    else if (requested_field == "qs") 
    {
        push_unique("q_snow");
        push_unique("snow");
        push_unique("qice");
    } 
    else if (requested_field == "qg") 
    {
        push_unique("q_graupel");
        push_unique("graupel");
        push_unique("qh");
        push_unique("hail");
    } 
    else if (requested_field == "qi") 
    {
        push_unique("q_ice");
        push_unique("ice");
        push_unique("qice");
        push_unique("qh");
        push_unique("hail");
    } 
    else if (requested_field == "qc") 
    {
        push_unique("q_cloud");
        push_unique("cloud");
    } 
    else if (requested_field == "w") 
    {
        push_unique("wz");
        push_unique("vertical_velocity");
        push_unique("updraft");
    } 
    else if (requested_field == "theta") 
    {
        push_unique("pt");
        push_unique("potential_temperature");
        push_unique("temperature");
    } 
    else if (requested_field == "vorticity" || requested_field == "zeta" || requested_field == "vertical_vorticity") 
    {
        push_unique("vorticity_z");
        push_unique("vorticity_magnitude");
        push_unique("omega");
    } 
    else if (requested_field == "vorticity_z") 
    {
        push_unique("zeta");
        push_unique("vorticity");
        push_unique("vertical_vorticity");
    }
    else if (requested_field == "vorticity_magnitude") 
    {
        push_unique("vorticity");
        push_unique("zeta");
        push_unique("vorticity_z");
    }

    return candidates;
}

/**
 * @brief Validate all requested input datasets without creating Vulkan resources.
 */
int run_input_validation(const vkcpp::Options& options) 
{
    std::vector<std::string> requested_fields;
    if (!options.fields_csv.empty()) 
    {
        requested_fields = split_csv_fields(options.fields_csv);
    } 
    else 
    {
        requested_fields.push_back(options.field);
    }

    if (requested_fields.empty()) 
    {
        std::cerr << "[vulkan][validate-input] no fields specified\n";
        return 1;
    }

    bool success = true;
    for (const std::string& raw_field : requested_fields) 
    {
        const std::string field = tmv::strutil::trim_and_lower(raw_field);
        if (field.empty()) 
        {
            std::cerr << "[vulkan][validate-input] empty field token encountered\n";
            success = false;
            continue;
        }

        std::string resolved_field;
        std::string scan_error;
        const std::vector<std::string> candidates = field_alias_candidates(field);
        for (const std::string& candidate : candidates) 
        {
            oglcpp::ExportDataset probe(options.input_dir, candidate);
            if (probe.scan(scan_error)) 
            {
                resolved_field = candidate;
                break;
            }
        }

        if (resolved_field.empty()) 
        {
            std::cerr << "[vulkan][validate-input] field '" << field
                      << "' scan failed: " << scan_error << "\n";
            success = false;
            continue;
        }

        if (resolved_field != field) 
        {
            std::cout << "[vulkan][validate-input] field alias: '" << field
                      << "' -> '" << resolved_field << "'\n";
        }

        oglcpp::ExportDataset dataset(options.input_dir, resolved_field);
        if (!dataset.scan(scan_error)) 
        {
            std::cerr << "[vulkan][validate-input] field '" << field
                      << "' scan failed after alias resolution: " << scan_error << "\n";
            success = false;
            continue;
        }

        std::size_t total_nan = 0;
        std::size_t total_inf = 0;
        std::size_t total_sanitized = 0;
        bool load_ok = true;

        for (std::size_t frame = 0; frame < dataset.frame_count(); ++frame) 
        {
            oglcpp::VolumeFrame volume;
            std::string load_error;

            if (!dataset.load_frame(frame, volume, load_error)) 
            {
                std::cerr << "[vulkan][validate-input] field '" << field
                          << "' (resolved as '" << resolved_field << "') frame " << frame
                          << " load failed: " << load_error << "\n";
                success = false;
                load_ok = false;
                break;
            }

            total_nan += volume.nan_count;
            total_inf += volume.inf_count;
            total_sanitized += volume.sanitized_nonfinite_count;
        }

        if (!load_ok) {continue;}

        std::cout << "[vulkan][validate-input] field=" << field
                  << " resolved=" << resolved_field
                  << " frames=" << dataset.frame_count()
                  << " dims=" << dataset.nx() << "x" << dataset.ny() << "x" << dataset.nz()
                  << " nan=" << total_nan
                  << " inf=" << total_inf
                  << " sanitized=" << total_sanitized
                  << "\n";

        if (total_nan + total_inf > 0) {success = false;}
    }

    return success ? 0 : 1;
}

/**
 * @brief Print CLI usage and supported runtime options.
 */
void print_help() 
{
    std::cout << "Vulkan Viewer Scaffold\n"
              << "Usage:\n"
              << "  bin/vulkan_viewer [options]\n\n"
              << "Options:\n"
              << "  --validation      Enable VK_LAYER_KHRONOS_validation\n"
              << "  --list-devices    Print discovered Vulkan devices and chosen target\n"
              << "  --device-index    Select a specific physical device index\n"
              << "  --window-test     Run window + swapchain render loop test\n"
              << "  --window-width    Window width for --window-test (default: 1280)\n"
              << "  --window-height   Window height for --window-test (default: 720)\n"
              << "  --window-frames   Frame count for --window-test (default: 0 = unlimited)\n"
              << "  --render-backend  Render backend for window mode (default: clear)\n"
              << "  --input           Input export directory for volume backend (default: data/exports)\n"
              << "  --field           Single field name for volume backend (default: theta)\n"
              << "  --fields          Comma-separated fields for multi-field volume rendering\n"
              << "  --volume-mode     supercell|composite|isolated|cycle (default: supercell)\n"
              << "  --isolate-field   Field to render in isolated/cycle-start mode (default: first resolved)\n"
              << "  --component-cycle-fps Cycle speed for --volume-mode cycle (default: 0.50)\n"
              << "  --texture-mode    smooth|natural (default: natural)\n"
              << "  --camera-mode     fixed|orbit|freefly (default: orbit)\n"
              << "  --camera-orbit-fps Camera orbits/sec in orbit mode (default: 0.02)\n"
              << "  --camera-distance Camera distance from storm center (default: 2.25)\n"
              << "  --camera-height   Camera Y height above domain center (default: 0.85)\n"
              << "  --camera-fov-deg  Camera field-of-view in degrees (default: 55.0)\n"
              << "  --style           Volume style: default|cinematic-bw (default: default)\n"
              << "  --playback-fps    Frame playback speed for volume backend (default: 1.0)\n"
              << "  --ray-steps       Raymarch step count (default: 192)\n"
              << "  --ray-threshold   Density threshold in [0,1] (default: 0.30)\n"
              << "  --ray-opacity     Extinction/opacity scale (default: 1.15)\n"
              << "  --ray-brightness  In-scatter brightness scale (default: 1.10)\n"
              << "  --ray-ambient     Ambient sky fill scale (default: 0.90)\n"
              << "  --ray-anisotropy  HG anisotropy in [-0.85,0.85] (default: 0.58)\n"
              << "  --ray-max-distance Max ray distance through volume (default: 5.0)\n"
              << "  --sun-dir         Comma-separated sun direction x,y,z (default: 0.66,0.34,0.67)\n"
              << "  --validate-input  Validate export datasets without creating Vulkan instance\n"
              << "  --dry-run         Initialize Vulkan and exit\n"
              << "  --help            Show this help\n";
}

}  // namespace

/**
 * @brief Program entrypoint for Vulkan viewer tooling.
 */
int main(int argc, char** argv) 
{
    vkcpp::Options options;

    for (int i = 1; i < argc; ++i) 
    {
        const std::string arg = argv[i];

        auto require_value = [&](const std::string& option) -> const char* 
        {
            if (i + 1 >= argc) 
            {
                std::cerr << "Missing value for " << option << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--help" || arg == "-h") {print_help(); return 0;}

        if (arg == "--validation") {options.enable_validation = true; continue;}

        if (arg == "--list-devices") {options.list_devices = true; continue;}

        if (arg == "--device-index") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            long parsed = 0;
            if (!parse_int_arg(value, 0, std::numeric_limits<int>::max(), parsed)) 
            {
                std::cerr << "Invalid --device-index value: " << value << "\n";
                return 1;
            }

            options.preferred_device_index = static_cast<int>(parsed);
            continue;
        }
        if (arg == "--dry-run") {options.dry_run = true; continue;}

        if (arg == "--validate-input") {options.validate_input = true; continue;}

        if (arg == "--window-test") {options.window_test = true; continue;}

        if (arg == "--window-width") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            long parsed = 0;
            if (!parse_int_arg(value, 1, std::numeric_limits<int>::max(), parsed)) 
            {
                std::cerr << "Invalid --window-width value: " << value << "\n";
                return 1;
            }

            options.window_width = static_cast<uint32_t>(parsed);
            continue;
        }
        if (arg == "--window-height") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            long parsed = 0;
            if (!parse_int_arg(value, 1, std::numeric_limits<int>::max(), parsed)) 
            {
                std::cerr << "Invalid --window-height value: " << value << "\n";
                return 1;
            }

            options.window_height = static_cast<uint32_t>(parsed);
            continue;
        }
        if (arg == "--window-frames") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            long parsed = 0;
            if (!parse_int_arg(value, 0, std::numeric_limits<int>::max(), parsed)) 
            {
                std::cerr << "Invalid --window-frames value: " << value << " (expected >= 0)\n";
                return 1;
            }

            options.window_frames = static_cast<int>(parsed);
            continue;
        }
        if (arg == "--render-backend") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            options.render_backend = value;
            if (options.render_backend.empty()) 
            {
                std::cerr << "Invalid --render-backend value\n";
                return 1;
            }

            std::transform(options.render_backend.begin(),
                           options.render_backend.end(),
                           options.render_backend.begin(),
                           [](unsigned char c) {
                               return static_cast<char>(std::tolower(c));
                           });
            continue;
        }
        if (arg == "--input") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            options.input_dir = value;
            if (options.input_dir.empty()) 
            {
                std::cerr << "Invalid --input value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--field") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            options.field = value;

            if (options.field.empty()) 
            {
                std::cerr << "Invalid --field value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--fields") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            options.fields_csv = tmv::strutil::trim_copy(value);

            if (options.fields_csv.empty()) 
            {
                std::cerr << "Invalid --fields value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--volume-mode") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            std::string normalized_mode;
            if (!parse_volume_mode_arg(value, normalized_mode)) 
            {
                std::cerr << "Invalid --volume-mode value: " << value
                          << " (expected supercell|composite|isolated|cycle)\n";
                return 1;
            }
            options.volume_mode = normalized_mode;
            continue;
        }
        if (arg == "--isolate-field") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            options.isolate_field = tmv::strutil::trim_and_lower(value);
            if (options.isolate_field.empty()) 
            {
                std::cerr << "Invalid --isolate-field value\n";
                return 1;
            }
            continue;
        }
        if (arg == "--component-cycle-fps") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            if (!parse_float_arg(value, 0.01f, 30.0f, options.component_cycle_fps)) 
            {
                std::cerr << "Invalid --component-cycle-fps value: " << value << " (expected 0.01..30)\n";
                return 1;
            }
            continue;
        }
        if (arg == "--texture-mode") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            std::string normalized_mode;
            if (!parse_texture_mode_arg(value, normalized_mode)) 
            {
                std::cerr << "Invalid --texture-mode value: " << value
                          << " (expected smooth|natural)\n";
                return 1;
            }
            options.texture_mode = normalized_mode;
            continue;
        }
        if (arg == "--camera-mode") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            std::string normalized_mode;
            if (!parse_camera_mode_arg(value, normalized_mode)) 
            {
                std::cerr << "Invalid --camera-mode value: " << value
                          << " (expected fixed|orbit|freefly)\n";
                return 1;
            }
            options.camera_mode = normalized_mode;
            continue;
        }
        if (arg == "--camera-orbit-fps") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            if (!parse_float_arg(value, 0.0f, 2.0f, options.camera_orbit_fps)) 
            {
                std::cerr << "Invalid --camera-orbit-fps value: " << value << " (expected 0..2)\n";
                return 1;
            }
            continue;
        }
        if (arg == "--camera-distance") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            if (!parse_float_arg(value, 0.80f, 20.0f, options.camera_distance)) {
                std::cerr << "Invalid --camera-distance value: " << value << " (expected 0.8..20)\n";
                return 1;
            }
            continue;
        }
        if (arg == "--camera-height") {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            if (!parse_float_arg(value, -2.0f, 5.0f, options.camera_height)) 
            {
                std::cerr << "Invalid --camera-height value: " << value << " (expected -2..5)\n";
                return 1;
            }
            continue;
        }
        if (arg == "--camera-fov" || arg == "--camera-fov-deg") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            if (!parse_float_arg(value, 20.0f, 110.0f, options.camera_fov_deg)) {
                std::cerr << "Invalid --camera-fov-deg value: " << value << " (expected 20..110)\n";
                return 1;
            }
            continue;
        }
        if (arg == "--style") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            std::string style = tmv::strutil::trim_and_lower(value);
            if (style == "bw" || style == "cinematic") 
            {
                style = "cinematic-bw";
            }

            if (style != "default" && style != "cinematic-bw") 
            {
                std::cerr << "Invalid --style value: " << value
                          << " (expected default|cinematic-bw)\n";
                return 1;
            }
            options.style = style;
            continue;
        }
        if (arg == "--playback-fps") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            if (!parse_float_arg(value, 0.01f, 120.0f, options.playback_fps)) 
            {
                std::cerr << "Invalid --playback-fps value: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (arg == "--ray-steps") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            long parsed = 0;
            if (!parse_int_arg(value, 16, 512, parsed)) 
            {
                std::cerr << "Invalid --ray-steps value: " << value << " (expected 16..512)\n";
                return 1;
            }
            options.ray_steps = static_cast<int>(parsed);
            continue;
        }
        if (arg == "--ray-threshold") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            if (!parse_float_arg(value, 0.0f, 1.0f, options.ray_threshold)) 
            {
                std::cerr << "Invalid --ray-threshold value: " << value << " (expected 0..1)\n";
                return 1;
            }
            continue;
        }
        if (arg == "--ray-opacity") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            if (!parse_float_arg(value, 0.01f, 20.0f, options.ray_opacity)) 
            {
                std::cerr << "Invalid --ray-opacity value: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (arg == "--ray-brightness") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            if (!parse_float_arg(value, 0.01f, 20.0f, options.ray_brightness)) 
            {
                std::cerr << "Invalid --ray-brightness value: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (arg == "--ray-ambient") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            if (!parse_float_arg(value, 0.0f, 5.0f, options.ray_ambient)) 
            {
                std::cerr << "Invalid --ray-ambient value: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (arg == "--ray-anisotropy") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            if (!parse_float_arg(value, -0.85f, 0.85f, options.ray_anisotropy)) 
            {
                std::cerr << "Invalid --ray-anisotropy value: " << value << " (expected -0.85..0.85)\n";
                return 1;
            }
            continue;
        }
        if (arg == "--ray-max-distance") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            if (!parse_float_arg(value, 0.01f, 1000.0f, options.ray_max_distance)) 
            {
                std::cerr << "Invalid --ray-max-distance value: " << value << "\n";
                return 1;
            }
            continue;
        }
        if (arg == "--sun-dir") 
        {
            const char* value = require_value(arg);
            if (value == nullptr) {return 1;}

            if (!parse_vec3_csv(value, options.sun_dir_x, options.sun_dir_y, options.sun_dir_z)) 
            {
                std::cerr << "Invalid --sun-dir value: " << value << " (expected x,y,z)\n";
                return 1;
            }
            continue;
        }

        std::cerr << "Unknown argument: " << arg << "\n";
        print_help();
        return 1;
    }

    if (options.window_test && options.dry_run) {std::cout << "[vulkan] --window-test takes precedence over --dry-run\n";}

    if (options.validate_input) {return run_input_validation(options);}

    vkcpp::VulkanBootstrap app;
    return app.run(options);
}
