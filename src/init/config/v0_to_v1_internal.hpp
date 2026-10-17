#pragma once

#include "core/runtime/yaml_helpers.hpp"
#include "init/config/v0_to_v1.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace tmv::init::config::detail
{

inline std::string to_lower(std::string value)
{
    std::transform(value.begin(),
                   value.end(),
                   value.begin(),
                   [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
    return value;
}

inline bool valid_runtime_seconds(double value, bool allow_zero)
{
    const bool sign_valid = allow_zero ? value >= 0.0 : value > 0.0;
    return sign_valid && std::trunc(value) == value && value <= static_cast<double>(std::numeric_limits<int>::max());
}

class V0ToV1Migrator
{
public:
    MigrationResult migrate(const std::string& v0_path);

private:
    std::string                                file_;
    std::unordered_map<std::string, YamlField> yaml_;
    std::unordered_set<std::string>            consumed_;
    std::vector<MigrationWarning>              warnings_;
    std::vector<LoadError>                     errors_;
    ConfigV1                                   config_;

    bool has(const std::string& key) const
    {
        return yaml_.count(key) > 0;
    }
    int line_of(const std::string& key) const
    {
        const auto it = yaml_.find(key);
        return (it != yaml_.end()) ? it->second.line_number : 0;
    }

    void warn_drop(const std::string& v0_key, const std::string& reason);
    void warn_rename(const std::string& v0_key, const std::string& v1_key);
    void warn_unit(const std::string& v0_key, const std::string& v1_key, const std::string& conversion);
    void warn_synth(const std::string& v0_key, const std::string& action, const std::string& reason);
    void error(int line, const std::string& key, const std::string& message);

    bool take_string(const std::string& v0_key, std::string& out);
    bool take_int(const std::string& v0_key, int& out);
    bool take_double(const std::string& v0_key, double& out);
    bool take_bool(const std::string& v0_key, bool& out);
    void consume_silent(const std::string& v0_key);

    void migrate_top_level();
    void migrate_grid_and_time();
    void migrate_environment();
    void migrate_sounding_subsection();
    void migrate_hodograph_subsection();
    void migrate_trigger();
    void migrate_dynamics();
    void migrate_microphysics();
    void migrate_radiation();
    void migrate_boundary_layer_and_surface();
    void migrate_turbulence();
    void migrate_numerics();
    void migrate_terrain();
    void migrate_output();
    void migrate_chaos();
    void migrate_validation();
    void migrate_metadata_sections();
    void report_unmapped_keys();
};

} // namespace tmv::init::config::detail
