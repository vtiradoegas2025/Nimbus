#pragma once

#include <filesystem>
#include <string>

namespace nimbus::output
{

inline constexpr const char* kHistoryIndexFilename = "history_index.json";

bool prepare_history_index_update(const std::filesystem::path& series_directory,
                                  const std::string&           manifest_json,
                                  std::string&                 error);

// Rebuild and atomically publish the per-series history index after a complete
// step manifest becomes durable. Markerless test/legacy manifests are ignored.
bool publish_history_index_after_manifest(const std::filesystem::path& series_directory,
                                          const std::string&           manifest_json,
                                          std::string&                 error);

// Rebuild an existing index after retention. A series without an index remains
// compatible with pre-index output and is left unchanged.
bool rebuild_history_index_if_present(const std::filesystem::path& series_directory, std::string& error);

} // namespace nimbus::output
