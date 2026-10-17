#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace nimbus::output
{

struct RetentionResult
{
    std::size_t removed_items = 0;
    std::uintmax_t reclaimed_bytes = 0;
    std::size_t dependency_frames = 0;
    std::size_t removed_geometry_versions = 0;
};

/**
 * Keep the newest completed step directories and every manifest-declared ZFP
 * ancestor needed to decode them. Zero disables pruning. Incomplete frames,
 * unexpected names, and symbolic links are never removed.
 */
bool enforce_step_retention(const std::filesystem::path& series_directory,
                            std::size_t                  keep_last_frames,
                            RetentionResult&             result,
                            std::string&                 error);

/**
 * Keep the newest complete checkpoint binary/sidecar pairs. The protected path
 * is retained even when it falls outside the numeric tail (for example, a
 * branch's source checkpoint). Zero disables pruning.
 */
bool enforce_checkpoint_retention(const std::filesystem::path& checkpoint_directory,
                                  std::size_t                  keep_last_checkpoints,
                                  const std::filesystem::path& protected_checkpoint,
                                  RetentionResult&             result,
                                  std::string&                 error);

/** Fail closed when the filesystem cannot preserve the configured reserve. */
bool require_free_space(const std::filesystem::path& destination,
                        std::uintmax_t               required_bytes,
                        std::string&                 error);

} // namespace nimbus::output
