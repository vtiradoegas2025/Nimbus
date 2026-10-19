#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "boundary_layer/runtime.hpp"

namespace tmv::checkpoint
{

inline constexpr std::uint32_t boundary_layer_checkpoint_version = 5;

// Serializes persistent boundary-layer state into the portable v5 format.
bool serialize_boundary_layer_runtime(const BoundaryLayerRuntimeState& state,
                                      std::vector<std::uint8_t>&       bytes,
                                      std::string&                     error);

// Restores v1-v5 only after the complete payload validates. V1 reconstructs
// the nominal deadline; pre-v3 active states force a conserved-source refresh.
bool deserialize_boundary_layer_runtime(const std::vector<std::uint8_t>& bytes,
                                        BoundaryLayerRuntimeState&       state,
                                        std::string&                     error);

// Publishes through a temporary file and atomic rename.
bool write_boundary_layer_checkpoint(const std::filesystem::path&     path,
                                     const BoundaryLayerRuntimeState& state,
                                     std::string&                     error);

// Reads and transactionally restores a portable checkpoint.
bool read_boundary_layer_checkpoint(const std::filesystem::path& path,
                                    BoundaryLayerRuntimeState&   state,
                                    std::string&                 error);

} // namespace tmv::checkpoint
