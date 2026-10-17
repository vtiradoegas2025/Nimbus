#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "core/runtime/simulation_state.hpp"
#include "diagnostics/trajectories.hpp"

namespace tmv::checkpoint
{

inline constexpr std::uint32_t simulation_checkpoint_version = 2;

struct SimulationCheckpointAttachments
{
    std::optional<tmv::TrajectoryCheckpointState> trajectories;
};

struct SimulationCheckpointIntegrity
{
    std::uint64_t bytes = 0;
    std::string fnv1a64;
    std::string sha256;
};

// Full-run checkpoints are taken between completed model timesteps. Call-local
// kernel aliases and intermediate Runge-Kutta stages are reconstructed.
bool serialize_simulation_storage(const SimulationStorageState& state,
                                  std::vector<std::uint8_t>& bytes,
                                  std::string& error);
bool serialize_simulation_storage(const SimulationStorageState& state,
                                  const tmv::TrajectorySet* trajectories,
                                  std::vector<std::uint8_t>& bytes,
                                  std::string& error);
bool deserialize_simulation_storage(const std::vector<std::uint8_t>& bytes,
                                    SimulationStorageState& state,
                                    std::string& error);
bool deserialize_simulation_storage(const std::vector<std::uint8_t>& bytes,
                                    SimulationStorageState& state,
                                    SimulationCheckpointAttachments& attachments,
                                    std::string& error);
bool write_simulation_checkpoint(const std::filesystem::path& path,
                                 const SimulationStorageState& state,
                                 std::string& error);
bool write_simulation_checkpoint(const std::filesystem::path& path,
                                 const SimulationStorageState& state,
                                 const tmv::TrajectorySet* trajectories,
                                 std::string& error);
bool write_simulation_checkpoint(const std::filesystem::path& path,
                                 const SimulationStorageState& state,
                                 const tmv::TrajectorySet* trajectories,
                                 SimulationCheckpointIntegrity& integrity,
                                 std::string& error,
                                 std::uint64_t minimum_free_bytes_after_publication = 0);
bool read_simulation_checkpoint(const std::filesystem::path& path,
                                SimulationStorageState& state,
                                std::string& error);
bool read_simulation_checkpoint(const std::filesystem::path& path,
                                SimulationStorageState& state,
                                SimulationCheckpointAttachments& attachments,
                                std::string& error);
bool read_simulation_checkpoint(const std::filesystem::path& path,
                                SimulationStorageState& state,
                                SimulationCheckpointAttachments& attachments,
                                SimulationCheckpointIntegrity& integrity,
                                std::string& error);

// Fail-closed compatibility gate used before a checkpoint replaces a live
// configuration. Backend/device identity is intentionally excluded: portable
// checkpoints may move between CPU, unified-memory Vulkan, and discrete GPUs.
bool simulation_checkpoint_compatible(const SimulationStorageState& requested,
                                      const SimulationStorageState& checkpoint,
                                      std::string& error);

// Apply only branch-local, non-scientific configuration values that the
// compatibility gate permits to differ from a portable checkpoint.
void adopt_checkpoint_branch_configuration(
    const RuntimeConfigurationState& requested,
    RuntimeConfigurationState& restored);

} // namespace tmv::checkpoint

namespace tmv
{

// Public owner for an independent simulation state. Legacy globals remain an
// adapter over global_simulation_storage while orchestration migrates.
class RunContext
{
public:
    RunContext() = default;
    explicit RunContext(GridStorageState grid) : storage_(std::move(grid)) {}

    SimulationStorageState& storage() noexcept { return storage_; }
    const SimulationStorageState& storage() const noexcept { return storage_; }
    SimulationStateView state() { return make_simulation_state_view(storage_); }

    bool save(const std::filesystem::path& path, std::string& error) const
    {
        return tmv::checkpoint::write_simulation_checkpoint(path, storage_, error);
    }

    bool load(const std::filesystem::path& path, std::string& error)
    {
        return tmv::checkpoint::read_simulation_checkpoint(path, storage_, error);
    }

private:
    SimulationStorageState storage_;
};

} // namespace tmv
