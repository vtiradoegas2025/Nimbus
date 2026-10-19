#pragma once

#include "compute/parity_check.hpp"

#include <vector>

namespace nimbus::parity {

// Qualification/test inventory. This API intentionally lives under tests so
// production targets do not carry fixture paths or compile-time test anchors.

enum class ShaderStatus
{
    Active,
    Deferred,
    Orphan,
};

enum class HorizonRule
{
    SingleDispatch,
    AccumulatedSubsteps,
};

enum class ReplayOwner
{
    RuntimeCallSite,
    HardwareFixture,
    RuntimeAndHardware,
    Waived,
};

enum class RequiredHardware
{
    None,
    Vulkan,
    Metal,
    VulkanAndMetal,
};

struct ParityCoverage
{
    // Comma-separated public outputs checked by the owning replay/fixture.
    const char* outputs = nullptr;
    const char* runtime_sample = nullptr;
    // Physical backend fixtures are tracked independently: a passing Vulkan
    // fixture is not evidence for the separately translated Metal kernel.
    const char* hardware_test = nullptr;
    const char* metal_hardware_test = nullptr;
    HorizonRule horizon = HorizonRule::SingleDispatch;
    ReplayOwner replay_owner = ReplayOwner::Waived;
    RequiredHardware required_hardware = RequiredHardware::None;
    const char* waiver = nullptr;
};

struct ParityEntry
{
    const char* shader;
    const char* cpu_reference;
    Tier tier;
    enum class Coordinate { Cartesian, Any } coordinate;
    ComparisonDomain comparison_domain;
    ShaderStatus status;
    ParityCoverage coverage;
};

// The full registry is derived from the backend-neutral kernel manifest.
const std::vector<ParityEntry>& registry();
bool validate_registry(const std::vector<ParityEntry>& entries,
                       const char** reason = nullptr) noexcept;

}  // namespace nimbus::parity
