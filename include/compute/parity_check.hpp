#pragma once

// Runtime CPU/GPU parity harness. Enabled GPU call sites replay their CPU
// reference and apply the tiered GPU audit section 2.3 tolerances.
// Expected fp32/fp64 rounding is tolerated; real stencil or physics drift is
// accumulated for the final report. Qualification pairing and fixture
// ownership live in tests/compute/support/parity_registry.hpp.

#include "compute/parity_types.hpp"

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <utility>

namespace nimbus::parity {

// Pure comparison result. Invalid inputs, empty domains, non-finite values,
// and values outside tolerance all fail closed.
struct FieldComparison
{
    bool passed = false;
    const char* invalid_reason = nullptr;
    std::uint64_t cells_compared = 0;
    std::uint64_t tolerance_breaches = 0;
    std::uint64_t nonfinite_cells = 0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    int worst_i = -1;
    int worst_j = -1;
    int worst_k = -1;
    float worst_cpu = 0.0f;
    float worst_gpu = 0.0f;
};

struct ParityFieldStats
{
    Tier tier = Tier::Pointwise;
    ComparisonDomain domain = ComparisonDomain::StrictInterior;
    std::uint64_t samples = 0;
    std::uint64_t failed_samples = 0;
    std::uint64_t empty_samples = 0;
    std::uint64_t cells_compared = 0;
    std::uint64_t breach_cells = 0;
    std::uint64_t nonfinite_cells = 0;
    double worst_abs = 0.0;
    double worst_rel = 0.0;
    int worst_i = -1;
    int worst_j = -1;
    int worst_k = -1;
    float worst_cpu = 0.0f;
    float worst_gpu = 0.0f;
};

// Per-simulation sampling cadence and accumulated CPU/GPU comparison results.
struct ParityRuntimeState
{
    using Key = std::tuple<std::string, std::string, ComparisonDomain>;

    mutable std::mutex mutex;
    std::map<Key, ParityFieldStats> stats;
    std::map<std::string, std::uint64_t> call_counters;
    std::uint64_t total_samples = 0;
    bool any_breach = false;
    int env_enabled = -1;
    int env_every = -1;

    ParityRuntimeState() = default;

    ParityRuntimeState(const ParityRuntimeState& other)
    {
        std::lock_guard<std::mutex> lock(other.mutex);
        copy_from(other);
    }

    ParityRuntimeState& operator=(const ParityRuntimeState& other)
    {
        if (this == &other) return *this;
        std::scoped_lock lock(mutex, other.mutex);
        copy_from(other);
        return *this;
    }

    ParityRuntimeState(ParityRuntimeState&& other) noexcept
    {
        std::lock_guard<std::mutex> lock(other.mutex);
        move_from(std::move(other));
    }

    ParityRuntimeState& operator=(ParityRuntimeState&& other) noexcept
    {
        if (this == &other) return *this;
        std::scoped_lock lock(mutex, other.mutex);
        move_from(std::move(other));
        return *this;
    }

private:
    void copy_from(const ParityRuntimeState& other)
    {
        stats = other.stats;
        call_counters = other.call_counters;
        total_samples = other.total_samples;
        any_breach = other.any_breach;
        env_enabled = other.env_enabled;
        env_every = other.env_every;
    }

    void move_from(ParityRuntimeState&& other)
    {
        stats = std::move(other.stats);
        call_counters = std::move(other.call_counters);
        total_samples = other.total_samples;
        any_breach = other.any_breach;
        env_enabled = other.env_enabled;
        env_every = other.env_every;
        other.total_samples = 0;
        other.any_breach = false;
        other.env_enabled = -1;
        other.env_every = -1;
    }
};

extern ParityRuntimeState& global_parity_runtime_state;

// Returns the calibrated section 2.3 tolerance.
Tolerance tolerance_for(Tier tier);

const char* tier_name(Tier tier);

const char* domain_name(ComparisonDomain domain);

// Fail-closed comparison primitive shared by runtime and standalone tests.
// The explicit domain prevents whole-field or face kernels from accidentally
// qualifying only their interior. A cell passes when
// |gpu-cpu| <= abs + rel*|cpu|.
FieldComparison compare_field(const float* cpu, const float* gpu,
                              int nr, int nth, int nz,
                              Tolerance tolerance,
                              ComparisonDomain domain);

// Config flag OR NIMBUS_VALIDATE_PARITY=1.
bool enabled();
bool enabled(ParityRuntimeState& runtime, bool config_enabled);

// Samples every NIMBUS_PARITY_EVERY-th eligible call per kernel (default 1).
bool sample(const char* kernel);
bool sample(ParityRuntimeState& runtime, bool config_enabled,
            const char* kernel);

// Records tiered comparison statistics for one field/domain. Microphysics
// permits 0.01% finite knife-edge cells because fp32/fp64 temperatures can
// straddle a discontinuous freezing branch; non-finite cells never qualify.
// `horizon` scales both bands linearly for accumulated sequential rounding.
// Invalid inputs, empty identifiers, and empty domains fail closed.
bool check_field(const char* kernel, const char* field,
                 const float* cpu, const float* gpu,
                 int nr, int nth, int nz, Tier tier,
                 ComparisonDomain domain, int horizon = 1);

bool check_field(ParityRuntimeState& runtime,
                 const char* kernel, const char* field,
                 const float* cpu, const float* gpu,
                 int nr, int nth, int nz, Tier tier,
                 ComparisonDomain domain, int horizon = 1);

// Compatibility form for strict-interior kernels only.
inline bool check_field(const char* kernel, const char* field,
                        const float* cpu, const float* gpu,
                        int nr, int nth, int nz, Tier tier, int horizon = 1)
{
    return check_field(kernel, field, cpu, gpu, nr, nth, nz, tier,
                       ComparisonDomain::StrictInterior, horizon);
}

bool any_breach();
bool any_breach(const ParityRuntimeState& runtime);

// Enabled runs with zero samples are vacuous; CI checks this count.
std::uint64_t total_samples();
std::uint64_t total_samples(const ParityRuntimeState& runtime);

// Prints the accumulated report; no-op when no samples were taken.
void print_report();
void print_report(const ParityRuntimeState& runtime);

// Clears accumulated state and cached environment reads.
void reset();
void reset(ParityRuntimeState& runtime);

}  // namespace nimbus::parity
