#pragma once

#include "chaos/config.hpp"
#include "chaos/state.hpp"
#include "core/runtime/portable_scheme_state.hpp"
#include "turbulence/turbulence_base.hpp"

#include <memory>
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

class ComputeBackend;

namespace tmv::chaos
{

struct HorizontalStochasticStatePhase
{
    bool initialized = false;
    std::uint64_t current_epoch = 0;
    double cadence_dt_s = 0.0;
    std::int64_t earth_origin_i_cells = 0;
    std::int64_t earth_origin_j_cells = 0;
    double continuous_offset_x_m = 0.0;
    double continuous_offset_y_m = 0.0;
    int nr = 0;
    int nth = 0;
    int nz = 0;
};

struct HorizontalStochasticStorageIdentity
{
    static constexpr std::size_t maximum_processes = 3;
    std::size_t process_count = 0;
    std::array<const float*, maximum_processes> current_controls{};
    std::array<const float*, maximum_processes> previous_controls{};
    std::array<std::size_t, maximum_processes> current_control_capacities{};
    std::array<std::size_t, maximum_processes> previous_control_capacities{};
    std::array<const float*, maximum_processes> current_rendered{};
    std::array<const float*, maximum_processes> previous_rendered{};
};

class ChaosScheme
{
public:
    virtual ~ChaosScheme() = default;
    virtual std::unique_ptr<ChaosScheme> clone() const { return nullptr; }
    virtual PortableSchemeState portable_state() const { return {}; }
    virtual bool restore_portable_state(const PortableSchemeState& state,
                                        std::string& error)
    {
        if (empty_portable_scheme_state(state)) return true;
        error = "chaos scheme does not accept portable state";
        return false;
    }

    virtual std::string name() const = 0;
    virtual void configure_compute_backend(ComputeBackend*, bool) noexcept {}
    virtual void initialize(const ChaosConfig& cfg, const GridMetrics& grid) = 0;

    // Initial-condition, tendency, and noise hooks are independently optional.
    virtual void apply_initial_conditions(const ChaosConfig&,
                                          const GridMetrics&,
                                          SimulationState&,
                                          ChaosDiagnostics* = nullptr)
    {
    }
    virtual void apply_tendencies(const ChaosConfig&,
                                  const GridMetrics&,
                                  const ChaosStateView&,
                                  ChaosTendencies&,
                                  ChaosDiagnostics* = nullptr)
    {
    }
    virtual void step_noise(const ChaosConfig&, const GridMetrics&, double)
    {
    }

    virtual bool set_continuous_horizontal_earth_origin(
        double,
        double,
        const ChaosConfig&,
        const GridMetrics&,
        std::string& error)
    {
        error = "chaos scheme has no evolving Earth-coordinate state";
        return false;
    }

    /**
     * Translate persistent Earth-addressed stochastic state for a whole-cell
     * recycled-window move. Implementations must mutate only the receiving
     * scheme object; recycling invokes this on a detached clone before commit.
     */
    virtual bool translate_horizontal_stochastic_state(
        int,
        int,
        const ChaosConfig&,
        const GridMetrics&,
        std::string& error)
    {
        error = "chaos scheme has no evolving Earth-coordinate state";
        return false;
    }

    /**
     * Prepare a translated process window in reusable, non-durable storage.
     * The live scheme must remain unchanged on both success and failure.
     */
    virtual bool prepare_horizontal_stochastic_state_translation(
        int,
        int,
        const ChaosConfig&,
        const GridMetrics&,
        std::unique_ptr<ChaosScheme>&,
        HorizontalStochasticStatePhase&,
        std::string& error) const
    {
        error = "chaos scheme has no evolving Earth-coordinate state";
        return false;
    }

    /**
     * Exchange the live process window with a successfully prepared one.
     * Calling this method twice with the same workspace restores the original
     * window exactly.
     */
    virtual void exchange_prepared_horizontal_stochastic_state_translation(
        std::unique_ptr<ChaosScheme>&) noexcept
    {
    }

    virtual bool horizontal_stochastic_storage_identity(
        HorizontalStochasticStorageIdentity&,
        std::string& error) const
    {
        error = "chaos scheme has no evolving Earth-coordinate state";
        return false;
    }

    /** Describe the persistent Earth-addressed process phase. */
    virtual bool horizontal_stochastic_state_phase(
        HorizontalStochasticStatePhase&,
        std::string& error) const
    {
        error = "chaos scheme has no evolving Earth-coordinate state";
        return false;
    }

    /**
     * Initialize this scheme's window from a parent scheme at an exact prior
     * stochastic epoch. This is a reconstruction of one shared process, not
     * interpolation of rendered random values.
     */
    virtual bool initialize_nested_horizontal_stochastic_state(
        const ChaosScheme&,
        std::int64_t,
        std::int64_t,
        int,
        int,
        int,
        std::uint64_t,
        double,
        const ChaosConfig&,
        const GridMetrics&,
        std::string& error)
    {
        error = "chaos scheme cannot initialize a nested Earth-coordinate state";
        return false;
    }
};

} // namespace tmv::chaos

using tmv::chaos::ChaosScheme;
