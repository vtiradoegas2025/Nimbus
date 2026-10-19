#pragma once

#include "core/field/field3d.hpp"
#include "compute/dispatch/chaos.hpp"

#include <cstdint>
#include <array>
#include <string>
#include <vector>

class ComputeBackend;

namespace tmv::chaos
{

/**
 * Earth-fixed, compact-support stochastic pattern state.
 *
 * Independent Gaussian AR(1) controls live on a coarse physical lattice.
 * Model cells sample a normalized tensor-product cubic B-spline from the
 * sixteen neighboring controls.  The representation is independent of the
 * current computational window: overlapping windows address identical
 * controls, and a translated window reconstructs only control nodes that have
 * not previously been resident.
 */
class EarthCorrelatedNoise
{
public:
    EarthCorrelatedNoise() = default;

    void configure(
        std::uint64_t seed,
        int member_id,
        std::uint64_t initial_process_key,
        std::uint64_t innovation_process_key,
        double correlation_length_x_m,
        double correlation_length_y_m,
        double dx_m,
        double dy_m);
    void configure_native_evolution(ComputeBackend* backend, bool required) noexcept;

    bool initialized() const noexcept { return initialized_; }
    int nr() const noexcept { return nr_; }
    int nth() const noexcept { return nth_; }
    int nz() const noexcept { return nz_; }
    std::int64_t earth_origin_i_cells() const noexcept { return earth_origin_i_cells_; }
    std::int64_t earth_origin_j_cells() const noexcept { return earth_origin_j_cells_; }
    double continuous_offset_x_m() const noexcept { return continuous_offset_x_m_; }
    double continuous_offset_y_m() const noexcept { return continuous_offset_y_m_; }

    void initialize_domain(int nr, int nth, int nz);
    /**
     * Initialize a differently resolved window from the same Earth process.
     * Controls already resident in `source` are copied bit-for-bit whenever
     * their epoch matches; older epochs are deterministically reconstructed.
     */
    void initialize_domain_from(
        const EarthCorrelatedNoise& source,
        std::int64_t earth_origin_i_cells,
        std::int64_t earth_origin_j_cells,
        int nr,
        int nth,
        int nz,
        std::uint64_t source_epoch,
        std::uint64_t target_epoch,
        double rho);
    void advance(double rho, std::uint64_t innovation_epoch);

    /**
     * Translate the computational window without changing the Earth field.
     * New control nodes are reconstructed from epoch zero through
     * `current_epoch`; retained controls are copied bit-for-bit.
     */
    void translate(
        int shift_i_cells,
        int shift_j_cells,
        std::uint64_t current_epoch,
        double rho);

    void set_continuous_earth_origin(
        double origin_x_m,
        double origin_y_m,
        std::uint64_t current_epoch,
        double rho);

    /** Build a translated window in reusable detached storage. */
    void prepare_translated_window(
        EarthCorrelatedNoise& destination,
        int shift_i_cells,
        int shift_j_cells,
        std::uint64_t current_epoch,
        double rho) const;

    /** Exchange complete process-window ownership without allocation. */
    void swap_window_state(EarthCorrelatedNoise& other) noexcept;

    void render_current(Field3D& output) const;
    void render_previous(Field3D& output) const;
    void render_current_preallocated(Field3D& output) const noexcept;
    void render_previous_preallocated(Field3D& output) const noexcept;
    bool current_render_matches(const Field3D& field) const noexcept;
    bool previous_render_matches(const Field3D& field) const noexcept;

    const Field3D& current_controls() const noexcept { return current_controls_; }
    const Field3D& previous_controls() const noexcept { return previous_controls_; }
    std::int64_t control_i_min() const noexcept { return control_i_min_; }
    std::int64_t control_j_min() const noexcept { return control_j_min_; }

    bool restore(
        const Field3D& current_controls,
        const Field3D& previous_controls,
        int nr,
        int nth,
        int nz,
        std::int64_t earth_origin_i_cells,
        std::int64_t earth_origin_j_cells,
        std::int64_t control_i_min,
        std::int64_t control_j_min,
        double continuous_offset_x_m,
        double continuous_offset_y_m,
        std::string& error);
    bool restore(
        const Field3D& current_controls,
        const Field3D& previous_controls,
        int nr,
        int nth,
        int nz,
        std::int64_t earth_origin_i_cells,
        std::int64_t earth_origin_j_cells,
        std::int64_t control_i_min,
        std::int64_t control_j_min,
        std::string& error)
    {
        return restore(
            current_controls, previous_controls, nr, nth, nz,
            earth_origin_i_cells, earth_origin_j_cells,
            control_i_min, control_j_min, 0.0, 0.0, error);
    }

    /** Restore a legacy state whose control bounds were not serialized.
     * The bounds are an exact function of configuration, process phase,
     * Earth-cell origin, and window shape, so no scientific state is guessed.
     */
    bool restore_inferred_control_bounds(
        const Field3D& current_controls,
        const Field3D& previous_controls,
        int nr,
        int nth,
        int nz,
        std::int64_t earth_origin_i_cells,
        std::int64_t earth_origin_j_cells,
        std::string& error);

    bool restore_uninitialized_origin(
        std::int64_t earth_origin_i_cells,
        std::int64_t earth_origin_j_cells,
        std::string& error);

    static const char* algorithm_id() noexcept;

private:
    struct ControlBounds
    {
        std::int64_t i_min = 0;
        std::int64_t i_max = -1;
        std::int64_t j_min = 0;
        std::int64_t j_max = -1;
    };

    struct AxisStencil
    {
        std::array<int, 4> control_indices{};
        std::array<double, 4> normalized_weights{};
    };

    ControlBounds required_control_bounds(
        std::int64_t origin_i,
        std::int64_t origin_j,
        int nr,
        int nth,
        double continuous_offset_x_m = 0.0,
        double continuous_offset_y_m = 0.0) const;
    double gaussian(
        std::uint64_t process_key,
        std::uint64_t epoch,
        std::int64_t control_i,
        std::int64_t control_j,
        int k) const noexcept;
    double reconstruct(
        std::int64_t control_i,
        std::int64_t control_j,
        int k,
        std::uint64_t epoch,
        double rho) const noexcept;
    void rebuild_stencils();
    void render(const Field3D& controls, Field3D& output) const;
    void render_preallocated(
        const Field3D& controls, Field3D& output) const noexcept;
    bool render_matches(
        const Field3D& controls, const Field3D& field) const noexcept;
    bool dispatch_native_evolution(
        Field3D& destination,
        const Field3D& previous,
        std::uint64_t epoch,
        double rho,
        EarthControlEvolutionMode mode,
        std::int64_t control_i_min,
        std::int64_t control_j_min) const;
    void initialize_domain_in_place(int nr, int nth, int nz);
    void initialize_domain_from_in_place(
        const EarthCorrelatedNoise& source,
        std::int64_t earth_origin_i_cells,
        std::int64_t earth_origin_j_cells,
        int nr,
        int nth,
        int nz,
        std::uint64_t source_epoch,
        std::uint64_t target_epoch,
        double rho);
    void prepare_translated_window_in_place(
        EarthCorrelatedNoise& destination,
        int shift_i_cells,
        int shift_j_cells,
        std::uint64_t current_epoch,
        double rho,
        double target_offset_x_m,
        double target_offset_y_m) const;

    std::uint64_t seed_ = 42;
    int member_id_ = 0;
    std::uint64_t initial_process_key_ = 0;
    std::uint64_t innovation_process_key_ = 0;
    double dx_m_ = 0.0;
    double dy_m_ = 0.0;
    double control_spacing_x_m_ = 0.0;
    double control_spacing_y_m_ = 0.0;
    double control_phase_x_ = 0.0;
    double control_phase_y_ = 0.0;

    int nr_ = 0;
    int nth_ = 0;
    int nz_ = 0;
    std::int64_t earth_origin_i_cells_ = 0;
    std::int64_t earth_origin_j_cells_ = 0;
    double continuous_offset_x_m_ = 0.0;
    double continuous_offset_y_m_ = 0.0;
    std::int64_t control_i_min_ = 0;
    std::int64_t control_j_min_ = 0;
    bool configured_ = false;
    bool initialized_ = false;
    Field3D current_controls_;
    Field3D previous_controls_;
    std::vector<AxisStencil> i_stencils_;
    std::vector<AxisStencil> j_stencils_;
    ComputeBackend* native_backend_ = nullptr;
    bool native_evolution_required_ = false;
};

} // namespace tmv::chaos
